"""
ml/training/trainer.py — SignalForge ML Training Pipeline
===========================================================
Fix log:
  v1.0 — evaluate() now calls ensemble.evaluate_cv() with explicit
          feature_cols param (not self.feature_cols which is set in fit())
        — train() sets feature_cols before evaluate to avoid empty-cols bug
"""
from __future__ import annotations

import os
import sys
import glob
import ast
import json
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.features import extract, FEATURE_NAMES
from ml.model import SignalForgeEnsemble, ModelMeta
from config.settings import DATA_CACHE_DIR, JOURNAL_DIR, ML_MODELS_DIR, BACKTEST_TIMEFRAME, LIVE_TIMEFRAME

INTRADAY_INTERVALS = ("1minute", "5minute", "15minute", "30minute")


class SignalForgeTrainer:

    def __init__(
        self,
        min_samples: int = 100,
        include_archive: bool = True,
        csv_paths: list[str] | None = None,
        label_mode: str = "classification",
        model_path: str | Path | None = None,
    ):
        self.min_samples = min_samples
        self.include_archive = include_archive
        self.csv_paths = [str(Path(p)) for p in (csv_paths or []) if str(p).strip()]
        self.label_mode = str(label_mode or "classification").strip().lower()
        if self.label_mode not in {"classification", "return"}:
            raise ValueError("label_mode must be one of: classification, return")
        self.model_path = Path(model_path) if model_path else None
        self.ensemble    = SignalForgeEnsemble()
        self._candle_cache: pd.DataFrame | None = None

    # ── STEP 1: LOAD JOURNAL ─────────────────────────────────────────────────

    def load_journal(self) -> pd.DataFrame:
        csv_files = self._resolve_csv_files()
        if not csv_files:
            raise FileNotFoundError(
                f"No labeled CSVs found.\n"
                f"Checked explicit paths: {', '.join(self.csv_paths) or 'none'}\n"
                f"Default journal dir: {JOURNAL_DIR}/"
            )

        dfs = []
        for f in csv_files:
            try:
                dfs.append(pd.read_csv(f))
            except Exception as e:
                logger.warning(f"[Trainer] Skipping {f}: {e}")

        df = pd.concat(dfs, ignore_index=True)
        df = self._normalize_labeled_rows(df)
        if "signal_id" not in df.columns:
            df["signal_id"] = ""
        if "lifecycle_status" not in df.columns:
            df["lifecycle_status"] = ""

        labeled = df[df["outcome_eod"].isin(["WIN", "LOSS"])].copy()
        if "lifecycle_status" in labeled.columns:
            closed = labeled["lifecycle_status"].fillna("").eq("CLOSED")
            if closed.any():
                labeled = labeled[closed | labeled["lifecycle_status"].fillna("").eq("")]

        signal_ids = labeled["signal_id"].fillna("").astype(str)
        clean_labeled = labeled[signal_ids.ne("")].copy()
        if not clean_labeled.empty:
            clean_labeled = clean_labeled.drop_duplicates(
                subset=["signal_id"], keep="last"
            )
        if len(clean_labeled) >= self.min_samples:
            labeled = clean_labeled
        else:
            labeled["_dedup_key"] = signal_ids.where(
                signal_ids.ne(""),
                (
                    labeled["date"].astype(str)
                    + "|"
                    + labeled["time"].astype(str)
                    + "|"
                    + labeled.get("option_symbol", pd.Series("", index=labeled.index)).astype(str)
                    + "|"
                    + labeled["direction"].astype(str)
                    + "|"
                    + labeled.get("strategies_fired", pd.Series("", index=labeled.index)).astype(str)
                ),
            )
            labeled = labeled.drop_duplicates(subset=["_dedup_key"], keep="last").copy()

        labeled["label"] = (labeled["outcome_eod"] == "WIN").astype(int)
        labeled["target_return"] = labeled["pnl_pct"].astype(float).fillna(0.0)
        if self.label_mode == "return":
            logger.info("[Trainer] label_mode=return (model still optimizes ranking probability; return is used in weighting/analysis)")
        labeled = labeled.sort_values(["date", "time"]).reset_index(drop=True)
        if "_dedup_key" in labeled.columns:
            labeled = labeled.drop(columns=["_dedup_key"])

        logger.info(
            f"[Trainer] Journal: {len(df)} total | {len(labeled)} labeled | "
            f"win_rate={labeled['label'].mean()*100:.1f}% | "
            f"archive={'on' if self.include_archive else 'off'}"
        )

        if len(labeled) < self.min_samples:
            raise ValueError(
                f"Only {len(labeled)} labeled samples. Need ≥ {self.min_samples}."
            )
        return labeled

    @staticmethod
    def _normalize_labeled_rows(df: pd.DataFrame) -> pd.DataFrame:
        norm = df.copy()
        for col in ("date", "time", "signal_id", "direction", "regime", "strategies_fired", "outcome_eod"):
            if col not in norm.columns:
                norm[col] = ""
        for col in ("strategy_conf", "votes", "realized_pnl", "pnl_pct"):
            if col not in norm.columns:
                norm[col] = 0.0

        # Parse date/time from entry_time or signal_id when missing.
        dt_parts = pd.to_datetime(norm.get("entry_time", pd.Series("", index=norm.index)), errors="coerce")
        missing_date = norm["date"].astype(str).str.strip().eq("")
        missing_time = norm["time"].astype(str).str.strip().eq("")
        norm.loc[missing_date & dt_parts.notna(), "date"] = dt_parts[missing_date & dt_parts.notna()].dt.date.astype(str)
        norm.loc[missing_time & dt_parts.notna(), "time"] = dt_parts[missing_time & dt_parts.notna()].dt.strftime("%H:%M")

        # signal_id starts with ISO timestamp in this codebase.
        signal_prefix = norm["signal_id"].astype(str).str.split("|").str[0]
        id_ts = pd.to_datetime(signal_prefix, errors="coerce")
        norm.loc[missing_date & id_ts.notna(), "date"] = id_ts[missing_date & id_ts.notna()].dt.date.astype(str)
        norm.loc[missing_time & id_ts.notna(), "time"] = id_ts[missing_time & id_ts.notna()].dt.strftime("%H:%M")

        norm["date"] = norm["date"].astype(str).str.strip()
        norm["time"] = norm["time"].astype(str).str.strip()
        norm["outcome_eod"] = norm["outcome_eod"].astype(str).str.upper().str.strip()
        norm["pnl_pct"] = pd.to_numeric(norm.get("pnl_pct"), errors="coerce").fillna(0.0)
        norm["realized_pnl"] = pd.to_numeric(norm.get("realized_pnl"), errors="coerce").fillna(0.0)
        norm["votes"] = pd.to_numeric(norm.get("votes"), errors="coerce").fillna(0).astype(int)
        norm["strategy_conf"] = pd.to_numeric(norm.get("strategy_conf"), errors="coerce").fillna(0.0)
        return norm

    def _resolve_csv_files(self) -> list[str]:
        if self.csv_paths:
            files: list[str] = []
            seen: set[str] = set()
            for raw_path in self.csv_paths:
                path = Path(raw_path)
                matches = [path] if path.exists() else [Path(p) for p in glob.glob(raw_path)]
                for match in sorted(matches):
                    resolved = str(match)
                    if resolved not in seen and match.is_file():
                        files.append(resolved)
                        seen.add(resolved)
            return files

        csv_files = sorted(glob.glob(f"{JOURNAL_DIR}/signals_*.csv"))
        if self.include_archive:
            csv_files.extend(sorted(glob.glob(f"{JOURNAL_DIR}/old_csv/signals_*.csv")))
            csv_files = list(dict.fromkeys(csv_files))
        return csv_files

    # ── STEP 2: LOAD CANDLES ─────────────────────────────────────────────────

    def load_candles(self) -> pd.DataFrame:
        if self._candle_cache is not None:
            return self._candle_cache
        from backtesting.engine import _load_primary_cache

        df = _load_primary_cache(force_sqlite_archive=True)
        cache_file = str(df.attrs.get("cache_file", ""))
        cache_interval = str(df.attrs.get("cache_interval", BACKTEST_TIMEFRAME))
        if cache_interval != BACKTEST_TIMEFRAME:
            logger.warning(
                f"[Trainer] Missing merged {BACKTEST_TIMEFRAME} cache coverage. "
                f"Using {cache_interval} source instead: {cache_file or 'merged cache'}"
            )
        logger.info(
            f"[Trainer] Candles: {len(df)} | source={cache_file or 'merged cache'} | "
            f"interval={cache_interval}"
        )
        if not df.empty:
            logger.info(
                f"[Trainer] Candle coverage: {df.index.min()} → {df.index.max()} | "
                f"days={len({idx.date() for idx in df.index})}"
            )
        self._candle_cache = df
        return df

    # ── STEP 3: BUILD FEATURES ────────────────────────────────────────────────

    def build_training_dataset(
        self, labeled: pd.DataFrame, candles: pd.DataFrame, lookback: int = 60
    ) -> tuple[pd.DataFrame, np.ndarray, list, pd.DataFrame]:
        """
        Returns (X, y, feature_cols) — feature_cols is the canonical
        ordered list used for all training and inference.
        """
        import pytz
        IST = pytz.timezone("Asia/Kolkata")

        rows, labels, skipped = [], [], 0
        meta_rows: list[dict] = []

        for _, row in labeled.iterrows():
            try:
                signal_ts = self._safe_signal_ts(
                    date_str=str(row.get("date", "")),
                    time_str=str(row.get("time", "")),
                    fallback_raw=str(row.get("entry_time", "")),
                    ist=IST,
                )
                if signal_ts is None:
                    skipped += 1
                    continue

                window = candles[candles.index <= signal_ts].tail(lookback)
                if len(window) < 30:
                    skipped += 1
                    continue

                signal_context = self._extract_signal_context(row)
                feats = extract(
                    df=window,
                    conf=float(row.get("strategy_conf", 0.65)),
                    votes=int(row.get("votes", 2)),
                    direction=str(row.get("direction", "BUY_CALL")),
                    lookback=lookback,
                    regime_info={
                        "label": str(row.get("regime", "RANGING") or "RANGING"),
                        "confidence": float(row.get("regime_confidence", 0.5) or 0.5),
                        "atr_ratio": float(row.get("regime_atr_ratio", 1.0) or 1.0),
                    },
                    signal_context=signal_context,
                    strategies_fired=self._parse_strategies(row.get("strategies_fired", "")),
                )
                if feats is None:
                    skipped += 1
                    continue

                rows.append(feats)
                labels.append(int(row["label"]))
                setup = signal_context.get("setup", {}) or {}
                structure = signal_context.get("market_structure", {}) or {}
                entry_validation = structure.get("entry_validation", {}) or {}
                liquidity_event = structure.get("liquidity_event", {}) or {}
                meta_rows.append({
                    "date": str(row.get("date", "")),
                    "time": str(row.get("time", "")),
                    "timestamp": signal_ts.isoformat(),
                    "regime": str(row.get("regime", "")),
                    "direction": str(row.get("direction", "")),
                    "combo": str(row.get("strategies_fired", "")),
                    "signal_id": str(row.get("signal_id", "")),
                    "setup_type": str(setup.get("setup_type", "unknown") or "unknown"),
                    "liquidity_event": str(liquidity_event.get("type", "NONE") or "NONE"),
                    "middle_range": bool(entry_validation.get("middle_range", False)),
                    "realized_pnl": float(row.get("realized_pnl", 0.0) or 0.0),
                    "pnl_pct": float(row.get("pnl_pct", 0.0) or 0.0),
                    "label": int(row["label"]),
                })

            except Exception as e:
                logger.debug(f"[Trainer] Feature error: {e}")
                skipped += 1

        if not rows:
            raise ValueError("No features extracted. Check candle/journal timestamps.")

        logger.info(f"[Trainer] Built: {len(rows)} samples | {skipped} skipped")
        if skipped and rows:
            logger.info(
                f"[Trainer] Feature coverage: kept={len(rows)} skipped={skipped} "
                f"skip_rate={skipped / max(len(rows) + skipped, 1):.1%}"
            )

        # FIX: use FEATURE_NAMES as canonical column order
        # Only include features that actually exist in extracted data
        feature_cols = [c for c in FEATURE_NAMES if c in rows[0]]
        X = pd.DataFrame(rows)[feature_cols].fillna(0.0)
        y = np.array(labels)
        meta_df = pd.DataFrame(meta_rows).reset_index(drop=True)

        logger.info(f"[Trainer] Features: {len(feature_cols)} | Samples: {len(y)}")
        return X, y, feature_cols, meta_df

    def build_features(
        self, labeled: pd.DataFrame, candles: pd.DataFrame, lookback: int = 60
    ) -> tuple[pd.DataFrame, np.ndarray, list]:
        X, y, feature_cols, _ = self.build_training_dataset(labeled, candles, lookback)
        return X, y, feature_cols

    @staticmethod
    def _safe_signal_ts(
        date_str: str,
        time_str: str,
        fallback_raw: str,
        ist,
    ):
        timestamp = pd.NaT
        if date_str.strip() and time_str.strip():
            timestamp = pd.to_datetime(f"{date_str} {time_str}", errors="coerce")
        if pd.isna(timestamp) and fallback_raw.strip():
            timestamp = pd.to_datetime(fallback_raw, errors="coerce")
        if pd.isna(timestamp):
            return None
        if timestamp.tzinfo is None:
            return timestamp.tz_localize(ist)
        return timestamp.tz_convert(ist)

    @staticmethod
    def _parse_strategies(raw_value) -> list[str]:
        if isinstance(raw_value, list):
            return [str(item).strip() for item in raw_value if str(item).strip()]
        text = str(raw_value or "").strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, list):
                    return [str(item).strip() for item in parsed if str(item).strip()]
            except Exception:
                pass
        delimiter = "|" if "|" in text else ","
        return [part.strip() for part in text.split(delimiter) if part.strip()]

    @staticmethod
    def _parse_contract_snapshot(raw_snapshot) -> dict:
        if isinstance(raw_snapshot, dict):
            return raw_snapshot
        text = str(raw_snapshot or "").strip()
        if not text or text.lower() in {"nan", "none", "{}"}:
            return {}
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            pass
        try:
            parsed = ast.literal_eval(text)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}

    def _extract_signal_context(self, row: pd.Series) -> dict:
        snapshot = self._parse_contract_snapshot(row.get("contract_snapshot", {}))
        setup = snapshot.get("setup", {}) or {}
        setup_context = setup.get("context", {}) or {}
        structure = setup_context.get("market_structure", {}) or {}
        regime = str(row.get("regime", "RANGING") or "RANGING")
        return {
            "setup": setup,
            "market_structure": structure,
            "detailed_regime": {
                "label": regime,
                "confidence": float(row.get("regime_confidence", 0.5) or 0.5),
                "atr_ratio": float(row.get("regime_atr_ratio", 1.0) or 1.0),
            },
            "weighted_vote": {
                "winner_avg_weight": float(row.get("winner_avg_weight", 1.0) or 1.0),
            },
            "strategies_fired": self._parse_strategies(row.get("strategies_fired", "")),
        }

    @staticmethod
    def _compute_sample_weights(meta_df: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        if len(meta_df) != len(y):
            return np.ones(len(y), dtype=float)

        temp = meta_df.copy()
        temp["label"] = y
        temp["pattern_key"] = (
            temp["regime"].astype(str).fillna("NA")
            + "|"
            + temp["setup_type"].astype(str).fillna("unknown")
            + "|"
            + temp["combo"].astype(str).fillna("none")
            + "|"
            + temp["liquidity_event"].astype(str).fillna("NONE")
            + "|"
            + temp["middle_range"].astype(str).fillna("False")
        )
        stats = (
            temp.groupby("pattern_key", as_index=False)
            .agg(
                n=("label", "size"),
                loss_rate=("label", lambda s: 1.0 - float(np.mean(s))),
                avg_pnl=("pnl_pct", "mean"),
            )
        )
        stats["confidence"] = np.minimum(1.0, stats["n"] / 12.0)
        stats["pattern_penalty"] = (
            stats["loss_rate"] * stats["confidence"] * 0.9
            + (stats["avg_pnl"] < 0).astype(float) * 0.2
        )
        penalty_map = dict(zip(stats["pattern_key"], stats["pattern_penalty"]))
        penalties = temp["pattern_key"].map(penalty_map).fillna(0.0).astype(float).values

        # Base class balancing + targeted loss upweighting for repeated bad patterns.
        pos_rate = float(np.mean(y)) if len(y) else 0.5
        neg_rate = 1.0 - pos_rate
        pos_w = 0.5 / max(pos_rate, 1e-6)
        neg_w = 0.5 / max(neg_rate, 1e-6)
        class_w = np.where(y == 1, pos_w, neg_w)
        loss_boost = np.where(y == 0, 1.0 + penalties, 1.0 + penalties * 0.25)
        weights = class_w * loss_boost
        return np.clip(weights, 0.5, 3.5)

    # ── STEP 4: EVALUATE + STEP 5: TRAIN ─────────────────────────────────────

    def train(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        feature_cols: list,
        sample_weight: np.ndarray | None = None,
    ) -> tuple[dict, dict]:
        """
        FIX: evaluate_cv uses explicit feature_cols param.
             fit() called after eval so feature_cols is set correctly.
        Returns (train_scores, eval_results).
        """
        self.ensemble.build()

        # FIX: evaluate BEFORE fit — uses evaluate_cv with explicit feature_cols
        cv_splits = min(5, max(2, len(X) - 1))
        if len(X) < 8:
            raise ValueError(
                f"Only {len(X)} usable feature samples after candle alignment. "
                "Need at least 8. Check candle cache coverage for the CSV date range."
            )
        eval_results = self.ensemble.evaluate_cv(
            X,
            y,
            feature_cols,
            n_splits=cv_splits,
            sample_weight=sample_weight,
        )

        # Train on full dataset
        train_scores = self.ensemble.fit(X, y, feature_cols, sample_weight=sample_weight)

        return train_scores, eval_results

    # ── STEP 6: SAVE ─────────────────────────────────────────────────────────

    def save(
        self, X: pd.DataFrame, y: np.ndarray,
        feature_cols: list, train_auc: float, val_auc: float,
        val_precision: float, val_recall: float, recommended_threshold: float,
        calibration: dict | None = None,
        walk_forward: dict | None = None,
    ) -> Path:
        os.makedirs(ML_MODELS_DIR, exist_ok=True)
        path = self.model_path or Path(ML_MODELS_DIR) / f"nifty_{LIVE_TIMEFRAME}.pkl"

        meta = ModelMeta(
            n_samples    = len(y),
            n_features   = len(feature_cols),
            win_rate     = float(y.mean()),
            train_auc    = train_auc,
            val_auc      = val_auc,
            val_precision = val_precision,
            val_recall    = val_recall,
            recommended_threshold = recommended_threshold,
            calibration_method = str((calibration or {}).get("method", "") or ""),
            calibration_coef = float((calibration or {}).get("coef", 1.0) or 1.0),
            calibration_intercept = float((calibration or {}).get("intercept", 0.0) or 0.0),
            walk_forward_auc = float((walk_forward or {}).get("avg_auc", 0.0) or 0.0),
            walk_forward_precision = float((walk_forward or {}).get("avg_precision", 0.0) or 0.0),
            walk_forward_windows = int((walk_forward or {}).get("windows", 0) or 0),
            feature_cols = feature_cols,
            model_names  = list(self.ensemble.models.keys()),
        )
        self.ensemble.save(path, meta)
        return path

    @staticmethod
    def _walk_forward_summary(
        X: pd.DataFrame,
        y: np.ndarray,
        feature_cols: list[str],
        sample_weight: np.ndarray | None = None,
    ) -> dict:
        from sklearn.metrics import precision_score, roc_auc_score
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.ensemble import RandomForestClassifier

        if len(X) < 40:
            return {"windows": 0, "avg_auc": 0.0, "avg_precision": 0.0}

        splitter = TimeSeriesSplit(n_splits=4)
        aucs: list[float] = []
        precisions: list[float] = []
        for tr_idx, te_idx in splitter.split(X):
            y_test = y[te_idx]
            if len(np.unique(y_test)) < 2:
                continue
            model = RandomForestClassifier(
                n_estimators=180,
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=4,
                random_state=42,
                n_jobs=-1,
            )
            X_train = X.iloc[tr_idx][feature_cols]
            X_test = X.iloc[te_idx][feature_cols]
            fit_kwargs = {}
            if sample_weight is not None and len(sample_weight) == len(y):
                fit_kwargs["sample_weight"] = np.asarray(sample_weight)[tr_idx]
            model.fit(X_train, y[tr_idx], **fit_kwargs)
            proba = model.predict_proba(X_test)[:, 1]
            preds = (proba >= 0.60).astype(int)
            aucs.append(float(roc_auc_score(y_test, proba)))
            precisions.append(float(precision_score(y_test, preds, zero_division=0)))
        if not aucs:
            return {"windows": 0, "avg_auc": 0.0, "avg_precision": 0.0}
        return {
            "windows": len(aucs),
            "avg_auc": round(float(np.mean(aucs)), 4),
            "avg_precision": round(float(np.mean(precisions)), 4),
        }

    # ── FULL PIPELINE ─────────────────────────────────────────────────────────

    def run(self, lookback: int = 60) -> dict:
        results = {}

        labeled = self.load_journal()
        results["n_labeled"] = len(labeled)

        candles = self.load_candles()

        X, y, feature_cols, meta_df = self.build_training_dataset(labeled, candles, lookback)
        sample_weight = self._compute_sample_weights(meta_df, y)
        results["n_features"] = len(feature_cols)
        results["n_samples"]  = len(y)
        results["win_rate"]   = float(y.mean())
        results["sample_weight_mean"] = float(np.mean(sample_weight)) if len(sample_weight) else 1.0

        # FIX: train() now takes feature_cols explicitly and returns both scores
        train_scores, eval_results = self.train(X, y, feature_cols, sample_weight=sample_weight)
        wf_summary = self._walk_forward_summary(X, y, feature_cols, sample_weight)

        results["train_auc"] = float(np.mean(list(train_scores.values())))
        results["val_auc"]   = eval_results.get("val_auc", 0.0)
        results["val_precision"] = eval_results.get("val_precision", 0.0)
        results["val_recall"] = eval_results.get("val_recall", 0.0)
        results["recommended_threshold"] = eval_results.get("recommended_threshold", 0.62)
        results["walk_forward_auc"] = wf_summary.get("avg_auc", 0.0)
        results["walk_forward_precision"] = wf_summary.get("avg_precision", 0.0)
        results["walk_forward_windows"] = wf_summary.get("windows", 0)

        path = self.save(
            X, y, feature_cols,
            results["train_auc"], results["val_auc"],
            results["val_precision"], results["val_recall"],
            results["recommended_threshold"],
            eval_results.get("calibration"),
            wf_summary,
        )
        results["model_path"] = str(path)
        results["models"]     = list(self.ensemble.models.keys())

        return results
