"""
ml/training/trainer.py — SignalForge ML Training Pipeline
===========================================================
Reads signal journal CSVs (with outcome_eod filled in),
builds feature matrix, trains ensemble, evaluates, saves model.

Called by scripts/train_ml.py — do not call directly.

Data flow:
    journal/signals_*.csv
        → load labeled rows (WIN/LOSS)
        → match each signal to its candle window
        → extract 40+ features per signal
        → train XGBoost + LightGBM + RandomForest
        → evaluate with TimeSeriesSplit (no data leakage)
        → save to ml/saved_models/nifty_5minute.pkl
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from loguru import logger
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, confusion_matrix,
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.features import extract, FEATURE_NAMES
from ml.model import SignalForgeEnsemble, ModelMeta
from config.settings import JOURNAL_DIR, DATA_CACHE_DIR, ML_MODELS_DIR


class SignalForgeTrainer:

    def __init__(self, min_samples: int = 100):
        self.min_samples = min_samples
        self.ensemble    = SignalForgeEnsemble()
        self._candle_cache: pd.DataFrame | None = None

    # ── STEP 1: LOAD JOURNAL DATA ─────────────────────────────────────────────

    def load_journal(self) -> pd.DataFrame:
        """Load all signal journal CSVs with WIN/LOSS outcomes."""
        csv_files = sorted(glob.glob(f"{JOURNAL_DIR}/signals_*.csv"))
        if not csv_files:
            raise FileNotFoundError(
                f"No journal CSVs found in {JOURNAL_DIR}/\n"
                f"Run SignalForge in OBSERVE mode first to collect signal data."
            )

        dfs = []
        for f in csv_files:
            try:
                dfs.append(pd.read_csv(f))
            except Exception as e:
                logger.warning(f"[Trainer] Skipping {f}: {e}")

        if not dfs:
            raise ValueError("No valid journal files found")

        df = pd.concat(dfs, ignore_index=True)

        # Keep only labeled rows
        labeled = df[df["outcome_eod"].isin(["WIN", "LOSS"])].copy()
        labeled["label"] = (labeled["outcome_eod"] == "WIN").astype(int)
        labeled = labeled.sort_values(["date", "time"]).reset_index(drop=True)

        logger.info(f"[Trainer] Journal: {len(df)} total | {len(labeled)} labeled")
        logger.info(
            f"[Trainer] Labels: {labeled['label'].sum()} WIN | "
            f"{(1-labeled['label']).sum()} LOSS | "
            f"win_rate={labeled['label'].mean()*100:.1f}%"
        )

        if len(labeled) < self.min_samples:
            raise ValueError(
                f"Only {len(labeled)} labeled samples. Need ≥ {self.min_samples}.\n"
                f"Keep running SignalForge in OBSERVE mode to collect more data.\n"
                f"Current status: {len(labeled)}/{self.min_samples} samples"
            )

        return labeled

    # ── STEP 2: LOAD CANDLE DATA ──────────────────────────────────────────────

    def load_candles(self) -> pd.DataFrame:
        """Load cached NIFTY 5-min candle data for feature extraction."""
        if self._candle_cache is not None:
            return self._candle_cache

        # Try all broker caches
        cache_files = sorted(
            Path(DATA_CACHE_DIR).glob("NIFTY_5minute_*.parquet"),
            key=lambda p: p.stat().st_size,
            reverse=True,
        )
        if not cache_files:
            raise FileNotFoundError(
                f"No NIFTY 5-min cache found in {DATA_CACHE_DIR}/\n"
                f"Run: python scripts/download_history.py"
            )

        df = pd.read_parquet(cache_files[0])
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        logger.info(
            f"[Trainer] Candles: {len(df)} rows | "
            f"{df.index[0].date()} → {df.index[-1].date()} | "
            f"source={cache_files[0].name}"
        )
        self._candle_cache = df
        return df

    # ── STEP 3: BUILD FEATURE MATRIX ──────────────────────────────────────────

    def build_features(
        self,
        labeled:  pd.DataFrame,
        candles:  pd.DataFrame,
        lookback: int = 60,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """
        For each labeled signal, find the candle window at signal time
        and extract features.

        Returns (X, y) ready for training.
        """
        import pytz
        IST = pytz.timezone("Asia/Kolkata")

        rows, labels, skipped = [], [], 0

        for _, row in labeled.iterrows():
            try:
                # Build signal timestamp
                signal_ts = pd.Timestamp(
                    f"{row['date']} {row['time']}"
                ).tz_localize(IST)

                # Get candle window ending at signal time
                window = candles[candles.index <= signal_ts].tail(lookback)
                if len(window) < 30:
                    skipped += 1
                    continue

                feats = extract(
                    df        = window,
                    conf      = float(row.get("strategy_conf", 0.65)),
                    votes     = int(row.get("votes", 2)),
                    direction = str(row.get("direction", "BUY_CALL")),
                    lookback  = lookback,
                )
                if feats is None:
                    skipped += 1
                    continue

                rows.append(feats)
                labels.append(int(row["label"]))

            except Exception as e:
                logger.debug(f"[Trainer] Feature extraction error: {e}")
                skipped += 1

        if not rows:
            raise ValueError(
                "No features extracted. Check that candle timestamps "
                "match journal dates."
            )

        logger.info(
            f"[Trainer] Features: {len(rows)} samples built | "
            f"{skipped} skipped"
        )

        # Use only feature names we know about
        cols = [c for c in FEATURE_NAMES if c in rows[0]]
        X    = pd.DataFrame(rows)[cols].fillna(0.0)
        y    = np.array(labels)

        return X, y

    # ── STEP 4: EVALUATE ──────────────────────────────────────────────────────

    def evaluate(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        n_splits: int = 5,
    ) -> dict:
        """
        Time-series cross-validation.
        No data leakage — future data never used to predict past.
        """
        logger.info(f"[Trainer] Evaluating with {n_splits}-fold TimeSeriesSplit...")

        tscv     = TimeSeriesSplit(n_splits=n_splits)
        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx],       y[val_idx]

            if len(np.unique(y_val)) < 2:
                continue

            # Quick fold model for eval
            try:
                import xgboost as xgb
                fold_model = xgb.XGBClassifier(
                    n_estimators=100, max_depth=4, learning_rate=0.1,
                    use_label_encoder=False, eval_metric="logloss",
                    verbosity=0, n_jobs=-1,
                )
                fold_model.fit(X_train[self.ensemble.feature_cols or list(X.columns)], y_train)
                proba = fold_model.predict_proba(X_val[self.ensemble.feature_cols or list(X.columns)])[:, 1]
                preds = (proba >= 0.62).astype(int)

                fold_results.append({
                    "fold":      fold + 1,
                    "auc":       roc_auc_score(y_val, proba),
                    "precision": precision_score(y_val, preds, zero_division=0),
                    "recall":    recall_score(y_val, preds, zero_division=0),
                    "f1":        f1_score(y_val, preds, zero_division=0),
                    "n_val":     len(y_val),
                })
                logger.info(
                    f"[Trainer] Fold {fold+1}: "
                    f"AUC={fold_results[-1]['auc']:.3f} | "
                    f"P={fold_results[-1]['precision']:.3f} | "
                    f"R={fold_results[-1]['recall']:.3f}"
                )
            except Exception as e:
                logger.warning(f"[Trainer] Fold {fold+1} error: {e}")

        if not fold_results:
            return {"val_auc": 0.0, "val_precision": 0.0, "val_recall": 0.0}

        avg = {
            "val_auc":       np.mean([r["auc"] for r in fold_results]),
            "val_precision": np.mean([r["precision"] for r in fold_results]),
            "val_recall":    np.mean([r["recall"] for r in fold_results]),
            "val_f1":        np.mean([r["f1"] for r in fold_results]),
            "fold_results":  fold_results,
        }
        logger.info(
            f"[Trainer] CV Results: "
            f"AUC={avg['val_auc']:.3f} | "
            f"Precision={avg['val_precision']:.3f} | "
            f"Recall={avg['val_recall']:.3f}"
        )
        return avg

    # ── STEP 5: TRAIN FULL MODEL ──────────────────────────────────────────────

    def train(self, X: pd.DataFrame, y: np.ndarray) -> dict:
        """Train ensemble on full dataset."""
        logger.info(f"[Trainer] Training on {len(X)} samples...")
        self.ensemble.build()
        scores = self.ensemble.fit(X, y, list(X.columns))
        return scores

    # ── STEP 6: SAVE ─────────────────────────────────────────────────────────

    def save(
        self,
        X:         pd.DataFrame,
        y:         np.ndarray,
        train_auc: float,
        val_auc:   float,
    ) -> Path:
        """Save trained ensemble to disk."""
        os.makedirs(ML_MODELS_DIR, exist_ok=True)
        path = Path(ML_MODELS_DIR) / "nifty_5minute.pkl"

        meta = ModelMeta(
            n_samples    = len(y),
            n_features   = X.shape[1],
            win_rate     = float(y.mean()),
            train_auc    = train_auc,
            val_auc      = val_auc,
            feature_cols = list(X.columns),
            model_names  = list(self.ensemble.models.keys()),
        )
        self.ensemble.save(path, meta)
        return path

    # ── FULL PIPELINE ─────────────────────────────────────────────────────────

    def run(self, lookback: int = 60) -> dict:
        """
        Run full training pipeline end-to-end.
        Returns summary dict.
        """
        results = {}

        # Step 1
        labeled = self.load_journal()
        results["n_labeled"] = len(labeled)

        # Step 2
        candles = self.load_candles()

        # Step 3
        X, y = self.build_features(labeled, candles, lookback)
        results["n_features"] = X.shape[1]
        results["n_samples"]  = len(y)
        results["win_rate"]   = float(y.mean())

        # Step 4 — eval before training on full set
        eval_results = self.evaluate(X, y)
        results["val_auc"]       = eval_results.get("val_auc", 0.0)
        results["val_precision"] = eval_results.get("val_precision", 0.0)

        # Step 5 — train on full dataset
        train_scores = self.train(X, y)
        results["train_auc"] = float(np.mean(list(train_scores.values())))

        # Step 6 — save
        path = self.save(X, y, results["train_auc"], results["val_auc"])
        results["model_path"] = str(path)
        results["models"]     = list(self.ensemble.models.keys())

        return results