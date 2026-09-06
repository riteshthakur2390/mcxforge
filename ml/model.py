"""
ml/model.py — SignalForge ML Model Ensemble
=============================================
Wraps XGBoost + LightGBM + RandomForest into a single ensemble.

Fix log:
  v1.0 — Fixed predict_proba KeyError when feature_cols empty
        — Fixed evaluate() using wrong feature_cols before fit()
        — Added strict column alignment between training and inference
"""

import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore", message=".*XGBoost, please export the model.*")
from pathlib import Path
from datetime import datetime
from loguru import logger
from dataclasses import dataclass, field
from typing import Union

# Suppress the joblib/scikit-learn parallel delayed UserWarning emitted by LightGBM/XGBoost
warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")


@dataclass
class ModelMeta:
    trained_at:   str   = ""
    n_samples:    int   = 0
    n_features:   int   = 0
    win_rate:     float = 0.0
    train_auc:    float = 0.0
    val_auc:      float = 0.0
    val_precision: float = 0.0
    val_recall:    float = 0.0
    recommended_threshold: float = 0.62
    calibration_method: str = ""
    calibration_coef: float = 1.0
    calibration_intercept: float = 0.0
    walk_forward_auc: float = 0.0
    walk_forward_precision: float = 0.0
    walk_forward_windows: int = 0
    feature_cols: list  = field(default_factory=list)
    model_names:  list  = field(default_factory=list)


class SignalForgeEnsemble:
    """
    Ensemble of XGBoost + LightGBM + RandomForest.
    Single interface for training, predicting, saving, loading.
    """

    WEIGHTS = {"xgb": 0.40, "lgb": 0.40, "rf": 0.20}

    def __init__(self):
        self.models:       dict      = {}
        self.feature_cols: list      = []
        self.meta:         ModelMeta = ModelMeta()
        self._trained:     bool      = False

    # ── BUILD ─────────────────────────────────────────────────────────────────

    def build(self) -> None:
        """Instantiate all three models with optimized hyperparameters."""
        try:
            import xgboost as xgb
            self.models["xgb"] = xgb.XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                min_child_weight=3, gamma=0.1,
                reg_alpha=0.1, reg_lambda=1.0,
                use_label_encoder=False, eval_metric="logloss",
                random_state=42, n_jobs=-1, verbosity=0,
            )
        except Exception as e:
            logger.warning(f"[Ensemble] xgboost unavailable — skipping ({e})")

        try:
            import lightgbm as lgb
            self.models["lgb"] = lgb.LGBMClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0,
                random_state=42, n_jobs=-1, verbose=-1,
            )
        except Exception as e:
            logger.warning(f"[Ensemble] lightgbm unavailable — skipping ({e})")

        try:
            from sklearn.ensemble import RandomForestClassifier
            self.models["rf"] = RandomForestClassifier(
                n_estimators=200, max_depth=8,
                min_samples_split=10, min_samples_leaf=5,
                max_features="sqrt", random_state=42, n_jobs=-1,
            )
        except Exception as e:
            logger.warning(f"[Ensemble] scikit-learn unavailable — skipping ({e})")

    # ── TRAIN ─────────────────────────────────────────────────────────────────

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        feature_cols: list,
        sample_weight: Union[np.ndarray, None] = None,
    ) -> dict:
        """
        Train all models. feature_cols defines the column order
        used for ALL future predictions — stored in self.feature_cols.
        """
        from sklearn.metrics import roc_auc_score

        # FIX: set feature_cols BEFORE any predict call
        self.feature_cols = feature_cols
        scores = {}

        for name, model in self.models.items():
            try:
                fit_kwargs = {}
                if sample_weight is not None and len(sample_weight) == len(y):
                    fit_kwargs["sample_weight"] = sample_weight
                model.fit(X[feature_cols], y, **fit_kwargs)
                prob         = model.predict_proba(X[feature_cols])[:, 1]
                auc          = roc_auc_score(y, prob)
                scores[name] = round(auc, 4)
                logger.info(f"[Ensemble] {name.upper():12} train AUC = {auc:.4f}")
            except Exception as e:
                logger.error(f"[Ensemble] {name} fit failed: {e}")
                scores[name] = 0.0

        self._trained = True
        return scores

    # ── PREDICT ───────────────────────────────────────────────────────────────

    def predict_proba(self, features: dict) -> float:
        """
        Predict win probability from a feature dict.
        Returns weighted average P(win) in [0, 1].

        FIX: guards against empty feature_cols and missing keys.
        """
        if not self._trained or not self.models:
            return 0.0

        # FIX: if feature_cols empty after load, rebuild from features keys
        if not self.feature_cols:
            logger.warning("[Ensemble] feature_cols empty — using all available features")
            self.feature_cols = list(features.keys())

        try:
            # FIX: fill missing features with 0.0 instead of raising KeyError
            row = {col: features.get(col, 0.0) for col in self.feature_cols}
            X   = pd.DataFrame([row])[self.feature_cols].fillna(0.0)

            total_w, weighted_p = 0.0, 0.0
            for name, model in self.models.items():
                w    = self.WEIGHTS.get(name, 0.33)
                prob = float(model.predict_proba(X)[0][1])
                weighted_p += w * prob
                total_w    += w

            raw = weighted_p / total_w if total_w > 0 else 0.0
            return round(self._calibrate_probability(raw), 4)

        except Exception as e:
            logger.debug(f"[Ensemble] predict error: {e}")
            return 0.0

    def _calibrate_probability(self, raw_proba: float) -> float:
        raw = max(0.0, min(1.0, float(raw_proba)))
        try:
            from config.settings import ML_CALIBRATION_ENABLED
            if not ML_CALIBRATION_ENABLED:
                return raw
        except Exception as e:
            logger.debug(f"[Ensemble] Could not check ML_CALIBRATION_ENABLED: {e}")

        method = str(getattr(self.meta, "calibration_method", "") or "").lower()
        if method != "platt":
            return raw
        coef = float(getattr(self.meta, "calibration_coef", 1.0) or 1.0)
        intercept = float(getattr(self.meta, "calibration_intercept", 0.0) or 0.0)
        z = coef * raw + intercept
        calibrated = 1.0 / (1.0 + np.exp(-z))
        return max(0.0, min(1.0, float(calibrated)))

    # ── EVALUATE (cross-val before full fit) ──────────────────────────────────

    def evaluate_cv(
        self,
        X:            pd.DataFrame,
        y:            np.ndarray,
        feature_cols: list,
        n_splits:     int = 5,
        include_oof:  bool = False,
        sample_weight: Union[np.ndarray, None] = None,
    ) -> dict:
        """
        Time-series cross-validation BEFORE fitting on full data.
        FIX: uses feature_cols param directly (not self.feature_cols which
             is only set during fit()).
        """
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import roc_auc_score, precision_score, recall_score
        from sklearn.linear_model import LogisticRegression

        logger.info(f"[Ensemble] {n_splits}-fold TimeSeriesSplit CV...")
        tscv         = TimeSeriesSplit(n_splits=n_splits)
        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx][feature_cols], X.iloc[val_idx][feature_cols]
            y_train, y_val = y[train_idx], y[val_idx]

            if len(np.unique(y_val)) < 2:
                continue

            try:
                fold_model = None
                try:
                    import xgboost as xgb
                    fold_model = xgb.XGBClassifier(
                        n_estimators=100, max_depth=4, learning_rate=0.1,
                        use_label_encoder=False, eval_metric="logloss",
                        verbosity=0, n_jobs=-1,
                    )
                except Exception:
                    try:
                        from sklearn.ensemble import RandomForestClassifier
                        fold_model = RandomForestClassifier(
                            n_estimators=200,
                            max_depth=8,
                            min_samples_split=10,
                            min_samples_leaf=5,
                            max_features="sqrt",
                            random_state=42 + fold,
                            n_jobs=-1,
                        )
                    except Exception as inner:
                        raise RuntimeError(f"No CV model available: {inner}") from inner

                if sample_weight is not None and len(sample_weight) == len(y):
                    fold_weight = np.asarray(sample_weight)[train_idx]
                    fold_model.fit(X_train, y_train, sample_weight=fold_weight)
                else:
                    fold_model.fit(X_train, y_train)
                proba = fold_model.predict_proba(X_val)[:, 1]
                preds = (proba >= 0.62).astype(int)

                fold_results.append({
                    "fold":      fold + 1,
                    "auc":       roc_auc_score(y_val, proba),
                    "precision": precision_score(y_val, preds, zero_division=0),
                    "recall":    recall_score(y_val, preds, zero_division=0),
                    "proba":     proba,
                    "y_true":    y_val,
                    "val_idx":   val_idx,
                })
                logger.info(
                    f"[Ensemble] Fold {fold+1}: "
                    f"AUC={fold_results[-1]['auc']:.3f} | "
                    f"P={fold_results[-1]['precision']:.3f} | "
                    f"R={fold_results[-1]['recall']:.3f}"
                )
            except Exception as e:
                logger.warning(f"[Ensemble] Fold {fold+1} skipped: {e}")

        if not fold_results:
            return {
                "val_auc": 0.0,
                "val_precision": 0.0,
                "val_recall": 0.0,
                "recommended_threshold": 0.62,
                "threshold_analysis": [],
                "fold_results": [],
            }

        oof_proba = np.concatenate([r["proba"] for r in fold_results])
        oof_y = np.concatenate([r["y_true"] for r in fold_results])
        threshold_analysis = self._threshold_sweep(oof_y, oof_proba)
        recommended_threshold = threshold_analysis[0]["threshold"] if threshold_analysis else 0.62
        calibration = {
            "method": "",
            "coef": 1.0,
            "intercept": 0.0,
        }
        try:
            calibrator = LogisticRegression(random_state=42)
            calibrator.fit(oof_proba.reshape(-1, 1), oof_y)
            calibration = {
                "method": "platt",
                "coef": float(calibrator.coef_[0][0]),
                "intercept": float(calibrator.intercept_[0]),
            }
        except Exception as e:
            logger.debug(f"[Ensemble] Calibration skipped: {e}")

        return {
            "val_auc":       round(float(np.mean([r["auc"] for r in fold_results])), 4),
            "val_precision": round(float(np.mean([r["precision"] for r in fold_results])), 4),
            "val_recall":    round(float(np.mean([r["recall"] for r in fold_results])), 4),
            "recommended_threshold": recommended_threshold,
            "calibration": calibration,
            "threshold_analysis": threshold_analysis,
            "fold_results":  [
                {
                    "fold": r["fold"],
                    "auc": r["auc"],
                    "precision": r["precision"],
                    "recall": r["recall"],
                }
                for r in fold_results
            ],
            **(
                {
                    "oof_proba": oof_proba,
                    "oof_y_true": oof_y,
                    "oof_indices": np.concatenate([r["val_idx"] for r in fold_results]),
                }
                if include_oof else
                {}
            ),
        }

    @staticmethod
    def _threshold_sweep(y_true: np.ndarray, proba: np.ndarray) -> list[dict]:
        from sklearn.metrics import precision_score, recall_score, f1_score

        if len(y_true) == 0:
            return []

        min_signals = max(3, int(len(y_true) * 0.08))
        candidates: list[dict] = []
        for threshold in np.arange(0.35, 0.76, 0.02):
            preds = (proba >= threshold).astype(int)
            selected = int(preds.sum())
            if selected < min_signals:
                continue
            precision = precision_score(y_true, preds, zero_division=0)
            recall = recall_score(y_true, preds, zero_division=0)
            f1 = f1_score(y_true, preds, zero_division=0)
            coverage = selected / max(len(y_true), 1)
            candidates.append({
                "threshold": round(float(threshold), 2),
                "precision": round(float(precision), 4),
                "recall": round(float(recall), 4),
                "f1": round(float(f1), 4),
                "coverage": round(float(coverage), 4),
                "selected": selected,
            })

        candidates.sort(
            key=lambda row: (
                row["precision"],
                row["f1"],
                row["coverage"],
                row["threshold"],
            ),
            reverse=True,
        )
        return candidates

    # ── FEATURE IMPORTANCE ────────────────────────────────────────────────────

    def get_feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        importances = {}
        for name, model in self.models.items():
            try:
                if hasattr(model, "feature_importances_"):
                    for feat, val in zip(self.feature_cols, model.feature_importances_):
                        importances[feat] = importances.get(feat, 0) + val * self.WEIGHTS.get(name, 0.33)
            except Exception:
                pass

        if not importances:
            return pd.DataFrame()

        df = pd.DataFrame([
            {"feature": k, "importance": v}
            for k, v in sorted(importances.items(), key=lambda x: x[1], reverse=True)
        ])
        return df.head(top_n)

    # ── SAVE / LOAD ───────────────────────────────────────────────────────────

    def save(self, path: Path, meta: Union[ModelMeta, None] = None) -> None:
        if meta:
            self.meta              = meta
            self.meta.feature_cols = self.feature_cols
            self.meta.model_names  = list(self.models.keys())
            self.meta.trained_at   = datetime.now().isoformat()

        joblib.dump({
            "models":       self.models,
            "feature_cols": self.feature_cols,
            "meta":         self.meta,
        }, path)
        logger.success(f"[Ensemble] Saved → {path}")

    def load(self, path: Path) -> bool:
        if not path.exists():
            return False
        try:
            data              = joblib.load(path)
            self.models       = data.get("models", {})
            self.feature_cols = data.get("feature_cols", [])
            self.meta         = data.get("meta", ModelMeta())
            self._trained     = bool(self.models)

            if self._trained:
                logger.info(
                    f"[Ensemble] Loaded {path.name} | "
                    f"models={list(self.models.keys())} | "
                    f"features={len(self.feature_cols)} | "
                    f"trained={self.meta.trained_at[:10] if self.meta.trained_at else 'unknown'}"
                )
            return self._trained
        except Exception as e:
            logger.error(f"[Ensemble] Load failed: {e}")
            return False

    @property
    def is_trained(self) -> bool:
        return self._trained and bool(self.models)

    @property
    def model_count(self) -> int:
        return len(self.models)

    @property
    def decision_threshold(self) -> float:
        threshold = float(getattr(self.meta, "recommended_threshold", 0.62) or 0.62)
        return max(0.0, min(1.0, threshold))
