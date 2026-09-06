"""
ml/training/walk_forward.py — Walk-Forward Validation
=======================================================
The most important missing piece for backtest reliability.

WHAT IT DOES:
  Splits 66 days of data into time-ordered windows:
    Window 1: Train on days 1-30  → Test on days 31-40
    Window 2: Train on days 1-40  → Test on days 41-50
    Window 3: Train on days 1-50  → Test on days 51-60
    Window 4: Train on days 1-60  → Test on days 61-66

  Model NEVER sees future data.
  Out-of-sample performance on each test window is the true estimate.

WHY THIS MATTERS:
  Current system trains on all 66 days then backtests on the same 66 days.
  That is data leakage — the model has already seen the answers.
  Walk-forward is the ONLY way to know if ML adds real edge.

  Professional rule: if walk-forward OOS AUC > 0.55 consistently → ML adds edge.
  If OOS AUC ≈ 0.50 → ML is noise, remove it and use strategy votes alone.

USAGE:
  from ml.training.walk_forward import WalkForwardValidator
  wfv = WalkForwardValidator(train_days=30, test_days=10, step_days=10)
  results = wfv.run(candles_df, journal_df)
  wfv.report(results)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from dataclasses import dataclass


@dataclass
class WFWindow:
    window_id:     int
    train_start:   str
    train_end:     str
    test_start:    str
    test_end:      str
    n_train:       int
    n_test:        int
    oos_auc:       float
    oos_precision: float
    oos_win_rate:  float


class WalkForwardValidator:
    """
    Walk-forward validation for SignalForge ML model.
    Provides honest out-of-sample performance estimates.
    """

    def __init__(
        self,
        train_days: int = 30,   # minimum days to train on
        test_days:  int = 10,   # days in each test window
        step_days:  int = 10,   # step forward by this many days
    ):
        self.train_days = train_days
        self.test_days  = test_days
        self.step_days  = step_days

    def run(
        self,
        candles: pd.DataFrame,
        journal: pd.DataFrame,
        lookback: int = 60,
    ) -> list[WFWindow]:
        """
        Run walk-forward validation across all available data.
        Returns list of WFWindow results (one per test window).
        """
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from ml.features import extract, FEATURE_NAMES
        from ml.training.trainer import SignalForgeTrainer

        results = []
        all_dates = sorted(journal["date"].unique())

        if len(all_dates) < self.train_days + self.test_days:
            logger.warning(
                f"[WFV] Not enough data: {len(all_dates)} days, "
                f"need {self.train_days + self.test_days}"
            )
            return results

        window_id = 0
        start_idx = 0

        while start_idx + self.train_days + self.test_days <= len(all_dates):
            train_dates = all_dates[start_idx : start_idx + self.train_days]
            test_dates  = all_dates[
                start_idx + self.train_days :
                start_idx + self.train_days + self.test_days
            ]

            train_journal = journal[journal["date"].isin(train_dates)]
            test_journal  = journal[journal["date"].isin(test_dates)]

            if len(train_journal) < 10:
                start_idx += self.step_days
                continue
            if len(test_journal) < 3:
                start_idx += self.step_days
                continue

            window_id += 1
            logger.info(
                f"[WFV] Window {window_id}: "
                f"train={train_dates[0]}→{train_dates[-1]} ({len(train_journal)} trades) | "
                f"test={test_dates[0]}→{test_dates[-1]} ({len(test_journal)} trades)"
            )

            # Build features
            trainer = SignalForgeTrainer(min_samples=5)
            trainer._candle_cache = candles

            try:
                X_train, y_train, feat_cols = trainer.build_features(
                    train_journal, candles, lookback
                )
                X_test, y_test, _ = trainer.build_features(
                    test_journal, candles, lookback
                )
            except Exception as e:
                logger.warning(f"[WFV] Window {window_id} feature build failed: {e}")
                start_idx += self.step_days
                continue

            if len(X_train) < 5 or len(X_test) < 2:
                start_idx += self.step_days
                continue

            # Train on training window only
            try:
                import xgboost as xgb
                from sklearn.metrics import roc_auc_score, precision_score

                model = xgb.XGBClassifier(
                    n_estimators=100, max_depth=4, learning_rate=0.1,
                    subsample=0.8, min_child_weight=3,
                    use_label_encoder=False, eval_metric="logloss",
                    verbosity=0, n_jobs=-1, random_state=42,
                )
                # Align columns
                common_cols = [c for c in feat_cols if c in X_test.columns]
                model.fit(X_train[common_cols], y_train)
                proba = model.predict_proba(X_test[common_cols])[:, 1]
                preds = (proba >= 0.55).astype(int)

                oos_auc = roc_auc_score(y_test, proba) if len(np.unique(y_test)) > 1 else 0.5
                oos_prec = precision_score(y_test, preds, zero_division=0)
                oos_wr   = float(y_test.mean())

                result = WFWindow(
                    window_id     = window_id,
                    train_start   = str(train_dates[0]),
                    train_end     = str(train_dates[-1]),
                    test_start    = str(test_dates[0]),
                    test_end      = str(test_dates[-1]),
                    n_train       = len(y_train),
                    n_test        = len(y_test),
                    oos_auc       = round(oos_auc, 4),
                    oos_precision = round(oos_prec, 4),
                    oos_win_rate  = round(oos_wr, 4),
                )
                results.append(result)
                logger.info(
                    f"[WFV]   OOS AUC={oos_auc:.3f} | "
                    f"Precision={oos_prec:.3f} | "
                    f"WinRate={oos_wr:.1%}"
                )
            except Exception as e:
                logger.warning(f"[WFV] Window {window_id} train/eval failed: {e}")

            start_idx += self.step_days

        return results

    def report(self, results: list[WFWindow]) -> dict:
        """Print and return summary of walk-forward results."""
        if not results:
            logger.warning("[WFV] No results to report")
            return {}

        avg_auc  = float(np.mean([r.oos_auc for r in results]))
        avg_prec = float(np.mean([r.oos_precision for r in results]))
        avg_wr   = float(np.mean([r.oos_win_rate for r in results]))

        print()
        print("━" * 60)
        print("  Walk-Forward Validation Results")
        print("━" * 60)
        print(f"  {'Window':<8} {'Train Period':<25} {'Test Period':<20} {'OOS AUC':<10} {'Precision'}")
        print(f"  {'-'*8} {'-'*25} {'-'*20} {'-'*10} {'-'*9}")
        for r in results:
            print(
                f"  {r.window_id:<8} "
                f"{r.train_start}→{r.train_end:<14} "
                f"{r.test_start}→{r.test_end:<8} "
                f"{r.oos_auc:<10.3f} "
                f"{r.oos_precision:.3f}"
            )
        print(f"  {'Average':<8} {'':25} {'':20} {avg_auc:<10.3f} {avg_prec:.3f}")
        print("━" * 60)
        print()
        print(f"  Verdict:")
        if avg_auc >= 0.60:
            print(f"  ✅ OOS AUC={avg_auc:.3f} — ML adds REAL edge. Keep it.")
        elif avg_auc >= 0.55:
            print(f"  ⚠️  OOS AUC={avg_auc:.3f} — Marginal edge. Monitor carefully.")
        else:
            print(f"  ❌ OOS AUC={avg_auc:.3f} — ML is noise. Disable it, use votes only.")
            print(f"     Set ML_MIN_CONFIDENCE=0.0 in settings.py to bypass.")
        print()

        return {
            "windows":      len(results),
            "avg_oos_auc":  avg_auc,
            "avg_precision": avg_prec,
            "avg_win_rate": avg_wr,
            "has_edge":     avg_auc >= 0.55,
        }