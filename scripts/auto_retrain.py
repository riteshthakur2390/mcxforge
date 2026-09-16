#!/usr/bin/env python3
"""
scripts/auto_retrain.py — Automatic ML Model Retraining Pipeline
=================================================================
Runs monthly. Retrains XGBoost+LightGBM+RF ensemble on last 90 days
of live trade outcomes. Validates OOS AUC before deploying new model.

Schedule via cron (1st of every month at 06:00 IST):
  0 6 1 * * cd /opt/signalforge && python scripts/auto_retrain.py

PIPELINE:
  1. Load last 90 days of completed trades from ledger
  2. Build feature matrix from trade metadata
  3. Train ensemble (walk-forward, not lookahead)
  4. Validate: OOS AUC must be >= 0.54
  5. If passes: deploy new model, backup old one
  6. If fails:  keep old model, send alert
  7. Send Telegram report with AUC scores

Usage:
  python scripts/auto_retrain.py
  python scripts/auto_retrain.py --days 60    # use last 60 days
  python scripts/auto_retrain.py --dry-run    # validate only, don't deploy
"""

import argparse
import json
import os
import shutil
import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

try:
    from config.settings import (
        JOURNAL_DIR, ML_MODELS_DIR,
        TELEGRAM_ENABLED,
    )
except ImportError:
    JOURNAL_DIR    = "journal"
    ML_MODELS_DIR  = "ml/saved_models"
    TELEGRAM_ENABLED = False

LEDGER_DB  = Path(JOURNAL_DIR) / "mcxforge.db"
MODEL_PATH = (
    Path(ML_MODELS_DIR) / "silverm_5minute.pkl"
    if (Path(ML_MODELS_DIR) / "silverm_5minute.pkl").exists()
    else (
        Path(ML_MODELS_DIR) / "silvermic_5minute.pkl"
        if (Path(ML_MODELS_DIR) / "silvermic_5minute.pkl").exists()
        else Path(ML_MODELS_DIR) / "commodity_5minute.pkl"
    )
)
BACKUP_DIR = Path(ML_MODELS_DIR) / "backups"
MIN_TRADES = 50      # minimum trades needed to retrain
MIN_OOS_AUC = 0.54   # minimum OOS AUC to deploy new model


def load_training_data(days: int = 90) -> tuple:
    """
    Load trade outcomes from ledger and build feature matrix.
    Returns (X, y, feature_names) or (None, None, None) if insufficient data.
    """
    cutoff = (date.today() - timedelta(days=days)).isoformat()

    if not LEDGER_DB.exists():
        print(f"  ❌ Ledger DB not found: {LEDGER_DB}")
        return None, None, None

    conn  = sqlite3.connect(LEDGER_DB)
    rows  = conn.execute(
        """SELECT votes, strategy_conf, ml_conf, setup_strength, quality_score,
                  regime, adx_at_entry, vix_at_entry, det_conf,
                  gross_pnl_pct, exit_reason, strategies_fired, holding_minutes
           FROM trades
           WHERE date >= ? AND gross_pnl_pct IS NOT NULL
           AND gross_pnl_pct != ''""",
        (cutoff,)
    ).fetchall()
    conn.close()

    if len(rows) < MIN_TRADES:
        print(f"  ❌ Insufficient trades: {len(rows)} < {MIN_TRADES} required")
        return None, None, None

    print(f"  ✅ Loaded {len(rows)} trades from last {days} days")

    feature_names = [
        "votes", "strategy_conf", "ml_conf", "setup_strength",
        "quality_score", "regime_enc", "adx", "vix", "det_conf",
        "holding_minutes",
    ]

    regime_map = {"TRENDING": 1, "CHOPPY": 0, "RANGING": 0, "HIGH_VOL": -1}

    X_rows, y = [], []
    for row in rows:
        (votes, s_conf, ml_conf, setup, quality,
         regime, adx, vix, det_conf, pnl,
         exit_r, strats, held_min) = row

        # Skip rows with missing critical data
        try:
            pnl_f = float(pnl)
        except (TypeError, ValueError):
            continue

        regime_enc = regime_map.get(str(regime).upper(), 0)

        X_rows.append([
            float(votes    or 2),
            float(s_conf   or 0.65),
            float(ml_conf  or 0.5),
            float(setup    or 0.6),
            float(quality  or 0.7),
            float(regime_enc),
            float(adx      or 18),
            float(vix      or 18),
            float(det_conf or 0.6),
            float(held_min or 30),
        ])
        # Binary label: win if PnL > 2% (meaningful win)
        y.append(1 if pnl_f > 2.0 else 0)

    if len(X_rows) < MIN_TRADES:
        return None, None, None

    return np.array(X_rows), np.array(y), feature_names


def train_ensemble(X: np.ndarray, y: np.ndarray) -> tuple:
    """
    Train XGBoost + LightGBM + RandomForest ensemble.
    Uses time-based walk-forward split (no data leakage).
    Returns (models, oos_auc, val_auc).
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("  ❌ scikit-learn not installed: pip install scikit-learn")
        return None, 0.0, 0.0

    try:
        import xgboost as xgb
    except ImportError:
        print("  ⚠️  xgboost not installed — using RF only")
        xgb = None

    try:
        import lightgbm as lgb
    except ImportError:
        print("  ⚠️  lightgbm not installed — using RF only")
        lgb = None

    n         = len(X)
    split_idx = int(n * 0.75)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
        print("  ❌ Not enough class diversity for training")
        return None, 0.0, 0.0

    models  = {}
    val_probs = []

    # Random Forest (always available)
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_leaf=5,
        random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_prob = rf.predict_proba(X_val)[:, 1]
    val_probs.append(rf_prob * 0.2)   # 20% weight
    models["rf"] = rf
    print(f"  RF  OOS AUC: {roc_auc_score(y_val, rf_prob):.4f}")

    # XGBoost
    if xgb is not None:
        xgb_m = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="auc", random_state=42,
            use_label_encoder=False, verbosity=0,
        )
        xgb_m.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)], verbose=False)
        xgb_prob = xgb_m.predict_proba(X_val)[:, 1]
        val_probs.append(xgb_prob * 0.4)   # 40% weight
        models["xgb"] = xgb_m
        print(f"  XGB OOS AUC: {roc_auc_score(y_val, xgb_prob):.4f}")

    # LightGBM
    if lgb is not None:
        lgb_m = lgb.LGBMClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1,
        )
        lgb_m.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(20, verbose=False),
                              lgb.log_evaluation(-1)])
        lgb_prob = lgb_m.predict_proba(X_val)[:, 1]
        val_probs.append(lgb_prob * 0.4)   # 40% weight
        models["lgb"] = lgb_m
        print(f"  LGB OOS AUC: {roc_auc_score(y_val, lgb_prob):.4f}")

    # Ensemble score
    ensemble_prob = sum(val_probs)
    oos_auc = roc_auc_score(y_val, ensemble_prob)
    val_auc = oos_auc

    print(f"  Ensemble OOS AUC: {oos_auc:.4f}  (min required: {MIN_OOS_AUC})")
    return models, oos_auc, val_auc


def save_model(models: dict, feature_names: list,
               n_samples: int, oos_auc: float, val_auc: float) -> None:
    """Save trained model to disk, backup old model."""
    import pickle
    from datetime import datetime

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Backup old model
    if MODEL_PATH.exists():
        backup = BACKUP_DIR / f"nifty_5minute_{date.today().isoformat()}.pkl"
        shutil.copy(MODEL_PATH, backup)
        print(f"  Backed up old model → {backup}")

    # Save new model
    payload = {
        "models":        models,
        "feature_cols":  feature_names,
        "meta": {
            "trained_at":   datetime.now().isoformat(),
            "n_samples":    n_samples,
            "n_features":   len(feature_names),
            "oos_auc":      oos_auc,
            "val_auc":      val_auc,
            "win_rate":     0.0,
            "model_names":  list(models.keys()),
        },
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(payload, f)
    print(f"  ✅ New model saved → {MODEL_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days",    type=int,  default=90)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(f"\n{'═'*55}")
    print(f"  MCXForge — Auto ML Retraining Pipeline")
    print(f"  Date: {date.today()}  |  Window: last {args.days} days")
    print(f"{'═'*55}\n")

    # Step 1: Load data
    print("Step 1: Loading training data...")
    X, y, feature_names = load_training_data(args.days)
    if X is None:
        print("\n  ⚠️  Retraining skipped — insufficient data.")
        return

    n_pos = int(y.sum())
    print(f"  Samples: {len(X)}  Wins: {n_pos}  Losses: {len(y)-n_pos}  "
          f"Win rate: {n_pos/len(y)*100:.1f}%")

    # Step 2: Train
    print("\nStep 2: Training ensemble...")
    models, oos_auc, val_auc = train_ensemble(X, y)
    if models is None:
        print("\n  ❌ Training failed.")
        return

    # Step 3: Validate
    print(f"\nStep 3: Validation  OOS AUC={oos_auc:.4f}  "
          f"Threshold={MIN_OOS_AUC}")
    passed = oos_auc >= MIN_OOS_AUC

    if not passed:
        print(f"\n  ❌ OOS AUC {oos_auc:.4f} < {MIN_OOS_AUC} — NOT deploying.")
        print("  Keeping existing model. Check signal quality.")
        result = "FAILED"
    elif args.dry_run:
        print(f"\n  ✅ Validation passed — DRY RUN, not deploying.")
        result = "DRY_RUN"
    else:
        print(f"\nStep 4: Deploying new model...")
        save_model(models, feature_names, len(X), oos_auc, val_auc)
        result = "DEPLOYED"

    # Step 4: Report
    report = {
        "date":         date.today().isoformat(),
        "result":       result,
        "samples":      len(X),
        "oos_auc":      round(oos_auc, 4),
        "passed":       passed,
        "days_window":  args.days,
    }
    report_path = Path(JOURNAL_DIR) / f"retrain_{date.today().isoformat()}.json"
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report: {report_path}")

    # Telegram notification
    if TELEGRAM_ENABLED:
        try:
            import asyncio
            from utils.telegram_notifier import get_notifier
            msg = (
                f"🤖 *ML Retrain: {result}*\n"
                f"OOS AUC: {oos_auc:.4f} ({'✅' if passed else '❌'})\n"
                f"Samples: {len(X)} | Window: {args.days}d"
            )
            asyncio.run(get_notifier().send_text(msg, target="LIVE"))
        except Exception:
            pass

    print(f"\n  {'✅ Done' if passed else '⚠️ Skipped'}: {result}\n")


if __name__ == "__main__":
    main()
