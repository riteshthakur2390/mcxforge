"""
scripts/train_commodity_ml.py — Dedicated Commodity ML Training Pipeline
========================================================================
Trains the 3-model SignalForge Ensemble (XGBoost, LightGBM, RandomForest)
specifically on MCX SILVERM 5-minute candle history and labeled backtest/paper
option trades. Saves the production model directly to:
    ml/saved_models/silvermic_5minute.pkl
"""

from __future__ import annotations

import sys
import os
import glob
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ml.features import extract, FEATURE_NAMES
from ml.model import SignalForgeEnsemble, ModelMeta
from ml.training.trainer import SignalForgeTrainer
from config.settings import ML_MODELS_DIR


def load_commodity_candles() -> pd.DataFrame:
    """Load primary 5m candles for SILVERM."""
    csv_path = REPO_ROOT / "data" / "historical" / "SILVERM_dhan_5m.csv"
    if not csv_path.exists():
        csv_path = REPO_ROOT / "data" / "historical" / "SILVERMIC_dhan_5m.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing historical candles at {csv_path}")

    df = pd.read_csv(csv_path)
    ts_col = [c for c in df.columns if "time" in c.lower() or "date" in c.lower()][0]
    df[ts_col] = pd.to_datetime(df[ts_col])
    df = df.set_index(ts_col).sort_index()
    # Normalize column names to lowercase
    df.columns = [c.lower() for c in df.columns]
    for req in ("open", "high", "low", "close", "volume"):
        if req not in df.columns:
            raise ValueError(f"Candle data missing required column '{req}'")
    logger.info(f"Loaded {len(df)} 5-minute candles from {csv_path.name} ({df.index.min()} to {df.index.max()})")
    return df


def load_labeled_commodity_trades() -> pd.DataFrame:
    """Collect all labeled trade records from backtest results and journals."""
    potential_files = []
    # 1. Backtest results
    potential_files.extend(glob.glob(str(REPO_ROOT / "analysis" / "**" / "trade_ledger.csv"), recursive=True))
    potential_files.extend(glob.glob(str(REPO_ROOT / "scratch" / "**" / "trade_ledger.csv"), recursive=True))
    potential_files.extend(glob.glob(str(Path("/Users/vishranti/.gemini/antigravity-ide/brain/c017d470-e356-4eb9-9bdc-2887d65e201b/scratch") / "**" / "trade_ledger.csv"), recursive=True))
    # 2. Live / paper journals
    potential_files.extend(glob.glob(str(REPO_ROOT / "journal" / "*.csv")))
    potential_files.extend(glob.glob(str(REPO_ROOT / "journal" / "old_csv" / "*.csv")))

    dfs = []
    seen_files = set()
    for f in potential_files:
        p = Path(f)
        if p.exists() and str(p) not in seen_files:
            seen_files.add(str(p))
            try:
                raw = pd.read_csv(p)
                if len(raw) > 0:
                    dfs.append(raw)
                    logger.debug(f"Loaded {len(raw)} trades from {p.name}")
            except Exception as e:
                logger.warning(f"Failed loading {p}: {e}")

    if not dfs:
        raise ValueError("No trade ledger or journal files found for training.")

    df_all = pd.concat(dfs, ignore_index=True)
    logger.info(f"Total raw trades loaded across sources: {len(df_all)}")

    # Normalize fields
    norm = pd.DataFrame()
    # Timestamp parsing
    dt = pd.to_datetime(df_all["entry_time"], errors="coerce")
    norm["entry_time"] = df_all["entry_time"].astype(str)
    norm["date"] = dt.dt.strftime("%Y-%m-%d")
    norm["time"] = dt.dt.strftime("%H:%M")

    # Direction (CALL / PUT / BUY / SELL)
    if "direction" in df_all.columns:
        sig = df_all["direction"].astype(str).str.upper()
        norm["direction"] = np.where(sig.str.contains("BUY|CALL|LONG"), "CALL", "PUT")
    else:
        norm["direction"] = "CALL"

    # Outcome
    if "net_pnl_inr" in df_all.columns:
        norm["pnl_inr"] = pd.to_numeric(df_all["net_pnl_inr"], errors="coerce").fillna(0.0)
        norm["outcome_eod"] = np.where(norm["pnl_inr"] > 0, "WIN", "LOSS")
    elif "is_win" in df_all.columns:
        norm["outcome_eod"] = np.where(df_all["is_win"].astype(bool), "WIN", "LOSS")
    else:
        norm["outcome_eod"] = "LOSS"

    if "strategy_conf" in df_all.columns:
        norm["strategy_conf"] = pd.to_numeric(df_all["strategy_conf"], errors="coerce").fillna(0.75).astype(float)
    else:
        norm["strategy_conf"] = 0.75

    if "votes" in df_all.columns:
        norm["votes"] = pd.to_numeric(df_all["votes"], errors="coerce").fillna(5).astype(int)
    else:
        norm["votes"] = 5

    if "pnl_pct" in df_all.columns:
        norm["pnl_pct"] = pd.to_numeric(df_all["pnl_pct"], errors="coerce").fillna(0.0).astype(float)
    else:
        norm["pnl_pct"] = 0.0

    if "strategies_fired" in df_all.columns:
        norm["strategies_fired"] = df_all["strategies_fired"].astype(str)
    elif "strategies" in df_all.columns:
        norm["strategies_fired"] = df_all["strategies"].astype(str)
    else:
        norm["strategies_fired"] = "TrendFollowing"

    # Filter strictly WIN or LOSS and valid timestamps
    valid_mask = norm["outcome_eod"].isin(["WIN", "LOSS"]) & norm["date"].notna() & norm["time"].notna()
    labeled = norm[valid_mask].copy()
    labeled["label"] = (labeled["outcome_eod"] == "WIN").astype(int)
    labeled = labeled.drop_duplicates(subset=["date", "time", "direction", "outcome_eod"]).reset_index(drop=True)

    logger.info(f"Clean unique labeled trades for training: {len(labeled)} (Win Rate: {labeled['label'].mean()*100:.1f}%)")
    return labeled


def train_commodity_model():
    logger.info("=== Starting MCX Commodity 5-Minute ML Training Pipeline ===")
    candles = load_commodity_candles()
    labeled = load_labeled_commodity_trades()

    trainer = SignalForgeTrainer(min_samples=20, label_mode="classification")
    lookback = 60

    logger.info(f"Extracting features using lookback={lookback} bars across {len(labeled)} trade points...")
    X, y, feature_cols, meta_df = trainer.build_training_dataset(labeled, candles, lookback=lookback)
    logger.info(f"Feature matrix built: X shape={X.shape}, y balance={np.bincount(y)}")

    # Train ensemble
    train_scores, eval_results = trainer.train(X, y, feature_cols)
    logger.info(f"Training Complete. Train Scores: {train_scores}")
    logger.info(f"Cross-Validation Evaluation Results: {eval_results}")

    # Set Model Metadata
    meta = ModelMeta(
        trained_at=datetime.now().isoformat(),
        n_samples=len(X),
        n_features=len(feature_cols),
        win_rate=float(np.mean(y)),
        train_auc=float(np.mean(list(train_scores.values()))),
        val_auc=float(eval_results.get("val_auc", 0.0) or 0.0),
        val_precision=float(eval_results.get("val_precision", 0.0) or 0.0),
        val_recall=float(eval_results.get("val_recall", 0.0) or 0.0),
        recommended_threshold=float(eval_results.get("recommended_threshold", 0.62) or 0.62),
        calibration_method=str((eval_results.get("calibration", {}) or {}).get("method", "") or ""),
        calibration_coef=float((eval_results.get("calibration", {}) or {}).get("coef", 1.0) or 1.0),
        calibration_intercept=float((eval_results.get("calibration", {}) or {}).get("intercept", 0.0) or 0.0),
        feature_cols=feature_cols,
        model_names=list(trainer.ensemble.models.keys()),
    )
    trainer.ensemble.meta = meta

    # Ensure output directory exists
    models_dir = Path(ML_MODELS_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)
    target_path = models_dir / "silverm_5minute.pkl"

    # Save ensemble
    trainer.ensemble.save(target_path)
    logger.success(f"✅ Production Commodity ML Model successfully saved to {target_path}")

    # Also save compatibility aliases: silvermic_5minute.pkl and commodity_5minute.pkl
    for alias_name in ("silvermic_5minute.pkl", "commodity_5minute.pkl"):
        alias_path = models_dir / alias_name
        trainer.ensemble.save(alias_path)
        logger.info(f"✅ Commodity compatibility alias saved to {alias_path}")

    return target_path, meta


if __name__ == "__main__":
    train_commodity_model()
