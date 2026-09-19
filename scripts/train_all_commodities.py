"""
scripts/train_all_commodities.py — Concurrent Multi-Commodity ML Training Pipeline
================================================================================
Trains dedicated 3-model SignalForge Ensembles (XGBoost, LightGBM, RandomForest)
concurrently for all 4 supported MCX commodity mini instruments:
  1. SILVERM   -> ml/saved_models/silverm_5minute.pkl
  2. GOLDM     -> ml/saved_models/goldm_5minute.pkl
  3. CRUDEOILM -> ml/saved_models/crudeoilm_5minute.pkl (and crudeoil_5minute.pkl)
  4. NATGASM   -> ml/saved_models/natgasm_5minute.pkl (and naturalgas_5minute.pkl)

Supports multicore parallel execution so training all 4 completes rapidly.
"""

from __future__ import annotations

import sys
import os
import glob
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ml.features import extract, FEATURE_NAMES
from ml.model import SignalForgeEnsemble, ModelMeta
from ml.training.trainer import SignalForgeTrainer
from config.settings import ML_MODELS_DIR

CANDLE_FILES = {
    "SILVERM": ["SILVERM_dhan_5m.csv", "SILVERMIC_dhan_5m.csv", "SILVERM_5m.csv"],
    "GOLDM": ["GOLDM_dhan_5m.csv", "GOLDM_5m.csv"],
    "CRUDEOILM": ["CRUDEOIL_dhan_5m.csv", "CRUDEOIL_5m.csv"],
    "NATGASM": ["NATGAS_dhan_5m.csv", "NATGAS_5m.csv"],
}


def load_candles(symbol: str) -> pd.DataFrame:
    hist_dir = REPO_ROOT / "data" / "historical"
    candidates = CANDLE_FILES.get(symbol, [f"{symbol}_dhan_5m.csv"])
    target_csv = None
    for c in candidates:
        p = hist_dir / c
        if p.exists():
            target_csv = p
            break

    if not target_csv:
        raise FileNotFoundError(f"Missing historical candles for {symbol} in {hist_dir}")

    df = pd.read_csv(target_csv)
    ts_col = [col for col in df.columns if "time" in col.lower() or "date" in col.lower()][0]
    df[ts_col] = pd.to_datetime(df[ts_col])
    df = df.set_index(ts_col).sort_index()
    df.columns = [c.lower() for c in df.columns]
    for req in ("open", "high", "low", "close", "volume"):
        if req not in df.columns:
            raise ValueError(f"Candle data for {symbol} missing required column '{req}'")
    logger.info(f"[{symbol}] Loaded {len(df)} 5-minute candles ({df.index.min()} to {df.index.max()})")
    return df


def load_trades_for_symbol(symbol: str) -> pd.DataFrame:
    potential_files = []
    potential_files.extend(glob.glob(str(REPO_ROOT / "analysis" / "**" / "trade_ledger.csv"), recursive=True))
    potential_files.extend(glob.glob(str(REPO_ROOT / "scratch" / "**" / "trade_ledger.csv"), recursive=True))
    potential_files.extend(glob.glob(str(REPO_ROOT / "journal" / "*.csv")))

    dfs = []
    seen = set()
    for f in potential_files:
        p = Path(f)
        if p.exists() and str(p) not in seen:
            seen.add(str(p))
            try:
                raw = pd.read_csv(p)
                if len(raw) > 0:
                    dfs.append(raw)
            except Exception:
                pass

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    norm = pd.DataFrame()
    norm["date"] = combined.get("date", combined.get("trade_date", ""))
    norm["time"] = combined.get("entry_time", combined.get("time", combined.get("timestamp", "")))
    norm["direction"] = combined.get("direction", "")
    norm["outcome_eod"] = combined.get("outcome_eod", combined.get("exit_reason", "")).str.upper()
    norm["symbol"] = combined.get("symbol", combined.get("option_symbol", "")).str.upper()

    sym_clean = symbol.upper()
    prefixes = [sym_clean]
    if "SILVER" in sym_clean: prefixes = ["SILVERM", "SILVERMIC", "SILVER"]
    elif "GOLD" in sym_clean: prefixes = ["GOLDM", "GOLD"]
    elif "CRUDE" in sym_clean: prefixes = ["CRUDEOILM", "CRUDEOIL", "CRUDE"]
    elif "NAT" in sym_clean: prefixes = ["NATGASM", "NATGASMINI", "NATGAS", "NATURALGAS"]

    sym_mask = norm["symbol"].apply(lambda s: any(p in str(s) for p in prefixes))
    symbol_df = norm[sym_mask].copy()

    if len(symbol_df) < 15:
        symbol_df = norm.copy()

    symbol_df["outcome_eod"] = symbol_df["outcome_eod"].apply(
        lambda x: "WIN" if any(w in str(x) for w in ("WIN", "TARGET", "TRAIL", "PROFIT"))
        else ("LOSS" if any(l in str(x) for l in ("LOSS", "SL_HIT", "STOP")) else "UNKNOWN")
    )
    valid = symbol_df[symbol_df["outcome_eod"].isin(["WIN", "LOSS"])].copy()
    valid["label"] = (valid["outcome_eod"] == "WIN").astype(int)
    valid = valid.drop_duplicates(subset=["date", "time", "direction"]).reset_index(drop=True)
    return valid


def generate_candle_based_labels(candles: pd.DataFrame, symbol: str = "SILVERM", forward_bars: int = 6) -> pd.DataFrame:
    close = candles["close"].to_numpy()
    high = candles["high"].to_numpy()
    low = candles["low"].to_numpy()
    n = len(candles)

    sym_clean = symbol.upper()
    if "GOLD" in sym_clean:
        target_gain, target_loss = 0.0020, 0.0015  # 0.20% move on Gold (approx Rs 300)
    elif "SILVER" in sym_clean:
        target_gain, target_loss = 0.0035, 0.0025  # 0.35% move on Silver (approx Rs 800)
    elif "CRUDE" in sym_clean:
        target_gain, target_loss = 0.0040, 0.0030  # 0.40% move on Crude (approx Rs 40)
    else:
        target_gain, target_loss = 0.0050, 0.0035  # 0.50% move on Natural Gas (approx Rs 1.4)
    
    records = []
    step = 12
    for i in range(60, n - forward_bars, step):
        idx = candles.index[i]
        curr_close = close[i]
        future_high = np.max(high[i+1 : i+1+forward_bars])
        future_low = np.min(low[i+1 : i+1+forward_bars])
        
        long_gain = (future_high - curr_close) / curr_close
        long_loss = (curr_close - future_low) / curr_close
        long_win = 1 if long_gain >= target_gain and long_loss <= target_loss else 0
        records.append({
            "date": idx.strftime("%Y-%m-%d"),
            "time": idx.strftime("%H:%M:%S"),
            "direction": "BUY_CALL",
            "outcome_eod": "WIN" if long_win else "LOSS",
            "label": long_win,
        })
        
        short_gain = (curr_close - future_low) / curr_close
        short_loss = (future_high - curr_close) / curr_close
        short_win = 1 if short_gain >= target_gain and short_loss <= target_loss else 0
        records.append({
            "date": idx.strftime("%Y-%m-%d"),
            "time": idx.strftime("%H:%M:%S"),
            "direction": "BUY_PUT",
            "outcome_eod": "WIN" if short_win else "LOSS",
            "label": short_win,
        })

    return pd.DataFrame(records)


def train_single_commodity(symbol: str) -> tuple[str, dict]:
    logger.info(f"[{symbol}] Training Pipeline Initiated...")
    candles = load_candles(symbol)
    trades = load_trades_for_symbol(symbol)
    
    if len(trades) < 25:
        logger.info(f"[{symbol}] Generating bootstrap labels directly from price action ({len(trades)} journal rows available)")
        bootstrap_trades = generate_candle_based_labels(candles, symbol=symbol)
        trades = pd.concat([trades, bootstrap_trades], ignore_index=True) if len(trades) > 0 else bootstrap_trades

    trainer = SignalForgeTrainer(min_samples=20, label_mode="classification")
    lookback = 60

    X, y, feature_cols, meta_df = trainer.build_training_dataset(trades, candles, lookback=lookback)
    logger.info(f"[{symbol}] Feature matrix shape: X={X.shape}, y={np.bincount(y)}")

    # Compute balanced pattern and class weights
    sample_weights = trainer._compute_sample_weights(meta_df, y)

    train_scores, eval_results = trainer.train(X, y, feature_cols, sample_weight=sample_weights)
    logger.info(f"[{symbol}] CV Results: val_auc={eval_results.get('val_auc', 0):.3f}, val_prec={eval_results.get('val_precision', 0):.3f}")

    calibration = eval_results.get("calibration", {}) or {}
    meta = ModelMeta(
        trained_at=datetime.now().isoformat(),
        n_samples=len(X),
        n_features=len(feature_cols),
        win_rate=float(np.mean(y)),
        train_auc=float(np.mean(list(train_scores.values()))),
        val_auc=float(eval_results.get("val_auc", 0.0) or 0.0),
        val_precision=float(eval_results.get("val_precision", 0.0) or 0.0),
        val_recall=float(eval_results.get("val_recall", 0.0) or 0.0),
        recommended_threshold=float(eval_results.get("recommended_threshold", 0.55) or 0.55),
        calibration_method=str(calibration.get("method", "platt") or "platt"),
        calibration_coef=float(calibration.get("coef", 1.0) or 1.0),
        calibration_intercept=float(calibration.get("intercept", 0.0) or 0.0),
        feature_cols=feature_cols,
        model_names=list(trainer.ensemble.models.keys()),
    )
    trainer.ensemble.meta = meta

    models_dir = Path(ML_MODELS_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    primary_name = f"{symbol.lower()}_5minute.pkl"
    out_path = models_dir / primary_name
    trainer.ensemble.save(out_path)
    logger.success(f"[{symbol}] Model saved to {out_path}")

    aliases = []
    if symbol == "SILVERM": aliases = ["silvermic_5minute.pkl", "commodity_5minute.pkl"]
    elif symbol == "CRUDEOILM": aliases = ["crudeoil_5minute.pkl"]
    elif symbol == "NATGASM": aliases = ["naturalgas_5minute.pkl", "natgasmini_5minute.pkl"]
    elif symbol == "GOLDM": aliases = ["gold_5minute.pkl"]

    for a in aliases:
        trainer.ensemble.save(models_dir / a)

    return symbol, eval_results


def train_all_commodities_parallel(symbols: list[str] | None = None) -> dict[str, dict]:
    target_symbols = symbols or ["SILVERM", "GOLDM", "CRUDEOILM", "NATGASM"]
    logger.info(f"🚀 Launching Parallel Training for: {target_symbols}")
    
    results = {}
    with ProcessPoolExecutor(max_workers=min(4, len(target_symbols))) as executor:
        futures = {executor.submit(train_single_commodity, sym): sym for sym in target_symbols}
        for future in as_completed(futures):
            sym = futures[future]
            try:
                sym_name, eval_res = future.result()
                results[sym_name] = eval_res
                logger.success(f"🎉 [{sym_name}] Completed successfully.")
            except Exception as exc:
                logger.error(f"❌ [{sym}] Failed: {exc}")
                results[sym] = {"error": str(exc)}

    logger.info("=" * 60)
    logger.info("  MULTI-COMMODITY TRAINING COMPLETE")
    for sym, res in results.items():
        logger.info(f"  - {sym:<10}: AUC={res.get('val_auc', 0):.3f} | Precision={res.get('val_precision', 0):.3f}")
    logger.info("=" * 60)
    return results


if __name__ == "__main__":
    train_all_commodities_parallel()
