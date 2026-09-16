"""
Daily ML maintenance helper.

Use this before market open to keep training attached to operations:
    ./venv/bin/python scripts/retrain_if_needed.py
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytz

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config.settings import JOURNAL_DIR, ML_MODELS_DIR, ML_RETRAIN_DAYS, LIVE_TIMEFRAME
from ml.model import SignalForgeEnsemble
from ml.training.trainer import SignalForgeTrainer

IST = pytz.timezone("Asia/Kolkata")


def model_age_days(path: Path) -> int | None:
    ensemble = SignalForgeEnsemble()
    if not ensemble.load(path):
        return None
    trained_at = str(getattr(ensemble.meta, "trained_at", "") or "")
    if not trained_at:
        return None
    ts = datetime.fromisoformat(trained_at)
    if ts.tzinfo is None:
        ts = IST.localize(ts)
    else:
        ts = ts.astimezone(IST)
    return max((datetime.now(IST) - ts).days, 0)


def model_trained_at(path: Path) -> datetime | None:
    ensemble = SignalForgeEnsemble()
    if not ensemble.load(path):
        return None
    trained_at = str(getattr(ensemble.meta, "trained_at", "") or "")
    if not trained_at:
        return None
    ts = datetime.fromisoformat(trained_at)
    if ts.tzinfo is None:
        return IST.localize(ts)
    return ts.astimezone(IST)


def closed_trades_since(ts: datetime | None) -> int:
    csv_files = sorted(Path(JOURNAL_DIR).glob("signals_*.csv"))
    if not csv_files:
        return 0
    frames: list[pd.DataFrame] = []
    for path in csv_files[-15:]:
        try:
            frames.append(pd.read_csv(path))
        except Exception:
            continue
    if not frames:
        return 0
    df = pd.concat(frames, ignore_index=True)
    if "outcome_eod" not in df.columns:
        return 0
    df = df[df["outcome_eod"].astype(str).isin(["WIN", "LOSS"])].copy()
    if df.empty:
        return 0
    if "entry_time" in df.columns:
        entry_ts = pd.to_datetime(df["entry_time"], errors="coerce")
    else:
        date = df.get("date", pd.Series("", index=df.index)).astype(str)
        time = df.get("time", pd.Series("", index=df.index)).astype(str)
        entry_ts = pd.to_datetime(date + " " + time, errors="coerce")
    if ts is not None:
        cutoff = pd.Timestamp(ts)
        df = df[entry_ts >= cutoff]
    if "signal_id" in df.columns:
        df = df.drop_duplicates(subset=["signal_id"], keep="last")
    return len(df)


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain MCXForge ML model if stale")
    parser.add_argument("--min-trades", type=int, default=200)
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--max-age-days", type=int, default=ML_RETRAIN_DAYS)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-archive", action="store_true")
    parser.add_argument("--live-and-backtest", action="store_true")
    parser.add_argument(
        "--min-new-closed-trades",
        type=int,
        default=25,
        help="Retrain early when this many new closed trades are available since last model train.",
    )
    args = parser.parse_args()

    active_sym = os.getenv("INSTRUMENT", "SILVERM").lower()
    model_path = Path(ML_MODELS_DIR) / f"{active_sym}_{LIVE_TIMEFRAME}.pkl"
    if not model_path.exists():
        fallback_model = Path(ML_MODELS_DIR) / f"silvermic_{LIVE_TIMEFRAME}.pkl"
        if fallback_model.exists():
            model_path = fallback_model
    age_days = model_age_days(model_path) if model_path.exists() else None
    trained_at = model_trained_at(model_path) if model_path.exists() else None
    new_closed = closed_trades_since(trained_at)

    needs_retrain = bool(args.force)
    reasons: list[str] = []
    if age_days is None:
        needs_retrain = True
        reasons.append("missing_or_invalid_model_age")
    elif age_days >= args.max_age_days:
        needs_retrain = True
        reasons.append(f"age={age_days}d >= {args.max_age_days}d")

    if new_closed >= args.min_new_closed_trades:
        needs_retrain = True
        reasons.append(f"new_closed_trades={new_closed} >= {args.min_new_closed_trades}")

    if not needs_retrain:
        print(
            f"Model is fresh enough: {model_path.name} age={age_days}d "
            f"< threshold={args.max_age_days}d | "
            f"new_closed_trades={new_closed} < {args.min_new_closed_trades}"
        )
        return

    print(
        f"Retraining model | path={model_path} | "
        f"age={'missing' if age_days is None else str(age_days) + 'd'} | "
        f"new_closed_trades={new_closed} | "
        f"reason={'; '.join(reasons) if reasons else 'force'}"
    )
    trainer = SignalForgeTrainer(
        min_samples=args.min_trades,
        include_archive=not args.no_archive,
        csv_paths=(
            [
                "journal/signals_*.csv",
                "journal/old_csv/signals_*.csv",
                "backtesting/results/backtest_trades_*.csv",
            ]
            if args.live_and_backtest else None
        ),
    )
    results = trainer.run(lookback=args.lookback)
    print(
        f"Training complete | samples={results['n_samples']} | "
        f"val_auc={results['val_auc']:.4f} | "
        f"val_precision={results['val_precision']:.4f} | "
        f"threshold={results['recommended_threshold']:.2f}"
    )


if __name__ == "__main__":
    main()
