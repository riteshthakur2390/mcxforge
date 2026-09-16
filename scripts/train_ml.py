"""
scripts/train_ml.py — MCXForge ML training entrypoint
=====================================================
Single entrypoint for the shared ML pipeline:
    journal -> ml.features.extract() -> ml.model.CommodityMLEnsemble
"""

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from config.settings import ML_MODELS_DIR, BACKTEST_TIMEFRAME, LIVE_TIMEFRAME
from core.bus import Topic
from ml.model import SignalForgeEnsemble
from ml.training.trainer import SignalForgeTrainer


def notify_system(model_path: str) -> None:
    """
    Publish MODEL_RETRAINED to the running container so Agent 3 can reload
    without a full restart. Safe to skip if docker is not running.
    """
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=mcxforge", "--filter", "name=signalforge", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        running_names = result.stdout.strip().splitlines()
        target_container = None
        for name in running_names:
            if "mcxforge" in name or "signalforge" in name:
                target_container = name
                break

        if not target_container:
            print("  ℹ️  Container not running — restart to load new model.")
            return

        cmd = (
            "python -c \""
            "import asyncio, sys; sys.path.insert(0, '/app'); "
            "from core.bus import Topic, get_bus; "
            "bus = get_bus(); "
            "asyncio.run(bus.publish(Topic.MODEL_RETRAINED, "
            "{'model_path': '" + model_path + "'}, 'train_script'))"
            "\""
        )
        subprocess.run(
            ["docker", "exec", target_container, "bash", "-c", cmd],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        print(f"  ✅ {Topic.MODEL_RETRAINED} event sent — Agent 3 can reload without restart.")
    except Exception as e:
        print(f"  ⚠️  Could not notify container: {e}")
        print("     Run: docker-compose restart to load the new model.")


def _load_model_metrics(path: Path) -> dict:
    ensemble = SignalForgeEnsemble()
    if not ensemble.load(path):
        return {}
    meta = ensemble.meta
    return {
        "val_auc": float(getattr(meta, "val_auc", 0.0) or 0.0),
        "val_precision": float(getattr(meta, "val_precision", 0.0) or 0.0),
        "walk_forward_auc": float(getattr(meta, "walk_forward_auc", 0.0) or 0.0),
        "walk_forward_precision": float(getattr(meta, "walk_forward_precision", 0.0) or 0.0),
        "walk_forward_windows": int(getattr(meta, "walk_forward_windows", 0) or 0),
        "n_samples": int(getattr(meta, "n_samples", 0) or 0),
    }


def _promotion_decision(
    results: dict,
    current_metrics: dict,
    min_val_auc: float,
    min_walk_forward_auc: float,
    min_walk_forward_windows: int,
    require_current_improvement: bool,
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    val_auc = float(results.get("val_auc", 0.0) or 0.0)
    wf_auc = float(results.get("walk_forward_auc", 0.0) or 0.0)
    wf_windows = int(results.get("walk_forward_windows", 0) or 0)

    if val_auc < min_val_auc:
        failures.append(f"val_auc {val_auc:.4f} < {min_val_auc:.4f}")
    if wf_windows < min_walk_forward_windows:
        failures.append(f"walk_forward_windows {wf_windows} < {min_walk_forward_windows}")
    if wf_windows and wf_auc < min_walk_forward_auc:
        failures.append(f"walk_forward_auc {wf_auc:.4f} < {min_walk_forward_auc:.4f}")

    current_val_auc = float(current_metrics.get("val_auc", 0.0) or 0.0)
    if require_current_improvement and current_val_auc > 0 and val_auc + 0.005 < current_val_auc:
        failures.append(f"candidate val_auc {val_auc:.4f} is worse than current {current_val_auc:.4f}")

    return not failures, failures


def _promote_candidate(candidate_path: Path, production_path: Path) -> Path | None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = None
    production_path.parent.mkdir(parents=True, exist_ok=True)
    if production_path.exists():
        backup_path = production_path.with_name(
            f"{production_path.stem}.backup_{timestamp}{production_path.suffix}"
        )
        shutil.copy2(production_path, backup_path)
    shutil.copy2(candidate_path, production_path)
    return backup_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MCXForge ML models")
    parser.add_argument(
        "--min-trades",
        type=int,
        default=200,
        help="Minimum labeled trades required before training",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=60,
        help="Candles to include in each feature window",
    )
    parser.add_argument(
        "--no-notify",
        action="store_true",
        help="Skip sending MODEL_RETRAINED to a running container",
    )
    parser.add_argument(
        "--no-archive",
        action="store_true",
        help="Ignore journal/old_csv archive rows during training",
    )
    parser.add_argument(
        "--csv",
        action="append",
        default=[],
        help="Explicit CSV file or glob to use for training. Repeatable.",
    )
    parser.add_argument(
        "--backtest-results",
        action="store_true",
        help="Train from backtesting/results/backtest_trades_*.csv exports.",
    )
    parser.add_argument(
        "--live-and-backtest",
        action="store_true",
        help="Train from both live journal CSVs and backtest result CSVs.",
    )
    parser.add_argument(
        "--label-mode",
        type=str,
        default="classification",
        choices=["classification", "return"],
        help="Label objective mode (classification=WIN/LOSS, return=expected return aware).",
    )
    parser.add_argument(
        "--unsafe-direct-save",
        action="store_true",
        help="Save directly to the live model path, bypassing candidate promotion safeguards.",
    )
    parser.add_argument(
        "--force-promote",
        action="store_true",
        help="Promote candidate even if safety gates fail.",
    )
    parser.add_argument(
        "--no-current-compare",
        action="store_true",
        help="Do not require candidate validation AUC to stay near the current model.",
    )
    parser.add_argument(
        "--min-val-auc",
        type=float,
        default=0.55,
        help="Minimum validation AUC required for safe promotion.",
    )
    parser.add_argument(
        "--min-walk-forward-auc",
        type=float,
        default=0.52,
        help="Minimum walk-forward AUC required when walk-forward windows are available.",
    )
    parser.add_argument(
        "--min-walk-forward-windows",
        type=int,
        default=2,
        help="Minimum walk-forward windows required for safe promotion.",
    )
    args = parser.parse_args()

    csv_inputs = list(args.csv or [])
    if args.backtest_results:
        csv_inputs.append(str(REPO_ROOT / "backtesting" / "results" / "backtest_trades_*.csv"))
    if args.live_and_backtest:
        csv_inputs.extend([
            str(REPO_ROOT / "journal" / "signals_*.csv"),
            str(REPO_ROOT / "journal" / "old_csv" / "signals_*.csv"),
            str(REPO_ROOT / "backtesting" / "results" / "backtest_trades_*.csv"),
        ])

    print()
    print("━" * 60)
    print("  MCXForge ML Training")
    print("━" * 60)
    print(f"  Data timeframe:  {BACKTEST_TIMEFRAME}")
    print(f"  Model timeframe: {LIVE_TIMEFRAME}")
    print(f"  Min trades: {args.min_trades}")
    print(f"  Lookback:   {args.lookback}")
    print(f"  Archive:    {'off' if args.no_archive else 'on'}")
    print(f"  Source:     {'mixed live/backtest CSVs' if args.live_and_backtest else ('explicit CSV inputs' if csv_inputs else 'journal CSVs')}")
    production_path = Path(ML_MODELS_DIR) / f"nifty_{LIVE_TIMEFRAME}.pkl"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate_path = (
        production_path
        if args.unsafe_direct_save
        else production_path.with_name(f"{production_path.stem}.candidate_{timestamp}{production_path.suffix}")
    )
    current_metrics = _load_model_metrics(production_path) if production_path.exists() else {}
    if current_metrics:
        print(
            f"  Current:    val_auc={current_metrics.get('val_auc', 0.0):.4f} | "
            f"wf_auc={current_metrics.get('walk_forward_auc', 0.0):.4f} | "
            f"samples={current_metrics.get('n_samples', 0)}"
        )
    print(f"  Candidate:  {candidate_path}")
    print()

    trainer = SignalForgeTrainer(
        min_samples=args.min_trades,
        include_archive=not args.no_archive,
        csv_paths=csv_inputs,
        label_mode=args.label_mode,
        model_path=candidate_path,
    )

    try:
        results = trainer.run(lookback=args.lookback)
    except FileNotFoundError as e:
        print(f"  ❌ {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"  ⚠️  {e}")
        sys.exit(1)

    print("  Training complete")
    print(f"  Samples:    {results['n_samples']}")
    print(f"  Features:   {results['n_features']}")
    print(f"  Win rate:   {results['win_rate'] * 100:.1f}%")
    print(f"  Train AUC:  {results['train_auc']:.4f}")
    print(f"  Val AUC:    {results['val_auc']:.4f}")
    print(f"  Val Prec:   {results['val_precision']:.4f}")
    print(f"  Val Recall: {results['val_recall']:.4f}")
    print(
        f"  WalkFwd:    auc={results.get('walk_forward_auc', 0.0):.4f} | "
        f"precision={results.get('walk_forward_precision', 0.0):.4f} | "
        f"windows={results.get('walk_forward_windows', 0)}"
    )
    print(f"  Threshold:  {results['recommended_threshold']:.2f}")
    print(f"  Models:     {', '.join(results['models'])}")
    print(f"  Saved to:   {results['model_path']}")
    print()

    promoted = args.unsafe_direct_save
    backup_path = None
    if not args.unsafe_direct_save:
        should_promote, promotion_failures = _promotion_decision(
            results=results,
            current_metrics=current_metrics,
            min_val_auc=args.min_val_auc,
            min_walk_forward_auc=args.min_walk_forward_auc,
            min_walk_forward_windows=args.min_walk_forward_windows,
            require_current_improvement=not args.no_current_compare,
        )
        if should_promote or args.force_promote:
            backup_path = _promote_candidate(candidate_path, production_path)
            promoted = True
            print(f"  Promoted:   {production_path}")
            if backup_path:
                print(f"  Backup:     {backup_path}")
            if promotion_failures and args.force_promote:
                print("  ⚠️  Force-promoted despite:")
                for failure in promotion_failures:
                    print(f"     - {failure}")
        else:
            print("  Kept current live model. Candidate was not promoted:")
            for failure in promotion_failures:
                print(f"     - {failure}")
            print(f"  Candidate retained for review: {candidate_path}")

    if results["val_auc"] >= 0.65:
        print("  ✅ Good AUC (>= 0.65) — ML should help signal quality")
    elif results["val_auc"] >= 0.55:
        print("  ⚠️  Marginal AUC (0.55–0.65) — collect more labeled trades")
    else:
        print("  🔴 Low AUC (< 0.55) — do not trust ML gate yet")

    print()
    print("  Evaluate in detail:")
    print("  python3 ml/training/evaluate.py")

    if not args.no_notify and promoted:
        print()
        notify_system(str(production_path))
    elif not promoted:
        print()
        print("  MODEL_RETRAINED not sent because production model was unchanged.")

    print()
    print("  Saved models directory:")
    print(f"  {Path(ML_MODELS_DIR).resolve()}")
    print("━" * 60)
    print()


if __name__ == "__main__":
    main()
