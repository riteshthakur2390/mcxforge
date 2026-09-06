"""
ml/training/evaluate.py — Model Evaluation Tools
==================================================
Run after training to get detailed performance metrics.
Shows feature importance, confusion matrix, threshold analysis.

Usage:
    python ml/training/evaluate.py
    python ml/training/evaluate.py --threshold 0.65
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Evaluate SignalForge ML model")
    parser.add_argument("--threshold", type=float, default=0.62,
                        help="Prediction threshold (default: 0.62)")
    parser.add_argument("--top-features", type=int, default=15,
                        help="Number of top features to show")
    parser.add_argument(
        "--thresholds",
        type=str,
        default="0.52,0.55,0.58,0.63",
        help="Comma-separated thresholds for sweep report",
    )
    parser.add_argument(
        "--min-group-trades",
        type=int,
        default=12,
        help="Minimum trades required to show combo/regime breakdown rows",
    )
    args = parser.parse_args()

    from ml.model import SignalForgeEnsemble
    from config.settings import ML_MODELS_DIR, LIVE_TIMEFRAME, ML_THRESHOLD_OVERRIDE
    from ml.training.trainer import SignalForgeTrainer

    model_path = Path(ML_MODELS_DIR) / f"nifty_{LIVE_TIMEFRAME}.pkl"
    if not model_path.exists():
        print(f"\n❌ No trained model found at {model_path}")
        print("   Run: python scripts/train_ml.py")
        sys.exit(1)

    print(f"\n{'━'*60}")
    print(f"  SignalForge ML — Model Evaluation")
    print(f"{'━'*60}\n")

    # Load model
    ensemble = SignalForgeEnsemble()
    if not ensemble.load(model_path):
        print("❌ Failed to load model")
        sys.exit(1)

    meta = ensemble.meta
    print(f"  Model: {model_path.name}")
    print(f"  Trained:   {meta.trained_at[:19] if meta.trained_at else 'unknown'}")
    print(f"  Samples:   {meta.n_samples}")
    print(f"  Features:  {meta.n_features}")
    print(f"  Win rate:  {meta.win_rate*100:.1f}% (in training data)")
    print(f"  Train AUC: {meta.train_auc:.4f}")
    print(f"  Val AUC:   {meta.val_auc:.4f}")
    print(f"  Val Prec:  {meta.val_precision:.4f}")
    print(f"  Val Recall:{meta.val_recall:.4f}")
    print(f"  Models:    {', '.join(meta.model_names)}")
    print(f"  Calibration: {meta.calibration_method or 'none'}")
    if meta.calibration_method:
        print(
            f"  Calibration params: coef={meta.calibration_coef:.4f} "
            f"| intercept={meta.calibration_intercept:.4f}"
        )
    print()

    # Feature importance
    print(f"  Top {args.top_features} Features by Importance:")
    print(f"  {'Feature':<25} {'Importance':>12}")
    print(f"  {'-'*40}")
    imp_df = ensemble.get_feature_importance(args.top_features)
    if not imp_df.empty:
        for _, row in imp_df.iterrows():
            bar = "█" * int(row["importance"] * 100)
            print(f"  {row['feature']:<25} {row['importance']:>8.4f}  {bar}")
    else:
        print("  (feature importance not available)")

    print()

    trainer = SignalForgeTrainer(min_samples=1, include_archive=True)
    labeled = trainer.load_journal()
    candles = trainer.load_candles()
    X, y, feature_cols, meta_df = trainer.build_training_dataset(labeled, candles, lookback=60)
    eval_results = ensemble.evaluate_cv(X, y, feature_cols, n_splits=5, include_oof=True)

    oof_df = meta_df.iloc[eval_results["oof_indices"]].copy().reset_index(drop=True)
    oof_df["proba"] = np.array(eval_results["oof_proba"])
    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]

    print("  Threshold Sweep (OOF / time-series CV):")
    print(f"  {'Thr':>5} {'Trades':>8} {'WR':>7} {'Prec':>7} {'Recall':>8} {'PF':>7} {'INR':>12}")
    print(f"  {'-'*68}")
    for threshold in thresholds:
        row = _summarize_threshold(oof_df, threshold)
        marker = "  ← current" if abs(threshold - ensemble.decision_threshold) < 0.005 else ""
        print(
            f"  {threshold:>5.2f} {row['trades']:>8} {row['win_rate']:>6.1f}% "
            f"{row['precision']:>6.1f}% {row['recall']:>7.1f}% "
            f"{row['profit_factor']:>7.2f} {row['realized_inr']:>12,.0f}{marker}"
        )
    print()
    _print_policy_recommendation(
        thresholds=thresholds,
        oof_df=oof_df,
        model_threshold=ensemble.decision_threshold,
        live_override=ML_THRESHOLD_OVERRIDE,
    )

    print()
    print("  By Combo")
    for threshold in thresholds:
        print(f"  Threshold {threshold:.2f}")
        _print_group_table(
            _group_summary(oof_df, threshold, "combo", args.min_group_trades),
            key_name="combo",
        )
        print()

    print("  By Regime")
    for threshold in thresholds:
        print(f"  Threshold {threshold:.2f}")
        _print_group_table(
            _group_summary(oof_df, threshold, "regime", max(6, args.min_group_trades // 2)),
            key_name="regime",
        )
        print()

    print(f"  Current threshold in model: {ensemble.decision_threshold:.2f}")
    print("  Use this report to tune ML threshold and secondary lane coverage.")
    print(f"\n{'━'*60}\n")


def _selected_df(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    return df[df["proba"] >= threshold].copy()


def _profit_factor(frame: pd.DataFrame) -> float:
    gains = float(frame.loc[frame["realized_pnl"] > 0, "realized_pnl"].sum())
    losses = float(-frame.loc[frame["realized_pnl"] < 0, "realized_pnl"].sum())
    if losses <= 0:
        return 999.0 if gains > 0 else 0.0
    return gains / losses


def _summarize_threshold(df: pd.DataFrame, threshold: float) -> dict:
    selected = _selected_df(df, threshold)
    wins = int(selected["label"].sum()) if not selected.empty else 0
    total = len(selected)
    positives = int(df["label"].sum()) if len(df) else 0
    return {
        "trades": total,
        "win_rate": (wins / total * 100.0) if total else 0.0,
        "precision": (wins / total * 100.0) if total else 0.0,
        "recall": (wins / positives * 100.0) if positives else 0.0,
        "profit_factor": _profit_factor(selected) if total else 0.0,
        "realized_inr": float(selected["realized_pnl"].sum()) if total else 0.0,
    }


def _group_summary(df: pd.DataFrame, threshold: float, group_col: str, min_trades: int) -> pd.DataFrame:
    selected = _selected_df(df, threshold)
    if selected.empty:
        return pd.DataFrame()

    rows = []
    total_losses = float(-selected.loc[selected["realized_pnl"] < 0, "realized_pnl"].sum())
    positives = int(selected["label"].sum())
    for key, group in selected.groupby(group_col):
        if len(group) < min_trades:
            continue
        wins = int(group["label"].sum())
        losses = float(-group.loc[group["realized_pnl"] < 0, "realized_pnl"].sum())
        rows.append({
            group_col: str(key or "UNKNOWN"),
            "trades": len(group),
            "win_rate": wins / len(group) * 100.0,
            "precision": wins / len(group) * 100.0,
            "recall_share": wins / positives * 100.0 if positives else 0.0,
            "profit_factor": _profit_factor(group),
            "realized_inr": float(group["realized_pnl"].sum()),
            "loss_share": losses / total_losses * 100.0 if total_losses else 0.0,
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["realized_inr", "win_rate", "trades"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def _print_group_table(df: pd.DataFrame, key_name: str) -> None:
    if df.empty:
        print("  (no groups met minimum trade count)")
        return
    print(
        f"  {key_name:<32} {'n':>5} {'WR':>7} {'Recall':>8} {'PF':>7} "
        f"{'Loss%':>8} {'INR':>12}"
    )
    print(f"  {'-'*86}")
    for _, row in df.iterrows():
        print(
            f"  {str(row[key_name])[:32]:<32} {int(row['trades']):>5} "
            f"{row['win_rate']:>6.1f}% {row['recall_share']:>7.1f}% "
            f"{row['profit_factor']:>7.2f} {row['loss_share']:>7.1f}% "
            f"{row['realized_inr']:>12,.0f}"
        )


def _print_policy_recommendation(
    *,
    thresholds: list[float],
    oof_df: pd.DataFrame,
    model_threshold: float,
    live_override: float,
) -> None:
    rows = [
        {"threshold": t, **_summarize_threshold(oof_df, t)}
        for t in thresholds
    ]
    viable_60 = [r for r in rows if r["precision"] >= 60.0 and r["trades"] > 0]
    viable_70 = [r for r in rows if r["precision"] >= 70.0 and r["trades"] > 0]
    coverage_pick = max(viable_60, key=lambda r: (r["realized_inr"], r["trades"]), default=None)
    precision_pick = max(viable_70, key=lambda r: (r["precision"], r["profit_factor"]), default=None)

    print("  Policy Recommendation:")
    print(
        f"  Model threshold={model_threshold:.2f} | "
        f"Live override={live_override:.2f}"
    )
    if coverage_pick:
        print(
            f"  Coverage-first pick: {coverage_pick['threshold']:.2f} | "
            f"trades={coverage_pick['trades']} | "
            f"precision={coverage_pick['precision']:.1f}% | "
            f"INR={coverage_pick['realized_inr']:,.0f}"
        )
    if precision_pick:
        print(
            f"  Precision-first pick: {precision_pick['threshold']:.2f} | "
            f"trades={precision_pick['trades']} | "
            f"precision={precision_pick['precision']:.1f}% | "
            f"PF={precision_pick['profit_factor']:.2f}"
        )
    if live_override > 0 and abs(live_override - model_threshold) > 0.005:
        print(
            "  Live policy differs from saved model threshold. "
            "That is acceptable if you are intentionally trading for more coverage."
        )


if __name__ == "__main__":
    main()
