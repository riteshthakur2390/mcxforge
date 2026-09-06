from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from config.settings import (
    BACKTEST_BROKERAGE_PER_ORDER,
    BACKTEST_TRANSACTION_COST_PCT,
    FINAL_DECISION_MAX_RISK_PCT,
    FINAL_DECISION_MIN_EXPECTANCY_INR,
    FINAL_DECISION_MIN_HISTORY,
    FINAL_DECISION_MIN_RISK_PCT,
    FINAL_DECISION_LOT_FALLBACK_RISK_MULT,
    FINAL_DECISION_SETUP_STRENGTH_DEFAULT,
    FINAL_DECISION_QUALITY_FLOOR_MAX,
    FINAL_DECISION_RANK_WEIGHT,
    FINAL_DECISION_SETUP_WEIGHT,
    FINAL_DECISION_WIN_PROB_MIN,
    FINAL_DECISION_WIN_PROB_MAX,
    FINAL_DECISION_QUALITY_WEIGHT,
    FINAL_DECISION_HIST_WEIGHT,
    FINAL_DECISION_LOSS_FLOOR,
    FINAL_DECISION_AVG_WIN_GROSS_WEIGHT,
    FINAL_DECISION_AVG_WIN_HIST_WEIGHT,
    FINAL_DECISION_AVG_WIN_SETUP_WEIGHT,
    FINAL_DECISION_AVG_LOSS_GROSS_WEIGHT,
    FINAL_DECISION_AVG_LOSS_HIST_WEIGHT,
    FINAL_DECISION_RISK_ANCHOR_ATR,
    FINAL_DECISION_VOL_FACTOR_MIN,
    FINAL_DECISION_VOL_FACTOR_MAX,
    FINAL_DECISION_FALLBACK_WIN_RATE,
    FINAL_DECISION_FALLBACK_AVG_WIN_PCT,
    FINAL_DECISION_FALLBACK_AVG_LOSS_PCT,
    FINAL_DECISION_JOURNAL_LOOKBACK,
    JOURNAL_DIR,
    MAX_POSITION_LOTS,
    NIFTY_LOT_SIZE,
    PAPER_TRADING_CAPITAL,
    PLANNER_ENTRY_SLIPPAGE_PCT,
)
from utils.option_utils import apply_option_slippage, estimate_round_trip_costs


@dataclass(frozen=True)
class HistoricalStats:
    sample_size: int
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    source: str


@dataclass(frozen=True)
class FinalDecision:
    tradeable: bool
    reason: str
    entry_premium: float
    target_premium: float
    stop_premium: float
    risk_pct: float
    risk_budget_inr: float
    desired_lots: int
    quantity: int
    win_prob: float
    avg_win_inr: float
    avg_loss_inr: float
    expectancy_inr: float
    reward_risk: float
    slippage_pct: float
    historical_stats: HistoricalStats


class FinalDecisionLayer:
    def __init__(
        self,
        *,
        journal_dir: str = JOURNAL_DIR,
        capital: float = PAPER_TRADING_CAPITAL,
        slippage_pct: float = PLANNER_ENTRY_SLIPPAGE_PCT,
    ) -> None:
        self._journal_dir = Path(journal_dir)
        self._capital = float(capital)
        self._slippage_pct = float(max(slippage_pct, 0.0))
        self._cache_key: tuple[tuple[str, float], ...] | None = None
        self._rows: list[dict] = []

    def evaluate(
        self,
        *,
        entry_premium: float,
        target_premium: float,
        stop_premium: float,
        ml_success_prob: float,
        ml_rank_score: float,
        atr_pct: float,
        strategies: list[str],
        regime: str,
        setup: dict | None = None,
        vix_regime: str = "FLAT",
        lot_size: int = 65,
    ) -> FinalDecision:
        lot_size = max(int(lot_size or 65), 1)
        stats = self._historical_stats(strategies=strategies, regime=regime)
        setup = setup or {}
        setup_strength = max(0.0, min(1.0, float(setup.get("setup_strength", FINAL_DECISION_SETUP_STRENGTH_DEFAULT) or FINAL_DECISION_SETUP_STRENGTH_DEFAULT)))
        adjusted_entry = apply_option_slippage(entry_premium, "BUY", self._slippage_pct)
        adjusted_target = apply_option_slippage(target_premium, "SELL", self._slippage_pct)
        adjusted_stop = apply_option_slippage(stop_premium, "SELL", self._slippage_pct)

        # ML is a ranker in this pipeline, so expectancy should respect the
        # approved rank/setup quality instead of re-blocking on raw model output.
        quality_floor = max(
            float(ml_success_prob),
            min(FINAL_DECISION_QUALITY_FLOOR_MAX, float(ml_rank_score) * FINAL_DECISION_RANK_WEIGHT + setup_strength * FINAL_DECISION_SETUP_WEIGHT),
        )
        win_prob = max(
            FINAL_DECISION_WIN_PROB_MIN,
            min(FINAL_DECISION_WIN_PROB_MAX, quality_floor * FINAL_DECISION_QUALITY_WEIGHT + stats.win_rate * FINAL_DECISION_HIST_WEIGHT),
        )
        signal_strength = max(
            0.0,
            min(
                1.0,
                (
                    quality_floor
                    + float(ml_rank_score)
                    + setup_strength
                ) / 3,
            ),
        )
        risk_pct = self._risk_pct(signal_strength=signal_strength, atr_pct=atr_pct)
        risk_budget_inr = round(self._capital * risk_pct / 100, 2)

        gross_win_per_lot = max(adjusted_target - adjusted_entry, 0.0) * lot_size
        gross_loss_per_lot = max(adjusted_entry - adjusted_stop, FINAL_DECISION_LOSS_FLOOR) * lot_size
        costs_target = estimate_round_trip_costs(
            adjusted_entry,
            adjusted_target,
            lot_size,
            BACKTEST_BROKERAGE_PER_ORDER,
            BACKTEST_TRANSACTION_COST_PCT,
        )
        costs_stop = estimate_round_trip_costs(
            adjusted_entry,
            adjusted_stop,
            lot_size,
            BACKTEST_BROKERAGE_PER_ORDER,
            BACKTEST_TRANSACTION_COST_PCT,
        )

        hist_win_inr = adjusted_entry * lot_size * (stats.avg_win_pct / 100)
        hist_loss_inr = adjusted_entry * lot_size * (stats.avg_loss_pct / 100)
        setup_win_inr = max(float(setup.get("expected_move", 0.0) or 0.0), 0.0) * lot_size
        avg_win_inr = round(
            max(
                0.0,
                gross_win_per_lot * FINAL_DECISION_AVG_WIN_GROSS_WEIGHT + hist_win_inr * FINAL_DECISION_AVG_WIN_HIST_WEIGHT + setup_win_inr * FINAL_DECISION_AVG_WIN_SETUP_WEIGHT - costs_target,
            ),
            2,
        )
        avg_loss_inr = round(max(FINAL_DECISION_LOSS_FLOOR, gross_loss_per_lot * FINAL_DECISION_AVG_LOSS_GROSS_WEIGHT + hist_loss_inr * FINAL_DECISION_AVG_LOSS_HIST_WEIGHT + costs_stop), 2)
        expectancy_inr = round((win_prob * avg_win_inr) - ((1 - win_prob) * avg_loss_inr), 2)
        reward_risk = round(avg_win_inr / max(avg_loss_inr, FINAL_DECISION_LOSS_FLOOR), 3)

        # Capital-based lot calculation
        if adjusted_entry > 0 and lot_size > 0:
            raw_lots = int(risk_budget_inr / (gross_loss_per_lot or 1.0))
            # Fallback to simple capital check if risk budget is tiny
            if raw_lots < 1 and risk_budget_inr >= (adjusted_entry * lot_size * FINAL_DECISION_LOT_FALLBACK_RISK_MULT):
                raw_lots = 1
            desired_lots = max(1, min(raw_lots, MAX_POSITION_LOTS))
        else:
            desired_lots = 1
            
        quantity = desired_lots * lot_size

        if avg_win_inr <= costs_target:
            return self._decision(
                False,
                "expected reward does not clear transaction costs",
                adjusted_entry,
                adjusted_target,
                adjusted_stop,
                risk_pct,
                risk_budget_inr,
                0,
                0,
                win_prob,
                avg_win_inr,
                avg_loss_inr,
                expectancy_inr,
                reward_risk,
                stats,
            )
        if expectancy_inr <= FINAL_DECISION_MIN_EXPECTANCY_INR:
            return self._decision(
                False,
                f"expectancy {expectancy_inr:.2f} <= {FINAL_DECISION_MIN_EXPECTANCY_INR:.2f}",
                adjusted_entry,
                adjusted_target,
                adjusted_stop,
                risk_pct,
                risk_budget_inr,
                0,
                0,
                win_prob,
                avg_win_inr,
                avg_loss_inr,
                expectancy_inr,
                reward_risk,
                stats,
            )
        if desired_lots < 1:
            return self._decision(
                False,
                "risk budget too small for one lot",
                adjusted_entry,
                adjusted_target,
                adjusted_stop,
                risk_pct,
                risk_budget_inr,
                0,
                0,
                win_prob,
                avg_win_inr,
                avg_loss_inr,
                expectancy_inr,
                reward_risk,
                stats,
            )
        return self._decision(
            True,
            "positive expectancy",
            adjusted_entry,
            adjusted_target,
            adjusted_stop,
            risk_pct,
            risk_budget_inr,
            desired_lots,
            quantity,
            win_prob,
            avg_win_inr,
            avg_loss_inr,
            expectancy_inr,
            reward_risk,
            stats,
        )

    def _decision(
        self,
        tradeable: bool,
        reason: str,
        entry_premium: float,
        target_premium: float,
        stop_premium: float,
        risk_pct: float,
        risk_budget_inr: float,
        desired_lots: int,
        quantity: int,
        win_prob: float,
        avg_win_inr: float,
        avg_loss_inr: float,
        expectancy_inr: float,
        reward_risk: float,
        stats: HistoricalStats,
    ) -> FinalDecision:
        return FinalDecision(
            tradeable=tradeable,
            reason=reason,
            entry_premium=round(entry_premium, 1),
            target_premium=round(target_premium, 1),
            stop_premium=round(stop_premium, 1),
            risk_pct=round(risk_pct, 3),
            risk_budget_inr=round(risk_budget_inr, 2),
            desired_lots=desired_lots,
            quantity=quantity,
            win_prob=round(win_prob, 4),
            avg_win_inr=round(avg_win_inr, 2),
            avg_loss_inr=round(avg_loss_inr, 2),
            expectancy_inr=round(expectancy_inr, 2),
            reward_risk=round(reward_risk, 3),
            slippage_pct=round(self._slippage_pct, 3),
            historical_stats=stats,
        )

    def _risk_pct(self, *, signal_strength: float, atr_pct: float) -> float:
        base = FINAL_DECISION_MIN_RISK_PCT + (
            (FINAL_DECISION_MAX_RISK_PCT - FINAL_DECISION_MIN_RISK_PCT) * signal_strength
        )
        anchor_atr_pct = FINAL_DECISION_RISK_ANCHOR_ATR
        current_atr = max(float(atr_pct or 0.0), 0.001)
        vol_factor = max(FINAL_DECISION_VOL_FACTOR_MIN, min(FINAL_DECISION_VOL_FACTOR_MAX, anchor_atr_pct / current_atr))
        return max(
            FINAL_DECISION_MIN_RISK_PCT,
            min(FINAL_DECISION_MAX_RISK_PCT, base * vol_factor),
        )

    def _historical_stats(self, *, strategies: list[str], regime: str) -> HistoricalStats:
        rows = self._load_rows()
        strategy_key = "|".join(sorted(strategies))
        exact = [r for r in rows if r["strategies_fired"] == strategy_key]
        if len(exact) >= FINAL_DECISION_MIN_HISTORY:
            return self._aggregate(exact, f"strategy:{strategy_key}")
        same_regime = [r for r in rows if r["regime"] == regime]
        if len(same_regime) >= FINAL_DECISION_MIN_HISTORY:
            return self._aggregate(same_regime, f"regime:{regime}")
        if rows:
            return self._aggregate(rows, "global")
        return HistoricalStats(
            sample_size=0,
            win_rate=FINAL_DECISION_FALLBACK_WIN_RATE,
            avg_win_pct=FINAL_DECISION_FALLBACK_AVG_WIN_PCT,
            avg_loss_pct=FINAL_DECISION_FALLBACK_AVG_LOSS_PCT,
            source="fallback",
        )

    def _load_rows(self) -> list[dict]:
        if not self._journal_dir.exists():
            return []
        csv_files = sorted(self._journal_dir.glob("signals_*.csv"))
        cache_key = tuple((f.name, f.stat().st_mtime) for f in csv_files[-FINAL_DECISION_JOURNAL_LOOKBACK:])
        if cache_key == self._cache_key:
            return self._rows
        rows: list[dict] = []
        for path in csv_files[-FINAL_DECISION_JOURNAL_LOOKBACK:]:
            try:
                with path.open("r", newline="") as handle:
                    reader = csv.DictReader(handle)
                    for row in reader:
                        outcome = str(row.get("outcome_eod", "")).upper()
                        pnl_text = str(row.get("pnl_pct", "") or "").strip()
                        regime = str(row.get("regime", "") or "").upper()
                        strategies = str(row.get("strategies_fired", "") or "").strip()
                        if outcome not in {"WIN", "LOSS"} or not pnl_text:
                            continue
                        try:
                            pnl_pct = float(pnl_text)
                        except Exception:
                            continue
                        rows.append(
                            {
                                "outcome": outcome,
                                "pnl_pct": pnl_pct,
                                "regime": regime,
                                "strategies_fired": strategies,
                            }
                        )
            except Exception:
                continue
        self._cache_key = cache_key
        self._rows = rows
        return rows

    @staticmethod
    def _aggregate(rows: list[dict], source: str) -> HistoricalStats:
        wins = [float(r["pnl_pct"]) for r in rows if str(r["outcome"]).upper() == "WIN" and float(r["pnl_pct"]) > 0]
        losses = [abs(float(r["pnl_pct"])) for r in rows if str(r["outcome"]).upper() == "LOSS" and float(r["pnl_pct"]) < 0]
        total = len(rows)
        win_rate = (len(wins) / total) if total else 0.0
        avg_win_pct = sum(wins) / len(wins) if wins else FINAL_DECISION_FALLBACK_AVG_WIN_PCT
        avg_loss_pct = sum(losses) / len(losses) if losses else FINAL_DECISION_FALLBACK_AVG_LOSS_PCT
        return HistoricalStats(
            sample_size=total,
            win_rate=round(win_rate, 4),
            avg_win_pct=round(avg_win_pct, 4),
            avg_loss_pct=round(avg_loss_pct, 4),
            source=source,
        )
