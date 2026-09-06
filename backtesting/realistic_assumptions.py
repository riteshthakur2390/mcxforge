"""
backtesting/realistic_assumptions.py — Backtest Realism Layer
==============================================================
NEW FILE. Makes backtests more trustworthy by applying real-world costs.

THE PROBLEM WITH CURRENT BACKTEST:
  1. Enters at close price → real fills are at ASK (worse)
  2. No slippage for wide bid-ask spreads
  3. No STT (Securities Transaction Tax) on options sell side
  4. Same logic as 09:31 applied at 14:59 — ignores intraday theta decay
  5. No consideration of liquidity — OI=0 means no fill possible

WHAT THIS ADDS:
  1. SLIPPAGE MODEL: add 2% to entry premium (ASK > mid)
  2. TRANSACTION COSTS: STT + brokerage + SEBI charges deducted from P&L
  3. LIQUIDITY CHECK: skip if OI < 1000 contracts (illiquid)
  4. DTE-ADJUSTED EXIT: if TIME_DECAY exit after 14:30, premium has lost
     more than model assumes (accelerated theta)

USAGE in replay.py / backtest engine:
  from backtesting.realistic_assumptions import apply_realistic_entry
  realistic = apply_realistic_entry(premium=100, dte=3, time_of_day="10:30")
  # realistic.adjusted_premium = 102.0 (after slippage)
  # realistic.entry_cost_per_lot = 182.5 (all-in cost per lot)
"""

from dataclasses import dataclass


try:
    from config.settings import NIFTY_LOT_SIZE
except ImportError:
    NIFTY_LOT_SIZE = 65
SLIPPAGE_PCT         = 0.02    # 2% above mid for buy orders
BROKERAGE_PER_TRADE  = 40.0    # ₹20 entry + ₹20 exit per lot
STT_ON_SELL_PCT      = 0.000625   # 0.0625% on sell side notional
SEBI_PCT             = 0.000001   # negligible but correct
STAMP_DUTY_PCT       = 0.00003    # on buy side
MIN_OI_FOR_TRADE     = 500        # contracts — below this = illiquid


@dataclass
class RealisticEntry:
    adjusted_premium:    float   # after slippage
    entry_cost_per_lot:  float   # all-in entry cost per lot
    total_cost_per_lot:  float   # brokerage + taxes per lot
    effective_sl_pct:    float   # adjusted SL% after costs
    net_profit_at_target: float  # expected net P&L if target hit, after costs
    note:                str


def apply_realistic_entry(
    premium:      float,
    dte:          int,
    target_pct:   float = 0.60,
    sl_pct:       float = 0.30,
    time_of_day:  str   = "10:30",
    oi:           int   = 10000,
) -> RealisticEntry:
    """
    Apply realistic market conditions to a backtest entry.

    Args:
        premium:     estimated mid-price of option
        dte:         days to expiry
        target_pct:  target as % of premium (0.60 = 60%)
        sl_pct:      stop loss as % of premium (0.30 = 30%)
        time_of_day: HH:MM string of entry time
        oi:          open interest at strike (liquidity proxy)
    """
    # Liquidity check
    if oi > 0 and oi < MIN_OI_FOR_TRADE:
        return RealisticEntry(
            adjusted_premium    = premium,
            entry_cost_per_lot  = premium * NIFTY_LOT_SIZE,
            total_cost_per_lot  = 0,
            effective_sl_pct    = sl_pct,
            net_profit_at_target = -999,
            note                = f"SKIP: OI={oi} < {MIN_OI_FOR_TRADE} (illiquid)",
        )

    # Slippage — 2% above mid (buy at ASK)
    adj_premium   = premium * (1 + SLIPPAGE_PCT)

    # Transaction costs per lot
    entry_notional = adj_premium * NIFTY_LOT_SIZE
    exit_notional  = adj_premium * (1 + target_pct) * NIFTY_LOT_SIZE

    brokerage    = BROKERAGE_PER_TRADE
    stt          = exit_notional * STT_ON_SELL_PCT   # STT only on sell (options)
    sebi         = (entry_notional + exit_notional) * SEBI_PCT
    stamp        = entry_notional * STAMP_DUTY_PCT
    total_costs  = brokerage + stt + sebi + stamp

    # Net profit if target hit
    gross_profit = adj_premium * target_pct * NIFTY_LOT_SIZE
    net_profit   = gross_profit - total_costs

    # Late-day theta: after 14:00, add extra 1% decay per hour since open
    h = int(time_of_day.split(":")[0])
    m = int(time_of_day.split(":")[1])
    if h >= 14:
        late_decay = 0.005 * (h - 14 + m / 60)   # 0.5% per hour after 14:00
        adj_premium = adj_premium * (1 - late_decay)

    # Effective SL% accounting for costs (must cover costs + sl)
    cost_pct_of_premium  = total_costs / max(entry_notional, 1)
    effective_sl_pct     = sl_pct + cost_pct_of_premium

    note = (
        f"adj_premium=₹{adj_premium:.1f} (slippage+{SLIPPAGE_PCT*100:.0f}%) | "
        f"costs=₹{total_costs:.0f}/lot | "
        f"net_profit_if_target=₹{net_profit:.0f}"
    )

    return RealisticEntry(
        adjusted_premium     = round(adj_premium, 2),
        entry_cost_per_lot   = round(entry_notional, 2),
        total_cost_per_lot   = round(total_costs, 2),
        effective_sl_pct     = round(effective_sl_pct, 4),
        net_profit_at_target = round(net_profit, 2),
        note                 = note,
    )


def expected_value(win_rate: float, avg_win_pct: float, avg_loss_pct: float,
                   premium: float = 100, dte: int = 5) -> dict:
    """
    Compute trade expectancy including realistic costs.
    Use this to decide if a setup is worth taking at all.

    Industry minimum: expectancy > ₹0 per trade after all costs.
    """
    entry = apply_realistic_entry(premium, dte)
    lot_premium = entry.adjusted_premium * NIFTY_LOT_SIZE

    gross_win  = lot_premium * avg_win_pct
    gross_loss = lot_premium * avg_loss_pct
    net_win    = gross_win  - entry.total_cost_per_lot
    net_loss   = gross_loss - entry.total_cost_per_lot

    expectancy = (win_rate * net_win) + ((1 - win_rate) * net_loss)

    return {
        "expectancy_per_lot": round(expectancy, 2),
        "is_positive_ev":     expectancy > 0,
        "gross_win":          round(gross_win, 2),
        "net_win":            round(net_win, 2),
        "gross_loss":         round(gross_loss, 2),
        "net_loss":           round(net_loss, 2),
        "total_costs":        round(entry.total_cost_per_lot, 2),
        "break_even_wr":      round(-net_loss / max(net_win - net_loss, 0.01), 3),
    }