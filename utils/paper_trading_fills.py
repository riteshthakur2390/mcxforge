"""
utils/paper_trading_fills.py — Paper Trading with Real Broker Fills
====================================================================
Current OBSERVE mode uses Black-Scholes estimated premiums.
This module uses ACTUAL broker LTP for fills so simulation P&L
matches what live trading would have been exactly.

DIFFERENCE:
  Current:  entry_premium = estimate_atm_premium(spot, dte)
            → may differ from actual by 5-15%
  
  This:     entry_premium = broker.get_option_ltp(symbol)
            → exact match to live fill price
            → slippage simulation included

WHY IT MATTERS:
  If paper trade shows +6.45% but uses estimated premiums,
  live trade may show +4.2% due to:
  - Bid-ask spread (options have wide spreads)
  - IV differences between BS estimate and market IV
  - Time-of-day premium variations
  
  Real fills in paper mode gives true signal validation.

USAGE:
  fills = PaperTradingFills(broker)
  entry = await fills.simulate_entry(symbol, expected_premium=150)
  # Returns actual LTP from broker, not estimate
  
  exit_ = await fills.simulate_exit(symbol, entry_price=152.5)
  # Returns actual LTP + slippage simulation
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import pytz

IST = pytz.timezone("Asia/Kolkata")

try:
    from config.settings import JOURNAL_DIR
except ImportError:
    JOURNAL_DIR = "journal"

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# Realistic slippage for NIFTY options (bid-ask half-spread)
SLIPPAGE_MODEL = {
    "ATM":  0.3,   # ₹0.30 slippage on ATM (tight spread)
    "OTM1": 0.5,   # ₹0.50 for 1 strike OTM
    "OTM2": 1.0,   # ₹1.00 for 2+ strikes OTM
    "ITM":  0.4,   # ₹0.40 for ITM
}
DEFAULT_SLIPPAGE = 0.5   # ₹0.50 default


@dataclass
class PaperFill:
    symbol:           str
    expected_premium: float     # BS estimate or signal price
    actual_ltp:       float     # real broker LTP
    fill_price:       float     # actual_ltp + slippage
    slippage:         float     # ₹ slippage added
    slippage_pct:     float     # % of actual_ltp
    fill_type:        str       # "real_ltp" | "estimated" | "fallback"
    timestamp:        str


@dataclass
class PaperTradeResult:
    symbol:        str
    entry:         PaperFill
    exit_:         PaperFill
    pnl_pct:       float
    pnl_inr:       float        # per lot
    lot_size:      int
    holding_min:   int
    realistic_pnl: float        # pnl after realistic costs


class PaperTradingFills:
    """
    Simulates paper trades using real broker LTPs.
    Tracks all paper trades for comparison with live.
    """

    PAPER_LOG = Path(JOURNAL_DIR) / "paper_trades.json"

    def __init__(self, broker=None) -> None:
        self._broker    = broker
        self._paper_log: list[dict] = self._load_log()

    async def simulate_entry(
        self,
        symbol:           str,
        expected_premium: float,
        lot_size:         int = 75,
    ) -> PaperFill:
        """
        Get actual entry fill using real broker LTP.
        Falls back to expected_premium if broker unavailable.
        """
        actual_ltp = 0.0

        if self._broker is not None:
            try:
                actual_ltp = float(self._broker.get_option_ltp(symbol))
            except Exception as e:
                logger.debug(f"[PaperFills] get_ltp failed: {e}")

        if actual_ltp <= 0:
            actual_ltp  = expected_premium
            fill_type   = "estimated"
        else:
            fill_type   = "real_ltp"

        # Add realistic bid-ask slippage (buying = pay ask = LTP + slip)
        slippage   = DEFAULT_SLIPPAGE
        fill_price = round(actual_ltp + slippage, 2)
        slip_pct   = slippage / actual_ltp * 100 if actual_ltp > 0 else 0

        fill = PaperFill(
            symbol           = symbol,
            expected_premium = expected_premium,
            actual_ltp       = actual_ltp,
            fill_price       = fill_price,
            slippage         = slippage,
            slippage_pct     = round(slip_pct, 3),
            fill_type        = fill_type,
            timestamp        = datetime.now(IST).isoformat(),
        )

        logger.info(
            f"[PaperFills] ENTRY {symbol} | "
            f"Expected=₹{expected_premium:.1f} | "
            f"LTP=₹{actual_ltp:.1f} | "
            f"Fill=₹{fill_price:.1f} ({fill_type})"
        )
        return fill

    async def simulate_exit(
        self,
        symbol:      str,
        entry_price: float,
        lot_size:    int   = 75,
        entry_time:  str   = "",
    ) -> PaperFill:
        """Get actual exit fill using real broker LTP."""
        actual_ltp = 0.0

        if self._broker is not None:
            try:
                actual_ltp = float(self._broker.get_option_ltp(symbol))
            except Exception:
                pass

        if actual_ltp <= 0:
            # Fallback — use entry as proxy (0 P&L, conservative)
            actual_ltp = entry_price
            fill_type  = "fallback"
        else:
            fill_type  = "real_ltp"

        # Exit: sell at bid = LTP - slippage
        slippage   = DEFAULT_SLIPPAGE
        fill_price = max(0.05, round(actual_ltp - slippage, 2))
        slip_pct   = slippage / actual_ltp * 100 if actual_ltp > 0 else 0

        return PaperFill(
            symbol           = symbol,
            expected_premium = actual_ltp,
            actual_ltp       = actual_ltp,
            fill_price       = fill_price,
            slippage         = slippage,
            slippage_pct     = round(slip_pct, 3),
            fill_type        = fill_type,
            timestamp        = datetime.now(IST).isoformat(),
        )

    def record_paper_trade(
        self,
        entry:    PaperFill,
        exit_:    PaperFill,
        lot_size: int,
        strategies: list[str] = None,
    ) -> PaperTradeResult:
        """Record completed paper trade and compute realistic P&L."""
        pnl_pct = (exit_.fill_price - entry.fill_price) / entry.fill_price * 100
        pnl_inr = (exit_.fill_price - entry.fill_price) * lot_size

        # Realistic costs: STT + brokerage
        stt       = entry.fill_price * lot_size * 0.00025
        brokerage = 40.0
        net_pnl   = pnl_inr - stt - brokerage

        result = PaperTradeResult(
            symbol       = entry.symbol,
            entry        = entry,
            exit_        = exit_,
            pnl_pct      = round(pnl_pct, 3),
            pnl_inr      = round(pnl_inr, 2),
            lot_size     = lot_size,
            holding_min  = 0,
            realistic_pnl= round(net_pnl, 2),
        )

        # Save to log
        self._paper_log.append({
            "symbol":        entry.symbol,
            "entry_fill":    entry.fill_price,
            "exit_fill":     exit_.fill_price,
            "entry_type":    entry.fill_type,
            "exit_type":     exit_.fill_type,
            "pnl_pct":       result.pnl_pct,
            "pnl_inr":       result.pnl_inr,
            "realistic_pnl": result.realistic_pnl,
            "strategies":    strategies or [],
            "timestamp":     datetime.now(IST).isoformat(),
        })
        self._save_log()

        logger.info(
            f"[PaperFills] TRADE COMPLETE | {entry.symbol} | "
            f"Entry=₹{entry.fill_price:.1f} Exit=₹{exit_.fill_price:.1f} | "
            f"PnL={pnl_pct:+.2f}% (₹{pnl_inr:+.0f}) | "
            f"Net after costs=₹{net_pnl:+.0f}"
        )
        return result

    def get_paper_summary(self) -> dict:
        if not self._paper_log:
            return {"trades": 0}
        pnls = [t["pnl_pct"] for t in self._paper_log]
        wins = [p for p in pnls if p > 0]
        return {
            "trades":    len(pnls),
            "win_rate":  round(len(wins) / len(pnls) * 100, 1),
            "avg_pnl":   round(sum(pnls) / len(pnls), 2),
            "total_pnl": round(sum(pnls), 2),
            "total_net": round(sum(t["realistic_pnl"] for t in self._paper_log), 2),
        }

    def _load_log(self) -> list:
        try:
            if self.PAPER_LOG.exists():
                with open(self.PAPER_LOG) as f:
                    return json.load(f)
        except Exception:
            pass
        return []

    def _save_log(self) -> None:
        try:
            Path(JOURNAL_DIR).mkdir(exist_ok=True)
            with open(self.PAPER_LOG, "w") as f:
                json.dump(self._paper_log, f, indent=2)
        except Exception:
            pass
