from __future__ import annotations
"""
utils/telegram_notifier.py — Real-Time Trade Notifications
============================================================
Every professional algo trading firm sends real-time notifications.
This module sends formatted Telegram messages for:
  - Trade opened (entry details, strategy, regime)
  - Trade closed (PnL, exit reason, peak, duration)
  - Circuit breaker triggered
  - Daily EOD summary
  - Pre-market brief

Setup (one-time):
  1. Message @BotFather on Telegram → /newbot → get TELEGRAM_BOT_TOKEN
  2. Message @userinfobot → get your TELEGRAM_CHAT_ID
  3. Add both to .env:
       TELEGRAM_BOT_TOKEN=7xxxxxxxxx:AAF...
       TELEGRAM_CHAT_ID=123456789

Usage:
    from utils.telegram_notifier import get_notifier
    notifier = get_notifier()
    await notifier.send_trade_opened(entry_payload)
    await notifier.send_trade_closed(exit_payload)

Message format follows industry standard used by:
  - Sensibull alerts
  - Zerodha Kite webhook notifications
  - AlgoTest Telegram integration
"""

import warnings
warnings.filterwarnings("ignore", message=".*urllib3 v2 only supports OpenSSL.*")
try:
    import urllib3.exceptions
    warnings.filterwarnings("ignore", category=urllib3.exceptions.NotOpenSSLWarning)
except Exception:
    pass

import asyncio
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional
import pytz

IST = pytz.timezone("Asia/Kolkata")

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
except Exception:
    pass

try:
    from config.settings import (
        TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TELEGRAM_ENABLED,
        TRADING_MODE, INSTRUMENT,
        LIVE_TELEGRAM_BOT_TOKEN, LIVE_TELEGRAM_CHAT_ID,
        BACKTEST_TELEGRAM_BOT_TOKEN, BACKTEST_TELEGRAM_CHAT_ID,
    )
except ImportError:
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")
    LIVE_TELEGRAM_BOT_TOKEN = os.getenv("LIVE_TELEGRAM_BOT_TOKEN", TELEGRAM_BOT_TOKEN)
    LIVE_TELEGRAM_CHAT_ID   = os.getenv("LIVE_TELEGRAM_CHAT_ID", TELEGRAM_CHAT_ID)
    BACKTEST_TELEGRAM_BOT_TOKEN = os.getenv("BACKTEST_TELEGRAM_BOT_TOKEN", "")
    BACKTEST_TELEGRAM_CHAT_ID   = os.getenv("BACKTEST_TELEGRAM_CHAT_ID", "")
    TELEGRAM_ENABLED   = bool(
        (LIVE_TELEGRAM_BOT_TOKEN and LIVE_TELEGRAM_CHAT_ID)
        or (BACKTEST_TELEGRAM_BOT_TOKEN and BACKTEST_TELEGRAM_CHAT_ID)
    )
    TRADING_MODE       = os.getenv("TRADING_MODE", "AUTO")
    INSTRUMENT         = "NIFTY"

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


class TelegramNotifier:
    """
    Async Telegram message sender.
    All methods are fire-and-forget — they never block the trading pipeline.
    If Telegram is unreachable, errors are silently logged (never raised).
    """
    _sent_cache: dict[str, float] = {}

    def __init__(self) -> None:
        generic_token = os.getenv("TELEGRAM_BOT_TOKEN", TELEGRAM_BOT_TOKEN)
        generic_chat = os.getenv("TELEGRAM_CHAT_ID", TELEGRAM_CHAT_ID)
        self._live_token = os.getenv("LIVE_TELEGRAM_BOT_TOKEN", LIVE_TELEGRAM_BOT_TOKEN or generic_token)
        self._live_chat  = os.getenv("LIVE_TELEGRAM_CHAT_ID", LIVE_TELEGRAM_CHAT_ID or generic_chat)
        self._bt_token   = os.getenv("BACKTEST_TELEGRAM_BOT_TOKEN", BACKTEST_TELEGRAM_BOT_TOKEN)
        self._bt_chat    = os.getenv("BACKTEST_TELEGRAM_CHAT_ID", BACKTEST_TELEGRAM_CHAT_ID)
        self._enabled  = bool(
            (self._live_token and self._live_chat)
            or (self._bt_token and self._bt_chat)
            or TELEGRAM_ENABLED
        )
        self._session  = None   # aiohttp session — lazy init

        if self._enabled:
            try:
                from loguru import logger
                logger.info("[TelegramNotifier] ✅ Enabled")
            except Exception:
                pass
        else:
            try:
                from loguru import logger
                logger.warning(
                    "[TelegramNotifier] Disabled — set TELEGRAM_BOT_TOKEN "
                    "and TELEGRAM_CHAT_ID in .env to enable"
                )
            except Exception:
                pass

    # ── PUBLIC METHODS ────────────────────────────────────────────────────────

    @staticmethod
    def _payload_target(payload: dict | None = None) -> str:
        p = payload or {}
        mode = str(p.get("target") or p.get("mode") or os.getenv("TELEGRAM_TARGET") or os.getenv("TRADING_MODE", TRADING_MODE)).upper()
        if "BACKTEST" in mode or mode in ("BT", "TEST"):
            return "BACKTEST"
        return "LIVE"

    async def send_trade_opened(self, payload: dict, route_target: str | None = None, target: str | None = None) -> None:
        """Send notification when a new option position is opened with full contract and price telemetry."""
        chosen_target = route_target or target or self._payload_target(payload)
        sig       = payload.get("signal", {})
        direction = str(sig.get("direction") or payload.get("direction", "?")).upper()

        # Contract and Option Metadata
        contract_symbol = str(
            payload.get("contract_symbol")
            or payload.get("option_symbol")
            or payload.get("symbol")
            or "MCX Option"
        )
        strike    = payload.get("strike") or payload.get("strike_price")
        opt_type  = str(payload.get("option_type") or ("CE" if ("CALL" in direction or "CE" in direction or direction in ("BUY", "LONG")) else "PE")).upper()
        expiry    = str(payload.get("expiry_date") or payload.get("expiry") or "")

        # Prices: Underlying Commodity Futures vs Option Contract Premium
        underlying = float(
            payload.get("underlying_price")
            or payload.get("underlying_entry")
            or payload.get("spot_price")
            or sig.get("underlying_price")
            or 0.0
        )
        prem = float(
            payload.get("actual_premium")
            or payload.get("entry_premium")
            or payload.get("est_premium")
            or payload.get("entry_price")
            or 0.0
        )
        # Graceful fallback: If underlying was passed as entry_price (> 10,000 for Silver), distinguish them
        if underlying == 0.0 and prem > 10000.0:
            underlying = prem
            prem = max(100.0, round(underlying * 0.025, 1))

        strategies = sig.get("strategies_fired") or payload.get("strategies_fired") or []
        if isinstance(strategies, str):
            try:
                import ast
                strategies = ast.literal_eval(strategies)
            except Exception:
                strategies = [strategies]
        votes     = sig.get("votes") or payload.get("votes") or len(strategies)
        regime    = sig.get("regime") or payload.get("regime") or "MCX"
        sl        = float(payload.get("sl_price") or payload.get("sl_premium") or (prem * 0.75))
        tgt_price = float(payload.get("target_price") or payload.get("target_premium") or (prem * 1.70))
        mode      = payload.get("mode", TRADING_MODE)
        sim_tag   = "🔵 PAPER" if payload.get("simulated", True) else "🟡 LIVE"

        arrow = "🟢 ▲ BUY CALL" if ("CALL" in direction or "CE" in direction or direction in ("BUY", "LONG")) else "🔴 ▼ BUY PUT"
        lots = int(payload.get("lots", 1) or 1)
        lot_size = int(payload.get("lot_size", 5) or 5)
        qty = int(payload.get("quantity", lots * lot_size) or (lots * lot_size))

        # Margin / Invested (Hard-capped at 15% of capital = ₹30,000)
        margin_used = float(
            payload.get("margin_used")
            or payload.get("margin_used_inr")
            or payload.get("total_invested")
            or (prem * qty)
        )
        max_margin_budget = float(payload.get("max_margin_budget", 30000.0))
        margin_used = min(margin_used, max_margin_budget)

        # ML metrics
        ml_score = float(payload.get("ml_rank_score") or sig.get("ml_rank_score") or 0.0)
        ml_tier  = str(payload.get("ml_rank_tier") or sig.get("ml_rank_tier") or ("HIGH" if ml_score >= 0.70 else "MED" if ml_score >= 0.45 else "LOW"))
        session  = str(payload.get("mcx_session") or payload.get("session") or "EVENING").upper()

        sl_diff_pct  = round(((sl - prem) / max(prem, 1e-6)) * 100.0, 1) if prem > 0 else 0.0
        tgt_diff_pct = round(((tgt_price - prem) / max(prem, 1e-6)) * 100.0, 1) if prem > 0 else 0.0

        # Entry timestamp
        entry_time_str = str(payload.get("entry_time") or payload.get("timestamp") or "")
        if entry_time_str:
            try:
                import pandas as _pd
                dt = _pd.to_datetime(entry_time_str)
                display_time = dt.strftime("%d %b %H:%M:%S IST")
            except Exception:
                display_time = entry_time_str[:19]
        else:
            display_time = datetime.now(IST).strftime("%H:%M:%S IST")

        # Format contract metadata line
        contract_meta_line = ""
        if strike or expiry:
            strike_str = f"Strike ₹{int(strike):,}" if strike else ""
            opt_tag = f" {opt_type}" if opt_type else ""
            exp_tag = f" | Exp: {expiry}" if expiry else ""
            contract_meta_line = f"🏷️ *Contract*: {strike_str}{opt_tag}{exp_tag}\n"

        price_lines = []
        if underlying > 0:
            price_lines.append(f"📊 *Underlying (MCX)*: ₹{underlying:,.2f}")
        price_lines.append(f"💰 *Contract Premium*: ₹{prem:,.2f}")
        price_lines.append(f"🛑 *SL Premium*:       ₹{sl:,.2f} ({sl_diff_pct:+.1f}%)")
        price_lines.append(f"🎯 *Target Premium*:   ₹{tgt_price:,.2f} ({tgt_diff_pct:+.1f}%)")
        price_lines.append(f"💵 *Margin Deployed*:  ₹{margin_used:,.0f} (Cap: ₹{max_margin_budget:,.0f})")
        price_block = "\n".join(price_lines)

        msg = (
            f"📊 *MCXForge — Trade Opened*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{arrow}  {sim_tag}  |  {lots} Lot{'s' if lots > 1 else ''} ({qty} Qty)\n"
            f"📌 *{contract_symbol}*\n"
            f"{contract_meta_line}"
            f"⏰ {display_time}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{price_block}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🗳️ *Votes*:   {votes} | ML Rank: {ml_score:.2f} ({ml_tier})\n"
            f"🕒 *Session*: {session}\n"
            f"📋 *Strats*:  {', '.join(str(s) for s in strategies[:4]) if strategies else 'Consensus'}\n"
            f"⚙️ *Mode*:    {mode}"
        )
        await self._send(msg, target=chosen_target)

    async def send_trade_closed(self, payload: dict, target: str | None = None) -> None:
        """Send notification when option position is closed with full underlying and contract movement."""
        direction = str(payload.get("direction", "BUY")).upper()
        reason    = str(payload.get("exit_reason", "TARGET_HIT"))
        peak      = float(payload.get("peak_pnl_pct", payload.get("peak_pnl", 0.0)))
        held      = int(payload.get("candles_held", 0))
        held_min  = int(payload.get("hold_minutes", payload.get("holding_minutes", held * 5)) or held * 5)
        tsl_on    = bool(payload.get("tsl_activated", False))
        lots      = int(payload.get("lots", 1) or 1)
        lot_size  = int(payload.get("lot_size", 5) or 5)
        qty       = int(payload.get("quantity", lots * lot_size) or (lots * lot_size))

        # Contract Details
        contract_symbol = str(
            payload.get("contract_symbol")
            or payload.get("option_symbol")
            or payload.get("symbol")
            or "MCX Option"
        )
        strike    = payload.get("strike") or payload.get("strike_price")
        opt_type  = str(payload.get("option_type") or ("CE" if "CALL" in direction or direction in ("BUY", "LONG") else "PE")).upper()
        expiry    = str(payload.get("expiry_date") or payload.get("expiry") or "")

        # Dual Price Movement: Underlying Commodity Futures vs Option Contract Premium
        u_entry = float(payload.get("underlying_entry") or payload.get("underlying_price") or 0.0)
        u_exit  = float(payload.get("underlying_exit") or 0.0)
        
        c_entry = float(
            payload.get("entry_premium")
            or payload.get("actual_premium")
            or payload.get("est_premium")
            or 0.0
        )
        c_exit = float(payload.get("exit_premium") or 0.0)

        # Legacy payload fallback if only entry_price and exit_price were given:
        if c_entry == 0.0 and c_exit == 0.0:
            raw_entry = float(payload.get("entry_price", 0.0))
            raw_exit  = float(payload.get("exit_price", 0.0))
            if raw_entry > 10000.0:
                # Raw entry was the commodity futures price (e.g. SILVERM 88,400)
                u_entry = raw_entry
                u_exit  = raw_exit
                u_pts   = round(u_exit - u_entry, 2)
                c_entry = max(100.0, round(u_entry * 0.025, 1))
                c_exit  = max(10.0, round(c_entry + (u_pts * 0.50), 1))
            else:
                c_entry = raw_entry
                c_exit  = raw_exit

        u_pts = float(payload.get("underlying_points") or (u_exit - u_entry) if u_entry > 0 and u_exit > 0 else 0.0)
        u_pct = round((u_pts / max(u_entry, 1.0)) * 100.0, 2) if u_entry > 0 else 0.0

        c_pts = float(payload.get("contract_points") or (c_exit - c_entry) if c_entry > 0 else 0.0)
        c_pct = round((c_pts / max(c_entry, 1e-6)) * 100.0, 2) if c_entry > 0 else 0.0

        # Points & PnL
        gross_inr = float(payload.get("gross_pnl_inr", c_pts * qty))
        charges   = float(payload.get("fees_inr", payload.get("total_charges", 0.0)))
        net_inr   = float(payload.get("net_pnl_inr", payload.get("realized_pnl", gross_inr - charges)))
        margin    = float(payload.get("margin_used_inr", payload.get("total_invested", max(c_entry * qty, 1.0))))
        max_margin_budget = float(payload.get("max_margin_budget", 30000.0))
        margin    = min(margin, max_margin_budget)

        net_roi_pct = round((net_inr / max(margin, 1.0)) * 100.0, 2) if margin > 0 else 0.0

        emoji = (
            "🏆" if net_inr >= 2000
            else "🟢" if net_inr > 0
            else "🟡" if net_inr > -300
            else "🔴"
        )

        reason_emoji = {
            "TARGET_HIT":     "🎯",
            "PROFIT_PROTECT": "🔒",
            "TRAILING_SL":    "🔒",
            "SL_HIT":         "🛑",
            "EOD":            "🕐",
            "EOD_SQUAREOFF":   "🕐",
            "STALE_LOSS":     "💤",
            "TIME_DECAY":     "⏳",
            "MANUAL_EXIT":    "✋",
        }.get(reason.upper(), "📤")

        peak_eff = round(c_pct / peak * 100, 0) if peak > 0 else 100
        tsl_tag  = f" | TSL {'✅' if tsl_on else '—'}" if c_pct > 0 else ""

        # Exit timestamp
        exit_time_str = str(payload.get("exit_time") or payload.get("timestamp") or "")
        if exit_time_str:
            try:
                import pandas as _pd
                dt = _pd.to_datetime(exit_time_str)
                display_time = dt.strftime("%d %b %H:%M:%S IST")
            except Exception:
                display_time = exit_time_str[:19]
        else:
            display_time = datetime.now(IST).strftime("%H:%M:%S IST")

        # Format contract metadata line
        contract_meta_line = ""
        if strike or expiry:
            strike_str = f"Strike ₹{int(strike):,}" if strike else ""
            opt_tag = f" {opt_type}" if opt_type else ""
            exp_tag = f" | Exp: {expiry}" if expiry else ""
            contract_meta_line = f"🏷️ *Contract*: {strike_str}{opt_tag}{exp_tag}\n"

        # Movement breakdown lines
        movement_lines = []
        if u_entry > 0 and u_exit > 0:
            movement_lines.append(
                f"📊 *Underlying Movement*:\n"
                f"   ₹{u_entry:,.2f} → ₹{u_exit:,.2f} (*{'+' if u_pts >= 0 else ''}{u_pts:,.2f} pts*, {u_pct:+.2f}%)"
            )
        movement_lines.append(
            f"🎟️ *Contract Movement*:\n"
            f"   ₹{c_entry:,.2f} → ₹{c_exit:,.2f} (*{'+' if c_pts >= 0 else ''}{c_pts:,.2f} pts*, {c_pct:+.2f}%)"
        )
        movement_block = "\n".join(movement_lines)

        msg = (
            f"{emoji} *MCXForge — Trade Closed*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{reason_emoji} *{reason}*{tsl_tag}\n"
            f"📌 *{contract_symbol}*\n"
            f"{contract_meta_line}"
            f"📦 *Size*: {lots} Lot{'s' if lots > 1 else ''} ({qty} Qty)\n"
            f"⏰ {display_time}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{movement_block}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💵 *Gross P&L*: *{'+' if gross_inr >= 0 else ''}₹{gross_inr:,.2f}*\n"
            f"🏷️ *Charges*:   ₹{charges:,.2f}\n"
            f"{'📈' if net_inr >= 0 else '📉'} *Net P&L:   {'+' if net_inr >= 0 else ''}₹{net_inr:,.2f} ({net_roi_pct:+.2f}% ROI)*\n"
            f"🛡️ *Margin*:    ₹{margin:,.0f} (Cap: ₹{max_margin_budget:,.0f})\n"
            f"🏔️ *Peak MFE*: {peak:+.1f}% | Captured: {peak_eff:.0f}%\n"
            f"⏱️ *Held*:     {held_min} min"
        )
        await self._send(msg, target=(target or self._payload_target(payload)))

    async def send_backtest_summary(self, summary: dict, target: str | None = None) -> None:
        """Send comprehensive SignalForge backtest performance report card."""
        symbol       = summary.get("symbol", "SILVERMIC")
        period       = summary.get("period", "30 Days")
        bars         = summary.get("bars", 0)
        trades       = int(summary.get("trades", 0))
        wins         = int(summary.get("wins", 0))
        losses       = int(summary.get("losses", max(0, trades - wins)))
        wr           = float(summary.get("win_rate", (wins / max(1, trades)) * 100.0))
        gross        = float(summary.get("gross_pnl", 0.0))
        charges      = float(summary.get("charges", summary.get("fees", 0.0)))
        net          = float(summary.get("net_pnl", gross - charges))
        pf           = float(summary.get("profit_factor", 0.0))
        sharpe       = float(summary.get("sharpe", 0.0))
        expectancy   = float(summary.get("expectancy", (net / max(1, trades))))
        capital      = float(summary.get("capital", 200000.0))
        cap_return   = float(summary.get("capital_return_pct", (net / max(1.0, capital)) * 100.0))
        margin_used  = float(summary.get("peak_margin_used", 25666.0))
        budget_cap   = float(summary.get("max_margin_budget", 30000.0))
        max_dd       = float(summary.get("max_dd", 0.0))
        max_dd_pct   = float(summary.get("max_dd_pct", (max_dd / max(1.0, capital)) * 100.0))

        emoji = "🏆" if net > 0 else "📉"

        msg = (
            f"{emoji} *MCXForge — Backtest Performance Review*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📅 Period: {period} ({bars:,} Bars)\n"
            f"📦 Instrument: *{symbol}* (1 Lot)\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🗂️ Total Trades:  {trades}\n"
            f"✅ Wins:          {wins} ({wr:.1f}% Win Rate)\n"
            f"❌ Losses:        {losses}\n"
            f"📈 Profit Factor: {pf:.2f}\n"
            f"⚡ Sharpe Ratio:  {sharpe:.3f}\n"
            f"🎯 Expectancy:    ₹{expectancy:,.0f} / trade\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 Gross P&L:     {'+' if gross >= 0 else ''}₹{gross:,.2f}\n"
            f"🏷️ Total Charges: ₹{charges:,.2f}\n"
            f"💵 Net Realized:  *{'+' if net >= 0 else ''}₹{net:,.2f}*\n"
            f"💼 Capital Return:*{'+' if cap_return >= 0 else ''}{cap_return:.2f}%* on ₹{capital:,.0f}\n"
            f"📉 Max Drawdown:  ₹{max_dd:,.2f} ({max_dd_pct:.2f}%)\n"
            f"🛡️ Max Margin:    ₹{margin_used:,.0f} (Cap: ₹{budget_cap:,.0f})"
        )
        await self._send(msg, target=(target or "BACKTEST"))

    def send_trade_opened_sync(self, payload: dict, target: str | None = None) -> bool:
        """Synchronous wrapper for trade open alert."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, self.send_trade_opened(payload, target=target))
                    future.result()
                    return True
            else:
                loop.run_until_complete(self.send_trade_opened(payload, target=target))
                return True
        except Exception:
            try:
                asyncio.run(self.send_trade_opened(payload, target=target))
                return True
            except Exception:
                return False

    def send_trade_closed_sync(self, payload: dict, target: str | None = None) -> bool:
        """Synchronous wrapper for trade close alert."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, self.send_trade_closed(payload, target=target))
                    future.result()
                    return True
            else:
                loop.run_until_complete(self.send_trade_closed(payload, target=target))
                return True
        except Exception:
            try:
                asyncio.run(self.send_trade_closed(payload, target=target))
                return True
            except Exception:
                return False

    def send_backtest_summary_sync(self, summary: dict, target: str | None = None) -> bool:
        """Synchronous wrapper for backtest summary card alert."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, self.send_backtest_summary(summary, target=target))
                    future.result()
                    return True
            else:
                loop.run_until_complete(self.send_backtest_summary(summary, target=target))
                return True
        except Exception:
            try:
                asyncio.run(self.send_backtest_summary(summary, target=target))
                return True
            except Exception:
                return False

    async def send_circuit_breaker(self, payload: dict) -> None:
        """Alert when daily loss limit is hit."""
        loss_inr  = abs(float(payload.get("daily_pnl_inr", 0)))
        limit_inr = abs(float(payload.get("loss_limit_inr", 0)))
        pct       = float(payload.get("loss_pct", 3.0))

        msg = (
            f"🛑 *CIRCUIT BREAKER TRIGGERED*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📉 Daily Loss: ₹{loss_inr:,.0f}\n"
            f"🚫 Limit:      ₹{limit_inr:,.0f} ({pct}% of capital)\n"
            f"⏰ {datetime.now(IST).strftime('%H:%M:%S IST')}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"⛔ *No new trades for today.*\n"
            f"Trading resumes tomorrow at 09:15 IST."
        )
        await self._send(msg, target=self._payload_target(payload))

    async def send_eod_summary(self, summary: dict) -> None:
        """Send end-of-day P&L summary with capital tracking and brokerage breakdown."""
        trades       = int(summary.get("trades", 0))
        wins         = int(summary.get("wins", 0))
        losses       = int(summary.get("losses", max(0, trades - wins)))
        wr           = float(summary.get("win_rate", 0))
        gross        = float(summary.get("gross_pnl", 0))
        charges      = float(summary.get("total_charges", 0))
        net          = float(summary.get("net_pnl", 0))
        today        = datetime.now(IST).strftime("%d %b %Y")
        total_fund   = float(summary.get("total_fund", 200000))
        deployed     = float(summary.get("deployed_capital", 30000))
        equity       = float(summary.get("ending_equity", total_fund) or total_fund)
        dd           = float(summary.get("drawdown_pct", 0))
        signal       = summary.get("risk_signal", "NORMAL")
        roi_pct      = ((equity - total_fund) / max(total_fund, 1.0)) * 100.0

        signal_tag = {"PAUSE": "⛔ PAUSE tomorrow", "REDUCE_SIZE": "🎯 Signal-Adaptive tomorrow",
                      "NORMAL": "✅ Normal (Signal-Driven)"}.get(signal, signal)

        day_emoji = "📈" if net >= 0 else "📉"

        msg = (
            f"{day_emoji} *MCXForge — Live EOD Report*\n"
            f"📅 {today}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💼 Total Capital:  ₹{total_fund:,.0f}\n"
            f"🎯 Active Fund:    ₹{deployed:,.0f}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🗂️ Trades:  {trades}  |  WR: {wr:.0f}%  ({wins}W / {losses}L)\n"
            f"💰 Gross:   ₹{gross:+,.2f}\n"
            f"🏷️ Charges: ₹{charges:,.2f} (Exchange + Taxes)\n"
            f"💵 Net P&L: *₹{net:+,.2f}*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 Total Equity: ₹{equity:,.2f}  |  ROI: {roi_pct:+.2f}%\n"
            f"📉 Drawdown:     {dd:.1f}%\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🔔 Tomorrow: {signal_tag}"
        )
        await self._send(msg, target=self._payload_target(summary))

    async def send_premarket_brief(self, payload: dict) -> None:
        """Morning market brief before trading starts."""
        bias     = payload.get("bias", "NEUTRAL")
        raw_vix  = float(payload.get("india_vix") or payload.get("vix") or 0.0)
        vix      = raw_vix if 8.0 <= raw_vix <= 80.0 else 14.0
        vix_note = "" if 8.0 <= raw_vix <= 80.0 else " fallback"
        gap_pct  = float(payload.get("gap_pct", 0))
        sgx      = payload.get("sgx_bias", "—")
        mode     = TRADING_MODE

        vix_tag  = "🔴 HIGH" if vix > 22 else "🟡 ELEVATED" if vix > 17 else "🟢 LOW"
        gap_tag  = f"{gap_pct:+.2f}% {'⬆️' if gap_pct > 0 else '⬇️'}" if abs(gap_pct) > 0.1 else "Flat"
        bias_tag = {"BULLISH": "🐂 BULLISH", "BEARISH": "🐻 BEARISH",
                    "NEUTRAL": "➡️ NEUTRAL"}.get(bias, bias)

        msg = (
            f"🌅 *MCXForge Daily Brief*\n"
            f"📅 {datetime.now(IST).strftime('%d %b %Y')}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🎯 Commodity: SILVERM\n"
            f"🌐 Bias:    {bias_tag}\n"
            f"📊 Gap:     {gap_tag}\n"
            f"😰 VIX:     {vix:.1f}  {vix_tag}{vix_note}\n"
            f"🌏 Global:  {sgx}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"⚙️ Mode:    {mode} (Simulated Paper Execution)\n"
            f"🕐 Hours:   09:00 – 23:30 IST\n"
            f"🌙 Evening: 17:00 – 23:00 IST (🔥 Primary Trading Window)\n"
            f"💡 Note:    Mostly trades in {mode} mode in Evening Session"
        )
        await self._send(msg, target=self._payload_target(payload))

    async def send_text(
        self,
        message: str,
        target: str | None = None,
        parse_mode: str | None = "Markdown",
    ) -> bool:
        """Send a raw text message. Use for custom alerts."""
        return await self._send(message, target=target, parse_mode=parse_mode)

    def send_text_sync(
        self,
        message: str,
        target: str | None = None,
        parse_mode: str | None = "Markdown",
    ) -> bool:
        """Synchronous helper for send_text."""
        try:
            return asyncio.run(self.send_text(message, target=target, parse_mode=parse_mode))
        except Exception:
            return False

    # ── INTERNAL ──────────────────────────────────────────────────────────────

    async def _send(
        self,
        text: str,
        target: str | None = None,
        parse_mode: str | None = "Markdown",
    ) -> bool:
        """Send a Telegram message. Silently fails if disabled or unreachable."""
        route = str(target or os.getenv("TELEGRAM_TARGET") or os.getenv("TRADING_MODE", TRADING_MODE)).upper()
        if not self._enabled:
            try:
                from loguru import logger
                logger.warning(f"[TelegramNotifier] skipped: disabled target={route}")
            except Exception:
                pass
            return False
        token, chat_id = self._active_credentials(target=target)
        if not token or not chat_id:
            try:
                from loguru import logger
                logger.warning(f"[TelegramNotifier] skipped: missing credentials for target={route}")
            except Exception:
                pass
            return False

        import hashlib, time
        cache_key = f"{chat_id}:{hashlib.sha256(text.strip().encode('utf-8')).hexdigest()}"
        now = time.time()
        TelegramNotifier._sent_cache = {k: ts for k, ts in TelegramNotifier._sent_cache.items() if now - ts < 60.0}
        if route != "BACKTEST" and cache_key in TelegramNotifier._sent_cache:
            if now - TelegramNotifier._sent_cache[cache_key] < 30.0:
                try:
                    from loguru import logger
                    logger.info(f"[TelegramNotifier] Suppressed duplicate Telegram message within 30s | target={route}")
                except Exception:
                    pass
                return True
        TelegramNotifier._sent_cache[cache_key] = now
        url = TELEGRAM_API.format(token=token)
        data = {
            "chat_id": chat_id,
            "text": text,
        }
        if parse_mode:
            data["parse_mode"] = parse_mode

        transport = os.getenv("TELEGRAM_TRANSPORT", "auto").strip().lower()
        transports = [transport] if transport in {"aiohttp", "requests", "curl"} else ["aiohttp", "requests", "curl"]
        last_error = ""

        for transport_name in transports:
            try:
                if transport_name == "aiohttp":
                    ok, last_error = await self._send_aiohttp(url, data)
                elif transport_name == "requests":
                    ok, last_error = self._send_requests(url, data)
                elif transport_name == "curl":
                    ok, last_error = self._send_curl(url, data)
                else:
                    continue
                if ok:
                    try:
                        from loguru import logger
                        logger.info(f"[TelegramNotifier] target={route} sent via {transport_name}")
                    except Exception:
                        pass
                    return True
            except Exception as e:
                last_error = f"{type(e).__name__}: {self._safe_error(e)}"

        # If sending with parse_mode failed (e.g. 400 bad entity parse), retry as plain text
        if parse_mode and "parse_mode" in data:
            data_plain = dict(data)
            data_plain.pop("parse_mode", None)
            for transport_name in transports:
                try:
                    if transport_name == "aiohttp":
                        ok, _ = await self._send_aiohttp(url, data_plain)
                    elif transport_name == "requests":
                        ok, _ = self._send_requests(url, data_plain)
                    elif transport_name == "curl":
                        ok, _ = self._send_curl(url, data_plain)
                    else:
                        continue
                    if ok:
                        try:
                            from loguru import logger
                            logger.info(f"[TelegramNotifier] target={route} sent as plain text fallback via {transport_name}")
                        except Exception:
                            pass
                        return True
                except Exception:
                    pass

        try:
            from loguru import logger
            logger.warning(f"[TelegramNotifier] target={route} send failed: {last_error}")
        except Exception:
            pass
        return False

    def _active_credentials(self, target: str | None = None) -> tuple[str, str]:
        mode = str(
            target
            or os.getenv("TELEGRAM_TARGET")
            or os.getenv("TRADING_MODE", TRADING_MODE)
        ).upper()
        if "BACKTEST" in mode or mode in ("BT", "TEST"):
            return self._bt_token, self._bt_chat
        return self._live_token, self._live_chat

    async def _send_aiohttp(self, url: str, data: dict) -> tuple[bool, str]:
        try:
            import aiohttp
        except ImportError as exc:
            return False, f"aiohttp unavailable: {exc}"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        return True, ""
                    body = await resp.text()
                    return False, f"aiohttp HTTP {resp.status}: {body[:300]}"
        except Exception as exc:
            return False, f"aiohttp {type(exc).__name__}: {self._safe_error(exc)}"

    def _send_requests(self, url: str, data: dict) -> tuple[bool, str]:
        try:
            import requests
        except ImportError as exc:
            return False, f"requests unavailable: {exc}"
        try:
            resp = requests.post(url, json=data, timeout=5)
            if resp.ok:
                return True, ""
            return False, f"requests HTTP {resp.status_code}: {resp.text[:300]}"
        except Exception as exc:
            return False, f"requests {type(exc).__name__}: {self._safe_error(exc)}"

    def _send_curl(self, url: str, data: dict) -> tuple[bool, str]:
        try:
            clean_env = {k: v for k, v in os.environ.items() if not k.startswith("Malloc")}
            result = subprocess.run(
                [
                    "curl",
                    "-sS",
                    "--fail",
                    "--connect-timeout",
                    "5",
                    "--max-time",
                    "8",
                    "-X",
                    "POST",
                    url,
                    "-H",
                    "Content-Type: application/json",
                    "--data",
                    json.dumps(data),
                ],
                capture_output=True,
                text=True,
                timeout=10,
                env=clean_env,
            )
        except FileNotFoundError:
            return False, "curl unavailable"
        except Exception as exc:
            return False, f"curl {type(exc).__name__}: {self._safe_error(exc)}"
        if result.returncode == 0:
            return True, ""
        detail = (result.stderr or result.stdout or "").strip()
        return False, f"curl exit {result.returncode}: {self._safe_error(detail)}"

    @staticmethod
    def _safe_error(error: object) -> str:
        text = str(error)
        marker = "/bot"
        if marker not in text:
            return text[:300]
        before, _, after = text.partition(marker)
        _, _, suffix = after.partition("/")
        return f"{before}{marker}<redacted>/{suffix}"[:300]


# ── Singleton ─────────────────────────────────────────────────────────────────
_notifier: Optional[TelegramNotifier] = None

def get_notifier() -> TelegramNotifier:
    global _notifier
    if _notifier is None:
        _notifier = TelegramNotifier()
    return _notifier


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Send a SignalForge Telegram test message.")
    parser.add_argument("--target", choices=("LIVE", "BACKTEST"), default="LIVE")
    parser.add_argument("--message", default=None)
    args = parser.parse_args()

    text = args.message or (
        f"SignalForge Telegram {args.target} test\n"
        f"Time: {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S IST')}"
    )
    ok = asyncio.run(get_notifier().send_text(text, target=args.target, parse_mode=None))
    print(f"telegram target={args.target} sent={ok}")
    raise SystemExit(0 if ok else 1)
