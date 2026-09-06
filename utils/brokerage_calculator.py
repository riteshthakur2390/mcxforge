"""
utils/brokerage_calculator.py
=============================
Comprehensive Brokerage & Statutory Taxes Calculator for MCX Commodity Futures
and Indian Financial Markets.

MCX Commodity Futures Charges (SEBI & MCX circulars):
  1. Brokerage: ₹20 per executed order (flat ₹40 round-trip)
  2. CTT (Commodities Transaction Tax): 0.01% on sell-side futures turnover
  3. Exchange Transaction Charges (MCX): ~0.0021% on total turnover (buy + sell)
  4. SEBI Turnover Charges: ₹10 per crore (0.0001% of total turnover)
  5. Stamp Duty: 0.002% on buy-side turnover
  6. GST: 18% on (Brokerage + MCX Exchange Charges + SEBI Charges)
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class TradeCharges:
    brokerage: float
    stt: float            # CTT for commodities / STT for options
    exchange_charges: float
    sebi_charges: float
    stamp_duty: float
    gst: float
    total_charges: float
    gross_pnl_inr: float
    net_pnl_inr: float
    net_pnl_pct: float

    def to_dict(self) -> dict:
        return {
            "brokerage": self.brokerage,
            "stt": self.stt,
            "ctt": self.stt,
            "exchange_charges": self.exchange_charges,
            "sebi_charges": self.sebi_charges,
            "stamp_duty": self.stamp_duty,
            "gst": self.gst,
            "total_charges": self.total_charges,
            "gross_pnl_inr": self.gross_pnl_inr,
            "net_pnl_inr": self.net_pnl_inr,
            "net_pnl_pct": self.net_pnl_pct,
        }


def calculate_commodity_trade_charges(
    entry_price: float,
    exit_price: float,
    quantity: int,
    direction: str = "BUY",
    tick_size: float = 1.0,
    tick_value: float = 1.0,
    order_brokerage: float = 20.0,
    orders_count: int = 2,
) -> TradeCharges:
    """
    Calculate exact brokerage and statutory charges for an MCX commodity futures trade.

    Args:
        entry_price: execution price at entry
        exit_price: execution price at exit
        quantity: total lots/units traded
        direction: 'BUY' (long) or 'SELL' (short)
        tick_size: minimum price fluctuation of instrument
        tick_value: rupee value per tick per lot
        order_brokerage: brokerage per order (default ₹20)
        orders_count: number of orders (default 2: entry + exit)

    Returns:
        TradeCharges dataclass with itemized taxes and net PnL
    """
    qty = max(float(quantity or 1), 1.0)
    mult = (tick_value / tick_size) if tick_size > 0 else 1.0
    # Determine physical trade units:
    # If qty is already in physical units (e.g. qty >= mult and mult > 1.0), use qty;
    # Otherwise if qty was given in lots (e.g. 1 lot), units = qty * mult.
    trade_units = qty if (qty >= mult and mult > 1.0) else (qty * mult)

    is_long = str(direction).upper() in ("BUY", "BUY_CALL", "LONG")
    if is_long:
        buy_price = entry_price
        sell_price = exit_price
        gross_pnl_inr = round((exit_price - entry_price) * trade_units, 2)
    else:
        buy_price = exit_price
        sell_price = entry_price
        gross_pnl_inr = round((entry_price - exit_price) * trade_units, 2)

    buy_turnover = buy_price * trade_units
    sell_turnover = sell_price * trade_units
    total_turnover = buy_turnover + sell_turnover

    # 1. Brokerage (flat ₹20 per order)
    brokerage = round(order_brokerage * orders_count, 2)

    # 2. CTT (Commodities Transaction Tax): 0.01% on sell side futures turnover
    ctt = round(sell_turnover * 0.00010, 2)

    # 3. MCX Exchange Transaction Charges: 0.0021% on total turnover
    exchange_charges = round(total_turnover * 0.000021, 2)

    # 4. SEBI Turnover Charges: ₹10 per crore (0.0001% of total turnover)
    sebi_charges = round(total_turnover * 0.000001, 2)

    # 5. Stamp Duty: 0.002% on buy value
    stamp_duty = round(buy_turnover * 0.00002, 2)

    # 6. GST: 18% on (Brokerage + Exchange Charges + SEBI Charges)
    gst = round((brokerage + exchange_charges + sebi_charges) * 0.18, 2)

    total_charges = round(brokerage + ctt + exchange_charges + sebi_charges + stamp_duty + gst, 2)
    net_pnl_inr = round(gross_pnl_inr - total_charges, 2)
    margin_est = max(buy_turnover * 0.12, 1000.0)
    net_pnl_pct = round((net_pnl_inr / margin_est) * 100.0, 2)

    return TradeCharges(
        brokerage=brokerage,
        stt=ctt,
        exchange_charges=exchange_charges,
        sebi_charges=sebi_charges,
        stamp_duty=stamp_duty,
        gst=gst,
        total_charges=total_charges,
        gross_pnl_inr=gross_pnl_inr,
        net_pnl_inr=net_pnl_inr,
        net_pnl_pct=net_pnl_pct,
    )


def calculate_option_trade_charges(
    entry_premium: float,
    exit_premium: float,
    quantity: int,
    order_brokerage: float = 20.0,
    orders_count: int = 2,
) -> TradeCharges:
    """Legacy option trade charges calculator for backward compatibility."""
    entry_val = max(float(entry_premium or 0.0), 0.0) * max(int(quantity or 0), 1)
    exit_val = max(float(exit_premium or 0.0), 0.0) * max(int(quantity or 0), 1)
    total_turnover = entry_val + exit_val

    brokerage = round(order_brokerage * orders_count, 2)
    stt = round(exit_val * 0.0010, 2)
    exchange_charges = round(total_turnover * 0.00050, 2)
    sebi_charges = round(total_turnover * 0.000001, 2)
    stamp_duty = round(entry_val * 0.00003, 2)
    gst = round((brokerage + exchange_charges + sebi_charges) * 0.18, 2)
    total_charges = round(brokerage + stt + exchange_charges + sebi_charges + stamp_duty + gst, 2)

    gross_pnl_inr = round(exit_val - entry_val, 2)
    net_pnl_inr = round(gross_pnl_inr - total_charges, 2)
    net_pnl_pct = round((net_pnl_inr / max(entry_val, 1.0)) * 100.0, 2)

    return TradeCharges(
        brokerage=brokerage,
        stt=stt,
        exchange_charges=exchange_charges,
        sebi_charges=sebi_charges,
        stamp_duty=stamp_duty,
        gst=gst,
        total_charges=total_charges,
        gross_pnl_inr=gross_pnl_inr,
        net_pnl_inr=net_pnl_inr,
        net_pnl_pct=net_pnl_pct,
    )

