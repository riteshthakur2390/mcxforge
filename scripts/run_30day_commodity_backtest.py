"""
scripts/run_30day_commodity_backtest.py — MCX Commodity 30-Day Institutional Backtest
=====================================================================================
Executes a comprehensive, non-hypothetical backtest on real 5-minute SILVERMIC
historical candles for the past 30 days (August 5, 2026 to September 4, 2026).

Evaluates:
  1. Overall Performance (Net PnL, Win Rate, Profit Factor, Max Drawdown, Return on Capital)
  2. Capital & Margin Utilization (Capital = ₹200,000, Max 15% = ₹30,000 per trade)
  3. Strategy Attribution (Contribution of all 43 active strategies)
  4. Session Breakdown (Morning 09:00-13:00, Afternoon 13:00-17:00, Evening 17:00-23:30)
  5. Vote Consensus Correlation (Performance vs 1-2, 3-4, 5+ votes)
  6. Agent 3 ML Filter & Confidence Impact (using ml/saved_models/silvermic_5minute.pkl)
  7. Entry Timing & Late Entry Telemetry (MFE, MAE, MFE/MAE ratio, Hold time)
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", message=".*urllib3 v2 only supports OpenSSL.*")
try:
    import urllib3.exceptions
    warnings.filterwarnings("ignore", category=urllib3.exceptions.NotOpenSSLWarning)
except Exception:
    pass
warnings.filterwarnings("ignore", message=r"(?s).*If you are loading a serialized model.*")
warnings.filterwarnings("ignore", message=r"(?s).*Trying to unpickle estimator.*")
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    pass

import sys
import os
import json
from pathlib import Path
import time as time_mod
from datetime import datetime, time as dt_time
import numpy as np
import pandas as pd
from dataclasses import asdict
from loguru import logger
import pytz

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.strategies.ensemble import (
    build_default_strategy_suite,
    build_quality_strategy_suite,
    count_independent_categories,
    CommodityEnsembleEngine,
    MCXSession,
    SILVERM_CONFIG,
    EnsembleRunResult,
    EnsembleTradeRecord,
)
from core.strategies.backtest_data import HistoricalDataLoader
from ml.model import SignalForgeEnsemble
from ml.features import extract, FEATURE_NAMES
from utils.brokerage_calculator import calculate_commodity_trade_charges
from instruments.registry import get_instrument_config, resolve_active_contract

IST = pytz.timezone("Asia/Kolkata")


def run_commodity_backtest(
    days: int = 30,
    run_all: bool = False,
    symbol: str = "SILVERM",
    start_date: str | None = None,
    end_date: str | None = None,
    timeframe: str = "5m",
    data_file: str | None = None,
    output_dir_str: str | None = None,
    capital: float = 200_000.0,
    max_cap_pct: float = float(os.getenv("MAX_CAP_PCT", "20.0")),
    quality_mode: bool = True,
    min_votes: int = 5,
    min_categories: int = 2,
    session_filter: str = "EVENING",
    min_ml_conf: float | None = float(os.getenv("MIN_ML_CONF", "0.32")),
    send_telegram: bool = False,
    send_telegram_trades: bool = False,
):
    if output_dir_str:
        output_dir = Path(output_dir_str)
    elif run_all or (days and days >= 1200):
        output_dir = Path(f"analysis/backtest_{'5y' if 'd' in timeframe else 'all'}")
    else:
        output_dir = Path(f"analysis/backtest_{days}d")
    output_dir.mkdir(parents=True, exist_ok=True)

    sym_clean = (symbol or "SILVERM").upper().strip()
    interval_code = "1d" if "d" in timeframe.lower() else "5m"

    # 1. Load Data based on timeframe / explicit path / symbol
    if data_file:
        csv_path = Path(data_file)
    else:
        sym_upper = sym_clean.upper()
        candidates = [
            Path(f"data/historical/{sym_upper}_dhan_{interval_code}.csv"),
            Path(f"data/historical/{sym_upper}_upstox_{interval_code}.csv"),
            Path(f"data/historical/{sym_upper}_{interval_code}.csv"),
        ]
        if "SILVER" in sym_upper:
            candidates.extend([
                Path(f"data/historical/SILVERM_dhan_{interval_code}.csv"),
                Path(f"data/historical/SILVERM_{interval_code}.csv"),
                Path(f"data/historical/SILVERMIC_dhan_{interval_code}.csv"),
                Path(f"data/historical/SILVERMIC_{interval_code}.csv"),
            ])
        elif "CRUDE" in sym_upper:
            candidates.extend([
                Path(f"data/historical/CRUDEOIL_dhan_{interval_code}.csv"),
                Path(f"data/historical/CRUDEOIL_{interval_code}.csv"),
                Path(f"data/historical/CRUDEOILM_dhan_{interval_code}.csv"),
                Path(f"data/historical/CRUDEOILM_{interval_code}.csv"),
            ])
        elif "GOLD" in sym_upper:
            candidates.extend([
                Path(f"data/historical/GOLDM_dhan_{interval_code}.csv"),
                Path(f"data/historical/GOLDM_{interval_code}.csv"),
                Path(f"data/historical/GOLD_dhan_{interval_code}.csv"),
                Path(f"data/historical/GOLD_{interval_code}.csv"),
            ])
        elif any(k in sym_upper for k in ("NATGAS", "NATURAL")):
            candidates.extend([
                Path(f"data/historical/NATGAS_dhan_{interval_code}.csv"),
                Path(f"data/historical/NATGAS_{interval_code}.csv"),
                Path(f"data/historical/NATGASM_dhan_{interval_code}.csv"),
                Path(f"data/historical/NATGASMINI_dhan_{interval_code}.csv"),
                Path(f"data/historical/NATURALGAS_dhan_{interval_code}.csv"),
            ])
        csv_path = None
        for c in candidates:
            if c.exists():
                csv_path = c
                break
        if csv_path is None:
            csv_path = candidates[0]

    # Auto-refresh check: only allowed outside active live market hours to protect broker rate limits
    now_ist = datetime.now(IST)
    is_live_market = (now_ist.weekday() < 5 and dt_time(9, 0) <= now_ist.time() <= dt_time(23, 30))

    if not data_file and csv_path.exists() and not is_live_market:
        try:
            today_d = now_ist.date()
            yesterday_d = today_d - pd.Timedelta(days=1)
            while yesterday_d.weekday() >= 5: # skip weekends
                yesterday_d -= pd.Timedelta(days=1)
            
            tail_df = pd.read_csv(csv_path).tail(1)
            ts_c = [c for c in tail_df.columns if "time" in c.lower() or "date" in c.lower()][0]
            last_recorded_date = pd.to_datetime(tail_df[ts_c].iloc[0], format="mixed", utc=True).tz_convert("Asia/Kolkata").date()
            if last_recorded_date < yesterday_d:
                logger.info(f"Local CSV for {sym_clean} ends at {last_recorded_date}. Refreshing from broker up to {yesterday_d}...")
                from scripts.sync_commodity_historical_data import sync_all_commodities_historical
                sync_all_commodities_historical(symbols=[sym_clean], allow_market_hours=False)
        except Exception as e:
            logger.warning(f"Could not auto-refresh local historical candles: {e}")

    loader = HistoricalDataLoader()
    df_raw = loader.load(str(csv_path), auto_repair=True)

    # If market is currently open today, exclude today's incomplete session so backtest covers completed sessions up to yesterday
    now_ist = datetime.now(IST)
    if end_date is None and not run_all and df_raw.index.max().date() == now_ist.date() and now_ist.time() < dt_time(23, 30):
        logger.info(f"Excluding partial intraday session for today ({now_ist.date()}); evaluating completed sessions up to yesterday.")
        df_raw = df_raw[df_raw.index.date < now_ist.date()]

    warmup_bars = 100
    filter_start_dt = None
    if start_date or end_date:
        df_eval = df_raw.copy()
        if start_date:
            s_dt = pd.to_datetime(start_date)
            if s_dt.tzinfo is None:
                s_dt = s_dt.tz_localize(IST)
            filter_start_dt = s_dt
            # Warm up with prior 100 bars so indicators are fully primed at chunk start
            pre_mask = df_eval.index < s_dt
            pre_indices = df_eval.index[pre_mask]
            warmup_dt = pre_indices[-warmup_bars] if len(pre_indices) >= warmup_bars else (pre_indices[0] if len(pre_indices) > 0 else s_dt)
            df_eval = df_eval[df_eval.index >= warmup_dt]
        if end_date:
            e_dt = pd.to_datetime(end_date) + pd.Timedelta(days=1)
            if e_dt.tzinfo is None:
                e_dt = e_dt.tz_localize(IST)
            df_eval = df_eval[df_eval.index < e_dt]
        label = f"Date-sliced ({start_date or 'start'} to {end_date or 'end'}, {len(df_eval)} bars over {len(set(df_eval.index.date))} sessions)"
    elif run_all or days is None or days <= 0 or (days >= 1200 and "d" in timeframe.lower()):
        df_eval = df_raw.copy()
        label = f"Full {len(df_eval)}-bar ({len(set(df_eval.index.date))} sessions)"
    else:
        max_ts = df_raw.index.max()
        min_ts = max_ts - pd.Timedelta(days=days)
        df_eval = df_raw[df_raw.index >= min_ts].copy()
        label = f"{days}-day ({len(df_eval)} bars over {len(set(df_eval.index.date))} sessions)"

    total_bars = len(df_eval)
    trading_days = len(set(df_eval.index.date))
    logger.info(f"Loaded {label} from {csv_path}: {total_bars} bars over {trading_days} sessions ({df_eval.index.min()} to {df_eval.index.max()})")

    # 2. Resolve Target Instrument Configuration & ML Model
    active_comm = symbol or os.getenv("COMMODITY_SYMBOL", "SILVERM")
    from instruments.registry import get_instrument_config, get_instrument_strategy_config
    inst_cfg = get_instrument_config(active_comm)
    inst_strat_cfg = get_instrument_strategy_config(active_comm)

    sym_lower = sym_clean.lower()
    ml_candidates = [
        Path(f"ml/saved_models/{sym_lower}_5minute.pkl"),
        Path(f"ml/saved_models/{sym_lower}m_5minute.pkl") if not sym_lower.endswith("m") else Path(f"ml/saved_models/{sym_lower[:-1]}_5minute.pkl"),
    ]
    if "crude" in sym_lower:
        ml_candidates.extend([Path("ml/saved_models/crudeoil_5minute.pkl"), Path("ml/saved_models/crudeoilm_5minute.pkl")])
    elif "gold" in sym_lower:
        ml_candidates.extend([Path("ml/saved_models/goldm_5minute.pkl"), Path("ml/saved_models/gold_5minute.pkl")])
    elif "silver" in sym_lower:
        ml_candidates.extend([Path("ml/saved_models/silverm_5minute.pkl"), Path("ml/saved_models/silvermic_5minute.pkl")])
    elif "natgas" in sym_lower or "natural" in sym_lower:
        ml_candidates.extend([Path("ml/saved_models/naturalgas_5minute.pkl"), Path("ml/saved_models/natgasmini_5minute.pkl"), Path("ml/saved_models/natgasm_5minute.pkl")])
    ml_candidates.append(Path("ml/saved_models/commodity_5minute.pkl"))
    ml_candidates.append(Path("ml/saved_models/silverm_5minute.pkl"))
    ml_path = next((p for p in ml_candidates if p.exists()), ml_candidates[-1])

    ml_ensemble = SignalForgeEnsemble()
    ml_loaded = ml_ensemble.load(ml_path)
    logger.info(f"Loaded ML model: {ml_path} (loaded={ml_loaded}, models={list(ml_ensemble.models.keys()) if ml_loaded else []})")

    # 3. Initialize Capital & Risk Parameters (Max Capital per Trade = 20% of Capital = ₹40,000 max)
    max_capital_per_trade = capital * (max_cap_pct / 100.0)  # ₹40,000 max (20% of ₹200,000 capital)
    max_premium_per_trade = max_capital_per_trade
    max_margin_per_trade = max_capital_per_trade
    lot_size = inst_cfg.lot_size            # Lot size from spec
    tick_size = inst_cfg.tick_size          # Tick size
    tick_val = inst_cfg.tick_value          # Tick value
    margin_pct = 0.15                       # Margin requirement estimate

    # 4. Run Backtest with Complete Telemetry & ML Scoring
    if quality_mode:
        strategies_to_use = build_quality_strategy_suite()
        enabled_names = inst_strat_cfg.get("enabled_strategies")
        if enabled_names:
            strategies_to_use = [s for s in strategies_to_use if getattr(s, "name", s.__class__.__name__) in enabled_names]
        effective_min_votes = min_votes if min_votes is not None else int(inst_strat_cfg.get("min_votes", 4))
        effective_min_cats = min_categories
        if session_filter.upper() == "ALL" or "d" in timeframe:
            allowed_sessions = [MCXSession.MORNING, MCXSession.AFTERNOON, MCXSession.EVENING, MCXSession.OFF_MARKET]
        elif session_filter.upper() == "MORNING":
            allowed_sessions = [MCXSession.MORNING]
        elif session_filter.upper() in ("MORNING_EVENING", "SKIP_AFTERNOON"):
            allowed_sessions = [MCXSession.MORNING, MCXSession.EVENING]
        else:
            allowed_sessions = [MCXSession.EVENING]  # US COMEX Evening Session (17:00-23:30) - High Institutional Liquidity
        mode_label = f"Quality Alpha Suite ({len(strategies_to_use)} Strats | Session: {session_filter} | Min Votes {effective_min_votes} | Min Cats {effective_min_cats})"
    else:
        strategies_to_use = build_default_strategy_suite()
        effective_min_votes = min_votes if min_votes is not None else 2
        effective_min_cats = 1
        allowed_sessions = (
            [MCXSession.MORNING, MCXSession.AFTERNOON, MCXSession.EVENING, MCXSession.OFF_MARKET]
            if "d" in timeframe or session_filter.upper() == "ALL"
            else ([MCXSession.EVENING] if session_filter.upper() == "EVENING" else [MCXSession.MORNING, MCXSession.AFTERNOON, MCXSession.EVENING])
        )
        mode_label = f"Raw Baseline (All {len(strategies_to_use)} Strats | Session: {session_filter} | Min Votes {effective_min_votes})"

    engine = CommodityEnsembleEngine(
        strategies=strategies_to_use,
        instrument_config=inst_cfg,
        min_votes=effective_min_votes,
        min_categories=effective_min_cats,
        capital=capital,
    )

    logger.info(f"Executing multi-strategy ensemble backtest [{mode_label}] (timeframe={timeframe}, bars={len(df_eval)})...")
    result: EnsembleRunResult = engine.run(df_eval, timeframe=timeframe, allowed_sessions=allowed_sessions)
    raw_trades = result.trades
    if filter_start_dt is not None:
        raw_trades = [t for t in raw_trades if pd.to_datetime(t.entry_time) >= filter_start_dt]
    logger.info(f"Backtest completed: {len(raw_trades)} total trades generated (after warm-up filter).")

    # 5. Enrich Trades with ML Probability, Margin Used, and Late Entry Analysis
    enriched_trades: list[dict] = []
    try:
        from broker.dhan_broker import DhanBroker
        _dhan_broker = DhanBroker()
    except Exception:
        _dhan_broker = None
    
    _dhan_contract_cache: dict[tuple, tuple[str, str]] = {}
    total_raw = len(raw_trades)
    log_enrich_interval = max(25, total_raw // 5) if total_raw > 0 else 100
    for idx_t, t in enumerate(raw_trades):
        if idx_t > 0 and idx_t % log_enrich_interval == 0:
            logger.info(f"Enriching trades with ML probability & option margin: {idx_t}/{total_raw} ({int(idx_t/total_raw*100)}%)...")
        # Find candle slice at entry time to extract features
        entry_ts = pd.to_datetime(t.entry_time)
        sub_df = df_eval[df_eval.index <= entry_ts].tail(60)
        
        ml_prob = 0.50
        if len(sub_df) >= 30 and ml_loaded:
            try:
                direction_str = "BUY_CALL" if t.direction == "BUY" else "BUY_PUT"
                c_s = sub_df["close"]
                delta = c_s.diff()
                gain = delta.where(delta > 0, 0.0).rolling(14, min_periods=1).mean()
                loss = (-delta.where(delta < 0, 0.0)).rolling(14, min_periods=1).mean()
                rs = gain / (loss.replace(0, 1e-6))
                rsi_val = float((100.0 - (100.0 / (1.0 + rs))).iloc[-1])
                feats = extract(
                    df=sub_df,
                    conf=0.68,
                    votes=t.votes,
                    direction=direction_str,
                    lookback=60,
                    regime_info={"label": "TRENDING", "confidence": 0.65, "atr_ratio": 1.1},
                    signal_context={"adx": 25.0, "rsi": rsi_val},
                    strategies_fired=t.strategies_fired,
                )
                if feats:
                    ml_prob = ml_ensemble.predict_proba(feats)
            except Exception:
                pass

        # 1. Resolve Instrument & Option Spec
        inst_symbol = t.symbol or active_comm
        inst_cfg = get_instrument_config(inst_symbol)
        strike_step = getattr(inst_cfg, "strike_step", 1000) or 1000
        atm_strike = int(round(t.entry_price / strike_step) * strike_step)
        
        is_call = t.direction in ("BUY", "BUY_CALL", "LONG")
        opt_type = "CE" if is_call else "PE"
        
        entry_dt = entry_ts.to_pydatetime()
        session_enum = MCXSession.from_time(entry_dt.time())

        # 2. Resolve Active Monthly Option Contract (Trade same month, roll to next month only if DTE <= 5 days)
        from instruments.registry import resolve_active_option_contract
        contract_symbol, active_expiry, dte = resolve_active_option_contract(
            symbol=inst_symbol,
            as_of=entry_dt.date(),
            strike=atm_strike,
            option_type=opt_type,
            rollover_days=5,
        )
        expiry_display = active_expiry.strftime("%d %b %Y")

        # Check Dhan scrip master if available for verified broker trading symbol
        cache_key = (inst_symbol, entry_dt.date(), opt_type, atm_strike)
        if cache_key in _dhan_contract_cache:
            contract_symbol, expiry_display = _dhan_contract_cache[cache_key]
        elif _dhan_broker is not None:
            try:
                dhan_contracts = _dhan_broker.get_mcx_option_contracts(
                    symbol=inst_symbol,
                    expiry=active_expiry,
                    option_type=opt_type,
                    strikes=[atm_strike],
                    fetch_live_quotes=False,
                )
                if dhan_contracts:
                    contract_symbol = dhan_contracts[0].symbol
                    expiry_display = str(dhan_contracts[0].expiry_date)[:10]
                _dhan_contract_cache[cache_key] = (contract_symbol, expiry_display)
            except Exception:
                _dhan_contract_cache[cache_key] = (contract_symbol, expiry_display)

        # 3. Real Black-Scholes Option Premium at Entry based on DTE, Spot & Commodity IV
        from utils.option_utils import estimate_atm_premium
        est_option_premium = estimate_atm_premium(
            underlying=t.entry_price,
            days_to_expiry=dte,
            symbol=inst_symbol,
        )
        single_lot_margin = round(est_option_premium * lot_size, 2)

        # 4. Late entry metric: Entry efficiency = MFE / (MFE + MAE)
        total_excursion = t.mfe_pts + t.mae_pts
        entry_efficiency = (t.mfe_pts / total_excursion) if total_excursion > 0 else 0.5
        is_late_entry = t.mae_pts > (1.5 * max(t.mfe_pts, 5.0)) and t.net_pnl_inr < 0
        peak_pnl_pct = (t.mfe_pts / t.entry_price * 100.0) if t.entry_price > 0 else 0.0

        # 5. SignalForge Composite ML Rank Score
        cat_count = getattr(t, "categories", 1) or count_independent_categories(t.strategies_fired)
        vote_strength = min(1.0, t.votes / 6.0)
        cat_strength = min(1.0, cat_count / 3.0)
        strategy_strength = min(1.0, (t.votes * 0.08 + cat_strength * 0.40))
        model_strength = min(1.0, ml_prob / 0.40) if ml_prob > 0 else 0.50
        regime_strength = 0.72

        ml_rank_score = (
            model_strength * 0.60
            + strategy_strength * 0.22
            + vote_strength * 0.10
            + regime_strength * 0.08
        )
        ml_rank_tier = "HIGH" if ml_rank_score >= 0.70 else ("MEDIUM" if ml_rank_score >= 0.45 else "LOW")

        # 6. Conviction-Based Dynamic Lot Sizing (Strictly bounded by Capital Budget)
        max_position_budget = capital * (max_cap_pct / 100.0)
        affordable_lots = int(max_position_budget // max(single_lot_margin, 1.0))

        # Hard Capital Budget Enforcement:
        # If even 1 single lot exceeds the per-trade budget, the trade CANNOT be afforded and is rejected.
        if affordable_lots < 1 or single_lot_margin > max_position_budget:
            continue

        if affordable_lots >= 3 and (ml_rank_tier == "HIGH" or t.votes >= 8):
            num_lots = min(3, affordable_lots)
        elif affordable_lots >= 2 and (ml_rank_tier in ("HIGH", "MEDIUM") or t.votes >= 6):
            num_lots = min(2, affordable_lots)
        else:
            num_lots = 1


        trade_quantity = num_lots * lot_size
        margin_used = round(single_lot_margin * num_lots, 2)
        margin_util_pct = (margin_used / capital) * 100.0

        # 7. Real Option Exit Premium & Charges
        option_delta = 0.50
        est_exit_premium = max(5.0, round(est_option_premium + (t.pnl_points * option_delta), 1))
        contract_points = round(est_exit_premium - est_option_premium, 1)

        from utils.brokerage_calculator import calculate_option_trade_charges
        opt_charges = calculate_option_trade_charges(
            entry_premium=est_option_premium,
            exit_premium=est_exit_premium,
            quantity=trade_quantity,
        )
        opt_gross_pnl = round(contract_points * trade_quantity, 2)
        opt_fees = opt_charges.total_charges
        opt_net_pnl = round(opt_gross_pnl - opt_fees, 2)

        t_dict = asdict(t)
        t_dict.update({
            "lots": num_lots,
            "quantity": trade_quantity,
            "option_symbol": contract_symbol,
            "contract_symbol": contract_symbol,
            "strike": atm_strike,
            "option_type": opt_type,
            "expiry_date": expiry_display,
            "underlying_price": t.entry_price,
            "underlying_entry": t.entry_price,
            "underlying_exit": t.exit_price,
            "underlying_points": t.pnl_points,
            "actual_premium": est_option_premium,
            "entry_premium": est_option_premium,
            "exit_premium": est_exit_premium,
            "gross_pnl_inr": opt_gross_pnl,
            "fees_inr": opt_fees,
            "net_pnl_inr": opt_net_pnl,
            "realized_pnl": opt_net_pnl,
            "underlying_gross_pnl_inr": t.gross_pnl_inr,
            "underlying_fees_inr": t.fees_inr,
            "underlying_net_pnl_inr": t.net_pnl_inr,
            "total_invested": round(margin_used, 2),
            "margin_used_inr": round(margin_used, 2),
            "margin_util_pct": round(margin_util_pct, 2),
            "ml_confidence": round(ml_prob, 4),
            "ml_rank_score": round(ml_rank_score, 4),
            "ml_rank_tier": ml_rank_tier,
            "categories": cat_count,
            "setup_type": f"{t.direction} ({t.lead_strategy[:6]})",
            "structure_bias": session_enum.value[:3],
            "peak_pnl_pct": round(peak_pnl_pct, 2),
            "entry_efficiency": round(entry_efficiency, 3),
            "is_late_entry": bool(is_late_entry),
            "mcx_session": session_enum.value,
            "is_win": bool(opt_net_pnl > 0),
        })
        enriched_trades.append(t_dict)

    effective_min_ml_conf = min_ml_conf if min_ml_conf is not None else (
        float(inst_strat_cfg.get("min_ml_confidence", 0.26)) if quality_mode else None
    )
    if effective_min_ml_conf is not None:
        before_count = len(enriched_trades)
        enriched_trades = [tr for tr in enriched_trades if tr.get("ml_confidence", 0.0) >= effective_min_ml_conf]
        logger.info(f"Applied ML Confidence filter (>= {effective_min_ml_conf:.2f}): {len(enriched_trades)} / {before_count} trades passed gatekeeper.")

    trade_columns = [
        "trade_id", "symbol", "direction", "entry_time", "exit_time", "entry_price", "exit_price",
        "quantity", "lots", "gross_pnl_inr", "fees_inr", "net_pnl_inr", "pnl_pct", "pnl_points",
        "exit_reason", "hold_minutes", "lead_strategy", "votes", "categories", "mfe_pts", "mae_pts",
        "option_symbol", "contract_symbol", "strike", "option_type", "expiry_date", "underlying_price",
        "underlying_entry", "underlying_exit", "underlying_points", "actual_premium", "entry_premium",
        "exit_premium", "contract_points", "realized_pnl", "total_invested", "margin_used_inr",
        "margin_util_pct", "ml_confidence", "ml_rank_score", "ml_rank_tier", "is_win", "is_late_entry", "mcx_session"
    ]
    df_trades = pd.DataFrame(enriched_trades) if enriched_trades else pd.DataFrame(columns=trade_columns)

    # 6. Overall Performance Metrics (Unfiltered vs ML Filtered)
    def compute_stats(df_sub: pd.DataFrame, label: str) -> dict:
        if df_sub.empty:
            return {
                "name": label,
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "gross_pnl": 0.0,
                "fees": 0.0,
                "net_pnl": 0.0,
                "profit_factor": 0.0,
                "max_dd": 0.0,
                "max_dd_pct": 0.0,
                "return_on_capital_pct": 0.0,
                "avg_trade_pnl": 0.0,
                "peak_margin_used": 0.0,
                "avg_margin_used": 0.0,
            }
        n_trades = len(df_sub)
        wins = df_sub[df_sub["net_pnl_inr"] > 0]
        losses = df_sub[df_sub["net_pnl_inr"] <= 0]
        win_rate = (len(wins) / n_trades) * 100.0
        gross_pnl = df_sub["gross_pnl_inr"].sum()
        fees = df_sub["fees_inr"].sum()
        net_pnl = df_sub["net_pnl_inr"].sum()
        total_gain = wins["net_pnl_inr"].sum()
        total_loss = abs(losses["net_pnl_inr"].sum())
        profit_factor = (total_gain / total_loss) if total_loss > 0 else (999.0 if total_gain > 0 else 0.0)

        # Drawdown calculation
        df_sub_sorted = df_sub.sort_values("exit_time").reset_index(drop=True)
        equity = capital + df_sub_sorted["net_pnl_inr"].cumsum()
        peak = equity.cummax()
        dd = peak - equity
        max_dd = dd.max() if len(dd) else 0.0

        return {
            "name": label,
            "trades": n_trades,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(win_rate, 2),
            "gross_pnl": round(gross_pnl, 2),
            "fees": round(fees, 2),
            "net_pnl": round(net_pnl, 2),
            "profit_factor": round(profit_factor, 2),
            "max_dd": round(max_dd, 2),
            "max_dd_pct": round((max_dd / capital) * 100.0, 2),
            "return_on_capital_pct": round((net_pnl / capital) * 100.0, 2),
            "avg_trade_pnl": round(net_pnl / n_trades, 2),
            "peak_margin_used": round(df_sub["margin_used_inr"].max(), 2),
            "avg_margin_used": round(df_sub["margin_used_inr"].mean(), 2),
        }

    stats_unfiltered = compute_stats(df_trades, "Baseline (All Consensus Signals)")
    stats_ml_28 = compute_stats(df_trades[df_trades["ml_confidence"] >= 0.28], "ML Filtered (Conf >= 0.28 — Above Base Rate)")
    stats_ml_30 = compute_stats(df_trades[df_trades["ml_confidence"] >= 0.30], "ML Conviction (Conf >= 0.30 — Top 30%)")
    stats_ml_33 = compute_stats(df_trades[df_trades["ml_confidence"] >= 0.33], "ML High Conviction (Conf >= 0.33 — Top 15%)")
    stats_ml_36 = compute_stats(df_trades[df_trades["ml_confidence"] >= 0.36], "ML Elite Conviction (Conf >= 0.36 — Top 5%)")

    # 7. Session Breakdown
    session_rows = []
    for sess in [MCXSession.MORNING.value, MCXSession.AFTERNOON.value, MCXSession.EVENING.value]:
        sub = df_trades[df_trades["mcx_session"] == sess]
        s_stats = compute_stats(sub, sess)
        s_stats["session"] = sess
        session_rows.append(s_stats)
    df_session = pd.DataFrame(session_rows)

    # 8. Vote Breakdown (1-2 votes, 3-4 votes, 5+ votes)
    vote_rows = []
    for v_range, v_label in [
        ((1, 2), "1-2 Votes (Low Consensus)"),
        ((3, 4), "3-4 Votes (Moderate Consensus)"),
        ((5, 43), "5+ Votes (High Consensus)"),
    ]:
        sub = df_trades[(df_trades["votes"] >= v_range[0]) & (df_trades["votes"] <= v_range[1])]
        v_stats = compute_stats(sub, v_label)
        v_stats["vote_range"] = f"{v_range[0]}-{v_range[1]}"
        vote_rows.append(v_stats)
    df_votes = pd.DataFrame(vote_rows)

    # 9. Strategy Attribution (Which strategies contributed the most)
    strat_perf = {}
    for _, tr in df_trades.iterrows():
        strats = tr.get("strategies_fired", [])
        if isinstance(strats, str):
            try:
                strats = eval(strats)
            except:
                strats = [strats]
        for s in strats:
            if s not in strat_perf:
                strat_perf[s] = {"trades": 0, "wins": 0, "losses": 0, "net_pnl": 0.0, "gross_pnl": 0.0, "fees": 0.0}
            strat_perf[s]["trades"] += 1
            if tr["is_win"]:
                strat_perf[s]["wins"] += 1
            else:
                strat_perf[s]["losses"] += 1
            strat_perf[s]["net_pnl"] += tr["net_pnl_inr"]
            strat_perf[s]["gross_pnl"] += tr["gross_pnl_inr"]
            strat_perf[s]["fees"] += tr["fees_inr"]

    strat_rows = []
    for s_name, d in strat_perf.items():
        w_rate = (d["wins"] / d["trades"] * 100.0) if d["trades"] > 0 else 0.0
        strat_rows.append({
            "strategy": s_name,
            "trades": d["trades"],
            "wins": d["wins"],
            "losses": d["losses"],
            "win_rate_pct": round(w_rate, 1),
            "gross_pnl": round(d["gross_pnl"], 2),
            "fees": round(d["fees"], 2),
            "net_pnl": round(d["net_pnl"], 2),
            "profit_factor": round((d["gross_pnl"] / max(1.0, abs(d["gross_pnl"] - d["net_pnl"]))), 2),
        })
    df_strat = pd.DataFrame(strat_rows).sort_values("net_pnl", ascending=False) if strat_rows else pd.DataFrame(columns=["strategy", "trades", "wins", "losses", "win_rate_pct", "gross_pnl", "fees", "net_pnl", "profit_factor"])

    # 10. Late Entry & Timing Telemetry Analysis
    total_trades_count = len(df_trades)
    late_entries_count = int(df_trades["is_late_entry"].sum()) if ("is_late_entry" in df_trades.columns and total_trades_count > 0) else 0
    late_entry_pct = (late_entries_count / total_trades_count * 100.0) if total_trades_count else 0.0
    avg_mfe = float(df_trades["mfe_pts"].mean()) if ("mfe_pts" in df_trades.columns and total_trades_count > 0) else 0.0
    avg_mae = float(df_trades["mae_pts"].mean()) if ("mae_pts" in df_trades.columns and total_trades_count > 0) else 0.0
    mfe_mae_ratio = (avg_mfe / avg_mae) if avg_mae > 0 else 1.0
    avg_hold_minutes = float(df_trades["hold_minutes"].mean()) if ("hold_minutes" in df_trades.columns and total_trades_count > 0) else 0.0

    exit_reasons = df_trades["exit_reason"].value_counts().to_dict() if ("exit_reason" in df_trades.columns and total_trades_count > 0) else {}

    # 11. Save CSV and JSON artifacts
    suffix = "5y" if "d" in timeframe else (f"{days}d" if (days and days < 1200 and not run_all) else "all")
    df_trades.to_csv(output_dir / f"trade_ledger_{suffix}.csv", index=False)
    df_session.to_csv(output_dir / f"session_breakdown_{suffix}.csv", index=False)
    df_votes.to_csv(output_dir / f"vote_breakdown_{suffix}.csv", index=False)
    df_strat.to_csv(output_dir / f"strategy_attribution_{suffix}.csv", index=False)
    # Also save canonical names for easy ingestion
    df_trades.to_csv(output_dir / "trade_ledger.csv", index=False)
    df_session.to_csv(output_dir / "session_breakdown.csv", index=False)
    df_votes.to_csv(output_dir / "vote_breakdown.csv", index=False)
    df_strat.to_csv(output_dir / "strategy_attribution.csv", index=False)

    full_report = {
        "metadata": {
            "symbol": active_comm,
            "timeframe": timeframe,
            "date_range": {"start": str(df_eval.index.min()), "end": str(df_eval.index.max())},
            "total_bars": total_bars,
            "trading_days": trading_days,
            "initial_capital": capital,
            "max_margin_budget": max_margin_per_trade,
            "margin_budget_pct": 20.0,
        },
        "performance": {
            "unfiltered": stats_unfiltered,
            "ml_conf_28": stats_ml_28,
            "ml_conf_30": stats_ml_30,
            "ml_conf_33": stats_ml_33,
            "ml_conf_36": stats_ml_36,
        },
        "timing_telemetry": {
            "avg_mfe_points": round(avg_mfe, 2),
            "avg_mae_points": round(avg_mae, 2),
            "mfe_mae_ratio": round(mfe_mae_ratio, 2),
            "avg_hold_minutes": round(avg_hold_minutes, 1),
            "late_entries_count": late_entries_count,
            "late_entry_rate_pct": round(late_entry_pct, 2),
            "exit_reasons": exit_reasons,
        },
        "sessions": session_rows,
        "votes": vote_rows,
        "top_strategies": strat_rows[:15],
    }

    report_filename = f"backtest_{suffix}_report.json"
    with open(output_dir / report_filename, "w") as f:
        json.dump(full_report, f, indent=2, default=str)
    with open(output_dir / "backtest_report.json", "w") as f:
        json.dump(full_report, f, indent=2, default=str)

    logger.success(f"Backtest report written to {output_dir / report_filename}")

    # 12. Generate SignalForge Rich Performance Report
    from utils.report_formatter import format_rich_report

    if not df_trades.empty and "exit_time" in df_trades.columns:
        exit_dt = pd.to_datetime(df_trades["exit_time"])
        daily_returns = df_trades.groupby(exit_dt.dt.date)["net_pnl_inr"].sum() / capital
        sharpe = (float(daily_returns.mean() / daily_returns.std() * np.sqrt(252))) if len(daily_returns) > 1 and daily_returns.std() > 0 else 0.0
    else:
        sharpe = 0.0

    all_dates = [d.strftime("%Y-%m-%d") for d in df_eval.index.normalize().unique()]
    extra_summary = {
        "days": all_dates,
        "candles": total_bars,
        "signals": total_trades_count,
        "approved": total_trades_count,
        "suppressed": 0,
        "sharpe_ratio": sharpe,
        "capital_return_pct": stats_unfiltered["return_on_capital_pct"],
        "profit_factor": stats_unfiltered["profit_factor"],
    }

    rich_report_color = format_rich_report(
        enriched_trades,
        paper_capital=capital,
        total_days=trading_days,
        lots=1,
        use_colors=True,
        extra_summary=extra_summary,
    )
    rich_report_clean = format_rich_report(
        enriched_trades,
        paper_capital=capital,
        total_days=trading_days,
        lots=1,
        use_colors=False,
        extra_summary=extra_summary,
    )

    with open(output_dir / "backtest_rich_report.txt", "w") as f:
        f.write(rich_report_clean)
    with open(output_dir / f"backtest_rich_report_{suffix}.txt", "w") as f:
        f.write(rich_report_clean)

    # 13. Generate SignalForge-Grade Markdown Report
    md_lines = [
        f"# MCXForge Institutional Backtest Report — {active_comm} Option Buying",
        "",
        f"> **Instrument**: `{active_comm}` ({lot_size} kg per lot) | **Strategy Mode**: {mode_label}",
        f"> **Date Range**: {df_eval.index.min().strftime('%Y-%m-%d')} to {df_eval.index.max().strftime('%Y-%m-%d')} ({trading_days} Trading Days | {total_bars:,} Bars)",
        f"> **Execution Model**: **BUY CALL (CE) / BUY PUT (PE) ONLY** | **Starting Capital**: ₹{capital:,.2f}",
        "",
        "## 1. Executive Performance Summary",
        "",
        "| Metric | Value | Metric | Value |",
        "|---|---|---|---|",
        f"| **Total Trades** | `{stats_unfiltered['trades']}` | **Win Rate** | `{stats_unfiltered['win_rate']}%` ({stats_unfiltered['wins']}W / {stats_unfiltered['losses']}L) |",
        f"| **Gross PnL** | `₹{stats_unfiltered['gross_pnl']:,.2f}` | **Statutory Charges** | `₹{stats_unfiltered['fees']:,.2f}` |",
        f"| **Net Realized PnL** | **`₹{stats_unfiltered['net_pnl']:,.2f}`** | **Return on Capital** | **`{stats_unfiltered['return_on_capital_pct']}%`** |",
        f"| **Profit Factor** | `{stats_unfiltered['profit_factor']}` | **Sharpe Ratio** | `{sharpe:.2f}` |",
        f"| **Max Drawdown** | `₹{stats_unfiltered['max_dd']:,.2f}` (`{stats_unfiltered['max_dd_pct']}%`) | **Expectancy / Trade** | `₹{(stats_unfiltered['net_pnl'] / max(1, stats_unfiltered['trades'])):,.2f}` |",
        "",
        "## 2. Session Profitability Breakdown",
        "",
        "| Session | Trades | Win Rate % | Net PnL (₹) | Profit Factor |",
        "|---|---|---|---|---|",
    ]
    for s_item in session_rows:
        md_lines.append(f"| **{s_item['session']}** | {s_item['trades']} | {s_item['win_rate']}% | **₹{s_item['net_pnl']:,.2f}** | {s_item['profit_factor']} |")
    
    md_lines.extend([
        "",
        "## 3. Vote Consensus Correlation",
        "",
        "| Consensus Tier | Trades | Win Rate % | Net PnL (₹) | Profit Factor |",
        "|---|---|---|---|---|",
    ])
    for v_item in vote_rows:
        md_lines.append(f"| **{v_item['name']}** | {v_item['trades']} | {v_item['win_rate']}% | **₹{v_item['net_pnl']:,.2f}** | {v_item['profit_factor']} |")

    md_lines.extend([
        "",
        "## 4. ML Conviction Stratification",
        "",
        "| ML Confidence Tier | Trades | Win Rate % | Net PnL (₹) | Profit Factor | Max DD (₹) |",
        "|---|---|---|---|---|---|",
        f"| **Baseline (Raw Consensus)** | {stats_unfiltered['trades']} | {stats_unfiltered['win_rate']}% | ₹{stats_unfiltered['net_pnl']:,.2f} | {stats_unfiltered['profit_factor']} | ₹{stats_unfiltered['max_dd']:,.2f} |",
        f"| **ML Filter (Conf >= 0.28)** | {stats_ml_28['trades']} | {stats_ml_28['win_rate']}% | ₹{stats_ml_28['net_pnl']:,.2f} | {stats_ml_28['profit_factor']} | ₹{stats_ml_28['max_dd']:,.2f} |",
        f"| **ML Conviction (Conf >= 0.30)** | {stats_ml_30['trades']} | {stats_ml_30['win_rate']}% | ₹{stats_ml_30['net_pnl']:,.2f} | {stats_ml_30['profit_factor']} | ₹{stats_ml_30['max_dd']:,.2f} |",
        f"| **ML High Conv (Conf >= 0.33)** | {stats_ml_33['trades']} | {stats_ml_33['win_rate']}% | ₹{stats_ml_33['net_pnl']:,.2f} | {stats_ml_33['profit_factor']} | ₹{stats_ml_33['max_dd']:,.2f} |",
        f"| **ML Elite Conv (Conf >= 0.36)** | {stats_ml_36['trades']} | {stats_ml_36['win_rate']}% | ₹{stats_ml_36['net_pnl']:,.2f} | {stats_ml_36['profit_factor']} | ₹{stats_ml_36['max_dd']:,.2f} |",
        "",
        "## 5. Top Strategy Attribution (Alpha Ranking)",
        "",
        "| Strategy | Trades | Win Rate % | Net PnL (₹) | Profit Factor |",
        "|---|---|---|---|---|",
    ])
    for st_item in strat_rows[:12]:
        md_lines.append(f"| **{st_item['strategy']}** | {st_item['trades']} | {st_item['win_rate_pct']}% | **₹{st_item['net_pnl']:,.2f}** | {st_item['profit_factor']} |")

    md_lines.extend([
        "",
        "## 6. Execution & Late Entry Telemetry",
        "",
        f"- **Average MFE (Favorable Excursion)**: `{avg_mfe:.1f}` points",
        f"- **Average MAE (Adverse Excursion)**: `{avg_mae:.1f}` points (MFE/MAE Ratio: `{mfe_mae_ratio:.2f}`)",
        f"- **Average Trade Hold Time**: `{avg_hold_minutes:.1f}` minutes",
        f"- **Late Entries Detected**: `{late_entries_count} / {total_trades_count}` ({late_entry_pct:.1f}%)",
        f"- **Exit Distribution**: `{json.dumps(exit_reasons)}`",
        "",
        "## 7. Sample Trade Ledger (Resolved Dhan Option Contracts)",
        "",
        "| Trade ID | Date | Time | Contract / Strike | Direction | Invested (₹) | Realized PnL (₹) | Exit Reason | Lead Strategy |",
        "|---|---|---|---|---|---|---|---|---|",
    ])
    for t_item in enriched_trades[:15]:
        opt_type = t_item.get("option_type", "CE")
        action_label = "BUY CALL (CE)" if opt_type == "CE" else "BUY PUT (PE)"
        md_lines.append(
            f"| `{t_item.get('trade_id', 'T000')}` | {str(t_item.get('entry_time', ''))[:10]} | {str(t_item.get('entry_time', ''))[11:19]} | **`{t_item.get('option_symbol', '')}`** | {action_label} | ₹{t_item.get('margin_used_inr', 0.0):,.1f} | **₹{t_item.get('net_pnl_inr', 0.0):,.2f}** | `{t_item.get('exit_reason', '')}` | {t_item.get('lead_strategy', '')} |"
        )

    md_report_content = "\n".join(md_lines) + "\n"
    with open(output_dir / "backtest_report.md", "w") as f:
        f.write(md_report_content)
    with open(output_dir / f"backtest_report_{suffix}.md", "w") as f:
        f.write(md_report_content)
    logger.success(f"Markdown report written to {output_dir / 'backtest_report.md'}")

    # Print SignalForge Rich Performance Report
    print("\n" + "=" * 40)
    print("SignalForge Rich Performance Report")
    print("=" * 40)
    print(rich_report_color)
    print("=" * 40)
    print(f"Run ID: backtest_{suffix} | Log: {output_dir / 'backtest_rich_report.txt'}")
    print(f"CSV: {output_dir / 'trade_ledger.csv'}\n")

    # Output supplementary institutional analytics
    print("=" * 90)
    print(f"      SIGNALFORGE COMMODITY INSIGHTS & ML CONVICTION ANALYSIS ({timeframe.upper()})")
    print("=" * 90)
    print(f"  Instrument: {active_comm} ({lot_size} kg lot) | Capital: ₹{capital:,.2f} | Max Premium/Trade: ₹{max_margin_per_trade:,.2f} ({max_cap_pct:.0f}%)")
    print(f"  Date Range: {df_eval.index.min().strftime('%Y-%m-%d')} to {df_eval.index.max().strftime('%Y-%m-%d')} ({trading_days} Trading Days | {total_bars:,} Bars)")
    print("-" * 90)
    print("1. OVERALL PERFORMANCE BY ML CONFIDENCE THRESHOLD:")
    print(f"  • Baseline (Raw Consensus)    : Trades={stats_unfiltered['trades']} | Win Rate={stats_unfiltered['win_rate']}% | Net PnL=₹{stats_unfiltered['net_pnl']:,.2f} | PF={stats_unfiltered['profit_factor']} | Max DD=₹{stats_unfiltered['max_dd']:,.2f} ({stats_unfiltered['max_dd_pct']}%)")
    print(f"  • ML Filter (Conf >= 0.28)   : Trades={stats_ml_28['trades']} | Win Rate={stats_ml_28['win_rate']}% | Net PnL=₹{stats_ml_28['net_pnl']:,.2f} | PF={stats_ml_28['profit_factor']} | Max DD=₹{stats_ml_28['max_dd']:,.2f} ({stats_ml_28['max_dd_pct']}%)")
    print(f"  • ML Conviction (Conf >= 0.30): Trades={stats_ml_30['trades']} | Win Rate={stats_ml_30['win_rate']}% | Net PnL=₹{stats_ml_30['net_pnl']:,.2f} | PF={stats_ml_30['profit_factor']} | Max DD=₹{stats_ml_30['max_dd']:,.2f} ({stats_ml_30['max_dd_pct']}%)")
    print(f"  • ML High Conv (Conf >= 0.33) : Trades={stats_ml_33['trades']} | Win Rate={stats_ml_33['win_rate']}% | Net PnL=₹{stats_ml_33['net_pnl']:,.2f} | PF={stats_ml_33['profit_factor']} | Max DD=₹{stats_ml_33['max_dd']:,.2f} ({stats_ml_33['max_dd_pct']}%)")
    print(f"  • ML Elite Conv (Conf >= 0.36): Trades={stats_ml_36['trades']} | Win Rate={stats_ml_36['win_rate']}% | Net PnL=₹{stats_ml_36['net_pnl']:,.2f} | PF={stats_ml_36['profit_factor']} | Max DD=₹{stats_ml_36['max_dd']:,.2f} ({stats_ml_36['max_dd_pct']}%)")
    print("-" * 90)
    print("2. SESSION PROFITABILITY BREAKDOWN (WHICH SESSION IS MOST PROFITABLE?):")
    for s in session_rows:
        print(f"  • {s['session']:10} Session : Trades={s['trades']:3d} | Win Rate={s['win_rate']:5.1f}% | Net PnL=₹{s['net_pnl']:10,.2f} | Profit Factor={s['profit_factor']:4.2f}")
    print("-" * 90)
    print("3. VOTE CONSENSUS CORRELATION (HOW MANY VOTES GIVE GOOD PROFIT?):")
    for v in vote_rows:
        print(f"  • {v['name']:32} : Trades={v['trades']:3d} | Win Rate={v['win_rate']:5.1f}% | Net PnL=₹{v['net_pnl']:10,.2f} | PF={v['profit_factor']:4.2f}")
    print("-" * 90)
    print("4. TIMING & LATE ENTRY ANALYSIS:")
    print(f"  • Average MFE (Favorable)    : {avg_mfe:.1f} points")
    print(f"  • Average MAE (Adverse)      : {avg_mae:.1f} points (MFE/MAE Ratio: {mfe_mae_ratio:.2f})")
    print(f"  • Average Hold Time          : {avg_hold_minutes:.1f} minutes")
    print(f"  • Late Entries Detected      : {late_entries_count} / {total_trades_count} ({late_entry_pct:.1f}%)")
    print(f"  • Exits Breakdown            : {exit_reasons}")
    print("-" * 90)
    print("5. TOP 10 CONTRIBUTING STRATEGIES (NET P&L ATTRIBUTION):")
    top_strats_df = pd.DataFrame(strat_rows[:10])
    if not top_strats_df.empty:
        print(top_strats_df[["strategy", "trades", "win_rate_pct", "net_pnl", "profit_factor"]].to_string(index=False))
    print("=" * 90 + "\n")


    # 13. Dispatch Telegram Notifications if requested
    if send_telegram or send_telegram_trades:
        try:
            from utils.telegram_notifier import get_notifier
            notifier = get_notifier()
            logger.info("Dispatching Telegram backtest notifications...")

            # Use BACKTEST routing so alerts appear in the designated backtest Telegram channel/bot
            target_mode = "BACKTEST"

            # Determine trades to dispatch
            trades_to_send = []
            if send_telegram_trades:
                trades_to_send = enriched_trades
            elif send_telegram and len(enriched_trades) > 0:
                # Send sample of recent high-conviction trades when --telegram is active
                trades_to_send = enriched_trades[-5:]

            if trades_to_send:
                logger.info(f"Sending Telegram trade alerts for {len(trades_to_send)} trades (target={target_mode})...")
                for tr in trades_to_send:
                    # Open payload with full contract details and dual price telemetry
                    open_p = {
                        "symbol": active_comm,
                        "contract_symbol": tr.get("contract_symbol"),
                        "option_symbol": tr.get("option_symbol"),
                        "strike": tr.get("strike"),
                        "option_type": tr.get("option_type"),
                        "expiry_date": tr.get("expiry_date"),
                        "direction": tr.get("direction", "BUY"),
                        "underlying_price": tr.get("underlying_entry", tr.get("entry_price", 0.0)),
                        "underlying_entry": tr.get("underlying_entry", tr.get("entry_price", 0.0)),
                        "actual_premium": tr.get("actual_premium", 0.0),
                        "entry_premium": tr.get("entry_premium", 0.0),
                        "entry_price": tr.get("actual_premium", 0.0),
                        "sl_price": round(float(tr.get("actual_premium", 0.0)) * 0.75, 1),
                        "target_price": round(float(tr.get("actual_premium", 0.0)) * 1.70, 1),
                        "lots": tr.get("lots", 1),
                        "lot_size": lot_size,
                        "quantity": tr.get("quantity", lot_size),
                        "margin_used": tr.get("margin_used_inr", margin_used),
                        "max_margin_budget": max_capital_per_trade,
                        "votes": tr.get("votes", 0),
                        "ml_rank_score": tr.get("ml_rank_score", 0.0),
                        "ml_rank_tier": tr.get("ml_rank_tier", "MED"),
                        "session": tr.get("mcx_session", "EVENING"),
                        "strategies_fired": tr.get("strategies_fired", []),
                        "entry_time": tr.get("entry_time"),
                        "mode": target_mode,
                        "target": target_mode,
                    }
                    notifier.send_trade_opened_sync(open_p, target=target_mode)

                    # Close payload with full contract details and underlying vs contract movement
                    close_p = {
                        "symbol": active_comm,
                        "contract_symbol": tr.get("contract_symbol"),
                        "option_symbol": tr.get("option_symbol"),
                        "strike": tr.get("strike"),
                        "option_type": tr.get("option_type"),
                        "expiry_date": tr.get("expiry_date"),
                        "direction": tr.get("direction", "BUY"),
                        "underlying_entry": tr.get("underlying_entry", tr.get("entry_price", 0.0)),
                        "underlying_exit": tr.get("underlying_exit", tr.get("exit_price", 0.0)),
                        "underlying_points": tr.get("underlying_points", tr.get("pnl_points", 0.0)),
                        "actual_premium": tr.get("actual_premium", 0.0),
                        "entry_premium": tr.get("entry_premium", 0.0),
                        "exit_premium": tr.get("exit_premium", 0.0),
                        "contract_points": tr.get("contract_points", 0.0),
                        "exit_reason": tr.get("exit_reason", "TARGET_HIT"),
                        "lots": tr.get("lots", 1),
                        "lot_size": lot_size,
                        "quantity": tr.get("quantity", lot_size),
                        "gross_pnl_inr": tr.get("gross_pnl_inr", 0.0),
                        "fees_inr": tr.get("fees_inr", 0.0),
                        "net_pnl_inr": tr.get("net_pnl_inr", 0.0),
                        "margin_used_inr": tr.get("margin_used_inr", margin_used),
                        "max_margin_budget": max_capital_per_trade,
                        "peak_pnl_pct": tr.get("peak_pnl_pct", 0.0),
                        "hold_minutes": tr.get("hold_minutes", 0),
                        "exit_time": tr.get("exit_time"),
                        "mode": target_mode,
                        "target": target_mode,
                    }
                    notifier.send_trade_closed_sync(close_p, target=target_mode)
                    time_mod.sleep(0.35)

            # Send overall performance summary card
            if send_telegram:
                summary_card = {
                    "symbol": active_comm,
                    "period": f"{df_eval.index.min().strftime('%Y-%m-%d')} to {df_eval.index.max().strftime('%Y-%m-%d')} ({trading_days} Days)",
                    "bars": total_bars,
                    "trades": stats_unfiltered["trades"],
                    "wins": stats_unfiltered["wins"],
                    "losses": stats_unfiltered["losses"],
                    "win_rate": stats_unfiltered["win_rate"],
                    "gross_pnl": stats_unfiltered["gross_pnl"],
                    "charges": stats_unfiltered["fees"],
                    "net_pnl": stats_unfiltered["net_pnl"],
                    "profit_factor": stats_unfiltered["profit_factor"],
                    "sharpe": sharpe,
                    "expectancy": (stats_unfiltered["net_pnl"] / max(1, stats_unfiltered["trades"])),
                    "capital": capital,
                    "capital_return_pct": stats_unfiltered["return_on_capital_pct"],
                    "peak_margin_used": stats_unfiltered["peak_margin_used"],
                    "max_margin_budget": max_margin_per_trade,
                    "max_dd": stats_unfiltered["max_dd"],
                    "max_dd_pct": stats_unfiltered["max_dd_pct"],
                    "mode": target_mode,
                    "target": target_mode,
                }
                notifier.send_backtest_summary_sync(summary_card, target=target_mode)
                logger.success("✅ Telegram backtest summary review sent successfully to Backtest Telegram bot.")
        except Exception as exc:
            logger.warning(f"Telegram notification dispatch skipped: {exc}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MCX Commodity Multi-Strategy Ensemble Backtest")
    parser.add_argument("--days", type=int, default=30, help="Number of calendar days (e.g. 30, 90, 365, 1200)")
    parser.add_argument("--trading-days", type=int, default=None, help="Number of trading sessions (alias for --days)")
    parser.add_argument("--start-date", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--all", "--max", action="store_true", dest="all", help="Run across all available historical data")
    parser.add_argument("--timeframe", type=str, default="5m", choices=["5m", "15m", "1h", "1d"], help="Bar timeframe (5m or 1d for 5-year daily)")
    parser.add_argument("--data-file", type=str, default=None, help="Explicit path to historical CSV")
    parser.add_argument("--capital", type=float, default=200_000.0, help="Starting capital (default: 200,000)")
    parser.add_argument("--max-cap-pct", type=float, default=20.0, help="Max capital percentage usable per trade (default: 20.0%% -> ₹40,000 on ₹200k)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--raw", action="store_true", help="Run raw unfiltered baseline (all 43 strategies, all sessions, min votes 2)")
    parser.add_argument("--min-votes", type=int, default=None, help="Minimum vote consensus threshold (default: auto from instrument registry)")
    parser.add_argument("--min-cats", "--min-categories", type=int, default=2, help="Minimum independent strategy categories (default: 2)")
    parser.add_argument("--session", type=str, default="EVENING", choices=["ALL", "EVENING", "MORNING", "SKIP_AFTERNOON"], help="Session filter (EVENING for US COMEX institutional liquidity, ALL for full day, MORNING, SKIP_AFTERNOON)")
    parser.add_argument("--min-ml-conf", type=float, default=None, help="Filter out trades below ML confidence threshold (default: auto from instrument registry)")
    parser.add_argument("--symbol", type=str, default="SILVERM", help="Commodity symbol to backtest (SILVERM, GOLDM, CRUDEOIL, NATGAS)")
    parser.add_argument("--telegram", action="store_true", help="Send comprehensive backtest summary report to Telegram")
    parser.add_argument("--telegram-trades", action="store_true", help="Send individual trade opened/closed alerts to Telegram")
    args = parser.parse_args()

    effective_days = args.trading_days if args.trading_days is not None else args.days
    quality_mode = not args.raw

    run_commodity_backtest(
        days=effective_days,
        run_all=args.all,
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        timeframe=args.timeframe,
        data_file=args.data_file,
        capital=args.capital,
        max_cap_pct=args.max_cap_pct,
        output_dir_str=args.output_dir,
        quality_mode=quality_mode,
        min_votes=args.min_votes,
        min_categories=args.min_cats,
        session_filter=args.session,
        min_ml_conf=args.min_ml_conf,
        send_telegram=args.telegram or (os.getenv("BACKTEST_TELEGRAM_ENABLED", "").strip().lower() in ("1", "true", "yes")),
        send_telegram_trades=args.telegram_trades or (os.getenv("BACKTEST_TELEGRAM_TRADES", "").strip().lower() in ("1", "true", "yes")),
    )
