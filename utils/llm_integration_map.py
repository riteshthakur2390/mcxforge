"""
llm_integration_map.py — SignalForge Module × LLM Call Mapping
===============================================================
Drop-in usage examples for every LLM call across SignalForge modules.
Import `call_llm_async` and `TaskType` from `utils.llm`, then adapt the relevant snippet.
"""

from utils.llm import TaskType, call_llm_async as call_llm


# ═══════════════════════════════════════════════════════════════
# MODULE 1: signal_generator.py
# Routing: DeepSeek (signal explanations are non-sensitive)
# ═══════════════════════════════════════════════════════════════

def explain_signal(symbol: str, signal_type: str, indicators: dict) -> str:
    """
    Called after a buy/sell signal fires — explains why in plain English.
    STANDARD → DeepSeek
    """
    prompt = f"""
Symbol: {symbol}
Signal: {signal_type}
Indicators:
  RSI: {indicators.get('rsi')}
  MACD: {indicators.get('macd')} | Signal: {indicators.get('macd_signal')}
  BB Upper: {indicators.get('bb_upper')} | BB Lower: {indicators.get('bb_lower')}
  Price: {indicators.get('close')}

Explain in 2-3 sentences why this {signal_type} signal fired based on the above indicators.
Be specific and concise. No disclaimers.
"""
    return call_llm(
        task_type=TaskType.SIGNAL_EXPLANATION,
        prompt=prompt,
        system="You are a technical analysis expert for Indian equity markets."
    )


# ═══════════════════════════════════════════════════════════════
# MODULE 2: backtester.py
# Routing: DeepSeek (backtest results are historical, not live account data)
# ═══════════════════════════════════════════════════════════════

def narrate_backtest(results: dict) -> str:
    """
    Summarizes a completed backtest run in plain English.
    STANDARD → DeepSeek
    """
    prompt = f"""
Backtest Results for strategy: {results.get('strategy_name')}
Period: {results.get('start_date')} to {results.get('end_date')}
Symbol: {results.get('symbol')}

Metrics:
  Total Trades: {results.get('total_trades')}
  Win Rate: {results.get('win_rate')}%
  Total PnL: {results.get('total_pnl')}
  Max Drawdown: {results.get('max_drawdown')}%
  Sharpe Ratio: {results.get('sharpe_ratio')}
  Avg Trade Duration: {results.get('avg_duration_mins')} mins

Write a 3-4 sentence performance summary. Highlight strengths and weaknesses.
Suggest one concrete improvement if performance is below expectations.
"""
    return call_llm(
        task_type=TaskType.BACKTEST_NARRATION,
        prompt=prompt,
        system="You are a quantitative analyst reviewing algorithmic trading backtest results."
    )


# ═══════════════════════════════════════════════════════════════
# MODULE 3: indicator_engine.py
# Routing: DeepSeek (indicator commentary is purely analytical)
# ═══════════════════════════════════════════════════════════════

def comment_on_indicators(symbol: str, timeframe: str, indicators: dict) -> str:
    """
    Generates plain-English commentary on current indicator state.
    STANDARD → DeepSeek
    """
    prompt = f"""
Symbol: {symbol} | Timeframe: {timeframe}

Current Indicator Readings:
  RSI(14): {indicators.get('rsi')}
  MACD: {indicators.get('macd')} | Signal Line: {indicators.get('macd_signal')}
  ATR(14): {indicators.get('atr')}
  Bollinger %B: {indicators.get('bb_pct_b')}
  Volume Ratio vs 20-day avg: {indicators.get('volume_ratio')}x

Give a 2-sentence market condition summary based solely on these readings.
State if the instrument is overbought/oversold/trending/ranging.
"""
    return call_llm(
        task_type=TaskType.INDICATOR_COMMENTARY,
        prompt=prompt,
        system="You are a technical analyst specializing in NSE-listed Indian equities and derivatives."
    )


# ═══════════════════════════════════════════════════════════════
# MODULE 4: news_fetcher.py / sentiment_analyzer.py
# Routing: DeepSeek (public news, no account data involved)
# ═══════════════════════════════════════════════════════════════

def score_news_sentiment(headline: str, body: str, symbol: str) -> dict:
    """
    Scores a news article for market sentiment.
    STANDARD → DeepSeek
    Returns structured JSON-like response.
    """
    prompt = f"""
Stock: {symbol}
Headline: {headline}
Article snippet: {body[:500]}

Respond in this exact format (no extra text):
SENTIMENT: <bullish|bearish|neutral>
CONFIDENCE: <0-100>
REASON: <one sentence>
IMPACT_HORIZON: <intraday|swing|long-term>
"""
    return call_llm(
        task_type=TaskType.NEWS_SENTIMENT,
        prompt=prompt,
        system="You are a financial news sentiment analyzer for Indian equity markets."
    )


# ═══════════════════════════════════════════════════════════════
# MODULE 5: strategy_manager.py
# Routing: DeepSeek (strategy descriptions are non-sensitive config)
# ═══════════════════════════════════════════════════════════════

def describe_strategy(strategy_config: dict) -> str:
    """
    Generates a human-readable description of a strategy config.
    STANDARD → DeepSeek
    """
    prompt = f"""
Strategy Name: {strategy_config.get('name')}
Entry Conditions: {strategy_config.get('entry_conditions')}
Exit Conditions: {strategy_config.get('exit_conditions')}
Timeframe: {strategy_config.get('timeframe')}
Instruments: {strategy_config.get('instruments')}
Position Sizing: {strategy_config.get('position_sizing')}

Write a 3-sentence plain-English description of this strategy.
Suitable for a strategy card in a trading dashboard.
"""
    return call_llm(
        task_type=TaskType.STRATEGY_DESCRIPTION,
        prompt=prompt,
        system="You are a quantitative trading strategy documentation writer."
    )


# ═══════════════════════════════════════════════════════════════
# MODULE 6: pattern_detector.py
# Routing: DeepSeek (pattern narration is analytical, no account data)
# ═══════════════════════════════════════════════════════════════

def narrate_pattern(symbol: str, pattern_name: str, context: dict) -> str:
    """
    Explains a detected candlestick / chart pattern.
    STANDARD → DeepSeek
    """
    prompt = f"""
Symbol: {symbol}
Pattern Detected: {pattern_name}
Context:
  Prior Trend: {context.get('prior_trend')}
  Pattern Location: {context.get('location')}  (e.g., near support/resistance)
  Volume on pattern candles: {context.get('volume_note')}

Explain this pattern in 2-3 sentences. State its typical implication and
what confirmation signal a trader should wait for before acting.
"""
    return call_llm(
        task_type=TaskType.PATTERN_SUMMARY,
        prompt=prompt,
        system="You are a technical analysis expert specializing in candlestick patterns."
    )


# ═══════════════════════════════════════════════════════════════
# MODULE 7: alert_manager.py
# Routing: DeepSeek (alert text generation, no sensitive data)
# ═══════════════════════════════════════════════════════════════

def generate_alert_text(alert_type: str, symbol: str, details: dict) -> str:
    """
    Generates a concise alert message for Telegram/email notifications.
    STANDARD → DeepSeek
    """
    prompt = f"""
Alert Type: {alert_type}
Symbol: {symbol}
Details: {details}

Write a concise alert message (max 2 lines) suitable for a Telegram notification.
Format: Emoji + symbol + core message. No fluff.
Example: 🔴 NIFTY | RSI Divergence detected on 15m chart. Possible reversal zone.
"""
    return call_llm(
        task_type=TaskType.ALERT_TEXT,
        prompt=prompt,
        system="You are a trading alert system generating concise notifications for traders."
    )


# ═══════════════════════════════════════════════════════════════
# MODULE 8: risk_manager.py
# Routing: Claude Haiku ← SENSITIVE (real position/exposure data)
# ═══════════════════════════════════════════════════════════════

def validate_risk(order: dict, portfolio_state: dict) -> str:
    """
    Validates an order against risk rules before execution.
    SENSITIVE → Claude Haiku
    """
    prompt = f"""
Proposed Order:
  Symbol: {order.get('symbol')}
  Side: {order.get('side')}
  Quantity: {order.get('quantity')}
  Order Type: {order.get('order_type')}
  Price: {order.get('price')}

Current Portfolio State:
  Total Capital: {portfolio_state.get('total_capital')}
  Capital at Risk: {portfolio_state.get('capital_at_risk')}
  Open Positions: {portfolio_state.get('open_positions')}
  Daily PnL: {portfolio_state.get('daily_pnl')}
  Max Daily Loss Limit: {portfolio_state.get('max_daily_loss')}

Rules:
  - Max single trade risk: 2% of capital
  - Max daily loss: 5% of capital
  - Max concurrent positions: 5

Respond with APPROVED or REJECTED, followed by one-sentence reason.
If REJECTED, state which rule was violated.
"""
    return call_llm(
        task_type=TaskType.RISK_CHECK,
        prompt=prompt,
        system="You are a strict risk management engine for an algorithmic trading system. Never approve orders that violate rules."
    )


# ═══════════════════════════════════════════════════════════════
# MODULE 9: order_executor.py
# Routing: Claude Haiku ← SENSITIVE (live order placement)
# ═══════════════════════════════════════════════════════════════

def pre_execution_check(order: dict) -> str:
    """
    Final sanity check before sending order to broker API.
    SENSITIVE → Claude Haiku
    """
    prompt = f"""
Order about to be placed:
  Symbol: {order.get('symbol')}
  Exchange: {order.get('exchange')}
  Transaction Type: {order.get('transaction_type')}
  Product: {order.get('product')}   (CNC/MIS/NRML)
  Order Type: {order.get('order_type')}
  Quantity: {order.get('quantity')}
  Price: {order.get('price')}
  Trigger Price: {order.get('trigger_price')}

Check for:
1. Any obviously incorrect values (e.g. negative qty, zero price for LIMIT order)
2. Product type appropriate for instrument type
3. Any anomaly that suggests a data pipeline error

Respond: PROCEED or ABORT + one-sentence reason.
"""
    return call_llm(
        task_type=TaskType.ORDER_EXECUTION,
        prompt=prompt,
        system="You are a pre-execution validation engine. Be conservative. When in doubt, ABORT."
    )


# ═══════════════════════════════════════════════════════════════
# MODULE 10: portfolio_tracker.py
# Routing: Claude Haiku ← SENSITIVE (real account/PnL data)
# ═══════════════════════════════════════════════════════════════

def summarize_portfolio(holdings: list, pnl_data: dict) -> str:
    """
    Generates a portfolio health summary — contains real account data.
    SENSITIVE → Claude Haiku
    """
    prompt = f"""
Portfolio Holdings:
{holdings}

PnL Summary:
  Realized PnL (today): {pnl_data.get('realized_today')}
  Unrealized PnL: {pnl_data.get('unrealized')}
  Total PnL (month): {pnl_data.get('total_month')}
  Best Performer: {pnl_data.get('best_performer')}
  Worst Performer: {pnl_data.get('worst_performer')}

Write a 3-sentence portfolio health summary.
Flag any positions with unrealized loss > 5%.
"""
    return call_llm(
        task_type=TaskType.PORTFOLIO_SUMMARY,
        prompt=prompt,
        system="You are a portfolio management assistant. Be factual and concise."
    )
