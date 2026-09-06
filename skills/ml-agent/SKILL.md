# SKILL: ML Filter Agent (Agent 3)

## Purpose
Gate 2 of the signal pipeline. Secondary validation and ranking classifier. Extracts 40+ quantitative features and predicts trade win probability for **Commodity Option Buying (BUY CALL / BUY PUT ONLY)** signals.
Signal passes to planning only if $P(win) \ge \text{ML\_MIN\_CONFIDENCE}$ (default: 0.55).

## Key Files
- `agents_code/agent3_ml/filter.py` — ML filter agent
- `ml/features.py` — Shared feature extraction pipeline
- `ml/model.py` — `SignalForgeEnsemble` (XGBoost + LightGBM + RandomForest)
- `scripts/train_ml.py` — Model training CLI

## Feature Engineering for Commodities (40+ Features)
- **Price Action**: Body-to-range ratio, wick ratios, candle momentum streak
- **Volatility**: ATR(14) normalized by price, Bollinger Band bandwidth, Squeeze indicator
- **Trend & Momentum**: ADX, +DI/-DI, MACD histogram, RSI(14), RSI slope
- **Microstructure & Volume**: Volume ratio vs 20-period SMA, Cumulative Volume Delta (CVD) imbalance
- **Macro & Cross-Asset**: USD-INR 1-day rate of change, Gold/Silver ratio z-score
- **Session Phase**: One-hot encoded market window (Morning, Afternoon, Evening COMEX)
- **Consensus Metadata**: Raw strategy votes, mean confidence, top strategy agreement

## Model Ensemble
- **XGBoost Classifier** (40% weight)
- **LightGBM Classifier** (40% weight)
- **RandomForest Classifier** (20% weight)
- Probability output: Calibrated $P(win) \in [0.0, 1.0]$

## Fallback Mode
If no trained model is present in `ml/saved_models/`, the agent transparently falls back to the raw strategy ensemble confidence, ensuring operational continuity.

## Bus Topics
- Subscribes: `RAW_SIGNAL`, `CANDLES_READY`, `MARKET_REGIME`
- Publishes: `SIGNAL_APPROVED` (if probability meets gate), `SIGNAL_REJECTED` (with explicit rejection reason)
