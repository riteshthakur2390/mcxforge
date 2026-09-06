# SKILL: Market Regime Agent (Agent 9)

## Purpose
Gate 1 of the signal pipeline. Classifies commodity market conditions and session liquidity dynamics BEFORE strategies generate actionable trade signals for **Commodity Option Buying (BUY CALL / BUY PUT ONLY)**.
Option buying requires strong directional momentum to overcome premium theta decay.

## Key Files
- `agents_code/agent9_regime/classifier.py` — Regime classifier
- `agents_code/agent9_regime/regime_detector.py` — Indicator math (ADX, Choppiness, Volatility)

## Classification Logic
```
Session == MORNING && NOISE_FILTER_ON → CHOPPY    → SUPPRESS (Avoid Asian whipsaws)
Tender Period Active                  → LOCKOUT   → SUPPRESS (Avoid physical delivery risk)
ADX < 18 OR Chop > 61.8              → CHOPPY    → SUPPRESS (Protects option buyer against theta decay)
ADX 18–25                            → RANGING   → Selective Mean-Reversion eligible
ADX > 25                             → TRENDING  → Trend Following & Breakout eligible (Ideal for Option Buying)
ATR Ratio > 2.0                      → HIGH_VOL  → Tighten stops / reduce position sizing
```

## Session Dynamics
- **Morning (09:00–13:00 IST)**: Suppressed by default due to high noise-to-signal ratio and thin Asian volume.
- **Afternoon (13:00–17:00 IST)**: Selective evaluation.
- **Evening (17:00–23:30 IST)**: Unlocked for full consensus execution. US COMEX institutional volume provides genuine follow-through, expanding option deltas.

## Bus Topics
- Subscribes: `CANDLES_READY`, `PREMARKET_BIAS`
- Publishes: `MARKET_REGIME` (when active), `SIGNAL_SUPPRESSED` (when filtered)

## Critical Rule
`SIGNAL_SUPPRESSED` prevents unwanted noise and churn. On low-liquidity or choppy sessions, silence is the edge.
