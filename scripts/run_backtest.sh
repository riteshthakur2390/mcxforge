#!/bin/bash

# ./scripts/run_backtest.sh --runs 1 --days 30
# ./scripts/run_backtest.sh --runs 1 --days 60 --live-parity
# ./scripts/run_backtest.sh --runs 1 --max
# ./scripts/run_backtest.sh --runs 3 --days 90 --python python3

# Default values
RUNS=10
DAYS=30
MAX_MODE=false
TRADING_DAYS=""
PYTHON_PATH="venv/bin/python"
SCRIPT_PATH="scripts/backtest_runner.py"
BACKTEST_TIMEFRAME="5minute"
LIVE_PARITY=false

# Parse arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --runs) RUNS="$2"; shift ;;
    --days) DAYS="$2"; shift ;;
    --trading-days) TRADING_DAYS="$2"; shift ;;
    --max) MAX_MODE=true ;;
    --live-parity) LIVE_PARITY=true ;;
    --python) PYTHON_PATH="$2"; shift ;;
    --script) SCRIPT_PATH="$2"; shift ;;
    --timeframe) BACKTEST_TIMEFRAME="$2"; shift ;;
    --commodity|--mcx) SCRIPT_PATH="scripts/run_30day_commodity_backtest.py" ;;
    *) echo "Unknown parameter passed: $1"; exit 1 ;;
  esac
  shift
done

unset MallocStackLogging
unset MallocStackLoggingNoCompact
unset MallocNanoZone

export BACKTEST_TIMEFRAME="$BACKTEST_TIMEFRAME"
export TRADING_MODE=BACKTEST
export TELEGRAM_TARGET=BACKTEST
export BACKTEST_TELEGRAM_ENABLED="${BACKTEST_TELEGRAM_ENABLED:-true}"
export BACKTEST_SIGNAL="$BACKTEST_TIMEFRAME"
export SIGNAL_TIMEFRAME="$BACKTEST_TIMEFRAME"
export PYTHONWARNINGS="ignore::ImportWarning,ignore:.*sklearn\\.utils\\.parallel\\.delayed.*:UserWarning:sklearn.utils.parallel"
export LIVE_TIMEFRAME="$BACKTEST_TIMEFRAME"
export BACKTEST_ENABLE_HISTORICAL_OPTION_PREMIUM=true
export BACKTEST_STRICT_OPTION_BROKER=true 
export BACKTEST_REQUIRE_HISTORICAL_OPTION_PREMIUM=true

BACKTEST_ARGS=()
if [[ "$LIVE_PARITY" == "true" ]]; then
  BACKTEST_ARGS+=(--live-parity)
fi

# Execution loop
for ((i=1; i<=RUNS; i++)); do
  SELECTED_DAYS="${TRADING_DAYS:-$DAYS}"
  if [[ "$MAX_MODE" == "true" ]]; then
    SELECTED_DAYS="all_available"
  fi
  MODE="historical-research"
  if [[ "$LIVE_PARITY" == "true" ]]; then
    MODE="live-parity"
  fi
  echo "Run $i started at $(date) | timeframe=$BACKTEST_TIMEFRAME | trading_days=$SELECTED_DAYS | mode=$MODE"
  if [[ "$MAX_MODE" == "true" ]]; then
    "$PYTHON_PATH" "$SCRIPT_PATH" --max "${BACKTEST_ARGS[@]}"
  elif [[ -n "$TRADING_DAYS" ]]; then
    "$PYTHON_PATH" "$SCRIPT_PATH" --trading-days "$TRADING_DAYS" "${BACKTEST_ARGS[@]}"
  else
    "$PYTHON_PATH" "$SCRIPT_PATH" --trading-days "$DAYS" "${BACKTEST_ARGS[@]}"
  fi
  echo "Run $i completed at $(date)"
  echo "----------------------------------------"
done
