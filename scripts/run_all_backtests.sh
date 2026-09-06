#!/bin/bash

LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

export TRADING_MODE=BACKTEST
export TELEGRAM_TARGET=BACKTEST
export BACKTEST_TELEGRAM_ENABLED="${BACKTEST_TELEGRAM_ENABLED:-true}"

#for DAYS in 30 60 90 120 150 180 210 240 270 300 360 450 540 630 720 810 900 990 1080 1170 1260 1350 1440 1530 1620 1710 1800
for DAYS in 60 90 120 150 300 600
#for DAYS in 300 600
#for DAYS in 1230
do
LOG_FILE="${LOG_DIR}/backtest_$(date +%Y%m%d)-${DAYS}days.log"

echo "========================================" >> "$LOG_FILE"
echo "Starting backtest for $DAYS days at $(date)" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

./scripts/run_backtest.sh --runs 1 --days "$DAYS" >> "$LOG_FILE" 2>&1

echo "Completed $DAYS days run at $(date)" >> "$LOG_FILE"

done
