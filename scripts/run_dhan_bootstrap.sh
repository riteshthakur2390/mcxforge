#!/usr/bin/env bash
set -euo pipefail

PHASE="${1:-}"
shift || true

if [[ -z "$PHASE" ]]; then
  echo "Usage: $0 underlying|options3|options10|all|status|validate [extra args]"
  exit 1
fi

mkdir -p logs
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="logs/dhan_bootstrap_${PHASE}_${STAMP}.log"

case "$PHASE" in
  underlying)
    CMD=(python3 scripts/dhan_bootstrap.py underlying --intervals 1minute,5minute "$@")
    ;;
  options3)
    CMD=(python3 scripts/dhan_bootstrap.py options --interval 5 --offsets 3 "$@")
    ;;
  options10)
    CMD=(python3 scripts/dhan_bootstrap.py options --interval 5 --offsets 10 "$@")
    ;;
  all)
    CMD=(python3 scripts/dhan_bootstrap.py all --intervals 1minute,5minute --interval 5 --offsets 3 "$@")
    ;;
  status)
    exec python3 scripts/dhan_bootstrap.py status "$@"
    ;;
  validate)
    exec python3 scripts/dhan_bootstrap.py validate "$@"
    ;;
  *)
    echo "Unknown phase: $PHASE"
    exit 1
    ;;
esac

nohup "${CMD[@]}" >"$LOG" 2>&1 &
PID=$!
echo "$PID" > "${LOG}.pid"
echo "Started PID $PID"
echo "Log: $LOG"
echo "Monitor: tail -f $LOG"
echo "Status: ./scripts/run_dhan_bootstrap.sh status"
