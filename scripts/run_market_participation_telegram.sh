#!/usr/bin/env bash
set -u

cd /Users/vishranti/Downloads/projects/signalforge || exit 1
mkdir -p logs data/market_participation

export PYTHONPYCACHEPREFIX=/private/tmp/signalforge_pycache
exec ./venv/bin/python scripts/send_market_participation_telegram.py "$@"
