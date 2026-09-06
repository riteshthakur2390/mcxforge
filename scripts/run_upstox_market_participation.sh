#!/usr/bin/env bash
set -u

cd /Users/vishranti/Downloads/projects/signalforge || exit 1
mkdir -p logs data/market_participation

export PYTHONPYCACHEPREFIX=/private/tmp/signalforge_pycache
exec ./venv/bin/python scripts/fetch_upstox_market_participation.py "$@"
