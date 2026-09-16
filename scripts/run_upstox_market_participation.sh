#!/usr/bin/env bash
set -u

cd /Users/vishranti/Downloads/projects/mcxforge || exit 1
mkdir -p logs data/market_participation

export PYTHONPYCACHEPREFIX=/private/tmp/mcxforge_pycache
exec ./venv/bin/python scripts/fetch_upstox_market_participation.py "$@"
