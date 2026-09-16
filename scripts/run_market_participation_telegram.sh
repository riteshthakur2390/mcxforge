#!/usr/bin/env bash
set -u

cd /Users/vishranti/Downloads/projects/mcxforge || exit 1
mkdir -p logs data/market_participation

export PYTHONPYCACHEPREFIX=/private/tmp/mcxforge_pycache
exec ./venv/bin/python scripts/send_market_participation_telegram.py "$@"
