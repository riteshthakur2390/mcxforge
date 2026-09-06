#!/usr/bin/env bash
set -u

cd /Users/vishranti/Downloads/projects/signalforge || exit 1
mkdir -p logs

export PYTHONPYCACHEPREFIX=/private/tmp/signalforge_pycache
exec ./venv/bin/python scripts/midday_expiry_health_check.py \
  --probe-timeout 8 \
  --telegram \
  --telegram-always
