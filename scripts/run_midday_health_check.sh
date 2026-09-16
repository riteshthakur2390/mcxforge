#!/usr/bin/env bash
set -u

cd /Users/vishranti/Downloads/projects/mcxforge || exit 1
mkdir -p logs

export PYTHONPYCACHEPREFIX=/private/tmp/mcxforge_pycache
exec ./venv/bin/python scripts/live_health_check.py \
  --probe-timeout 8 \
  --telegram \
  --telegram-always \
  "$@"
