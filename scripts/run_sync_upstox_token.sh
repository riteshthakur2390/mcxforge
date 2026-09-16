#!/usr/bin/env bash
set -u

cd /Users/vishranti/Downloads/projects/mcxforge || exit 1
mkdir -p logs

export PYTHONPYCACHEPREFIX=/private/tmp/mcxforge_pycache
exec ./venv/bin/python scripts/sync_upstox_token.py \
  --notify \
  "$@"
