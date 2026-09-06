#!/usr/bin/env bash
set -u

cd /Users/vishranti/Downloads/projects/signalforge || exit 1
mkdir -p logs

export PYTHONPYCACHEPREFIX=/private/tmp/signalforge_pycache
exec ./venv/bin/python scripts/dhan_renew_tokens.py \
  --env /Users/vishranti/Downloads/projects/swingforge/.env \
  --no-default-sync-env \
  --telegram \
  --telegram-always \
  "$@"
