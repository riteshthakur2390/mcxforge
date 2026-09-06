#!/usr/bin/env bash
set -u

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR" || exit 1
mkdir -p logs

export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/private/tmp/mcxforge_pycache}"
exec ./venv/bin/python scripts/dhan_renew_tokens.py \
  --telegram \
  --telegram-always \
  "$@"
