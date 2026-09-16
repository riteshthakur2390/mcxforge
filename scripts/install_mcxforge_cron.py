#!/usr/bin/env python3
"""
Install / update MCXForge crontab block safely without overwriting other projects' cron jobs.

Usage:
    python scripts/install_mcxforge_cron.py
    python scripts/install_mcxforge_cron.py --dry-run
    python scripts/install_mcxforge_cron.py --uninstall
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

START_MARKER = "# >>> MCXFORGE MANAGED CRON >>>"
END_MARKER = "# <<< MCXFORGE MANAGED CRON <<<"

CRON_TEMPLATE = """# >>> MCXFORGE MANAGED CRON >>>
# Safe install: ./venv/bin/python scripts/install_mcxforge_cron.py
# Operating Hours: 09:00 - 23:30 IST (Mon - Fri)

# 1. 08:45 AM Mon-Fri: Force recreate / restart containers before morning session
45 8 * * 1-5 cd /Users/vishranti/Downloads/projects/mcxforge && PYTHONPYCACHEPREFIX=/private/tmp/mcxforge_pycache ./venv/bin/python scripts/cron_telegram_notify.py --name "MCXForge Docker restart" -- /bin/bash scripts/dev.sh fr >> logs/docker_restart_cron.log 2>&1

# 2. 08:50 AM Mon-Fri: End-to-End Live Readiness Audit (10m before 09:00 MCX open)
50 8 * * 1-5 cd /Users/vishranti/Downloads/projects/mcxforge && PYTHONPYCACHEPREFIX=/private/tmp/mcxforge_pycache ./venv/bin/python scripts/e2e_live_readiness_audit.py --telegram >> logs/e2e_readiness_audit_cron.log 2>&1

# 3. 09:15 AM Mon-Fri: Early Market Health Check (15m after MCX open)
15 9 * * 1-5 cd /Users/vishranti/Downloads/projects/mcxforge && PYTHONPYCACHEPREFIX=/private/tmp/mcxforge_pycache ./venv/bin/python scripts/live_health_check.py --probe-timeout 8 --telegram --telegram-always >> logs/early_live_health_cron.log 2>&1

# 4. 10:00 AM Mon-Fri: Morning Health Check (after 09:30 signal generation starts)
0 10 * * 1-5 cd /Users/vishranti/Downloads/projects/mcxforge && PYTHONPYCACHEPREFIX=/private/tmp/mcxforge_pycache ./venv/bin/python scripts/live_health_check.py --probe-timeout 8 --telegram --telegram-always >> logs/live_health_cron.log 2>&1

# 5. 14:00 PM Mon-Fri: Midday Session Health Check
0 14 * * 1-5 cd /Users/vishranti/Downloads/projects/mcxforge && PYTHONPYCACHEPREFIX=/private/tmp/mcxforge_pycache ./venv/bin/python scripts/live_health_check.py --probe-timeout 8 --telegram >> logs/midday_health_cron.log 2>&1

# 6. 17:05 PM Mon-Fri: Evening Session Open Health Check (US/COMEX open - peak liquidity)
5 17 * * 1-5 cd /Users/vishranti/Downloads/projects/mcxforge && PYTHONPYCACHEPREFIX=/private/tmp/mcxforge_pycache ./venv/bin/python scripts/live_health_check.py --probe-timeout 8 --telegram --telegram-always >> logs/evening_session_health_cron.log 2>&1

# 7. 20:30 PM Mon-Fri: Peak Evening Session Health Check
30 20 * * 1-5 cd /Users/vishranti/Downloads/projects/mcxforge && PYTHONPYCACHEPREFIX=/private/tmp/mcxforge_pycache ./venv/bin/python scripts/live_health_check.py --probe-timeout 8 --telegram >> logs/peak_evening_health_cron.log 2>&1

# 8. 23:15 PM Mon-Fri: Emergency EOD Square-off (MCX Intraday Cutoff)
15 23 * * 1-5 cd /Users/vishranti/Downloads/projects/mcxforge && PYTHONPYCACHEPREFIX=/private/tmp/mcxforge_pycache ./venv/bin/python scripts/emergency_eod_squareoff.py >> logs/emergency_eod_squareoff_cron.log 2>&1

# 9. 23:35 PM Mon-Fri: EOD Diagnostic Report (post MCX 23:30 close)
35 23 * * 1-5 cd /Users/vishranti/Downloads/projects/mcxforge && PYTHONPYCACHEPREFIX=/private/tmp/mcxforge_pycache ./venv/bin/python scripts/cron_telegram_notify.py --name "MCXForge EOD diagnostic" -- ./venv/bin/python scripts/eod_diagnostic.py --no-color >> logs/eod_cron.log 2>&1

# 10. 01:00 AM Daily: Nightly Database & State Backup with retention
0 1 * * * cd /Users/vishranti/Downloads/projects/mcxforge && PYTHONPYCACHEPREFIX=/private/tmp/mcxforge_pycache ./venv/bin/python scripts/cron_telegram_notify.py --name "MCXForge Nightly backup" -- /bin/sh -lc "./venv/bin/python scripts/manage_backups.py backup && ./venv/bin/python scripts/manage_backups.py retention 2 3" >> logs/backup_cron.log 2>&1
# <<< MCXFORGE MANAGED CRON <<<"""


def get_current_crontab() -> str:
    res = subprocess.run(["crontab", "-l"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        return ""
    return res.stdout


def remove_mcxforge_block(content: str) -> str:
    lines = content.splitlines()
    filtered = []
    in_block = False
    for line in lines:
        if START_MARKER in line:
            in_block = True
            continue
        if END_MARKER in line:
            in_block = False
            continue
        if not in_block:
            filtered.append(line)
    return "\n".join(filtered).strip()


def install(dry_run: bool = False, uninstall: bool = False) -> None:
    current = get_current_crontab()
    cleaned = remove_mcxforge_block(current)

    if uninstall:
        new_crontab = cleaned + "\n" if cleaned else ""
        print("ℹ️ Removing MCXForge cron block...")
    else:
        new_crontab = (cleaned + "\n\n" if cleaned else "") + CRON_TEMPLATE + "\n"
        print("ℹ️ Installing/Updating MCXForge cron block...")

    if dry_run:
        print("\n--- Proposed Crontab ---")
        print(new_crontab)
        print("--- End of Proposed Crontab ---")
        return

    # Ensure logs directory exists
    logs_dir = Path("/Users/vishranti/Downloads/projects/mcxforge/logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    proc = subprocess.run(["crontab", "-"], input=new_crontab, text=True, capture_output=True)
    if proc.returncode != 0:
        print(f"❌ Failed to update crontab: {proc.stderr}", file=sys.stderr)
        sys.exit(proc.returncode)
    print("✅ Crontab successfully updated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage MCXForge Crontab entries safely.")
    parser.add_argument("--dry-run", action="store_true", help="Print crontab without applying changes")
    parser.add_argument("--uninstall", action="store_true", help="Remove MCXForge cron block")
    args = parser.parse_args()
    install(dry_run=args.dry_run, uninstall=args.uninstall)
