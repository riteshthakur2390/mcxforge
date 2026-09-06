# MCXForge Health Check & Operational Cron Schedule (Commodity Option Buying)

Use these entrypoints for automated health checks, Dhan token verification, and nightly EOD backups aligned with MCX Commodity market hours (09:00 to 23:30/23:55 IST) for the **Commodity Option Buying (BUY CALL / BUY PUT ONLY)** engine.

```cron
# 1. Post-Market Nightly Backup & Log Pruning (00:15 IST)
15 0 * * 2-6 cd /Users/vishranti/Downloads/projects/mcxforge && PYTHONPYCACHEPREFIX=/private/tmp/mcxforge_pycache ./venv/bin/python scripts/cron_telegram_notify.py --name "MCXForge Nightly Backup" -- /bin/sh -lc "./venv/bin/python scripts/manage_backups.py backup && ./venv/bin/python scripts/manage_backups.py retention 2 3" >> logs/backup_cron.log 2>&1

# 2. Pre-Market Dhan Token & Feed Verification (08:30 IST)
30 8 * * 1-5 cd /Users/vishranti/Downloads/projects/mcxforge && ./venv/bin/python scripts/live_health_check.py --probe-timeout 8 --telegram >> logs/early_live_health_cron.log 2>&1

# 3. Midday European Open Check (13:30 IST)
30 13 * * 1-5 cd /Users/vishranti/Downloads/projects/mcxforge && ./venv/bin/python scripts/live_health_check.py --probe-timeout 8 >> logs/midday_health_cron.log 2>&1

# 4. Pre-US COMEX Session Prime Check (16:45 IST)
45 16 * * 1-5 cd /Users/vishranti/Downloads/projects/mcxforge && ./venv/bin/python scripts/live_health_check.py --probe-timeout 8 --telegram-always >> logs/pre_evening_health_cron.log 2>&1

# 5. EOD Trade Reconciliation Audit (23:35 IST)
35 23 * * 1-5 cd /Users/vishranti/Downloads/projects/mcxforge && ./venv/bin/python scripts/cron_telegram_notify.py --name "EOD Reconciliation" -- /bin/sh -lc "cat state/live_paper_journal.json" >> logs/eod_reconcile_cron.log 2>&1
```

### Operational Schedule Notes
- **08:30 IST**: Validates Dhan credentials and feeds before the 09:00 Asian market open.
- **16:45 IST**: Verifies feed latency, websocket status, and data cache 15 minutes before the high-alpha **US COMEX Evening Session (17:00–23:30 IST)** begins.
- **23:35 IST**: Performs position reconciliation, checks that all intraday positions are closed flat, and creates an audit log of statutory charges.
