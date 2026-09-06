#!/usr/bin/env python3
"""
scripts/nightly_backup.py — Automated Nightly Backup
=====================================================
Every night at 22:00 IST: zip critical data and upload to Google Drive or S3.

WHAT GETS BACKED UP:
  - journal/   (trade ledger, equity curve, snapshots — irreplaceable)
  - logs/      (all diagnostic logs)
  - ml/saved_models/ (trained models)
  - config/settings/ (current configuration)

WHY THIS IS CRITICAL:
  - Trade history is the ML training data — losing it = losing future edge
  - Docker volumes can be wiped on restart
  - Single disk failure = months of work lost

DESTINATIONS:
  1. Local archive: /backups/signalforge_YYYYMMDD.zip (always)
  2. Google Drive:  if GDRIVE_FOLDER_ID set in .env
  3. AWS S3:        if S3_BUCKET set in .env

SETUP:
  # Google Drive (using rclone — easiest)
  pip install rclone
  rclone config  (follow prompts to link Google Drive)
  Add to .env: GDRIVE_FOLDER_ID=your_folder_id

  # Cron (22:00 IST daily)
  0 22 * * * cd /opt/signalforge && python scripts/nightly_backup.py
"""

import os
import shutil
import sys
import zipfile
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytz
IST = pytz.timezone("Asia/Kolkata")

try:
    from config.settings import (
        JOURNAL_DIR, LOGS_DIR, ML_MODELS_DIR,
        TELEGRAM_ENABLED,
    )
except ImportError:
    JOURNAL_DIR   = "journal"
    LOGS_DIR      = "logs"
    ML_MODELS_DIR = "ml/saved_models"
    TELEGRAM_ENABLED = False

BACKUP_DIR   = Path("backups")
GDRIVE_ID    = os.getenv("GDRIVE_FOLDER_ID", "")
S3_BUCKET    = os.getenv("S3_BUCKET", "")
KEEP_LOCAL_N = 7   # keep last 7 local backups


def create_zip(backup_path: Path) -> Path:
    """Create zip archive of critical data."""
    today    = date.today().strftime("%Y%m%d")
    zip_name = f"signalforge_{today}.zip"
    zip_path = backup_path / zip_name

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    include_dirs = [
        Path(JOURNAL_DIR),
        Path(LOGS_DIR),
        Path(ML_MODELS_DIR).parent,
        Path("config"),
    ]

    total_files = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for src_dir in include_dirs:
            if not src_dir.exists():
                continue
            for file_path in src_dir.rglob("*"):
                if file_path.is_file() and "backups" not in str(file_path):
                    arcname = str(file_path)
                    zf.write(file_path, arcname)
                    total_files += 1

    size_mb = zip_path.stat().st_size / 1024 / 1024
    print(f"  ✅ Zip created: {zip_path} ({total_files} files, {size_mb:.1f} MB)")
    return zip_path


def upload_gdrive(zip_path: Path) -> bool:
    """Upload to Google Drive using rclone."""
    if not GDRIVE_ID:
        return False
    try:
        import subprocess
        result = subprocess.run(
            ["rclone", "copy", str(zip_path), f"gdrive:{GDRIVE_ID}"],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0:
            print(f"  ✅ Google Drive: uploaded {zip_path.name}")
            return True
        else:
            print(f"  ❌ Google Drive failed: {result.stderr[:100]}")
    except Exception as e:
        print(f"  ❌ Google Drive error: {e}")
    return False


def upload_s3(zip_path: Path) -> bool:
    """Upload to AWS S3."""
    if not S3_BUCKET:
        return False
    try:
        import boto3
        s3 = boto3.client("s3")
        s3.upload_file(str(zip_path), S3_BUCKET, f"signalforge/{zip_path.name}")
        print(f"  ✅ S3: uploaded to s3://{S3_BUCKET}/signalforge/{zip_path.name}")
        return True
    except Exception as e:
        print(f"  ❌ S3 error: {e}")
    return False


def cleanup_old_backups(backup_dir: Path, keep_n: int = 7) -> None:
    """Remove old local backups keeping only last N."""
    zips = sorted(backup_dir.glob("signalforge_*.zip"),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    for old in zips[keep_n:]:
        old.unlink()
        print(f"  🗑  Removed old backup: {old.name}")


def main() -> None:
    now = datetime.now(IST).strftime("%Y-%m-%d %H:%M IST")
    print(f"\n{'='*50}")
    print(f"  Nightly Backup | {now}")
    print(f"{'='*50}\n")

    # Step 1: Create zip
    zip_path = create_zip(BACKUP_DIR)

    # Step 2: Upload to cloud
    gdrive_ok = upload_gdrive(zip_path)
    s3_ok     = upload_s3(zip_path)

    if not gdrive_ok and not S3_BUCKET:
        print("  ℹ️  No cloud destination configured — local backup only")
        print("     Set GDRIVE_FOLDER_ID or S3_BUCKET in .env for cloud backup")

    # Step 3: Cleanup old
    cleanup_old_backups(BACKUP_DIR, KEEP_LOCAL_N)

    # Step 4: Telegram
    if TELEGRAM_ENABLED:
        try:
            import asyncio
            from utils.telegram_notifier import get_notifier
            size_mb = zip_path.stat().st_size / 1024 / 1024
            msg = (
                f"💾 *Nightly Backup Complete*\n"
                f"Size: {size_mb:.1f} MB\n"
                f"Cloud: {'✅' if gdrive_ok or s3_ok else '❌ local only'}\n"
                f"{now}"
            )
            asyncio.run(get_notifier().send_text(msg, target="LIVE"))
        except Exception:
            pass

    print(f"\n  Backup complete ✅\n")


if __name__ == "__main__":
    main()
