"""
scripts/manage_backups.py
A utility script to handle backup and restoration of critical data files.

This script helps prevent data loss from corruption by creating versioned backups
of the historical database and Parquet cache files. It also includes verification
and pruning of old backups.

Usage:
    - Create and verify a DB-only backup:
      python scripts/manage_backups.py db-backup

    - Create and verify a backup:
      python scripts/manage_backups.py backup

    - Restore from the latest backup:
      python scripts/manage_backups.py restore

    - List all available backups:
      python scripts/manage_backups.py status

    - Prune old backups, keeping the 5 most recent:
      python scripts/manage_backups.py prune

    - Prune old backups, keeping the N most recent:
      python scripts/manage_backups.py prune <N>

    - Apply tight disk retention:
      python scripts/manage_backups.py retention 2 3
"""
import os
import sys
import shutil
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

# Define project root and data directories
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
HISTORICAL_DIR = DATA_DIR / "historical"
CACHE_DIR = DATA_DIR / "cache"
BACKUP_ROOT_DIR = PROJECT_ROOT / "backups"
DB_BACKUP_DIR = BACKUP_ROOT_DIR / "db"
LOG_DIR = PROJECT_ROOT / "logs"

# Setup logging to file
LOG_DIR.mkdir(exist_ok=True)
log_file = LOG_DIR / "backup_manager.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("BackupManager")

# --- Files and Directories to be backed up ---
FILES_TO_BACKUP = [
    HISTORICAL_DIR / "market_history.sqlite3",
]
DIRS_TO_BACKUP = [
    CACHE_DIR,
]


def get_dir_size(dir_path: Path) -> int:
    """Calculates the total size of a directory in bytes."""
    total_size = 0
    for dirpath, _, filenames in os.walk(dir_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size


def format_size(size_bytes: int) -> str:
    """Formats size in bytes to a human-readable string."""
    if size_bytes > 1024 * 1024:
        return f"{size_bytes / (1024*1024):.2f} MB"
    if size_bytes > 1024:
        return f"{size_bytes / 1024:.2f} KB"
    return f"{size_bytes} bytes"


def verify_backup(backup_dir: Path) -> bool:
    """Verifies a backup by checking file existence and size."""
    logger.info(f"Verifying backup: {backup_dir.name}")
    all_verified = True

    for file_path in FILES_TO_BACKUP:
        if not file_path.exists():
            continue
        backup_file = backup_dir / file_path.name
        if not backup_file.exists():
            logger.error(f"  - ❌ FAILED: Missing file {backup_file.name}")
            all_verified = False
            continue
        if file_path.suffix == ".sqlite3":
            try:
                with sqlite3.connect(backup_file) as conn:
                    result = conn.execute("PRAGMA integrity_check").fetchone()
                if not result or result[0] != "ok":
                    logger.error(f"  - FAILED: SQLite integrity check failed for {file_path.name}: {result}")
                    all_verified = False
                else:
                    logger.info(f"  - OK: {file_path.name} integrity_check=ok (size: {backup_file.stat().st_size})")
            except sqlite3.DatabaseError as exc:
                logger.error(f"  - FAILED: SQLite backup unreadable for {file_path.name}: {exc}")
                all_verified = False
        elif file_path.stat().st_size != backup_file.stat().st_size:
            logger.error(f"  - FAILED: Size mismatch for {file_path.name}")
            all_verified = False
        else:
            logger.info(f"  - OK: {file_path.name} (size: {file_path.stat().st_size})")

    # Verify directories
    for dir_path in DIRS_TO_BACKUP:
        if not dir_path.exists():
            continue
        backup_subdir = backup_dir / dir_path.name
        if not backup_subdir.exists():
            logger.error(f"  - FAILED: Missing directory {backup_subdir.name}")
            all_verified = False
            continue
        source_size = get_dir_size(dir_path)
        backup_size = get_dir_size(backup_subdir)
        if source_size != backup_size:
            logger.error(f"  - FAILED: Size mismatch for directory {dir_path.name} (source: {source_size} vs backup: {backup_size})")
            all_verified = False
        else:
            logger.info(f"  - OK: {dir_path.name} (size: {source_size})")

    return all_verified


def backup_sqlite_db(source: Path, target: Path) -> None:
    """Create a transactionally consistent SQLite backup, including WAL state."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(source, timeout=60) as src:
        src.execute("PRAGMA busy_timeout=60000")
        with sqlite3.connect(target) as dst:
            src.backup(dst)


def verify_sqlite_file(path: Path) -> bool:
    try:
        with sqlite3.connect(path) as conn:
            result = conn.execute("PRAGMA integrity_check").fetchone()
        return bool(result and result[0] == "ok")
    except sqlite3.DatabaseError:
        return False


def do_db_backup() -> Path:
    """Create a DB-only backup using SQLite online backup."""
    source = HISTORICAL_DIR / "market_history.sqlite3"
    if not source.exists():
        logger.error(f"Historical DB not found: {source}")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    DB_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    target = DB_BACKUP_DIR / f"market_history_{timestamp}.sqlite3"
    logger.info(f"Creating DB backup: {target}")
    backup_sqlite_db(source, target)
    if not verify_sqlite_file(target):
        logger.error(f"DB backup failed integrity check: {target}")
        target.unlink(missing_ok=True)
        sys.exit(1)
    logger.info(f"DB backup verified: {target} ({format_size(target.stat().st_size)})")
    return target


def do_db_restore_latest() -> None:
    """Restore the latest DB-only backup after confirmation."""
    backups = sorted(DB_BACKUP_DIR.glob("market_history_*.sqlite3"), reverse=True)
    if not backups:
        logger.error(f"No DB backups found in {DB_BACKUP_DIR}")
        sys.exit(1)
    latest = backups[0]
    if not verify_sqlite_file(latest):
        logger.error(f"Latest DB backup failed integrity check: {latest}")
        sys.exit(1)
    confirm = input(f"Restore {latest.name} over current market_history.sqlite3? (y/n): ").lower()
    if confirm != "y":
        logger.info("DB restore cancelled.")
        return
    target = HISTORICAL_DIR / "market_history.sqlite3"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(latest, target)
    logger.info(f"DB restored from {latest}")


def do_backup():
    """Creates a new timestamped backup and verifies it."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_target_dir = BACKUP_ROOT_DIR / f"backup_{timestamp}"

    logger.info(f"--- STARTING BACKUP: {timestamp} ---")
    try:
        backup_target_dir.mkdir(parents=True, exist_ok=True)

        for file_path in FILES_TO_BACKUP:
            if file_path.exists():
                logger.info(f"  - Backing up file: {file_path}")
                backup_file = backup_target_dir / file_path.name
                if file_path.suffix == ".sqlite3":
                    backup_sqlite_db(file_path, backup_file)
                else:
                    shutil.copy2(file_path, backup_file)
            else:
                logger.warning(f"  - Skipping missing file: {file_path}")

        for dir_path in DIRS_TO_BACKUP:
            if dir_path.exists():
                target_dir = backup_target_dir / dir_path.name
                logger.info(f"  - Backing up directory: {dir_path} to {target_dir}")
                shutil.copytree(dir_path, target_dir)
            else:
                logger.warning(f"  - Skipping missing directory: {dir_path}")

        logger.info("✅ Backup creation complete.")
        logger.info(f"   Backup size: {format_size(get_dir_size(backup_target_dir))}")

        if verify_backup(backup_target_dir):
            logger.info("✅ Backup verified successfully.")
            logger.info(f"--- BACKUP FINISHED SUCCESSFULLY ---")
        else:
            logger.error("❌ Backup verification failed. Deleting incomplete backup.")
            shutil.rmtree(backup_target_dir)
            sys.exit(1)

    except Exception as e:
        logger.error(f"❌ Error during backup process: {e}")
        if backup_target_dir.exists():
            shutil.rmtree(backup_target_dir)
        sys.exit(1)


def do_restore():
    """Restores data from the most recent valid backup."""
    if not BACKUP_ROOT_DIR.exists() or not any(BACKUP_ROOT_DIR.iterdir()):
        logger.error("❌ No backups found. Cannot restore.")
        sys.exit(1)

    try:
        latest_backup = max(
            (d for d in BACKUP_ROOT_DIR.iterdir() if d.is_dir()),
            key=lambda d: d.name
        )
    except ValueError:
        logger.error("❌ No valid backup directories found.")
        sys.exit(1)

    logger.info(f"Found latest backup: {latest_backup.name}")
    print(f"\nThis backup will be restored. It contains:")
    for item in sorted(latest_backup.iterdir()):
        if item.is_dir():
            print(f"  - Directory: {item.name}")
        else:
            print(f"  - File:      {item.name}")

    confirm = input("\nThis will overwrite current data files. Are you sure? (y/n): ").lower()
    if confirm != 'y':
        logger.info("Restore operation cancelled.")
        return

    logger.info(f"--- STARTING RESTORE FROM {latest_backup.name} ---")
    try:
        for file_path in FILES_TO_BACKUP:
            backup_file = latest_backup / file_path.name
            if backup_file.exists():
                logger.info(f"  - Restoring file: {file_path.name}")
                file_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(backup_file, file_path)
            else:
                logger.warning(f"  - File not in backup, skipping: {file_path.name}")

        for dir_path in DIRS_TO_BACKUP:
            backup_dir = latest_backup / dir_path.name
            if backup_dir.exists():
                logger.info(f"  - Restoring directory: {dir_path.name}")
                if dir_path.exists():
                    shutil.rmtree(dir_path)
                shutil.copytree(backup_dir, dir_path)
            else:
                logger.warning(f"  - Directory not in backup, skipping: {dir_path.name}")

        logger.info("✅ Restore completed successfully.")
        logger.info(f"--- RESTORE FINISHED SUCCESSFULLY ---")

    except Exception as e:
        logger.error(f"❌ Error during restore process: {e}")
        sys.exit(1)


def do_status():
    """Lists all available backups and their details."""
    if not BACKUP_ROOT_DIR.exists() or not any(BACKUP_ROOT_DIR.iterdir()):
        print("No backups found.")
        return

    print("\nAvailable backups (newest first):")
    print("-" * 60)

    backups = sorted(
        (d for d in BACKUP_ROOT_DIR.iterdir() if d.is_dir() and d.name.startswith("backup_")),
        key=lambda d: d.name,
        reverse=True
    )

    if not backups:
        print("No valid backup directories found.")
        return

    for backup_path in backups:
        try:
            backup_date = datetime.strptime(backup_path.name, "backup_%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
            backup_size = format_size(get_dir_size(backup_path))
            print(f"  - {backup_path.name}  |  Date: {backup_date}  |  Size: {backup_size}")
        except ValueError:
            print(f"  - {backup_path.name} (unrecognized format)")

    print("-" * 60)


def _backup_timestamp(path: Path, prefix: str, suffix: str = "") -> datetime | None:
    name = path.name
    if suffix and name.endswith(suffix):
        name = name[: -len(suffix)]
    try:
        return datetime.strptime(name, f"{prefix}%Y%m%d_%H%M%S")
    except ValueError:
        return None


def _older_than(path: Path, *, prefix: str, max_age_days: int, suffix: str = "") -> bool:
    timestamp = _backup_timestamp(path, prefix=prefix, suffix=suffix)
    if timestamp is None:
        return False
    return timestamp < datetime.now() - timedelta(days=max_age_days)


def do_prune(keep: int = 5, max_age_days: int | None = None):
    """Deletes old full backups by count and optional age."""
    age_text = f", max age {max_age_days} day(s)" if max_age_days is not None else ""
    logger.info(f"--- STARTING PRUNE (keeping {keep} most recent{age_text}) ---")
    if not BACKUP_ROOT_DIR.exists():
        logger.warning("Backup directory not found. Nothing to prune.")
        return

    backups = sorted(
        (d for d in BACKUP_ROOT_DIR.iterdir() if d.is_dir() and d.name.startswith("backup_")),
        key=lambda d: d.name,
        reverse=True
    )

    to_delete_set = set(backups[keep:])
    if max_age_days is not None:
        to_delete_set.update(
            backup_path
            for backup_path in backups
            if _older_than(backup_path, prefix="backup_", max_age_days=max_age_days)
        )
    to_delete = sorted(to_delete_set, key=lambda d: d.name, reverse=True)

    if not to_delete:
        logger.info(f"Found {len(backups)} backups, within retention policy. No deletions.")
        return

    logger.info(f"Found {len(backups)} backups. Deleting {len(to_delete)} old backup(s):")
    for backup_path in to_delete:
        try:
            logger.info(f"  - Deleting {backup_path.name}...")
            shutil.rmtree(backup_path)
        except Exception as e:
            logger.error(f"    ❌ Error deleting {backup_path.name}: {e}")

    logger.info("✅ Pruning complete.")


def do_prune_db_backups(keep: int = 2, max_age_days: int | None = None) -> None:
    """Deletes old DB-only backup snapshots by count and optional age."""
    age_text = f", max age {max_age_days} day(s)" if max_age_days is not None else ""
    logger.info(f"--- STARTING DB PRUNE (keeping {keep} most recent{age_text}) ---")
    if not DB_BACKUP_DIR.exists():
        logger.info("DB backup directory not found. Nothing to prune.")
        return

    backups = sorted(
        DB_BACKUP_DIR.glob("market_history_*.sqlite3"),
        key=lambda path: path.name,
        reverse=True,
    )

    to_delete_set = set(backups[keep:])
    if max_age_days is not None:
        to_delete_set.update(
            backup_path
            for backup_path in backups
            if _older_than(
                backup_path,
                prefix="market_history_",
                suffix=".sqlite3",
                max_age_days=max_age_days,
            )
        )
    to_delete = sorted(to_delete_set, key=lambda path: path.name, reverse=True)

    if not to_delete:
        logger.info(f"Found {len(backups)} DB backups, within retention policy. No deletions.")
        return

    logger.info(f"Found {len(backups)} DB backups. Deleting {len(to_delete)} old snapshot(s):")
    for backup_path in to_delete:
        try:
            logger.info(f"  - Deleting {backup_path.name}...")
            backup_path.unlink(missing_ok=True)
            for suffix in ("-wal", "-shm"):
                sidecar = backup_path.with_name(f"{backup_path.name}{suffix}")
                if sidecar.exists():
                    logger.info(f"  - Deleting {sidecar.name}...")
                    sidecar.unlink(missing_ok=True)
        except Exception as e:
            logger.error(f"    ❌ Error deleting {backup_path.name}: {e}")

    logger.info("✅ DB pruning complete.")


def do_retention(keep: int = 2, max_age_days: int = 3) -> None:
    """Apply disk retention to full backups and DB-only snapshots."""
    do_prune(keep=keep, max_age_days=max_age_days)
    do_prune_db_backups(keep=keep, max_age_days=max_age_days)


def main():
    """Main function to parse command-line arguments."""
    BACKUP_ROOT_DIR.mkdir(exist_ok=True)

    if len(sys.argv) < 2:
        print("Usage: python scripts/manage_backups.py [db-backup|db-restore-latest|backup|restore|status|prune|prune-db|retention]")
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "backup":
        do_backup()
    elif command == "db-backup":
        do_db_backup()
    elif command == "db-restore-latest":
        do_db_restore_latest()
    elif command == "restore":
        do_restore()
    elif command == "status":
        do_status()
    elif command == "prune":
        keep_n = 5
        if len(sys.argv) > 2:
            try:
                keep_n = int(sys.argv[2])
                if keep_n < 1:
                    print("Error: Number of backups to keep must be at least 1.")
                    sys.exit(1)
            except ValueError:
                print(f"Error: Invalid number '{sys.argv[2]}'. Please provide an integer.")
                sys.exit(1)
        do_prune(keep=keep_n)
    elif command == "prune-db":
        keep_n = 2
        if len(sys.argv) > 2:
            try:
                keep_n = int(sys.argv[2])
                if keep_n < 1:
                    print("Error: Number of DB backups to keep must be at least 1.")
                    sys.exit(1)
            except ValueError:
                print(f"Error: Invalid number '{sys.argv[2]}'. Please provide an integer.")
                sys.exit(1)
        do_prune_db_backups(keep=keep_n)
    elif command == "retention":
        keep_n = 2
        max_age_days = 3
        if len(sys.argv) > 2:
            try:
                keep_n = int(sys.argv[2])
                if keep_n < 1:
                    print("Error: Number of backups to keep must be at least 1.")
                    sys.exit(1)
            except ValueError:
                print(f"Error: Invalid number '{sys.argv[2]}'. Please provide an integer.")
                sys.exit(1)
        if len(sys.argv) > 3:
            try:
                max_age_days = int(sys.argv[3])
                if max_age_days < 1:
                    print("Error: Max age days must be at least 1.")
                    sys.exit(1)
            except ValueError:
                print(f"Error: Invalid max age days '{sys.argv[3]}'. Please provide an integer.")
                sys.exit(1)
        do_retention(keep=keep_n, max_age_days=max_age_days)
    else:
        print(f"Unknown command: '{command}'")
        print("Usage: python scripts/manage_backups.py [db-backup|db-restore-latest|backup|restore|status|prune N|prune-db N|retention N MAX_AGE_DAYS]")
        sys.exit(1)


if __name__ == "__main__":
    main()
