"""BKP-1 — Command line for the SQLite backups: run, list, verify, restore.

The daily backup runs on its own inside the backend (``src/persistence/
backup_daemon.py``). This script is for the hands-on moments: a backup on
demand, a look at what the store holds, and — the one that matters — the
restore.

    # Back up now (snapshot → upload → prune to the retention window)
    python scripts/backup_sqlite.py run

    # What is in the store?
    python scripts/backup_sqlite.py list

    # Is the store healthy? (same verdict the daily verifier uses; exits 1 if not)
    python scripts/backup_sqlite.py verify

    # Restore the most recent backup into an empty directory, then check it
    python scripts/backup_sqlite.py restore --target /tmp/restore-test

    # Restore one specific backup
    python scripts/backup_sqlite.py restore --key mia-backup-20260913T031700Z.tar.gz \
        --target /tmp/restore-test

    # Snapshot to a local file without touching any remote store
    python scripts/backup_sqlite.py snapshot --out ./backups

Credentials are read from the environment only (see docs/ops/sauvegardes-sqlite.md).
Exit code is 0 on success and 1 on any failure, so cron and CI can rely on it.

Full procedure: docs/ops/sauvegardes-sqlite.md
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow "python scripts/backup_sqlite.py" from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.persistence.backup_service import (  # noqa: E402
    data_dir,
    max_age_hours,
    restore_latest,
    retention_days,
    run_backup,
    size_tolerance,
    verify_backups,
)
from src.persistence.backup_storage import destination_from_env  # noqa: E402
from src.persistence.sqlite_backup import (  # noqa: E402
    BackupError,
    create_archive,
    discover_databases,
    verify_archive,
)


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
    )


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return f"{n:,.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} GB"


# --------------------------------------------------------------------------- #

def cmd_run(args: argparse.Namespace) -> int:
    result = run_backup(prune=not args.no_prune)
    if args.json:
        print(json.dumps(result.as_dict(), indent=2, default=str))
    elif result.ok:
        print(f"OK  {result.archive}")
        print(f"    {len(result.databases)} databases: {', '.join(result.databases)}")
        print(
            f"    {_human(result.source_bytes)} snapshotted → "
            f"{_human(result.size_bytes)} stored in {result.duration_s:.1f}s"
        )
        if result.pruned:
            print(f"    pruned {len(result.pruned)}: {', '.join(result.pruned)}")
        verify = result.verify or {}
        if not verify.get("healthy", True):
            print("    WARNING — the store is not healthy:")
            for p in verify.get("problems", []):
                print(f"      - {p}")
    else:
        print(f"FAILED  {result.error}", file=sys.stderr)
    return 0 if result.ok else 1


def cmd_snapshot(args: argparse.Namespace) -> int:
    src = Path(args.data_dir) if args.data_dir else data_dir()
    archive = create_archive(src, args.out, now=datetime.now(timezone.utc))
    verify_archive(archive.path)
    print(f"OK  {archive.path}")
    print(
        f"    {len(archive.files)} databases, {_human(archive.source_bytes)} → "
        f"{_human(archive.size_bytes)} (ratio {archive.compression_ratio:.3f}) "
        f"in {archive.duration_s:.1f}s"
    )
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    dest = destination_from_env()
    objects = dest.list_backups()
    if args.json:
        print(
            json.dumps(
                [
                    {
                        "key": o.key,
                        "size": o.size,
                        "stamp": o.stamp.isoformat() if o.stamp else None,
                    }
                    for o in objects
                ],
                indent=2,
            )
        )
        return 0
    print(f"{dest.describe()} — {len(objects)} backup(s), retention {retention_days()} days")
    for o in sorted(objects, key=lambda x: x.key):
        stamp = o.stamp.strftime("%Y-%m-%d %H:%M UTC") if o.stamp else "unparsable name"
        print(f"  {stamp}  {_human(o.size):>10}  {o.key}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    dest = destination_from_env()
    report = verify_backups(
        dest, max_age_h=max_age_hours(), tolerance=size_tolerance()
    )
    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
    else:
        print(f"{report.destination} — {report.count} backup(s)")
        if report.latest_key:
            print(
                f"  latest: {report.latest_key} — {_human(report.latest_size or 0)}, "
                f"{report.latest_age_hours}h old"
            )
        if report.healthy:
            print("  HEALTHY")
        else:
            print("  NOT HEALTHY:")
            for p in report.problems:
                print(f"    - {p}")
    return 0 if report.healthy else 1


def cmd_restore(args: argparse.Namespace) -> int:
    report = restore_latest(args.target, key=args.key, force=args.force)
    if args.json:
        print(json.dumps(report, indent=2, default=str))
        return 0
    print(f"OK  restored {report['source_key']} → {report['target']}")
    for name, info in report["files"].items():
        print(f"    {name:<30} {_human(info['size']):>10}  integrity_check={info['integrity_check']}")
    return 0


def cmd_databases(args: argparse.Namespace) -> int:
    src = Path(args.data_dir) if args.data_dir else data_dir()
    dbs = discover_databases(src)
    total = 0
    print(f"{src} — {len(dbs)} database(s) would be backed up")
    for p in dbs:
        size = p.stat().st_size
        total += size
        print(f"  {p.name:<30} {_human(size):>10}")
    print(f"  {'TOTAL':<30} {_human(total):>10}")
    return 0


# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="backup_sqlite",
        description="Consistent SQLite backups to S3/R2, and their restore (BKP-1).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Full procedure: docs/ops/sauvegardes-sqlite.md",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="snapshot, upload, prune to the retention window")
    r.add_argument("--no-prune", action="store_true", help="keep every old backup")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("snapshot", help="local archive only, no upload")
    s.add_argument("--out", default="./backups", help="where to write the archive")
    s.add_argument("--data-dir", default=None, help="override DATA_DIR")
    s.set_defaults(func=cmd_snapshot)

    li = sub.add_parser("list", help="what the store holds")
    li.set_defaults(func=cmd_list)

    v = sub.add_parser("verify", help="judge the store; exit 1 when unhealthy")
    v.set_defaults(func=cmd_verify)

    re_ = sub.add_parser("restore", help="download and restore a backup")
    re_.add_argument("--target", required=True, help="directory to restore into")
    re_.add_argument("--key", default=None, help="a specific backup (default: the latest)")
    re_.add_argument("--force", action="store_true", help="allow a non-empty target")
    re_.set_defaults(func=cmd_restore)

    d = sub.add_parser("databases", help="list what would be backed up")
    d.add_argument("--data-dir", default=None, help="override DATA_DIR")
    d.set_defaults(func=cmd_databases)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    try:
        return args.func(args)
    except BackupError as exc:
        print(f"FAILED  {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:  # pragma: no cover
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    # .env is convenient locally; on Render the variables are already in the env.
    try:
        from dotenv import load_dotenv

        load_dotenv(override=False)
    except ImportError:  # pragma: no cover
        pass
    sys.exit(main())
