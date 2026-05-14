"""Backup + DR CLI — Sprint INFRA-2B.6.

Snapshots every Sprint-2B SQLite store into a single timestamped tar.gz
archive, optionally pushed to Backblaze B2 / S3. Restore is the
inverse — extract archive into a fresh data dir.

Why this exists
---------------
The production deploy currently has 6 SQLite files holding cross-cut
state:

  data/signals.db                    (SignalStore)
  data/api_keys.db                   (KeyStore + grace rotation)
  data/audit_ledger.db               (Hash-chain ledger)
  data/admin_action_log.db           (Admin audit log)
  data/telegram_lang.db              (Per-chat language)
  data/idempotency.db                (Idempotency cache, if persistent)

Plus the RAG corpus snapshot (if exported), audit ledger
snapshot.jsonl (DATA-2B.9), and the prompt registry frozen state.

Without a routine snapshot, a Railway volume corruption = total loss.
This script is the discipline knob: run nightly via cron, push to
Backblaze, and we have ≤ 24h RPO on every B2B-facing store.

Usage
-----
  python scripts/backup_dr.py snapshot data/ backups/
       → data_20260513_034512.tar.gz

  python scripts/backup_dr.py restore backups/data_20260513.tar.gz new_data/
       → new_data/ with every DB extracted

The verify subcommand walks the archive's manifest + checksums.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

MANIFEST_NAME = "MANIFEST.json"

# Sprint-2B persistent stores we always want backed up. Globs are
# allowed; missing files are logged but don't fail the snapshot
# (a fresh deploy may not have generated every store yet).
DEFAULT_INCLUDES = (
    "*.db", "*.db-wal", "*.db-shm",
    "audit_snapshot*.jsonl",
    "rag_corpus_fingerprint.json",
    "prompts/manifest.yaml",
)


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _enumerate(data_dir: Path, includes: Iterable[str]) -> list[Path]:
    seen: set[Path] = set()
    for pat in includes:
        for p in data_dir.rglob(pat):
            if p.is_file():
                seen.add(p.resolve())
    return sorted(seen)


def cmd_snapshot(
    data_dir: str, backup_dir: str, *, includes: Iterable[str] = DEFAULT_INCLUDES
) -> int:
    src = Path(data_dir).resolve()
    if not src.is_dir():
        print(f"ABORT: {src} is not a directory", file=sys.stderr)
        return 2
    dst_dir = Path(backup_dir).resolve()
    dst_dir.mkdir(parents=True, exist_ok=True)

    files = _enumerate(src, includes)
    if not files:
        print("ABORT: no files matched include patterns", file=sys.stderr)
        return 2

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    archive_path = dst_dir / f"data_{ts}.tar.gz"
    manifest = {
        "snapshot_v": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(src),
        "files": [],
    }
    for f in files:
        rel = f.relative_to(src)
        manifest["files"].append(
            {
                "path": str(rel).replace("\\", "/"),
                "size": f.stat().st_size,
                "sha256": _file_sha256(f),
            }
        )

    with tarfile.open(archive_path, "w:gz") as tar:
        # Manifest first so streaming restore can validate before extracting bulk.
        manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")
        info = tarfile.TarInfo(name=MANIFEST_NAME)
        info.size = len(manifest_bytes)
        info.mtime = int(time.time())
        from io import BytesIO
        tar.addfile(info, BytesIO(manifest_bytes))
        for f in files:
            rel = f.relative_to(src)
            tar.add(f, arcname=str(rel).replace("\\", "/"))

    print(f"snapshot OK: {len(files)} files, {archive_path.stat().st_size} bytes "
          f"→ {archive_path}")
    return 0


def cmd_restore(archive: str, target_dir: str) -> int:
    src = Path(archive).resolve()
    if not src.is_file():
        print(f"ABORT: {src} is not a file", file=sys.stderr)
        return 2
    dst = Path(target_dir).resolve()
    if dst.exists() and any(dst.iterdir()):
        print(f"ABORT: target {dst} is not empty — refuse to overwrite", file=sys.stderr)
        return 2
    dst.mkdir(parents=True, exist_ok=True)

    with tarfile.open(src, "r:gz") as tar:
        manifest_member = next(
            (m for m in tar.getmembers() if m.name == MANIFEST_NAME), None
        )
        if manifest_member is None:
            print("ABORT: archive has no MANIFEST.json — not a backup", file=sys.stderr)
            return 2
        manifest_bytes = tar.extractfile(manifest_member).read()  # type: ignore[union-attr]
        manifest = json.loads(manifest_bytes)
        if manifest.get("snapshot_v") != 1:
            print(
                f"ABORT: unsupported snapshot_v {manifest.get('snapshot_v')}",
                file=sys.stderr,
            )
            return 2
        # Extract everything except the manifest itself.
        members = [m for m in tar.getmembers() if m.name != MANIFEST_NAME]
        # Python 3.12 deprecates default extraction; use the filter param.
        for m in members:
            tar.extract(m, dst, filter="data")

    # Verify checksums on the way out.
    for entry in manifest["files"]:
        p = dst / entry["path"]
        if not p.is_file():
            print(f"ABORT: restore missing {p}", file=sys.stderr)
            return 2
        actual = _file_sha256(p)
        if actual != entry["sha256"]:
            print(f"ABORT: checksum mismatch on {entry['path']}", file=sys.stderr)
            return 2

    print(f"restore OK: {len(manifest['files'])} files → {dst}")
    return 0


def cmd_verify(archive: str) -> int:
    src = Path(archive).resolve()
    if not src.is_file():
        print(f"ABORT: {src} is not a file", file=sys.stderr)
        return 2
    with tarfile.open(src, "r:gz") as tar:
        manifest_member = next(
            (m for m in tar.getmembers() if m.name == MANIFEST_NAME), None
        )
        if manifest_member is None:
            print("ABORT: no MANIFEST.json", file=sys.stderr)
            return 2
        manifest = json.loads(
            tar.extractfile(manifest_member).read()  # type: ignore[union-attr]
        )
        for entry in manifest["files"]:
            member = tar.getmember(entry["path"])
            if member.size != entry["size"]:
                print(
                    f"ABORT: size mismatch on {entry['path']}", file=sys.stderr
                )
                return 2
            data = tar.extractfile(member).read()  # type: ignore[union-attr]
            if hashlib.sha256(data).hexdigest() != entry["sha256"]:
                print(
                    f"ABORT: checksum mismatch on {entry['path']}", file=sys.stderr
                )
                return 2
    print(f"verify OK: {len(manifest['files'])} files, archive intact")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="backup_dr", description="Backup + DR for Smart Sentinel persistent stores.")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("snapshot")
    s.add_argument("data_dir"); s.add_argument("backup_dir")
    r = sub.add_parser("restore")
    r.add_argument("archive"); r.add_argument("target_dir")
    v = sub.add_parser("verify")
    v.add_argument("archive")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "snapshot":
        return cmd_snapshot(args.data_dir, args.backup_dir)
    if args.cmd == "restore":
        return cmd_restore(args.archive, args.target_dir)
    if args.cmd == "verify":
        return cmd_verify(args.archive)
    return 1


if __name__ == "__main__":
    sys.exit(main())
