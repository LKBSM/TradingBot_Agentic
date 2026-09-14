"""BKP-1 — Consistent snapshots of the SQLite stores, and their restore.

Why not a file copy
-------------------
Every store in ``DATA_DIR`` runs in **WAL** mode (``src/persistence/sqlite_pragmas``).
In WAL, the freshly written pages live in ``<name>.db-wal``, not in ``<name>.db``.
Copying the three files (``.db``, ``-wal``, ``-shm``) reads them at three different
instants, so the copy can carry a *torn* database — and a SHA-256 manifest taken
over those bytes will happily declare that torn copy "intact". That is a backup
that reports success and restores garbage, which is worse than no backup at all.

This module uses SQLite's **online backup API** (``sqlite3.Connection.backup``)
instead. It copies page by page under the source's own locking and *restarts the
copy* when a concurrent writer changes a page already copied, so the destination
is always a transactionally consistent database — while the backend keeps writing.

Shape of an archive
-------------------
One ``.tar.gz`` per run::

    mia-backup-20260913T032000Z.tar.gz
      MANIFEST.json          # written FIRST so a streaming reader validates early
      candles.db             # consistent snapshot, integrity_check already passed
      market_readings.db
      accounts.db
      ...

The manifest records, per file: relative path, byte size, SHA-256, the
``PRAGMA integrity_check`` verdict taken on the *snapshot* (not on the live
database), and the source path it came from.

Nothing here touches the network — see ``backup_storage`` for that, and
``backup_service`` for the orchestration. Keeping the three apart is what lets
the whole snapshot/verify/restore cycle be tested without any credentials.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import shutil
import sqlite3
import tarfile
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Iterable, Optional, Sequence

logger = logging.getLogger(__name__)

MANIFEST_NAME = "MANIFEST.json"
MANIFEST_VERSION = 2
ARCHIVE_PREFIX = "mia-backup-"
ARCHIVE_SUFFIX = ".tar.gz"

#: Sidecars are deliberately EXCLUDED: the online backup API folds the WAL into
#: the snapshot, so shipping ``-wal``/``-shm`` would only add torn bytes.
_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")

#: Scratch databases a dev run leaves behind in ``data/``. Never worth shipping,
#: and their presence in an archive would confuse a restore.
_EXCLUDED_STEMS = ("_tmp_", "test_")


class BackupError(RuntimeError):
    """Raised when a snapshot, archive or restore cannot be trusted."""


# --------------------------------------------------------------------------- #
# Naming
# --------------------------------------------------------------------------- #

def utc_stamp(now: Optional[datetime] = None) -> str:
    """``20260913T032000Z`` — sorts lexicographically in chronological order."""
    return (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")


def archive_name(now: Optional[datetime] = None) -> str:
    return f"{ARCHIVE_PREFIX}{utc_stamp(now)}{ARCHIVE_SUFFIX}"


def parse_archive_stamp(name: str) -> Optional[datetime]:
    """Recover the UTC instant from an archive key. None when it isn't ours.

    Accepts a bare name or a full object key with a prefix
    (``mia/prod/mia-backup-20260913T032000Z.tar.gz``).
    """
    base = name.rsplit("/", 1)[-1]
    if not base.startswith(ARCHIVE_PREFIX) or not base.endswith(ARCHIVE_SUFFIX):
        return None
    stamp = base[len(ARCHIVE_PREFIX) : -len(ARCHIVE_SUFFIX)]
    try:
        return datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #

def discover_databases(data_dir: str | Path) -> list[Path]:
    """Every ``*.db`` worth backing up in ``data_dir``, sorted.

    Discovery rather than a hard-coded list: BKP-1 chose to ship the whole
    directory precisely so that adding a store later cannot silently fall out of
    the backup. Sidecars and dev scratch files are the only exclusions.
    """
    root = Path(data_dir)
    if not root.is_dir():
        raise BackupError(f"data dir not found: {root}")
    out: list[Path] = []
    for p in sorted(root.glob("*.db")):
        if not p.is_file():
            continue
        if p.name.endswith(_SIDECAR_SUFFIXES):
            continue
        if any(p.stem.startswith(s) for s in _EXCLUDED_STEMS):
            logger.debug("backup: skipping scratch database %s", p.name)
            continue
        out.append(p)
    return out


# --------------------------------------------------------------------------- #
# Snapshot
# --------------------------------------------------------------------------- #

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def integrity_check(db_path: str | Path) -> str:
    """``PRAGMA integrity_check`` verdict. ``"ok"`` means the file is sound."""
    try:
        conn = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True)
        try:
            row = conn.execute("PRAGMA integrity_check").fetchone()
            return str(row[0]) if row else "no result"
        finally:
            conn.close()
    except sqlite3.Error as exc:
        # Not a verdict of "ok", which is all any caller checks — and the reason
        # travels with it.
        return f"unreadable: {type(exc).__name__}: {exc}"


def snapshot_database(src: str | Path, dst: str | Path) -> Path:
    """Copy a live SQLite database to ``dst`` with the online backup API.

    Safe against a concurrent writer: SQLite restarts the copy if a page already
    copied is modified. The destination is then checked with
    ``PRAGMA integrity_check`` — a snapshot that does not pass is a failure, not
    a warning, because shipping it would create the illusion of a backup.
    """
    src_path, dst_path = Path(src), Path(dst)
    if not src_path.is_file():
        raise BackupError(f"source database not found: {src_path}")
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists():
        dst_path.unlink()

    # Every sqlite failure becomes a BackupError: callers must have one thing to
    # catch, and "the backup blew up with a DatabaseError" is not a distinction
    # anyone acts on differently.
    try:
        source = sqlite3.connect(f"file:{src_path.as_posix()}?mode=ro", uri=True, timeout=30.0)
        try:
            target = sqlite3.connect(str(dst_path))
            try:
                # pages=-1 → copy in one step; SQLite still yields to writers and
                # restarts as needed. sleep=0.25s spaces out retries under load.
                source.backup(target, pages=-1, sleep=0.25)
            finally:
                target.close()
        finally:
            source.close()
    except sqlite3.Error as exc:
        raise BackupError(f"snapshot of {src_path.name} failed: {type(exc).__name__}: {exc}") from exc

    verdict = integrity_check(dst_path)
    if verdict != "ok":
        raise BackupError(f"snapshot of {src_path.name} failed integrity_check: {verdict}")
    return dst_path


# --------------------------------------------------------------------------- #
# Archive
# --------------------------------------------------------------------------- #

@dataclass
class ArchiveResult:
    """What a run produced, for logging, the manifest and the state file."""

    path: Path
    name: str
    size_bytes: int
    source_bytes: int
    files: list[dict] = field(default_factory=list)
    duration_s: float = 0.0

    @property
    def compression_ratio(self) -> float:
        return (self.size_bytes / self.source_bytes) if self.source_bytes else 0.0


def create_archive(
    data_dir: str | Path,
    out_dir: str | Path,
    *,
    databases: Optional[Sequence[str | Path]] = None,
    now: Optional[datetime] = None,
    staging_dir: Optional[str | Path] = None,
) -> ArchiveResult:
    """Snapshot every database and pack them into one timestamped ``.tar.gz``.

    Peak scratch usage is roughly ``sum(db sizes) + archive size`` — the archive
    is streamed as the snapshots are added, so the snapshots are never gzipped
    twice. ``staging_dir`` defaults to a temporary directory next to ``out_dir``;
    point it elsewhere (``BACKUP_STAGING_DIR``) when the data disk is tight.
    """
    started = time.time()
    data_path = Path(data_dir)
    dbs = [Path(p) for p in databases] if databases is not None else discover_databases(data_path)
    if not dbs:
        raise BackupError(f"no database found in {data_path} — refusing an empty backup")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    name = archive_name(now)
    archive_path = out_path / name

    stage_parent = Path(staging_dir) if staging_dir else out_path
    stage_parent.mkdir(parents=True, exist_ok=True)

    entries: list[dict] = []
    source_bytes = 0
    with tempfile.TemporaryDirectory(prefix="mia-bkp-", dir=str(stage_parent)) as stage:
        stage_dir = Path(stage)
        snapshots: list[tuple[Path, Path]] = []
        for db in dbs:
            snap = snapshot_database(db, stage_dir / db.name)
            size = snap.stat().st_size
            source_bytes += size
            entries.append(
                {
                    "path": db.name,
                    "source": str(db),
                    "size": size,
                    "sha256": _sha256(snap),
                    "integrity_check": "ok",  # snapshot_database raises otherwise
                }
            )
            snapshots.append((db, snap))
            logger.info("backup: snapshot %s (%d bytes) ok", db.name, size)

        manifest = {
            "manifest_v": MANIFEST_VERSION,
            "created_utc": (now or datetime.now(timezone.utc)).isoformat(),
            "archive": name,
            "data_dir": str(data_path),
            "method": "sqlite3.Connection.backup (online backup API)",
            "files": entries,
        }
        manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")

        with tarfile.open(archive_path, "w:gz", compresslevel=6) as tar:
            info = tarfile.TarInfo(name=MANIFEST_NAME)
            info.size = len(manifest_bytes)
            info.mtime = int(time.time())
            tar.addfile(info, BytesIO(manifest_bytes))
            for _, snap in snapshots:
                tar.add(snap, arcname=snap.name)

    result = ArchiveResult(
        path=archive_path,
        name=name,
        size_bytes=archive_path.stat().st_size,
        source_bytes=source_bytes,
        files=entries,
        duration_s=round(time.time() - started, 2),
    )
    logger.info(
        "backup: archive %s — %d databases, %d bytes source, %d bytes compressed "
        "(ratio %.3f) in %.1fs",
        result.name,
        len(entries),
        result.source_bytes,
        result.size_bytes,
        result.compression_ratio,
        result.duration_s,
    )
    return result


# --------------------------------------------------------------------------- #
# Verify / restore
# --------------------------------------------------------------------------- #

def read_manifest(archive: str | Path) -> dict:
    """The manifest of an archive, without extracting anything else."""
    path = Path(archive)
    if not path.is_file():
        raise BackupError(f"archive not found: {path}")
    with tarfile.open(path, "r:gz") as tar:
        member = next((m for m in tar.getmembers() if m.name == MANIFEST_NAME), None)
        if member is None:
            raise BackupError(f"{path.name} carries no {MANIFEST_NAME} — not one of our backups")
        fh = tar.extractfile(member)
        if fh is None:
            raise BackupError(f"{path.name}: unreadable manifest")
        manifest = json.loads(fh.read())
    if manifest.get("manifest_v") not in (1, MANIFEST_VERSION):
        raise BackupError(f"unsupported manifest_v {manifest.get('manifest_v')}")
    return manifest


def verify_archive(archive: str | Path) -> dict:
    """Check every member against the manifest, without writing to disk.

    Sizes and SHA-256 only: proving the *bytes* survived the trip. The stronger
    proof — that the bytes are a working database — is what ``restore_archive``
    does with ``PRAGMA integrity_check``.
    """
    path = Path(archive)
    manifest = read_manifest(path)
    with tarfile.open(path, "r:gz") as tar:
        for entry in manifest["files"]:
            try:
                member = tar.getmember(entry["path"])
            except KeyError as exc:
                raise BackupError(f"{path.name}: manifest lists {entry['path']}, archive has not") from exc
            if member.size != entry["size"]:
                raise BackupError(
                    f"{path.name}: size mismatch on {entry['path']} "
                    f"({member.size} vs {entry['size']} expected)"
                )
            fh = tar.extractfile(member)
            if fh is None:
                raise BackupError(f"{path.name}: unreadable member {entry['path']}")
            h = hashlib.sha256()
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
            if h.hexdigest() != entry["sha256"]:
                raise BackupError(f"{path.name}: checksum mismatch on {entry['path']}")
    logger.info("backup: verify %s — %d files intact", path.name, len(manifest["files"]))
    return manifest


def restore_archive(
    archive: str | Path,
    target_dir: str | Path,
    *,
    force: bool = False,
    only: Optional[Iterable[str]] = None,
) -> dict:
    """Extract an archive into ``target_dir`` and prove each database is usable.

    Refuses a non-empty target unless ``force`` — overwriting a live data
    directory by accident is exactly the kind of mistake a restore must not make.
    Every restored file is checked twice: SHA-256 against the manifest, then
    ``PRAGMA integrity_check`` on the actual database.
    """
    src = Path(archive)
    dst = Path(target_dir)
    if dst.exists() and any(dst.iterdir()) and not force:
        raise BackupError(f"target {dst} is not empty — pass force=True to overwrite")
    dst.mkdir(parents=True, exist_ok=True)

    manifest = read_manifest(src)
    wanted = set(only) if only is not None else None
    entries = [e for e in manifest["files"] if wanted is None or e["path"] in wanted]
    if wanted is not None and len(entries) != len(wanted):
        missing = wanted - {e["path"] for e in entries}
        raise BackupError(f"{src.name}: archive has no {sorted(missing)}")

    with tarfile.open(src, "r:gz") as tar:
        for entry in entries:
            member = tar.getmember(entry["path"])
            tar.extract(member, dst, filter="data")

    report: dict[str, dict] = {}
    for entry in entries:
        p = dst / entry["path"]
        if not p.is_file():
            raise BackupError(f"restore: {entry['path']} did not land in {dst}")
        digest = _sha256(p)
        if digest != entry["sha256"]:
            raise BackupError(f"restore: checksum mismatch on {entry['path']}")
        verdict = integrity_check(p)
        if verdict != "ok":
            raise BackupError(f"restore: {entry['path']} fails integrity_check: {verdict}")
        report[entry["path"]] = {"size": p.stat().st_size, "integrity_check": verdict}
        logger.info("backup: restored %s (%d bytes) integrity_check=ok", entry["path"], p.stat().st_size)

    return {"archive": src.name, "target": str(dst), "files": report, "manifest": manifest}


# --------------------------------------------------------------------------- #
# Single-file helpers (used by the ops runbook)
# --------------------------------------------------------------------------- #

def gzip_file(src: str | Path, dst: Optional[str | Path] = None) -> Path:
    """Compress one file, returning the ``.gz`` path. Used by the runbook."""
    s = Path(src)
    d = Path(dst) if dst else s.with_suffix(s.suffix + ".gz")
    with open(s, "rb") as fin, gzip.open(d, "wb", compresslevel=6) as fout:
        shutil.copyfileobj(fin, fout, 1 << 20)
    return d


def free_space_bytes(path: str | Path) -> int:
    """Bytes available where ``path`` lives (the directory is created if needed)."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(str(p)).free


__all__ = [
    "ARCHIVE_PREFIX",
    "ARCHIVE_SUFFIX",
    "ArchiveResult",
    "BackupError",
    "MANIFEST_NAME",
    "archive_name",
    "create_archive",
    "discover_databases",
    "free_space_bytes",
    "gzip_file",
    "integrity_check",
    "parse_archive_stamp",
    "read_manifest",
    "restore_archive",
    "snapshot_database",
    "utc_stamp",
    "verify_archive",
]
