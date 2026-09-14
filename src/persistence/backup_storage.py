"""BKP-1 — Where the backups go: Cloudflare R2 (S3 API), or a local directory.

Credentials come from the environment and **only** from the environment. Nothing
here reads a file in the repository, and nothing here logs a secret: the access
key is never emitted, and the endpoint is logged host-only.

Two destinations behind one interface:

``S3Destination``
    Cloudflare R2 through its S3-compatible API (``boto3``, already a dependency
    — ``requirements.txt`` line 74 — so this adds no new package). Works
    unchanged against Backblaze B2 or plain S3 by pointing the endpoint
    elsewhere; the code has nothing R2-specific in it.

``LocalDirDestination``
    A directory on disk. Its reason to exist is that it makes the whole
    upload → list → prune → download → restore cycle testable with no
    credentials and no network, so the code path exercised by the test suite is
    the same one production runs.

``destination_from_env`` picks one: an explicit ``file://`` or bare path in
``BACKUP_DESTINATION`` gives the local one, otherwise the S3 variables are read.
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Protocol

from src.persistence.sqlite_backup import (
    ARCHIVE_PREFIX,
    ARCHIVE_SUFFIX,
    BackupError,
    parse_archive_stamp,
)

logger = logging.getLogger(__name__)

# Env vars — documented in docs/ops/sauvegardes-sqlite.md and render.yaml.
ENV_DESTINATION = "BACKUP_DESTINATION"          # file:///path  → local dir
ENV_S3_ENDPOINT = "BACKUP_S3_ENDPOINT"          # https://<account>.r2.cloudflarestorage.com
ENV_S3_BUCKET = "BACKUP_S3_BUCKET"
ENV_S3_ACCESS_KEY = "BACKUP_S3_ACCESS_KEY_ID"
ENV_S3_SECRET_KEY = "BACKUP_S3_SECRET_ACCESS_KEY"
ENV_S3_REGION = "BACKUP_S3_REGION"              # R2 ignores it; "auto" is its convention
ENV_S3_PREFIX = "BACKUP_S3_PREFIX"              # optional key prefix, e.g. "prod/"


@dataclass(frozen=True)
class RemoteObject:
    """One stored backup, as the destination reports it."""

    key: str
    size: int
    last_modified: Optional[datetime] = None

    @property
    def name(self) -> str:
        return self.key.rsplit("/", 1)[-1]

    @property
    def stamp(self) -> Optional[datetime]:
        """The instant encoded in the NAME — the only trustworthy clock here.

        A store's own ``last_modified`` is the upload time, which a re-upload or
        a lifecycle rewrite can move. The name is stamped once, by us.
        """
        return parse_archive_stamp(self.key)


class BackupDestination(Protocol):
    """The four operations a backup store has to support."""

    def describe(self) -> str: ...
    def upload(self, local_path: str | Path, key: str) -> RemoteObject: ...
    def list_backups(self) -> list[RemoteObject]: ...
    def download(self, key: str, local_path: str | Path) -> Path: ...
    def delete(self, key: str) -> None: ...


# --------------------------------------------------------------------------- #
# Local directory
# --------------------------------------------------------------------------- #

class LocalDirDestination:
    """A plain directory. Used by the tests, and by a local dry run."""

    def __init__(self, root: str | Path, prefix: str = "") -> None:
        self._root = Path(root)
        self._prefix = prefix.strip("/")
        self._root.mkdir(parents=True, exist_ok=True)

    def describe(self) -> str:
        return f"local:{self._root}" + (f"/{self._prefix}" if self._prefix else "")

    def _path(self, key: str) -> Path:
        return self._root / key

    def upload(self, local_path: str | Path, key: str) -> RemoteObject:
        dst = self._path(key)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(local_path), str(dst))
        st = dst.stat()
        return RemoteObject(
            key=key,
            size=st.st_size,
            last_modified=datetime.fromtimestamp(st.st_mtime, tz=timezone.utc),
        )

    def list_backups(self) -> list[RemoteObject]:
        base = self._root / self._prefix if self._prefix else self._root
        if not base.is_dir():
            return []
        out: list[RemoteObject] = []
        for p in sorted(base.glob(f"{ARCHIVE_PREFIX}*{ARCHIVE_SUFFIX}")):
            st = p.stat()
            key = f"{self._prefix}/{p.name}" if self._prefix else p.name
            out.append(
                RemoteObject(
                    key=key,
                    size=st.st_size,
                    last_modified=datetime.fromtimestamp(st.st_mtime, tz=timezone.utc),
                )
            )
        return out

    def download(self, key: str, local_path: str | Path) -> Path:
        src = self._path(key)
        if not src.is_file():
            raise BackupError(f"no such backup: {key}")
        dst = Path(local_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dst))
        return dst

    def delete(self, key: str) -> None:
        p = self._path(key)
        if p.is_file():
            p.unlink()


# --------------------------------------------------------------------------- #
# S3 / Cloudflare R2
# --------------------------------------------------------------------------- #

class S3Destination:
    """Cloudflare R2 (or any S3-compatible store) through ``boto3``."""

    def __init__(
        self,
        *,
        bucket: str,
        endpoint_url: Optional[str] = None,
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None,
        region: str = "auto",
        prefix: str = "",
        client=None,
    ) -> None:
        if not bucket:
            raise BackupError(f"{ENV_S3_BUCKET} is empty — no bucket to write to")
        self._bucket = bucket
        self._prefix = prefix.strip("/")
        self._endpoint = endpoint_url
        if client is not None:
            self._client = client
            return
        try:
            import boto3  # noqa: PLC0415 — optional at import time, required at use
            from botocore.config import Config
        except ImportError as exc:  # pragma: no cover - boto3 is in requirements.txt
            raise BackupError(
                "boto3 is required for the S3/R2 destination (requirements.txt line 74)"
            ) from exc
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            region_name=region,
            # R2 speaks SigV4 only. Retries matter: a nightly job gets one shot.
            config=Config(
                signature_version="s3v4",
                retries={"max_attempts": 5, "mode": "standard"},
                connect_timeout=20,
                read_timeout=120,
            ),
        )

    def describe(self) -> str:
        host = (self._endpoint or "s3.amazonaws.com").split("//")[-1].split("/")[0]
        return f"s3://{self._bucket}/{self._prefix} @ {host}"

    def _key(self, name: str) -> str:
        return f"{self._prefix}/{name}" if self._prefix else name

    def upload(self, local_path: str | Path, key: str) -> RemoteObject:
        p = Path(local_path)
        full = key if key.startswith(f"{self._prefix}/") or not self._prefix else self._key(key)
        self._client.upload_file(str(p), self._bucket, full)
        return RemoteObject(key=full, size=p.stat().st_size, last_modified=datetime.now(timezone.utc))

    def list_backups(self) -> list[RemoteObject]:
        out: list[RemoteObject] = []
        token: Optional[str] = None
        list_prefix = f"{self._prefix}/{ARCHIVE_PREFIX}" if self._prefix else ARCHIVE_PREFIX
        while True:
            kwargs = {"Bucket": self._bucket, "Prefix": list_prefix}
            if token:
                kwargs["ContinuationToken"] = token
            resp = self._client.list_objects_v2(**kwargs)
            for item in resp.get("Contents", []) or []:
                key = item["Key"]
                if not key.endswith(ARCHIVE_SUFFIX):
                    continue
                lm = item.get("LastModified")
                if lm is not None and lm.tzinfo is None:
                    lm = lm.replace(tzinfo=timezone.utc)
                out.append(RemoteObject(key=key, size=int(item.get("Size", 0)), last_modified=lm))
            if not resp.get("IsTruncated"):
                break
            token = resp.get("NextContinuationToken")
            if not token:
                break
        return sorted(out, key=lambda o: o.key)

    def download(self, key: str, local_path: str | Path) -> Path:
        dst = Path(local_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        self._client.download_file(self._bucket, key, str(dst))
        return dst

    def delete(self, key: str) -> None:
        self._client.delete_object(Bucket=self._bucket, Key=key)


# --------------------------------------------------------------------------- #
# Wiring from the environment
# --------------------------------------------------------------------------- #

def destination_from_env(env: Optional[dict] = None) -> BackupDestination:
    """Build the destination the environment describes.

    Raises ``BackupError`` with a message naming the missing variable — a
    misconfigured backup has to fail at the first run, loudly, not months later
    when someone needs to restore.
    """
    e = env if env is not None else os.environ

    raw = (e.get(ENV_DESTINATION) or "").strip()
    if raw:
        path = raw[len("file://") :] if raw.startswith("file://") else raw
        logger.info("backup destination: local directory %s", path)
        return LocalDirDestination(path, prefix=(e.get(ENV_S3_PREFIX) or "").strip("/"))

    bucket = (e.get(ENV_S3_BUCKET) or "").strip()
    endpoint = (e.get(ENV_S3_ENDPOINT) or "").strip() or None
    access = (e.get(ENV_S3_ACCESS_KEY) or "").strip() or None
    secret = (e.get(ENV_S3_SECRET_KEY) or "").strip() or None
    missing = [
        name
        for name, val in (
            (ENV_S3_BUCKET, bucket),
            (ENV_S3_ENDPOINT, endpoint),
            (ENV_S3_ACCESS_KEY, access),
            (ENV_S3_SECRET_KEY, secret),
        )
        if not val
    ]
    if missing:
        raise BackupError(
            "backup destination not configured — set "
            + ", ".join(missing)
            + f" (or {ENV_DESTINATION}=file:///path for a local target)"
        )
    dest = S3Destination(
        bucket=bucket,
        endpoint_url=endpoint,
        access_key_id=access,
        secret_access_key=secret,
        region=(e.get(ENV_S3_REGION) or "auto").strip(),
        prefix=(e.get(ENV_S3_PREFIX) or "").strip("/"),
    )
    logger.info("backup destination: %s", dest.describe())
    return dest


def destination_configured(env: Optional[dict] = None) -> bool:
    """True when a destination *could* be built. Never raises."""
    try:
        e = env if env is not None else os.environ
        if (e.get(ENV_DESTINATION) or "").strip():
            return True
        return all(
            (e.get(k) or "").strip()
            for k in (ENV_S3_BUCKET, ENV_S3_ENDPOINT, ENV_S3_ACCESS_KEY, ENV_S3_SECRET_KEY)
        )
    except Exception:  # pragma: no cover - defensive
        return False


__all__ = [
    "BackupDestination",
    "LocalDirDestination",
    "RemoteObject",
    "S3Destination",
    "destination_configured",
    "destination_from_env",
]
