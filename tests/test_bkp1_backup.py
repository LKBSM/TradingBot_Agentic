"""BKP-1 — tests for the daily SQLite backup, its verification and its restore.

The whole cycle (snapshot → archive → upload → list → prune → download →
restore → integrity_check) runs here against ``LocalDirDestination``, which is
the same code path production takes with ``S3Destination`` swapped in. The S3
side is covered separately with a fake client, so the key handling and the
pagination are exercised without credentials.
"""

from __future__ import annotations

import json
import sqlite3
import tarfile
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.persistence.backup_daemon import BackupDaemon
from src.persistence.backup_service import (
    DEFAULT_MAX_AGE_HOURS,
    health_snapshot,
    prune_old_backups,
    read_state,
    restore_latest,
    run_backup,
    schedule_utc,
    verify_backups,
)
from src.persistence.backup_storage import (
    LocalDirDestination,
    RemoteObject,
    S3Destination,
    destination_configured,
    destination_from_env,
)
from src.persistence.sqlite_backup import (
    ARCHIVE_PREFIX,
    ARCHIVE_SUFFIX,
    MANIFEST_NAME,
    BackupError,
    archive_name,
    create_archive,
    discover_databases,
    integrity_check,
    parse_archive_stamp,
    read_manifest,
    restore_archive,
    snapshot_database,
    verify_archive,
)

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

def _make_db(path: Path, rows: int = 200, table: str = "t") -> Path:
    """A WAL database with some content — the shape production actually has."""
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(f"CREATE TABLE IF NOT EXISTS {table} (id INTEGER PRIMARY KEY, v TEXT)")
        conn.executemany(
            f"INSERT INTO {table} (v) VALUES (?)", [(f"value-{i}" * 8,) for i in range(rows)]
        )
        conn.commit()
    finally:
        conn.close()
    return path


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    d = tmp_path / "data"
    d.mkdir()
    _make_db(d / "candles.db", rows=500)
    _make_db(d / "market_readings.db", rows=120)
    _make_db(d / "accounts.db", rows=10)
    return d


@pytest.fixture()
def env(tmp_path: Path, data_dir: Path) -> dict:
    """An isolated environment mapping — never os.environ."""
    return {
        "DATA_DIR": str(data_dir),
        "BACKUP_ENABLED": "1",
        "BACKUP_DESTINATION": f"file://{tmp_path / 'store'}",
        "BACKUP_STAGING_DIR": str(tmp_path / "stage"),
        "BACKUP_RETENTION_DAYS": "30",
    }


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #

def test_discovery_takes_every_db_and_no_sidecar(data_dir: Path):
    # WAL sidecars must never be shipped: the online backup API already folds
    # the WAL into the snapshot, so a sidecar would only add torn bytes.
    (data_dir / "candles.db-wal").write_bytes(b"junk")
    (data_dir / "candles.db-shm").write_bytes(b"junk")
    (data_dir / "_tmp_scratch.db").write_bytes(b"junk")

    names = [p.name for p in discover_databases(data_dir)]

    assert names == ["accounts.db", "candles.db", "market_readings.db"]


def test_discovery_is_not_a_hard_coded_list(data_dir: Path):
    # The point of discovering rather than listing: a store added later cannot
    # silently fall out of the backup.
    _make_db(data_dir / "brand_new_store.db", rows=5)

    assert "brand_new_store.db" in [p.name for p in discover_databases(data_dir)]


def test_discovery_rejects_a_missing_directory(tmp_path: Path):
    with pytest.raises(BackupError, match="data dir not found"):
        discover_databases(tmp_path / "nope")


# --------------------------------------------------------------------------- #
# Snapshot — the reason this mission exists
# --------------------------------------------------------------------------- #

def test_snapshot_is_consistent_while_a_writer_is_hammering(data_dir: Path, tmp_path: Path):
    """A file copy would tear here. The online backup API must not."""
    src = data_dir / "candles.db"
    stop = threading.Event()
    errors: list[Exception] = []

    def writer():
        conn = sqlite3.connect(str(src), timeout=30.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            i = 0
            while not stop.is_set():
                conn.execute("INSERT INTO t (v) VALUES (?)", (f"live-{i}" * 20,))
                conn.commit()
                i += 1
        except Exception as exc:  # pragma: no cover - surfaced by the assert below
            errors.append(exc)
        finally:
            conn.close()

    th = threading.Thread(target=writer, daemon=True)
    th.start()
    time.sleep(0.05)
    try:
        snap = snapshot_database(src, tmp_path / "snap.db")
    finally:
        stop.set()
        th.join(timeout=10)

    assert not errors, f"the writer failed: {errors}"
    # snapshot_database raises unless integrity_check says "ok"; assert it again
    # explicitly so the guarantee is visible in the test, not just implied.
    assert integrity_check(snap) == "ok"
    conn = sqlite3.connect(str(snap))
    try:
        # A consistent point in time: at least the rows that existed before the
        # writer started, and a whole number of committed transactions.
        assert conn.execute("SELECT COUNT(*) FROM t").fetchone()[0] >= 500
    finally:
        conn.close()


def test_snapshot_refuses_a_corrupt_source(tmp_path: Path):
    bad = tmp_path / "broken.db"
    bad.write_bytes(b"SQLite format 3\x00" + b"\x00" * 200)  # header only, no pages
    with pytest.raises(BackupError):
        snapshot_database(bad, tmp_path / "out.db")


def test_snapshot_refuses_a_missing_source(tmp_path: Path):
    with pytest.raises(BackupError, match="source database not found"):
        snapshot_database(tmp_path / "absent.db", tmp_path / "out.db")


# --------------------------------------------------------------------------- #
# Archive / verify / restore
# --------------------------------------------------------------------------- #

def test_archive_round_trip_restores_usable_databases(data_dir: Path, tmp_path: Path):
    archive = create_archive(data_dir, tmp_path / "out")

    assert archive.name.startswith(ARCHIVE_PREFIX)
    assert archive.name.endswith(ARCHIVE_SUFFIX)
    assert archive.size_bytes < archive.source_bytes  # it is compressed

    verify_archive(archive.path)
    report = restore_archive(archive.path, tmp_path / "restored")

    assert set(report["files"]) == {"accounts.db", "candles.db", "market_readings.db"}
    for name, info in report["files"].items():
        assert info["integrity_check"] == "ok", name
    # And the content actually came back.
    conn = sqlite3.connect(str(tmp_path / "restored" / "candles.db"))
    try:
        assert conn.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 500
    finally:
        conn.close()


def test_manifest_is_the_first_member(data_dir: Path, tmp_path: Path):
    # A streaming reader must be able to validate before pulling gigabytes.
    archive = create_archive(data_dir, tmp_path / "out")
    with tarfile.open(archive.path, "r:gz") as tar:
        assert tar.getnames()[0] == MANIFEST_NAME


def test_manifest_records_the_method_and_the_integrity_verdict(data_dir: Path, tmp_path: Path):
    archive = create_archive(data_dir, tmp_path / "out")
    manifest = read_manifest(archive.path)

    assert "online backup API" in manifest["method"]
    assert {e["path"] for e in manifest["files"]} == {
        "accounts.db",
        "candles.db",
        "market_readings.db",
    }
    for entry in manifest["files"]:
        assert entry["integrity_check"] == "ok"
        assert len(entry["sha256"]) == 64


def test_verify_catches_a_tampered_member(data_dir: Path, tmp_path: Path):
    """The manifest has to be able to call out bytes that changed in transit."""
    archive = create_archive(data_dir, tmp_path / "out")
    work = tmp_path / "work"
    work.mkdir()
    with tarfile.open(archive.path, "r:gz") as tar:
        tar.extractall(work, filter="data")
    # Same manifest, one database quietly altered.
    with open(work / "accounts.db", "r+b") as f:
        f.seek(4096)
        f.write(b"\xde\xad\xbe\xef")
    tampered = tmp_path / "tampered.tar.gz"
    with tarfile.open(tampered, "w:gz") as tar:
        tar.add(work / MANIFEST_NAME, arcname=MANIFEST_NAME)
        for name in ("accounts.db", "candles.db", "market_readings.db"):
            tar.add(work / name, arcname=name)

    with pytest.raises(BackupError, match="checksum mismatch"):
        verify_archive(tampered)


def test_verify_rejects_an_archive_that_is_not_ours(tmp_path: Path):
    foreign = tmp_path / "foreign.tar.gz"
    (tmp_path / "x.txt").write_text("hello")
    with tarfile.open(foreign, "w:gz") as tar:
        tar.add(tmp_path / "x.txt", arcname="x.txt")

    with pytest.raises(BackupError, match="carries no MANIFEST"):
        verify_archive(foreign)


def test_restore_refuses_a_non_empty_target(data_dir: Path, tmp_path: Path):
    archive = create_archive(data_dir, tmp_path / "out")
    target = tmp_path / "occupied"
    target.mkdir()
    (target / "precious.db").write_text("do not overwrite me")

    with pytest.raises(BackupError, match="not empty"):
        restore_archive(archive.path, target)

    # force is the explicit way through.
    restore_archive(archive.path, target, force=True)
    assert (target / "candles.db").is_file()
    assert (target / "precious.db").is_file()


def test_restore_can_pick_one_database(data_dir: Path, tmp_path: Path):
    archive = create_archive(data_dir, tmp_path / "out")
    report = restore_archive(archive.path, tmp_path / "one", only=["accounts.db"])

    assert set(report["files"]) == {"accounts.db"}
    assert not (tmp_path / "one" / "candles.db").exists()


def test_archive_refuses_an_empty_data_dir(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(BackupError, match="refusing an empty backup"):
        create_archive(empty, tmp_path / "out")


# --------------------------------------------------------------------------- #
# Naming
# --------------------------------------------------------------------------- #

def test_names_sort_chronologically_and_parse_back():
    early = archive_name(datetime(2026, 9, 13, 3, 17, tzinfo=timezone.utc))
    late = archive_name(datetime(2026, 9, 14, 3, 17, tzinfo=timezone.utc))

    assert early < late  # lexicographic order == chronological order
    assert parse_archive_stamp(early) == datetime(2026, 9, 13, 3, 17, tzinfo=timezone.utc)
    assert parse_archive_stamp(f"prod/nested/{early}") is not None
    assert parse_archive_stamp("something-else.tar.gz") is None
    assert parse_archive_stamp(f"{ARCHIVE_PREFIX}not-a-date{ARCHIVE_SUFFIX}") is None


# --------------------------------------------------------------------------- #
# Destinations
# --------------------------------------------------------------------------- #

def test_local_destination_round_trip(tmp_path: Path, data_dir: Path):
    dest = LocalDirDestination(tmp_path / "store")
    archive = create_archive(data_dir, tmp_path / "out")

    remote = dest.upload(archive.path, archive.name)
    assert remote.size == archive.size_bytes
    assert [o.key for o in dest.list_backups()] == [archive.name]

    back = dest.download(archive.name, tmp_path / "back.tar.gz")
    verify_archive(back)

    dest.delete(archive.name)
    assert dest.list_backups() == []


def test_destination_from_env_names_what_is_missing():
    with pytest.raises(BackupError) as exc:
        destination_from_env({})
    message = str(exc.value)
    for var in (
        "BACKUP_S3_BUCKET",
        "BACKUP_S3_ENDPOINT",
        "BACKUP_S3_ACCESS_KEY_ID",
        "BACKUP_S3_SECRET_ACCESS_KEY",
    ):
        assert var in message, f"{var} should be named so the fix is obvious"


def test_destination_configured_never_raises():
    assert destination_configured({}) is False
    assert destination_configured({"BACKUP_DESTINATION": "file:///tmp/x"}) is True
    assert (
        destination_configured(
            {
                "BACKUP_S3_BUCKET": "b",
                "BACKUP_S3_ENDPOINT": "https://e",
                "BACKUP_S3_ACCESS_KEY_ID": "k",
                "BACKUP_S3_SECRET_ACCESS_KEY": "s",
            }
        )
        is True
    )


class _FakeS3:
    """Enough of the boto3 S3 client to exercise S3Destination, paging included."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.page_size = 2
        self.calls: list[str] = []

    def upload_file(self, filename, bucket, key):
        self.calls.append(f"upload:{key}")
        self.objects[key] = Path(filename).read_bytes()

    def download_file(self, bucket, key, filename):
        self.calls.append(f"download:{key}")
        Path(filename).write_bytes(self.objects[key])

    def delete_object(self, Bucket, Key):  # noqa: N803 — boto3's own casing
        self.calls.append(f"delete:{Key}")
        self.objects.pop(Key, None)

    def list_objects_v2(self, **kwargs):
        self.calls.append("list")
        prefix = kwargs.get("Prefix", "")
        keys = sorted(k for k in self.objects if k.startswith(prefix))
        start = int(kwargs.get("ContinuationToken") or 0)
        page = keys[start : start + self.page_size]
        nxt = start + self.page_size
        return {
            "Contents": [
                {"Key": k, "Size": len(self.objects[k]), "LastModified": datetime.now(timezone.utc)}
                for k in page
            ],
            "IsTruncated": nxt < len(keys),
            "NextContinuationToken": str(nxt),
        }


def test_s3_destination_prefixes_keys_and_pages_the_listing(tmp_path: Path):
    fake = _FakeS3()
    dest = S3Destination(bucket="mia-backups", prefix="prod", client=fake)

    for day in range(1, 6):
        blob = tmp_path / f"a{day}.tar.gz"
        blob.write_bytes(b"x" * (100 + day))
        dest.upload(blob, archive_name(datetime(2026, 9, day, 3, 17, tzinfo=timezone.utc)))

    listed = dest.list_backups()
    # 5 objects through a 2-per-page fake: the paging loop has to gather them all.
    assert len(listed) == 5
    assert all(o.key.startswith("prod/") for o in listed)
    assert all(o.stamp is not None for o in listed)
    assert "s3://mia-backups/prod" in dest.describe()

    dest.delete(listed[0].key)
    assert len(dest.list_backups()) == 4


def test_s3_destination_ignores_foreign_objects(tmp_path: Path):
    fake = _FakeS3()
    fake.objects["notes.txt"] = b"unrelated"
    fake.objects["mia-backup-20260913T031700Z.tar.gz"] = b"ours"
    dest = S3Destination(bucket="b", client=fake)

    assert [o.name for o in dest.list_backups()] == ["mia-backup-20260913T031700Z.tar.gz"]


def test_remote_object_trusts_the_name_not_the_store_clock():
    # A re-upload moves last_modified; the stamp in the name is written once.
    obj = RemoteObject(
        key="prod/mia-backup-20260901T031700Z.tar.gz",
        size=10,
        last_modified=datetime(2026, 9, 30, tzinfo=timezone.utc),
    )
    assert obj.stamp == datetime(2026, 9, 1, 3, 17, tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# Verification — the part that must fail loudly
# --------------------------------------------------------------------------- #

def _seed(dest: LocalDirDestination, sizes: dict[datetime, int]) -> None:
    for when, size in sizes.items():
        blob = Path(dest.describe().split(":", 1)[1]) / archive_name(when)
        blob.write_bytes(b"x" * size)


def test_verify_says_so_when_the_store_is_empty(tmp_path: Path):
    report = verify_backups(LocalDirDestination(tmp_path / "store"))

    assert report.healthy is False
    assert "no backup found" in report.problems[0]


def test_verify_catches_a_stale_backup(tmp_path: Path):
    now = datetime(2026, 9, 13, 6, 0, tzinfo=timezone.utc)
    dest = LocalDirDestination(tmp_path / "store")
    _seed(dest, {now - timedelta(days=3): 1000})

    report = verify_backups(dest, now=now)

    assert report.healthy is False
    assert any("old" in p for p in report.problems)
    assert report.latest_age_hours == pytest.approx(72.0, abs=0.1)


def test_verify_accepts_a_fresh_backup(tmp_path: Path):
    now = datetime(2026, 9, 13, 6, 0, tzinfo=timezone.utc)
    dest = LocalDirDestination(tmp_path / "store")
    _seed(dest, {now - timedelta(hours=3): 1000})

    report = verify_backups(dest, now=now)

    assert report.healthy is True
    assert report.problems == []
    assert report.count == 1


def test_verify_catches_a_backup_that_shrank(tmp_path: Path):
    """The classic silent failure: the job 'succeeded' and shipped near-nothing."""
    now = datetime(2026, 9, 13, 6, 0, tzinfo=timezone.utc)
    dest = LocalDirDestination(tmp_path / "store")
    _seed(
        dest,
        {
            now - timedelta(days=4): 1000,
            now - timedelta(days=3): 1010,
            now - timedelta(days=2): 990,
            now - timedelta(days=1): 1005,
            now - timedelta(hours=3): 40,  # today: a fortieth of the usual
        },
    )

    report = verify_backups(dest, now=now)

    assert report.healthy is False
    assert any("median" in p for p in report.problems)
    assert report.median_size == 1002


def test_verify_tolerates_normal_growth(tmp_path: Path):
    now = datetime(2026, 9, 13, 6, 0, tzinfo=timezone.utc)
    dest = LocalDirDestination(tmp_path / "store")
    _seed(
        dest,
        {
            now - timedelta(days=4): 1000,
            now - timedelta(days=3): 1020,
            now - timedelta(days=2): 1040,
            now - timedelta(days=1): 1060,
            now - timedelta(hours=3): 1090,
        },
    )

    assert verify_backups(dest, now=now).healthy is True


def test_verify_waits_for_enough_history_before_judging_size(tmp_path: Path):
    # Two backups is not a median. Refusing to judge beats a false alarm on day 2.
    now = datetime(2026, 9, 13, 6, 0, tzinfo=timezone.utc)
    dest = LocalDirDestination(tmp_path / "store")
    _seed(dest, {now - timedelta(days=1): 5000, now - timedelta(hours=2): 10})

    report = verify_backups(dest, now=now)

    assert report.healthy is True
    assert report.median_size is None


def test_verify_reports_rather_than_raises_when_the_store_is_unreachable():
    class Broken:
        def describe(self):
            return "broken://"

        def list_backups(self):
            raise ConnectionError("R2 unreachable")

    report = verify_backups(Broken())

    assert report.healthy is False
    assert "cannot list" in report.problems[0]


# --------------------------------------------------------------------------- #
# Retention
# --------------------------------------------------------------------------- #

def test_prune_drops_only_what_is_past_the_window(tmp_path: Path):
    now = datetime(2026, 9, 13, 6, 0, tzinfo=timezone.utc)
    dest = LocalDirDestination(tmp_path / "store")
    _seed(
        dest,
        {
            now - timedelta(days=40): 100,
            now - timedelta(days=31): 100,
            now - timedelta(days=29): 100,
            now - timedelta(days=1): 100,
            now: 100,
        },
    )

    deleted = prune_old_backups(dest, days=30, now=now)

    assert len(deleted) == 2
    remaining = {o.stamp for o in dest.list_backups()}
    assert len(remaining) == 3


def test_prune_can_never_empty_the_store(tmp_path: Path):
    """If the clock or the window is wrong, retention must not be the disaster."""
    now = datetime(2026, 9, 13, tzinfo=timezone.utc)
    dest = LocalDirDestination(tmp_path / "store")
    _seed(dest, {now - timedelta(days=d): 100 for d in (10, 5, 1)})

    deleted = prune_old_backups(dest, days=1, now=now + timedelta(days=3650))

    assert len(dest.list_backups()) == 2, "the two most recent must always survive"
    assert len(deleted) == 1


# --------------------------------------------------------------------------- #
# A whole run
# --------------------------------------------------------------------------- #

def test_run_backup_uploads_verifies_and_records(env: dict, tmp_path: Path):
    result = run_backup(env=env, now=datetime(2026, 9, 13, 3, 17, tzinfo=timezone.utc))

    assert result.ok is True, result.error
    assert sorted(result.databases) == ["accounts.db", "candles.db", "market_readings.db"]
    assert result.size_bytes > 0

    dest = destination_from_env(env)
    assert [o.name for o in dest.list_backups()] == [result.archive]

    state = read_state(env)
    assert state["last_archive"] == result.archive
    assert state["last_error"] is None
    assert state["last_success_utc"].startswith("2026-09-13")


def test_run_backup_leaves_no_scratch_behind(env: dict, tmp_path: Path):
    stage = Path(env["BACKUP_STAGING_DIR"])
    run_backup(env=env)

    leftovers = [p for p in stage.iterdir()] if stage.is_dir() else []
    assert leftovers == [], f"staging not cleaned: {leftovers}"


def test_run_backup_records_the_failure_instead_of_raising(env: dict, tmp_path: Path):
    env = dict(env, DATA_DIR=str(tmp_path / "gone"))

    result = run_backup(env=env)

    assert result.ok is False
    # An empty data dir is never a backup worth uploading: shipping a valid,
    # empty archive is how a store fills with 30 days of nothing.
    assert "refusing an empty backup" in (result.error or "")
    assert read_state(env)["last_error"] == result.error


def test_run_backup_refuses_when_there_is_no_room(env: dict, tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "src.persistence.backup_service.free_space_bytes", lambda _p: 1024
    )

    result = run_backup(env=env)

    assert result.ok is False
    assert "not enough room" in (result.error or "")
    assert "BACKUP_STAGING_DIR" in (result.error or ""), "the message must say how to fix it"


def test_run_backup_prunes_past_the_window_but_never_the_last_two(env: dict, tmp_path: Path):
    now = datetime(2026, 9, 13, 3, 17, tzinfo=timezone.utc)
    dest = destination_from_env(env)
    store = tmp_path / "store"
    store.mkdir(exist_ok=True)
    old = [now - timedelta(days=d) for d in (60, 50, 40)]
    for when in old:
        (store / archive_name(when)).write_bytes(b"x" * 100)

    result = run_backup(env=env, now=now)

    assert result.ok is True
    # The two oldest go; the newest of the old ones survives on the keep_minimum
    # floor even though it is past 30 days — retention must never be able to
    # leave the store with fewer than two backups.
    assert sorted(result.pruned) == sorted(archive_name(w) for w in old[:2])
    remaining = sorted(o.name for o in dest.list_backups())
    assert remaining == sorted([archive_name(old[2]), result.archive])


def test_restore_latest_round_trip(env: dict, tmp_path: Path):
    run_backup(env=env)

    report = restore_latest(tmp_path / "recovered", env=env)

    assert set(report["files"]) == {"accounts.db", "candles.db", "market_readings.db"}
    for info in report["files"].values():
        assert info["integrity_check"] == "ok"
    conn = sqlite3.connect(str(tmp_path / "recovered" / "market_readings.db"))
    try:
        assert conn.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 120
    finally:
        conn.close()


def test_restore_latest_picks_the_newest(env: dict, tmp_path: Path):
    first = run_backup(env=env, now=datetime(2026, 9, 11, 3, 17, tzinfo=timezone.utc))
    second = run_backup(env=env, now=datetime(2026, 9, 12, 3, 17, tzinfo=timezone.utc))

    report = restore_latest(tmp_path / "recovered", env=env)

    assert report["source_key"] == second.archive
    assert report["source_key"] != first.archive


def test_restore_latest_says_so_when_there_is_nothing(env: dict, tmp_path: Path):
    with pytest.raises(BackupError, match="no backup to restore"):
        restore_latest(tmp_path / "recovered", env=env)


# --------------------------------------------------------------------------- #
# Health + daemon
# --------------------------------------------------------------------------- #

def test_health_snapshot_is_absent_when_backups_are_off(env: dict):
    assert health_snapshot(dict(env, BACKUP_ENABLED="0")) is None


def test_health_snapshot_reports_fresh_then_stale(env: dict):
    run_backup(env=env)
    fresh = health_snapshot(env)
    assert fresh is not None
    assert fresh["fresh"] is True
    assert fresh["last_backup_age_hours"] < 1

    # Rewind the recorded success well past the limit.
    from src.persistence.backup_service import write_state

    write_state(
        {
            "last_success_utc": (
                datetime.now(timezone.utc) - timedelta(hours=DEFAULT_MAX_AGE_HOURS + 5)
            ).isoformat()
        },
        env,
    )
    assert health_snapshot(env)["fresh"] is False


def test_health_snapshot_survives_a_corrupt_state_file(env: dict):
    from src.persistence.backup_service import state_path

    state_path(env).parent.mkdir(parents=True, exist_ok=True)
    state_path(env).write_text("{not json", encoding="utf-8")

    snap = health_snapshot(env)

    assert snap is not None
    assert snap["fresh"] is False


def test_daemon_stays_off_unless_asked(env: dict):
    assert BackupDaemon(dict(env, BACKUP_ENABLED="0")).start() is False


def test_daemon_refuses_to_pretend_without_a_destination(env: dict, caplog):
    """Enabled + nowhere to write is the worst case: it must shout, not sulk."""
    blind = {k: v for k, v in env.items() if k != "BACKUP_DESTINATION"}
    with caplog.at_level("ERROR"):
        started = BackupDaemon(blind).start()

    assert started is False
    assert "NOTHING WILL BE BACKED UP" in caplog.text


def test_daemon_schedules_and_stops(env: dict):
    pytest.importorskip("apscheduler")
    daemon = BackupDaemon(dict(env, BACKUP_HOUR_UTC="4", BACKUP_MINUTE_UTC="5"))
    try:
        assert daemon.start() is True
        assert daemon.running is True
        jobs = daemon._scheduler.get_jobs()  # noqa: SLF001 — asserting the wiring
        assert [j.id for j in jobs] == ["daily-sqlite-backup"]
    finally:
        daemon.stop()
    assert daemon.running is False


def test_schedule_reads_the_env_and_clamps_nonsense():
    assert schedule_utc({"BACKUP_HOUR_UTC": "4", "BACKUP_MINUTE_UTC": "5"}) == (4, 5)
    assert schedule_utc({"BACKUP_HOUR_UTC": "26", "BACKUP_MINUTE_UTC": "70"}) == (2, 10)
    assert schedule_utc({"BACKUP_HOUR_UTC": "oops"}) == (3, 17)


# --------------------------------------------------------------------------- #
# The line: no secret ever reaches the repository
# --------------------------------------------------------------------------- #

def test_no_backup_credential_is_committed():
    """A guard, not a formality: this is how a key ends up in a public repo."""
    root = Path(__file__).resolve().parents[1]
    targets = [
        root / "render.yaml",
        root / ".github" / "workflows" / "backup-verify.yml",
        root / "scripts" / "backup_sqlite.py",
        root / "docs" / "ops" / "sauvegardes-sqlite.md",
        *(root / "src" / "persistence").glob("backup*.py"),
        root / "src" / "persistence" / "sqlite_backup.py",
    ]
    for path in targets:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            # The variable NAMES are everywhere; a VALUE next to them is the bug.
            for var in ("BACKUP_S3_ACCESS_KEY_ID", "BACKUP_S3_SECRET_ACCESS_KEY"):
                if f"{var}=" in line and "${{" not in line and "$" not in line:
                    assert line.strip().endswith("=") or "<" in line, (
                        f"{path.name} looks like it carries a value for {var}: {line.strip()}"
                    )
        # R2 account ids are 32 hex characters; none should be written down.
        assert "r2.cloudflarestorage.com" not in text or "<account" in text or "${{" in text or "account-id" in text, (
            f"{path.name} hard-codes an R2 endpoint"
        )


def test_state_file_never_carries_a_credential(env: dict):
    run_backup(env=env)
    raw = json.dumps(read_state(env))

    for needle in ("ACCESS_KEY", "SECRET", "password"):
        assert needle not in raw


# --------------------------------------------------------------------------- #
# Shutdown wiring
# --------------------------------------------------------------------------- #

def test_shutdown_stops_the_daemon_before_the_stores_close(monkeypatch):
    """A snapshot caught mid-teardown would read a database closing under it."""
    from src.api.app import _auto_register_default_handlers
    from src.api.dependencies import AppState
    from src.api.shutdown import GracefulShutdownCoordinator

    monkeypatch.setenv("BACKUP_ENABLED", "1")
    coord = GracefulShutdownCoordinator()
    state = AppState(signal_store=type("S", (), {"close": lambda self: None})())
    _auto_register_default_handlers(coord, state)

    names = [r.name for r in coord._registrations]  # noqa: SLF001
    assert "backup-daemon" in names
    assert names.index("backup-daemon") < names.index("signal-store")


def test_shutdown_registers_nothing_when_backups_are_off(monkeypatch):
    from src.api.app import _auto_register_default_handlers
    from src.api.dependencies import AppState
    from src.api.shutdown import GracefulShutdownCoordinator

    monkeypatch.delenv("BACKUP_ENABLED", raising=False)
    coord = GracefulShutdownCoordinator()
    state = AppState(signal_store=type("S", (), {"close": lambda self: None})())
    _auto_register_default_handlers(coord, state)

    assert "backup-daemon" not in [r.name for r in coord._registrations]  # noqa: SLF001


def test_the_app_boots_backs_up_and_says_so_on_health(tmp_path: Path, monkeypatch):
    """The whole chain, through create_app: boot → catch-up backup → /health.

    Guards the wiring, not the algorithm: a refactor that stops starting the
    daemon, or drops the `backup` field, fails here rather than in production
    three weeks later.
    """
    from fastapi.testclient import TestClient

    data = tmp_path / "data"
    data.mkdir()
    _make_db(data / "candles.db", rows=80)
    _make_db(data / "accounts.db", rows=5)
    store = tmp_path / "store"

    for key, value in {
        "DATA_DIR": str(data),
        "BACKUP_ENABLED": "1",
        "BACKUP_DESTINATION": f"file://{store}",
        "BACKUP_STAGING_DIR": str(tmp_path / "stage"),
        "SENTINEL_TESTING_MODE": "1",
        "BOOTSTRAP_ENABLED": "false",
        "SCHEDULER_ENABLED": "false",
        "CHATBOT_ENABLED": "false",
        "NEWS_PIPELINE_ENABLED": "false",
        "LIVE_TICK_ENABLED": "false",
    }.items():
        monkeypatch.setenv(key, value)

    from src.api.app import create_app
    from src.persistence.backup_daemon import stop_backup_daemon

    state_file = data / "backup_state.json"
    try:
        with TestClient(create_app()) as client:
            # The catch-up runs in a thread; wait for the state, not the clock.
            for _ in range(120):
                time.sleep(0.25)
                if state_file.is_file() and json.loads(state_file.read_text()).get(
                    "last_success_utc"
                ):
                    break
            body = client.get("/health").json()
    finally:
        stop_backup_daemon()

    archives = sorted(p.name for p in store.glob("*.tar.gz"))
    assert len(archives) == 1, f"expected one backup, got {archives}"

    backup = body["backup"]
    assert backup["enabled"] is True
    assert backup["fresh"] is True
    assert backup["last_error"] is None
    assert backup["last_archive"] == archives[0]
    assert body["status"] == "healthy"
