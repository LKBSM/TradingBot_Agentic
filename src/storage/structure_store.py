"""LB-1 — persisted structure: detected zones (OB/FVG) + BOS/CHOCH event journal.

The mission's real unblock: the history is analysed ONCE and its zones and events
are written here with their lifecycle, so a refresh processes only the newest
candle and the event journal exists *by construction*. Same pattern as
``CandlesCacheStore`` (WAL, ``RLock``, connection-per-call, ``schema_version``).

Two tables:

  * ``detected_zones`` — one row per zone (stable id = type+direction+formation
    time), carrying its live lifecycle (status, tested, mitigation, fill level)
    and whether it is currently ``surfaced``. A zone that becomes consumed while
    still inside the analysis window is flipped to ``surfaced=0`` — never left
    stale-active.
  * ``structure_events`` — the append-only BOS/CHOCH journal, keyed by
    (type, event_ts) so replaying a candle twice adds nothing.

Nothing here detects: it stores what the engine + mappers already produced. The
detection rules are untouched (LB-1 discipline).
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Surfaced lifecycle states (a zone the product would show). Anything else
# (consumed / invalidated / filled) is retained for history but not surfaced.
_SURFACED_STATES = ("active", "mitigated", "partially_filled", "tested")


def zone_id(zone_type: str, direction: str, created_at: datetime) -> str:
    """Stable zone id — type + direction + formation time. A bar carries at most
    one bullish and one bearish OB and one FVG, so this is unique per zone."""
    return f"{zone_type.upper()}_{direction}_{created_at.strftime('%Y%m%d%H%M%S')}"


def _iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if isinstance(dt, datetime) else (str(dt) if dt else None)


@dataclass(frozen=True)
class StoredZone:
    zone_type: str
    direction: str
    level_high: float
    level_low: float
    created_at: Optional[datetime]
    status: str
    tested: bool
    mitigated_at: Optional[datetime]
    fill_level: Optional[float]


class StructureStore:
    SCHEMA_VERSION = 1
    DEFAULT_DB_PATH = "./data/structure.db"
    DB_PATH_ENV_VAR = "STRUCTURE_DB_PATH"

    def __init__(self, db_path: Optional[str] = None) -> None:
        self._db_path = self._resolve_db_path(db_path)
        self._lock = threading.RLock()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        logger.info("StructureStore initialised at %s", self._db_path)

    @classmethod
    def _resolve_db_path(cls, db_path: Optional[str]) -> Path:
        if db_path:
            return Path(db_path)
        env = os.environ.get(cls.DB_PATH_ENV_VAR)
        if env:
            return Path(env)
        return Path(cls.DEFAULT_DB_PATH)

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_database(self) -> None:
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)"
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS detected_zones (
                        instrument TEXT NOT NULL,
                        timeframe  TEXT NOT NULL,
                        zone_id    TEXT NOT NULL,
                        zone_type  TEXT NOT NULL,
                        direction  TEXT NOT NULL,
                        level_high REAL NOT NULL,
                        level_low  REAL NOT NULL,
                        created_at TEXT,
                        status     TEXT NOT NULL,
                        tested     INTEGER NOT NULL DEFAULT 0,
                        mitigated_at TEXT,
                        fill_level REAL,
                        surfaced   INTEGER NOT NULL DEFAULT 1,
                        updated_at TEXT,
                        PRIMARY KEY (instrument, timeframe, zone_id)
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_zones_lookup "
                    "ON detected_zones(instrument, timeframe, created_at)"
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS structure_events (
                        instrument TEXT NOT NULL,
                        timeframe  TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        direction  TEXT NOT NULL,
                        level      REAL,
                        event_ts   TEXT NOT NULL,
                        PRIMARY KEY (instrument, timeframe, event_type, event_ts)
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_events_lookup "
                    "ON structure_events(instrument, timeframe, event_ts)"
                )
                row = conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
                if row is None:
                    conn.execute(
                        "INSERT INTO schema_version (version) VALUES (?)", (self.SCHEMA_VERSION,)
                    )
                conn.commit()
            finally:
                conn.close()

    # ------------------------------------------------------------------ #
    # Write — apply the current window's snapshot
    # ------------------------------------------------------------------ #
    def apply_snapshot(
        self,
        instrument: str,
        timeframe: str,
        zones: Dict[str, List[dict]],
        events: Dict[str, List[dict]],
        *,
        window_start: Optional[datetime],
        processed_ts: Optional[datetime],
    ) -> None:
        """Persist one detection snapshot.

        ``zones`` / ``events`` are exactly what ``collect_zones`` /
        ``collect_structure_events`` returned for the current window. Surfaced
        zones are upserted; any zone that WAS surfaced, still sits inside the
        window (``created_at >= window_start``) but is absent now is flipped to
        consumed — so an invalidated/filled zone never lingers as active. A zone
        older than the window is left frozen (it aged out, it was not consumed).
        Events are upserted idempotently into the journal.
        """
        surfaced_rows = self._zone_rows(instrument, timeframe, zones, processed_ts)
        surfaced_ids = {r[2] for r in surfaced_rows}
        ws = _iso(window_start)
        with self._lock:
            conn = self._get_connection()
            try:
                conn.executemany(
                    """
                    INSERT INTO detected_zones
                        (instrument, timeframe, zone_id, zone_type, direction,
                         level_high, level_low, created_at, status, tested,
                         mitigated_at, fill_level, surfaced, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
                    ON CONFLICT(instrument, timeframe, zone_id) DO UPDATE SET
                        level_high=excluded.level_high,
                        level_low=excluded.level_low,
                        status=excluded.status,
                        tested=excluded.tested,
                        mitigated_at=excluded.mitigated_at,
                        fill_level=excluded.fill_level,
                        surfaced=1,
                        updated_at=excluded.updated_at
                    """,
                    surfaced_rows,
                )
                # Reconcile consumed zones: surfaced before, inside the window,
                # absent now → mark consumed (surfaced=0). Frozen if aged out.
                # Runs even when the new window surfaces NOTHING (all consumed).
                params: List[Any] = [instrument, timeframe]
                ts_clause = ""
                if ws is not None:
                    ts_clause = "AND created_at >= ?"
                    params.append(ws)
                notin_clause = ""
                if surfaced_ids:
                    placeholders = ",".join("?" for _ in surfaced_ids)
                    notin_clause = f"AND zone_id NOT IN ({placeholders})"
                    params.extend(surfaced_ids)
                conn.execute(
                    f"""
                    UPDATE detected_zones SET surfaced=0, status='consumed'
                    WHERE instrument=? AND timeframe=? AND surfaced=1 {ts_clause} {notin_clause}
                    """,
                    params,
                )
                # Events reconcile WITHIN the current window, exactly like zones:
                # a break detected on a partial window can be revised as more bars
                # arrive (fractal confirmation is not monotonic), so appending
                # would keep transient events the full recompute discards. Replace
                # the in-window slice with the current snapshot; events that have
                # aged out below the window stay frozen — by then they were
                # computed with full forward context, so they are the confirmed
                # journal. (This is what makes the journal == a full recompute.)
                if ws is not None:
                    conn.execute(
                        "DELETE FROM structure_events WHERE instrument=? AND timeframe=? "
                        "AND event_ts >= ?",
                        (instrument, timeframe, ws),
                    )
                else:
                    conn.execute(
                        "DELETE FROM structure_events WHERE instrument=? AND timeframe=?",
                        (instrument, timeframe),
                    )
                event_rows = self._event_rows(instrument, timeframe, events)
                if event_rows:
                    conn.executemany(
                        """
                        INSERT INTO structure_events
                            (instrument, timeframe, event_type, direction, level, event_ts)
                        VALUES (?, ?, ?, ?, ?, ?)
                        ON CONFLICT(instrument, timeframe, event_type, event_ts) DO UPDATE SET
                            direction=excluded.direction, level=excluded.level
                        """,
                        event_rows,
                    )
                conn.commit()
            finally:
                conn.close()

    @staticmethod
    def _zone_rows(instrument, timeframe, zones, processed_ts):
        rows = []
        updated = _iso(processed_ts)
        for zone_type, key in (("ob", "order_blocks"), ("fvg", "fair_value_gaps")):
            for z in zones.get(key, []) or []:
                created = z.get("created_at")
                if created is None:
                    continue
                zid = zone_id(zone_type, z["direction"], created)
                rows.append((
                    instrument, timeframe, zid, zone_type, z["direction"],
                    float(z["level_high"]), float(z["level_low"]), _iso(created),
                    str(z.get("status", "active")), 1 if z.get("tested") else 0,
                    _iso(z.get("mitigated_at")),
                    float(z["fill_level"]) if z.get("fill_level") is not None else None,
                    updated,
                ))
        return rows

    @staticmethod
    def _event_rows(instrument, timeframe, events):
        rows = []
        for event_type, key in (("bos", "bos_events"), ("choch", "choch_events")):
            for e in events.get(key, []) or []:
                ts = e.get("broken_at")
                if ts is None:
                    continue
                rows.append((
                    instrument, timeframe, event_type, e["direction"],
                    float(e["level"]) if e.get("level") is not None else None, _iso(ts),
                ))
        return rows

    # ------------------------------------------------------------------ #
    # Read
    # ------------------------------------------------------------------ #
    def get_current_zones(self, instrument: str, timeframe: str) -> List[StoredZone]:
        with self._lock:
            conn = self._get_connection()
            try:
                rows = conn.execute(
                    "SELECT * FROM detected_zones WHERE instrument=? AND timeframe=? "
                    "AND surfaced=1 ORDER BY created_at",
                    (instrument, timeframe),
                ).fetchall()
            finally:
                conn.close()
        return [self._to_zone(r) for r in rows]

    def get_event_journal(
        self, instrument: str, timeframe: str, event_type: Optional[str] = None
    ) -> List[dict]:
        q = "SELECT event_type, direction, level, event_ts FROM structure_events " \
            "WHERE instrument=? AND timeframe=?"
        params: List[Any] = [instrument, timeframe]
        if event_type:
            q += " AND event_type=?"
            params.append(event_type)
        q += " ORDER BY event_ts"
        with self._lock:
            conn = self._get_connection()
            try:
                rows = conn.execute(q, params).fetchall()
            finally:
                conn.close()
        return [dict(r) for r in rows]

    def get_zones_in_window(
        self,
        instrument: str,
        timeframe: str,
        *,
        time_from: Optional[datetime] = None,
        price_low: Optional[float] = None,
        price_high: Optional[float] = None,
        surfaced_only: bool = True,
    ) -> List[StoredZone]:
        """Zones intersecting a visible window (LB-1 window-bounded loading).

        A zone intersects the price band when [level_low, level_high] overlaps
        [price_low, price_high]; ``time_from`` keeps zones formed at/after it.
        """
        q = "SELECT * FROM detected_zones WHERE instrument=? AND timeframe=?"
        params: List[Any] = [instrument, timeframe]
        if surfaced_only:
            q += " AND surfaced=1"
        if time_from is not None:
            q += " AND (created_at IS NULL OR created_at >= ?)"
            params.append(_iso(time_from))
        if price_low is not None and price_high is not None:
            q += " AND level_low <= ? AND level_high >= ?"
            params.extend([price_high, price_low])
        q += " ORDER BY created_at"
        with self._lock:
            conn = self._get_connection()
            try:
                rows = conn.execute(q, params).fetchall()
            finally:
                conn.close()
        return [self._to_zone(r) for r in rows]

    def get_events_in_window(
        self,
        instrument: str,
        timeframe: str,
        *,
        time_from: Optional[datetime] = None,
    ) -> List[dict]:
        q = "SELECT event_type, direction, level, event_ts FROM structure_events " \
            "WHERE instrument=? AND timeframe=?"
        params: List[Any] = [instrument, timeframe]
        if time_from is not None:
            q += " AND event_ts >= ?"
            params.append(_iso(time_from))
        q += " ORDER BY event_ts"
        with self._lock:
            conn = self._get_connection()
            try:
                rows = conn.execute(q, params).fetchall()
            finally:
                conn.close()
        return [dict(r) for r in rows]

    def count_zones(self, instrument: str, timeframe: str, surfaced_only: bool = True) -> int:
        q = "SELECT COUNT(*) AS n FROM detected_zones WHERE instrument=? AND timeframe=?"
        if surfaced_only:
            q += " AND surfaced=1"
        with self._lock:
            conn = self._get_connection()
            try:
                return int(conn.execute(q, (instrument, timeframe)).fetchone()["n"])
            finally:
                conn.close()

    @staticmethod
    def _to_zone(row: sqlite3.Row) -> StoredZone:
        def _dt(v):
            return datetime.fromisoformat(v) if v else None
        return StoredZone(
            zone_type=row["zone_type"],
            direction=row["direction"],
            level_high=row["level_high"],
            level_low=row["level_low"],
            created_at=_dt(row["created_at"]),
            status=row["status"],
            tested=bool(row["tested"]),
            mitigated_at=_dt(row["mitigated_at"]),
            fill_level=row["fill_level"],
        )
