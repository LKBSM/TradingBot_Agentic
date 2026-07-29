"""Shared base for the per-organism official adapters (NW-1b).

An ``OfficialSourceProvider`` turns the versioned, auditable catalog
(``config/calendar_catalog.json``) + a dated release schedule into neutral
``ProviderEvent``s for ONE source. Metadata (series code, organism, unit,
publisher timezone, licence, confirmed publication time) comes from the catalog;
the dated instances (and any published values / revisions) come from a
``date_source`` callable.

The default ``date_source`` reads ``config/calendar_schedule.json`` — a
versioned file, one auditable row per release (date + official schedule URL +
last-verified). It ships EMPTY, so by default the source is honestly empty
rather than showing fabricated dates. A live network feed (organism ``.ics`` /
data API) can be injected without touching anything else: the seam is one
callable, and no source-specific format ever crosses ``ProviderEvent``.

Values are shown AS PUBLISHED — never converted, never rounded (ECB policy
forbids modifying statistics). A release whose publication TIME is not confirmed
(``time_confirmed=false`` in the catalog) is marked and excluded from amplitude
measures; its time is never approximated.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional
from zoneinfo import ZoneInfo

from src.intelligence.calendar_providers.base import (
    CalendarProvider,
    ProviderAttribution,
    ProviderEvent,
    ProviderFetch,
)

logger = logging.getLogger(__name__)

_ENV_CATALOG_PATH = "CALENDAR_CATALOG_PATH"
_ENV_SCHEDULE_PATH = "CALENDAR_SCHEDULE_PATH"


def _config_dir() -> Path:
    return Path(__file__).resolve().parents[4] / "config"


def _default_catalog_path() -> Path:
    return Path(os.environ.get(_ENV_CATALOG_PATH) or (_config_dir() / "calendar_catalog.json"))


def _default_schedule_path() -> Path:
    return Path(os.environ.get(_ENV_SCHEDULE_PATH) or (_config_dir() / "calendar_schedule.json"))


@dataclass(frozen=True)
class CatalogEvent:
    """One recurring publication, as declared in the catalog (no dates)."""

    key: str
    source: str
    event: str
    currency: str
    series_code: Optional[str]
    value_unit: Optional[str]
    periodicity: Optional[str]
    source_timezone: Optional[str]
    release_time_local: Optional[str]
    time_confirmed: bool
    organism: Optional[str]
    license_label: Optional[str]


@dataclass(frozen=True)
class ReleaseInstance:
    """A single dated occurrence of a catalog event, plus any published values.

    Dates/values come from the schedule config or a live feed; the base joins
    them with the catalog metadata. Values are AS PUBLISHED — no conversion.
    """

    event_key: str
    release_date: str            # ISO date "YYYY-MM-DD" in the publisher timezone
    actual: Optional[float] = None
    actual_initial: Optional[float] = None
    previous: Optional[float] = None
    revised: bool = False
    revised_at: Optional[datetime] = None


def load_catalog(path: Optional[Path] = None) -> Dict[str, CatalogEvent]:
    """Load the catalog as ``{event_key: CatalogEvent}``. On any error → empty
    (⇒ nothing emitted), the honest failure mode."""
    p = path or _default_catalog_path()
    try:
        raw = json.loads(Path(p).read_text(encoding="utf-8"))
        sources = raw.get("sources", {})
        out: Dict[str, CatalogEvent] = {}
        for ev in raw.get("events", []):
            src = str(ev.get("source", ""))
            smeta = sources.get(src, {}) if isinstance(sources, dict) else {}
            out[str(ev["key"])] = CatalogEvent(
                key=str(ev["key"]),
                source=src,
                event=str(ev.get("event", "")),
                currency=str(ev.get("currency", "")).upper(),
                series_code=ev.get("series_code"),
                value_unit=ev.get("value_unit"),
                periodicity=ev.get("periodicity"),
                source_timezone=ev.get("source_timezone"),
                release_time_local=ev.get("release_time_local"),
                time_confirmed=bool(ev.get("time_confirmed", False)),
                organism=smeta.get("organism"),
                license_label=smeta.get("license_label"),
            )
        return out
    except (OSError, ValueError, TypeError, KeyError) as exc:
        logger.error("failed to load calendar catalog (%s): %s — empty", p, exc)
        return {}


def load_schedule(path: Optional[Path] = None) -> List[ReleaseInstance]:
    """Load the dated release schedule. Ships empty; on any error → empty."""
    p = path or _default_schedule_path()
    try:
        text = Path(p).read_text(encoding="utf-8")
    except OSError:
        return []
    try:
        raw = json.loads(text)
        out: List[ReleaseInstance] = []
        for r in raw.get("releases", []):
            if not isinstance(r, dict) or not r.get("event_key") or not r.get("date"):
                continue
            revised_at = r.get("revised_at")
            out.append(
                ReleaseInstance(
                    event_key=str(r["event_key"]),
                    release_date=str(r["date"]),
                    actual=r.get("actual"),
                    actual_initial=r.get("actual_initial"),
                    previous=r.get("previous"),
                    revised=bool(r.get("revised", False)),
                    revised_at=_parse_dt(revised_at) if revised_at else None,
                )
            )
        return out
    except (ValueError, TypeError, KeyError) as exc:
        logger.error("failed to load calendar schedule (%s): %s — empty", p, exc)
        return []


def _parse_dt(value: str) -> Optional[datetime]:
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def _schedule_date_source(source_key: str) -> Callable[[Dict[str, CatalogEvent]], List[ReleaseInstance]]:
    """Default date source: the schedule config, filtered to this source's keys."""

    def _source(catalog: Dict[str, CatalogEvent]) -> List[ReleaseInstance]:
        mine = {k for k, c in catalog.items() if c.source == source_key}
        return [r for r in load_schedule() if r.event_key in mine]

    return _source


class OfficialSourceProvider(CalendarProvider):
    """Base official adapter for ONE source. Subclasses set ``SOURCE`` and may
    inject a live ``date_source``; everything else is shared."""

    SOURCE: str = "official"
    POLICY_URL: str = ""

    def __init__(
        self,
        catalog: Optional[Dict[str, CatalogEvent]] = None,
        date_source: Optional[Callable[[Dict[str, CatalogEvent]], List[ReleaseInstance]]] = None,
    ) -> None:
        self._catalog = catalog if catalog is not None else load_catalog()
        self._date_source = date_source or _schedule_date_source(self.SOURCE)

    @property
    def source_name(self) -> str:
        return self.SOURCE

    def _my_events(self) -> Dict[str, CatalogEvent]:
        return {k: c for k, c in self._catalog.items() if c.source == self.SOURCE}

    def attributions(self) -> List[ProviderAttribution]:
        for c in self._my_events().values():
            if c.organism and c.license_label:
                return [
                    ProviderAttribution(
                        source=self.SOURCE,
                        organism=c.organism,
                        license_label=c.license_label,
                        policy_url=self.POLICY_URL,
                    )
                ]
        return []

    def _scheduled_at(self, cat: CatalogEvent, release_date: str) -> Optional[datetime]:
        """Combine the release DATE with the catalog's confirmed local TIME in
        the publisher timezone, DST-safe, → UTC. Missing tz/time → None (dropped)."""
        if not cat.source_timezone or not cat.release_time_local:
            return None
        try:
            y, m, d = (int(x) for x in release_date.split("-"))
            hh, mm = (int(x) for x in cat.release_time_local.split(":"))
            local = datetime(y, m, d, tzinfo=ZoneInfo(cat.source_timezone)).replace(
                hour=hh, minute=mm, second=0, microsecond=0
            )
            return local.astimezone(timezone.utc)
        except (ValueError, KeyError, TypeError):
            return None

    def fetch(self) -> ProviderFetch:
        mine = self._my_events()
        if not mine:
            return ProviderFetch(events=[])
        try:
            instances = self._date_source(self._catalog)
        except Exception as exc:  # defensive — a feed failure is graceful
            logger.warning("%s date source failed: %s — empty fetch", self.SOURCE, exc)
            return ProviderFetch(events=[])

        events: List[ProviderEvent] = []
        for inst in instances:
            cat = mine.get(inst.event_key)
            if cat is None:
                continue
            scheduled_at = self._scheduled_at(cat, inst.release_date)
            if scheduled_at is None:
                continue
            events.append(
                ProviderEvent(
                    source=self.SOURCE,
                    provider_ref=f"{inst.event_key}:{inst.release_date}",
                    series_code=cat.series_code,
                    event=cat.event,
                    currency=cat.currency,
                    scheduled_at=scheduled_at,
                    source_timezone=cat.source_timezone,
                    time_confirmed=cat.time_confirmed,
                    organism=cat.organism,
                    periodicity=cat.periodicity,
                    value_unit=cat.value_unit,
                    actual=inst.actual,
                    actual_initial=inst.actual_initial,
                    previous=inst.previous,
                    revised=inst.revised,
                    revised_at=inst.revised_at,
                    license_label=cat.license_label,
                )
            )
        if not events:
            return ProviderFetch(events=[])
        starts = [e.scheduled_at for e in events]
        return ProviderFetch(events=events, coverage_start=min(starts), coverage_end=max(starts))


__all__ = [
    "OfficialSourceProvider",
    "CatalogEvent",
    "ReleaseInstance",
    "load_catalog",
    "load_schedule",
]
