"""ForexFactory news pipeline — fills the MarketReading ``events`` block (Chantier 3).

Architecture doc §2.4 ("impact contextualisé") + §3.3 ("pipeline news") + §3.6
("News upcoming/just_published recalculées en continu").

Source: the **stable public ForexFactory JSON feed** that MT4/MT5 EAs have
relied on for 10+ years (no fragile HTML scraping):
    https://nfs.faireconomy.media/ff_calendar_thisweek.json
    https://nfs.faireconomy.media/ff_calendar_nextweek.json
The feed already provides ``country`` (currency code) and ``impact``
(high/medium/low), so no keyword classification is needed.

Niveau 1.5 strict — non négociable:
  ``potential_effect_description`` is **template-generated** (no LLM), purely
  descriptive/factual, never directive. It states *what the event is* and
  *that movement is possible*, never *what to do*. Every generated string is
  guarded by ``contains_forbidden_tokens`` (reused from market_reading_mappers).

Cost economy (architecture doc §3.3):
  - low/holiday/non-economic impact dropped at normalization
  - feed fetched at most once per ``ttl_seconds`` (default 120s), cached in
    SQLite (``NewsCacheStore``); network errors keep the stale cache (graceful)

Testability: the network fetch is injectable via ``fetch_fn`` and the wall
clock via ``clock`` / per-call ``now``, so unit tests need no network access.
"""

from __future__ import annotations

import hashlib
import json
import logging
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, List, Optional, Sequence

from src.intelligence.market_reading_mappers import contains_forbidden_tokens
from src.intelligence.market_reading_schema import (
    ImpactLevel,
    NewsJustPublished,
    NewsUpcoming,
    SurpriseDirection,
)
from src.storage.news_cache_store import NewsCacheStore, NormalizedNewsEvent

logger = logging.getLogger(__name__)

FF_URLS = (
    "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
    "https://nfs.faireconomy.media/ff_calendar_nextweek.json",
)

# FF impact → schema impact. low / holiday / non-economic are intentionally
# absent (dropped at normalization for economy, per the brief).
_FF_IMPACT_MAP = {"high": "high", "medium": "medium"}

_IMPACT_FR = {"high": "élevé", "medium": "moyen", "low": "faible"}

_NUMERIC_SUFFIX = {
    "k": 1e3, "K": 1e3,
    "m": 1e6, "M": 1e6,
    "b": 1e9, "B": 1e9,
    "t": 1e12, "T": 1e12,
}

# ForexFactory's default timezone is US Eastern; the feed emits offsets when
# known, naive timestamps otherwise. We assume ET for naive values — DST-aware
# (a fixed UTC-5 is wrong half the year: EDT is UTC-4; audit 2026-06-12 §2.6).
# Fallback to fixed EST only if the IANA database is unavailable.
try:
    from zoneinfo import ZoneInfo

    _FF_DEFAULT_TZ: Any = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover — missing tzdata on minimal installs
    _FF_DEFAULT_TZ = timezone(timedelta(hours=-5))


# ===================================================================== #
# Normalization helpers (pure functions — no I/O)
# ===================================================================== #
def _parse_ff_datetime(date_str: str) -> Optional[datetime]:
    """Parse a FF ISO timestamp into aware UTC. Returns None if malformed."""
    try:
        dt = datetime.fromisoformat(date_str)
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_FF_DEFAULT_TZ)
    return dt.astimezone(timezone.utc)


def _parse_numeric(val: Any) -> Optional[float]:
    """Parse a FF numeric field ('180K', '3.2%', '-0.1', '') leniently → float|None."""
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    neg = s.startswith("-")
    s = s.lstrip("+-").replace(",", "").replace("%", "").strip()
    if not s:
        return None
    mult = 1.0
    if s[-1] in _NUMERIC_SUFFIX:
        mult = _NUMERIC_SUFFIX[s[-1]]
        s = s[:-1]
    try:
        value = float(s) * mult
    except ValueError:
        return None
    return -value if neg else value


def _event_id(title: str, currency: str, scheduled_at: datetime) -> str:
    raw = f"{title}|{currency}|{scheduled_at.isoformat()}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def normalize_ff_event(ev: dict) -> Optional[NormalizedNewsEvent]:
    """Convert one raw FF JSON event into a NormalizedNewsEvent.

    Returns None for malformed rows OR rows whose impact is not medium/high
    (low / holiday / non-economic are dropped for economy).
    """
    title = (ev.get("title") or "").strip()
    country = (ev.get("country") or "").strip().upper()
    date_str = ev.get("date") or ""
    if not (title and country and date_str):
        return None

    impact = _FF_IMPACT_MAP.get((ev.get("impact") or "").lower())
    if impact is None:
        return None

    scheduled_at = _parse_ff_datetime(date_str)
    if scheduled_at is None:
        return None

    return NormalizedNewsEvent(
        event_id=_event_id(title, country, scheduled_at),
        event=title,
        currency=country,
        impact=impact,
        scheduled_at=scheduled_at,
        actual=_parse_numeric(ev.get("actual")),
        forecast=_parse_numeric(ev.get("forecast")),
        previous=_parse_numeric(ev.get("previous")),
    )


def _surprise_direction(
    actual: Optional[float], forecast: Optional[float]
) -> Optional[SurpriseDirection]:
    """Numeric comparison of actual vs forecast — neutral market term, no judgement."""
    if actual is None or forecast is None:
        return None
    if actual > forecast:
        return "beat"
    if actual < forecast:
        return "miss"
    return "in_line"


# ===================================================================== #
# Niveau 1.5 strict description templates (factual, never directive)
# ===================================================================== #
def _upcoming_effect_description(event: NormalizedNewsEvent, minutes: int) -> str:
    impact_fr = _IMPACT_FR.get(event.impact, event.impact)
    desc = (
        f"{event.event} ({event.currency}) — impact {impact_fr}, "
        f"prévu dans {minutes} min. Publication macro {event.currency} "
        f"pouvant générer du mouvement sur XAUUSD et EURUSD."
    )
    return _ensure_clean(desc, event)


def _just_published_effect_description(
    event: NormalizedNewsEvent,
    minutes_ago: int,
    surprise: Optional[SurpriseDirection],
) -> str:
    impact_fr = _IMPACT_FR.get(event.impact, event.impact)
    desc = (
        f"{event.event} ({event.currency}) — impact {impact_fr}, "
        f"publié il y a {minutes_ago} min."
    )
    if surprise == "beat":
        desc += " Résultat supérieur au consensus."
    elif surprise == "miss":
        desc += " Résultat inférieur au consensus."
    elif surprise == "in_line":
        desc += " Résultat conforme au consensus."
    return _ensure_clean(desc, event)


def _ensure_clean(desc: str, event: NormalizedNewsEvent) -> str:
    """Last-resort niveau 1.5 guard: if a token slips in, fall back to a
    minimal factual string. Templates are designed never to trip this, but
    event titles come from an external feed and could in theory contain a
    forbidden word — so we re-check the *whole* string including the title.
    """
    forbidden = contains_forbidden_tokens(desc)
    if forbidden is None:
        return desc
    logger.warning(
        "news description tripped forbidden token %r (event=%r) — using minimal fallback",
        forbidden, event.event,
    )
    impact_fr = _IMPACT_FR.get(event.impact, event.impact)
    # Minimal, title-free fallback (the title carried the offending token).
    return f"Événement macro {event.currency}, impact {impact_fr}."


# ===================================================================== #
# Default network fetcher
# ===================================================================== #
def _default_ff_fetch() -> List[dict]:
    """Fetch + concatenate the FF this-week / next-week JSON feeds."""
    rows: List[dict] = []
    for url in FF_URLS:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, json.JSONDecodeError, ValueError) as exc:
            logger.warning("FF feed fetch failed for %s: %s", url, exc)
            continue
        if isinstance(data, list):
            rows.extend(data)
    return rows


# ===================================================================== #
# Pipeline
# ===================================================================== #
class NewsPipeline:
    """Fetches, caches and maps ForexFactory news into MarketReading events.

    Lifecycle: one instance per process. Thread-safety is inherited from the
    underlying ``NewsCacheStore`` (RLock).
    """

    DEFAULT_TTL_SECONDS = 120
    DEFAULT_CURRENCIES = ("USD", "EUR")
    DEFAULT_LOOKAHEAD_MIN = 240
    DEFAULT_LOOKBACK_MIN = 60

    def __init__(
        self,
        cache_store: Optional[NewsCacheStore] = None,
        fetch_fn: Optional[Callable[[], List[dict]]] = None,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self._cache = cache_store if cache_store is not None else NewsCacheStore()
        self._fetch_fn = fetch_fn or _default_ff_fetch
        self._ttl_seconds = ttl_seconds
        self._clock = clock

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def get_upcoming(
        self,
        currency_filter: Optional[Sequence[str]] = DEFAULT_CURRENCIES,
        lookahead_minutes: int = DEFAULT_LOOKAHEAD_MIN,
        now: Optional[datetime] = None,
    ) -> List[NewsUpcoming]:
        """Upcoming medium/high-impact news in the next ``lookahead_minutes``."""
        now = self._coerce_now(now)
        self._maybe_refresh(now)
        until = now + timedelta(minutes=lookahead_minutes)
        cf = self._normalize_filter(currency_filter)

        out: List[NewsUpcoming] = []
        for ev in self._cache.get_events_between(now, until):
            if cf is not None and ev.currency not in cf:
                continue
            if ev.impact not in ("medium", "high"):
                continue
            minutes = int((ev.scheduled_at - now).total_seconds() // 60)
            if minutes < 0:
                continue
            out.append(
                NewsUpcoming(
                    event=ev.event,
                    scheduled_at=ev.scheduled_at,
                    time_to_event_min=minutes,
                    impact=ev.impact,  # type: ignore[arg-type]
                    currency=ev.currency,
                    potential_effect_description=_upcoming_effect_description(ev, minutes),
                )
            )
        return out

    def get_just_published(
        self,
        currency_filter: Optional[Sequence[str]] = DEFAULT_CURRENCIES,
        lookback_minutes: int = DEFAULT_LOOKBACK_MIN,
        now: Optional[datetime] = None,
    ) -> List[NewsJustPublished]:
        """Medium/high-impact news published in the last ``lookback_minutes``."""
        now = self._coerce_now(now)
        self._maybe_refresh(now)
        start = now - timedelta(minutes=lookback_minutes)
        cf = self._normalize_filter(currency_filter)

        out: List[NewsJustPublished] = []
        for ev in self._cache.get_events_between(start, now):
            if cf is not None and ev.currency not in cf:
                continue
            if ev.impact not in ("medium", "high"):
                continue
            minutes_ago = int((now - ev.scheduled_at).total_seconds() // 60)
            if minutes_ago < 0:
                continue
            surprise = _surprise_direction(ev.actual, ev.forecast)
            out.append(
                NewsJustPublished(
                    event=ev.event,
                    published_at=ev.scheduled_at,
                    actual=ev.actual,
                    forecast=ev.forecast,
                    previous=ev.previous,
                    surprise_direction=surprise,
                    currency=ev.currency,
                    impact=ev.impact,  # type: ignore[arg-type]
                    potential_effect_description=_just_published_effect_description(
                        ev, minutes_ago, surprise
                    ),
                )
            )
        return out

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _coerce_now(self, now: Optional[datetime]) -> datetime:
        ts = now if now is not None else self._clock()
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc)

    def _normalize_filter(
        self, currency_filter: Optional[Sequence[str]]
    ) -> Optional[set]:
        if currency_filter is None:
            return None
        return {c.upper() for c in currency_filter}

    def _maybe_refresh(self, now: datetime) -> None:
        """Fetch the FF feed at most once per TTL; persist into the cache.

        Network/parse errors are swallowed (logged) so a transient feed outage
        never breaks MarketReading generation — we serve the stale cache.
        """
        last = self._cache.last_fetch_at()
        if last is not None and (now - last).total_seconds() < self._ttl_seconds:
            return
        try:
            raw = self._fetch_fn()
        except Exception as exc:  # defensive — any fetcher failure
            logger.warning("news fetch_fn failed: %s — keeping cached events", exc)
            return

        events: List[NormalizedNewsEvent] = []
        for item in raw or []:
            if not isinstance(item, dict):
                continue
            norm = normalize_ff_event(item)
            if norm is not None:
                events.append(norm)
        if events:
            self._cache.upsert_events(events, fetched_at=now)
        else:
            logger.debug("news refresh produced 0 medium/high events")


__all__ = [
    "FF_URLS",
    "NewsPipeline",
    "normalize_ff_event",
]
