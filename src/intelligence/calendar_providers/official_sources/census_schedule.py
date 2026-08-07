"""U.S. Census Bureau release calendar (HTML) → release dates (NW-9).

Census, unlike BLS / BEA / Eurostat, publishes NO iCalendar feed — only an HTML
"list view" release calendar (the current year at ``calendar-listview.html`` and
per-year archives at ``calendar-listview-YYYY.html``). Its release dates are
needed so the publication-measures engine has PAST releases for the Census
indicators (advance retail sales, new residential construction / housing starts,
advance durable goods); without them those pages show no engine measures, since
the curated schedule holds only a handful of upcoming dates.

This module fetches those pages (stdlib only, browser UA, short timeout,
graceful) and parses each calendar row into a ``(summary, 'YYYY-MM-DD')`` pair —
the SAME shape :func:`ics_feed.parse_ics` returns — so the official adapter maps
them to catalog keys with the identical ``ics_match`` keyword rules. The exact
release datetime is carried by each row's ``sorttable_customkey="YYYYMMDDHHMM"``
attribute; we take its DATE and let the catalog apply the confirmed local
publication time (exactly as the .ics path does), never inventing a time.

Pure parse: :func:`parse_census_calendar` takes text, so the pytest drives it
with a fixed HTML fixture and no network. A fetch/parse failure yields an empty
list — the adapter then falls back to the curated schedule; a source is never
erased. Public domain (17 U.S.C. §105) — not endorsed by the Census Bureau.
"""

from __future__ import annotations

import logging
import re
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_TIMEOUT_S = 12
_UA = "Mozilla/5.0 (compatible; MIA-Markets-Calendar/1.0; +https://mia-markets)"
_BASE = "https://www.census.gov/economic-indicators/calendar-listview{suffix}.html"

# How many prior YEAR archives to read in addition to the current-year page. Two
# years of monthly releases is ~24 per indicator — well past the reliability
# floor and the measures' own 24-release cap, while keeping the fetch small.
_DEFAULT_YEARS_BACK = 2

_ROW_RE = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.S | re.I)
_TD_RE = re.compile(r"<td\b[^>]*>(.*?)</td>", re.S | re.I)
_KEY_RE = re.compile(r'sorttable_customkey="(\d{8,12})"')
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _strip(cell: str) -> str:
    """Cell inner HTML → normalised plain text (tags removed, whitespace collapsed)."""
    return _WS_RE.sub(" ", _TAG_RE.sub(" ", cell)).strip()


def _key_to_date(digits: str) -> Optional[str]:
    """Reduce a ``sorttable_customkey`` (YYYYMMDD[HHMM]) to ``YYYY-MM-DD``, or None."""
    if len(digits) < 8:
        return None
    y, m, d = digits[0:4], digits[4:6], digits[6:8]
    if not ("2000" <= y <= "2100" and "01" <= m <= "12" and "01" <= d <= "31"):
        return None
    return f"{y}-{m}-{d}"


def parse_census_calendar(text: str) -> List[Tuple[str, str]]:
    """Parse a Census list-view calendar page → ``[(indicator_name, 'YYYY-MM-DD')]``.

    Each release is one ``<tr>`` whose first ``<td>`` names the indicator and
    whose date cell carries ``sorttable_customkey="YYYYMMDDHHMM"`` (the machine
    date+time). We keep the indicator name + the DATE; the time comes from the
    catalog. Rows without a parseable key or a name are skipped. Pure — no network.
    """
    out: List[Tuple[str, str]] = []
    seen: set = set()
    for row in _ROW_RE.finditer(text or ""):
        block = row.group(1)
        km = _KEY_RE.search(block)
        if not km:
            continue
        date = _key_to_date(km.group(1))
        if date is None:
            continue
        cells = _TD_RE.findall(block)
        if not cells:
            continue
        name = _strip(cells[0])
        if not name:
            continue
        pair = (name, date)
        if pair not in seen:
            seen.add(pair)
            out.append(pair)
    return out


def _fetch(url: str, timeout: int = _TIMEOUT_S) -> str:
    """Return the raw HTML, or "" on ANY failure (graceful)."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _UA, "Accept": "text/html"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (trusted official URL)
            return resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError, OSError) as exc:
        logger.warning("Census calendar fetch failed for %s: %s — falling back", url, exc)
        return ""


def census_calendar_urls(now: Optional[datetime] = None, years_back: int = _DEFAULT_YEARS_BACK) -> List[str]:
    """The current-year page (no suffix) + the prior ``years_back`` year archives.

    The suffix-less page is always the current calendar year; older years live at
    ``calendar-listview-YYYY.html``. A non-existent archive simply 404s and is
    skipped by the graceful fetch."""
    ref = now or datetime.now(timezone.utc)
    urls = [_BASE.format(suffix="")]
    for y in range(ref.year - 1, ref.year - 1 - max(0, years_back), -1):
        urls.append(_BASE.format(suffix=f"-{y}"))
    return urls


def fetch_census_calendar(
    fetch_fn: Optional[Callable[[str], str]] = None,
    *,
    now: Optional[datetime] = None,
    years_back: int = _DEFAULT_YEARS_BACK,
) -> List[Tuple[str, str]]:
    """Fetch + parse the current + prior-year Census calendar pages, deduped.

    ``fetch_fn`` is injectable (default: the live HTTP GET). Any page that fails
    contributes nothing. Returns ``[(indicator_name, 'YYYY-MM-DD')]``."""
    fetch = fetch_fn or _fetch
    out: List[Tuple[str, str]] = []
    seen: set = set()
    for url in census_calendar_urls(now=now, years_back=years_back):
        for pair in parse_census_calendar(fetch(url)):
            if pair not in seen:
                seen.add(pair)
                out.append(pair)
    return out


def census_date_source(
    source_key: str,
    fetch_fn: Optional[Callable[[str], str]] = None,
    *,
    now: Optional[datetime] = None,
    years_back: int = _DEFAULT_YEARS_BACK,
):
    """Live date source for Census: fetch the HTML calendar and map each row's
    indicator name to catalog keys via the same ``ics_match`` keyword rules as the
    .ics path. Empty on any failure (⇒ caller falls back to the curated schedule).

    Returns a callable ``(catalog) -> List[ReleaseInstance]`` — the exact seam the
    base official adapter expects (mirrors :func:`base_official.ics_date_source`).
    """
    # Imported lazily to avoid a cycle (base_official imports this module's source).
    from src.intelligence.calendar_providers.official_sources.base_official import (
        CatalogEvent,
        ReleaseInstance,
        match_ics_keys,
    )

    def _source(catalog: Dict[str, "CatalogEvent"]) -> List["ReleaseInstance"]:
        mine = {k: c for k, c in catalog.items() if c.source == source_key}
        if not mine or not any(c.ics_match for c in mine.values()):
            return []
        rows = fetch_census_calendar(fetch_fn, now=now, years_back=years_back)
        if not rows:
            return []
        seen: set = set()
        out: List["ReleaseInstance"] = []
        for name, day in rows:
            for key in match_ics_keys(name, mine):
                if (key, day) not in seen:
                    seen.add((key, day))
                    out.append(ReleaseInstance(event_key=key, release_date=day))
        return out

    return _source


__all__ = [
    "parse_census_calendar",
    "fetch_census_calendar",
    "census_calendar_urls",
    "census_date_source",
]
