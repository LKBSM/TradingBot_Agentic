"""FOMC meeting calendar (HTML) → rate-decision dates (NW-9).

The Federal Reserve publishes no .ics feed for FOMC meetings — only the HTML
calendar ``fomccalendars.htm`` (past + scheduled). The rate DECISION is announced
on the SECOND day of each two-day meeting at 14:00 ET; the publication-measures
engine needs those PAST decision instants to measure price behaviour around them
(without them the FOMC page shows no measures — the curated schedule holds only a
few upcoming dates).

This module fetches that page (stdlib only, browser UA, short timeout, graceful)
and parses each meeting into its decision DATE. Each year is a panel
(``YYYY FOMC Meetings``); each meeting row carries a ``fomc-meeting__month``
(possibly ``April/May`` when the two days straddle a month boundary) and a
``fomc-meeting__date`` day range (``29-30``, ``30-1``). The decision is the
SECOND day, in the SECOND month when the meeting straddles months. The time
(14:00 ET) comes from the catalog, exactly as the .ics path applies it — never
invented. Pure parse: :func:`parse_fomc_calendar` takes text, no network in
tests. Public domain (17 U.S.C. §105)."""

from __future__ import annotations

import logging
import re
import urllib.error
import urllib.request
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

_TIMEOUT_S = 12
_UA = "Mozilla/5.0 (compatible; MIA-Markets-Calendar/1.0; +https://mia-markets)"
_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"

# The catalog event this calendar dates: the rate decision (all scheduled
# meetings). Minutes (≈3 weeks later) and the dot-plot (projection meetings only)
# are distinct events on other dates and are not inferred here.
_RATE_KEY = "us_fomc_rate"

_MONTHS = {
    m: i
    for i, m in enumerate(
        [
            "January", "February", "March", "April", "May", "June", "July",
            "August", "September", "October", "November", "December",
        ],
        1,
    )
}

_YEAR_SPLIT = re.compile(r'<h4><a id="\d+">(\d{4}) FOMC Meetings</a>')
_MEETING_RE = re.compile(
    r"fomc-meeting__month[^>]*>\s*(?:<strong>)?\s*([A-Za-z]+)"
    r"(?:\s*/\s*([A-Za-z]+))?\s*(?:</strong>)?"
    r".*?fomc-meeting__date[^>]*>\s*([\d/]+)\s*-\s*([\d/]+)",
    re.S,
)


def _iso(year: int, month_name: str, day_str: str) -> Optional[str]:
    month = _MONTHS.get(month_name)
    digits = re.sub(r"\D", "", day_str)
    if month is None or not digits:
        return None
    day = int(digits)
    if not (1 <= day <= 31):
        return None
    return f"{year:04d}-{month:02d}-{day:02d}"


def parse_fomc_calendar(text: str) -> List[str]:
    """Parse the FOMC calendar page → sorted unique decision dates ``YYYY-MM-DD``.

    The decision is the second day of the meeting; a meeting that straddles two
    months (month field ``April/May``) resolves to the second month. Pure — no
    network. Junk or an unparseable row contributes nothing (never a fabricated
    date)."""
    out: set = set()
    panels = _YEAR_SPLIT.split(text or "")
    # panels = [preamble, 'YYYY', block, 'YYYY', block, ...]
    for i in range(1, len(panels) - 1, 2):
        try:
            year = int(panels[i])
        except ValueError:
            continue
        block = panels[i + 1]
        for m in _MEETING_RE.finditer(block):
            month1, month2, _d1, d2 = m.group(1), m.group(2), m.group(3), m.group(4)
            iso = _iso(year, month2 or month1, d2)
            if iso is not None:
                out.add(iso)
    return sorted(out)


def _fetch(url: str, timeout: int = _TIMEOUT_S) -> str:
    """Return the raw HTML, or "" on ANY failure (graceful)."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _UA, "Accept": "text/html"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (trusted official URL)
            return resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError, OSError) as exc:
        logger.warning("FOMC calendar fetch failed for %s: %s — falling back", url, exc)
        return ""


def fomc_date_source(
    source_key: str, fetch_fn: Optional[Callable[[str], str]] = None
):
    """Live date source for the Federal Reserve: fetch the FOMC calendar and date
    the rate decision for every scheduled meeting. Empty on any failure (⇒ caller
    falls back to the curated schedule). Returns a callable ``(catalog) ->
    List[ReleaseInstance]`` — the seam the base official adapter expects."""
    from src.intelligence.calendar_providers.official_sources.base_official import (
        CatalogEvent,
        ReleaseInstance,
    )

    fetch = fetch_fn or _fetch

    def _source(catalog: Dict[str, "CatalogEvent"]) -> List["ReleaseInstance"]:
        cat = catalog.get(_RATE_KEY)
        if cat is None or cat.source != source_key:
            return []
        dates = parse_fomc_calendar(fetch(_URL))
        return [ReleaseInstance(event_key=_RATE_KEY, release_date=d) for d in dates]

    return _source


__all__ = ["parse_fomc_calendar", "fomc_date_source"]
