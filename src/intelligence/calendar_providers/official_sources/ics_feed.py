"""iCalendar (.ics) feed fetch + parse for the official adapters (NW-1b).

The machine-readable release calendars of BLS, BEA and Eurostat are iCalendar
feeds. This module fetches one (stdlib only, browser User-Agent, short timeout,
graceful) and parses its ``VEVENT`` blocks into ``(summary, date)`` pairs. It is
pure: no catalog knowledge, no network in tests (``parse_ics`` takes text).

Format handled (verified against the real BEA feed):
  · line folding (RFC 5545: a line beginning with SPACE/TAB continues the prior);
  · ``SUMMARY`` with escaped commas/semicolons (``\\,`` ``\\;`` ``\\n``);
  · ``DTSTART;VALUE=DATE-TIME:20250130T133000Z`` and ``DTSTART;VALUE=DATE:20260812``
    → both reduced to the calendar date ``YYYY-MM-DD`` (the adapter applies the
    catalog's confirmed local time in the publisher timezone).

A fetch failure (network, 403, timeout, parse) yields an empty list — the
adapter then falls back to the curated schedule; a source is never erased.
"""

from __future__ import annotations

import logging
import urllib.error
import urllib.request
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

_TIMEOUT_S = 10
# Some official hosts reject default urllib UAs; present a browser-like UA.
_UA = "Mozilla/5.0 (compatible; MIA-Markets-Calendar/1.0; +https://mia-markets)"


def fetch_ics(url: str, timeout: int = _TIMEOUT_S) -> str:
    """Return the raw .ics text, or "" on ANY failure (graceful)."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _UA, "Accept": "text/calendar"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (trusted official URL)
            raw = resp.read()
        return raw.decode("utf-8", errors="replace")
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError, OSError) as exc:
        logger.warning("ICS fetch failed for %s: %s — falling back", url, exc)
        return ""


def _unfold(text: str) -> List[str]:
    """RFC 5545 line unfolding: a line starting with SPACE/TAB continues the prior."""
    out: List[str] = []
    for raw in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if raw[:1] in (" ", "\t") and out:
            out[-1] += raw[1:]
        else:
            out.append(raw)
    return out


def _unescape(value: str) -> str:
    return (
        value.replace("\\n", " ").replace("\\N", " ")
        .replace("\\,", ",").replace("\\;", ";").replace("\\\\", "\\")
        .strip()
    )


def _to_date(dtstart_value: str) -> Optional[str]:
    """Reduce a DTSTART value to ``YYYY-MM-DD`` (its calendar date)."""
    digits = "".join(ch for ch in dtstart_value if ch.isdigit())
    if len(digits) < 8:
        return None
    y, m, d = digits[0:4], digits[4:6], digits[6:8]
    if not ("1900" <= y <= "2100" and "01" <= m <= "12" and "01" <= d <= "31"):
        return None
    return f"{y}-{m}-{d}"


def parse_ics(text: str) -> List[Tuple[str, str]]:
    """Parse .ics text → list of ``(summary, 'YYYY-MM-DD')`` for each VEVENT that
    has both a SUMMARY and a parseable DTSTART. Pure — no network."""
    events: List[Tuple[str, str]] = []
    in_event = False
    summary: Optional[str] = None
    dtstart: Optional[str] = None
    for line in _unfold(text):
        upper = line.upper()
        if upper.startswith("BEGIN:VEVENT"):
            in_event, summary, dtstart = True, None, None
        elif upper.startswith("END:VEVENT"):
            if summary and dtstart:
                events.append((summary, dtstart))
            in_event = False
        elif in_event and ":" in line:
            name, _, value = line.partition(":")
            name = name.split(";", 1)[0].upper()
            if name == "SUMMARY":
                summary = _unescape(value)
            elif name == "DTSTART":
                dtstart = _to_date(value)
    return events


__all__ = ["fetch_ics", "parse_ics"]
