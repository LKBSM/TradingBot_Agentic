"""Census value fetcher — EITS time-series API, free registration key (NW-4 ch.2).

Fetches the latest published value (+ previous) for a Census economic-indicator
series by its stable catalog code (``MARTS-RSAFS`` → advance retail & food-service
sales; ``RESCONST-HOUST`` → housing starts; ``ADVM3-DGORDER`` → durable-goods new
orders). Registers only when ``CENSUS_API_KEY`` is set — the EITS endpoint now
rejects keyless requests with a "Missing Key" page (verified 2026-08-01, revising
the NW-D2 assumption that Census was keyless). Absent the key it is not wired and
the event stays honestly ``unfetched``. Values AS PUBLISHED. Public domain.

Each EITS program is a cube; the ONE headline cell for each series is selected by
the dimension codes in ``_CENSUS_SERIES`` (auditable), keyed by catalog code. An
unknown code, or a response that does not reduce to a clean single monthly
series, yields ``None`` — never a guessed cell (a wrong number is worse than an
absent one). The retail (MARTS) codes are the well-documented headline; the
housing-starts and durable-goods codes are to be confirmed on the first live run
before merge.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, List, Optional, Tuple

from src.intelligence.calendar_providers.values.base_value import (
    SeriesPoint,
    ValueFetcher,
    ValuePoint,
)

logger = logging.getLogger(__name__)

_BASE = "https://api.census.gov/data/timeseries/eits"
_UA = "Mozilla/5.0 (compatible; MIA-Markets-Calendar/1.0; +https://mia-markets)"
_TIMEOUT_S = 12


class _Series:
    __slots__ = ("program", "category_code", "data_type_code", "seasonally_adj")

    def __init__(
        self, program: str, category_code: str, data_type_code: str, seasonally_adj: str = "yes"
    ) -> None:
        self.program = program
        self.category_code = category_code
        self.data_type_code = data_type_code
        self.seasonally_adj = seasonally_adj


# The single headline cell per catalog series (auditable). MARTS retail is the
# documented 44X72 / SM (sales, monthly) SA headline. RESCONST / ADVM3 codes are
# best-known and flagged for live confirmation before merge.
_CENSUS_SERIES: Dict[str, _Series] = {
    # Advance retail & food-services sales, SA, monthly ($ millions).
    "MARTS-RSAFS": _Series("marts", "44X72", "SM", "yes"),
    # New privately-owned housing units STARTED, total, SA annual rate (thousands).
    "RESCONST-HOUST": _Series("resconst", "STARTS", "TOTAL", "yes"),
    # Advance durable-goods NEW ORDERS, total, SA, monthly ($ millions).
    "ADVM3-DGORDER": _Series("advm3", "MDM", "NO", "yes"),
}


class CensusValueFetcher(ValueFetcher):
    def __init__(self, api_key: Optional[str] = None, http_get=None) -> None:
        self._key = api_key if api_key is not None else os.environ.get("CENSUS_API_KEY", "")
        self._get = http_get or _http_get

    def _fetch_points(self, spec: "_Series") -> List[Tuple[str, float]]:
        if not self._key:
            return []
        params = [
            ("get", "cell_value,time"),
            ("category_code", spec.category_code),
            ("data_type_code", spec.data_type_code),
            ("seasonally_adj", spec.seasonally_adj),
            ("for", "us"),
            ("time", "from 2024-01"),
            ("key", self._key),
        ]
        url = f"{_BASE}/{spec.program}?{urllib.parse.urlencode(params)}"
        text = self._get(url)
        if not text:
            return []
        if _is_key_error(text):
            # Census answers a rejected/unactivated key with an HTML "Invalid Key"
            # page at HTTP 200 — indistinguishable from "no data" unless we say so.
            # Log it distinctly (once per series) so a misconfigured key surfaces
            # in the operator's logs instead of a silent, permanently-unfetched event.
            logger.warning(
                "Census rejected CENSUS_API_KEY (Invalid/Missing Key) for '%s' — the emailed "
                "key must be ACTIVATED at https://api.census.gov/data/key_signup.html before it "
                "works; the event stays honestly unfetched until then.",
                spec.program,
            )
            return []
        return _parse_eits(text)

    def fetch(self, series_code: str) -> Optional[ValuePoint]:
        spec = _CENSUS_SERIES.get(series_code)
        if spec is None:
            return None
        points = self._fetch_points(spec)
        if not points:
            return None
        points.sort(key=lambda p: p[0])  # chronological by "YYYY-MM"
        vals = [v for _, v in points]
        actual = vals[-1]
        previous = vals[-2] if len(vals) >= 2 else None
        return ValuePoint(actual=actual, previous=previous)

    def fetch_series(self, series_code: str, limit: int = 12) -> List[SeriesPoint]:
        spec = _CENSUS_SERIES.get(series_code)
        if spec is None or limit < 1:
            return []
        points = self._fetch_points(spec)
        if not points:
            return []
        points.sort(key=lambda p: p[0])
        return [SeriesPoint(period=per, value=val) for per, val in points][-int(limit):]


def _http_get(url: str) -> str:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _UA, "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:  # noqa: S310 (trusted official URL)
            return resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError, OSError) as exc:
        logger.warning("Census value fetch failed for %s: %s", url, exc)
        return ""


def _is_key_error(text: str) -> bool:
    """True when the body is Census's HTML key-gate page ("Invalid Key" /
    "Missing Key") rather than a JSON array — so an unactivated/misconfigured key
    is reported distinctly instead of silently looking like an empty series. Both
    pages arrive at HTTP 200, so the status code alone cannot tell them apart."""
    head = text.lstrip()[:600].lower()
    if "<html" not in head and "<!doctype" not in head:
        return False
    return "invalid key" in head or "missing key" in head


def _parse_eits(text: str) -> List[Tuple[str, float]]:
    """Extract (period, value) pairs from an EITS 2-D array response. The first
    row is the header; ``cell_value`` and ``time`` columns give the observation.
    Returns [] on any shape mismatch or a non-JSON "Missing Key" page — never
    fabricates a period or a value."""
    try:
        data = json.loads(text)
        if not isinstance(data, list) or len(data) < 2:
            return []
        header = data[0]
        if not isinstance(header, list):
            return []
        try:
            vi = header.index("cell_value")
            ti = header.index("time")
        except ValueError:
            return []
        out: List[Tuple[str, float]] = []
        for row in data[1:]:
            if not isinstance(row, list) or len(row) <= max(vi, ti):
                continue
            period = str(row[ti]).strip()
            try:
                value = float(str(row[vi]).replace(",", "").strip())
            except (ValueError, TypeError):
                continue
            if period:
                out.append((period, value))
        return out
    except (ValueError, TypeError, KeyError):
        return []


__all__ = ["CensusValueFetcher"]
