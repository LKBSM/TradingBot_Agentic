"""BEA value fetcher — public API, free registration key (NW-4 ch.2).

Fetches the latest published value (+ previous) for a BEA NIPA series by its
stable catalog code (``NIPA-T10101`` → real-GDP % change; ``NIPA-T20804`` → PCE
price index). Registers only when ``BEA_API_KEY`` is set — absent the key it is
not wired and the event stays honestly ``unfetched`` (never a fabricated number).
Values AS PUBLISHED — no conversion, no re-rounding. Public domain (17 U.S.C. §105).

A NIPA table carries many lines; the ONE headline line for each series lives in
``_BEA_SERIES`` (auditable), keyed by catalog code. An unknown code yields
``None`` rather than a guessed line — a wrong number is worse than an absent one.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Dict, List, Optional

from src.intelligence.calendar_providers.values.base_value import (
    SeriesPoint,
    ValueFetcher,
    ValuePoint,
)

logger = logging.getLogger(__name__)

_BASE = "https://apps.bea.gov/api/data"
# A plain, current browser UA — some government API hosts reject bot-style
# User-Agents with a 403, which reads as an empty curve in prod (NW-9).
_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
_TIMEOUT_S = 15


class _Series:
    __slots__ = ("table", "frequency", "line")

    def __init__(self, table: str, frequency: str, line: int) -> None:
        self.table = table            # NIPA TableName, e.g. "T10101"
        self.frequency = frequency    # "Q" quarterly | "M" monthly | "A" annual
        self.line = line              # the ONE headline LineNumber to read


# The single headline line per catalog series (auditable — never a guess at call
# time). Real GDP is line 1 of the "% change" table; the PCE price index is line
# 1 of the PCE price-index-by-type table.
_BEA_SERIES: Dict[str, _Series] = {
    # Real GDP, % change from preceding period (Table 1.1.1), quarterly.
    "NIPA-T10101": _Series("T10101", "Q", 1),
    # PCE price index by major type (Table 2.8.4), line 1 = all PCE, monthly.
    "NIPA-T20804": _Series("T20804", "M", 1),
}


class BEAValueFetcher(ValueFetcher):
    def __init__(self, api_key: Optional[str] = None, http_get=None) -> None:
        raw = api_key if api_key is not None else os.environ.get("BEA_API_KEY", "")
        # Strip surrounding whitespace/newline pasted into the env var (NW-9).
        self._key = (raw or "").strip()
        self._get = http_get or _http_get

    def _years_param(self) -> str:
        # Two most recent calendar years cover the latest release + its previous
        # period, including a Q1/Jan value whose "previous" is in the prior year.
        y = datetime.now(timezone.utc).year
        return f"{y - 1},{y}"

    def _fetch_rows(self, spec: "_Series") -> List[dict]:
        if not self._key:
            return []
        params = {
            "UserID": self._key,
            "method": "GetData",
            "datasetname": "NIPA",
            "TableName": spec.table,
            "Frequency": spec.frequency,
            "Year": self._years_param(),
            "ResultFormat": "JSON",
        }
        url = f"{_BASE}?{urllib.parse.urlencode(params)}"
        text = self._get(url)
        if not text:
            return []
        rows = _parse_bea_data(text)
        # Keep only the headline line for this series.
        return [r for r in rows if str(r.get("LineNumber")) == str(spec.line)]

    def fetch(self, series_code: str) -> Optional[ValuePoint]:
        spec = _BEA_SERIES.get(series_code)
        if spec is None:
            return None
        rows = self._fetch_rows(spec)
        if not rows:
            return None
        rows.sort(key=lambda r: _period_key(str(r.get("TimePeriod", ""))))
        vals = [
            v for v in (_to_float(r.get("DataValue")) for r in rows) if v is not None
        ]
        if not vals:
            return None
        actual = vals[-1]
        previous = vals[-2] if len(vals) >= 2 else None
        return ValuePoint(actual=actual, previous=previous)

    def fetch_series(
        self, series_code: str, limit: int = 12, kind: str = "level"
    ) -> List[SeriesPoint]:  # ``kind`` unused: BEA series are served as levels.
        spec = _BEA_SERIES.get(series_code)
        if spec is None or limit < 1:
            return []
        rows = self._fetch_rows(spec)
        if not rows:
            return []
        rows.sort(key=lambda r: _period_key(str(r.get("TimePeriod", ""))))
        out: List[SeriesPoint] = []
        for r in rows:
            v = _to_float(r.get("DataValue"))
            period = _normalize_period(str(r.get("TimePeriod", "")))
            if v is not None and period:
                out.append(SeriesPoint(period=period, value=v))
        return out[-int(limit):]


def _http_get(url: str) -> str:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _UA, "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:  # noqa: S310 (trusted official URL)
            return resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError, OSError) as exc:
        logger.warning("BEA value fetch failed for %s: %s", url, exc)
        return ""


def _parse_bea_data(text: str) -> List[dict]:
    """Extract the Data rows from a BEA GetData JSON envelope. Returns [] on any
    shape mismatch or an API error object (never raises, never fabricates)."""
    try:
        data = json.loads(text)
        results = ((data.get("BEAAPI") or {}).get("Results") or {})
        # A GetData error is reported as {"Error": ...} — treat as no data.
        if isinstance(results, dict):
            if results.get("Error"):
                return []
            rows = results.get("Data")
        elif isinstance(results, list):
            rows = []
            for block in results:
                if isinstance(block, dict) and isinstance(block.get("Data"), list):
                    rows.extend(block["Data"])
        else:
            rows = None
        return [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else []
    except (ValueError, TypeError, KeyError):
        return []


def _to_float(value) -> Optional[float]:
    """Parse a BEA DataValue AS PUBLISHED (strip thousands commas only — no
    conversion, no rounding). Non-numeric placeholders → None."""
    if value is None:
        return None
    try:
        return float(str(value).replace(",", "").strip())
    except (ValueError, TypeError):
        return None


def _period_key(tp: str) -> tuple:
    """Chronological sort key for a BEA TimePeriod: "2026Q2", "2026M06", "2026"."""
    tp = tp.strip()
    try:
        if "Q" in tp:
            y, q = tp.split("Q")
            return (int(y), int(q))
        if "M" in tp:
            y, m = tp.split("M")
            return (int(y), int(m))
        return (int(tp), 0)
    except (ValueError, TypeError):
        return (0, 0)


def _normalize_period(tp: str) -> str:
    """BEA TimePeriod → a stable label: "2026M06"→"2026-06", "2026Q2"→"2026-Q2"."""
    tp = tp.strip()
    if "M" in tp:
        y, _, m = tp.partition("M")
        return f"{y}-{m.zfill(2)}" if y and m else ""
    if "Q" in tp:
        y, _, q = tp.partition("Q")
        return f"{y}-Q{q}" if y and q else ""
    return tp


__all__ = ["BEAValueFetcher"]
