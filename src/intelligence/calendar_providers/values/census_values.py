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
# best-known and flagged for live confirmation: the read-only diagnostic
# ``CensusValueFetcher.diagnose`` (exposed at ``GET /api/publications/{key}/values/debug``)
# lists EVERY valid (category_code, data_type_code) cell of the program with its
# value, so the correct headline cell is picked by evidence against the real API
# — never guessed. Correct a code here once the diagnostic confirms it.
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

    def fetch_series(
        self, series_code: str, limit: int = 12, kind: str = "level"
    ) -> List[SeriesPoint]:  # ``kind`` unused: Census series are served as levels.
        spec = _CENSUS_SERIES.get(series_code)
        if spec is None or limit < 1:
            return []
        points = self._fetch_points(spec)
        if not points:
            return []
        points.sort(key=lambda p: p[0])
        return [SeriesPoint(period=per, value=val) for per, val in points][-int(limit):]

    # ------------------------------------------------------------------
    # Read-only diagnostic (NW-9). WHY a Census curve is empty in prod: is the
    # key wired, does the CONFIGURED cell return data, and — crucially — what are
    # the VALID (category_code, data_type_code) cells for the program, so a wrong
    # "à confirmer" code is fixed against the real API instead of guessed. The API
    # key is NEVER echoed. Needs the key (Census rejects keyless data queries).
    # ------------------------------------------------------------------
    def diagnose(self, series_code: str, probe_month: Optional[str] = None) -> dict:
        out: dict = {
            "series_code": series_code,
            "key_present": bool(self._key),
            "configured": None,
            "attempt": None,
            "program_cells": None,
        }
        spec = _CENSUS_SERIES.get(series_code)
        if spec is None:
            out["error"] = "unknown series_code"
            return out
        out["configured"] = {
            "program": spec.program,
            "category_code": spec.category_code,
            "data_type_code": spec.data_type_code,
            "seasonally_adj": spec.seasonally_adj,
        }
        if not self._key:
            out["attempt"] = {"skipped": "no CENSUS_API_KEY in env"}
            return out
        pts = self._fetch_points(spec)
        out["attempt"] = {
            "point_count": len(pts),
            "sample": [{"period": p, "value": v} for p, v in pts[-3:]],
        }
        out["program_cells"] = self._probe_program_cells(spec, probe_month)
        return out

    def _probe_program_cells(self, spec: "_Series", probe_month: Optional[str]) -> dict:
        """Every (category_code, data_type_code) cell of the program for ONE recent
        month, with its value — so the correct headline cell is picked by evidence.
        Returns {"month", "rows":[{category_code,data_type_code,value}], "error"?}."""
        since = probe_month or "2025-01"
        params = [
            ("get", "cell_value,category_code,data_type_code,time"),
            ("seasonally_adj", spec.seasonally_adj),
            ("for", "us"),
            ("time", f"from {since}"),
            ("key", self._key),
        ]
        url = f"{_BASE}/{spec.program}?{urllib.parse.urlencode(params)}"
        text = self._get(url)
        if not text:
            return {"error": "empty response (key rejected or network)"}
        try:
            data = json.loads(text)
        except (ValueError, TypeError):
            return {"error": "non-JSON response (likely a 'Missing Key' page)", "head": text[:160]}
        if not isinstance(data, list) or len(data) < 2 or not isinstance(data[0], list):
            return {"error": "unexpected shape", "head": str(data)[:160]}
        header = data[0]
        try:
            vi = header.index("cell_value")
            ci = header.index("category_code")
            di = header.index("data_type_code")
            ti = header.index("time")
        except ValueError:
            return {"error": "missing expected columns", "header": header}
        # Keep only the latest month present, and one row per (cat, dtype) cell.
        latest = max((str(r[ti]) for r in data[1:] if len(r) > ti), default=None)
        rows: list = []
        for r in data[1:]:
            if len(r) <= max(vi, ci, di, ti) or str(r[ti]) != latest:
                continue
            rows.append(
                {"category_code": str(r[ci]), "data_type_code": str(r[di]), "value": str(r[vi])}
            )
        rows.sort(key=lambda x: (x["category_code"], x["data_type_code"]))
        return {"month": latest, "cell_count": len(rows), "rows": rows[:120]}


def _http_get(url: str) -> str:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _UA, "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:  # noqa: S310 (trusted official URL)
            return resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError, OSError) as exc:
        logger.warning("Census value fetch failed for %s: %s", url, exc)
        return ""


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
