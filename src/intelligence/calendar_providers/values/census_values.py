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
import re
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
# A plain, current browser UA. The Census API host rejects some non-browser
# "bot"-style User-Agents with a 403 (which read as an empty curve in prod, NW-9);
# a standard browser UA reaches the data path. Only used for the Census value API.
_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
_TIMEOUT_S = 15


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
        raw = api_key if api_key is not None else os.environ.get("CENSUS_API_KEY", "")
        # Strip surrounding whitespace/newline — a env-var value pasted with a
        # trailing newline makes Census reject the key ("Invalid Key") even though
        # it is set, which read as an empty curve (NW-9 prod diagnosis).
        self._key = (raw or "").strip()
        self._get = http_get or _http_get

    def _fetch_points(self, spec: "_Series") -> List[Tuple[str, float]]:
        if not self._key:
            return []
        params = [
            # ``time`` is a datetime PREDICATE, not a gettable variable — asking for
            # it in ``get`` returns HTTP 400 "unknown variable 'time'" (NW-9 prod).
            # The period is read from ``time_slot_date`` / ``time_slot_name``; the
            # ``time`` filter below still bounds the range.
            ("get", "cell_value,time_slot_date,time_slot_name"),
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
        # Key shape (never the key itself) — a length of 0, or surrounding
        # whitespace, or a non-hex/non-ascii char, is a common env-var cause.
        out["key_diag"] = {
            "length": len(self._key),
            "is_ascii": self._key.isascii(),
            "hex_40": len(self._key) == 40 and all(c in "0123456789abcdefABCDEF" for c in self._key),
        }
        if not self._key:
            out["attempt"] = {"skipped": "no CENSUS_API_KEY in env"}
            return out
        # Detailed single request that NAMES the failure (status / redirect /
        # exception) instead of swallowing it — this is what pinpoints a
        # Render-side 403 / redirect / timeout / rejected key.
        out["raw_probe"] = self._raw_http_probe(spec)
        pts = self._fetch_points(spec)
        out["attempt"] = {
            "point_count": len(pts),
            "sample": [{"period": p, "value": v} for p, v in pts[-3:]],
        }
        out["program_cells"] = self._probe_program_cells(spec, probe_month)
        return out

    def _raw_http_probe(self, spec: "_Series") -> dict:
        """One detailed request that does NOT swallow the error and does NOT follow
        redirects, so the exact Census response is named: HTTP 200 + data, a 302 to
        "Missing Key"/"Invalid Key", a 403 (blocked UA/IP), or a timeout/network
        exception. The API key is redacted from the echoed URL."""
        import urllib.error
        import urllib.request

        params = [
            ("get", "cell_value,time_slot_date,time_slot_name"),
            ("category_code", spec.category_code),
            ("data_type_code", spec.data_type_code),
            ("seasonally_adj", spec.seasonally_adj),
            ("for", "us"),
            ("time", "from 2025-06"),
            ("key", self._key),
        ]
        url = f"{_BASE}/{spec.program}?{urllib.parse.urlencode(params)}"
        redacted = url.replace(self._key, "<KEY>") if self._key else url
        result: dict = {"url": redacted}

        class _NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *a, **k):  # noqa: ANN001 - stdlib signature
                return None  # surface the 3xx as an HTTPError instead of following

        opener = urllib.request.build_opener(_NoRedirect)
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": _UA, "Accept": "application/json"}
            )
            with opener.open(req, timeout=_TIMEOUT_S) as resp:  # noqa: S310 (trusted)
                body = resp.read().decode("utf-8", errors="replace")
                result.update(status=getattr(resp, "status", 200), body_head=body[:200])
        except urllib.error.HTTPError as exc:
            head = ""
            try:
                head = exc.read().decode("utf-8", errors="replace")[:200]
            except Exception:  # pragma: no cover - defensive
                pass
            loc = ""
            try:
                loc = exc.headers.get("Location", "") if exc.headers else ""
            except Exception:  # pragma: no cover - defensive
                pass
            title = ""
            m = re.search(r"<title>([^<]+)</title>", head, re.I)
            if m:
                title = m.group(1).strip()
            result.update(
                status=exc.code,
                error=f"HTTPError {exc.code} {exc.reason}",
                redirect_location=loc,
                page_title=title,
                body_head=head,
            )
        except Exception as exc:  # timeout / URLError / SSL / DNS
            result.update(error=f"{type(exc).__name__}: {exc}")
        return result

    def _probe_program_cells(self, spec: "_Series", probe_month: Optional[str]) -> dict:
        """Every (category_code, data_type_code) cell of the program for ONE recent
        month, with its value — so the correct headline cell is picked by evidence.
        Returns {"month", "rows":[{category_code,data_type_code,value}], "error"?}."""
        since = probe_month or "2025-01"
        params = [
            ("get", "cell_value,category_code,data_type_code,time_slot_date,time_slot_name"),
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
        except ValueError:
            return {"error": "missing expected columns", "header": header}
        # Keep only the latest period present, and one row per (cat, dtype) cell.
        periods = [p for p in (_period_of(header, r) for r in data[1:]) if p]
        latest = max(periods, default=None)
        rows: list = []
        for r in data[1:]:
            if len(r) <= max(vi, ci, di) or _period_of(header, r) != latest:
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


_MON3 = {
    m: i
    for i, m in enumerate(
        ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1
    )
}


def _to_yyyymm(raw: str) -> Optional[str]:
    """Normalise a Census time-slot label to ``YYYY-MM``. Handles an ISO date
    (``2025-06-01`` / ``2025-06``) and a month name (``June 2025``). None if it
    matches none — never a fabricated period."""
    s = (raw or "").strip()
    m = re.search(r"(\d{4})-(\d{1,2})", s)
    if m and 1 <= int(m.group(2)) <= 12:
        return f"{m.group(1)}-{int(m.group(2)):02d}"
    m = re.search(r"([A-Za-z]{3,9})\.?\s+(\d{4})", s)
    if m and m.group(1)[:3].lower() in _MON3:
        return f"{m.group(2)}-{_MON3[m.group(1)[:3].lower()]:02d}"
    return None


def _period_of(header: List, row: List) -> Optional[str]:
    """The observation's ``YYYY-MM`` period, read from the first usable time column
    (``time_slot_date`` preferred, then the ``time`` predicate echo, then
    ``time_slot_name``)."""
    for col in ("time_slot_date", "time", "time_slot_name"):
        if col in header:
            i = header.index(col)
            if i < len(row):
                p = _to_yyyymm(str(row[i]))
                if p:
                    return p
    return None


def _parse_eits(text: str) -> List[Tuple[str, float]]:
    """Extract (period, value) pairs from an EITS 2-D array response. The first
    row is the header; ``cell_value`` holds the value and the period is read from a
    time-slot column (``time`` is predicate-only and not returned). Returns [] on
    any shape mismatch or a non-JSON "Missing Key" page — never fabricates."""
    try:
        data = json.loads(text)
        if not isinstance(data, list) or len(data) < 2:
            return []
        header = data[0]
        if not isinstance(header, list):
            return []
        try:
            vi = header.index("cell_value")
        except ValueError:
            return []
        out: List[Tuple[str, float]] = []
        for row in data[1:]:
            if not isinstance(row, list) or len(row) <= vi:
                continue
            period = _period_of(header, row)
            if not period:
                continue
            try:
                value = float(str(row[vi]).replace(",", "").strip())
            except (ValueError, TypeError):
                continue
            out.append((period, value))
        return out
    except (ValueError, TypeError, KeyError):
        return []


__all__ = ["CensusValueFetcher"]
