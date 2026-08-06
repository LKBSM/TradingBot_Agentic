"""BLS value fetcher — public API v2, free registration key (NW-1c §3A).

Fetches the latest published value (+ previous) for a BLS series by its stable
series id (``CUUR0000SA0``, ``CES0000000001``, ``WPSFD4``…). Registers only when
``BLS_API_KEY`` is set — absent the key it is not wired and the event stays
honestly ``unfetched`` (never a fabricated number). Values AS PUBLISHED.

The API returns the series' recent data points most-recent-first; we take the
two latest by (year, period). Public domain (17 U.S.C. §105).
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import List, Optional

from src.intelligence.calendar_providers.values.base_value import (
    SeriesPoint,
    ValueFetcher,
    ValuePoint,
)

logger = logging.getLogger(__name__)

_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
_UA = "Mozilla/5.0 (compatible; MIA-Markets-Calendar/1.0; +https://mia-markets)"
_TIMEOUT_S = 12


class BLSValueFetcher(ValueFetcher):
    def __init__(self, api_key: Optional[str] = None, http_post=None) -> None:
        self._key = api_key if api_key is not None else os.environ.get("BLS_API_KEY", "")
        self._post = http_post or _http_post

    def fetch(self, series_code: str) -> Optional[ValuePoint]:
        if not self._key:
            return None
        payload = json.dumps({
            "seriesid": [series_code],
            "registrationkey": self._key,
            "latest": True,
        })
        text = self._post(_URL, payload)
        if not text:
            return None
        obs = _parse_bls(text, series_code)
        if not obs:
            return None
        actual = obs[0]
        previous = obs[1] if len(obs) >= 2 else None
        return ValuePoint(actual=actual, previous=previous)

    def fetch_series(
        self, series_code: str, limit: int = 12, kind: str = "level"
    ) -> List[SeriesPoint]:
        """Last ``limit`` published MONTHLY observations (oldest→newest) with their
        reference-period labels ("YYYY-MM"), each AS PUBLISHED. Returns [] on any
        failure/absence (→ no curve), and only when a key is present — absent the
        key the source is not even wired. Never raises, never fabricates.

        ``kind``:
          · "level" (default) — the index level / count, as the series stores it.
          · "yoy_percent" — the series' OWN 12-month percent change, i.e. the
            headline inflation figure BLS publishes alongside the index. It is read
            from the API's ``calculations`` block (``pct_changes["12"]``), NOT
            recomputed here, so it remains a published value. Months without a
            12-month calculation (e.g. the 2025 appropriations-lapse gap) simply
            have no point — never fabricated.

        Unlike ``fetch`` (``latest``), a series needs a year range: we ask for the
        span that guarantees ``limit`` monthly points and keep the last ``limit``.
        The API's "annual average" rows (period ``M13``) and any non-monthly period
        are skipped — only true reference months form the curve.
        """
        if not self._key or limit < 1:
            return []
        want_yoy = kind == "yoy_percent"
        end_year = datetime.now(timezone.utc).year
        # A month-span of two full calendar years covers 12 points with margin,
        # tolerating a publication lag at the year boundary. For the 12-month
        # percent change the API needs the prior year in-window to compute it, so
        # widen by one extra year when yoy is requested.
        start_year = end_year - ((limit // 12) + (3 if want_yoy else 2))
        payload_obj = {
            "seriesid": [series_code],
            "registrationkey": self._key,
            "startyear": str(start_year),
            "endyear": str(end_year),
        }
        if want_yoy:
            payload_obj["calculations"] = True
        text = self._post(_URL, json.dumps(payload_obj))
        if not text:
            return []
        points = _parse_bls_series(text, yoy_percent=want_yoy)
        return points[-int(limit):] if len(points) > limit else points


def _http_post(url: str, body: str) -> str:
    try:
        req = urllib.request.Request(
            url, data=body.encode("utf-8"),
            headers={"User-Agent": _UA, "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:  # noqa: S310
            return resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError, OSError) as exc:
        logger.warning("BLS value fetch failed: %s", exc)
        return ""


def _parse_bls(text: str, series_code: str) -> List[float]:
    """Return the series' data values, most-recent-first, or [] on mismatch."""
    try:
        data = json.loads(text)
        if str(data.get("status", "")).upper() not in ("REQUEST_SUCCEEDED", ""):
            return []
        series = ((data.get("Results") or {}).get("series") or [])
        if not series:
            return []
        points = series[0].get("data") or []
        # BLS returns most-recent-first; keep that order, sorted defensively.
        def _key(p):
            return (str(p.get("year", "")), str(p.get("period", "")))
        points = sorted(points, key=_key, reverse=True)
        out = []
        for p in points:
            try:
                out.append(float(str(p.get("value")).replace(",", "")))
            except (TypeError, ValueError):
                continue
        return out
    except (ValueError, TypeError, KeyError):
        return []


def _parse_bls_series(text: str, yoy_percent: bool = False) -> List[SeriesPoint]:
    """Extract MONTHLY (period, value) observations in CHRONOLOGICAL order from a
    BLS v2 message. Period ``M06`` of year ``2026`` → label ``"2026-06"``. Rows
    that are not a reference month (``M13`` annual average, quarterly/semi-annual)
    are dropped. Returns [] on any shape mismatch — never fabricates.

    When ``yoy_percent`` is set, the point value is the API's own 12-month percent
    change (``calculations.pct_changes["12"]``), not the level; months without that
    calculation are skipped (no point), never fabricated."""
    try:
        data = json.loads(text)
        if str(data.get("status", "")).upper() not in ("REQUEST_SUCCEEDED", ""):
            return []
        series = ((data.get("Results") or {}).get("series") or [])
        if not series:
            return []
        points = series[0].get("data") or []
        out: List[SeriesPoint] = []
        for p in points:
            period = str(p.get("period", ""))
            if not (period.startswith("M") and period[1:].isdigit()):
                continue
            month = int(period[1:])
            if not (1 <= month <= 12):  # drops M13 (annual average)
                continue
            year = str(p.get("year", ""))
            if not year.isdigit():
                continue
            if yoy_percent:
                raw = (
                    ((p.get("calculations") or {}).get("pct_changes") or {}).get("12")
                )
                if raw is None:
                    continue  # no published 12-month change for this month → no point
            else:
                raw = p.get("value")
            try:
                value = float(str(raw).replace(",", ""))
            except (TypeError, ValueError):
                continue
            out.append(SeriesPoint(period=f"{year}-{month:02d}", value=value))
        # BLS returns most-recent-first; sort chronological by the label itself.
        out.sort(key=lambda sp: sp.period)
        return out
    except (ValueError, TypeError, KeyError):
        return []


__all__ = ["BLSValueFetcher"]
