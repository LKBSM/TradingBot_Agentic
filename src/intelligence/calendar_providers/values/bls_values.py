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
from typing import List, Optional

from src.intelligence.calendar_providers.values.base_value import ValueFetcher, ValuePoint

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


__all__ = ["BLSValueFetcher"]
