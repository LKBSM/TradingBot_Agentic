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
# A plain, current browser UA — some government API hosts reject bot-style
# User-Agents with a 403, which reads as an empty curve in prod (NW-9).
_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
_TIMEOUT_S = 15


class BLSValueFetcher(ValueFetcher):
    def __init__(self, api_key: Optional[str] = None, http_post=None) -> None:
        raw = api_key if api_key is not None else os.environ.get("BLS_API_KEY", "")
        # Strip surrounding whitespace/newline pasted into the env var (NW-9).
        self._key = (raw or "").strip()
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

        ``kind`` (NW-8) — all variations are BLS's OWN ``calculations``, read from
        the API, NEVER recomputed here, so they stay published values:
          · "level" (default) — the index level / count, as the series stores it.
          · "index_change" — an index whose HEADLINE is the 12-month % change:
            ``value`` = ``pct_changes["12"]`` (plotted), ``level`` = the index,
            ``change_mom`` = ``pct_changes["1"]``. Months without a 12-month change
            (e.g. the 2025 appropriations-lapse gap) simply have no point.
          · "count_change" — a count whose HEADLINE is the monthly ABSOLUTE change:
            ``value`` = ``net_changes["1"]`` (e.g. jobs created), ``level`` = the
            total count. Months without a 1-month change have no point.

        Unlike ``fetch`` (``latest``), a series needs a year range: we ask for the
        span that guarantees ``limit`` monthly points and keep the last ``limit``.
        The API's "annual average" rows (period ``M13``) and any non-monthly period
        are skipped — only true reference months form the curve.
        """
        if not self._key or limit < 1:
            return []
        want_calc = kind in ("index_change", "count_change")
        end_year = datetime.now(timezone.utc).year
        # Two full calendar years cover 12 points with margin. A 12-month change
        # needs the prior year in-window, so widen when a variation is requested.
        start_year = end_year - ((limit // 12) + (3 if want_calc else 2))
        payload_obj = {
            "seriesid": [series_code],
            "registrationkey": self._key,
            "startyear": str(start_year),
            "endyear": str(end_year),
        }
        if want_calc:
            payload_obj["calculations"] = True
        text = self._post(_URL, json.dumps(payload_obj))
        if not text:
            return []
        points = _parse_bls_series(text, kind=kind)
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


def _to_float(raw) -> Optional[float]:
    try:
        return float(str(raw).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _parse_bls_series(text: str, kind: str = "level") -> List[SeriesPoint]:
    """Extract MONTHLY observations in CHRONOLOGICAL order from a BLS v2 message.
    Period ``M06`` of year ``2026`` → label ``"2026-06"``. Non-monthly rows
    (``M13`` annual average, quarterly/semi-annual) are dropped. Returns [] on any
    shape mismatch — never fabricates. Variations come from the API's own
    ``calculations`` block (see ``fetch_series`` for the per-kind mapping); a month
    that lacks the headline calculation is skipped (no point)."""
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
            level = _to_float(p.get("value"))
            calc = p.get("calculations") or {}
            pct = calc.get("pct_changes") or {}
            net = calc.get("net_changes") or {}
            if kind == "index_change":
                yoy = _to_float(pct.get("12"))
                if yoy is None:
                    continue  # no published 12-month change → no point
                sp = SeriesPoint(
                    period=f"{year}-{month:02d}", value=yoy,
                    level=level, change_mom=_to_float(pct.get("1")),
                )
            elif kind == "count_change":
                mom = _to_float(net.get("1"))
                if mom is None:
                    continue  # no published 1-month change → no point
                sp = SeriesPoint(
                    period=f"{year}-{month:02d}", value=mom, level=level,
                )
            else:  # level
                if level is None:
                    continue
                sp = SeriesPoint(period=f"{year}-{month:02d}", value=level)
            out.append(sp)
        # BLS returns most-recent-first; sort chronological by the label itself.
        out.sort(key=lambda sp: sp.period)
        return out
    except (ValueError, TypeError, KeyError):
        return []


__all__ = ["BLSValueFetcher"]
