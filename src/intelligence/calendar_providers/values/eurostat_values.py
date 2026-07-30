"""Eurostat value fetcher — JSON-stat 2.0, no key (NW-1c §3A).

Fetches the latest published value (+ previous) for a euro-area indicator by its
stable dataset code (``prc_hicp_manr``, ``namq_10_gdp``, ``une_rt_m``). Values are
returned AS PUBLISHED. Graceful: any failure returns ``None`` (event stays
``unfetched``). CC BY 4.0 — attribution shown; the value is not modified.

A dataset needs headline dimension filters (euro area, main aggregate) to select
one series; those live in ``_DATASET_FILTERS`` (auditable), keyed by dataset code.
An unknown dataset yields ``None``.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, List, Optional

from src.intelligence.calendar_providers.values.base_value import ValueFetcher, ValuePoint

logger = logging.getLogger(__name__)

_BASE = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"
_UA = "Mozilla/5.0 (compatible; MIA-Markets-Calendar/1.0; +https://mia-markets)"
_TIMEOUT_S = 12

# Headline euro-area filters per dataset (auditable). One series is selected so
# the JSON-stat message reduces to a time vector.
_DATASET_FILTERS: Dict[str, Dict[str, str]] = {
    # HICP, annual rate of change, all-items, euro area.
    "prc_hicp_manr": {"geo": "EA20", "coicop": "CP00", "unit": "RCH_A"},
    # GDP, chain-linked % change vs previous quarter, SA, euro area.
    "namq_10_gdp": {"geo": "EA20", "na_item": "B1GQ", "unit": "CLV_PCH_PRE", "s_adj": "SCA"},
    # Unemployment rate, % of active population, SA, total, all ages, euro area.
    "une_rt_m": {"geo": "EA20", "unit": "PC_ACT", "s_adj": "SA", "sex": "T", "age": "TOTAL"},
}


class EurostatValueFetcher(ValueFetcher):
    def __init__(self, http_get=None) -> None:
        self._get = http_get or _http_get

    def fetch(self, series_code: str) -> Optional[ValuePoint]:
        filters = _DATASET_FILTERS.get(series_code)
        if filters is None:
            return None
        params = [("format", "JSON"), ("lang", "EN"), ("lastTimePeriod", "2")]
        params += list(filters.items())
        url = f"{_BASE}/{series_code}?{urllib.parse.urlencode(params)}"
        text = self._get(url)
        if not text:
            return None
        obs = _parse_jsonstat(text)
        if not obs:
            return None
        actual = obs[-1]
        previous = obs[-2] if len(obs) >= 2 else None
        return ValuePoint(actual=actual, previous=previous)


def _http_get(url: str) -> str:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _UA, "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:  # noqa: S310 (trusted official URL)
            return resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError, OSError) as exc:
        logger.warning("Eurostat value fetch failed for %s: %s", url, exc)
        return ""


def _parse_jsonstat(text: str) -> List[float]:
    """Extract observation values in chronological order from a JSON-stat 2.0
    message reduced to a single series (time is the only free dimension).
    Returns [] on any shape mismatch."""
    try:
        data = json.loads(text)
        values = data.get("value")
        if not isinstance(values, dict) or not values:
            return []
        # Order by the time dimension's category index, mapping to value keys.
        time = ((data.get("dimension") or {}).get("time") or {})
        index = ((time.get("category") or {}).get("index") or {})
        if isinstance(index, dict) and index:
            # {period: position}; a single-series message keys value by position.
            ordered_positions = sorted(index.values())
            out = []
            for pos in ordered_positions:
                v = values.get(str(pos))
                if isinstance(v, (int, float)):
                    out.append(float(v))
            if out:
                return out
        # Fallback: numeric-string keys in order.
        out = []
        for k in sorted(values.keys(), key=lambda s: int(s) if s.isdigit() else 0):
            v = values[k]
            if isinstance(v, (int, float)):
                out.append(float(v))
        return out
    except (ValueError, TypeError, KeyError):
        return []


__all__ = ["EurostatValueFetcher"]
