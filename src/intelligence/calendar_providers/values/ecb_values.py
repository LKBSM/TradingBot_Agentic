"""ECB value fetcher — SDMX-JSON, no key (NW-1c §3A).

Fetches the latest published value (+ previous) for an ECB series by its stable
key, e.g. ``FM.D.U2.EUR.4F.KR.MRR_FR.LEV``. Values are returned AS PUBLISHED — no
conversion, no rounding (a licence condition of the ECB). Graceful: any failure
returns ``None`` and the event stays ``unfetched``.

The ECB reuse policy requires the source be quoted and the statistics not be
modified; the calendar shows the value verbatim with the ECB attribution.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import List, Optional

from src.intelligence.calendar_providers.values.base_value import (
    SeriesPoint,
    ValueFetcher,
    ValuePoint,
)

logger = logging.getLogger(__name__)

_BASE = "https://data-api.ecb.europa.eu/service/data"
_UA = "Mozilla/5.0 (compatible; MIA-Markets-Calendar/1.0; +https://mia-markets)"
_TIMEOUT_S = 10


class ECBValueFetcher(ValueFetcher):
    def __init__(self, http_get=None) -> None:
        # Injectable for tests; default hits the live no-key SDMX endpoint.
        self._get = http_get or _http_get

    def fetch(self, series_code: str) -> Optional[ValuePoint]:
        # series_code = "FM.D.U2.EUR.4F.KR.MRR_FR.LEV" → flowRef "FM", key rest.
        flow, _, key = series_code.partition(".")
        if not flow or not key:
            return None
        # A policy rate is a STEP series (a value persists until it changes), so
        # ask for a wider window and derive "previous" as the last DISTINCT prior
        # value — not the mechanical second-to-last observation, which on a step
        # series is almost always equal to the current one (NW-1c §4).
        url = f"{_BASE}/{flow}/{key}?lastNObservations=250&format=jsondata"
        text = self._get(url)
        if not text:
            return None
        obs = _parse_observations(text)
        if not obs:
            return None
        actual = obs[-1]
        previous = _last_distinct_before(obs, actual)
        return ValuePoint(actual=actual, previous=previous)

    def fetch_series(
        self, series_code: str, limit: int = 12, kind: str = "level"
    ) -> List[SeriesPoint]:  # ``kind`` unused: the policy rate is served as a level.
        """The last ``limit`` DISTINCT rate LEVELS (oldest→newest), each labelled by
        the month its level took effect ("YYYY-MM"). A policy rate is a step series:
        plotting consecutive identical daily observations would be meaningless, so we
        collapse each run to its first date — the actual published decisions. Values
        AS PUBLISHED; [] on any failure/absence. Never raises, never fabricates."""
        flow, _, key = series_code.partition(".")
        if not flow or not key or limit < 1:
            return []
        url = f"{_BASE}/{flow}/{key}?lastNObservations=250&format=jsondata"
        text = self._get(url)
        if not text:
            return []
        dated = _parse_observation_points(text)
        if not dated:
            return []
        # Collapse consecutive equal levels to the FIRST date of each run — the
        # moment the level was decided, not every day it merely persisted.
        changes: List[SeriesPoint] = []
        last_value: Optional[float] = None
        for period, value in dated:
            if last_value is None or value != last_value:
                changes.append(SeriesPoint(period=period[:7], value=value))
                last_value = value
        return changes[-int(limit):] if len(changes) > limit else changes


def _last_distinct_before(obs: List[float], actual: float) -> Optional[float]:
    """The most recent value in ``obs`` (chronological) that differs from
    ``actual`` — i.e. the prior level on a step series. ``None`` if the series
    never changed within the window."""
    for v in reversed(obs[:-1]):
        if v != actual:
            return v
    return None


def _http_get(url: str) -> str:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _UA, "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:  # noqa: S310 (trusted official URL)
            return resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError, OSError) as exc:
        logger.warning("ECB value fetch failed for %s: %s", url, exc)
        return ""


def _parse_observations(text: str):
    """Extract the observation values in chronological order from an SDMX-JSON
    (jsondata) message. Returns a list of floats, or [] on any shape mismatch."""
    try:
        data = json.loads(text)
        datasets = data.get("dataSets") or []
        if not datasets:
            return []
        series = datasets[0].get("series") or {}
        if not series:
            return []
        # A single series is expected; take the first.
        first = next(iter(series.values()))
        observations = first.get("observations") or {}
        # Keys are observation indices as strings; order by their integer value.
        ordered = sorted(observations.items(), key=lambda kv: int(kv[0]))
        out = []
        for _, val in ordered:
            if isinstance(val, list) and val and isinstance(val[0], (int, float)):
                out.append(float(val[0]))
        return out
    except (ValueError, TypeError, KeyError, StopIteration):
        return []


def _parse_observation_points(text: str):
    """Extract (period_label, value) observations in chronological order from an
    SDMX-JSON (jsondata) message. The period label is read from the observation
    TIME dimension in ``structure`` (e.g. "2026-08-05"). Returns a list of
    (str, float), or [] on any shape mismatch — never fabricates a period."""
    try:
        data = json.loads(text)
        datasets = data.get("dataSets") or []
        if not datasets:
            return []
        series = datasets[0].get("series") or {}
        if not series:
            return []
        first = next(iter(series.values()))
        observations = first.get("observations") or {}
        if not observations:
            return []
        # Time labels: structure.dimensions.observation[0].values[idx].id
        obs_dims = (
            ((data.get("structure") or {}).get("dimensions") or {}).get("observation")
            or []
        )
        time_values = obs_dims[0].get("values") if obs_dims else None
        if not isinstance(time_values, list) or not time_values:
            return []
        ordered = sorted(observations.items(), key=lambda kv: int(kv[0]))
        out = []
        for idx_str, val in ordered:
            idx = int(idx_str)
            if not (0 <= idx < len(time_values)):
                continue
            period = str(time_values[idx].get("id", ""))
            if period and isinstance(val, list) and val and isinstance(val[0], (int, float)):
                out.append((period, float(val[0])))
        return out
    except (ValueError, TypeError, KeyError, StopIteration):
        return []


__all__ = ["ECBValueFetcher"]
