"""Engine-measured facts around a recurring economic publication (NW-3).

Computes, for ONE indicator's PAST releases measured on ONE attached market,
the three engine-measured facts the publication page renders:

  #1 ``calm_before``     — the hour BEFORE each release vs. the same hour-of-day
                           on non-release reference days (amplitude comparison).
  #2 ``structure_state`` — the structural context AT each release instant, by
                           REPLAYING detection on the window ending at that bar
                           (no look-ahead), plus the state NOW.
  #4 ``return_to_calm``  — minutes after each release until per-bar movement
                           returns to the hour's ordinary level (distribution).

Question #3 (zone lifecycle) is DEFERRED and NOT computed here — the schema has
no field for it, and this module never fabricates one.

This module binds to ``publication_measures_schema.PublicationMeasures`` and
never changes it. Discipline enforced (schema + mission NW-3):
  · Durations are WHOLE MINUTES, never candle counts.
  · Every measure carries a ``MeasureProvenance`` (method, denominator, market,
    period, reference-day count when a baseline is used, quote unit).
  · A measure that cannot be computed reliably is returned as ``None`` — never
    approximated. The reliability floor is < 4 measured releases → ``None``.
  · No central-value statistic is EXPOSED. Medians are used only INTERNALLY to
    define the ordinary hour-of-day reference; only ``reference_amount`` leaks.

``compute_publication_measures`` is PURE on its inputs (releases + a candle
DataFrame), so the pytest drives it with synthetic data. ``load_default_measures``
is the thin runtime loader that reads the bundled calendar + gold M15 CSVs.

Timezone assumption: all datetimes — the ``releases`` list, the candle
DataFrame index, and the CSV timestamps read by the loader — are treated as
UTC. The bundled CSVs store naive "UTC-like" timestamps; the loader localises
them to UTC without shifting the wall-clock value.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, List, Optional, Sequence

from src.intelligence.publication_measures_schema import (
    CalmBeforeMeasure,
    DurationTranche,
    MeasureExtreme,
    MeasureProvenance,
    PublicationMeasures,
    ReturnToCalmMeasure,
    StructureStateMeasure,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

# Minimum releases that must be measurable for a measure to be considered
# reliable. Fewer → the measure is omitted (None), never approximated.
MIN_RELIABLE_RELEASES = 4

# M15 timeframe constants. One bar is 15 minutes; four bars is one hour.
BAR_MINUTES = 15
BARS_PER_HOUR = 4

# How close (fraction of price) an intact liquidity pocket must sit to the
# close to count as "within" for #2. Mission-specified 0.5 %.
POCKET_WITHIN_FRACTION = 0.005

# Replay window length for #2 detection (bars ending at the release instant).
STRUCTURE_REPLAY_BARS = 1200


# ---------------------------------------------------------------------------
# Small internal candle object for the engine replay (#2). ``build_enriched_frame``
# only reads .open/.high/.low/.close/.volume/.ts, so a lightweight record does.
# ---------------------------------------------------------------------------


class _Candle:
    __slots__ = ("open", "high", "low", "close", "volume", "ts")

    def __init__(self, ts, o, h, l, c, v):
        self.ts = ts
        self.open = o
        self.high = h
        self.low = l
        self.close = c
        self.volume = v


def _as_utc(dt: datetime) -> datetime:
    """Normalise a datetime to tz-aware UTC (assume naive == UTC)."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _median(values: Sequence[float]) -> Optional[float]:
    """Ordinary central value — used INTERNALLY only, never exposed."""
    vals = sorted(v for v in values if v is not None)
    n = len(vals)
    if n == 0:
        return None
    mid = n // 2
    if n % 2 == 1:
        return float(vals[mid])
    return float((vals[mid - 1] + vals[mid]) / 2.0)


# ---------------------------------------------------------------------------
# Hour-of-day reference (shared by #1 and #4)
# ---------------------------------------------------------------------------


def _build_hour_reference(
    candles: "pd.DataFrame",
    release_days: set,
    reference_days: int,
):
    """Build the ordinary per-bar and 60-min amplitudes per hour-of-day.

    Reference days = calendar days (UTC) with NO release in ``release_days``.
    For each hour-of-day (0..23) we gather, from the most-recent ``reference_days``
    clean days, the per-bar ranges (high-low of each M15 bar in that hour) and the
    rolling 60-min ranges (max(high)-min(low) over each set of 4 consecutive bars
    inside that hour). The ordinary value is the median (internal only).

    Returns ``(per_bar_ref: dict[int,float], hour60_ref: dict[int,float])``.
    """
    import pandas as pd  # local import keeps module import cheap for tests

    idx = candles.index
    hours = idx.hour
    days = idx.normalize()

    # per-bar range for every bar
    bar_range = (candles["high"] - candles["low"]).astype(float)

    per_bar_buckets: dict[int, list] = {h: [] for h in range(24)}
    hour60_buckets: dict[int, list] = {h: [] for h in range(24)}
    # Track how many distinct clean days already contributed per hour bucket, so
    # we cap at the most-recent ``reference_days`` clean days per bucket.
    days_seen: dict[int, set] = {h: set() for h in range(24)}

    # Iterate most-recent day first so the cap keeps recent clean days.
    unique_days = sorted({d for d in days}, reverse=True)
    for day in unique_days:
        day_date = day.date()
        if day_date in release_days:
            continue
        day_mask = days == day
        sub = candles[day_mask]
        sub_hours = sub.index.hour
        sub_range = bar_range[day_mask]
        for h in range(24):
            if reference_days and len(days_seen[h]) >= reference_days:
                continue
            hmask = sub_hours == h
            if not hmask.any():
                continue
            highs = sub["high"][hmask].astype(float)
            lows = sub["low"][hmask].astype(float)
            ranges = sub_range[hmask]
            if len(ranges) == 0:
                continue
            days_seen[h].add(day_date)
            per_bar_buckets[h].extend(float(r) for r in ranges.values)
            # 60-min amplitude for the hour = full-hour high-low (up to 4 bars).
            hour60_buckets[h].append(float(highs.max() - lows.min()))

    per_bar_ref = {}
    hour60_ref = {}
    for h in range(24):
        m = _median(per_bar_buckets[h])
        if m is not None:
            per_bar_ref[h] = m
        m60 = _median(hour60_buckets[h])
        if m60 is not None:
            hour60_ref[h] = m60
    return per_bar_ref, hour60_ref


# ---------------------------------------------------------------------------
# #1 calm_before
# ---------------------------------------------------------------------------


def _compute_calm_before(
    candles: "pd.DataFrame",
    releases: List[datetime],
    hour60_ref: dict,
    market: str,
    reference_days: int,
) -> Optional[CalmBeforeMeasure]:
    measured: list[tuple[datetime, float, float]] = []  # (release, pre_amp, ref)
    for t in releases:
        hour = t.hour
        ref = hour60_ref.get(hour)
        if ref is None:
            continue
        # 4 M15 bars in [T-60min, T)
        lo = t - timedelta(minutes=60)
        window = candles[(candles.index >= lo) & (candles.index < t)]
        if len(window) < BARS_PER_HOUR:
            continue
        pre_amp = float(window["high"].max() - window["low"].min())
        measured.append((t, pre_amp, ref))

    if len(measured) < MIN_RELIABLE_RELEASES:
        return None

    calmer = sum(1 for _, amp, ref in measured if amp < ref)
    busier = len(measured) - calmer

    calmest_rel, calmest_amp, _ = min(measured, key=lambda m: m[1])
    busiest_rel, busiest_amp, _ = max(measured, key=lambda m: m[1])

    # reference_amount = the ordinary 60-min amplitude for the releases' hour(s).
    # CPI publishes at a fixed hour; expose that hour's ordinary amplitude (median
    # across the measured releases' hour references — internal median, exposed as
    # the single reference number).
    ref_amount = _median([ref for _, _, ref in measured]) or 0.0

    period_start = min(m[0] for m in measured)
    period_end = max(m[0] for m in measured)

    return CalmBeforeMeasure(
        provenance=MeasureProvenance(
            method_key="measure.calm_before.hour_of_day_amplitude",
            sample_size=len(measured),
            market=market,
            period_start=period_start,
            period_end=period_end,
            reference_days=reference_days,
            quote_unit="USD",
        ),
        reference_amount=round(float(ref_amount), 4),
        calmer_count=calmer,
        busier_count=busier,
        calmest=MeasureExtreme(observed_at=calmest_rel, amount=round(calmest_amp, 4)),
        busiest=MeasureExtreme(observed_at=busiest_rel, amount=round(busiest_amp, 4)),
    )


# ---------------------------------------------------------------------------
# #4 return_to_calm
# ---------------------------------------------------------------------------


def _compute_return_to_calm(
    candles: "pd.DataFrame",
    releases: List[datetime],
    per_bar_ref: dict,
    market: str,
    reference_days: int,
    settle_window_minutes: int,
) -> Optional[ReturnToCalmMeasure]:
    measured: list[tuple[datetime, Optional[int]]] = []  # (release, minutes or None)
    for t in releases:
        hour = t.hour
        ref = per_bar_ref.get(hour)
        if ref is None:
            continue
        hi = t + timedelta(minutes=settle_window_minutes)
        window = candles[(candles.index >= t) & (candles.index <= hi)]
        if len(window) < 2:
            continue
        highs = window["high"].astype(float).values
        lows = window["low"].astype(float).values
        ts = window.index
        # First bar at/after T where per-bar range <= ref for 2 CONSECUTIVE bars.
        settle_ts = None
        for i in range(len(window) - 1):
            r0 = highs[i] - lows[i]
            r1 = highs[i + 1] - lows[i + 1]
            if r0 <= ref and r1 <= ref:
                settle_ts = ts[i]
                break
        if settle_ts is None:
            measured.append((t, None))  # never settled within the window
            continue
        settle_dt = _as_utc(settle_ts.to_pydatetime() if hasattr(settle_ts, "to_pydatetime") else settle_ts)
        minutes = int(round((settle_dt - t).total_seconds() / 60.0))
        if minutes < 0:
            minutes = 0
        measured.append((t, minutes))

    if len(measured) < MIN_RELIABLE_RELEASES:
        return None

    settled = [(t, m) for t, m in measured if m is not None]
    never = len(measured) - len(settled)

    if not settled:
        return None

    # Tranches: [0,60), [60,180), [180+)
    tr0 = sum(1 for _, m in settled if 0 <= m < 60)
    tr1 = sum(1 for _, m in settled if 60 <= m < 180)
    tr2 = sum(1 for _, m in settled if m >= 180)
    tranches = [
        DurationTranche(lower_minutes=0, upper_minutes=60, count=tr0),
        DurationTranche(lower_minutes=60, upper_minutes=180, count=tr1),
        DurationTranche(lower_minutes=180, upper_minutes=None, count=tr2),
    ]

    fastest_rel, fastest_min = min(settled, key=lambda m: m[1])
    slowest_rel, slowest_min = max(settled, key=lambda m: m[1])

    period_start = min(m[0] for m in measured)
    period_end = max(m[0] for m in measured)

    return ReturnToCalmMeasure(
        provenance=MeasureProvenance(
            method_key="measure.return_to_calm.hour_of_day_perbar",
            sample_size=len(measured),
            market=market,
            period_start=period_start,
            period_end=period_end,
            reference_days=reference_days,
            quote_unit="USD",
        ),
        tranches=tranches,
        fastest=MeasureExtreme(observed_at=fastest_rel, minutes=int(fastest_min)),
        slowest=MeasureExtreme(observed_at=slowest_rel, minutes=int(slowest_min)),
        never_settled_count=never,
    )


# ---------------------------------------------------------------------------
# #2 structure_state (engine replay, no look-ahead)
# ---------------------------------------------------------------------------


def _range_bucket(close: float, low: float, high: float) -> Optional[str]:
    if high <= low:
        return None
    pos = (close - low) / (high - low)
    if pos < 1.0 / 3.0:
        return "lower"
    if pos < 2.0 / 3.0:
        return "middle"
    return "upper"


def _structure_at(candles: "pd.DataFrame", upto_ts: datetime):
    """Replay detection on the window ending at ``upto_ts`` (inclusive).

    Returns ``(inside_zone, intact_pocket_within, range_bucket)`` or None if the
    window is too small to build an enriched frame reliably.
    """
    from src.intelligence.market_reading_assembler import build_enriched_frame
    from src.intelligence.market_reading_mappers import (
        collect_liquidity_pools,
        collect_zones,
    )

    window = candles[candles.index <= upto_ts]
    if len(window) < 60:  # too little history to replay detection reliably
        return None
    window = window.tail(STRUCTURE_REPLAY_BARS)

    candle_objs = [
        _Candle(
            _as_utc(ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts),
            float(row.open), float(row.high), float(row.low),
            float(row.close), float(getattr(row, "volume", 0.0) or 0.0),
        )
        for ts, row in zip(window.index, window.itertuples(index=False))
    ]
    enriched, _cfg = build_enriched_frame(candle_objs)
    last = len(enriched) - 1
    if last < 0:
        return None

    zones = collect_zones(enriched, idx=last)
    pools = collect_liquidity_pools(enriched, idx=last)

    close_at_t = float(enriched["close"].iloc[last])

    # inside_zone: close within any OB or FVG [level_low, level_high]
    inside_zone = False
    for z in zones.get("order_blocks", []) + zones.get("fair_value_gaps", []):
        lo = z.get("level_low")
        hi = z.get("level_high")
        if lo is not None and hi is not None and lo <= close_at_t <= hi:
            inside_zone = True
            break

    # intact pocket within 0.5% of close
    intact_within = False
    for p in pools:
        if p.get("status") != "intact":
            continue
        level = p.get("level")
        if level is None or close_at_t == 0:
            continue
        if abs(level - close_at_t) / abs(close_at_t) <= POCKET_WITHIN_FRACTION:
            intact_within = True
            break

    # range position over the window (swing highs/lows if exposed, else H/L)
    range_high = None
    range_low = None
    for p in pools:
        if p.get("side") == "bsl":
            lvl = p.get("level")
            if lvl is not None:
                range_high = lvl if range_high is None else max(range_high, lvl)
        elif p.get("side") == "ssl":
            lvl = p.get("level")
            if lvl is not None:
                range_low = lvl if range_low is None else min(range_low, lvl)
    if range_high is None:
        range_high = float(enriched["high"].max())
    if range_low is None:
        range_low = float(enriched["low"].min())

    bucket = _range_bucket(close_at_t, range_low, range_high)
    return inside_zone, intact_within, bucket


def _compute_structure_state(
    candles: "pd.DataFrame",
    releases: List[datetime],
    market: str,
    now: Optional[datetime],
) -> Optional[StructureStateMeasure]:
    inside_count = 0
    pocket_count = 0
    lower = middle = upper = 0
    measured_releases: list[datetime] = []

    for t in releases:
        res = _structure_at(candles, t)
        if res is None:
            continue
        inside_zone, intact_within, bucket = res
        measured_releases.append(t)
        if inside_zone:
            inside_count += 1
        if intact_within:
            pocket_count += 1
        if bucket == "lower":
            lower += 1
        elif bucket == "middle":
            middle += 1
        elif bucket == "upper":
            upper += 1

    if len(measured_releases) < MIN_RELIABLE_RELEASES:
        return None

    # now_* — same computation on the full candle set's last bar.
    now_inside = None
    now_pocket = None
    now_pos = None
    now_ts = _as_utc(now) if now is not None else None
    if now_ts is None:
        last_ts = candles.index[-1]
        now_ts = _as_utc(last_ts.to_pydatetime() if hasattr(last_ts, "to_pydatetime") else last_ts)
    now_res = _structure_at(candles, now_ts)
    if now_res is not None:
        now_inside, now_pocket, now_pos = now_res

    return StructureStateMeasure(
        provenance=MeasureProvenance(
            method_key="measure.structure_state.replay_at_release",
            sample_size=len(measured_releases),
            market=market,
            period_start=min(measured_releases),
            period_end=max(measured_releases),
            reference_days=None,  # no baseline used
            quote_unit=None,
        ),
        inside_zone_count=inside_count,
        intact_pocket_count=pocket_count,
        range_lower_count=lower,
        range_middle_count=middle,
        range_upper_count=upper,
        now_inside_zone=now_inside,
        now_intact_pocket_within=now_pocket,
        now_range_position=now_pos,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_publication_measures(
    event_key: str,
    market: str,
    releases: List[datetime],
    candles: "pd.DataFrame",
    *,
    now: Optional[datetime] = None,
    reference_days: int = 60,
    settle_window_minutes: int = 360,
) -> PublicationMeasures:
    """Compute the three NW-3 measures for one publication on one market.

    Pure on its inputs. ``releases`` = UTC datetimes of past publications.
    ``candles`` = M15 OHLCV DataFrame indexed by UTC timestamp with columns
    open/high/low/close/volume. Any measure with < 4 usable releases is omitted
    (None). See module docstring for the UTC assumption.
    """
    if candles is None or len(candles) == 0 or not releases:
        return PublicationMeasures(event_key=event_key, market=market)

    # Normalise the candle index to tz-aware UTC.
    candles = candles.copy()
    if candles.index.tz is None:
        candles.index = candles.index.tz_localize(timezone.utc)
    else:
        candles.index = candles.index.tz_convert(timezone.utc)
    candles = candles.sort_index()

    cov_start = candles.index[0].to_pydatetime()
    cov_end = candles.index[-1].to_pydatetime()

    # Keep releases within candle coverage; normalise to UTC; dedupe (a release
    # instant with several sub-events — "CPI m/m" + "Core CPI m/m" — is ONE
    # publication moment).
    norm: list[datetime] = []
    for r in releases:
        ru = _as_utc(r)
        if cov_start <= ru <= cov_end:
            norm.append(ru)
    releases_utc = sorted(set(norm))

    if not releases_utc:
        return PublicationMeasures(event_key=event_key, market=market)

    release_days = {r.date() for r in releases_utc}

    per_bar_ref, hour60_ref = _build_hour_reference(
        candles, release_days, reference_days
    )

    calm = _compute_calm_before(
        candles, releases_utc, hour60_ref, market, reference_days
    )
    rtc = _compute_return_to_calm(
        candles, releases_utc, per_bar_ref, market, reference_days,
        settle_window_minutes,
    )
    structure = _compute_structure_state(candles, releases_utc, market, now)

    return PublicationMeasures(
        event_key=event_key,
        market=market,
        calm_before=calm,
        structure_state=structure,
        return_to_calm=rtc,
    )


# ---------------------------------------------------------------------------
# Runtime loader (NOT used by tests — reads the bundled, gitignored CSVs)
# ---------------------------------------------------------------------------


def load_default_measures(
    event_key: str = "us_cpi",
    market: str = "XAUUSD",
    max_releases: int = 24,
) -> Optional[PublicationMeasures]:
    """Read US CPI releases + gold M15 candles from the bundled CSVs and compute.

    Returns ``None`` when the data is unavailable (so callers render nothing).

    Only the ``max_releases`` MOST RECENT releases are measured (default 24 ≈ two
    years of a monthly indicator): it keeps the sample recent and comparable, and
    caps the per-release engine replay so the first (uncached) request returns in
    a few seconds rather than tens of seconds. The denominator is shown, so the
    bound is honest, not silent.

    Timezone: the bundled CSVs store naive "UTC-like" timestamps; we treat them
    as UTC (localise without shifting the wall-clock value) — consistent with the
    rest of the pipeline (see MEMORY: "Treat CSV timestamps as UTC").
    """
    import os

    try:
        import pandas as pd

        import config as _config
    except Exception:  # pragma: no cover - defensive
        return None

    # --- Calendar releases (USD CPI) --------------------------------------
    calendar_path = None
    cfg_path = _config.NEWS_FILTER_CONFIG.get("calendar_file")
    for candidate in (
        cfg_path,
        os.path.join(_config.DATA_DIR, "economic_calendar_HIGH_IMPACT_2019_2025.csv"),
    ):
        if candidate and os.path.exists(candidate):
            calendar_path = candidate
            break
    if calendar_path is None:
        return None

    try:
        cal = pd.read_csv(calendar_path)
    except Exception:  # pragma: no cover
        return None
    if not {"Date", "Currency", "Event"} <= set(cal.columns):
        return None

    is_usd = cal["Currency"].astype(str).str.upper() == "USD"
    is_cpi = cal["Event"].astype(str).str.contains("CPI", case=False, na=False)
    rows = cal[is_usd & is_cpi]
    if rows.empty:
        return None

    releases: list[datetime] = []
    for raw in rows["Date"]:
        ts = pd.to_datetime(raw, errors="coerce")
        if pd.isna(ts):
            continue
        dt = ts.to_pydatetime()
        releases.append(dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt)
    releases = sorted(set(releases))
    if not releases:
        return None
    # Keep only the most recent window (recent + comparable + bounded latency).
    if max_releases and len(releases) > max_releases:
        releases = releases[-max_releases:]

    # --- Gold M15 candles -------------------------------------------------
    gold_path = getattr(_config, "HISTORICAL_DATA_FILE", None)
    if not gold_path or not os.path.exists(gold_path):
        return None
    try:
        candles = pd.read_csv(gold_path)
    except Exception:  # pragma: no cover
        return None

    # Normalise columns → open/high/low/close/volume, UTC DatetimeIndex.
    rename = {}
    for col in candles.columns:
        low = col.lower()
        if low in ("date", "datetime", "time", "timestamp"):
            rename[col] = "ts"
        elif low in ("open", "high", "low", "close", "volume"):
            rename[col] = low
    candles = candles.rename(columns=rename)
    if "ts" not in candles.columns or not {
        "open", "high", "low", "close"
    } <= set(candles.columns):
        return None
    if "volume" not in candles.columns:
        candles["volume"] = 0.0
    candles["ts"] = pd.to_datetime(candles["ts"], errors="coerce")
    candles = candles.dropna(subset=["ts"]).set_index("ts").sort_index()
    if candles.empty:
        return None

    return compute_publication_measures(event_key, market, releases, candles)


__all__ = ["compute_publication_measures", "load_default_measures"]
