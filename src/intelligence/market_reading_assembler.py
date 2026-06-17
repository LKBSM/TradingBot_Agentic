"""MarketReadingAssembler — lazy on-demand generation of MarketReading objects.

Section 3.1 of the V2 architecture doc ("Moteur hybride lazy puis continu").
Chantier 2 implements the **lazy** path only — the recurring scheduler job
(continu after first access, auto-stop after 24h) is Chantier 3.

Lazy flow per get_or_generate():
  1. Compute expected last-candle-close timestamp for (instrument, timeframe).
  2. Read latest stored MarketReading for that pair.
  3. If it matches the expected close → return it (cache hit), still mark
     the combination as active.
  4. Otherwise: fetch candles via data_provider, persist into candles_store,
     run the SMC pipeline, build the structure/regime/events/conditions,
     persist via readings_store, mark active, return.

The SMC pipeline (smc_features extraction + optional confluence) is injectable
via the ``smc_pipeline`` callable so the assembler stays testable without
needing the heavy SmartMoneyEngine / ConfluenceDetector classes at unit-test
time. The default implementation lazily wires the real engine.

The Haiku description engine is also injectable (Étape 5). When absent, the
assembler uses the template fallback from market_reading_mappers.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

from src.intelligence.market_reading_mappers import (
    candles_to_regime,
    confluence_signal_to_structure,
    empty_events,
    tags_and_description,
)
from src.intelligence.market_reading_schema import (
    MarketReading,
    MarketReadingConditions,
    MarketReadingEvents,
    MarketReadingHeader,
)
from src.intelligence.provider_snapshot import snapshot_provider_response

logger = logging.getLogger(__name__)


# Timeframe minute durations (aligned with TwelveDataProvider._TIMEFRAME_MAP keys).
_TIMEFRAME_MINUTES: dict[str, int] = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 60,
    "H4": 240,
    "D1": 1440,
    "W1": 10080,
}


def expected_last_candle_close(timeframe: str, now: datetime) -> datetime:
    """Return the close timestamp of the most recent fully-closed candle.

    By convention here, candle_close_ts is the boundary at which a candle
    completes (e.g. M15 candle 13:45-14:00 closes at 14:00:00Z). The "last
    fully closed" candle's close ts is the most recent boundary that has
    elapsed at ``now``.
    """
    tf_key = timeframe.upper()
    if tf_key not in _TIMEFRAME_MINUTES:
        raise ValueError(
            f"Unsupported timeframe: {timeframe!r}. "
            f"Supported: {sorted(_TIMEFRAME_MINUTES)}"
        )
    minutes = _TIMEFRAME_MINUTES[tf_key]
    ts = now.astimezone(timezone.utc) if now.tzinfo else now.replace(tzinfo=timezone.utc)
    epoch_minutes = int(ts.timestamp() // 60)
    last_boundary_minutes = (epoch_minutes // minutes) * minutes
    return datetime.fromtimestamp(last_boundary_minutes * 60, tz=timezone.utc)


def drop_unclosed_candles(
    candles: Sequence[Any], timeframe: str, expected_close: datetime
) -> list[Any]:
    """Keep only the bars fully closed at ``expected_close``.

    Twelve Data labels bars by OPEN time and includes the still-forming bar in
    its response. Analysing that bar makes every detector repaint (audit
    DETECTION_QUALITY_REVIEW_2026_06_12 §T3): the same OB was published twice
    under two ids, and readings were not reproducible from final candles.

    A bar opened at ``ts`` is closed iff ``ts + timeframe <= expected_close``.
    This also drops any future-labelled bar (defence-in-depth against the §T2
    timezone mislabelling). Naive timestamps are treated as UTC, matching the
    provider parse and the candles cache convention.
    """
    minutes = _TIMEFRAME_MINUTES[timeframe.upper()]
    span = timedelta(minutes=minutes)
    out: list[Any] = []
    for candle in candles:
        ts = candle.ts
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if ts + span <= expected_close:
            out.append(candle)
    return out


SmcPipelineFn = Callable[[Sequence[Any]], Tuple[dict[str, float], Optional[Any]]]


def _default_smc_pipeline(candles: Sequence[Any]) -> Tuple[dict[str, float], Optional[Any]]:
    """Default SMC pipeline — runs SmartMoneyEngine on the candle list.

    Returns ``(smc_features, confluence_signal=None)`` because Chantier 2
    does not wire ConfluenceDetector by default (scoring is for the legacy
    InsightSignalV2 flow; MarketReading describes state, not setups).
    Callers wanting a ConfluenceSignal can inject a custom pipeline.

    Heavy imports are local so unit tests with mocks pay no import cost.
    """
    if not candles:
        return {}, None

    import pandas as pd

    from src.intelligence.smart_money import SmartMoneyEngine

    df = pd.DataFrame(
        [
            {
                "open": float(c.open),
                "high": float(c.high),
                "low": float(c.low),
                "close": float(c.close),
                "volume": float(getattr(c, "volume", 0.0)),
            }
            for c in candles
        ],
        index=[c.ts for c in candles],
    )
    df.index.name = "ts"

    engine = SmartMoneyEngine(data=df, config={}, verbose=False)
    # compute_divergence=False: the MarketReading mapper does not consume the
    # RSI divergence column, so we skip its O(n·k) pass on every reading (D2-9).
    enriched = engine.analyze(compute_divergence=False)
    last_row = enriched.iloc[-1].to_dict()

    smc_features: dict[str, float] = {}
    for k, v in last_row.items():
        if isinstance(v, bool):  # bool is a subclass of int — exclude
            smc_features[str(k)] = 1.0 if v else 0.0
        elif isinstance(v, (int, float)) and not pd.isna(v):
            smc_features[str(k)] = float(v)

    # Merge the REAL structural levels (BOS break level forward-filled, real
    # OB zone, real FVG bounds) so the structure mapper publishes them instead
    # of price ± ATR proxies (audit findings F1/F2/F3). Glue, not engine logic.
    from src.intelligence.market_reading_mappers import collect_zones, realized_levels
    last_idx = len(enriched) - 1
    smc_features.update(realized_levels(enriched, idx=last_idx))
    # Multi-zone registry: surface ALL still-relevant OB/FVG zones the engine
    # computed over the window, not just the last bar (audit §T1). Carried under
    # a reserved key the structure mapper consumes; never persisted.
    smc_features["_zones"] = collect_zones(enriched, idx=last_idx)
    return smc_features, None


class MarketReadingAssembler:
    """Lazy on-demand assembler for MarketReading objects.

    Lifecycle: one instance per process (or per worker). Thread-safety is
    inherited from the underlying stores (RLock).
    """

    # Indicator-grade context windows. Widened 2026-06-15 (was lookback=200,
    # news 240/60 min) so the product shows real history + a multi-day economic
    # calendar, not a 4h keyhole. All overridable via env in the bootstrap.
    DEFAULT_LOOKBACK = 500
    DEFAULT_NEWS_LOOKAHEAD_MIN = 4320   # 3 days — captures FOMC/NFP ahead
    DEFAULT_NEWS_LOOKBACK_MIN = 1440    # 24h — what moved the market today

    def __init__(
        self,
        data_provider: Any,
        readings_store: Any,
        candles_store: Any,
        smc_pipeline: Optional[SmcPipelineFn] = None,
        description_engine: Optional[Any] = None,
        news_pipeline: Optional[Any] = None,
        lookback: int = DEFAULT_LOOKBACK,
        mtf_provider: Optional[Callable[[str, str], Mapping[str, Sequence[Any]]]] = None,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        news_lookahead_min: int = DEFAULT_NEWS_LOOKAHEAD_MIN,
        news_lookback_min: int = DEFAULT_NEWS_LOOKBACK_MIN,
    ) -> None:
        self._data_provider = data_provider
        self._readings_store = readings_store
        self._candles_store = candles_store
        self._smc_pipeline: SmcPipelineFn = smc_pipeline or _default_smc_pipeline
        self._description_engine = description_engine
        self._news_pipeline = news_pipeline
        self._lookback = lookback
        self._mtf_provider = mtf_provider
        self._clock = clock
        self._news_lookahead_min = news_lookahead_min
        self._news_lookback_min = news_lookback_min

    # ------------------------------------------------------------------ #
    # Public accessors (used by the Chantier 3 scheduler / bootstrap)
    # ------------------------------------------------------------------ #
    @property
    def readings_store(self) -> Any:
        return self._readings_store

    @property
    def candles_store(self) -> Any:
        return self._candles_store

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def get_or_generate(self, instrument: str, timeframe: str) -> MarketReading:
        """Return the current MarketReading for (instrument, timeframe).

        Lazy: re-uses the stored payload if it matches the expected last
        candle close. Otherwise re-generates end-to-end. Always marks the
        combination as active (Section 3.3 of architecture doc — hybrid
        scheduler state for Chantier 3).
        """
        expected_close = expected_last_candle_close(timeframe, self._clock())

        existing = self._readings_store.get_latest_reading(instrument, timeframe)
        if existing is not None and self._payload_matches(existing, expected_close):
            self._readings_store.mark_combination_active(instrument, timeframe)
            return MarketReading.model_validate(existing)

        reading = self._build_fresh(instrument, timeframe, expected_close)
        self._readings_store.save_reading(
            instrument, timeframe, expected_close, reading.model_dump(mode="json")
        )
        self._readings_store.mark_combination_active(instrument, timeframe)
        return reading

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    @staticmethod
    def _payload_matches(payload: Mapping[str, Any], expected_close: datetime) -> bool:
        header = payload.get("header") if isinstance(payload, Mapping) else None
        if not isinstance(header, Mapping):
            return False
        stored_ts = header.get("candle_close_ts")
        if stored_ts is None:
            return False
        try:
            parsed = (
                stored_ts
                if isinstance(stored_ts, datetime)
                else datetime.fromisoformat(str(stored_ts).replace("Z", "+00:00"))
            )
        except ValueError:
            return False
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc) == expected_close

    def _build_fresh(
        self, instrument: str, timeframe: str, expected_close: datetime
    ) -> MarketReading:
        raw_candles = self._data_provider.fetch_candles(
            instrument, timeframe, self._lookback
        )
        # Raw response snapshot BEFORE any filtering — the only way to replay
        # a reading bit-for-bit later (the feed revises forming bars; audit §T3
        # found a reading whose close_price existed in no stored final candle).
        snapshot_provider_response(
            instrument, timeframe, raw_candles, fetched_at=self._clock()
        )
        # The forming bar (and any future-labelled bar) never reaches the SMC
        # pipeline nor the cache: candle_close_ts promises closed-candle data,
        # and /api/candles documents "stops at the last fully-closed candle".
        candles = drop_unclosed_candles(raw_candles, timeframe, expected_close)
        if candles:
            try:
                self._candles_store.upsert_candles(instrument, timeframe, candles)
            except Exception as exc:  # pragma: no cover — defensive cache write
                logger.warning(
                    "candles_store.upsert_candles failed for %s/%s: %s",
                    instrument, timeframe, exc,
                )

        smc_features, confluence_signal = self._smc_pipeline(candles)
        current_price = float(candles[-1].close) if candles else 0.0

        structure = confluence_signal_to_structure(
            confluence_signal=confluence_signal,
            smc_features=smc_features,
            bar_ts=expected_close,
            current_price=current_price,
        )

        mtf_candles: Mapping[str, Sequence[Any]] = {}
        if self._mtf_provider is not None:
            try:
                mtf_candles = self._mtf_provider(instrument, timeframe) or {}
            except Exception as exc:  # pragma: no cover — defensive
                logger.warning("mtf_provider failed for %s/%s: %s", instrument, timeframe, exc)

        candle_dicts = [
            {
                "open": float(c.open),
                "high": float(c.high),
                "low": float(c.low),
                "close": float(c.close),
            }
            for c in candles
        ]
        regime = candles_to_regime(candle_dicts, mtf_candles_above=mtf_candles)

        events = self._build_events()

        tags, fallback_description = tags_and_description(structure, regime)
        description, source = self._resolve_description(tags, regime, fallback_description)

        header = MarketReadingHeader(
            instrument=instrument,
            timeframe=timeframe,
            candle_close_ts=expected_close,
            close_price=current_price,
        )
        conditions = MarketReadingConditions(
            tags=tags,
            description=description,
            description_source=source,
        )

        return MarketReading(
            header=header,
            structure=structure,
            regime=regime,
            events=events,
            conditions=conditions,
        )

    def _build_events(self) -> MarketReadingEvents:
        """Fill ``news_upcoming`` / ``news_just_published`` from the pipeline.

        Compat Chantier 2: when no news pipeline is wired, returns an empty
        events block (identical to ``empty_events()``). News timing is
        independent of candle cadence (architecture doc §3.6) so the wall
        clock is used as the reference ``now``. ``technical_triggers_recent``
        is left empty here — out of scope for Chantier 3.
        """
        if self._news_pipeline is None:
            return empty_events()
        now = self._clock()
        try:
            upcoming = self._news_pipeline.get_upcoming(
                currency_filter=["USD", "EUR"],
                lookahead_minutes=self._news_lookahead_min,
                now=now,
            )
            published = self._news_pipeline.get_just_published(
                currency_filter=["USD", "EUR"],
                lookback_minutes=self._news_lookback_min,
                now=now,
            )
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning("news_pipeline failed: %s — emitting empty events", exc)
            return empty_events()
        return MarketReadingEvents(
            news_upcoming=upcoming,
            news_just_published=published,
        )

    def _resolve_description(
        self,
        tags: list[str],
        regime: Any,
        fallback: str,
    ) -> tuple[str, str]:
        """Use the Haiku engine if injected, else the template fallback string."""
        if self._description_engine is None:
            return fallback, "template_fallback"
        try:
            description, source = self._description_engine.generate(tags, regime)
            return description, source
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning("description_engine.generate failed: %s — using template", exc)
            return fallback, "template_fallback"


__all__ = [
    "MarketReadingAssembler",
    "SmcPipelineFn",
    "expected_last_candle_close",
]
