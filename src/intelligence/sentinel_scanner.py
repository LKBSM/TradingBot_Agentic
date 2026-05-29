"""Main scanning loop for Smart Sentinel AI.

Replaces LiveTradingLoop. Polls for M15 bar closes, runs the analysis
pipeline, and publishes scored signals with LLM narratives.

Pipeline per bar:
  DataProvider OHLCV → SmartMoneyEngine → RegimeAgent → NewsAgent
  → VolForecaster → ConfluenceDetector → SemanticCache → LLMNarrativeEngine → SignalStore

Supports multi-symbol scanning via MultiSymbolScanner.
"""

from __future__ import annotations

import logging
import time
import threading
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from src.intelligence.circuit_breaker import CircuitBreaker, CircuitOpenError
from src.intelligence.confluence_detector import (
    ConfluenceDetector,
    ConfluenceSignal,
    SignalTier,
)
from src.intelligence.insight_assembler import InsightAssembler
from src.intelligence.llm_narrative_engine import (
    LLMNarrativeEngine,
    NarrativeTier,
    SignalNarrative,
)
from src.intelligence.regime_filter import RegimeFilter
from src.intelligence.semantic_cache import SemanticCache
from src.intelligence.template_narrative_engine import TemplateNarrativeEngine
from src.intelligence.signal_state_machine import (
    BarInput,
    PublicState,
    SignalStateMachine,
    StateTransition,
)
from src.intelligence.state_persistence import (
    load_state_machine,
    save_state_machine,
)
from src.risk.kill_switch import KillSwitch

logger = logging.getLogger(__name__)


# =============================================================================
# TIER → NARRATIVE TIER MAPPING
# =============================================================================

TIER_TO_NARRATIVE: Dict[str, NarrativeTier] = {
    "VISUAL": NarrativeTier.VISUAL,
    "VALIDATOR": NarrativeTier.VALIDATOR,
    "NARRATOR": NarrativeTier.NARRATOR,
    "INSTITUTIONAL": NarrativeTier.INSTITUTIONAL,
}


# Order must match DEFAULT_FEATURE_NAMES in scoring/lgbm_scorer.py:
# ("smc_structure", "order_blocks", "fvg", "retest", "regime",
#  "vol_forecast", "news", "momentum_rsi_div")
# Map from ConfluenceSignal.components[i].name → feature index.
_COMPONENT_NAME_TO_INDEX: Dict[str, int] = {
    "bos": 0,            # smc_structure
    "order_block": 1,    # order_blocks
    "fvg": 2,            # fvg
    # index 3 (retest) — not currently exposed as a separate component
    "regime": 4,         # regime
    # index 5 (vol_forecast) — not currently exposed as a separate component
    "news": 6,           # news
    "momentum": 7,       # momentum_rsi_div (combined with rsi_divergence)
    "rsi_divergence": 7,
}


def _components_to_feature_vector(signal: ConfluenceSignal) -> np.ndarray:
    """Map a ConfluenceSignal's components → 8-element numpy vector.

    Returns an 8-element float vector in the order expected by
    CalibratedConvictionPipeline. Missing components contribute 0.0,
    matching the training-time placeholder convention (retest and
    vol_forecast are not persisted as standalone components yet —
    see reports/scoring_v2_brier_validation.md).
    """
    vec = np.zeros(8, dtype=float)
    if signal is None or not getattr(signal, "components", None):
        return vec
    for comp in signal.components:
        idx = _COMPONENT_NAME_TO_INDEX.get(getattr(comp, "name", ""))
        if idx is None:
            continue
        # Two components map to index 7 — accumulate (momentum + rsi_div).
        vec[idx] += float(getattr(comp, "weighted_score", 0.0) or 0.0)
    return vec


class SentinelScanner:
    """
    Main scanning loop: poll for M15 bars → run analysis → publish signals.

    All dependencies are injected via constructor for testability.
    """

    def __init__(
        self,
        data_provider: Any,
        smc_factory: Callable[[pd.DataFrame], Any],
        regime_agent: Any,
        news_agent: Any,
        confluence: ConfluenceDetector,
        llm_engine: LLMNarrativeEngine,
        cache: Optional[SemanticCache],
        signal_store: Any,
        notifier: Optional[Any] = None,
        vol_forecaster: Optional[Any] = None,
        symbol: str = "XAUUSD",
        timeframe: str = "M15",
        # Bumped 200→800 (2026-05-21) so the MTF resample yields ~200 H4 bars,
        # enough warm-up for SMA50/RSI14 on the higher TF. See feedback memory
        # ``feedback_multi_view_ux.md`` and the MTF Phase-1 rewiring plan.
        lookback_bars: int = 800,
        narrative_tier: NarrativeTier = NarrativeTier.NARRATOR,
        poll_interval_seconds: float = 60.0,
        llm_circuit_breaker: Optional[CircuitBreaker] = None,
        notifier_circuit_breaker: Optional[CircuitBreaker] = None,
        state_machine: Optional[SignalStateMachine] = None,
        persistence_path: Optional[Any] = None,
        persistence_max_staleness_bars: int = 4,
        kill_switch: Optional[KillSwitch] = None,
        regime_filter: Optional[RegimeFilter] = None,
        insight_assembler: Optional[InsightAssembler] = None,
    ):
        self._data_provider = data_provider
        self._smc_factory = smc_factory
        self._regime_agent = regime_agent
        self._news_agent = news_agent
        self._confluence = confluence
        self._llm_engine = llm_engine
        self._cache = cache
        self._signal_store = signal_store
        self._notifier = notifier
        self._vol_forecaster = vol_forecaster
        self._symbol = symbol
        self._timeframe = timeframe
        self._lookback = lookback_bars
        self._narrative_tier = narrative_tier
        self._poll_interval = poll_interval_seconds

        # Circuit breakers for external services
        self._llm_breaker = llm_circuit_breaker
        self._notifier_breaker = notifier_circuit_breaker

        # Algorithmic fallback engine — used when LLM circuit opens or any
        # narrative call raises. Sub-ms, $0, deterministic. Lazily reused.
        self._fallback_engine = TemplateNarrativeEngine()
        self._fallback_uses: int = 0

        # Operational kill-switch — when supplied, gates _publish_signal,
        # receives heartbeats from the data feed, and observes vol-forecast
        # readings. ``None`` keeps the scanner backwards-compatible.
        self._kill_switch = kill_switch
        self._signals_blocked_by_kill_switch: int = 0

        # Regime filter — drops signals from PF<1 buckets (NY session, top vol
        # quartile). When None the scanner is unfiltered (legacy behaviour).
        # Empirical lift on XAU M15 7-yr replay: PF 1.13 → 1.355 (see
        # `reports/feature_filter_audit.md`).
        self._regime_filter = regime_filter
        self._signals_dropped_by_regime_filter = 0

        # Insight assembler — when wired, every published signal is also
        # materialised as an InsightSignalV2 (v2.1.0 contract) and stored as
        # ``latest_insight`` for B2B/B2C surfaces to read. Legacy passthrough
        # when None — the scanner keeps emitting the v1 SignalRecord path
        # unchanged so existing tests and consumers are not disturbed.
        self._insight_assembler = insight_assembler
        from src.api.insight_signal_v2 import InsightSignalV2 as _InsightV2  # local import to avoid heavy module load on cold paths
        self._InsightSignalV2 = _InsightV2
        self._latest_insight: Optional[_InsightV2] = None
        self._insights_built: int = 0
        self._insight_build_failures: int = 0

        # Signal state machine — None means legacy passthrough (backward compat).
        self._state_machine = state_machine
        # Persistence (optional): save on shutdown, reload on start. Staleness
        # guard drops snapshots older than ``persistence_max_staleness_bars``.
        from pathlib import Path as _Path
        self._persistence_path: Optional[_Path] = (
            _Path(persistence_path) if persistence_path else None
        )
        self._persistence_max_staleness = int(persistence_max_staleness_bars)

        # State
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_bar_ts: Optional[str] = None
        # Last cache cleanup wall-clock time. The scanner runs cleanup_expired
        # every CACHE_CLEANUP_INTERVAL_S (default 1h) so the narrative cache
        # SQLite doesn't grow unbounded with TTL-expired zombies.
        self._last_cache_cleanup_ts: float = time.time()
        # Interruptible sleep for graceful shutdown. With time.sleep(60) the
        # shutdown() call had to wait up to 60s for the loop to exit;
        # _stop_event.wait(timeout) returns immediately when shutdown sets it.
        self._stop_event = threading.Event()

        # Multi-timeframe features — fitted lazily on first scan (Phase 1
        # rewiring, 2026-05-21). Uses ``MultiTimeframeFeatures`` from the
        # environment package which already has the look-ahead-safe causal
        # mask (l.272). Re-fit on every scan is cheap (~800 rows resampled
        # to ~200 H4 bars) but we keep the instance reusable to avoid the
        # allocator pressure of recreating the class every minute.
        self._mtf = None
        self._mtf_warmup_bars = 200  # need ≥200 H4 bars for SMA50 + RSI14 warm-up
        self._mtf_warmup_complete = False
        self._htf_features_computed: int = 0
        self._htf_features_skipped: int = 0

        # Stats
        self._bars_scanned = 0
        self._signals_generated = 0
        self._signals_held_by_state_machine = 0  # entries suppressed before confirmation
        self._state_transitions_emitted = 0
        self._cache_hits = 0
        self._llm_calls = 0
        self._llm_failures = 0
        self._notification_failures = 0
        self._errors = 0
        self._start_time: Optional[float] = None

    @property
    def state_machine(self) -> Optional[SignalStateMachine]:
        """Expose the state machine for API reads / snapshots."""
        return self._state_machine

    @property
    def latest_insight(self):
        """Most recent InsightSignalV2 produced by the assembler, if any."""
        return self._latest_insight

    # ------------------------------------------------------------------ #
    # LIFECYCLE
    # ------------------------------------------------------------------ #

    def start(self, blocking: bool = True) -> None:
        """Start the scanning loop.

        If ``persistence_path`` was supplied and a compatible snapshot
        exists on disk, the state machine is rehydrated from it before
        the loop begins. Stale snapshots (see
        ``persistence_max_staleness_bars``) are discarded.
        """
        self._restore_state_machine()
        self._running = True
        self._start_time = time.time()
        logger.info("SentinelScanner starting (symbol=%s, tf=%s)", self._symbol, self._timeframe)

        if blocking:
            self._run_loop()
        else:
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()

    def shutdown(self) -> None:
        """Graceful shutdown — persists state machine if configured."""
        self._running = False
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)
        self._persist_state_machine()
        logger.info("SentinelScanner stopped. Stats: %s", self.get_stats())

    # ------------------------------------------------------------------ #
    # PERSISTENCE
    # ------------------------------------------------------------------ #

    def _restore_state_machine(self) -> None:
        """Reload state from disk if a snapshot exists and isn't stale."""
        if self._persistence_path is None or self._state_machine is None:
            return
        # We don't know the current bar timestamp yet (no scan has run),
        # so skip the staleness check on cold-start. The scanner loop
        # will overwrite stale state on the next persist cycle.
        restored = load_state_machine(
            self._persistence_path,
            current_bar_ts=None,
            max_staleness_bars=0,  # cold start — no reference bar
        )
        if restored is not None:
            # Swap the live machine's state in-place to preserve caller's reference
            self._state_machine = restored

    def _persist_state_machine(self) -> None:
        """Write the current state to disk. No-op if persistence disabled."""
        if self._persistence_path is None or self._state_machine is None:
            return
        save_state_machine(self._state_machine, self._persistence_path)

    # ------------------------------------------------------------------ #
    # MAIN LOOP
    # ------------------------------------------------------------------ #

    CACHE_CLEANUP_INTERVAL_S = 3600.0  # 1h — narrative cache TTL is 24h, sweep hourly

    def _run_loop(self) -> None:
        while self._running and not self._stop_event.is_set():
            try:
                self._scan_once()
            except Exception as e:
                self._errors += 1
                logger.error("Scanner error: %s", e, exc_info=True)

            # Periodic narrative cache cleanup so the SQLite doesn't grow
            # unbounded with TTL-expired entries (eval 06 finding).
            self._maybe_cleanup_cache()

            if self._running:
                # Interruptible sleep — shutdown() sets _stop_event so we exit
                # immediately rather than waiting for the next tick.
                self._stop_event.wait(timeout=self._poll_interval)

    def _maybe_cleanup_cache(self) -> None:
        """Sweep narrative cache for TTL-expired entries every CACHE_CLEANUP_INTERVAL_S."""
        if self._cache is None:
            return
        now = time.time()
        if now - self._last_cache_cleanup_ts < self.CACHE_CLEANUP_INTERVAL_S:
            return
        try:
            deleted = self._cache.cleanup_expired()
            if deleted > 0:
                logger.info("Narrative cache cleanup: removed %d expired entries", deleted)
        except Exception as e:
            logger.warning("Narrative cache cleanup failed: %s", e)
        finally:
            self._last_cache_cleanup_ts = now

    def scan_once(self) -> Optional[ConfluenceSignal]:
        """Public single-scan method for testing and manual invocation."""
        return self._scan_once()

    def _scan_once(self) -> Optional[ConfluenceSignal]:
        """Execute one scan cycle."""
        # 1. Fetch OHLCV data
        try:
            df = self._data_provider.get_ohlcv(
                self._symbol, self._timeframe, self._lookback
            )
        except Exception as e:
            logger.error("Data provider error: %s", e)
            self._errors += 1
            return None

        if df is None or len(df) < 50:
            logger.debug("Insufficient data (%s bars)", len(df) if df is not None else 0)
            return None

        # 1a. Kill-switch heartbeat — a successful fetch proves the data feed
        # is live, which is what the BROKER_DISCONNECT rule actually monitors.
        if self._kill_switch is not None:
            self._kill_switch.heartbeat()

        # 1b. Data quality gate — refuses structurally broken feeds, warns on
        # soft issues (gaps, stale data). Without this, broker-feed corruption
        # silently corrupts pattern detection downstream.
        from src.intelligence.data_quality import DataQualityError, validate_ohlcv
        try:
            validate_ohlcv(df, self._symbol, self._timeframe, strict=True)
        except DataQualityError as e:
            logger.error("OHLCV quality gate failed: %s", e)
            self._errors += 1
            return None

        # Check for new bar
        bar_ts = str(df.index[-1]) if hasattr(df.index, '__len__') else str(len(df))
        if bar_ts == self._last_bar_ts:
            return None  # Same bar, skip
        self._last_bar_ts = bar_ts
        self._bars_scanned += 1

        # 2. Run SMC analysis
        try:
            smc_engine = self._smc_factory(df)
            enriched = smc_engine.analyze()
        except Exception as e:
            logger.error("SMC analysis error: %s", e)
            self._errors += 1
            return None

        latest = enriched.iloc[-1]
        price = float(latest.get("Close", latest.get("close", 0)))
        atr = float(latest.get("ATR", 0))

        # Extract SMC features from latest bar
        smc_features = {
            "BOS_SIGNAL": float(latest.get("BOS_SIGNAL", 0)),
            "BOS_EVENT": float(latest.get("BOS_EVENT", 0)),
            "BOS_RETEST_STATE": float(latest.get("BOS_RETEST_STATE", 0)),
            "BOS_RETEST_ARMED": float(latest.get("BOS_RETEST_ARMED", 0)),
            "FVG_SIGNAL": float(latest.get("FVG_SIGNAL", 0)),
            "OB_STRENGTH_NORM": float(latest.get("OB_STRENGTH_NORM", 0)),
            "RSI": float(latest.get("RSI", 50)),
            "MACD_Diff": float(latest.get("MACD_Diff", 0)),
            "CHOCH_SIGNAL": float(latest.get("CHOCH_SIGNAL", 0)),
        }

        # Volume
        volume = float(latest.get("Volume", latest.get("volume", 0)))
        vol_col = "Volume" if "Volume" in enriched.columns else "volume"
        volume_ma = float(enriched[vol_col].tail(20).mean()) if vol_col in enriched.columns else None

        # 3. Regime analysis
        regime = None
        try:
            prices = enriched["Close"].values if "Close" in enriched.columns else enriched["close"].values
            highs = enriched["High"].values if "High" in enriched.columns else enriched["high"].values
            lows = enriched["Low"].values if "Low" in enriched.columns else enriched["low"].values
            regime = self._regime_agent.analyze(prices, highs, lows)
        except Exception as e:
            logger.warning("Regime analysis failed: %s", e)

        # 4. News assessment
        news = None
        try:
            from src.agents.events import TradeProposal
            proposal = TradeProposal(
                action="BUY" if smc_features["BOS_SIGNAL"] > 0 else "SELL",
                asset=self._symbol,
                entry_price=price,
            )
            news = self._news_agent.evaluate_news_impact(proposal)
        except Exception as e:
            logger.warning("News analysis failed: %s", e)

        # 5. Volatility forecast (if forecaster available)
        vol_forecast = None
        if self._vol_forecaster is not None:
            try:
                vol_forecast = self._vol_forecaster.forecast(
                    enriched, pd.Timestamp(bar_ts) if bar_ts else None
                )
                logger.debug(
                    "Vol forecast: atr=%.4f (naive=%.4f, regime=%s)",
                    vol_forecast.forecast_atr, vol_forecast.naive_atr,
                    vol_forecast.regime_state,
                )
                # Feed the kill-switch's vol-spike z-score buffer.
                if self._kill_switch is not None and vol_forecast is not None:
                    self._kill_switch.update_volatility(vol_forecast.forecast_atr)
            except Exception as e:
                logger.warning("Vol forecast failed (using naive ATR): %s", e)

        # 5b. Multi-timeframe context — Phase 1 wiring (2026-05-21).
        # The HTF features are descriptive only at weight=0 in
        # ConfluenceDetector for Phase 1; the empirical validation (Phase 2)
        # decides whether to lift the weight in Phase 3. All failures fall
        # back to ``None`` so the renormalisation logic still produces a
        # well-scaled score.
        htf_features = self._compute_htf_features_safe(enriched)

        # 6. Confluence scoring
        signal = self._confluence.analyze(
            smc_features=smc_features,
            regime=regime,
            news=news,
            price=price,
            atr=atr,
            volume=volume,
            volume_ma=volume_ma,
            bar_timestamp=bar_ts,
            vol_forecast=vol_forecast,
            htf_features=htf_features,
        )

        # 6b. Regime filter — drops signals in NY session and top-quartile vol.
        # Validated on 7-yr XAU replay: PF 1.13 → 1.355 OOS without this gate
        # the NY × Q4_high vol bucket alone drains −23R (PF 0.64 on 288 trades).
        if signal is not None and self._regime_filter is not None:
            atr_col = "ATR" if "ATR" in enriched.columns else None
            atr_series = enriched[atr_col] if atr_col else None
            decision = self._regime_filter.evaluate(bar_ts, atr_series)
            if not decision.allowed:
                self._signals_dropped_by_regime_filter += 1
                logger.info(
                    "Signal %s dropped by regime filter: %s",
                    signal.signal_id, decision.reason,
                )
                signal = None

        # 7. State machine gating (optional). When configured, only signals
        # that survive hysteresis, confirmation, and lifetime rules are
        # published — killing flicker and giving the client a stable
        # HOLD / BUY / SELL contract. When not configured, passthrough.
        if self._state_machine is not None:
            transition = self._step_state_machine(
                bar_ts=bar_ts,
                enriched=enriched,
                signal=signal,
                vol_regime=(vol_forecast.regime_state if vol_forecast else None),
            )
            if transition is None:
                # No state change on this bar — nothing to publish.
                return signal
            if transition.to_state is PublicState.HOLD:
                # Exit event: the prior active signal just flipped to HOLD.
                self._publish_exit_transition(transition)
                return signal
            # BUY / SELL entry — use the active_signal from the transition.
            effective_signal = transition.active_signal or signal
            if effective_signal is None:
                return signal
            signal = effective_signal

        if signal is None:
            logger.debug("No signal at bar %s (low confluence or blocked)", bar_ts)
            return None

        self._signals_generated += 1

        # 8. Narrative generation, with optional cache (skipped when None —
        # e.g. template engine where generation is sub-ms so caching is pure
        # overhead and risks stale cross-engine reads).
        narrative_data = None
        cache_key: Optional[str] = None
        if self._cache is not None:
            cache_key = SemanticCache.generate_cache_key(signal)
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._cache_hits += 1
                narrative_data = cached
                logger.debug("Cache hit for signal %s", signal.signal_id)

        if narrative_data is None:
            narrative_data = self._generate_narrative_safe(signal)
            self._llm_calls += 1
            # Cache successful primary-engine results only (skip template
            # fallbacks so a transient LLM outage doesn't poison the cache).
            if (
                self._cache is not None
                and cache_key is not None
                and not narrative_data.get("fallback_used", False)
            ):
                self._cache.put(cache_key, narrative_data)

        # 9. Publish to signal store
        try:
            self._publish_signal(signal, narrative_data)
        except Exception as e:
            logger.error("Signal publish failed: %s", e)

        # 10. Push notification (circuit-breaker protected)
        self._send_notification_safe(signal, narrative_data)

        # 11. Assemble InsightSignalV2 (v2.1.0) for B2B/B2C surfaces. This
        # never blocks the legacy v1 emission path: any failure is logged
        # and counted, but the scanner returns the ConfluenceSignal as
        # before. Skipped when no assembler is wired.
        self._build_insight_safe(
            signal=signal,
            narrative_data=narrative_data,
            smc_features=smc_features,
            regime=regime,
            news=news,
            vol_forecast=vol_forecast,
            htf_features=htf_features,
        )

        return signal

    # ------------------------------------------------------------------ #
    # STATE MACHINE STEP
    # ------------------------------------------------------------------ #

    def _step_state_machine(
        self,
        bar_ts: str,
        enriched: pd.DataFrame,
        signal: Optional[ConfluenceSignal],
        vol_regime: Optional[str],
    ) -> Optional[StateTransition]:
        """Feed the latest bar into the state machine. Returns any transition.

        All failures are caught defensively — the state machine must never
        take down the scanner loop.
        """
        try:
            latest = enriched.iloc[-1]
            high = float(latest.get("High", latest.get("high", 0)))
            low = float(latest.get("Low", latest.get("low", 0)))
            close = float(latest.get("Close", latest.get("close", 0)))
            bar = BarInput(
                bar_timestamp=bar_ts,
                high=high,
                low=low,
                close=close,
                signal=signal,
                vol_regime=vol_regime,
            )
            _, transition = self._state_machine.on_bar(bar)
            if transition is not None:
                self._state_transitions_emitted += 1
            elif signal is not None:
                # Signal was generated but state machine suppressed publication.
                self._signals_held_by_state_machine += 1
            return transition
        except Exception as e:
            logger.error("State machine step failed at bar %s: %s", bar_ts, e, exc_info=True)
            self._errors += 1
            return None

    def _publish_exit_transition(self, transition: StateTransition) -> None:
        """Record an ACTIVE → HOLD transition to the signal store + notify.

        We do NOT create a new SignalRecord for HOLD. Instead we update the
        originating signal's outcome so history shows exactly why it ended.
        """
        sig = transition.active_signal
        if sig is None or transition.exit_reason is None:
            return
        signal_id = getattr(sig, "signal_id", None)
        if not signal_id:
            return
        # Compute pnl in pips/points from entry → exit
        entry = getattr(sig, "entry_price", None)
        exit_px = transition.exit_price
        pnl: Optional[float] = None
        if entry is not None and exit_px is not None:
            delta = exit_px - float(entry)
            direction = transition.direction.value if transition.direction else "LONG"
            pnl = delta if direction == "LONG" else -delta
        try:
            self._signal_store.update_outcome(
                signal_id=signal_id,
                outcome=transition.exit_reason.value,
                pnl_pips=float(pnl) if pnl is not None else 0.0,
            )
        except Exception as e:
            logger.error("Exit outcome update failed for %s: %s", signal_id, e)

        # Notify the exit (best-effort)
        if self._notifier is not None:
            try:
                notify_fn = getattr(self._notifier, "send_exit", None)
                if callable(notify_fn):
                    def _call_exit_notify():
                        notify_fn(sig, transition.exit_reason.value, exit_px)
                    if self._notifier_breaker is not None:
                        self._notifier_breaker.call(_call_exit_notify)
                    else:
                        _call_exit_notify()
            except CircuitOpenError:
                self._notification_failures += 1
            except Exception as e:
                self._notification_failures += 1
                logger.warning("Exit notification failed for %s: %s", signal_id, e)

        logger.info(
            "State exit: %s %s → HOLD (reason=%s, exit=%.4f)",
            self._symbol,
            transition.direction.value if transition.direction else "?",
            transition.exit_reason.value,
            float(exit_px) if exit_px is not None else float("nan"),
        )

    # ------------------------------------------------------------------ #
    # CIRCUIT-BREAKER WRAPPERS
    # ------------------------------------------------------------------ #

    def _generate_narrative_safe(self, signal: ConfluenceSignal) -> Dict:
        """Generate narrative with circuit-breaker protection + algo fallback.

        Always returns a dict. On LLM failure or open circuit, falls back to
        the deterministic ``TemplateNarrativeEngine`` so notifications never
        ship with a degraded "fallback:true" stub.
        """
        def _call_llm():
            narrative = self._llm_engine.generate_narrative(signal, self._narrative_tier)
            return narrative.to_dict()

        try:
            if self._llm_breaker is not None:
                return self._llm_breaker.call(_call_llm)
            return _call_llm()
        except CircuitOpenError:
            self._llm_failures += 1
            logger.warning(
                "LLM circuit OPEN for signal %s — falling back to TemplateNarrativeEngine",
                signal.signal_id,
            )
            return self._template_fallback(signal, reason="circuit_open")
        except Exception as e:
            self._llm_failures += 1
            logger.error(
                "LLM narrative failed for %s: %s — falling back to template",
                signal.signal_id, e,
            )
            return self._template_fallback(signal, reason=f"llm_error:{type(e).__name__}")

    def _template_fallback(self, signal: ConfluenceSignal, reason: str) -> Dict:
        """Run the deterministic template engine; tag result so we can audit fallbacks."""
        try:
            narrative = self._fallback_engine.generate_narrative(
                signal, self._narrative_tier
            )
            self._fallback_uses += 1
            data = narrative.to_dict()
            data["fallback_used"] = True
            data["fallback_reason"] = reason
            return data
        except Exception as e:
            # Last-resort minimal stub — only if even the template engine raises.
            logger.error("Template fallback also failed for %s: %s", signal.signal_id, e)
            return {
                "tier": self._narrative_tier.value,
                "summary": (
                    f"{signal.signal_type.value} {signal.symbol} "
                    f"— score {signal.confluence_score:.0f}"
                ),
                "fallback_used": True,
                "fallback_reason": f"template_error:{type(e).__name__}",
            }

    def _send_notification_safe(self, signal: ConfluenceSignal, narrative_data: Dict) -> None:
        """Send notification with circuit breaker protection."""
        if self._notifier is None:
            return

        def _call_notifier():
            self._notifier.send_signal(signal, narrative_data)

        try:
            if self._notifier_breaker is not None:
                self._notifier_breaker.call(_call_notifier)
            else:
                _call_notifier()
        except CircuitOpenError:
            self._notification_failures += 1
            logger.warning(
                "Telegram circuit OPEN — skipping notification for %s",
                signal.signal_id,
            )
        except Exception as e:
            self._notification_failures += 1
            logger.warning("Notification failed for %s: %s", signal.signal_id, e)

    # ------------------------------------------------------------------ #
    # PUBLISH
    # ------------------------------------------------------------------ #

    def _publish_signal(self, signal: ConfluenceSignal, narrative_data: Dict) -> None:
        """Persist signal to the signal store.

        Gated by the kill-switch when one is configured: a tripped switch
        drops the signal and emits an audit log entry. The signal store
        is not touched, so downstream consumers (dashboard, Telegram)
        never see signals that the operator has paused.
        """
        # Kill-switch gate (operational risk): refuse to publish when any
        # of the four hard rules has tripped (consecutive losses, daily DD,
        # volatility spike, broker disconnect).
        if self._kill_switch is not None and not self._kill_switch.check():
            self._signals_blocked_by_kill_switch += 1
            status = self._kill_switch.status()
            logger.error(
                "Kill-switch tripped (%s) — DROPPING signal %s [%s]: %s",
                status.get("reason"), signal.signal_id, signal.symbol,
                status.get("detail"),
            )
            return

        import json
        from src.api.signal_store import SignalRecord

        action = "OPEN_LONG" if signal.signal_type.value == "LONG" else "OPEN_SHORT"

        # Serialize volatility confidence interval as JSON
        vol_confidence = None
        if signal.vol_confidence_lower is not None and signal.vol_confidence_upper is not None:
            vol_confidence = json.dumps({
                "lower": signal.vol_confidence_lower,
                "upper": signal.vol_confidence_upper,
            })

        record = SignalRecord(
            signal_id=signal.signal_id,
            action=action,
            symbol=signal.symbol,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            rr_ratio=signal.rr_ratio,
            created_at=signal.created_at,
            vol_forecast_atr=signal.vol_forecast_atr,
            vol_regime=signal.vol_regime,
            vol_confidence=vol_confidence,
        )
        self._signal_store.publish(record)
        logger.info(
            "Published %s signal: %s score=%.1f tier=%s entry=%.2f SL=%.2f TP=%.2f",
            signal.signal_type.value,
            signal.signal_id,
            signal.confluence_score,
            signal.tier.value,
            signal.entry_price,
            signal.stop_loss,
            signal.take_profit,
        )

    # ------------------------------------------------------------------ #
    # INSIGHT V2.1.0 ASSEMBLY
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # MULTI-TIMEFRAME FEATURE COMPUTATION
    # ------------------------------------------------------------------ #

    def _compute_htf_features_safe(
        self, enriched: pd.DataFrame
    ) -> Optional[Dict[str, float]]:
        """Compute HTF features for the latest bar. Defensive.

        Returns ``None`` if (a) the dataframe is too short for the H4 SMA50
        warm-up, (b) the MTF module is unavailable, or (c) any computation
        raises. Phase 1: weight=0 in ConfluenceDetector, so missing HTF
        features change neither tier nor the score (renormalisation handles
        absence). Phase 2 validation harness reads this dict directly.
        """
        # Lazy import — keep cold-start fast and avoid coupling tests that
        # don't touch the scanner directly to the MTF dependency tree.
        if self._mtf is None:
            try:
                from src.environment.multi_timeframe_features import (
                    MultiTimeframeFeatures,
                )
                # Base TF mapping — only M15 is currently supported in the
                # MTF module; for other base TFs we degrade gracefully.
                base_tf = "15min" if self._timeframe.upper() in ("M15", "15M", "15MIN") else None
                if base_tf is None:
                    logger.debug(
                        "MTF disabled: unsupported base timeframe %s", self._timeframe
                    )
                    self._htf_features_skipped += 1
                    return None
                self._mtf = MultiTimeframeFeatures(
                    base_timeframe=base_tf,
                    include_1h=True,
                    include_4h=True,
                    include_session=True,
                )
            except Exception as exc:
                logger.debug("MTF init failed: %s", exc)
                self._htf_features_skipped += 1
                return None

        # Need enough M15 bars (200 H4 bars × 16 M15/H4) for stable H4 SMA50.
        # We use a softer guard than ``lookback_bars`` since cold-start
        # backtests can start with less. Below 800 M15 bars we return None
        # and let the absent_weight renormalisation in ConfluenceDetector
        # handle it; above 800 we mark warm-up complete.
        if len(enriched) < 800:
            if not self._mtf_warmup_complete:
                self._htf_features_skipped += 1
                return None
            # Already warm: continue with whatever we have (live data can
            # transiently shrink if a provider returns fewer bars).

        try:
            # The MTF module wants a DatetimeIndex; ``enriched`` from SMC
            # carries either a DatetimeIndex (live) or a RangeIndex (some
            # tests). The module handles both via the fit() branch.
            self._mtf.fit(enriched)
            self._mtf_warmup_complete = True
            features = self._mtf.get_features(idx=len(enriched) - 1)
            self._htf_features_computed += 1
            return features
        except Exception as exc:
            logger.debug("MTF compute failed: %s", exc)
            self._htf_features_skipped += 1
            return None

    def _build_insight_safe(
        self,
        signal: ConfluenceSignal,
        narrative_data: Dict,
        smc_features: Dict[str, float],
        regime: Any,
        news: Any,
        vol_forecast: Any,
        htf_features: Optional[Dict[str, float]] = None,
    ) -> None:
        """Compose an InsightSignalV2 from pipeline outputs. Defensive.

        Stores the result on ``self._latest_insight`` so API/notifier
        layers can read it. All failures are caught — the scanner must
        never crash because of v2 assembly. The legacy v1 publish path is
        unaffected.
        """
        if self._insight_assembler is None:
            return
        try:
            narrative_short = (
                narrative_data.get("summary")
                or narrative_data.get("narrative_short")
                or ""
            )
            narrative_long = (
                narrative_data.get("full_narrative")
                or narrative_data.get("narrative_long")
                or narrative_data.get("market_context")
                or ""
            )
            # Feature vector for the calibrated pipeline (LGBM input).
            # Built from ConfluenceSignal.components, ordered to match
            # DEFAULT_FEATURE_NAMES in scoring/lgbm_scorer.py.
            feature_vector = _components_to_feature_vector(signal)
            self._latest_insight = self._insight_assembler.assemble(
                instrument=self._symbol,
                timeframe=self._timeframe,
                confluence_signal=signal,
                smc_features=smc_features,
                volatility_forecast=vol_forecast,
                regime_analysis=regime,
                news_assessment=news,
                htf_features=htf_features,
                narrative_short=str(narrative_short),
                narrative_long=str(narrative_long),
                feature_vector=feature_vector,
                include_levels=False,
            )
            self._insights_built += 1
        except Exception as exc:  # pragma: no cover — defensive
            self._insight_build_failures += 1
            logger.warning(
                "InsightSignalV2 assembly failed for %s: %s",
                getattr(signal, "signal_id", "?"), exc,
            )

    # ------------------------------------------------------------------ #
    # TRADE OUTCOME HOOK
    # ------------------------------------------------------------------ #

    def on_trade_closed(self, r_multiple: float, pnl_dollars: float = 0.0) -> None:
        """Notify the kill-switch that a trade just closed.

        Callable from the position-tracker, Telegram callback, or any
        adapter that observes trade exits. ``r_multiple`` should be
        signed (negative for losses, positive for wins). ``pnl_dollars``
        feeds the daily-DD rule.

        No-op when no kill-switch is configured.
        """
        if self._kill_switch is None:
            return
        try:
            self._kill_switch.record_trade_outcome(
                r_multiple=r_multiple, pnl_dollars=pnl_dollars,
            )
        except Exception as e:  # pragma: no cover — defensive
            logger.error("kill_switch.record_trade_outcome failed: %s", e)

    # ------------------------------------------------------------------ #
    # STATS
    # ------------------------------------------------------------------ #

    def get_stats(self) -> Dict[str, Any]:
        uptime = time.time() - self._start_time if self._start_time else 0
        stats = {
            "running": self._running,
            "symbol": self._symbol,
            "timeframe": self._timeframe,
            "uptime_seconds": round(uptime, 1),
            "bars_scanned": self._bars_scanned,
            "signals_generated": self._signals_generated,
            "signals_held_by_state_machine": self._signals_held_by_state_machine,
            "state_transitions_emitted": self._state_transitions_emitted,
            "cache_hits": self._cache_hits,
            "llm_calls": self._llm_calls,
            "llm_failures": self._llm_failures,
            "notification_failures": self._notification_failures,
            "errors": self._errors,
            "last_bar_ts": self._last_bar_ts,
        }
        if self._llm_breaker is not None:
            stats["llm_circuit"] = self._llm_breaker.state.value
        if self._notifier_breaker is not None:
            stats["telegram_circuit"] = self._notifier_breaker.state.value
        if self._insight_assembler is not None:
            stats["insights_built"] = self._insights_built
            stats["insight_build_failures"] = self._insight_build_failures
            stats["latest_insight_id"] = (
                self._latest_insight.id if self._latest_insight is not None else None
            )
        if self._kill_switch is not None:
            stats["kill_switch"] = self._kill_switch.status()
            stats["signals_blocked_by_kill_switch"] = self._signals_blocked_by_kill_switch
        if self._regime_filter is not None:
            stats["regime_filter"] = self._regime_filter.stats()
            stats["signals_dropped_by_regime_filter"] = self._signals_dropped_by_regime_filter
        if self._state_machine is not None:
            sm_stats = self._state_machine.get_stats()
            sm_snap = self._state_machine.snapshot()
            stats["state_machine"] = {
                "public_state": sm_snap.state.value,
                "direction": sm_snap.direction.value if sm_snap.direction else None,
                "bars_in_state": sm_snap.bars_in_state,
                "bars_remaining": sm_snap.bars_remaining,
                "confirmation_progress": (
                    list(sm_snap.confirmation_progress)
                    if sm_snap.confirmation_progress else None
                ),
                "last_exit_reason": (
                    sm_snap.last_exit_reason.value
                    if sm_snap.last_exit_reason else None
                ),
                "signals_emitted": sm_stats["signals_emitted"],
                "exits_by_reason": sm_stats["exits_by_reason"],
                "avg_signal_lifetime_bars": sm_stats["avg_signal_lifetime_bars"],
                "confirmation_rate": sm_stats["confirmation_rate"],
            }
        return stats


# =============================================================================
# MULTI-SYMBOL SCANNER
# =============================================================================

class MultiSymbolScanner:
    """Manages multiple SentinelScanner instances, one per symbol.

    Coordinates scanning across symbols with shared resources
    (LLM engine, cache, signal store) and per-symbol state
    (vol forecaster, confluence detector, last bar tracking).

    Usage:
        from src.intelligence.volatility_forecaster import InstrumentConfig, get_instrument_registry

        registry = get_instrument_registry()
        multi = MultiSymbolScanner(
            symbols=["XAUUSD", "EURUSD", "BTCUSD"],
            instrument_registry=registry,
            data_provider=provider,
            smc_factory=factory,
            regime_agent=regime,
            news_agent=news,
            llm_engine=llm,
            cache=cache,
            signal_store=store,
        )
        multi.start()
    """

    def __init__(
        self,
        symbols: List[str],
        instrument_registry: Dict[str, Any],
        data_provider: Any,
        smc_factory: Callable[[pd.DataFrame], Any],
        regime_agent: Any,
        news_agent: Any,
        llm_engine: LLMNarrativeEngine,
        cache: Optional[SemanticCache],
        signal_store: Any,
        notifier: Optional[Any] = None,
        vol_forecaster_factory: Optional[Callable[[Any], Any]] = None,
        narrative_tier: NarrativeTier = NarrativeTier.NARRATOR,
        poll_interval_seconds: float = 60.0,
        state_machine_factory: Optional[Callable[[str], SignalStateMachine]] = None,
        state_persistence_dir: Optional[Any] = None,
        insight_assembler_factory: Optional[Callable[[str], InsightAssembler]] = None,
    ):
        self._symbols = symbols
        self._registry = instrument_registry
        self._scanners: Dict[str, SentinelScanner] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._poll_interval = poll_interval_seconds
        self._start_time: Optional[float] = None
        self._stop_event = threading.Event()
        from pathlib import Path as _Path
        self._persistence_dir: Optional[_Path] = (
            _Path(state_persistence_dir) if state_persistence_dir else None
        )

        # Create per-symbol scanners with shared resources
        for symbol in symbols:
            config = instrument_registry.get(symbol)

            # Per-symbol ConfluenceDetector with instrument-specific SL/TP
            confluence = ConfluenceDetector(
                symbol=symbol,
                instrument_config=config,
            )

            # Per-symbol vol forecaster if factory provided
            vol_forecaster = None
            if vol_forecaster_factory is not None and config is not None:
                vol_forecaster = vol_forecaster_factory(config)

            # Per-symbol state machine if factory provided
            sm = state_machine_factory(symbol) if state_machine_factory is not None else None

            # Per-symbol insight assembler if factory provided. The factory
            # signature receives the symbol so callers can wire
            # symbol-specific historical_stats callbacks (e.g. EUR vs XAU
            # backtest stats), or share a single instance by returning the
            # same object every call.
            assembler = (
                insight_assembler_factory(symbol)
                if insight_assembler_factory is not None else None
            )

            # Per-symbol persistence path under the shared directory
            persistence_path = (
                self._persistence_dir / f"state_{symbol}.json"
                if self._persistence_dir is not None else None
            )

            timeframe = getattr(config, "timeframe", "M15") if config else "M15"

            self._scanners[symbol] = SentinelScanner(
                data_provider=data_provider,
                smc_factory=smc_factory,
                regime_agent=regime_agent,
                news_agent=news_agent,
                confluence=confluence,
                llm_engine=llm_engine,
                cache=cache,
                signal_store=signal_store,
                notifier=notifier,
                vol_forecaster=vol_forecaster,
                symbol=symbol,
                timeframe=timeframe,
                narrative_tier=narrative_tier,
                poll_interval_seconds=0,  # We manage polling at the multi level
                state_machine=sm,
                persistence_path=persistence_path,
                insight_assembler=assembler,
            )

        logger.info(
            "MultiSymbolScanner initialized: %d symbols (%s)",
            len(symbols), ", ".join(symbols),
        )

    @property
    def symbols(self) -> List[str]:
        return list(self._symbols)

    @property
    def scanners(self) -> Dict[str, SentinelScanner]:
        return self._scanners

    # ------------------------------------------------------------------ #
    # LIFECYCLE
    # ------------------------------------------------------------------ #

    def start(self, blocking: bool = True) -> None:
        """Start scanning all symbols.

        Rehydrates each per-symbol state machine from its persistence
        snapshot (if one exists) before the scanning loop begins.
        """
        # Restore per-symbol state before the first scan runs
        for scanner in self._scanners.values():
            scanner._restore_state_machine()
        self._running = True
        self._start_time = time.time()
        logger.info("MultiSymbolScanner starting: %s", self._symbols)

        if blocking:
            self._run_loop()
        else:
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()

    def shutdown(self) -> None:
        """Graceful shutdown — persists each per-symbol state machine."""
        self._running = False
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=15)
        # Persist state for every symbol whose scanner has a persistence path
        for scanner in self._scanners.values():
            scanner._persist_state_machine()
        logger.info("MultiSymbolScanner stopped. Stats: %s", self.get_stats())

    def _run_loop(self) -> None:
        while self._running and not self._stop_event.is_set():
            self.scan_all_once()
            if self._running:
                self._stop_event.wait(timeout=self._poll_interval)

    # ------------------------------------------------------------------ #
    # SCANNING
    # ------------------------------------------------------------------ #

    def scan_all_once(self) -> Dict[str, Optional[ConfluenceSignal]]:
        """Scan all symbols once. Returns {symbol: signal_or_none}."""
        results: Dict[str, Optional[ConfluenceSignal]] = {}
        for symbol, scanner in self._scanners.items():
            try:
                signal = scanner.scan_once()
                results[symbol] = signal
            except Exception as e:
                logger.error("Error scanning %s: %s", symbol, e)
                results[symbol] = None
        return results

    def scan_symbol(self, symbol: str) -> Optional[ConfluenceSignal]:
        """Scan a single symbol. Raises KeyError if symbol not registered."""
        if symbol not in self._scanners:
            raise KeyError(f"Symbol {symbol} not registered. Available: {list(self._scanners.keys())}")
        return self._scanners[symbol].scan_once()

    # ------------------------------------------------------------------ #
    # CALIBRATION
    # ------------------------------------------------------------------ #

    def calibrate_forecasters(
        self,
        ohlcv_data: Dict[str, pd.DataFrame],
        calendar_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Calibrate vol forecasters for all symbols with available data.

        Args:
            ohlcv_data: {symbol: ohlcv_dataframe}
            calendar_df: Shared economic calendar.

        Returns:
            {symbol: calibration_stats}
        """
        results = {}
        for symbol, scanner in self._scanners.items():
            if scanner._vol_forecaster is None:
                results[symbol] = {"calibrated": False, "reason": "no forecaster"}
                continue
            if symbol not in ohlcv_data:
                results[symbol] = {"calibrated": False, "reason": "no data"}
                continue
            try:
                stats = scanner._vol_forecaster.calibrate(
                    ohlcv_data[symbol], calendar_df
                )
                results[symbol] = stats
                logger.info("Calibrated %s: %s", symbol, stats.get("blend_weight", "n/a"))
            except Exception as e:
                results[symbol] = {"calibrated": False, "error": str(e)}
                logger.error("Calibration failed for %s: %s", symbol, e)
        return results

    # ------------------------------------------------------------------ #
    # STATS
    # ------------------------------------------------------------------ #

    def get_stats(self) -> Dict[str, Any]:
        uptime = time.time() - self._start_time if self._start_time else 0
        per_symbol = {sym: scanner.get_stats() for sym, scanner in self._scanners.items()}
        total_signals = sum(s.get("signals_generated", 0) for s in per_symbol.values())
        total_errors = sum(s.get("errors", 0) for s in per_symbol.values())
        return {
            "running": self._running,
            "symbols": self._symbols,
            "uptime_seconds": round(uptime, 1),
            "total_signals": total_signals,
            "total_errors": total_errors,
            "per_symbol": per_symbol,
        }
