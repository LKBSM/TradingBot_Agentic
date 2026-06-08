"""High-level assembler — composes an InsightSignalV2 from pipeline outputs.

This is the single integration point between the internal pipeline objects
(``ConfluenceSignal``, ``VolatilityForecast``, ``RegimeGateOutput``,
``NewsAssessment``) and the v2.1.0 client contract.

Usage from :class:`SentinelScanner` is one call::

    assembler = InsightAssembler(
        instrument_config=...,
        calibrated_pipeline=load_calibrated_pipeline(...),
        historical_stats=lambda symbol: ...,  # callback or None
    )
    insight = assembler.assemble(
        confluence_signal=cs,
        smc_features=smc_features_dict,
        volatility_forecast=vf,
        regime_analysis=ra,
        regime_gate_output=gate,
        news_assessment=news,
        narrative_short="...",
        narrative_long="...",
        session_label="new_york",
    )
    notifier.send(to_telegram_b2c(insight))

Design rules
------------
- **Standalone.** No I/O, no threading, no globals. Fully testable.
- **Defensive.** Any input can be ``None`` — the corresponding readout is
  simply omitted from the final InsightSignal.
- **Compliance-safe.** Final InsightSignal always has ``edge_claim=False``
  and ``is_paper_demo=True`` unless explicitly overridden by caller.
- **Indicator stance.** Levels (entry/stop/target) are accepted as input
  for backward-compat with v2.0.0 consumers but they are NOT required.
  Production B2C surfaces should pass ``include_levels=False``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional

from src.api.insight_signal_v2 import (
    ComplianceMeta,
    InsightSignalV2,
    NarrativeLanguage,
    SetupDirection,
    SignalLevels,
    Timeframe,
)
from src.intelligence.readout_mappers import (
    map_breakdown_components,
    map_event_readout,
    map_historical_stats,
    map_mtf_readout,
    map_regime_readout,
    map_structure_readout,
    map_uncertainty_context,
    map_volatility_readout,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default historical stats callback signature
# ---------------------------------------------------------------------------


HistoricalStatsFn = Callable[[str], Optional[dict]]


@dataclass(frozen=True)
class AssemblerDefaults:
    """Default compliance + lifecycle settings applied to every assembled signal."""

    edge_claim: bool = False
    is_paper_demo: bool = True
    disclaimer_lang: NarrativeLanguage = NarrativeLanguage.FR
    validity_hours: int = 4
    expose_component_weights: bool = True  # True for B2B, False for B2C surface


# ---------------------------------------------------------------------------
# InsightAssembler
# ---------------------------------------------------------------------------


class InsightAssembler:
    """Compose an InsightSignalV2 from internal pipeline objects."""

    def __init__(
        self,
        calibrated_pipeline: Optional[Any] = None,
        historical_stats_fn: Optional[HistoricalStatsFn] = None,
        defaults: Optional[AssemblerDefaults] = None,
        backtest_window: str = "XAU M15 2019-2025 walk-forward",
    ):
        self._calibrated_pipeline = calibrated_pipeline
        self._historical_stats_fn = historical_stats_fn
        self._defaults = defaults or AssemblerDefaults()
        self._backtest_window = backtest_window

    # ------------------------------------------------------------------ #
    # MAIN ENTRY POINT
    # ------------------------------------------------------------------ #

    def assemble(
        self,
        *,
        instrument: str,
        timeframe: str,
        confluence_signal: Optional[Any],
        smc_features: Optional[dict] = None,
        volatility_forecast: Optional[Any] = None,
        regime_analysis: Optional[Any] = None,
        regime_gate_output: Optional[Any] = None,
        news_assessment: Optional[Any] = None,
        htf_features: Optional[dict] = None,
        session_label: Optional[str] = None,
        narrative_short: str = "",
        narrative_long: str = "",
        narrative_language: NarrativeLanguage = NarrativeLanguage.FR,
        feature_vector: Optional[Any] = None,
        include_levels: bool = False,
        edge_claim: Optional[bool] = None,
        is_paper_demo: Optional[bool] = None,
    ) -> InsightSignalV2:
        """Assemble a complete v2.1.0 InsightSignal from pipeline outputs.

        Parameters
        ----------
        instrument, timeframe : str
            Mandatory metadata.
        confluence_signal : ConfluenceSignal, optional
            If provided, used to derive direction, breakdown, and levels.
            If None, the assembled signal will be NEUTRAL with empty readouts.
        smc_features : dict, optional
            Per-bar SMC features for the StructureReadout.
        volatility_forecast, regime_analysis, regime_gate_output, news_assessment :
            Pipeline outputs — each independently optional.
        session_label : str, optional
            Trading session label fed into EventReadout.
        narrative_short, narrative_long : str
            Generated by the LLM narrative engine (outside this assembler).
        feature_vector : np.ndarray or list, optional
            8-feature vector for the calibrated pipeline. If None, conviction
            falls back to the raw confluence score (legacy behaviour).
        include_levels : bool, default False
            **Indicator stance**: by default we do NOT propagate
            entry/stop/target to the InsightSignal. Pass True only when
            consuming a legacy v2.0.0 client.
        edge_claim, is_paper_demo : bool, optional
            Override the default compliance flags. Use with extreme caution
            for edge_claim — must only become True after empirical edge
            validation (currently False per A1 verdict).

        Returns
        -------
        InsightSignalV2 (v2.1.0)
        """
        # Determine direction
        direction = self._derive_direction(confluence_signal)

        # Determine conviction (calibrated if pipeline available, else raw fallback)
        conviction_int, calibrated_cc = self._derive_conviction(
            confluence_signal=confluence_signal,
            feature_vector=feature_vector,
        )

        # Direction hint for regime readout
        sig_type = getattr(confluence_signal, "signal_type", None)
        sig_val = getattr(sig_type, "value", sig_type)
        direction_hint = sig_val if sig_val in ("LONG", "SHORT") else None

        # Build readouts via mappers
        structure = map_structure_readout(confluence_signal, smc_features)
        regime = map_regime_readout(regime_analysis, regime_gate_output, direction_hint)
        volatility = map_volatility_readout(volatility_forecast)
        event = map_event_readout(news_assessment, session_label)
        mtf = map_mtf_readout(htf_features, direction_hint)
        breakdown = map_breakdown_components(
            confluence_signal,
            expose_weights=self._defaults.expose_component_weights,
        )
        uncertainty = map_uncertainty_context(calibrated_cc)

        # Historical stats — pulled via callback if provided
        hist_stats = None
        if self._historical_stats_fn is not None:
            try:
                stats_dict = self._historical_stats_fn(instrument)
                if stats_dict:
                    hist_stats = map_historical_stats(
                        similar_setups_n=stats_dict.get("similar_setups_n"),
                        hit_rate=stats_dict.get("hit_rate"),
                        profit_factor=stats_dict.get("profit_factor"),
                        profit_factor_ci95=stats_dict.get("profit_factor_ci95"),
                        empirical_coverage=stats_dict.get("empirical_coverage"),
                        backtest_window=stats_dict.get("backtest_window") or self._backtest_window,
                    )
            except Exception as exc:
                logger.warning("Historical stats callback failed: %s", exc)

        # Levels: only when explicitly requested (legacy v2.0.0 consumers)
        levels = self._build_levels(confluence_signal, direction, include_levels)

        # Lifecycle
        created_at = self._derive_created_at(confluence_signal)
        valid_until = created_at + timedelta(hours=self._defaults.validity_hours)

        # Identity — prefer ConfluenceSignal's signal_id (deterministic SHA-1)
        signal_id = getattr(confluence_signal, "signal_id", None) or "neutral_no_setup"

        # Compliance
        compliance = ComplianceMeta(
            disclaimer_lang=self._defaults.disclaimer_lang
            if narrative_language is None else narrative_language,
            edge_claim=self._defaults.edge_claim if edge_claim is None else edge_claim,
            is_paper_demo=self._defaults.is_paper_demo if is_paper_demo is None else is_paper_demo,
        )

        return InsightSignalV2(
            id=str(signal_id),
            instrument=instrument,
            timeframe=Timeframe(timeframe),
            direction=direction,
            conviction_0_100=conviction_int,
            levels=levels,
            uncertainty=uncertainty,
            structure_readout=structure,
            regime_readout=regime,
            volatility_readout=volatility,
            event_readout=event,
            mtf_readout=mtf,
            breakdown_components=breakdown,
            historical_stats=hist_stats,
            narrative_short=narrative_short or self._fallback_narrative(direction, narrative_language),
            narrative_long=narrative_long,
            narrative_language=narrative_language,
            compliance=compliance,
            created_at_utc=created_at,
            valid_until_utc=valid_until,
        )

    # ------------------------------------------------------------------ #
    # Online feedback (ACI loop closure when an outcome is realised)
    # ------------------------------------------------------------------ #

    def observe_outcome(self, realised_r_multiple: float) -> None:
        """Forward a realised outcome to the calibrated pipeline's ACI scorer.

        Call this when a previously-published signal's outcome becomes known
        (target hit, invalidation, expiry). Keeps the conformal coverage
        guarantee under regime drift.
        """
        if self._calibrated_pipeline is not None:
            try:
                self._calibrated_pipeline.observe_outcome(realised_r_multiple)
            except Exception as exc:
                logger.warning("observe_outcome failed: %s", exc)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    @staticmethod
    def _derive_direction(confluence_signal: Optional[Any]) -> SetupDirection:
        if confluence_signal is None:
            return SetupDirection.NEUTRAL
        sig_type = getattr(confluence_signal, "signal_type", None)
        val = getattr(sig_type, "value", sig_type)
        if val == "LONG":
            return SetupDirection.BULLISH_SETUP
        if val == "SHORT":
            return SetupDirection.BEARISH_SETUP
        return SetupDirection.NEUTRAL

    def _derive_conviction(
        self,
        confluence_signal: Optional[Any],
        feature_vector: Optional[Any],
    ) -> tuple[int, Optional[Any]]:
        """Return (conviction_0_100, optional CalibratedConviction).

        Order of preference:
          1. Calibrated pipeline result (if pipeline fitted and features given)
          2. Raw confluence_score from confluence_signal
          3. 50 (neutral)
        """
        if (
            self._calibrated_pipeline is not None
            and feature_vector is not None
        ):
            try:
                cc = self._calibrated_pipeline.score_one(feature_vector)
                if not cc.is_fallback:
                    return int(cc.conviction_0_100), cc
                # Even on fallback we return the cc so the conformal interval
                # surfaces a degenerate [0, 100] band (visible "unknown").
                if confluence_signal is None:
                    return int(cc.conviction_0_100), cc
                # Fall through to use raw score but keep the degenerate cc
            except Exception as exc:
                logger.warning("Calibrated pipeline score_one failed: %s", exc)

        if confluence_signal is not None:
            raw = getattr(confluence_signal, "confluence_score", None)
            if raw is not None:
                return int(round(max(0.0, min(100.0, float(raw))))), None

        return 50, None

    def _build_levels(
        self,
        confluence_signal: Optional[Any],
        direction: SetupDirection,
        include_levels: bool,
    ) -> SignalLevels:
        """Build SignalLevels — empty by default (indicator stance)."""
        if not include_levels or direction == SetupDirection.NEUTRAL or confluence_signal is None:
            return SignalLevels()
        return SignalLevels(
            entry=getattr(confluence_signal, "entry_price", None),
            stop=getattr(confluence_signal, "stop_loss", None),
            target_1=getattr(confluence_signal, "take_profit", None),
        )

    @staticmethod
    def _derive_created_at(confluence_signal: Optional[Any]) -> datetime:
        """Try the ConfluenceSignal.bar_timestamp / created_at, else now()."""
        if confluence_signal is not None:
            for attr in ("bar_timestamp", "created_at"):
                ts = getattr(confluence_signal, attr, None)
                if ts is None:
                    continue
                if isinstance(ts, datetime):
                    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
                if isinstance(ts, str):
                    try:
                        parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
                    except ValueError:
                        continue
        return datetime.now(timezone.utc)

    @staticmethod
    def _fallback_narrative(direction: SetupDirection, language: NarrativeLanguage) -> str:
        """Minimal narrative when LLM hasn't generated one yet."""
        if language == NarrativeLanguage.FR:
            label_map = {
                SetupDirection.BULLISH_SETUP: "Lecture de marché haussière algorithmique.",
                SetupDirection.BEARISH_SETUP: "Lecture de marché baissière algorithmique.",
                SetupDirection.NEUTRAL: "Contexte neutre. Le système attend une nouvelle structure.",
            }
        else:
            label_map = {
                SetupDirection.BULLISH_SETUP: "Bullish algorithmic market readout.",
                SetupDirection.BEARISH_SETUP: "Bearish algorithmic market readout.",
                SetupDirection.NEUTRAL: "Neutral context. Awaiting structural break.",
            }
        return label_map[direction]


__all__ = ["InsightAssembler", "AssemblerDefaults", "HistoricalStatsFn"]
