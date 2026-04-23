"""Tests for Sprint 2: Pipeline Integration.

Tests cover:
  - ConfluenceDetector accepts vol_forecast and uses it for SL/TP
  - ConfluenceDetector high-regime SL widening
  - ConfluenceSignal includes vol fields
  - SentinelScanner accepts vol_forecaster
  - SignalRecord schema v3 with vol fields
  - AppState includes vol_forecaster
  - NarrativeResponse includes vol fields
"""

import json
import sqlite3
import tempfile
from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.intelligence.confluence_detector import (
    ConfluenceDetector,
    ConfluenceSignal,
    SignalTier,
    SignalType,
    SL_ATR_MULT,
    TP_ATR_MULT,
)
from src.intelligence.volatility_forecaster import (
    InstrumentConfig,
    VolatilityForecast,
    VolatilityForecaster,
)


# =============================================================================
# HELPERS
# =============================================================================

def _make_smc_features(bos: float = 1.0, fvg: float = 1.0) -> dict:
    """Create minimal SMC features for a LONG signal."""
    return {
        "BOS_SIGNAL": bos,
        "FVG_SIGNAL": fvg,
        "FVG_SIZE_NORM": 0.5,
        "OB_STRENGTH_NORM": 0.7,
        "RSI": 60.0,
        "MACD_Diff": 0.005,
        "CHOCH_SIGNAL": 1.0,
        "CHOCH_DIVERGENCE": 1,
    }


@dataclass
class MockRegime:
    regime: str = "strong_uptrend"
    trend_direction: str = "UP"
    confidence: float = 0.8
    trend_strength: float = 0.7


@dataclass
class MockNews:
    sentiment_score: float = 0.5
    sentiment_confidence: float = 0.7
    decision: str = "ALLOW"


def _make_vol_forecast(
    forecast_atr: float = 3.5,
    naive_atr: float = 3.0,
    regime: str = "normal",
) -> VolatilityForecast:
    return VolatilityForecast(
        forecast_atr=forecast_atr,
        naive_atr=naive_atr,
        confidence_lower=forecast_atr * 0.6,
        confidence_upper=forecast_atr * 1.4,
        regime_state=regime,
        regime_multiplier=1.0,
        diurnal_multiplier=1.1,
        calendar_multiplier=1.0,
        blend_weight=0.75,
        har_base=3.2,
    )


# =============================================================================
# CONFLUENCE DETECTOR + VOL_FORECAST TESTS
# =============================================================================

class TestConfluenceWithVolForecast:
    def test_analyze_without_vol_forecast(self):
        """Backward compat: works without vol_forecast."""
        detector = ConfluenceDetector()
        signal = detector.analyze(
            smc_features=_make_smc_features(),
            regime=MockRegime(),
            news=MockNews(),
            price=2000.0,
            atr=3.0,
        )
        assert signal is not None
        assert signal.vol_forecast_atr is None
        assert signal.vol_regime is None

    def test_analyze_with_vol_forecast_uses_forecast_atr(self):
        """When vol_forecast provided, SL/TP should use forecast_atr."""
        detector = ConfluenceDetector()
        vol = _make_vol_forecast(forecast_atr=5.0, naive_atr=3.0)

        signal = detector.analyze(
            smc_features=_make_smc_features(),
            regime=MockRegime(),
            news=MockNews(),
            price=2000.0,
            atr=3.0,  # naive
            vol_forecast=vol,
        )

        assert signal is not None
        # SL should use forecast_atr (5.0) not naive (3.0)
        expected_sl = 2000.0 - SL_ATR_MULT * 5.0  # 2000 - 10 = 1990
        assert abs(signal.stop_loss - expected_sl) < 0.01

    def test_high_regime_widens_sl(self):
        """In high-vol regime, SL should be 1.5x wider."""
        detector = ConfluenceDetector()
        vol = _make_vol_forecast(forecast_atr=4.0, regime="high")

        signal = detector.analyze(
            smc_features=_make_smc_features(),
            regime=MockRegime(),
            news=MockNews(),
            price=2000.0,
            atr=3.0,
            vol_forecast=vol,
        )

        assert signal is not None
        # sizing_atr = 4.0 * 1.5 = 6.0 (high regime widening)
        expected_sl = 2000.0 - SL_ATR_MULT * 6.0  # 2000 - 12 = 1988
        assert abs(signal.stop_loss - expected_sl) < 0.01

    def test_normal_regime_no_widening(self):
        """In normal regime, no SL widening."""
        detector = ConfluenceDetector()
        vol = _make_vol_forecast(forecast_atr=4.0, regime="normal")

        signal = detector.analyze(
            smc_features=_make_smc_features(),
            regime=MockRegime(),
            news=MockNews(),
            price=2000.0,
            atr=3.0,
            vol_forecast=vol,
        )

        assert signal is not None
        expected_sl = 2000.0 - SL_ATR_MULT * 4.0  # 2000 - 8 = 1992
        assert abs(signal.stop_loss - expected_sl) < 0.01

    def test_vol_fields_in_signal(self):
        """ConfluenceSignal should include vol forecast metadata."""
        detector = ConfluenceDetector()
        vol = _make_vol_forecast(forecast_atr=3.5, regime="low")

        signal = detector.analyze(
            smc_features=_make_smc_features(),
            regime=MockRegime(),
            news=MockNews(),
            price=2000.0,
            atr=3.0,
            vol_forecast=vol,
        )

        assert signal is not None
        assert signal.vol_forecast_atr == 3.5
        assert signal.vol_regime == "low"
        assert signal.vol_confidence_lower is not None
        assert signal.vol_confidence_upper is not None

    def test_vol_fields_in_to_dict(self):
        """Vol fields should appear in to_dict() output."""
        detector = ConfluenceDetector()
        vol = _make_vol_forecast(forecast_atr=3.5, regime="normal")

        signal = detector.analyze(
            smc_features=_make_smc_features(),
            regime=MockRegime(),
            news=MockNews(),
            price=2000.0,
            atr=3.0,
            vol_forecast=vol,
        )

        assert signal is not None
        d = signal.to_dict()
        assert "vol_forecast_atr" in d
        assert "vol_regime" in d
        assert d["vol_forecast_atr"] == 3.5
        assert d["vol_regime"] == "normal"

    def test_short_signal_with_vol_forecast(self):
        """Short signals should also use vol_forecast for SL/TP."""
        detector = ConfluenceDetector()
        vol = _make_vol_forecast(forecast_atr=5.0)

        signal = detector.analyze(
            smc_features=_make_smc_features(bos=-1.0, fvg=-1.0),
            regime=MockRegime(regime="strong_downtrend", trend_direction="DOWN"),
            news=MockNews(sentiment_score=-0.5),
            price=2000.0,
            atr=3.0,
            vol_forecast=vol,
        )

        assert signal is not None
        assert signal.signal_type == SignalType.SHORT
        # For SHORT: SL = price + sl_distance
        expected_sl = 2000.0 + SL_ATR_MULT * 5.0  # 2000 + 10 = 2010
        assert abs(signal.stop_loss - expected_sl) < 0.01


# =============================================================================
# SENTINEL SCANNER VOL_FORECASTER INJECTION
# =============================================================================

class TestScannerVolForecaster:
    def test_scanner_accepts_vol_forecaster(self):
        """SentinelScanner should accept vol_forecaster parameter."""
        from src.intelligence.sentinel_scanner import SentinelScanner

        mock_forecaster = MagicMock()
        scanner = SentinelScanner(
            data_provider=MagicMock(),
            smc_factory=MagicMock(),
            regime_agent=MagicMock(),
            news_agent=MagicMock(),
            confluence=MagicMock(),
            llm_engine=MagicMock(),
            cache=MagicMock(),
            signal_store=MagicMock(),
            vol_forecaster=mock_forecaster,
        )
        assert scanner._vol_forecaster is mock_forecaster

    def test_scanner_works_without_vol_forecaster(self):
        """SentinelScanner should work without vol_forecaster (backward compat)."""
        from src.intelligence.sentinel_scanner import SentinelScanner

        scanner = SentinelScanner(
            data_provider=MagicMock(),
            smc_factory=MagicMock(),
            regime_agent=MagicMock(),
            news_agent=MagicMock(),
            confluence=MagicMock(),
            llm_engine=MagicMock(),
            cache=MagicMock(),
            signal_store=MagicMock(),
        )
        assert scanner._vol_forecaster is None


# =============================================================================
# SIGNAL STORE SCHEMA V3
# =============================================================================

class TestSignalStoreV3:
    def test_signal_record_has_vol_fields(self):
        from src.api.signal_store import SignalRecord

        record = SignalRecord(
            signal_id="test-123",
            action="OPEN_LONG",
            symbol="XAUUSD",
            entry_price=2000.0,
            stop_loss=1994.0,
            take_profit=2012.0,
            rr_ratio=2.0,
            created_at="2024-01-01T00:00:00",
            vol_forecast_atr=3.5,
            vol_regime="normal",
            vol_confidence='{"lower": 2.1, "upper": 4.9}',
        )
        assert record.vol_forecast_atr == 3.5
        assert record.vol_regime == "normal"

    def test_schema_migration_v3(self):
        """SignalStore should migrate to v3 with vol columns."""
        from src.api.signal_store import SignalStore

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            store = SignalStore(db_path=db_path)
            assert store.SCHEMA_VERSION == 3

            # Check vol columns exist
            conn = sqlite3.connect(db_path)
            cur = conn.execute("PRAGMA table_info(signals)")
            columns = {row[1] for row in cur.fetchall()}
            conn.close()

            assert "vol_forecast_atr" in columns
            assert "vol_regime" in columns
            assert "vol_confidence" in columns
        finally:
            import os
            os.unlink(db_path)

    def test_publish_with_vol_fields(self):
        """publish() should persist vol fields correctly."""
        from src.api.signal_store import SignalRecord, SignalStore

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            store = SignalStore(db_path=db_path)
            record = SignalRecord(
                signal_id="vol-test-001",
                action="OPEN_LONG",
                symbol="XAUUSD",
                entry_price=2000.0,
                stop_loss=1994.0,
                take_profit=2012.0,
                rr_ratio=2.0,
                created_at="2024-01-01T00:00:00",
                vol_forecast_atr=3.5,
                vol_regime="high",
                vol_confidence='{"lower": 2.1, "upper": 4.9}',
            )
            store.publish(record)

            # Read back from DB
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cur = conn.execute(
                "SELECT vol_forecast_atr, vol_regime, vol_confidence "
                "FROM signals WHERE signal_id = ?",
                ("vol-test-001",),
            )
            row = cur.fetchone()
            conn.close()

            assert row["vol_forecast_atr"] == 3.5
            assert row["vol_regime"] == "high"
            conf = json.loads(row["vol_confidence"])
            assert conf["lower"] == 2.1
            assert conf["upper"] == 4.9
        finally:
            import os
            os.unlink(db_path)


# =============================================================================
# APP STATE
# =============================================================================

class TestAppState:
    def test_appstate_has_vol_forecaster(self):
        from src.api.dependencies import AppState

        mock_store = MagicMock()
        state = AppState(
            signal_store=mock_store,
            vol_forecaster=MagicMock(),
        )
        assert state.vol_forecaster is not None

    def test_appstate_vol_forecaster_optional(self):
        from src.api.dependencies import AppState

        mock_store = MagicMock()
        state = AppState(signal_store=mock_store)
        assert state.vol_forecaster is None


# =============================================================================
# NARRATIVE RESPONSE MODEL
# =============================================================================

class TestNarrativeResponseModel:
    def test_narrative_response_has_vol_fields(self):
        from src.api.models import NarrativeResponse

        resp = NarrativeResponse(
            signal_id="test-001",
            symbol="XAUUSD",
            action="OPEN_LONG",
            entry_price=2000.0,
            stop_loss=1994.0,
            take_profit=2012.0,
            rr_ratio=2.0,
            vol_forecast_atr=3.5,
            vol_regime="normal",
            vol_confidence_lower=2.1,
            vol_confidence_upper=4.9,
        )
        assert resp.vol_forecast_atr == 3.5
        assert resp.vol_regime == "normal"
        assert resp.vol_confidence_lower == 2.1
        assert resp.vol_confidence_upper == 4.9

    def test_narrative_response_vol_fields_optional(self):
        from src.api.models import NarrativeResponse

        resp = NarrativeResponse(
            signal_id="test-002",
            symbol="XAUUSD",
            action="OPEN_LONG",
            entry_price=2000.0,
            stop_loss=1994.0,
            take_profit=2012.0,
            rr_ratio=2.0,
        )
        assert resp.vol_forecast_atr is None
        assert resp.vol_regime is None


# =============================================================================
# END-TO-END PIPELINE (mock-based)
# =============================================================================

class TestEndToEndPipeline:
    def test_full_pipeline_with_vol_forecast(self):
        """
        Simulate the full pipeline:
        data → SMC → vol_forecast → ConfluenceDetector → signal with vol fields
        """
        # 1. Create vol forecast
        vol = _make_vol_forecast(forecast_atr=4.0, naive_atr=3.0, regime="normal")

        # 2. Score with detector
        detector = ConfluenceDetector()
        signal = detector.analyze(
            smc_features=_make_smc_features(),
            regime=MockRegime(),
            news=MockNews(),
            price=2000.0,
            atr=3.0,
            vol_forecast=vol,
        )

        assert signal is not None
        assert signal.confluence_score >= 40
        assert signal.vol_forecast_atr == 4.0
        assert signal.vol_regime == "normal"

        # SL/TP should use forecast (4.0), not naive (3.0)
        assert abs(signal.stop_loss - (2000.0 - 8.0)) < 0.01  # 2 * 4.0
        assert abs(signal.take_profit - (2000.0 + 16.0)) < 0.01  # 4 * 4.0

    def test_fallback_uses_naive_atr(self):
        """When vol_forecast is None, should use raw atr."""
        detector = ConfluenceDetector()
        signal = detector.analyze(
            smc_features=_make_smc_features(),
            regime=MockRegime(),
            news=MockNews(),
            price=2000.0,
            atr=3.0,
            vol_forecast=None,
        )

        assert signal is not None
        assert abs(signal.stop_loss - (2000.0 - 6.0)) < 0.01  # 2 * 3.0
        assert signal.vol_forecast_atr is None
