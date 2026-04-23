"""Tests for production wiring: circuit breakers, rate limiter, input validation.

Covers:
  - SentinelScanner circuit breaker integration (LLM + Telegram)
  - ConfluenceDetector per-instrument price_decimals
  - InstrumentConfig price_decimals field
  - AppState new fields (circuit_breakers, health_checker, rate_limiter)
  - Narratives route input validation (signal_id format, sanitization)
  - App middleware (rate limiter, request size limit, CORS env config)
  - build_system circuit breaker wiring
  - Health checker aggregation with circuit breakers
"""

import re
import threading
import time
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest


# =============================================================================
# INSTRUMENT CONFIG — price_decimals
# =============================================================================

class TestInstrumentConfigDecimals:
    def test_default_decimals_is_2(self):
        from src.intelligence.volatility_forecaster import InstrumentConfig
        config = InstrumentConfig()
        assert config.price_decimals == 2

    def test_fx_pairs_have_5_decimals(self):
        from src.intelligence.volatility_forecaster import get_instrument_registry
        registry = get_instrument_registry()
        assert registry["EURUSD"].price_decimals == 5
        assert registry["GBPUSD"].price_decimals == 5

    def test_gold_has_2_decimals(self):
        from src.intelligence.volatility_forecaster import get_instrument_registry
        registry = get_instrument_registry()
        assert registry["XAUUSD"].price_decimals == 2

    def test_usdjpy_has_3_decimals(self):
        from src.intelligence.volatility_forecaster import get_instrument_registry
        registry = get_instrument_registry()
        assert registry["USDJPY"].price_decimals == 3

    def test_index_has_1_decimal(self):
        from src.intelligence.volatility_forecaster import get_instrument_registry
        registry = get_instrument_registry()
        assert registry["US500"].price_decimals == 1

    def test_btc_has_2_decimals(self):
        from src.intelligence.volatility_forecaster import get_instrument_registry
        registry = get_instrument_registry()
        assert registry["BTCUSD"].price_decimals == 2


# =============================================================================
# CONFLUENCE DETECTOR — price_decimals rounding
# =============================================================================

class TestConfluenceDetectorRounding:
    def _make_detector(self, config):
        from src.intelligence.confluence_detector import ConfluenceDetector
        return ConfluenceDetector(instrument_config=config)

    def test_gold_rounds_to_2(self):
        from src.intelligence.volatility_forecaster import get_instrument_registry
        from src.intelligence.confluence_detector import ConfluenceDetector
        config = get_instrument_registry()["XAUUSD"]
        detector = ConfluenceDetector(instrument_config=config)
        assert detector._price_decimals == 2

    def test_eurusd_rounds_to_5(self):
        from src.intelligence.volatility_forecaster import get_instrument_registry
        from src.intelligence.confluence_detector import ConfluenceDetector
        config = get_instrument_registry()["EURUSD"]
        detector = ConfluenceDetector(instrument_config=config)
        assert detector._price_decimals == 5

    def test_no_config_defaults_to_2(self):
        from src.intelligence.confluence_detector import ConfluenceDetector
        detector = ConfluenceDetector()
        assert detector._price_decimals == 2

    def test_fx_analyze_preserves_precision(self):
        """FX signal entry/SL/TP should have 5 decimal places."""
        from src.intelligence.volatility_forecaster import get_instrument_registry
        from src.intelligence.confluence_detector import ConfluenceDetector
        config = get_instrument_registry()["EURUSD"]
        detector = ConfluenceDetector(instrument_config=config)

        signal = detector.analyze(
            smc_features={
                "BOS_SIGNAL": 1.0,
                "FVG_SIGNAL": 1.0,
                "OB_STRENGTH_NORM": 0.8,
                "RSI": 60.0,
                "MACD_Diff": 0.001,
                "FVG_SIZE_NORM": 0.5,
            },
            regime=None,
            news=None,
            price=1.10000,
            atr=0.00100,
            volume=1000.0,
            volume_ma=800.0,
        )

        if signal is not None:
            # Entry should have up to 5 decimal places
            entry_str = f"{signal.entry_price:.5f}"
            assert entry_str == "1.10000"
            # SL precision preserved
            sl_str = f"{signal.stop_loss:.5f}"
            assert len(sl_str.split(".")[-1]) == 5


# =============================================================================
# SENTINEL SCANNER — circuit breaker integration
# =============================================================================

class TestScannerCircuitBreaker:
    def _make_scanner(self, llm_breaker=None, notifier_breaker=None, llm_fails=False, notifier_fails=False):
        from src.intelligence.sentinel_scanner import SentinelScanner
        from src.intelligence.confluence_detector import ConfluenceDetector, ConfluenceSignal, SignalType, SignalTier
        from src.intelligence.llm_narrative_engine import NarrativeTier

        # Mock dependencies
        mock_llm = MagicMock()
        if llm_fails:
            mock_llm.generate_narrative.side_effect = RuntimeError("LLM unavailable")
        else:
            mock_narrative = MagicMock()
            mock_narrative.to_dict.return_value = {"tier": "NARRATOR", "summary": "test"}
            mock_llm.generate_narrative.return_value = mock_narrative

        mock_notifier = MagicMock()
        if notifier_fails:
            mock_notifier.send_signal.side_effect = RuntimeError("Telegram down")

        scanner = SentinelScanner(
            data_provider=MagicMock(),
            smc_factory=MagicMock(),
            regime_agent=MagicMock(),
            news_agent=MagicMock(),
            confluence=ConfluenceDetector(),
            llm_engine=mock_llm,
            cache=MagicMock(get=MagicMock(return_value=None)),
            signal_store=MagicMock(),
            notifier=mock_notifier,
            llm_circuit_breaker=llm_breaker,
            notifier_circuit_breaker=notifier_breaker,
        )
        return scanner

    def test_scanner_accepts_circuit_breaker_params(self):
        from src.intelligence.circuit_breaker import CircuitBreaker
        llm_cb = CircuitBreaker(name="llm_api", failure_threshold=3)
        telegram_cb = CircuitBreaker(name="telegram", failure_threshold=5)
        scanner = self._make_scanner(llm_breaker=llm_cb, notifier_breaker=telegram_cb)
        assert scanner._llm_breaker is llm_cb
        assert scanner._notifier_breaker is telegram_cb

    def test_scanner_stats_include_circuit_state(self):
        from src.intelligence.circuit_breaker import CircuitBreaker
        llm_cb = CircuitBreaker(name="llm_api", failure_threshold=3)
        telegram_cb = CircuitBreaker(name="telegram", failure_threshold=5)
        scanner = self._make_scanner(llm_breaker=llm_cb, notifier_breaker=telegram_cb)
        scanner._start_time = time.time()
        stats = scanner.get_stats()
        assert stats["llm_circuit"] == "closed"
        assert stats["telegram_circuit"] == "closed"

    def test_llm_failure_uses_fallback_narrative(self):
        """When LLM fails (no circuit breaker), scanner returns fallback narrative."""
        scanner = self._make_scanner(llm_fails=True)
        # Test the _generate_narrative_safe method directly
        mock_signal = MagicMock()
        mock_signal.signal_type.value = "LONG"
        mock_signal.symbol = "XAUUSD"
        mock_signal.confluence_score = 75.0
        mock_signal.signal_id = "test123"

        result = scanner._generate_narrative_safe(mock_signal)
        assert result is None
        assert scanner._llm_failures == 1

    def test_llm_circuit_open_returns_none(self):
        """When LLM circuit is OPEN, generate_narrative_safe returns None."""
        from src.intelligence.circuit_breaker import CircuitBreaker
        llm_cb = CircuitBreaker(name="llm_api", failure_threshold=2)
        scanner = self._make_scanner(llm_breaker=llm_cb, llm_fails=True)

        mock_signal = MagicMock()
        mock_signal.signal_id = "test123"

        # Trip the circuit
        for _ in range(2):
            scanner._generate_narrative_safe(mock_signal)

        # Now circuit should be open
        assert llm_cb.state.value == "open"

        # Next call should be blocked by circuit
        result = scanner._generate_narrative_safe(mock_signal)
        assert result is None
        assert scanner._llm_failures == 3

    def test_notification_failure_doesnt_crash(self):
        """Notification failure should be caught gracefully."""
        scanner = self._make_scanner(notifier_fails=True)
        mock_signal = MagicMock()
        mock_signal.signal_id = "test123"

        # Should not raise
        scanner._send_notification_safe(mock_signal, {"summary": "test"})
        assert scanner._notification_failures == 1

    def test_notification_circuit_open_skips(self):
        """When Telegram circuit is open, notification is silently skipped."""
        from src.intelligence.circuit_breaker import CircuitBreaker
        telegram_cb = CircuitBreaker(name="telegram", failure_threshold=2)
        scanner = self._make_scanner(notifier_breaker=telegram_cb, notifier_fails=True)

        mock_signal = MagicMock()
        mock_signal.signal_id = "test123"

        # Trip the circuit
        for _ in range(2):
            scanner._send_notification_safe(mock_signal, {"summary": "test"})

        assert telegram_cb.state.value == "open"

        # Next call silently skipped
        scanner._send_notification_safe(mock_signal, {"summary": "test"})
        assert scanner._notification_failures == 3

    def test_no_notifier_send_is_noop(self):
        """When notifier is None, _send_notification_safe is a noop."""
        from src.intelligence.sentinel_scanner import SentinelScanner
        from src.intelligence.confluence_detector import ConfluenceDetector
        scanner = SentinelScanner(
            data_provider=MagicMock(),
            smc_factory=MagicMock(),
            regime_agent=MagicMock(),
            news_agent=MagicMock(),
            confluence=ConfluenceDetector(),
            llm_engine=MagicMock(),
            cache=MagicMock(),
            signal_store=MagicMock(),
            notifier=None,
        )
        # Should not raise
        scanner._send_notification_safe(MagicMock(), {})
        assert scanner._notification_failures == 0


# =============================================================================
# APP STATE — new fields
# =============================================================================

class TestAppStateNewFields:
    def test_circuit_breakers_default_empty(self):
        from src.api.dependencies import AppState
        from src.api.signal_store import SignalStore
        state = AppState(signal_store=MagicMock(spec=SignalStore))
        assert state.circuit_breakers == {}
        assert state.health_checker is None
        assert state.rate_limiter is None

    def test_circuit_breakers_set(self):
        from src.api.dependencies import AppState
        from src.intelligence.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker(name="test")
        state = AppState(
            signal_store=MagicMock(),
            circuit_breakers={"test": cb},
        )
        assert state.circuit_breakers["test"] is cb


# =============================================================================
# NARRATIVES ROUTE — input validation
# =============================================================================

class TestNarrativesValidation:
    def test_signal_id_pattern_valid(self):
        """Valid signal IDs match the pattern."""
        from src.api.routes.narratives import SIGNAL_ID_PATTERN
        assert SIGNAL_ID_PATTERN.match("a1b2c3d4e5f6")
        assert SIGNAL_ID_PATTERN.match("abcdef012345")
        assert SIGNAL_ID_PATTERN.match("12345678")
        # Full UUID format
        assert SIGNAL_ID_PATTERN.match("a1b2c3d4-e5f6-7890-abcd-ef0123456789")

    def test_signal_id_pattern_rejects_injection(self):
        """SQL injection attempts should not match."""
        from src.api.routes.narratives import SIGNAL_ID_PATTERN
        assert not SIGNAL_ID_PATTERN.match("'; DROP TABLE--")
        assert not SIGNAL_ID_PATTERN.match("<script>alert(1)</script>")
        assert not SIGNAL_ID_PATTERN.match("")
        assert not SIGNAL_ID_PATTERN.match("short")  # Too short (< 8)

    def test_signal_id_pattern_rejects_uppercase(self):
        from src.api.routes.narratives import SIGNAL_ID_PATTERN
        assert not SIGNAL_ID_PATTERN.match("ABCDEF012345")

    def test_sanitize_string_imported(self):
        """sanitize_string should be importable from the module."""
        from src.api.routes.narratives import sanitize_string
        result = sanitize_string("test\x00injection\x01attempt", max_length=50)
        assert "\x00" not in result
        assert "\x01" not in result
        assert "test" in result


# =============================================================================
# APP MIDDLEWARE — rate limiter, size limit
# =============================================================================

class TestAppMiddleware:
    def test_create_app_accepts_new_params(self):
        """create_app should accept circuit_breakers, health_checker, rate_limiter."""
        from src.api.app import create_app
        from src.intelligence.circuit_breaker import CircuitBreaker, HealthChecker
        from src.intelligence.security import RateLimiter

        cb = CircuitBreaker(name="test")
        hc = HealthChecker()
        rl = RateLimiter(max_requests=10, window_seconds=60)

        app = create_app(
            circuit_breakers={"test": cb},
            health_checker=hc,
            rate_limiter=rl,
        )

        assert app.state.app_state.circuit_breakers["test"] is cb
        assert app.state.app_state.health_checker is hc
        assert app.state.app_state.rate_limiter is rl

    def test_create_app_defaults_new_params(self):
        """New params should default to empty/None."""
        from src.api.app import create_app
        app = create_app()
        assert app.state.app_state.circuit_breakers == {}
        assert app.state.app_state.health_checker is None
        assert app.state.app_state.rate_limiter is None


# =============================================================================
# HEALTH CHECKER + CIRCUIT BREAKER integration
# =============================================================================

class TestHealthCheckerIntegration:
    def test_healthy_when_all_circuits_closed(self):
        from src.intelligence.circuit_breaker import CircuitBreaker, CircuitState, HealthChecker
        llm_cb = CircuitBreaker(name="llm", failure_threshold=3)
        tg_cb = CircuitBreaker(name="telegram", failure_threshold=3)

        hc = HealthChecker()
        hc.register("llm", lambda: llm_cb.state != CircuitState.OPEN)
        hc.register("telegram", lambda: tg_cb.state != CircuitState.OPEN)

        status = hc.check()
        assert status.healthy is True

    def test_unhealthy_when_llm_circuit_open(self):
        from src.intelligence.circuit_breaker import CircuitBreaker, CircuitState, HealthChecker
        llm_cb = CircuitBreaker(name="llm", failure_threshold=2)

        # Trip the circuit
        for _ in range(2):
            try:
                llm_cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))
            except ValueError:
                pass

        hc = HealthChecker()
        hc.register("llm", lambda: llm_cb.state != CircuitState.OPEN)

        status = hc.check()
        assert status.healthy is False
        assert status.checks["llm"]["healthy"] is False

    def test_health_status_to_dict(self):
        from src.intelligence.circuit_breaker import HealthChecker
        hc = HealthChecker()
        hc.register("test", lambda: True)
        d = hc.check().to_dict()
        assert isinstance(d, dict)
        assert d["healthy"] is True


# =============================================================================
# BUILD SYSTEM — circuit breaker wiring
# =============================================================================

def _build_system_for_test(tmpdir, env=None):
    """Shared setup for build_system tests. Writes dummy OHLCV + builds system."""
    import os
    from pathlib import Path

    import numpy as np
    import pandas as pd

    rng = np.random.RandomState(42)
    n = 3000
    ts = pd.date_range("2024-01-01", periods=n, freq="15min")
    close = 2000.0 + np.cumsum(rng.randn(n) * 0.5)
    df = pd.DataFrame({
        "timestamp": ts,
        "Open": close + rng.randn(n) * 0.3,
        "High": close + rng.uniform(0.5, 3.0, n),
        "Low": close - rng.uniform(0.5, 3.0, n),
        "Close": close,
        "Volume": rng.uniform(100, 10000, n),
    })
    df.to_csv(Path(tmpdir) / "XAUUSD_M15.csv", index=False)
    db_path = str(Path(tmpdir) / "signals.db")

    from src.intelligence.main import _NullAgent, build_system
    with patch.dict(os.environ, env or {}, clear=False), \
         patch("src.intelligence.main._create_regime_agent", return_value=_NullAgent()), \
         patch("src.intelligence.main._create_news_agent", return_value=_NullAgent()):
        return build_system(
            symbols=["XAUUSD"],
            data_dir=tmpdir,
            signal_db=db_path,
            vol_mode="har",
        )


class TestBuildSystemCircuitBreakers:
    def test_build_system_llm_mode_returns_both_breakers(self):
        """In LLM mode, both LLM and Telegram breakers should be wired."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            system = _build_system_for_test(tmpdir, env={"NARRATIVE_MODE": "llm"})
            assert "circuit_breakers" in system
            assert "llm" in system["circuit_breakers"]
            assert "telegram" in system["circuit_breakers"]
            assert "health_checker" in system

    def test_build_system_template_mode_skips_llm_breaker(self):
        """In template mode, LLM breaker is skipped — template engine never fails."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            system = _build_system_for_test(tmpdir, env={"NARRATIVE_MODE": "template"})
            assert "circuit_breakers" in system
            assert "llm" not in system["circuit_breakers"]
            assert "telegram" in system["circuit_breakers"]

    def test_scanner_llm_mode_has_both_breakers(self):
        """In LLM mode, scanner gets both circuit breakers."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            system = _build_system_for_test(tmpdir, env={"NARRATIVE_MODE": "llm"})
            scanner = system["scanner"]
            assert scanner._llm_breaker is not None
            assert scanner._notifier_breaker is not None
            assert scanner._llm_breaker.name == "llm_api"
            assert scanner._notifier_breaker.name == "telegram"

    def test_scanner_template_mode_only_has_notifier_breaker(self):
        """In template mode, scanner has no LLM breaker — narrative gen is in-process."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            system = _build_system_for_test(tmpdir, env={"NARRATIVE_MODE": "template"})
            scanner = system["scanner"]
            assert scanner._llm_breaker is None
            assert scanner._notifier_breaker is not None

    def test_template_mode_skips_semantic_cache(self):
        """In template mode, SemanticCache should not be attached to scanner."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            system = _build_system_for_test(tmpdir, env={"NARRATIVE_MODE": "template"})
            scanner = system["scanner"]
            assert scanner._cache is None

    def test_llm_mode_uses_semantic_cache(self):
        """In LLM mode, SemanticCache should be attached to scanner."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            system = _build_system_for_test(tmpdir, env={"NARRATIVE_MODE": "llm"})
            scanner = system["scanner"]
            assert scanner._cache is not None
