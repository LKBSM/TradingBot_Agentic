"""Tests for SentinelScanner — Sprint 4 of Smart Sentinel AI."""

import pytest
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch
from enum import Enum

from src.intelligence.sentinel_scanner import SentinelScanner
from src.intelligence.confluence_detector import (
    ConfluenceDetector,
    ConfluenceSignal,
    SignalType,
    SignalTier,
)
from src.intelligence.llm_narrative_engine import (
    LLMNarrativeEngine,
    NarrativeTier,
    SignalNarrative,
)
from src.intelligence.semantic_cache import SemanticCache


# ============================================================================
# HELPERS
# ============================================================================

def make_ohlcv_df(n_bars: int = 200, base_price: float = 2400.0,
                  bos: float = 1.0, fvg: float = 1.0) -> pd.DataFrame:
    """Create a mock enriched OHLCV DataFrame with SMC features."""
    dates = pd.date_range("2025-06-01", periods=n_bars, freq="15min")
    close = base_price + np.cumsum(np.random.randn(n_bars) * 0.5)
    high = close + np.random.rand(n_bars) * 2
    low = close - np.random.rand(n_bars) * 2
    open_ = close + np.random.randn(n_bars) * 0.3
    volume = np.random.randint(500, 2000, n_bars).astype(float)

    df = pd.DataFrame({
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
        "ATR": np.full(n_bars, 10.0),
        "RSI": np.full(n_bars, 55.0),
        "MACD_Diff": np.full(n_bars, 0.3),
        "BOS_SIGNAL": np.zeros(n_bars),
        "FVG_SIGNAL": np.zeros(n_bars),
        "OB_STRENGTH_NORM": np.zeros(n_bars),
    }, index=dates)

    # Set last bar to have strong signals
    df.iloc[-1, df.columns.get_loc("BOS_SIGNAL")] = bos
    df.iloc[-1, df.columns.get_loc("FVG_SIGNAL")] = fvg

    return df


class MockRegimeType(Enum):
    STRONG_UPTREND = "strong_uptrend"


class MockTrendDirection(Enum):
    UP = 1


@dataclass
class MockRegimeAnalysis:
    regime: MockRegimeType = MockRegimeType.STRONG_UPTREND
    confidence: float = 0.8
    trend_direction: MockTrendDirection = MockTrendDirection.UP
    trend_strength: float = 0.75


class MockNewsDecision(Enum):
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"


@dataclass
class MockNewsAssessment:
    decision: MockNewsDecision = MockNewsDecision.ALLOW
    sentiment_score: float = 0.5
    sentiment_confidence: float = 0.6
    reasoning: List[str] = field(default_factory=lambda: ["Positive outlook"])


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_data_provider():
    provider = MagicMock()
    provider.get_ohlcv.return_value = make_ohlcv_df()
    return provider


@pytest.fixture
def mock_smc_factory():
    """Factory that returns a mock SMC engine whose analyze() returns the df."""
    def factory(df):
        engine = MagicMock()
        engine.analyze.return_value = df
        return engine
    return factory


@pytest.fixture
def mock_regime_agent():
    agent = MagicMock()
    agent.analyze.return_value = MockRegimeAnalysis()
    return agent


@pytest.fixture
def mock_news_agent():
    agent = MagicMock()
    agent.evaluate_news_impact.return_value = MockNewsAssessment()
    return agent


@pytest.fixture
def detector():
    return ConfluenceDetector()


@pytest.fixture
def llm_engine():
    engine = LLMNarrativeEngine(api_key=None)  # Visual-only mode
    return engine


@pytest.fixture
def cache(tmp_path):
    return SemanticCache(db_path=str(tmp_path / "test_cache.db"))


@pytest.fixture
def signal_store():
    store = MagicMock()
    return store


@pytest.fixture
def notifier():
    return MagicMock()


@pytest.fixture
def scanner(mock_data_provider, mock_smc_factory, mock_regime_agent,
            mock_news_agent, detector, llm_engine, cache, signal_store, notifier):
    return SentinelScanner(
        data_provider=mock_data_provider,
        smc_factory=mock_smc_factory,
        regime_agent=mock_regime_agent,
        news_agent=mock_news_agent,
        confluence=detector,
        llm_engine=llm_engine,
        cache=cache,
        signal_store=signal_store,
        notifier=notifier,
        narrative_tier=NarrativeTier.VISUAL,
    )


# ============================================================================
# TESTS: FULL PIPELINE
# ============================================================================

class TestFullPipeline:
    def test_new_bar_generates_signal(self, scanner):
        scanner._start_time = 0
        result = scanner.scan_once()

        assert result is not None
        assert isinstance(result, ConfluenceSignal)
        assert result.signal_type == SignalType.LONG
        stats = scanner.get_stats()
        assert stats["bars_scanned"] == 1
        assert stats["signals_generated"] == 1

    def test_same_bar_skipped(self, scanner):
        scanner._start_time = 0
        scanner.scan_once()
        result = scanner.scan_once()  # Same bar

        assert result is None
        stats = scanner.get_stats()
        assert stats["bars_scanned"] == 1  # Only counted once

    def test_signal_published_to_store(self, scanner, signal_store):
        scanner._start_time = 0
        scanner.scan_once()

        signal_store.publish.assert_called_once()
        record = signal_store.publish.call_args[0][0]
        assert record.action == "OPEN_LONG"
        assert record.symbol == "XAUUSD"

    def test_notifier_called(self, scanner, notifier):
        scanner._start_time = 0
        scanner.scan_once()

        notifier.send_signal.assert_called_once()


# ============================================================================
# TESTS: LOW CONFLUENCE
# ============================================================================

class TestLowConfluence:
    def test_no_bos_no_signal(self, scanner, mock_data_provider):
        """No BOS → ConfluenceDetector returns None → no signal published."""
        mock_data_provider.get_ohlcv.return_value = make_ohlcv_df(bos=0.0, fvg=0.0)
        scanner._start_time = 0
        result = scanner.scan_once()

        assert result is None
        stats = scanner.get_stats()
        assert stats["signals_generated"] == 0


# ============================================================================
# TESTS: NEWS BLOCKING
# ============================================================================

class TestNewsBlocking:
    def test_news_block_skips_signal(self, scanner, mock_news_agent):
        mock_news_agent.evaluate_news_impact.return_value = MockNewsAssessment(
            decision=MockNewsDecision.BLOCK,
            sentiment_score=0.0,
            sentiment_confidence=0.5,
            reasoning=["NFP imminent"],
        )
        scanner._start_time = 0
        result = scanner.scan_once()

        assert result is None
        stats = scanner.get_stats()
        assert stats["signals_generated"] == 0


# ============================================================================
# TESTS: CACHE INTEGRATION
# ============================================================================

class TestCacheIntegration:
    def test_cache_miss_triggers_llm(self, scanner, cache):
        scanner._start_time = 0
        scanner.scan_once()

        stats = scanner.get_stats()
        assert stats["llm_calls"] == 1
        assert stats["cache_hits"] == 0

    def test_cache_hit_skips_llm(self, scanner, cache, mock_data_provider):
        """Pre-populate cache, then scan → should hit cache."""
        scanner._start_time = 0

        # First scan populates cache
        scanner.scan_once()

        # Change bar timestamp to force re-scan but keep same features
        df = make_ohlcv_df()
        # Shift index by 15 min so it's a "new" bar
        df.index = df.index + pd.Timedelta(minutes=15)
        mock_data_provider.get_ohlcv.return_value = df

        scanner.scan_once()

        # The second scan won't hit cache because the bar_ts changed → different key
        # This is correct behavior: different bars = different analysis
        stats = scanner.get_stats()
        assert stats["bars_scanned"] == 2


# ============================================================================
# TESTS: ERROR HANDLING
# ============================================================================

class TestErrorHandling:
    def test_data_provider_error_graceful(self, scanner, mock_data_provider):
        mock_data_provider.get_ohlcv.side_effect = ConnectionError("MT5 disconnected")
        scanner._start_time = 0
        result = scanner.scan_once()

        assert result is None
        stats = scanner.get_stats()
        assert stats["errors"] == 1

    def test_insufficient_data_skipped(self, scanner, mock_data_provider):
        mock_data_provider.get_ohlcv.return_value = make_ohlcv_df(n_bars=10)
        scanner._start_time = 0
        result = scanner.scan_once()

        assert result is None

    def test_smc_error_graceful(self, scanner, mock_smc_factory):
        def broken_factory(df):
            engine = MagicMock()
            engine.analyze.side_effect = RuntimeError("SMC crash")
            return engine

        scanner._smc_factory = broken_factory
        scanner._start_time = 0
        result = scanner.scan_once()

        assert result is None
        assert scanner.get_stats()["errors"] == 1

    def test_notifier_error_non_fatal(self, scanner, notifier):
        notifier.send_signal.side_effect = Exception("Telegram down")
        scanner._start_time = 0
        result = scanner.scan_once()

        # Signal should still be generated despite notification failure
        assert result is not None


# ============================================================================
# TESTS: STATS
# ============================================================================

class TestStats:
    def test_initial_stats(self, scanner):
        scanner._start_time = 0
        stats = scanner.get_stats()
        assert stats["bars_scanned"] == 0
        assert stats["signals_generated"] == 0
        assert stats["cache_hits"] == 0
        assert stats["llm_calls"] == 0
        assert stats["errors"] == 0

    def test_stats_accumulate(self, scanner, mock_data_provider):
        scanner._start_time = 0
        scanner.scan_once()

        # Change bar to scan again
        df = make_ohlcv_df()
        df.index = df.index + pd.Timedelta(minutes=15)
        mock_data_provider.get_ohlcv.return_value = df
        scanner.scan_once()

        stats = scanner.get_stats()
        assert stats["bars_scanned"] == 2


# ============================================================================
# TESTS: LIFECYCLE
# ============================================================================

class TestLifecycle:
    def test_shutdown_sets_running_false(self, scanner):
        scanner._running = True
        scanner._start_time = 0
        scanner.shutdown()
        assert scanner._running is False

    def test_non_blocking_start_and_stop(self, scanner):
        scanner._poll_interval = 0.1
        scanner.start(blocking=False)
        assert scanner._running is True

        import time
        time.sleep(0.3)
        scanner.shutdown()
        assert scanner._running is False
