"""
Sprint 7 Tests: Async GARCH & Incremental Feature Computation.

Tests incremental RSI, MACD, Bollinger Bands, ATR accuracy vs batch computation,
and verifies AsyncGARCHManager non-blocking behavior.
"""

import time
import threading
import numpy as np
import pytest

from src.performance.incremental_features import (
    IncrementalRSI,
    IncrementalMACD,
    IncrementalBollingerBands,
    IncrementalATR,
    IncrementalFeatureEngine,
    IncrementalFeatureConfig,
)

# We import AsyncGARCHManager directly from risk_manager to avoid gymnasium
import importlib.util
import os

_rm_path = os.path.join(
    os.path.dirname(__file__), "..", "src", "environment", "risk_manager.py"
)
_spec = importlib.util.spec_from_file_location("risk_manager", _rm_path)
_rm = importlib.util.module_from_spec(_spec)

# Patch numpy into the module's namespace before exec
import sys
_rm.np = np
_rm.warnings = __import__("warnings")
_rm.threading = threading
_rm.logging = __import__("logging")

# Provide concurrent.futures
import concurrent.futures
_rm.ThreadPoolExecutor = concurrent.futures.ThreadPoolExecutor
_rm.Future = concurrent.futures.Future

_spec.loader.exec_module(_rm)
AsyncGARCHManager = _rm.AsyncGARCHManager
DynamicRiskManager = _rm.DynamicRiskManager


# =============================================================================
# HELPERS
# =============================================================================

def make_price_series(n: int = 1000, seed: int = 42) -> dict:
    """Create synthetic OHLC price data."""
    rng = np.random.RandomState(seed)
    close = 2000.0 + np.cumsum(rng.randn(n) * 5)
    high = close + np.abs(rng.randn(n) * 3)
    low = close - np.abs(rng.randn(n) * 3)
    return {'high': high, 'low': low, 'close': close}


def batch_rsi(closes: np.ndarray, period: int) -> np.ndarray:
    """Compute RSI using batch Wilder smoothing (reference implementation)."""
    changes = np.diff(closes)
    gains = np.maximum(changes, 0.0)
    losses = np.maximum(-changes, 0.0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    rsi_values = [0.0] * period  # placeholder for warmup
    for i in range(period, len(changes)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss < 1e-10:
            rsi_values.append(100.0 if avg_gain > 1e-10 else 50.0)
        else:
            rs = avg_gain / avg_loss
            rsi_values.append(100.0 - (100.0 / (1.0 + rs)))
    return np.array(rsi_values)


def batch_ema(data: np.ndarray, period: int) -> np.ndarray:
    """Compute EMA (reference implementation)."""
    alpha = 2.0 / (period + 1)
    ema = np.zeros_like(data, dtype=np.float64)
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
    return ema


# =============================================================================
# TEST: INCREMENTAL RSI
# =============================================================================

class TestIncrementalRSI:
    def test_seed_and_update_match_batch(self):
        """Incremental RSI should match batch RSI within floating-point tolerance."""
        prices = make_price_series(1000)
        closes = prices['close']
        period = 10

        # Batch reference: compute RSI on entire series
        # batch_rsi returns array of length len(closes)-1 (indexed by changes)
        batch = batch_rsi(closes, period)

        # Incremental: seed with all closes, building identical state
        rsi = IncrementalRSI(period=period)
        rsi.seed(closes[:201])  # seed with 201 closes -> processes changes 0..199

        # batch[199] = RSI after processing change[199], same as seed result
        assert abs(rsi.value - batch[199]) < 1e-6, (
            f"Seed mismatch: incremental={rsi.value:.6f} vs batch={batch[199]:.6f}"
        )

        # Update bar-by-bar and compare
        for i in range(201, len(closes)):
            inc_val = rsi.update(closes[i])
            # batch[i-1] because batch is indexed by changes (diff), offset by 1
            assert abs(inc_val - batch[i - 1]) < 1e-6, (
                f"Mismatch at bar {i}: incremental={inc_val:.6f} vs batch={batch[i-1]:.6f}"
            )

    def test_rsi_range(self):
        prices = make_price_series()
        rsi = IncrementalRSI(period=10)
        rsi.seed(prices['close'][:50])
        for c in prices['close'][50:200]:
            val = rsi.update(c)
            assert 0.0 <= val <= 100.0

    def test_is_warm(self):
        rsi = IncrementalRSI(period=10)
        assert not rsi.is_warm
        rsi.seed(np.arange(50, dtype=np.float64) + 100)
        assert rsi.is_warm


# =============================================================================
# TEST: INCREMENTAL MACD
# =============================================================================

class TestIncrementalMACD:
    def test_seed_and_update_match_batch(self):
        """Incremental MACD should match batch EMA-based MACD."""
        prices = make_price_series(1000)
        closes = prices['close']
        fast, slow, signal = 8, 17, 9

        # Batch reference
        ema_fast = batch_ema(closes, fast)
        ema_slow = batch_ema(closes, slow)
        macd_line = ema_fast - ema_slow
        macd_signal = batch_ema(macd_line, signal)

        # Incremental
        macd = IncrementalMACD(fast=fast, slow=slow, signal=signal)
        macd.seed(closes)

        # After full seed, should match batch
        assert abs(macd.values['macd_line'] - macd_line[-1]) < 1e-6
        assert abs(macd.values['macd_signal'] - macd_signal[-1]) < 1e-6

    def test_incremental_update_accuracy(self):
        """Bar-by-bar updates should track batch computation."""
        prices = make_price_series(500)
        closes = prices['close']
        fast, slow, signal = 8, 17, 9

        # Seed with first 100 bars
        macd = IncrementalMACD(fast=fast, slow=slow, signal=signal)
        macd.seed(closes[:100])

        # Batch reference for full series
        ema_fast = batch_ema(closes, fast)
        ema_slow = batch_ema(closes, slow)
        macd_line_batch = ema_fast - ema_slow
        macd_signal_batch = batch_ema(macd_line_batch, signal)

        # The seed processes all 100 bars, so the incremental state
        # should match batch at bar 99
        assert abs(macd.values['macd_line'] - macd_line_batch[99]) < 1e-6

        # Update remaining bars
        for i in range(100, len(closes)):
            vals = macd.update(closes[i])
            # Compare MACD line (EMA diff should match exactly)
            assert abs(vals['macd_line'] - macd_line_batch[i]) < 1e-6, (
                f"MACD line mismatch at bar {i}"
            )

    def test_is_warm(self):
        macd = IncrementalMACD(fast=8, slow=17, signal=9)
        assert not macd.is_warm
        macd.seed(np.arange(50, dtype=np.float64) + 2000)
        assert macd.is_warm


# =============================================================================
# TEST: INCREMENTAL BOLLINGER BANDS
# =============================================================================

class TestIncrementalBollingerBands:
    def test_seed_and_update_match_batch(self):
        """Bollinger Bands should match rolling mean/std."""
        prices = make_price_series(500)
        closes = prices['close']
        period = 20

        bb = IncrementalBollingerBands(period=period)
        bb.seed(closes[:100])

        # After seed, middle band should be mean of last 20 closes
        expected_mean = np.mean(closes[80:100])
        assert abs(bb.values['bb_middle'] - expected_mean) < 1e-6

        # Update bar-by-bar and compare to rolling calculation
        for i in range(100, 200):
            bb.update(closes[i])
            window = closes[i - period + 1:i + 1]
            exp_mean = np.mean(window)
            exp_std = np.std(window, ddof=0)  # population std
            assert abs(bb.values['bb_middle'] - exp_mean) < 1e-4, (
                f"BB middle mismatch at bar {i}"
            )
            assert abs(bb.values['bb_upper'] - (exp_mean + 2 * exp_std)) < 1e-4, (
                f"BB upper mismatch at bar {i}"
            )

    def test_is_warm(self):
        bb = IncrementalBollingerBands(period=20)
        assert not bb.is_warm
        for i in range(20):
            bb.update(float(i + 100))
        assert bb.is_warm


# =============================================================================
# TEST: INCREMENTAL ATR
# =============================================================================

class TestIncrementalATR:
    def test_seed_and_update_match_batch(self):
        """Incremental ATR should match batch Wilder-smoothed ATR."""
        prices = make_price_series(500)
        highs, lows, closes = prices['high'], prices['low'], prices['close']
        period = 7

        # Batch reference: True Range then Wilder smoothing
        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1])
            )
        )
        batch_atr = np.zeros(len(tr))
        batch_atr[period - 1] = np.mean(tr[:period])
        for i in range(period, len(tr)):
            batch_atr[i] = (batch_atr[i - 1] * (period - 1) + tr[i]) / period

        # Incremental: seed then update
        atr = IncrementalATR(period=period)
        seed_len = 100
        atr.seed(highs[:seed_len], lows[:seed_len], closes[:seed_len])

        # After seed, should match batch at bar seed_len-2 (since TR starts at bar 1)
        assert abs(atr.value - batch_atr[seed_len - 2]) < 1e-6, (
            f"ATR seed mismatch: inc={atr.value:.6f} vs batch={batch_atr[seed_len-2]:.6f}"
        )

        # Update bar-by-bar
        for i in range(seed_len, len(closes)):
            inc_val = atr.update(highs[i], lows[i], closes[i])
            # i-1 because batch_atr is 1-indexed relative to TR
            assert abs(inc_val - batch_atr[i - 1]) < 1e-6, (
                f"ATR mismatch at bar {i}: inc={inc_val:.6f} vs batch={batch_atr[i-1]:.6f}"
            )

    def test_atr_positive(self):
        prices = make_price_series()
        atr = IncrementalATR(period=7)
        atr.seed(prices['high'][:50], prices['low'][:50], prices['close'][:50])
        for i in range(50, 150):
            val = atr.update(prices['high'][i], prices['low'][i], prices['close'][i])
            assert val > 0

    def test_is_warm(self):
        atr = IncrementalATR(period=7)
        assert not atr.is_warm
        prices = make_price_series(50)
        atr.seed(prices['high'], prices['low'], prices['close'])
        assert atr.is_warm


# =============================================================================
# TEST: COMPOSITE ENGINE
# =============================================================================

class TestIncrementalFeatureEngine:
    def test_seed_and_update(self):
        prices = make_price_series(200)
        engine = IncrementalFeatureEngine()
        vals = engine.seed(prices['high'][:100], prices['low'][:100], prices['close'][:100])

        assert 'RSI' in vals
        assert 'ATR' in vals
        assert 'macd_line' in vals
        assert 'bb_upper' in vals
        assert engine.is_warm

        # Update
        for i in range(100, 150):
            vals = engine.update(prices['high'][i], prices['low'][i], prices['close'][i])
            assert len(vals) == 8  # RSI, ATR, macd_line/signal/diff, bb_upper/middle/lower

    def test_to_array_length(self):
        prices = make_price_series(100)
        engine = IncrementalFeatureEngine()
        engine.seed(prices['high'], prices['low'], prices['close'])
        arr = engine.to_array()
        assert arr.shape == (8,)
        assert np.all(np.isfinite(arr))

    def test_feature_names_match_values(self):
        prices = make_price_series(100)
        engine = IncrementalFeatureEngine()
        engine.seed(prices['high'], prices['low'], prices['close'])
        names = engine.feature_names
        vals = engine.current_values
        assert set(names) == set(vals.keys())

    def test_custom_config(self):
        cfg = IncrementalFeatureConfig(rsi_period=7, macd_fast=12, macd_slow=26, macd_signal=9)
        engine = IncrementalFeatureEngine(config=cfg)
        prices = make_price_series(100)
        engine.seed(prices['high'], prices['low'], prices['close'])
        assert engine.rsi._period == 7
        assert engine.macd._fast_period == 12

    def test_per_bar_latency(self):
        """Each update should complete in < 1ms."""
        prices = make_price_series(1000)
        engine = IncrementalFeatureEngine()
        engine.seed(prices['high'][:200], prices['low'][:200], prices['close'][:200])

        timings = []
        for i in range(200, 800):
            t0 = time.perf_counter()
            engine.update(prices['high'][i], prices['low'][i], prices['close'][i])
            timings.append(time.perf_counter() - t0)

        avg_us = np.mean(timings) * 1e6  # microseconds
        p99_us = np.percentile(timings, 99) * 1e6
        assert p99_us < 1000, f"P99 latency {p99_us:.0f}us exceeds 1ms"


# =============================================================================
# TEST: ASYNC GARCH MANAGER
# =============================================================================

class TestAsyncGARCHManager:
    def test_get_volatility_nonblocking(self):
        """get_volatility should return in < 1ms even during refit."""
        mgr = AsyncGARCHManager(refit_interval=10)
        returns = np.random.RandomState(42).randn(200) * 0.01

        timings = []
        for i in range(50, 200):
            t0 = time.perf_counter()
            sigma = mgr.get_volatility(returns[:i])
            elapsed = time.perf_counter() - t0
            timings.append(elapsed)
            assert sigma > 0

        avg_ms = np.mean(timings) * 1000
        assert avg_ms < 1.0, f"Average latency {avg_ms:.3f}ms exceeds 1ms"

    def test_ewma_sigma_positive(self):
        mgr = AsyncGARCHManager()
        returns = np.random.RandomState(42).randn(50) * 0.01
        for i in range(10, 50):
            sigma = mgr.get_volatility(returns[:i])
            assert sigma > 0

    def test_ewma_sigma_property(self):
        mgr = AsyncGARCHManager()
        returns = np.random.RandomState(42).randn(50) * 0.01
        mgr.get_volatility(returns)
        assert mgr.ewma_sigma > 0

    def test_reset_preserves_ewma(self):
        mgr = AsyncGARCHManager()
        returns = np.random.RandomState(42).randn(50) * 0.01
        mgr.get_volatility(returns)
        sigma_before = mgr.ewma_sigma
        mgr.reset()
        assert mgr.ewma_sigma == sigma_before
        assert mgr._steps_since_refit == 0

    def test_shutdown(self):
        mgr = AsyncGARCHManager()
        mgr.shutdown()
        # Should not raise

    def test_short_returns_fallback(self):
        mgr = AsyncGARCHManager()
        sigma = mgr.get_volatility(np.array([0.01]))
        assert sigma == 0.01


# =============================================================================
# TEST: RISK MANAGER ASYNC INTEGRATION
# =============================================================================

class TestRiskManagerAsyncIntegration:
    def _make_rm(self, use_async: bool = True):
        config = {
            'RISK_PERCENTAGE_PER_TRADE': 0.005,
            'STOP_LOSS_PERCENTAGE': 0.01,
            'TAKE_PROFIT_PERCENTAGE': 0.02,
            'MIN_TRADE_QUANTITY': 0.01,
            'GARCH_UPDATE_FREQUENCY': 50,
            'USE_ASYNC_GARCH': use_async,
        }
        return DynamicRiskManager(config)

    def test_async_garch_path_active(self):
        rm = self._make_rm(use_async=True)
        returns = np.random.RandomState(42).randn(200) * 0.01
        sigma = rm.calculate_garch_volatility(returns)
        assert sigma > 0
        assert rm._use_async_garch is True

    def test_sync_garch_path_still_works(self):
        rm = self._make_rm(use_async=False)
        returns = np.random.RandomState(42).randn(200) * 0.01
        sigma = rm.calculate_garch_volatility(returns)
        assert sigma > 0

    def test_force_update_bypasses_async(self):
        rm = self._make_rm(use_async=True)
        returns = np.random.RandomState(42).randn(200) * 0.01
        sigma = rm.calculate_garch_volatility(returns, force_update=True)
        assert sigma > 0

    def test_reset_resets_async_garch(self):
        rm = self._make_rm(use_async=True)
        returns = np.random.RandomState(42).randn(200) * 0.01
        rm.calculate_garch_volatility(returns)
        rm.reset()
        assert rm._async_garch._steps_since_refit == 0


# =============================================================================
# TEST: CONFIG FLAGS
# =============================================================================

class TestConfigFlags:
    def test_use_async_garch_flag_exists(self):
        import config
        assert hasattr(config, 'USE_ASYNC_GARCH')
        assert isinstance(config.USE_ASYNC_GARCH, bool)

    def test_use_incremental_features_flag_exists(self):
        import config
        assert hasattr(config, 'USE_INCREMENTAL_FEATURES')
        assert isinstance(config.USE_INCREMENTAL_FEATURES, bool)


# =============================================================================
# TEST: SOURCE VERIFICATION
# =============================================================================

class TestSourceVerification:
    def test_async_garch_in_risk_manager(self):
        src_path = os.path.join(
            os.path.dirname(__file__), "..", "src", "environment", "risk_manager.py"
        )
        with open(src_path, "r", encoding="utf-8") as f:
            source = f.read()
        assert "AsyncGARCHManager" in source
        assert "_use_async_garch" in source

    def test_incremental_engine_in_environment(self):
        src_path = os.path.join(
            os.path.dirname(__file__), "..", "src", "environment", "environment.py"
        )
        with open(src_path, "r", encoding="utf-8") as f:
            source = f.read()
        assert "_incremental_engine" in source
