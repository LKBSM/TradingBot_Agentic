# =============================================================================
# TESTS - Sprint 2: Performance & Latency
# =============================================================================
# Test suite for all performance components.
#
# Run with: pytest tests/test_sprint2_performance.py -v
# =============================================================================

import os
import time
import tempfile
import threading
import numpy as np
from pathlib import Path
from datetime import datetime
from concurrent.futures import Future
import pytest

# Import Sprint 2 components
from src.utils.ring_buffer import RingBuffer, ThreadSafeRingBuffer, TypedRingBuffer, PriceBuffer
from src.utils.latency_tracker import LatencyTracker, LatencyStats, track_latency, get_tracker
from src.utils.async_helpers import AsyncQueue, AsyncWorkerPool
from src.live_trading.async_order_manager import (
    AsyncOrderManager, OrderRequest, OrderResult, OrderPriority, OrderType, OrderStatus
)
from src.performance.async_audit_logger import AsyncAuditLogger, AuditLogConfig
from src.performance.vectorized_risk import VectorizedRiskCalculator, RiskMetrics


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def temp_dir():
    """Create temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_returns():
    """Generate sample returns for testing."""
    np.random.seed(42)
    return np.random.normal(0.001, 0.02, 1000)


@pytest.fixture
def sample_prices():
    """Generate sample prices for testing."""
    np.random.seed(42)
    returns = np.random.normal(0.0001, 0.01, 1000)
    prices = 100 * np.cumprod(1 + returns)
    return prices


# =============================================================================
# RING BUFFER TESTS
# =============================================================================

class TestRingBuffer:
    """Tests for RingBuffer."""

    def test_initialization(self):
        """Test buffer initialization."""
        buffer = RingBuffer(max_size=100)
        assert len(buffer) == 0
        assert buffer.is_empty
        assert not buffer.is_full

    def test_append_and_retrieve(self):
        """Test basic append and retrieval."""
        buffer = RingBuffer(max_size=10)

        for i in range(5):
            buffer.append(float(i))

        assert len(buffer) == 5
        assert buffer[-1] == 4.0  # Last element
        assert buffer[0] == 0.0   # First element

    def test_circular_overwrite(self):
        """Test circular buffer behavior."""
        buffer = RingBuffer(max_size=5)

        for i in range(10):
            buffer.append(float(i))

        assert len(buffer) == 5
        assert buffer.is_full

        # Should contain last 5 values: 5, 6, 7, 8, 9
        arr = buffer.to_array()
        np.testing.assert_array_equal(arr, [5, 6, 7, 8, 9])

    def test_get_last(self):
        """Test get_last method."""
        buffer = RingBuffer(max_size=100)

        for i in range(50):
            buffer.append(float(i))

        last_10 = buffer.get_last(10)
        expected = np.arange(40, 50, dtype=np.float64)
        np.testing.assert_array_equal(last_10, expected)

    def test_extend(self):
        """Test bulk append."""
        buffer = RingBuffer(max_size=100)
        values = np.arange(50, dtype=np.float64)
        buffer.extend(values)

        assert len(buffer) == 50
        np.testing.assert_array_equal(buffer.to_array(), values)

    def test_statistics(self):
        """Test statistical methods."""
        buffer = RingBuffer(max_size=100)
        buffer.extend([1.0, 2.0, 3.0, 4.0, 5.0])

        assert buffer.mean() == 3.0
        assert buffer.min() == 1.0
        assert buffer.max() == 5.0
        assert buffer.sum() == 15.0

    def test_returns_calculation(self):
        """Test returns calculation."""
        buffer = RingBuffer(max_size=100)
        buffer.extend([100.0, 102.0, 101.0, 103.0])

        returns = buffer.returns()
        assert len(returns) == 3
        np.testing.assert_almost_equal(returns[0], 0.02, decimal=5)

    def test_thread_safe_buffer(self):
        """Test thread-safe buffer."""
        buffer = ThreadSafeRingBuffer(max_size=1000)
        threads = []

        def append_values(start, count):
            for i in range(count):
                buffer.append(float(start + i))

        # Concurrent appends
        for t in range(4):
            thread = threading.Thread(target=append_values, args=(t * 100, 100))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert len(buffer) == 400


class TestTypedRingBuffer:
    """Tests for TypedRingBuffer (OHLCV)."""

    def test_initialization(self):
        """Test typed buffer initialization."""
        buffer = TypedRingBuffer(
            max_size=100,
            columns=['open', 'high', 'low', 'close', 'volume']
        )
        assert len(buffer) == 0

    def test_append_and_retrieve(self):
        """Test append and column retrieval."""
        buffer = TypedRingBuffer(
            max_size=100,
            columns=['open', 'high', 'low', 'close']
        )

        buffer.append({
            'open': 100.0,
            'high': 101.0,
            'low': 99.0,
            'close': 100.5
        })

        closes = buffer.get_column('close')
        assert len(closes) == 1
        assert closes[0] == 100.5


class TestPriceBuffer:
    """Tests for PriceBuffer."""

    def test_sma(self, sample_prices):
        """Test simple moving average."""
        buffer = PriceBuffer(max_size=10000)
        buffer.extend(sample_prices)

        sma_20 = buffer.sma(20)
        expected = np.mean(sample_prices[-20:])
        np.testing.assert_almost_equal(sma_20, expected, decimal=5)

    def test_volatility(self, sample_prices):
        """Test volatility calculation."""
        buffer = PriceBuffer(max_size=10000)
        buffer.extend(sample_prices)

        vol = buffer.volatility(window=20)
        assert vol > 0

    def test_bollinger_bands(self, sample_prices):
        """Test Bollinger Bands."""
        buffer = PriceBuffer(max_size=10000)
        buffer.extend(sample_prices)

        upper, middle, lower = buffer.bollinger_bands(period=20)
        assert upper > middle > lower


# =============================================================================
# LATENCY TRACKER TESTS
# =============================================================================

class TestLatencyTracker:
    """Tests for LatencyTracker."""

    def test_initialization(self):
        """Test tracker initialization."""
        tracker = LatencyTracker()
        assert tracker is not None

    def test_track_context_manager(self):
        """Test context manager tracking."""
        tracker = LatencyTracker()

        with tracker.track("test_operation"):
            time.sleep(0.01)  # 10ms

        stats = tracker.get_stats("test_operation")
        assert stats.count == 1
        assert stats.mean_ms >= 10  # At least 10ms

    def test_multiple_operations(self):
        """Test tracking multiple operations."""
        tracker = LatencyTracker()

        for _ in range(10):
            with tracker.track("op_a"):
                time.sleep(0.001)

        for _ in range(5):
            with tracker.track("op_b"):
                time.sleep(0.002)

        stats_a = tracker.get_stats("op_a")
        stats_b = tracker.get_stats("op_b")

        assert stats_a.count == 10
        assert stats_b.count == 5

    def test_percentiles(self):
        """Test percentile calculations."""
        tracker = LatencyTracker()

        # Add varying latencies
        for i in range(100):
            tracker.record("test_op", float(i))

        stats = tracker.get_stats("test_op")
        assert stats.p50_ms == pytest.approx(49.5, abs=1)
        assert stats.p99_ms == pytest.approx(99, abs=1)

    def test_threshold_check(self):
        """Test threshold checking."""
        tracker = LatencyTracker()

        for _ in range(10):
            tracker.record("fast_op", 5.0)
            tracker.record("slow_op", 150.0)

        violations = tracker.check_thresholds({
            "fast_op": 10.0,
            "slow_op": 100.0,
        })

        # slow_op should violate
        assert len(violations) == 1
        assert violations[0]['operation'] == 'slow_op'

    def test_decorator(self):
        """Test track_latency decorator."""
        @track_latency("decorated_func")
        def slow_function():
            time.sleep(0.01)
            return 42

        result = slow_function()
        assert result == 42

        tracker = get_tracker()
        stats = tracker.get_stats("decorated_func")
        assert stats.count >= 1

    def test_prometheus_export(self):
        """Test Prometheus format export."""
        tracker = LatencyTracker()

        for _ in range(5):
            with tracker.track("export_test"):
                time.sleep(0.001)

        prometheus_output = tracker.to_prometheus()
        assert "trading_latency_count" in prometheus_output
        assert "export_test" in prometheus_output


# =============================================================================
# ASYNC HELPERS TESTS
# =============================================================================

class TestAsyncQueue:
    """Tests for AsyncQueue."""

    def test_basic_processing(self):
        """Test basic queue processing."""
        results = []

        def processor(item):
            results.append(item * 2)

        q = AsyncQueue(processor=processor, num_workers=1)
        q.start()

        for i in range(10):
            q.put(i)

        time.sleep(0.5)  # Wait for processing
        q.stop()

        assert len(results) == 10
        assert 0 in results
        assert 18 in results  # 9 * 2


class TestAsyncWorkerPool:
    """Tests for AsyncWorkerPool."""

    def test_submit_and_result(self):
        """Test submitting work and getting results."""
        pool = AsyncWorkerPool(num_workers=2)
        pool.start()

        def double(x):
            return x * 2

        future = pool.submit(double, 21)
        result = future.result(timeout=5.0)

        assert result == 42

        pool.stop()

    def test_parallel_execution(self):
        """Test parallel execution."""
        pool = AsyncWorkerPool(num_workers=4)
        pool.start()

        def slow_task(x):
            time.sleep(0.1)
            return x * 2

        start = time.time()
        futures = [pool.submit(slow_task, i) for i in range(4)]
        results = [f.result(timeout=5.0) for f in futures]
        elapsed = time.time() - start

        # Should complete in ~100ms (parallel), not 400ms (sequential)
        assert elapsed < 0.3
        assert results == [0, 2, 4, 6]

        pool.stop()


# =============================================================================
# ASYNC ORDER MANAGER TESTS
# =============================================================================

class TestAsyncOrderManager:
    """Tests for AsyncOrderManager."""

    def test_initialization(self):
        """Test manager initialization."""
        manager = AsyncOrderManager(mt5_connector=None, pool_size=2)
        assert manager is not None
        assert not manager.is_running

    def test_start_stop(self):
        """Test start and stop lifecycle."""
        manager = AsyncOrderManager(mt5_connector=None)
        manager.start()
        assert manager.is_running

        manager.stop()
        assert not manager.is_running

    def test_submit_order_mock(self):
        """Test order submission with mock execution."""
        manager = AsyncOrderManager(mt5_connector=None)
        manager.start()

        order = OrderRequest(
            symbol="EURUSD",
            direction="BUY",
            volume=0.1,
            priority=OrderPriority.NORMAL
        )

        future = manager.submit_order(order)
        result = future.result(timeout=5.0)

        assert result.status == OrderStatus.FILLED
        assert result.symbol == "EURUSD"

        manager.stop()

    def test_order_priority(self):
        """Test order priority handling."""
        manager = AsyncOrderManager(mt5_connector=None, pool_size=1)
        manager.start()

        results = []

        def on_result(r):
            results.append(r.order_id)

        # Submit low priority first
        order_low = OrderRequest(
            symbol="EURUSD",
            direction="BUY",
            volume=0.1,
            priority=OrderPriority.LOW,
            callback=on_result
        )

        # Submit high priority second
        order_high = OrderRequest(
            symbol="GBPUSD",
            direction="BUY",
            volume=0.1,
            priority=OrderPriority.HIGH,
            callback=on_result
        )

        manager.submit_order(order_low)
        manager.submit_order(order_high)

        time.sleep(0.5)
        manager.stop()

        # High priority should have been processed
        assert len(results) >= 1

    def test_order_stats(self):
        """Test order statistics."""
        manager = AsyncOrderManager(mt5_connector=None)
        manager.start()

        for i in range(5):
            order = OrderRequest(
                symbol=f"TEST{i}",
                direction="BUY",
                volume=0.1
            )
            future = manager.submit_order(order)
            future.result(timeout=5.0)

        stats = manager.get_stats()
        assert stats['orders_submitted'] == 5
        assert stats['orders_filled'] == 5

        manager.stop()


# =============================================================================
# ASYNC AUDIT LOGGER TESTS
# =============================================================================

class TestAsyncAuditLogger:
    """Tests for AsyncAuditLogger."""

    def test_initialization(self, temp_dir):
        """Test logger initialization."""
        config = AuditLogConfig(
            output_dir=temp_dir,
            enable_hmac=False
        )
        logger = AsyncAuditLogger(config)
        assert logger is not None

    def test_basic_logging(self, temp_dir):
        """Test basic non-blocking logging."""
        config = AuditLogConfig(
            output_dir=temp_dir,
            enable_hmac=False,
            flush_interval_seconds=0.1
        )
        logger = AsyncAuditLogger(config)
        logger.start()

        # This should be non-blocking
        start = time.time()
        for i in range(100):
            logger.log({"event": "test", "index": i})
        elapsed = time.time() - start

        # Should be very fast (< 10ms for 100 logs)
        assert elapsed < 0.1

        time.sleep(0.5)  # Wait for flush
        logger.stop()

        # Check file was created
        log_files = list(Path(temp_dir).glob("audit_*.jsonl"))
        assert len(log_files) >= 1

    def test_trade_logging(self, temp_dir):
        """Test trade logging convenience method."""
        config = AuditLogConfig(
            output_dir=temp_dir,
            enable_hmac=False
        )
        logger = AsyncAuditLogger(config)
        logger.start()

        result = logger.log_trade(
            action="executed",
            symbol="EURUSD",
            direction="BUY",
            volume=0.1,
            price=1.0850
        )

        assert result is True

        logger.stop()

    def test_hmac_signing(self, temp_dir):
        """Test HMAC signing of entries."""
        config = AuditLogConfig(
            output_dir=temp_dir,
            enable_hmac=True,
            hmac_key=b"test_secret_key_32_bytes_long!!"
        )
        logger = AsyncAuditLogger(config)
        logger.start()

        logger.log({"important": "data"})

        time.sleep(0.5)
        logger.stop()

        # Verify signature was added (check file content)
        log_files = list(Path(temp_dir).glob("audit_*.jsonl"))
        if log_files:
            import json
            content = log_files[0].read_text()
            if content.strip():
                entry = json.loads(content.strip().split('\n')[0])
                assert '_signature' in entry


# =============================================================================
# VECTORIZED RISK CALCULATOR TESTS
# =============================================================================

class TestVectorizedRiskCalculator:
    """Tests for VectorizedRiskCalculator."""

    def test_initialization(self):
        """Test calculator initialization."""
        calc = VectorizedRiskCalculator()
        assert calc is not None

    def test_var_historical(self, sample_returns):
        """Test historical VaR."""
        calc = VectorizedRiskCalculator()

        var_95 = calc.var_historical(sample_returns, confidence=0.95)
        var_99 = calc.var_historical(sample_returns, confidence=0.99)

        assert var_95 > 0
        assert var_99 > var_95  # 99% VaR should be higher

    def test_var_parametric(self, sample_returns):
        """Test parametric VaR."""
        calc = VectorizedRiskCalculator()
        var = calc.var_parametric(sample_returns, confidence=0.95)
        assert var > 0

    def test_var_monte_carlo(self, sample_returns):
        """Test Monte Carlo VaR."""
        calc = VectorizedRiskCalculator()
        var = calc.var_monte_carlo(sample_returns, confidence=0.95, simulations=1000)
        assert var > 0

    def test_cvar(self, sample_returns):
        """Test Conditional VaR."""
        calc = VectorizedRiskCalculator()

        var = calc.var_historical(sample_returns, confidence=0.95)
        cvar = calc.cvar(sample_returns, confidence=0.95)

        # CVaR should be >= VaR
        assert cvar >= var

    def test_volatility(self, sample_returns):
        """Test volatility calculation."""
        calc = VectorizedRiskCalculator()

        vol = calc.volatility(sample_returns, annualize=False)
        vol_annual = calc.volatility(sample_returns, annualize=True)

        assert vol > 0
        assert vol_annual > vol  # Annualized should be higher

    def test_sharpe_ratio(self, sample_returns):
        """Test Sharpe ratio."""
        calc = VectorizedRiskCalculator()
        sharpe = calc.sharpe_ratio(sample_returns)
        assert isinstance(sharpe, float)

    def test_sortino_ratio(self, sample_returns):
        """Test Sortino ratio."""
        calc = VectorizedRiskCalculator()
        sortino = calc.sortino_ratio(sample_returns)
        assert isinstance(sortino, float)

    def test_max_drawdown(self, sample_returns):
        """Test maximum drawdown."""
        calc = VectorizedRiskCalculator()
        mdd = calc.max_drawdown(sample_returns)

        assert mdd >= 0
        assert mdd <= 1  # Should be a percentage

    def test_rolling_volatility(self, sample_returns):
        """Test rolling volatility."""
        calc = VectorizedRiskCalculator()
        rolling_vol = calc.rolling_volatility(sample_returns, window=20)

        expected_length = len(sample_returns) - 20 + 1
        assert len(rolling_vol) == expected_length

    def test_correlation_matrix(self):
        """Test correlation matrix."""
        calc = VectorizedRiskCalculator()

        # Generate correlated returns
        np.random.seed(42)
        n = 500
        returns1 = np.random.normal(0, 0.01, n)
        returns2 = returns1 * 0.5 + np.random.normal(0, 0.01, n) * 0.5

        returns_matrix = np.column_stack([returns1, returns2])
        corr = calc.correlation_matrix(returns_matrix)

        assert corr.shape == (2, 2)
        assert corr[0, 0] == pytest.approx(1.0)
        assert corr[0, 1] == corr[1, 0]  # Symmetric

    def test_portfolio_var(self):
        """Test portfolio VaR."""
        calc = VectorizedRiskCalculator()

        np.random.seed(42)
        returns_matrix = np.random.normal(0, 0.01, (500, 3))
        weights = np.array([0.4, 0.3, 0.3])

        port_var = calc.portfolio_var(returns_matrix, weights, confidence=0.95)
        assert port_var > 0

    def test_beta_alpha(self, sample_returns):
        """Test beta and alpha calculation."""
        calc = VectorizedRiskCalculator()

        # Create benchmark with some correlation
        np.random.seed(42)
        benchmark = sample_returns * 0.8 + np.random.normal(0, 0.01, len(sample_returns)) * 0.2

        beta = calc.beta(sample_returns, benchmark)
        alpha = calc.alpha(sample_returns, benchmark)

        assert isinstance(beta, float)
        assert isinstance(alpha, float)

    def test_calculate_all_metrics(self, sample_returns):
        """Test comprehensive metrics calculation."""
        calc = VectorizedRiskCalculator()
        metrics = calc.calculate_all_metrics(sample_returns)

        assert isinstance(metrics, RiskMetrics)
        assert metrics.var_95 > 0
        assert metrics.volatility > 0
        assert metrics.max_drawdown >= 0

        # Test to_dict
        metrics_dict = metrics.to_dict()
        assert 'var_95' in metrics_dict
        assert 'sharpe_ratio' in metrics_dict

    def test_performance_benchmark(self, sample_returns):
        """Test that vectorized operations are fast."""
        calc = VectorizedRiskCalculator()

        # Large dataset
        large_returns = np.random.normal(0, 0.02, 100000)

        start = time.time()
        calc.var_historical(large_returns, 0.95)
        calc.volatility(large_returns)
        calc.max_drawdown(large_returns)
        elapsed = time.time() - start

        # Should complete in < 100ms for 100k samples
        assert elapsed < 0.1


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for Sprint 2 components."""

    def test_latency_tracking_with_orders(self):
        """Test latency tracking integrated with order manager."""
        tracker = LatencyTracker()
        manager = AsyncOrderManager(mt5_connector=None)
        manager.start()

        for _ in range(10):
            order = OrderRequest(
                symbol="EURUSD",
                direction="BUY",
                volume=0.1
            )

            with tracker.track("order_submission"):
                future = manager.submit_order(order)

            with tracker.track("order_completion"):
                future.result(timeout=5.0)

        stats = tracker.get_stats("order_submission")
        assert stats.count == 10
        assert stats.mean_ms < 10  # Should be fast (< 10ms)

        manager.stop()

    def test_ring_buffer_with_risk_calc(self, sample_prices):
        """Test RingBuffer with VectorizedRiskCalculator."""
        buffer = PriceBuffer(max_size=10000)
        buffer.extend(sample_prices)

        # Get returns from buffer
        returns = buffer.returns()

        # Calculate risk
        calc = VectorizedRiskCalculator()
        metrics = calc.calculate_all_metrics(returns)

        assert metrics.var_95 > 0
        assert metrics.volatility > 0


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
