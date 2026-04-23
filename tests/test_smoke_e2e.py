"""End-to-end smoke test — validates the full pipeline from CSV data to API response.

Runs without real API keys or external services. Uses mocks for LLM/Telegram.
"""

from __future__ import annotations

import os
import sys
import json
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

# Ensure SENTINEL_TESTING_MODE=1 for smoke tests
os.environ["SENTINEL_TESTING_MODE"] = "1"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _make_ohlcv(n_bars: int = 300, symbol: str = "XAUUSD") -> pd.DataFrame:
    """Generate synthetic OHLCV data resembling Gold M15."""
    np.random.seed(42)
    base_price = 2000.0
    returns = np.random.normal(0.0001, 0.002, n_bars)
    closes = base_price * np.exp(np.cumsum(returns))
    highs = closes * (1 + np.abs(np.random.normal(0, 0.001, n_bars)))
    lows = closes * (1 - np.abs(np.random.normal(0, 0.001, n_bars)))
    opens = (closes + np.roll(closes, 1)) / 2
    opens[0] = base_price
    volumes = np.random.lognormal(10, 0.5, n_bars)

    idx = pd.date_range("2024-01-01", periods=n_bars, freq="15min")
    return pd.DataFrame({
        "Open": opens,
        "High": highs,
        "Low": lows,
        "Close": closes,
        "Volume": volumes,
    }, index=idx)


class TestSmokeEndToEnd(unittest.TestCase):
    """Full pipeline smoke test with synthetic data."""

    def test_build_system_creates_all_components(self):
        """build_system() returns scanner, api_app, and signal_store."""
        # Write synthetic CSV
        with tempfile.TemporaryDirectory() as tmpdir:
            df = _make_ohlcv(300)
            csv_path = os.path.join(tmpdir, "XAUUSD_M15.csv")
            df.to_csv(csv_path)

            signal_db = os.path.join(tmpdir, "signals.db")

            with patch("src.intelligence.main._create_regime_agent") as mock_regime, \
                 patch("src.intelligence.main._create_news_agent") as mock_news:
                from src.intelligence.main import _NullAgent
                mock_regime.return_value = _NullAgent()
                mock_news.return_value = _NullAgent()

                from src.intelligence.main import build_system
                system = build_system(
                    symbols=["XAUUSD"],
                    data_dir=tmpdir,
                    signal_db=signal_db,
                    vol_mode="har",
                    anthropic_key=None,
                    telegram_token=None,
                )

            self.assertIn("scanner", system)
            self.assertIn("api_app", system)
            self.assertIn("signal_store", system)
            self.assertIn("circuit_breakers", system)
            self.assertIn("health_checker", system)

    def test_scanner_scan_once_with_synthetic_data(self):
        """Scanner processes one bar and returns without crashing."""
        from src.intelligence.confluence_detector import ConfluenceDetector
        from src.intelligence.llm_narrative_engine import LLMNarrativeEngine
        from src.intelligence.semantic_cache import SemanticCache
        from src.intelligence.sentinel_scanner import SentinelScanner
        from src.intelligence.volatility_forecaster import (
            InstrumentConfig,
            VolatilityForecaster,
            get_instrument_registry,
        )

        df = _make_ohlcv(300)

        # Mock data provider
        data_provider = MagicMock()
        data_provider.get_ohlcv.return_value = df

        # SMC factory
        def smc_factory(data):
            engine = MagicMock()
            # Return the dataframe with minimal SMC columns
            enriched = data.copy()
            enriched["BOS_SIGNAL"] = 0
            enriched["FVG_SIGNAL"] = 0
            enriched["OB_STRENGTH_NORM"] = 0
            enriched["RSI"] = 50.0
            enriched["MACD_Diff"] = 0.0
            enriched["ATR"] = enriched["High"] - enriched["Low"]
            engine.analyze.return_value = enriched
            return engine

        # Minimal subsystems
        config = get_instrument_registry().get("XAUUSD", InstrumentConfig(symbol="XAUUSD"))
        confluence = ConfluenceDetector(symbol="XAUUSD", instrument_config=config)
        llm_engine = MagicMock(spec=LLMNarrativeEngine)
        cache = SemanticCache()
        signal_store = MagicMock()

        scanner = SentinelScanner(
            data_provider=data_provider,
            smc_factory=smc_factory,
            regime_agent=MagicMock(),
            news_agent=MagicMock(),
            confluence=confluence,
            llm_engine=llm_engine,
            cache=cache,
            signal_store=signal_store,
        )

        # Should not raise
        result = scanner.scan_once()
        stats = scanner.get_stats()
        self.assertIn("bars_scanned", stats)

    def test_api_health_endpoint_in_testing_mode(self):
        """Health endpoint returns 200 with testing_mode=True."""
        from fastapi.testclient import TestClient
        from src.api.app import create_app
        from src.api.signal_store import SignalStore

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "signals.db")
            store = SignalStore(db_path=db_path)
            app = create_app(signal_store=store)
            client = TestClient(app)

            resp = client.get("/health")
            self.assertEqual(resp.status_code, 200)
            body = resp.json()
            self.assertIn("status", body)
            self.assertTrue(body["testing_mode"])

    def test_api_narratives_no_auth_in_testing_mode(self):
        """Narrative endpoints accessible without API key in testing mode."""
        from fastapi.testclient import TestClient
        from src.api.app import create_app
        from src.api.signal_store import SignalStore

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "signals.db")
            store = SignalStore(db_path=db_path)
            app = create_app(signal_store=store)
            client = TestClient(app)

            # Scanner status should work without key
            resp = client.get("/api/v1/scanner/status")
            self.assertEqual(resp.status_code, 200)

    def test_api_chat_accessible_in_testing_mode(self):
        """Chat endpoint returns 404 (signal not found) not 403 (tier blocked)."""
        from fastapi.testclient import TestClient
        from src.api.app import create_app
        from src.api.signal_store import SignalStore

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "signals.db")
            store = SignalStore(db_path=db_path)
            app = create_app(signal_store=store)
            client = TestClient(app)

            resp = client.post(
                "/api/v1/narratives/chat",
                json={"signal_id": "abcdef12", "question": "What is the risk?"},
            )
            # Should NOT be 403 (tier-blocked) — testing mode bypasses tier check
            # Will be 404 (signal not found) since the DB is empty
            self.assertNotEqual(resp.status_code, 403)

    def test_json_logging_formatter(self):
        """JSONFormatter produces valid JSON."""
        from src.intelligence.main import JSONFormatter
        import logging

        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Hello %s",
            args=("world",),
            exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        self.assertEqual(parsed["msg"], "Hello world")
        self.assertEqual(parsed["level"], "INFO")
        self.assertEqual(parsed["logger"], "test")

    def test_health_response_includes_components(self):
        """Health response model includes components list."""
        from src.api.models import ComponentHealth, HealthResponse, SystemStatus

        resp = HealthResponse(
            status=SystemStatus.HEALTHY,
            uptime_seconds=100.0,
            testing_mode=True,
            components=[
                ComponentHealth(name="llm_api", healthy=True),
                ComponentHealth(name="telegram", healthy=False),
            ],
            scanner_running=True,
            signals_generated=42,
        )
        data = resp.model_dump()
        self.assertEqual(len(data["components"]), 2)
        self.assertTrue(data["scanner_running"])
        self.assertEqual(data["signals_generated"], 42)

    def test_circuit_breaker_wiring_in_scanner(self):
        """Scanner respects circuit breaker state."""
        from src.intelligence.circuit_breaker import CircuitBreaker
        from src.intelligence.sentinel_scanner import SentinelScanner

        llm_breaker = CircuitBreaker("llm", failure_threshold=2, recovery_timeout=1.0)
        scanner = SentinelScanner(
            data_provider=MagicMock(),
            smc_factory=lambda df: MagicMock(analyze=MagicMock(return_value=df)),
            regime_agent=MagicMock(),
            news_agent=MagicMock(),
            confluence=MagicMock(),
            llm_engine=MagicMock(),
            cache=MagicMock(),
            signal_store=MagicMock(),
            llm_circuit_breaker=llm_breaker,
        )
        self.assertIs(scanner._llm_breaker, llm_breaker)

    def test_dockerfile_entry_point(self):
        """Dockerfile CMD points to src.intelligence.main."""
        dockerfile_path = os.path.join(
            os.path.dirname(__file__), "..", "infrastructure", "Dockerfile"
        )
        if not os.path.exists(dockerfile_path):
            self.skipTest("Dockerfile not found")

        with open(dockerfile_path) as f:
            content = f.read()

        self.assertIn("src.intelligence.main", content)
        self.assertIn("8000", content)
        self.assertNotIn("src.main", content)


if __name__ == "__main__":
    unittest.main()
