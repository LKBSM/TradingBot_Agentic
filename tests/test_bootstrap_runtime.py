"""Tests for the runtime bootstrap + FastAPI lifespan wiring (Chantier 3 Étape 5)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.bootstrap import (
    BootstrapConfigurationError,
    build_market_reading_assembler,
    build_market_reading_scheduler,
    env_flag,
    env_int,
    is_bootstrap_enabled,
)


class _FakeAnthropic:
    """Minimal stand-in for the Anthropic SDK client.

    The LLM-backed factories (chatbot / scanner translator) only hold this as a
    client; no methods are called by the bootstrap path itself, so an empty
    object is sufficient. The MarketReading assembler needs no client — its
    narrated reading is composed by a deterministic template.
    """


@pytest.fixture(autouse=True)
def _no_deep_history_seed(monkeypatch):
    """Disable the NW-7 deep-history seed for every test in this module.

    ``build_market_reading_assembler`` imports the bundled seed (~180k M15 bars
    PER measured market) into candles.db. On production that is a one-time boot
    cost against a persistent disk. Here each test points candles.db at a fresh
    ``tmp_path``, so every call re-imported ~360k rows — minutes per test, which
    read as a hang rather than a slow suite.

    Autouse (not folded into ``isolated_env``) because the tests that build the
    assembler with their own hand-rolled env — e.g.
    ``test_assembler_builds_without_an_anthropic_key`` — do not take that fixture,
    and they are the ones that stalled. The seed module ships this opt-out for
    exactly this case; nothing asserted here depends on the seeded history.
    """
    monkeypatch.setenv("SENTINEL_SEED_MEASURES", "0")


@pytest.fixture
def isolated_env(tmp_path, monkeypatch):
    """Point every store at tmp + provide fake API keys + fake Anthropic client.

    The Anthropic SDK is optional in CI; we patch the client factory so the
    bootstrap can succeed even when ``anthropic`` is not installed locally.
    """
    monkeypatch.setenv("TWELVE_DATA_API_KEY", "test-twelve-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("MARKET_READINGS_DB_PATH", str(tmp_path / "mr.db"))
    monkeypatch.setenv("CANDLES_DB_PATH", str(tmp_path / "candles.db"))
    monkeypatch.setenv("NEWS_CACHE_DB_PATH", str(tmp_path / "news.db"))
    monkeypatch.setenv("SCHEDULER_TICK_INTERVAL_SECONDS", "3600")  # no tick during test
    monkeypatch.setattr(
        "src.api.bootstrap._build_anthropic_client", lambda: _FakeAnthropic()
    )
    yield


# ===================================================================== #
# Env helpers
# ===================================================================== #
class TestEnvHelpers:
    def test_env_flag_truthy(self, monkeypatch):
        for v in ("1", "true", "TRUE", "yes", "on"):
            monkeypatch.setenv("X", v)
            assert env_flag("X") is True

    def test_env_flag_falsy_and_default(self, monkeypatch):
        monkeypatch.setenv("X", "false")
        assert env_flag("X") is False
        monkeypatch.delenv("X", raising=False)
        assert env_flag("X", default=False) is False
        assert env_flag("X", default=True) is True

    def test_env_int(self, monkeypatch):
        monkeypatch.setenv("N", "42")
        assert env_int("N", 7) == 42
        monkeypatch.setenv("N", "notanint")
        assert env_int("N", 7) == 7
        monkeypatch.delenv("N", raising=False)
        assert env_int("N", 7) == 7

    def test_is_bootstrap_enabled_reads_BOOTSTRAP_ENABLED(self, monkeypatch):
        # Default = False (tests stay light unless explicitly opted-in).
        monkeypatch.delenv("BOOTSTRAP_ENABLED", raising=False)
        assert is_bootstrap_enabled() is False
        monkeypatch.setenv("BOOTSTRAP_ENABLED", "true")
        assert is_bootstrap_enabled() is True
        monkeypatch.setenv("BOOTSTRAP_ENABLED", "false")
        assert is_bootstrap_enabled() is False


# ===================================================================== #
# Bootstrap factories
# ===================================================================== #
class TestBootstrapFactories:
    def test_build_assembler_with_news(self, isolated_env):
        assembler = build_market_reading_assembler(enable_news=True)
        assert assembler is not None
        assert assembler.readings_store is not None
        assert assembler.candles_store is not None
        # News pipeline wired → assembler will populate events.
        assert assembler._news_pipeline is not None  # noqa: SLF001

    def test_build_assembler_without_news(self, isolated_env):
        assembler = build_market_reading_assembler(enable_news=False)
        assert assembler._news_pipeline is None  # noqa: SLF001

    def test_build_scheduler_bound_to_assembler_stores(self, isolated_env):
        assembler = build_market_reading_assembler(enable_news=False)
        scheduler = build_market_reading_scheduler(assembler)
        assert scheduler is not None
        assert scheduler.running is False  # built, not started

    def test_assembler_builds_without_an_anthropic_key(self, tmp_path, monkeypatch):
        """The MarketReading assembler needs NO ``ANTHROPIC_API_KEY`` — deliberate.

        This test used to assert the opposite. The narrated reading became a
        deterministic template (the LLM description engine was removed), so the
        assembler stopped needing a key and the assertion rotted into a permanent
        red. What is worth guarding is the CURRENT contract, in both directions:
        the assembler builds keyless (here), and every LLM-backed factory still
        fails fast and names the variable (next test).
        """
        monkeypatch.setenv("TWELVE_DATA_API_KEY", "test-key")
        monkeypatch.setenv("MARKET_READINGS_DB_PATH", str(tmp_path / "mr.db"))
        monkeypatch.setenv("CANDLES_DB_PATH", str(tmp_path / "candles.db"))
        monkeypatch.setenv("NEWS_CACHE_DB_PATH", str(tmp_path / "news.db"))
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        assembler = build_market_reading_assembler(enable_news=False)
        assert assembler is not None

    def test_llm_factories_fail_fast_and_name_the_missing_key(self, monkeypatch):
        """Every factory that DOES need the key errors clearly without it.

        Covers the scanner translator and the MIA-4S landing demo agent — both
        reach the Anthropic client directly, so a missing key must surface as an
        explicit misconfiguration, never as a half-wired service.
        """
        from src.api.bootstrap import build_demo_chat_agent, build_scanner_translator

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        for factory in (build_scanner_translator, build_demo_chat_agent):
            with pytest.raises(BootstrapConfigurationError) as exc_info:
                factory()
            # The message must name the env var so the operator knows the fix.
            assert "ANTHROPIC_API_KEY" in str(exc_info.value), factory.__name__

    def test_store_paths_read_from_env_vars(self, isolated_env, tmp_path):
        """Each store honours its *_DB_PATH env var (shared Fly.io volume in prod)."""
        assembler = build_market_reading_assembler(enable_news=True)
        # Reach into each store; their _db_path attribute is the resolved path.
        # The fixture set every path to tmp_path / "<name>.db" — confirm they
        # were picked up rather than the package default ./data/*.db paths.
        assert str(tmp_path) in str(assembler.readings_store._db_path)  # noqa: SLF001
        assert str(tmp_path) in str(assembler.candles_store._db_path)  # noqa: SLF001
        # News cache is owned by the pipeline, not exposed; the env var alone
        # is the contract — the store would have failed to open otherwise.


# ===================================================================== #
# Lifespan — env-gated auto bootstrap
# ===================================================================== #
class TestLifespanAutoBootstrap:
    def test_startup_bootstraps_assembler_and_starts_scheduler(
        self, isolated_env, monkeypatch
    ):
        monkeypatch.setenv("BOOTSTRAP_ENABLED", "true")
        monkeypatch.setenv("SCHEDULER_ENABLED", "true")
        monkeypatch.setenv("NEWS_PIPELINE_ENABLED", "true")

        app = create_app()
        with TestClient(app):
            state = app.state.app_state
            assert state.market_reading_assembler is not None
            assert state.market_reading_scheduler is not None
            assert state.market_reading_scheduler.running is True
        # After shutdown the coordinator must have stopped the scheduler.
        assert app.state.app_state.market_reading_scheduler.running is False

    def test_scheduler_disabled_builds_assembler_only(
        self, isolated_env, monkeypatch
    ):
        monkeypatch.setenv("BOOTSTRAP_ENABLED", "true")
        monkeypatch.setenv("SCHEDULER_ENABLED", "false")

        app = create_app()
        with TestClient(app):
            state = app.state.app_state
            assert state.market_reading_assembler is not None
            assert state.market_reading_scheduler is None

    def test_bootstrap_enabled_false_skips_assembler_build(
        self, isolated_env, monkeypatch
    ):
        """Brief: 'BOOTSTRAP_ENABLED=false permet de créer l'app sans bootstrap'."""
        monkeypatch.setenv("BOOTSTRAP_ENABLED", "false")
        monkeypatch.setenv("SCHEDULER_ENABLED", "true")  # ignored when no assembler

        app = create_app()
        with TestClient(app):
            state = app.state.app_state
            assert state.market_reading_assembler is None
            assert state.market_reading_scheduler is None


# ===================================================================== #
# Regression — create_app() with no env must stay light (Chantier 1/2 compat)
# ===================================================================== #
class TestNoBootstrapByDefault:
    def test_no_env_no_bootstrap(self, monkeypatch):
        # Ensure the auto-bootstrap flags are absent.
        monkeypatch.delenv("BOOTSTRAP_ENABLED", raising=False)
        monkeypatch.delenv("SCHEDULER_ENABLED", raising=False)

        app = create_app()
        with TestClient(app):
            state = app.state.app_state
            assert state.market_reading_assembler is None
            assert state.market_reading_scheduler is None

    def test_endpoint_503_without_assembler(self, monkeypatch):
        monkeypatch.delenv("BOOTSTRAP_ENABLED", raising=False)
        monkeypatch.delenv("SCHEDULER_ENABLED", raising=False)
        app = create_app()
        with TestClient(app) as client:
            resp = client.get("/api/market-reading?instrument=XAUUSD&timeframe=M15")
            assert resp.status_code == 503

    def test_endpoint_200_with_injected_assembler(self, monkeypatch):
        """Injected assembler → lifespan does not override it; endpoint 200."""
        from tests.test_market_reading_endpoint import _StubAssembler

        monkeypatch.delenv("BOOTSTRAP_ENABLED", raising=False)
        app = create_app(market_reading_assembler=_StubAssembler())
        with TestClient(app) as client:
            resp = client.get("/api/market-reading?instrument=XAUUSD&timeframe=M15")
            assert resp.status_code == 200
            assert resp.json()["header"]["instrument"] == "XAUUSD"
