"""Smart Sentinel AI — Main operational entry point.

Wires all subsystems together and starts the scanning pipeline + API.

Usage:
    # Environment variables:
    #   ANTHROPIC_API_KEY     — Claude API key (required for LLM narratives)
    #   TELEGRAM_BOT_TOKEN    — Telegram bot token (optional)
    #   TELEGRAM_CHAT_ID      — Default Telegram chat ID (optional)
    #   DATA_DIR              — Path to OHLCV CSV data (default: ./data)
    #   SIGNAL_DB_PATH        — SQLite path (default: ./data/signals.db)
    #   VOL_MODE              — Forecaster mode: har/lgbm/hybrid (default: har)
    #   SYMBOLS               — Comma-separated symbols (default: XAUUSD)
    #   API_PORT              — API port (default: 8000)
    #   LOG_LEVEL             — Logging level (default: INFO)
    #   LOG_FORMAT            — text or json (default: text)
    #   SENTINEL_TESTING_MODE — 1=all features unlocked (default: 0, fail-closed)
    #   NARRATIVE_MODE        — template or llm (default: template — zero-cost, deterministic)
    #   DATA_SOURCE           — csv or mt5 (default: csv)
    #   MT5_LOGIN             — MT5 account number (when DATA_SOURCE=mt5)
    #   MT5_PASSWORD          — MT5 password (when DATA_SOURCE=mt5)
    #   MT5_SERVER            — MT5 broker server (when DATA_SOURCE=mt5)

    python -m src.intelligence.main
"""

from __future__ import annotations

import logging
import os
import sys
import signal as signal_mod
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger("smart_sentinel")


class JSONFormatter(logging.Formatter):
    """Structured JSON log formatter for production."""

    def format(self, record: logging.LogRecord) -> str:
        import json
        log_entry = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry, default=str)


def setup_logging(level: str = "INFO") -> None:
    """Configure logging. Uses JSON in production, text in dev."""
    log_format = os.environ.get("LOG_FORMAT", "text").lower()
    log_level = getattr(logging, level.upper(), logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    if log_format == "json":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )

    logging.basicConfig(level=log_level, handlers=[handler])

    # Suppress noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("hmmlearn").setLevel(logging.WARNING)


def assert_safe_production_config() -> None:
    """Refuse to start if production env still has the auth-bypass on.

    Trip wire: ENVIRONMENT in {production, prod} + SENTINEL_TESTING_MODE=1
    means every endpoint is accessible without an API key and grants
    INSTITUTIONAL access. Shipping that to prod = giving the product away.
    """
    env = os.environ.get("ENVIRONMENT", "development").strip().lower()
    testing = os.environ.get("SENTINEL_TESTING_MODE", "0").strip() == "1"
    if env in ("production", "prod") and testing:
        msg = (
            "FATAL: SENTINEL_TESTING_MODE=1 in ENVIRONMENT=production. "
            "Auth bypass is on; refusing to start. "
            "Set SENTINEL_TESTING_MODE=0 (or unset ENVIRONMENT) to proceed."
        )
        logger.critical(msg)
        sys.stderr.write(msg + "\n")
        sys.exit(2)
    if testing:
        logger.warning(
            "SENTINEL_TESTING_MODE=1 — auth bypass active "
            "(ENVIRONMENT=%s). Do not deploy to production with this set.",
            env,
        )


def build_system(
    symbols: List[str],
    data_dir: str,
    signal_db: str,
    vol_mode: str,
    anthropic_key: Optional[str] = None,
    telegram_token: Optional[str] = None,
    telegram_chat_id: Optional[str] = None,
    discord_webhook_url: Optional[str] = None,
    calendar_path: Optional[str] = None,
    data_source: str = "csv",
    mt5_login: Optional[int] = None,
    mt5_password: Optional[str] = None,
    mt5_server: Optional[str] = None,
) -> Dict[str, Any]:
    """Build all subsystems and wire them together.

    Args:
        data_source: "csv" for CSV files, "mt5" for MetaTrader 5 live data.
        mt5_login/mt5_password/mt5_server: MT5 credentials (when data_source="mt5").

    Returns:
        Dict with keys: scanner, api_app, app_state
    """
    import pandas as pd
    from src.intelligence.circuit_breaker import CircuitBreaker, CircuitState, HealthChecker
    from src.intelligence.confluence_detector import ConfluenceDetector
    from src.intelligence.regime_filter import RegimeFilter
    from src.intelligence.signal_state_machine import (
        SignalStateMachine,
        StateMachineConfig,
    )
    from src.intelligence.llm_narrative_engine import LLMNarrativeEngine, NarrativeTier
    from src.intelligence.template_narrative_engine import TemplateNarrativeEngine
    from src.intelligence.security import RateLimiter
    from src.intelligence.semantic_cache import SemanticCache
    from src.intelligence.sentinel_scanner import MultiSymbolScanner, SentinelScanner
    from src.intelligence.volatility_forecaster import (
        InstrumentConfig,
        VolatilityForecaster,
        get_instrument_registry,
    )
    from src.intelligence.data_providers import CSVDataProvider, MT5DataProvider
    from src.api.signal_store import SignalStore
    from src.delivery.telegram_notifier import TelegramNotifier
    from src.delivery.discord_notifier import DiscordNotifier
    from src.api.dependencies import AppState
    from src.risk.kill_switch import KillSwitch, KillSwitchConfig

    # 1. Instrument registry
    registry = get_instrument_registry()

    # 2. Data provider (CSV or MT5 live)
    if data_source == "mt5":
        data_provider = MT5DataProvider()
        connected = data_provider.connect(
            login=mt5_login,
            password=mt5_password,
            server=mt5_server,
        )
        if not connected:
            raise RuntimeError(
                "MT5 connection failed. Check credentials and that MT5 terminal is running."
            )
        logger.info("Data provider: MT5 (live data)")
    else:
        data_provider = CSVDataProvider(data_dir)
        logger.info("Data provider: CSVDataProvider(%s)", data_dir)

    # 3. SMC factory (creates SmartMoneyEngine per call)
    def smc_factory(df):
        from src.environment.strategy_features import SmartMoneyEngine
        return SmartMoneyEngine(df, config={})

    # 4. Regime & News agents (best-effort, fallback to None)
    regime_agent = _create_regime_agent()
    news_agent = _create_news_agent()

    # 5. Vol forecaster factory
    def vol_factory(config: InstrumentConfig) -> VolatilityForecaster:
        return VolatilityForecaster.create(mode=vol_mode, config=config)

    # 6. Narrative engine (template by default — algorithmic, $0 cost;
    #    set NARRATIVE_MODE=llm to use Claude Haiku/Sonnet instead)
    narrative_mode = os.environ.get("NARRATIVE_MODE", "template").lower()
    if narrative_mode == "llm":
        llm_engine = LLMNarrativeEngine(api_key=anthropic_key)
        logger.info("Narrative engine: LLMNarrativeEngine (Claude API)")
    else:
        llm_engine = TemplateNarrativeEngine()
        logger.info("Narrative engine: TemplateNarrativeEngine (algorithmic, $0 cost)")

    # 7. Semantic cache & signal store (cache only meaningful with LLM — template
    # engine generates in <1ms, so caching adds overhead and risks stale reads
    # if NARRATIVE_MODE is flipped to llm later)
    cache = SemanticCache() if narrative_mode == "llm" else None
    signal_store = SignalStore(db_path=signal_db)

    # 8. Notifier (Discord preferred when webhook URL is set, else Telegram)
    notifier = None
    notifier_label = "none"
    if discord_webhook_url:
        notifier = DiscordNotifier(webhook_url=discord_webhook_url)
        notifier_label = "discord"
        logger.info("Notifier: Discord (webhook)")
    elif telegram_token:
        notifier = TelegramNotifier(
            bot_token=telegram_token,
            default_chat_id=telegram_chat_id,
        )
        notifier_label = "telegram"
        logger.info("Notifier: Telegram")
    else:
        logger.info("Notifier: disabled (no DISCORD_WEBHOOK_URL or TELEGRAM_BOT_TOKEN)")

    # 8b. Circuit breakers for external services. LLM breaker only exists in
    # LLM mode — wrapping the template engine is pointless because it never
    # raises and never makes network calls.
    llm_breaker: Optional[CircuitBreaker] = None
    if narrative_mode == "llm":
        llm_breaker = CircuitBreaker(
            name="llm_api",
            failure_threshold=3,
            recovery_timeout=60.0,
            success_threshold=1,
        )
    notifier_breaker = CircuitBreaker(
        name=notifier_label,
        failure_threshold=5,
        recovery_timeout=120.0,
        success_threshold=1,
    )
    circuit_breakers: Dict[str, CircuitBreaker] = {notifier_label: notifier_breaker}
    if llm_breaker is not None:
        circuit_breakers["llm"] = llm_breaker

    # 8c. Health checker
    health_checker = HealthChecker()
    if llm_breaker is not None:
        health_checker.register("llm_api", lambda: llm_breaker.state != CircuitState.OPEN)
    health_checker.register(notifier_label, lambda: notifier_breaker.state != CircuitState.OPEN)
    health_checker.register(
        "signal_store", lambda: signal_store is not None
    )

    # 8d. Rate limiter (global, used by API middleware)
    rate_limiter = RateLimiter(max_requests=100, window_seconds=60.0)

    # 8d-bis. Operational kill-switch (4 rules: streak / daily-DD / vol-spike /
    # broker-disconnect). Configurable via env; defaults match the XAU M15
    # personal-testing tier in baseline_2019_2025.md.
    ks_starting_equity = float(os.environ.get("KILL_SWITCH_EQUITY", "1000"))
    ks_max_streak = int(os.environ.get("KILL_SWITCH_MAX_LOSSES", "4"))
    ks_dd_pct = float(os.environ.get("KILL_SWITCH_DAILY_DD_PCT", "0.05"))
    ks_vol_z = float(os.environ.get("KILL_SWITCH_VOL_Z", "3.0"))
    ks_heartbeat_s = float(os.environ.get("KILL_SWITCH_HEARTBEAT_MAX_S", "120"))
    operational_kill_switch = KillSwitch(
        config=KillSwitchConfig(
            max_consecutive_losses=ks_max_streak,
            daily_dd_limit_pct=ks_dd_pct,
            vol_zscore_limit=ks_vol_z,
            heartbeat_max_silence_s=ks_heartbeat_s,
        ),
        starting_equity=ks_starting_equity,
    )
    logger.info(
        "Operational kill-switch armed: streak<=%d, DD<=%.0f%%, vol_z<=%.1fσ, heartbeat<=%.0fs",
        ks_max_streak, ks_dd_pct * 100, ks_vol_z, ks_heartbeat_s,
    )

    # 8e. Metrics registry — wired in so /metrics exposes the request_logging
    # histogram and any subsystem counters/gauges. Without this, /metrics
    # returns an empty body (eval 16 finding #1).
    from src.performance.metrics import MetricsRegistry
    metrics_registry = MetricsRegistry(prefix="sentinel")

    # 9. Build scanner (single or multi-symbol)
    if len(symbols) == 1:
        config = registry.get(symbols[0], InstrumentConfig(symbol=symbols[0]))
        vol_forecaster = vol_factory(config)
        confluence = ConfluenceDetector(
            symbol=symbols[0],
            instrument_config=config,
        )
        regime_filter = (
            RegimeFilter.from_env()
            if os.environ.get("REGIME_FILTER_ENABLED", "1") == "1"
            else None
        )
        if regime_filter is not None:
            logger.info(
                "Regime filter ON: ny_mode=%s, vol_pctl_max=%s",
                regime_filter.ny_mode, regime_filter.vol_pctl_max,
            )
        # State machine — empirical thresholds from the post-RegimeFilter
        # XAU 7-yr replay. The legacy 75/55 defaults were inherited from
        # marketing copy and made the gate mathematically unreachable
        # (see reports/audit/audit_report.md, score_max=55.5). 40/25 sits
        # at p75/p25 of the post-filter score distribution and matches the
        # config that produced PF 1.30 OOS in scripts/backtest_combo_E.py.
        sm_enter = float(os.environ.get("STATE_MACHINE_ENTER_THRESHOLD", "40"))
        sm_exit = float(os.environ.get("STATE_MACHINE_EXIT_THRESHOLD", "25"))
        state_machine = (
            SignalStateMachine(
                StateMachineConfig(
                    symbol=symbols[0],
                    enter_threshold=sm_enter,
                    exit_threshold=sm_exit,
                )
            )
            if os.environ.get("STATE_MACHINE_ENABLED", "1") == "1"
            else None
        )
        if state_machine is not None:
            logger.info(
                "State machine ON: enter=%.0f, exit=%.0f", sm_enter, sm_exit,
            )

        # Persistence — survive scanner restarts so a stale signal isn't
        # re-emitted from cold start. Honours STATE_MACHINE_PERSIST_PATH;
        # default sits next to the signal store DB.
        sm_persist_path = os.environ.get(
            "STATE_MACHINE_PERSIST_PATH",
            os.path.join(os.path.dirname(signal_db) or ".", "state_machine.json"),
        )

        scanner = SentinelScanner(
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
            symbol=symbols[0],
            timeframe=config.timeframe,
            llm_circuit_breaker=llm_breaker,
            notifier_circuit_breaker=notifier_breaker,
            regime_filter=regime_filter,
            kill_switch=operational_kill_switch,
            state_machine=state_machine,
            persistence_path=sm_persist_path if state_machine is not None else None,
        )
    else:
        scanner = MultiSymbolScanner(
            symbols=symbols,
            instrument_registry=registry,
            data_provider=data_provider,
            smc_factory=smc_factory,
            regime_agent=regime_agent,
            news_agent=news_agent,
            llm_engine=llm_engine,
            cache=cache,
            signal_store=signal_store,
            notifier=notifier,
            vol_forecaster_factory=vol_factory,
        )

    # 10. Calibrate vol forecasters with available data
    logger.info("Calibrating volatility forecasters...")
    calendar_df = None
    if calendar_path and os.path.exists(calendar_path):
        calendar_df = pd.read_csv(calendar_path)
        logger.info("Loaded calendar: %d events", len(calendar_df))

    _calibrate_system(scanner, data_provider, symbols, registry, calendar_df)

    # 11. Build FastAPI app
    from src.api.app import create_app
    api_app = create_app(
        signal_store=signal_store,
        llm_engine=llm_engine,
        scanner=scanner,
        circuit_breakers=circuit_breakers,
        health_checker=health_checker,
        rate_limiter=rate_limiter,
        metrics_registry=metrics_registry,
        operational_kill_switch=operational_kill_switch,
    )

    return {
        "scanner": scanner,
        "api_app": api_app,
        "signal_store": signal_store,
        "data_provider": data_provider,
        "circuit_breakers": circuit_breakers,
        "health_checker": health_checker,
        "notifier": notifier,
        "notifier_label": notifier_label,
        "operational_kill_switch": operational_kill_switch,
    }


def _check_coverage_or_abort(df: Any, symbol: str, timeframe: str, min_coverage: float = 0.95) -> None:
    """Fail-fast if the OHLCV feed has < min_coverage of expected bars.

    A 63%-coverage XAU feed (vs 97.6% for the same period from a different
    source) caused BOS to fire on 100% of bars during 2026-04-23 — silently
    corrupted every backtest. This gate blocks startup so the operator
    can re-download a clean feed before any signal is published.

    Bypass with COVERAGE_GATE=off when running on intentionally-sparse data
    (smoke tests, single-day fixtures).
    """
    import os as _os
    if _os.environ.get("COVERAGE_GATE", "on").lower() == "off":
        return
    if df is None or len(df) < 2:
        return  # not enough rows to estimate coverage; let calibration error out instead
    try:
        import pandas as pd
        from src.intelligence.data_quality import TIMEFRAME_MINUTES
        tf_min = TIMEFRAME_MINUTES.get(timeframe)
        if tf_min is None:
            return
        span_min = (df.index[-1] - df.index[0]) / pd.Timedelta(minutes=1)
        expected = span_min / tf_min
        if expected <= 0:
            return
        coverage = len(df) / expected
        if coverage < min_coverage:
            msg = (
                f"FATAL: {symbol} {timeframe} coverage {coverage:.1%} < "
                f"{min_coverage:.0%} threshold (got {len(df)} / expected ~{expected:.0f}). "
                f"Re-download from a clean feed before starting. "
                f"Bypass: COVERAGE_GATE=off."
            )
            logger.critical(msg)
            sys.stderr.write(msg + "\n")
            sys.exit(3)
        logger.info(
            "Coverage gate OK for %s %s: %.1f%% (%d / ~%.0f bars)",
            symbol, timeframe, coverage * 100, len(df), expected,
        )
    except SystemExit:
        raise
    except Exception as e:
        # Non-fatal if the index isn't a DatetimeIndex etc — calibration
        # downstream will surface the real shape problem.
        logger.warning("Coverage gate skipped for %s: %s", symbol, e)


def _calibrate_system(
    scanner: Any,
    data_provider: Any,
    symbols: List[str],
    registry: Dict[str, Any],
    calendar_df: Any,
) -> None:
    """Calibrate vol forecasters with training data."""
    import pandas as pd

    if hasattr(scanner, "calibrate_forecasters"):
        # MultiSymbolScanner
        ohlcv_data = {}
        for symbol in symbols:
            try:
                config = registry.get(symbol)
                tf = config.timeframe if config else "M15"
                df = data_provider.get_ohlcv(symbol, tf, 10000)
                _check_coverage_or_abort(df, symbol, tf)
                ohlcv_data[symbol] = df.reset_index()
                logger.info("Loaded %d bars for %s calibration", len(df), symbol)
            except SystemExit:
                raise
            except Exception as e:
                logger.warning("No calibration data for %s: %s", symbol, e)
        scanner.calibrate_forecasters(ohlcv_data, calendar_df)
    elif hasattr(scanner, "_vol_forecaster") and scanner._vol_forecaster is not None:
        # Single SentinelScanner
        try:
            df = data_provider.get_ohlcv(
                scanner._symbol, scanner._timeframe, 10000
            )
            _check_coverage_or_abort(df, scanner._symbol, scanner._timeframe)
            scanner._vol_forecaster.calibrate(df.reset_index(), calendar_df)
            logger.info("Calibrated vol forecaster for %s", scanner._symbol)
        except SystemExit:
            raise
        except Exception as e:
            logger.warning("Vol calibration failed: %s", e)


def _create_regime_agent() -> Any:
    """Try to create MarketRegimeAgent, return mock if unavailable."""
    try:
        from src.agents.market_regime_agent import MarketRegimeAgent
        return MarketRegimeAgent()
    except Exception:
        logger.warning("MarketRegimeAgent unavailable — regime analysis disabled")
        return _NullAgent()


def _create_news_agent() -> Any:
    """Try to create NewsAnalysisAgent, return mock if unavailable."""
    try:
        from src.agents.news_analysis_agent import NewsAnalysisAgent
        return NewsAnalysisAgent()
    except Exception:
        logger.warning("NewsAnalysisAgent unavailable — news analysis disabled")
        return _NullAgent()


class _NullAgent:
    """Stub agent that returns None for all calls."""
    def analyze(self, *args, **kwargs):
        return None
    def evaluate_news_impact(self, *args, **kwargs):
        return None


def main() -> None:
    """Main entry point."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    log_level = os.environ.get("LOG_LEVEL", "INFO")
    setup_logging(log_level)
    assert_safe_production_config()

    symbols_str = os.environ.get("SYMBOLS", "XAUUSD")
    symbols = [s.strip() for s in symbols_str.split(",") if s.strip()]
    data_dir = os.environ.get("DATA_DIR", "./data")
    signal_db = os.environ.get("SIGNAL_DB_PATH", "./data/signals.db")
    vol_mode = os.environ.get("VOL_MODE", "har")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    telegram_chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    discord_webhook_url = os.environ.get("DISCORD_WEBHOOK_URL")
    calendar_path = os.environ.get("CALENDAR_PATH")
    api_port = int(os.environ.get("API_PORT", "8000"))

    # MT5 live data settings
    data_source = os.environ.get("DATA_SOURCE", "csv").lower()
    mt5_login = os.environ.get("MT5_LOGIN")
    mt5_password = os.environ.get("MT5_PASSWORD")
    mt5_server = os.environ.get("MT5_SERVER")

    from src.api.auth import TESTING_MODE

    logger.info("=" * 60)
    logger.info("  Smart Sentinel AI — Starting")
    logger.info("  Symbols: %s", symbols)
    logger.info("  Data source: %s", data_source.upper())
    logger.info("  Vol mode: %s", vol_mode)
    logger.info("  API port: %d", api_port)
    logger.info("  Testing mode: %s", "ON (all features unlocked)" if TESTING_MODE else "OFF (tier-gated)")
    logger.info("=" * 60)

    if not anthropic_key:
        logger.warning("ANTHROPIC_API_KEY not set — LLM narratives disabled")

    # Build
    system = build_system(
        symbols=symbols,
        data_dir=data_dir,
        signal_db=signal_db,
        vol_mode=vol_mode,
        anthropic_key=anthropic_key,
        telegram_token=telegram_token,
        telegram_chat_id=telegram_chat_id,
        discord_webhook_url=discord_webhook_url,
        calendar_path=calendar_path,
        data_source=data_source,
        mt5_login=int(mt5_login) if mt5_login else None,
        mt5_password=mt5_password,
        mt5_server=mt5_server,
    )

    scanner = system["scanner"]
    api_app = system["api_app"]
    notifier = system.get("notifier")

    shutdown_event = threading.Event()
    scanner_crashed = threading.Event()

    def _signal_handler(sig, frame):
        logger.info("Shutdown signal received")
        shutdown_event.set()
        if hasattr(scanner, "shutdown"):
            scanner.shutdown()

    signal_mod.signal(signal_mod.SIGINT, _signal_handler)
    signal_mod.signal(signal_mod.SIGTERM, _signal_handler)

    def _ping_notifier(msg: str) -> None:
        """Fire-and-forget Discord/Telegram ping on fatal events. Never raises."""
        if notifier is None or not getattr(notifier, "is_configured", lambda: False)():
            return
        try:
            if hasattr(notifier, "send_raw"):
                notifier.send_raw(msg)
        except Exception as e:
            logger.warning("Notifier ping failed: %s", e)

    def _run_scanner() -> None:
        """Scanner thread entry. Logs full traceback on crash — previously
        the daemon thread died silently, leaving operators guessing why
        signals stopped."""
        try:
            if not shutdown_event.is_set():
                scanner.start(blocking=True)
        except Exception as e:
            logger.exception("Scanner thread crashed: %s", e)
            scanner_crashed.set()
            _ping_notifier(
                f":rotating_light: **Smart Sentinel CRASHED**\n"
                f"Scanner thread died: `{type(e).__name__}: {str(e)[:300]}`\n"
                f"API may still be up but no new signals will be published."
            )

    def _watchdog() -> None:
        """Check scanner thread liveness every 30s. Fires one notification
        if the thread dies while the process is still running (API thread
        would otherwise keep the process alive and mask the failure)."""
        fired = False
        while not shutdown_event.is_set():
            shutdown_event.wait(timeout=30.0)
            if shutdown_event.is_set():
                return
            if not scanner_thread.is_alive() and not fired:
                fired = True
                if not scanner_crashed.is_set():
                    logger.critical(
                        "Scanner thread stopped without exception — likely "
                        "a clean exit from inside scanner.start(). This "
                        "should not happen in normal operation."
                    )
                    _ping_notifier(
                        ":warning: Smart Sentinel scanner thread exited unexpectedly "
                        "(no exception). Process still alive but no new signals."
                    )

    scanner_thread = threading.Thread(target=_run_scanner, name="sentinel-scanner", daemon=True)
    scanner_thread.start()

    watchdog_thread = threading.Thread(target=_watchdog, name="sentinel-watchdog", daemon=True)
    watchdog_thread.start()

    _ping_notifier(
        f":white_check_mark: Smart Sentinel online — symbols={symbols} "
        f"source={data_source.upper()} vol={vol_mode}"
    )

    exit_code = 0
    try:
        import uvicorn
        uvicorn.run(api_app, host="0.0.0.0", port=api_port, log_level=log_level.lower())
    except ImportError:
        logger.critical("uvicorn not installed. Install with: pip install uvicorn")
        exit_code = 1
    except KeyboardInterrupt:
        logger.info("API server stopped by user")
    except Exception as e:
        logger.exception("API server crashed: %s", e)
        _ping_notifier(
            f":rotating_light: **Smart Sentinel CRASHED**\n"
            f"API server died: `{type(e).__name__}: {str(e)[:300]}`"
        )
        exit_code = 1
    finally:
        shutdown_event.set()
        if hasattr(scanner, "shutdown"):
            try:
                scanner.shutdown()
            except Exception as e:
                logger.warning("Scanner shutdown error: %s", e)
        data_provider = system.get("data_provider")
        if data_provider is not None and hasattr(data_provider, "disconnect"):
            try:
                data_provider.disconnect()
                logger.info("Data provider disconnected")
            except Exception as e:
                logger.warning("Data provider disconnect error: %s", e)
        if scanner_crashed.is_set() and exit_code == 0:
            exit_code = 1

    if exit_code != 0:
        import sys
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
