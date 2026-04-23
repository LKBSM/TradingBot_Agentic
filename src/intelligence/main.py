"""Smart Sentinel AI — Main operational entry point.

Wires all subsystems together and starts the scanning pipeline + API.

Usage:
    # Environment variables:
    #   ANTHROPIC_API_KEY     — Claude API key (required for LLM narratives)
    #   TELEGRAM_BOT_TOKEN    — Telegram bot token (optional)
    #   TELEGRAM_CHAT_ID      — Default Telegram chat ID (optional)
    #   DATA_DIR              — Path to OHLCV CSV data (default: ./data)
    #   SIGNAL_DB_PATH        — SQLite path (default: ./data/signals.db)
    #   VOL_MODE              — Forecaster mode: har/lgbm/hybrid (default: hybrid)
    #   SYMBOLS               — Comma-separated symbols (default: XAUUSD)
    #   API_PORT              — API port (default: 8000)
    #   LOG_LEVEL             — Logging level (default: INFO)
    #   LOG_FORMAT            — text or json (default: text)
    #   SENTINEL_TESTING_MODE — 1=all features unlocked (default: 1)
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

    # 9. Build scanner (single or multi-symbol)
    if len(symbols) == 1:
        config = registry.get(symbols[0], InstrumentConfig(symbol=symbols[0]))
        vol_forecaster = vol_factory(config)
        confluence = ConfluenceDetector(
            symbol=symbols[0],
            instrument_config=config,
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
    )

    return {
        "scanner": scanner,
        "api_app": api_app,
        "signal_store": signal_store,
        "data_provider": data_provider,
        "circuit_breakers": circuit_breakers,
        "health_checker": health_checker,
    }


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
                ohlcv_data[symbol] = df.reset_index()
                logger.info("Loaded %d bars for %s calibration", len(df), symbol)
            except Exception as e:
                logger.warning("No calibration data for %s: %s", symbol, e)
        scanner.calibrate_forecasters(ohlcv_data, calendar_df)
    elif hasattr(scanner, "_vol_forecaster") and scanner._vol_forecaster is not None:
        # Single SentinelScanner
        try:
            df = data_provider.get_ohlcv(
                scanner._symbol, scanner._timeframe, 10000
            )
            scanner._vol_forecaster.calibrate(df.reset_index(), calendar_df)
            logger.info("Calibrated vol forecaster for %s", scanner._symbol)
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
    # Parse environment
    log_level = os.environ.get("LOG_LEVEL", "INFO")
    setup_logging(log_level)

    symbols_str = os.environ.get("SYMBOLS", "XAUUSD")
    symbols = [s.strip() for s in symbols_str.split(",") if s.strip()]
    data_dir = os.environ.get("DATA_DIR", "./data")
    signal_db = os.environ.get("SIGNAL_DB_PATH", "./data/signals.db")
    vol_mode = os.environ.get("VOL_MODE", "hybrid")
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

    # Graceful shutdown
    shutdown_event = threading.Event()

    def _signal_handler(sig, frame):
        logger.info("Shutdown signal received")
        shutdown_event.set()
        if hasattr(scanner, "shutdown"):
            scanner.shutdown()

    signal_mod.signal(signal_mod.SIGINT, _signal_handler)
    signal_mod.signal(signal_mod.SIGTERM, _signal_handler)

    # Start scanner in background thread
    scanner_thread = threading.Thread(
        target=lambda: scanner.start(blocking=True) if not shutdown_event.is_set() else None,
        daemon=True,
    )
    scanner_thread.start()

    # Start API (blocking)
    try:
        import uvicorn
        uvicorn.run(api_app, host="0.0.0.0", port=api_port, log_level=log_level.lower())
    except ImportError:
        logger.error("uvicorn not installed. Install with: pip install uvicorn")
    except Exception:
        logger.info("API server stopped")
    finally:
        shutdown_event.set()
        if hasattr(scanner, "shutdown"):
            scanner.shutdown()
        # Disconnect data provider (MT5 connections must be explicitly closed)
        data_provider = system.get("data_provider")
        if data_provider is not None and hasattr(data_provider, "disconnect"):
            try:
                data_provider.disconnect()
                logger.info("Data provider disconnected")
            except Exception as e:
                logger.warning("Data provider disconnect error: %s", e)


if __name__ == "__main__":
    main()
