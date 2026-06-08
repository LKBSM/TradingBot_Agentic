"""Runtime bootstrap for the MIA Markets MarketReading engine (Chantier 3 Étape 5).

Factories that wire the full production assembler + hybrid scheduler from
environment configuration. Kept out of ``create_app`` so that:

  - tests calling ``create_app()`` (no env, no injection) pay no cost and hit
    no network — the endpoint simply returns 503 until an assembler is wired;
  - production opts in via env (``BOOTSTRAP_ENABLED=true`` etc.), and the
    FastAPI lifespan calls these factories at startup.

Failure modes (deliberate fail-fast — no silent degradation):
  - Missing ``ANTHROPIC_API_KEY``    → :class:`BootstrapConfigurationError`.
    Silently routing to template fallback would mask the misconfiguration
    and surface as a "haiku_generated" tier that never actually runs the LLM.
  - Missing ``TWELVE_DATA_API_KEY``  → ``TwelveDataProvider`` raises
    ``ValueError`` (same intent — no implicit fallback).
  - ``NEWS_PIPELINE_ENABLED=false``  → assembler emits empty events blocks
    (this *is* a legitimate operational toggle, not a misconfig).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)


class BootstrapConfigurationError(RuntimeError):
    """Raised when a required env var or dependency is missing at boot.

    Distinct from generic ``RuntimeError`` so the FastAPI lifespan can catch
    *only* configuration errors and degrade the endpoint to 503, without
    swallowing genuine runtime bugs (e.g. an import error inside a factory).
    """


# --------------------------------------------------------------------------- #
# Env helpers
# --------------------------------------------------------------------------- #
def env_flag(name: str, default: bool = False) -> bool:
    """Parse a boolean env var. Truthy: 1/true/yes/on (case-insensitive)."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("env %s=%r is not an int — using default %d", name, raw, default)
        return default


def is_bootstrap_enabled() -> bool:
    """Return True when the runtime bootstrap should run at app startup.

    Reads ``BOOTSTRAP_ENABLED`` (default ``False`` — production opts in
    explicitly). Centralised so app.py and tests agree on the env-var name
    and the falsy/truthy parsing. Keep ``False`` as the default so the
    28 existing test suites calling ``create_app()`` with no env stay
    light (no network, no Anthropic init).
    """
    return env_flag("BOOTSTRAP_ENABLED", default=False)


# --------------------------------------------------------------------------- #
# Factories
# --------------------------------------------------------------------------- #
def build_market_reading_assembler(enable_news: Optional[bool] = None) -> Any:
    """Instantiate the production MarketReadingAssembler from env config.

    Reads ``TWELVE_DATA_API_KEY`` (data provider), ``ANTHROPIC_API_KEY``
    (optional — Haiku description engine), and ``NEWS_PIPELINE_ENABLED``.
    """
    from src.intelligence.data_providers import TwelveDataProvider
    from src.intelligence.haiku_description_engine import HaikuDescriptionEngine
    from src.intelligence.market_reading_assembler import MarketReadingAssembler
    from src.intelligence.news_pipeline import NewsPipeline
    from src.storage import (
        CandlesCacheStore,
        HaikuDescriptionCacheStore,
        MarketReadingsStore,
        NewsCacheStore,
    )

    if enable_news is None:
        enable_news = env_flag("NEWS_PIPELINE_ENABLED", True)

    data_provider = TwelveDataProvider()  # reads TWELVE_DATA_API_KEY from env
    readings_store = MarketReadingsStore()
    candles_store = CandlesCacheStore()
    haiku_cache_store = HaikuDescriptionCacheStore()

    anthropic_client = _build_anthropic_client()  # raises if missing
    haiku_engine = HaikuDescriptionEngine(anthropic_client, haiku_cache_store)

    news_pipeline = NewsPipeline(NewsCacheStore()) if enable_news else None

    assembler = MarketReadingAssembler(
        data_provider=data_provider,
        readings_store=readings_store,
        candles_store=candles_store,
        description_engine=haiku_engine,
        news_pipeline=news_pipeline,
    )
    logger.info("MarketReadingAssembler built (news=%s)", "on" if news_pipeline else "off")
    return assembler


def build_chatbot(assembler: Any) -> Any:
    """Instantiate the niveau 1.5 strict Chatbot (Chantier 4) from env config.

    Requires the MarketReadingAssembler (for the signal_summary + the
    get_market_reading tool) and ``ANTHROPIC_API_KEY``. Fail-fast — a missing
    assembler or API key raises :class:`BootstrapConfigurationError` rather than
    silently producing a half-wired chatbot.
    """
    if assembler is None:
        raise BootstrapConfigurationError(
            "CHATBOT_ENABLED requires the MarketReadingAssembler to be "
            "bootstrapped first (set BOOTSTRAP_ENABLED=true)."
        )

    from src.intelligence.chatbot.adversarial_filter import AdversarialFilter
    from src.intelligence.chatbot.chatbot import Chatbot
    from src.intelligence.chatbot.output_filter import OutputFilter
    from src.intelligence.chatbot.signal_summary_provider import SignalSummaryProvider

    anthropic_client = _build_anthropic_client()  # raises if key/package missing

    chatbot = Chatbot(
        anthropic_client=anthropic_client,
        summary_provider=SignalSummaryProvider(assembler=assembler),
        assembler=assembler,
        adversarial_filter=AdversarialFilter(),
        output_filter=OutputFilter(),
    )
    logger.info("Chatbot (niveau 1.5 strict) built at startup")
    return chatbot


def build_market_reading_scheduler(assembler: Any) -> Any:
    """Instantiate the hybrid scheduler bound to an assembler's stores."""
    from src.intelligence.scheduler import MarketReadingScheduler

    return MarketReadingScheduler(
        assembler=assembler,
        readings_store=assembler.readings_store,
        candles_store=assembler.candles_store,
        tick_interval_seconds=env_int("SCHEDULER_TICK_INTERVAL_SECONDS", 60),
        auto_stop_hours=env_int("SCHEDULER_AUTO_STOP_HOURS", 24),
    )


def _build_anthropic_client() -> Any:
    """Instantiate the Anthropic SDK client or raise a clear config error.

    Two distinct failure paths, both with explicit messages:
      1. ``ANTHROPIC_API_KEY`` is unset → :class:`BootstrapConfigurationError`.
      2. The ``anthropic`` package isn't installed → same exception.

    Silent fallback to ``None`` (and template-only descriptions) is rejected
    by design — a production boot without the LLM is a misconfiguration, not
    a degraded but acceptable mode. Catch :class:`BootstrapConfigurationError`
    at the call site to choose how to degrade (e.g. 503 on the endpoint).
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise BootstrapConfigurationError(
            "ANTHROPIC_API_KEY env var is required to build the MarketReading "
            "runtime — without it the Haiku description engine cannot run. "
            "Set ANTHROPIC_API_KEY in the environment or disable bootstrap "
            "with BOOTSTRAP_ENABLED=false."
        )
    try:
        import anthropic
    except ImportError as exc:
        raise BootstrapConfigurationError(
            "The `anthropic` package is not installed but is required by the "
            "MarketReading runtime. Install it (`pip install anthropic`) or "
            "disable bootstrap with BOOTSTRAP_ENABLED=false."
        ) from exc

    return anthropic.Anthropic(api_key=api_key)


__all__ = [
    "BootstrapConfigurationError",
    "build_chatbot",
    "build_market_reading_assembler",
    "build_market_reading_scheduler",
    "env_flag",
    "env_int",
    "is_bootstrap_enabled",
]
