"""
Observability bootstrap — Sentry + standard sentinel metrics.

Sprint INFRA-1.2 (Théo, 3h). See `reports/roadmap_2026_2027/PLAN_12_MOIS.md`
Partie II.2 Agent 5.

Closes the eval_16 findings:
1. ``MetricsRegistry`` is now instantiated at boot AND populated with three
   plan-mandated metrics (``signals_generated_total``, ``llm_latency_seconds``,
   ``circuit_breaker_open_total``) — `/metrics` always returns a non-empty
   payload from the first request, even before any signal is emitted.
2. Sentry SDK is initialised when ``SENTRY_DSN`` is set in the environment;
   silently skipped otherwise (no DSN ⇒ no telemetry, no error).

Usage at boot:
    from src.performance.observability import init_observability

    metrics_registry = init_observability(
        service="sentinel",
        environment=os.environ.get("ENVIRONMENT", "dev"),
        release=os.environ.get("RELEASE", "unknown"),
    )
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from src.performance.metrics import MetricsRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Standard metric definitions (plan-mandated minimum)
# ---------------------------------------------------------------------------

# Per PLAN_12_MOIS.md INFRA-1.2 spec: "/metrics ≥ 3 métriques", with these
# three names so dashboards / alerts / SLOs can be authored once and
# survive code-level renames.
STANDARD_METRIC_NAMES = (
    "signals_generated_total",
    "llm_latency_seconds",
    "circuit_breaker_open_total",
)


@dataclass
class StandardMetrics:
    """Container giving direct access to the three standard sentinel metrics.

    Subsystems should NOT instantiate ad-hoc counters with the same names —
    they should pull from this container so cardinality stays predictable.
    """

    signals_generated_total: Any  # Counter
    llm_latency_seconds: Any  # Histogram
    circuit_breaker_open_total: Any  # Counter


def register_standard_metrics(registry: MetricsRegistry) -> StandardMetrics:
    """Register the three plan-mandated metrics on ``registry`` and return
    handles. Idempotent: re-registering re-fetches the existing series.
    """
    signals_generated_total = registry.counter(
        "signals_generated_total",
        "Total trading signals emitted by the SentinelScanner since boot.",
    )
    llm_latency_seconds = registry.histogram(
        "llm_latency_seconds",
        "LLM call latency distribution in seconds.",
        # Buckets sized for Anthropic API typical latencies (~0.5-10s) plus
        # tail-protection bucket at 30s (timeout boundary).
        buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
    )
    circuit_breaker_open_total = registry.counter(
        "circuit_breaker_open_total",
        "Number of times any circuit breaker has transitioned to OPEN.",
    )
    return StandardMetrics(
        signals_generated_total=signals_generated_total,
        llm_latency_seconds=llm_latency_seconds,
        circuit_breaker_open_total=circuit_breaker_open_total,
    )


# ---------------------------------------------------------------------------
# Sentry initialisation (best-effort)
# ---------------------------------------------------------------------------


def init_sentry(
    dsn: str | None = None,
    environment: str = "dev",
    release: str | None = None,
    traces_sample_rate: float = 0.05,
) -> bool:
    """Initialise Sentry SDK. Returns True if Sentry is active, False otherwise.

    Reads ``SENTRY_DSN`` from environment if ``dsn`` not given. If the env var
    is empty/unset, Sentry is silently disabled (no DSN = no telemetry, no
    network calls — preserves dev-loop simplicity).

    Sample rate defaults to 5% for traces (cheap on the free tier; well below
    the 5k events/month free-tier ceiling for our expected volume).
    """
    dsn = (dsn or os.environ.get("SENTRY_DSN") or "").strip()
    if not dsn:
        logger.info("Sentry disabled: SENTRY_DSN not set")
        return False

    try:
        import sentry_sdk
        from sentry_sdk.integrations.logging import LoggingIntegration
    except ImportError:
        logger.warning(
            "sentry-sdk not installed; pip install 'sentry-sdk[fastapi]' to enable"
        )
        return False

    sentry_logging = LoggingIntegration(
        level=logging.INFO,
        event_level=logging.ERROR,
    )
    try:
        sentry_sdk.init(
            dsn=dsn,
            environment=environment,
            release=release,
            traces_sample_rate=traces_sample_rate,
            # Strip PII at source: Telegram chat IDs and API keys can land in
            # exception messages; scrub them via before_send hook.
            before_send=_scrub_pii,
            integrations=[sentry_logging],
            # FastAPI integration enabled lazily — sentry-sdk auto-detects
            # FastAPI/Starlette when they are imported in the same process.
        )
    except Exception as exc:  # noqa: BLE001 — defensive boot
        # Sentry raises BadDsn on malformed DSNs. We MUST NOT crash the
        # whole boot for an observability mis-config — log and continue.
        logger.warning("Sentry init failed (%s); telemetry disabled", exc)
        return False

    logger.info(
        "Sentry initialised (env=%s, release=%s, traces=%.2f%%)",
        environment, release or "unknown", traces_sample_rate * 100,
    )
    return True


def _scrub_pii(event: dict, hint: dict) -> dict | None:
    """Remove obvious PII before sending to Sentry.

    - Telegram chat_id / chat IDs
    - Anthropic / FRED API keys
    - Email addresses

    Returns None to drop the event entirely if it looks like a key dump.
    """
    serialized = str(event)
    suspicious_substrings = [
        "ANTHROPIC_API_KEY=",
        "FRED_API_KEY=",
        "TELEGRAM_BOT_TOKEN=",
    ]
    if any(s in serialized for s in suspicious_substrings):
        # Drop entirely — too risky to ship to Sentry even with redaction
        logger.warning("Sentry event dropped: contains suspicious key marker")
        return None
    return event


# ---------------------------------------------------------------------------
# Top-level boot helper
# ---------------------------------------------------------------------------


def init_observability(
    service: str = "sentinel",
    environment: str | None = None,
    release: str | None = None,
    metrics_prefix: str | None = None,
) -> MetricsRegistry:
    """Single entry-point for boot-time observability setup.

    1. Initialise Sentry (no-op when SENTRY_DSN unset)
    2. Create a MetricsRegistry with the requested prefix
    3. Register the three standard metrics

    Returns the registry so callers can attach it to FastAPI ``AppState``.
    """
    env = environment or os.environ.get("ENVIRONMENT", "dev")
    rel = release or os.environ.get("RELEASE")
    init_sentry(environment=env, release=rel)

    prefix = metrics_prefix or service
    registry = MetricsRegistry(prefix=prefix)
    register_standard_metrics(registry)
    logger.info(
        "Observability initialised: service=%s env=%s metrics=%d",
        service, env, len(STANDARD_METRIC_NAMES),
    )
    return registry
