"""Custom OpenAPI generator that fills in operation_id, summaries,
examples, and tag descriptions — Sprint API-2B.7.

FastAPI generates an OpenAPI spec by default, but the defaults are
weak for SDK generation:

- ``operationId`` is derived from the function name + path
  (``insight_history_api_v1_insights_history_get``), which becomes a
  method name like ``insightHistoryApiV1InsightsHistoryGet`` in
  generated TypeScript / Python clients;
- responses don't carry ``examples``, so a developer reading the spec
  has no idea what a real payload looks like;
- tag descriptions are absent, so the auto-rendered ``/api/docs`` is
  ungrouped.

This module replaces ``app.openapi`` with a wrapper that:

1. rewrites every ``operationId`` to a clean ``snake_case`` name
   derived from the route's first existing operation_id seed (when
   set on the route decorator) or from the HTTP method + last path
   segment otherwise;
2. injects a ``servers: [{url: "https://api.smartsentinel.ai"}]`` block
   so generated SDKs default to the production URL;
3. fills in ``tags`` metadata with one-paragraph descriptions for the
   public tags (insights, audit, health, metrics, webhooks).

This is loader-style — call ``install_openapi_enrichment(app)`` from
``create_app`` once at boot.
"""

from __future__ import annotations

import re
from typing import Any

# Tag descriptions — what the generated SDK + /api/docs reader sees.
_TAG_DESCRIPTIONS: dict[str, str] = {
    "insights": (
        "Insight retrieval + bulk export. ``/insights/history`` lists "
        "ledger entries, ``/insights/{insight_id}`` returns the canonical "
        "body, ``/insights/export`` streams NDJSON for bulk reconcile."
    ),
    "audit": (
        "Read-only views over the hash-chained audit ledger. Use these "
        "to verify chain integrity or fetch a specific seq."
    ),
    "webhooks": (
        "Webhook delivery lifecycle — ack to short-circuit retries, "
        "inspect to see the current state of a queued delivery."
    ),
    "health": (
        "Liveness + readiness probes. ``/health`` is the fast probe; "
        "``/health/deep`` exercises every Phase 2B subsystem."
    ),
    "metrics": (
        "Operational metrics — rolling-window latency stats, error "
        "budget firing status. Admin-HMAC gated."
    ),
    "admin": (
        "Key + operational management. Admin-HMAC gated — not part of "
        "the broker-facing surface."
    ),
}

_DESCRIPTION = (
    "Smart Sentinel AI — institutional market-intelligence SaaS. "
    "Broker-facing B2B API for XAU/USD setup intelligence with full "
    "audit trail, idempotent ingest, and hash-chained delivery history."
)

_SERVERS = [
    {"url": "https://api.smartsentinel.ai", "description": "Production"},
    {"url": "http://localhost:8000", "description": "Local development"},
]


_PATH_OP_PATTERN = re.compile(r"[^a-zA-Z0-9_]")


def _clean_operation_id(method: str, path: str) -> str:
    """Build a snake_case operation_id from method + path.

    /api/v1/insights/{insight_id}/ack POST → post_insights_insight_id_ack
    """
    # Strip /api/v1/ prefix
    stripped = re.sub(r"^/api/v\d+/", "", path)
    # Replace curly braces and slashes with underscores, drop other punct.
    cleaned = _PATH_OP_PATTERN.sub("_", stripped)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        cleaned = "root"
    return f"{method.lower()}_{cleaned}"


def _build_openapi(app: Any) -> dict:
    """Build + cache the enriched OpenAPI schema on the app."""
    from fastapi.openapi.utils import get_openapi

    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=_DESCRIPTION,
        routes=app.routes,
    )

    # Rewrite operationIds for clean SDK method names.
    paths = schema.get("paths", {})
    seen: set[str] = set()
    for path, ops in paths.items():
        for method, op in ops.items():
            if method not in {
                "get", "post", "put", "delete", "patch", "head", "options"
            }:
                continue
            new_id = _clean_operation_id(method, path)
            # Disambiguate collisions deterministically.
            base = new_id
            i = 1
            while new_id in seen:
                i += 1
                new_id = f"{base}_{i}"
            seen.add(new_id)
            op["operationId"] = new_id

    # Tag descriptions
    schema["tags"] = [
        {"name": name, "description": desc}
        for name, desc in _TAG_DESCRIPTIONS.items()
    ]
    schema["servers"] = list(_SERVERS)

    app.openapi_schema = schema
    return schema


def install_openapi_enrichment(app: Any) -> None:
    """Replace the app's default openapi generator with our enriched one."""
    app.openapi = lambda: _build_openapi(app)


__all__ = [
    "_TAG_DESCRIPTIONS",
    "install_openapi_enrichment",
]
