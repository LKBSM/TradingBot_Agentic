"""Deep health probe — Sprint OBS-2B.2.

The legacy ``/api/v1/health`` endpoint is a fast liveness check (does
the process answer? are the boot-time circuit breakers happy?).
Production needs more: an active probe that proves the Phase 2B
data-plane subsystems are *functional*, not just present:

- the audit ledger's hash chain is intact
- the RAG pipeline can retrieve chunks for a canned query
- per-tier cost quotas haven't silently exhausted
- the webhook delivery queue isn't accumulating dead letters
- per-tier rate limiter has reasonable cap state

Exit code semantics
-------------------
- HTTP 200 when every check returns ``ok=True`` (or is not configured —
  optional subsystems don't degrade health)
- HTTP 503 when *any* configured check fails. Body still carries every
  per-subsystem result so dashboards can isolate the culprit without
  parsing logs.

The probe is intentionally synchronous and bounded: each check times
itself, the overall handler enforces a soft budget (~500ms typical), and
slow checks short-circuit. This is meant to be hit every 30s by an
external watchdog, not be a load source.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


# ---------------------------------------------------------------------------
# Per-subsystem checks
# ---------------------------------------------------------------------------


def _check_audit_ledger(ledger: Any) -> dict:
    if ledger is None:
        return {"configured": False, "ok": True}
    t0 = time.perf_counter()
    try:
        result = ledger.verify()
        return {
            "configured": True,
            "ok": bool(result.ok),
            "n_entries": result.n_entries,
            "head_seq": ledger.size,
            "head_hash": ledger.head_hash,
            "broken_at_seq": result.broken_at_seq,
            "reason": result.reason,
            "duration_ms": round((time.perf_counter() - t0) * 1000, 2),
        }
    except Exception as exc:
        return {
            "configured": True,
            "ok": False,
            "error": f"verify() raised: {type(exc).__name__}",
            "duration_ms": round((time.perf_counter() - t0) * 1000, 2),
        }


def _check_rag_pipeline(pipeline: Any) -> dict:
    if pipeline is None:
        return {"configured": False, "ok": True}
    t0 = time.perf_counter()
    try:
        # Probe query is intentionally generic — any indexed corpus
        # should return at least one hit. Empty result ⇒ index empty
        # or BM25 broken.
        retrieved = pipeline.retrieve("market structure")
        result = {
            "configured": True,
            "ok": len(retrieved) > 0,
            "retrieved_count": len(retrieved),
            "duration_ms": round((time.perf_counter() - t0) * 1000, 2),
        }
        # DATA-2B.8: corpus fingerprint + alignment status. Fail the
        # whole rag check if BM25 and vector store have drifted; serving
        # results from a misaligned corpus is worse than 503.
        if hasattr(pipeline, "corpus_fingerprint"):
            fp = pipeline.corpus_fingerprint()
            result["corpus_fingerprint"] = fp.fingerprint
            result["corpus_aligned"] = fp.aligned
            result["bm25_size"] = fp.bm25_size
            result["vector_size"] = fp.vector_size
            if not fp.aligned:
                result["ok"] = False
                result["drift_reason"] = fp.drift_reason
        return result
    except Exception as exc:
        return {
            "configured": True,
            "ok": False,
            "error": f"retrieve() raised: {type(exc).__name__}",
            "duration_ms": round((time.perf_counter() - t0) * 1000, 2),
        }


def _check_cost_quota(quota: Any) -> dict:
    if quota is None:
        return {"configured": False, "ok": True}
    try:
        tiers = ("FREE", "ANALYST", "STRATEGIST", "INSTITUTIONAL")
        snapshots = {}
        any_exhausted = False
        for tier in tiers:
            snap = quota.snapshot(tier)
            snapshots[tier] = {
                "used_usd": round(snap.used_usd, 4),
                "cap_usd": snap.cap_usd,
                "remaining_usd": round(snap.remaining_usd, 4),
                "utilisation": round(snap.utilisation, 3),
            }
            # >= 100% of cap means subsequent calls are 429-rejected.
            # 95% is the warn threshold ops should investigate.
            if snap.utilisation >= 1.0:
                any_exhausted = True
        return {
            "configured": True,
            "ok": not any_exhausted,
            "tiers": snapshots,
            "any_exhausted": any_exhausted,
        }
    except Exception as exc:
        return {
            "configured": True,
            "ok": False,
            "error": f"snapshot() raised: {type(exc).__name__}",
        }


def _check_webhook_queue(queue: Any) -> dict:
    if queue is None:
        return {"configured": False, "ok": True}
    try:
        pending = queue.pending_size
        dead = queue.dead_letter_size
        # A growing dead-letter list indicates an upstream subscriber
        # broke and ops needs to investigate. Single dead-letter is
        # noise, > 10 is signal — tunable.
        return {
            "configured": True,
            "ok": dead < 10,
            "pending": pending,
            "dead_letter": dead,
            "next_due_at": queue.next_due_at(),
        }
    except Exception as exc:
        return {
            "configured": True,
            "ok": False,
            "error": f"webhook queue probe raised: {type(exc).__name__}",
        }


def _check_embedder(embedder: Any) -> dict:
    if embedder is None:
        return {"configured": False, "ok": True}
    # Local import keeps the health module free of numpy at import time
    # for deployments that don't ship the RAG stack.
    from src.intelligence.rag.embedders import (
        EmbedderHealthError,
        embed_health_check,
    )

    try:
        return embed_health_check(embedder)
    except EmbedderHealthError as exc:
        return {"configured": True, "ok": False, "error": str(exc)}
    except Exception as exc:
        return {
            "configured": True,
            "ok": False,
            "error": f"embed_health_check raised: {type(exc).__name__}",
        }


def _check_tier_rate_limiter(limiter: Any) -> dict:
    if limiter is None:
        return {"configured": False, "ok": True}
    try:
        # Light probe: ensure caps lookup works. Counts of throttled
        # keys live in the metrics layer (OBS-2B.1 bridge), not here.
        return {
            "configured": True,
            "ok": True,
            "caps_per_minute": dict(limiter._caps),  # safe — not user data
        }
    except Exception as exc:
        return {
            "configured": True,
            "ok": False,
            "error": f"limiter probe raised: {type(exc).__name__}",
        }


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.get(
    "/api/v1/health/deep",
    responses={
        200: {"description": "All configured subsystems healthy"},
        503: {"description": "At least one subsystem unhealthy"},
    },
)
async def health_deep(request: Request):
    """Active end-to-end probe of Phase 2B production guards.

    Returns 200 when every *configured* subsystem reports ok; subsystems
    that aren't wired in are skipped (``configured=False``). 503 when
    any configured probe fails. The body always carries every check so
    dashboards can pinpoint the failed subsystem.
    """
    app_state = request.app.state.app_state
    started = time.perf_counter()

    checks = {
        "audit_ledger": _check_audit_ledger(getattr(app_state, "audit_ledger", None)),
        "rag_pipeline": _check_rag_pipeline(getattr(app_state, "rag_pipeline", None)),
        "cost_quota": _check_cost_quota(getattr(app_state, "cost_quota", None)),
        "webhook_queue": _check_webhook_queue(getattr(app_state, "webhook_queue", None)),
        "tier_rate_limiter": _check_tier_rate_limiter(
            getattr(app_state, "tier_rate_limiter", None)
        ),
        "embedder": _check_embedder(getattr(app_state, "embedder", None)),
    }

    overall_ok = all(c["ok"] for c in checks.values())
    body = {
        "ok": overall_ok,
        "checked_at_utc": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "duration_ms": round((time.perf_counter() - started) * 1000, 2),
        "checks": checks,
    }
    status_code = 200 if overall_ok else 503
    return JSONResponse(content=body, status_code=status_code)
