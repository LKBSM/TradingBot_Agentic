"""Insight history endpoint — Sprint API-2B.1.

Paginated, B2B-facing read view over the audit ledger built in DATA-2B.4.
The audit routes (DATA-2B.5) expose verification + per-seq + by-insight
lookup, but a broker reconciling delivery history needs to *list* — give
me the last 50 deliveries since 09:00 UTC, optionally filtered by
``insight_id`` for replay scenarios.

Cursor pagination (seq descending) keeps the contract stable as the
ledger grows: pages are append-only relative to a frozen cursor, so a
client paginating during heavy ingest never sees duplicates or skips.

Tier gate
---------
STRATEGIST+ tier required. The history endpoint is part of the B2B
revenue surface (rec/eval 27), not the consumer experience. TESTING_MODE
short-circuits the gate so dev exercises it without keys.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from src.api.auth import require_api_key

logger = logging.getLogger(__name__)


TESTING_MODE = os.environ.get("SENTINEL_TESTING_MODE", "0") == "1"
ALLOWED_TIERS: frozenset[str] = frozenset({"STRATEGIST", "INSTITUTIONAL"})

router = APIRouter(prefix="/api/v1/insights", tags=["insights"])


def _get_ledger(request: Request):
    ledger = getattr(request.app.state.app_state, "audit_ledger", None)
    if ledger is None:
        raise HTTPException(
            status_code=503, detail="Audit ledger not configured"
        )
    return ledger


def _check_tier(subscriber: dict) -> None:
    if TESTING_MODE or subscriber.get("testing_mode"):
        return
    tier = subscriber.get("tier", "FREE")
    if tier not in ALLOWED_TIERS:
        raise HTTPException(
            status_code=403,
            detail=(
                f"Tier {tier} cannot access /insights/history; "
                f"requires {sorted(ALLOWED_TIERS)}"
            ),
        )


@router.get(
    "/history",
    responses={
        400: {"description": "Invalid pagination parameters"},
        403: {"description": "Tier insufficient"},
        503: {"description": "Ledger not configured"},
    },
)
async def insight_history(
    request: Request,
    cursor: Optional[int] = Query(
        None, ge=1, description="Seq cursor — returns entries with seq < cursor"
    ),
    limit: int = Query(50, ge=1, le=500),
    insight_id: Optional[str] = Query(None, max_length=64),
    since: Optional[str] = Query(
        None,
        max_length=40,
        description="ISO-8601 lower bound on inserted_at_utc, inclusive",
    ),
    until: Optional[str] = Query(
        None,
        max_length=40,
        description="ISO-8601 upper bound on inserted_at_utc, inclusive",
    ),
    subscriber: dict = Depends(require_api_key),
):
    """Newest-first paginated listing of ledger entries.

    Returns entry metadata (seq, insight_id, inserted_at_utc, entry_hash)
    only — call ``/api/v1/audit/entry/{seq}`` for the full canonical body
    + prev_hash. This separation keeps history pages light (~1.5KB for a
    50-row page) so dashboards can poll without bandwidth pain.

    The ``ETag`` response header carries the head_hash, letting clients
    return early when the chain hasn't grown.
    """
    _check_tier(subscriber)
    ledger = _get_ledger(request)

    try:
        entries, next_cursor = ledger.paginate(
            cursor=cursor,
            limit=limit,
            insight_id=insight_id,
            since_iso=since,
            until_iso=until,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    head_seq = ledger.size
    head_hash = ledger.head_hash
    body = {
        "entries": [
            {
                "seq": e.seq,
                "insight_id": e.insight_id,
                "inserted_at_utc": e.inserted_at_utc,
                "entry_hash": e.entry_hash,
            }
            for e in entries
        ],
        "next_cursor": next_cursor,
        "has_more": next_cursor is not None,
        "head_seq": head_seq,
        "head_hash": head_hash,
        "limit": limit,
        "filters": {
            "insight_id": insight_id,
            "since": since,
            "until": until,
        },
    }
    return JSONResponse(
        content=body,
        headers={
            "ETag": f'W/"{head_hash[:16]}-{head_seq}"',
            "X-Ledger-Head-Seq": str(head_seq),
        },
    )
