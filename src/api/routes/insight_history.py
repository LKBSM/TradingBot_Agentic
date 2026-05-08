"""Insight history + by-id endpoints — Sprints API-2B.1 + API-2B.3.

Paginated, B2B-facing read view over the audit ledger built in DATA-2B.4.
The audit routes (DATA-2B.5) expose verification + per-seq + by-insight
lookup, but a broker reconciling delivery history needs to *list* (one
of /history) and *replay one* (one of /by-id) — give me the last 50
deliveries since 09:00 UTC, or "fetch the canonical body for insight
abc-123 with proper ETag/304 conditional caching".

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

import json
import logging
import os
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response

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


# ---------------------------------------------------------------------------
# Single-lookup endpoint — Sprint API-2B.3
# ---------------------------------------------------------------------------


def _etag_for(entry_hash: str) -> str:
    """Strong ETag — entry_hash is content-derived and immutable."""
    return f'"{entry_hash}"'


@router.get(
    "/{insight_id}",
    responses={
        304: {"description": "Not modified — entry_hash matches If-None-Match"},
        403: {"description": "Tier insufficient"},
        404: {"description": "Insight not found"},
        503: {"description": "Ledger not configured"},
    },
)
async def get_insight_by_id(
    insight_id: str,
    request: Request,
    if_none_match: Optional[str] = Header(None, alias="If-None-Match"),
    subscriber: dict = Depends(require_api_key),
):
    """Fetch the canonical InsightSignalV2 body that was sealed in the
    ledger under ``insight_id``.

    A broker reconciling a single delivery cares about the body, not the
    list — this is the lookup path. Conditional caching is supported:
    the response carries a strong ``ETag`` derived from the immutable
    ``entry_hash``, so a client storing it can ``If-None-Match: <etag>``
    and we'll return 304 Not Modified instead of re-shipping the body.

    Multi-match policy: an ``insight_id`` may legally appear more than
    once if the broker re-submitted with the same ``client_request_id``.
    We return the *latest* (highest seq) — that's the one currently
    active for delivery. The /audit/by-insight/{id} endpoint exposes
    the full history for forensics.
    """
    _check_tier(subscriber)
    ledger = _get_ledger(request)

    if not insight_id or len(insight_id) > 64:
        raise HTTPException(status_code=400, detail="invalid insight_id")

    entries = ledger.find_by_insight_id(insight_id)
    if not entries:
        raise HTTPException(
            status_code=404, detail=f"no ledger entry for insight {insight_id}"
        )

    # Latest = highest seq. find_by_insight_id is asc by seq.
    entry = entries[-1]
    etag = _etag_for(entry.entry_hash)

    # Conditional GET — strong-ETag match short-circuits with 304.
    if if_none_match and if_none_match.strip() == etag:
        return Response(
            status_code=304,
            headers={
                "ETag": etag,
                "X-Ledger-Seq": str(entry.seq),
            },
        )

    # canonical_json was the exact serialised body that was hashed —
    # parse it back so the response is JSON, not a string.
    try:
        body = json.loads(entry.canonical_json)
    except json.JSONDecodeError as exc:  # pragma: no cover — corrupted ledger
        logger.exception("ledger entry %s has malformed canonical_json", entry.seq)
        raise HTTPException(
            status_code=500, detail="ledger entry corrupted"
        ) from exc

    return JSONResponse(
        content={
            "insight": body,
            "audit": {
                "seq": entry.seq,
                "inserted_at_utc": entry.inserted_at_utc,
                "entry_hash": entry.entry_hash,
                "prev_hash": entry.prev_hash,
            },
            "is_latest_of": len(entries),
        },
        headers={
            "ETag": etag,
            "X-Ledger-Seq": str(entry.seq),
            "Cache-Control": "private, max-age=300",
        },
    )
