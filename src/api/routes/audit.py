"""Audit-trail public endpoints — Sprint DATA-2B.5.

Two B2B-facing routes that let a broker (or auditor) independently
verify the hash chain produced by DATA-2B.4:

- ``GET /api/v1/audit/verify`` — walks the chain end-to-end and returns
  ``{ok, n_entries, head_hash, broken_at_seq, reason}``.
- ``GET /api/v1/audit/entry/{seq}`` — returns one entry's full record
  including its canonical JSON, prev_hash, and entry_hash. The broker
  can recompute ``sha256(seq | ts | canonical_json | prev_hash)``
  themselves and compare.

The chain itself is constructed elsewhere (DATA-2B.4); these routes
are read-only.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from src.api.auth import require_api_key

logger = logging.getLogger(__name__)


router = APIRouter(prefix="/api/v1/audit", tags=["audit"])


def _get_ledger(request: Request):
    ledger = getattr(request.app.state.app_state, "audit_ledger", None)
    if ledger is None:
        raise HTTPException(
            status_code=503, detail="Audit ledger not configured"
        )
    return ledger


@router.get(
    "/verify",
    responses={503: {"description": "Ledger not configured"}},
)
async def verify_chain(
    request: Request,
    subscriber: dict = Depends(require_api_key),
):
    ledger = _get_ledger(request)
    result = ledger.verify()
    return {
        "ok": result.ok,
        "n_entries": result.n_entries,
        "head_hash": ledger.head_hash,
        "broken_at_seq": result.broken_at_seq,
        "reason": result.reason,
    }


@router.get(
    "/entry/{seq}",
    responses={
        404: {"description": "Sequence not found"},
        503: {"description": "Ledger not configured"},
    },
)
async def get_entry(
    seq: int,
    request: Request,
    subscriber: dict = Depends(require_api_key),
):
    if seq < 1:
        raise HTTPException(status_code=400, detail="seq must be >= 1")
    ledger = _get_ledger(request)
    entry = ledger.get(seq)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"seq {seq} not found")
    return {
        "seq": entry.seq,
        "inserted_at_utc": entry.inserted_at_utc,
        "insight_id": entry.insight_id,
        "canonical_json": entry.canonical_json,
        "prev_hash": entry.prev_hash,
        "entry_hash": entry.entry_hash,
    }


@router.get(
    "/by-insight/{insight_id}",
    responses={
        404: {"description": "Insight ID not found in ledger"},
        503: {"description": "Ledger not configured"},
    },
)
async def find_by_insight_id(
    insight_id: str,
    request: Request,
    subscriber: dict = Depends(require_api_key),
):
    """Lookup all ledger entries that carried a given insight_id.

    Most insights appear at most once. A broker that re-submits the
    same client_request_id will see multiple matches — useful for
    deduplication audits.
    """
    if not insight_id or len(insight_id) > 64:
        raise HTTPException(status_code=400, detail="invalid insight_id")
    ledger = _get_ledger(request)
    entries = ledger.find_by_insight_id(insight_id)
    if not entries:
        raise HTTPException(
            status_code=404, detail=f"no ledger entry for insight {insight_id}"
        )
    return {
        "insight_id": insight_id,
        "n_entries": len(entries),
        "entries": [
            {
                "seq": e.seq,
                "inserted_at_utc": e.inserted_at_utc,
                "entry_hash": e.entry_hash,
            }
            for e in entries
        ],
    }
