"""Admin audit log query endpoint — Sprint SECURITY-2B.1.

Read-only view over the AdminActionLog. Admin-HMAC gated (via the same
``require_admin`` dependency the existing /admin/* routes use), so an
internal dashboard / forensics console can pull "who did what when".

Why a query endpoint at all
---------------------------
Two reasons the SQLite file isn't enough:

1. ops doesn't always have shell access to the production volume; a
   tooled dashboard fetches over the API,
2. the query layer adds the same defensive limits and filters as
   /audit/* (max 1000 rows, ISO-8601 since filter, action/actor
   filters), so an inquisitive operator can't accidentally page-fault
   the SQLite reader.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from src.api.auth import require_admin

logger = logging.getLogger(__name__)


router = APIRouter(prefix="/api/v1/admin", tags=["admin"])


def _get_log(request: Request):
    log = getattr(request.app.state.app_state, "admin_action_log", None)
    if log is None:
        raise HTTPException(
            status_code=503, detail="Admin action log not configured"
        )
    return log


@router.get(
    "/audit-log",
    responses={
        503: {"description": "Admin action log not configured"},
    },
)
async def list_admin_actions(
    request: Request,
    actor: Optional[str] = Query(None, max_length=64),
    action: Optional[str] = Query(None, max_length=64),
    since: Optional[str] = Query(
        None, max_length=40, description="ISO-8601 lower bound on ts_utc"
    ),
    limit: int = Query(100, ge=1, le=1000),
    _: bool = Depends(require_admin),
):
    """Most-recent-first listing of admin actions.

    Filters compose with AND. ``since`` is an inclusive ISO-8601 lower
    bound on ``ts_utc``.
    """
    log = _get_log(request)
    try:
        records = log.query(
            actor=actor, action=action, since_iso=since, limit=limit
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return {
        "count": len(records),
        "records": [
            {
                "id": r.id,
                "ts_utc": r.ts_utc,
                "actor": r.actor,
                "action": r.action,
                "target": r.target,
                "payload_digest": r.payload_digest,
                "result": r.result,
                "request_id": r.request_id,
            }
            for r in records
        ],
        "filters": {"actor": actor, "action": action, "since": since},
        "limit": limit,
    }
