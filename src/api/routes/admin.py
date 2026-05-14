"""Admin key-management endpoints — HMAC-protected."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from src.api.auth import require_admin
from src.api.models import (
    KeyCreateRequest,
    KeyCreateResponse,
    KeyInfo,
    KeyListResponse,
    KeyRevokeResponse,
    KeyRotateRequest,
    KeyRotateResponse,
    UsageResponse,
)


class OperationalResumeRequest(BaseModel):
    """Body for POST /admin/operational-resume."""
    operator: str = Field(..., min_length=1, max_length=128)
    ack_phrase: str = Field(..., description='Must equal "I-ACCEPT-RISK"')


class OperationalResumeResponse(BaseModel):
    cleared: bool
    reason: str

router = APIRouter(
    prefix="/api/v1/admin",
    tags=["admin"],
    dependencies=[Depends(require_admin)],
)


@router.post("/keys", response_model=KeyCreateResponse)
async def create_key(body: KeyCreateRequest, request: Request):
    """Create a new subscriber API key."""
    key_store = request.app.state.app_state.key_store
    result = key_store.create_key(body.label)
    return KeyCreateResponse(
        key_id=result["key_id"],
        api_key=result["api_key"],
        label=result["label"],
    )


@router.delete("/keys/{key_id}", response_model=KeyRevokeResponse)
async def revoke_key(key_id: int, request: Request):
    """Revoke a subscriber API key."""
    key_store = request.app.state.app_state.key_store
    revoked = key_store.revoke_key(key_id)
    return KeyRevokeResponse(key_id=key_id, revoked=revoked)


@router.post(
    "/keys/{key_id}/rotate",
    response_model=KeyRotateResponse,
    responses={
        400: {"description": "Already-rotated / revoked / invalid grace"},
        404: {"description": "Key not found"},
    },
)
async def rotate_key(key_id: int, body: KeyRotateRequest, request: Request):
    """Issue a successor key — old one keeps verifying during grace.

    SECURITY-2B.2: a broker on a daily deploy cadence can pre-stage
    the new key value, ship the deploy, then have us automatically
    cut over once the grace window closes. Default grace is 24h.

    Pass ``grace_seconds: 0`` for emergency rotation (immediate cut).
    """
    grace = body.grace_seconds if body.grace_seconds is not None else 86400.0
    key_store = request.app.state.app_state.key_store
    try:
        result = key_store.rotate_key(key_id, grace_seconds=grace)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if result is None:
        raise HTTPException(status_code=404, detail=f"key {key_id} not found")
    # Best-effort admin audit
    log = getattr(request.app.state.app_state, "admin_action_log", None)
    if log is not None:
        try:
            log.record(
                actor="admin",
                action="rotate_key",
                target=str(key_id),
                payload={
                    "new_key_id": result["new_key_id"],
                    "grace_seconds": result["grace_seconds"],
                },
                result="ok",
            )
        except Exception:
            pass
    return KeyRotateResponse(**result)


@router.get("/keys", response_model=KeyListResponse)
async def list_keys(request: Request):
    """List all API keys (metadata only, no hashes)."""
    key_store = request.app.state.app_state.key_store
    keys = key_store.list_keys()
    return KeyListResponse(
        keys=[KeyInfo(**k) for k in keys],
    )


@router.get("/usage", response_model=UsageResponse)
async def get_usage(
    request: Request,
    key_id: int = Query(..., ge=1),
    days: int = Query(30, ge=1, le=365),
):
    """Usage stats for a given key, grouped by endpoint."""
    key_store = request.app.state.app_state.key_store
    stats = key_store.get_usage(key_id, days=days)
    return UsageResponse(key_id=key_id, days=days, usage=stats)


@router.post("/operational-resume", response_model=OperationalResumeResponse)
async def operational_resume(body: OperationalResumeRequest, request: Request):
    """Clear a tripped operational kill-switch.

    Requires the literal ack phrase ``"I-ACCEPT-RISK"`` and an operator
    identity (logged for legal traceability). Refuses to clear when the
    cause is ``BROKER_DISCONNECT`` — only a successful data-feed
    heartbeat does that, by design.
    """
    op_ks = getattr(request.app.state.app_state, "operational_kill_switch", None)
    if op_ks is None:
        raise HTTPException(
            status_code=503,
            detail="Operational kill-switch is not configured.",
        )
    cleared = op_ks.manual_reset(operator=body.operator, ack_phrase=body.ack_phrase)
    if not cleared:
        status = op_ks.status()
        reason = (
            "broker still disconnected — heartbeat must resume first"
            if status.get("reason") == "broker_disconnect"
            else "ack phrase mismatch (must be 'I-ACCEPT-RISK')"
        )
        return OperationalResumeResponse(cleared=False, reason=reason)
    return OperationalResumeResponse(cleared=True, reason="manual override accepted")
