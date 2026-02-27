"""Admin key-management endpoints — HMAC-protected."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request

from src.api.auth import require_admin
from src.api.models import (
    KeyCreateRequest,
    KeyCreateResponse,
    KeyInfo,
    KeyListResponse,
    KeyRevokeResponse,
    UsageResponse,
)

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
