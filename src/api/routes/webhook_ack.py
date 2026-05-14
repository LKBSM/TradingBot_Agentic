"""Webhook delivery acknowledgement endpoint — Sprint API-2B.5.

When a broker successfully ingests a webhook payload, they can call
this endpoint to:

- short-circuit our retry loop (queue won't keep attempting delivery),
- emit an audit record so we know they got it.

Idempotent by design: ack-ing a delivery_id that's already been acked,
already delivered, or never existed returns 200 with ``state:
"not_found"``. Brokers can safely retry the ack call if their own
request to us timed out.

Why a POST rather than DELETE
-----------------------------
Semantic clarity. DELETE /webhooks/deliveries/{id} would imply the
broker is rejecting the delivery; POST /ack expresses "received and
processed, you can stop trying". The two states have different audit
records and different downstream behaviour (a future DELETE could
flag the message as malformed for forensics).
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request

from src.api.auth import require_api_key

logger = logging.getLogger(__name__)


router = APIRouter(prefix="/api/v1/webhooks", tags=["webhooks"])


def _get_queue(request: Request):
    queue = getattr(request.app.state.app_state, "webhook_queue", None)
    if queue is None:
        raise HTTPException(
            status_code=503, detail="Webhook queue not configured"
        )
    return queue


def _audit_ack(request: Request, *, delivery_id: str, state: str, subscriber: dict) -> None:
    """Append an ack record to the admin action log when wired."""
    log = getattr(request.app.state.app_state, "admin_action_log", None)
    if log is None:
        return
    try:
        log.record(
            actor=f"key:{subscriber.get('key_id', 'unknown')}",
            action="webhook_ack",
            target=delivery_id,
            payload={"state": state, "tier": subscriber.get("tier", "-")},
            result="ok",
            request_id=str(getattr(request.state, "request_id", "-")),
        )
    except Exception:  # pragma: no cover — audit failure ≠ 500
        logger.exception("admin_action_log.record(webhook_ack) failed")


@router.post(
    "/deliveries/{delivery_id}/ack",
    responses={
        400: {"description": "Malformed delivery_id"},
        503: {"description": "Webhook queue not configured"},
    },
)
async def acknowledge_delivery(
    delivery_id: str,
    request: Request,
    subscriber: dict = Depends(require_api_key),
):
    """Broker confirms receipt — pulls the delivery out of retry rotation."""
    if not delivery_id or len(delivery_id) > 64:
        raise HTTPException(status_code=400, detail="invalid delivery_id")

    queue = _get_queue(request)
    state = queue.cancel(delivery_id)
    _audit_ack(
        request, delivery_id=delivery_id, state=state, subscriber=subscriber
    )
    return {
        "delivery_id": delivery_id,
        "state": state,
        "acknowledged": state in {"pending", "dead"},
    }


@router.get(
    "/deliveries/{delivery_id}",
    responses={
        404: {"description": "No such delivery"},
        503: {"description": "Webhook queue not configured"},
    },
)
async def inspect_delivery(
    delivery_id: str,
    request: Request,
    subscriber: dict = Depends(require_api_key),
):
    """Read the current state of a delivery — useful when the broker
    didn't get a callback and wants to know whether retries are still
    in flight."""
    queue = _get_queue(request)
    delivery = queue.find(delivery_id)
    if delivery is None:
        raise HTTPException(status_code=404, detail="delivery not found")
    return {
        "delivery_id": delivery.delivery_id,
        "url": delivery.url,
        "attempts": delivery.attempts,
        "last_status": delivery.last_status,
        "last_error": delivery.last_error,
        "next_attempt_at": delivery.next_attempt_at,
    }
