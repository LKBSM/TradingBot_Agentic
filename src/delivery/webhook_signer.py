"""HMAC-SHA256 webhook signer + verifier — Sprint INFRA-2B.6.

Why a signing scheme
--------------------
B2B webhook subscribers (brokers, prop desks) need cryptographic proof
that a payload they received originated from us and was not tampered
with in flight. Stripe's pattern (HMAC-SHA256 of timestamp.body) is the
industry standard:

- A shared secret per subscriber, generated server-side at register
  time. Delivered once via secure channel, never persisted server-side
  in plaintext (we'd hash + salt for storage in production).
- Signature header carries ``t=<unix_ts>,v1=<hmac_hex>`` so the
  signing algorithm is self-describing. ``v1`` is a version tag we'll
  bump if we ever rotate the algorithm.
- Verifier MUST check the timestamp tolerance to reject replay attacks
  (default 5 minutes), and MUST use ``hmac.compare_digest`` for the
  hex comparison so a timing oracle can't leak the secret.

Scope of this module
--------------------
Pure crypto + parsing helpers. The actual HTTP delivery worker (queue,
retry, dead-letter) sits on top in a follow-up sprint; here we ship the
primitive that makes any of that secure.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from dataclasses import dataclass
from typing import Optional


SIGNATURE_HEADER = "X-Sentinel-Signature"
SIGNATURE_VERSION = "v1"
DEFAULT_TOLERANCE_SECONDS = 300  # 5 minutes


# ---------------------------------------------------------------------------
# Sign
# ---------------------------------------------------------------------------


@dataclass
class SignedPayload:
    """Output of :func:`sign_payload` — header value + raw signature."""

    header_value: str  # ready to drop into X-Sentinel-Signature
    timestamp: int
    signature_hex: str


def _hmac_hex(secret: bytes, signed_string: bytes) -> str:
    return hmac.new(secret, signed_string, hashlib.sha256).hexdigest()


def sign_payload(
    body: str,
    secret: str,
    *,
    timestamp: Optional[int] = None,
) -> SignedPayload:
    """Sign a webhook body and return the header value to send.

    The signed string is ``f"{timestamp}.{body}"``. ``timestamp`` is
    auto-set to the current Unix time when not provided.

    The header value is ``"t=<ts>,v1=<sig>"``. Subscribers parse it
    via :func:`parse_signature_header`.
    """
    if timestamp is None:
        timestamp = int(time.time())
    signed_string = f"{timestamp}.{body}".encode("utf-8")
    sig = _hmac_hex(secret.encode("utf-8"), signed_string)
    header_value = f"t={timestamp},{SIGNATURE_VERSION}={sig}"
    return SignedPayload(
        header_value=header_value,
        timestamp=timestamp,
        signature_hex=sig,
    )


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------


@dataclass
class VerificationResult:
    ok: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.ok


def parse_signature_header(header_value: str) -> tuple[Optional[int], Optional[str]]:
    """Parse ``"t=...,v1=..."`` into ``(timestamp, signature_hex)``.

    Tolerates extra version tags (e.g. ``v0=…,v1=…``) — picks the
    highest-numbered ``v`` the parser knows about. Unknown tag layout
    returns ``(None, None)``.
    """
    timestamp: Optional[int] = None
    signature: Optional[str] = None
    for raw in header_value.split(","):
        kv = raw.strip().split("=", 1)
        if len(kv) != 2:
            continue
        k, v = kv[0].strip().lower(), kv[1].strip()
        if k == "t":
            try:
                timestamp = int(v)
            except ValueError:
                return None, None
        elif k == SIGNATURE_VERSION:
            signature = v
    return timestamp, signature


def verify_payload(
    body: str,
    header_value: str,
    secret: str,
    *,
    tolerance_seconds: int = DEFAULT_TOLERANCE_SECONDS,
    now: Optional[int] = None,
) -> VerificationResult:
    """Verify a webhook signature.

    Checks (in order):
    1. Header parses and contains both ``t`` and ``v1`` fields.
    2. Timestamp is within ``tolerance_seconds`` of ``now`` (replay guard).
    3. Recomputed HMAC matches via ``hmac.compare_digest``.

    ``tolerance_seconds`` defaults to 5 minutes — long enough to absorb
    clock skew and queue backlog, tight enough that captured signatures
    stop working before a sophisticated replay attack can exploit them.
    """
    timestamp, signature = parse_signature_header(header_value)
    if timestamp is None or signature is None:
        return VerificationResult(False, "header missing timestamp or v1 signature")

    now_ts = now if now is not None else int(time.time())
    if abs(now_ts - timestamp) > tolerance_seconds:
        return VerificationResult(
            False,
            f"timestamp outside tolerance "
            f"({abs(now_ts - timestamp)}s > {tolerance_seconds}s)",
        )

    signed_string = f"{timestamp}.{body}".encode("utf-8")
    expected = _hmac_hex(secret.encode("utf-8"), signed_string)
    if not hmac.compare_digest(expected, signature):
        return VerificationResult(False, "signature mismatch")

    return VerificationResult(True, "")


# ---------------------------------------------------------------------------
# Secret generation
# ---------------------------------------------------------------------------


def generate_webhook_secret(n_bytes: int = 32) -> str:
    """Generate a CSPRNG-backed shared secret.

    Returns a URL-safe token of ``n_bytes`` random bytes (base64-encoded).
    32 bytes ≈ 256 bits of entropy, ample headroom over the SHA-256
    HMAC's effective security.
    """
    if n_bytes < 16:
        raise ValueError("n_bytes must be >= 16 for adequate entropy")
    return secrets.token_urlsafe(n_bytes)


__all__ = [
    "DEFAULT_TOLERANCE_SECONDS",
    "SIGNATURE_HEADER",
    "SIGNATURE_VERSION",
    "SignedPayload",
    "VerificationResult",
    "generate_webhook_secret",
    "parse_signature_header",
    "sign_payload",
    "verify_payload",
]
