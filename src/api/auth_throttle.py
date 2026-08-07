"""In-process sliding-window throttle for the authentication endpoints.

AUTH-HARDENING — a first-layer brute-force / abuse brake on login, registration
and password-reset requests. The audit found these endpoints unthrottled (a
20-attempt burst passed unimpeded and the per-IP limiter in ``create_app`` is a
no-op in the ``asgi:app`` entrypoint, where ``rate_limiter is None``).

Design:
  · Keyed per client IP; login additionally per identifier, so a distributed
    attack on ONE account is slowed even from many IPs.
  · Login counts only FAILURES and is cleared on a successful auth, so a legit
    user typing a wrong password once never gets locked and the happy path is
    untouched. Registration / reset count every attempt (they have no "success
    clears" notion and are the enumeration / spam vectors).
  · Fixed-cap sliding window: once the cap is reached the window must drain
    (``window_s``) before new attempts are accepted — a refused attempt does NOT
    extend the window.

Scope / honesty: in-memory and PER-PROCESS. A multi-worker or multi-instance
deployment throttles per worker, so treat this as a brake, not a hard global
quota; a shared store (Redis) is the natural next layer. Limits are far above a
human's pace, so real users never see it. Disabled when ``max_attempts <= 0``.
"""

from __future__ import annotations

import os
import threading
import time
from collections import defaultdict, deque
from typing import Callable, Deque, Dict, Tuple

# Defaults chosen well above any human cadence but far below a brute-force run.
_DEFAULT_MAX_ATTEMPTS = 10
_DEFAULT_WINDOW_S = 300  # 5 minutes
# Bound the key map so a spray of unique IPs cannot grow memory without limit.
_MAX_TRACKED_KEYS = 50_000


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


class AuthThrottle:
    """Sliding-window attempt counter, thread-safe, per-process.

    ``max_attempts <= 0`` disables the throttle entirely (every call is allowed) —
    an explicit escape hatch for environments that front the app with their own
    limiter, and the default for constructions that pass 0.
    """

    def __init__(
        self,
        max_attempts: int | None = None,
        window_s: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._max = (
            max_attempts
            if max_attempts is not None
            else _int_env("AUTH_THROTTLE_MAX_ATTEMPTS", _DEFAULT_MAX_ATTEMPTS)
        )
        self._window = float(
            window_s
            if window_s is not None
            else _int_env("AUTH_THROTTLE_WINDOW_S", _DEFAULT_WINDOW_S)
        )
        self._clock = clock
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self._max > 0

    def _evict(self, dq: Deque[float], now: float) -> None:
        cutoff = now - self._window
        while dq and dq[0] <= cutoff:
            dq.popleft()

    def _retry_after(self, dq: Deque[float], now: float) -> int:
        # Time until the oldest in-window hit ages out and a slot frees up.
        if not dq:
            return 1
        return max(1, int(dq[0] + self._window - now) + 1)

    def blocked(self, key: str) -> Tuple[bool, int]:
        """Read-only: is ``key`` already at the cap? Returns (blocked, retry_after_s)."""
        if not self.enabled:
            return False, 0
        now = self._clock()
        with self._lock:
            dq = self._hits.get(key)
            if dq is None:
                return False, 0
            self._evict(dq, now)
            if len(dq) >= self._max:
                return True, self._retry_after(dq, now)
            return False, 0

    def hit(self, key: str) -> Tuple[bool, int]:
        """Record one attempt. Returns (allowed, retry_after_s).

        ``allowed=False`` when the window is already full — the attempt is refused
        and NOT recorded (so a hammering client cannot keep extending its own
        window)."""
        if not self.enabled:
            return True, 0
        now = self._clock()
        with self._lock:
            dq = self._hits[key]
            self._evict(dq, now)
            if len(dq) >= self._max:
                return False, self._retry_after(dq, now)
            dq.append(now)
            self._gc_locked()
            return True, 0

    def reset(self, key: str) -> None:
        """Clear a key's history (e.g. a successful login rewards the actor)."""
        with self._lock:
            self._hits.pop(key, None)

    def _gc_locked(self) -> None:
        # Opportunistic cleanup of drained keys; only runs when the map is large.
        if len(self._hits) <= _MAX_TRACKED_KEYS:
            return
        empty = [k for k, v in self._hits.items() if not v]
        for k in empty:
            self._hits.pop(k, None)


def client_ip(request) -> str:
    """The client IP as the app already derives it elsewhere (per-IP limiter /
    access log): ``request.client.host``. Behind a proxy this is the proxy hop
    unless uvicorn runs with ``--proxy-headers`` (standard on the render deploy),
    in which case Starlette resolves the real client. Never trusts a raw
    ``X-Forwarded-For`` header (spoofable)."""
    return request.client.host if getattr(request, "client", None) else "unknown"
