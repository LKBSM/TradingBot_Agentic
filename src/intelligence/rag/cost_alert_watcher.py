"""DG-052 — LLM cost monitoring + threshold alerts.

The existing ``CostTracker`` already accumulates per-call USD spend, and
``metrics_bridge`` exposes ``rag_cost_usd_total{kind="llm"}`` as a
Prometheus gauge. What was missing for Sprint 4 is the **alert leg** —
a daemon that watches the gauge and triggers a notification once the
daily LLM spend crosses a configurable threshold.

Design
------
This module is intentionally tiny and dependency-free. It exposes one
class — ``CostAlertWatcher`` — which:

1. takes a callable ``cost_provider()`` that returns the current
   running USD total (we call ``CostTracker.summary().llm_usd`` in
   prod, a lambda in tests),
2. takes a ``threshold_usd`` (read from ``LLM_DAILY_COST_ALERT_USD``
   by the caller),
3. takes a ``notify(message: str)`` callable. Default = ``logger.warning``,
4. exposes ``check()`` which the caller polls (or runs on a background
   timer / each scheduler tick).

State machine
-------------
- ``below``: nothing happened.
- Crossing ``threshold_usd`` flips to ``alerting`` and fires ``notify``
  exactly once.
- A ``cooldown_seconds`` window prevents re-fire even if the gauge
  keeps drifting up. After cooldown, the next observed crossing fires
  again. This avoids paging-loop on a wedged budget.
- ``reset()`` returns to ``below``. Use at UTC-midnight rollover or
  on explicit operator intervention.

The default cooldown is 1 hour — enough to avoid notification floods
when an LLM-tight-loop is the root cause and operators are already
paged; short enough that a *new* incident an hour later still fires.

Notifier helpers
----------------
``DiscordWebhookNotifier`` and ``LoggingNotifier`` are tiny adapters
matching the ``notify(message: str) -> None`` signature. Add SMTP /
Slack / PagerDuty by following the same pattern — no changes to the
watcher.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.request
from typing import Callable, Optional, Protocol


logger = logging.getLogger(__name__)


DEFAULT_THRESHOLD_USD = 5.0      # Conservative — matches kill-criteria green/red boundary
DEFAULT_COOLDOWN_SECONDS = 3600  # one hour


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


class _CostProvider(Protocol):
    """Anything that returns the current cumulative USD spend."""
    def __call__(self) -> float: ...


# ---------------------------------------------------------------------------
# Watcher
# ---------------------------------------------------------------------------


class CostAlertWatcher:
    """Polls a cost gauge and fires a notification on threshold crossing."""

    def __init__(
        self,
        *,
        cost_provider: _CostProvider,
        threshold_usd: float = DEFAULT_THRESHOLD_USD,
        notify: Optional[Callable[[str], None]] = None,
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
        clock: Optional[Callable[[], float]] = None,
        label: str = "LLM",
    ):
        if threshold_usd <= 0:
            raise ValueError("threshold_usd must be > 0")
        if cooldown_seconds < 0:
            raise ValueError("cooldown_seconds must be ≥ 0")
        self._cost_provider = cost_provider
        self._threshold = float(threshold_usd)
        self._notify = notify or (lambda msg: logger.warning(msg))
        self._cooldown = float(cooldown_seconds)
        self._clock = clock or time.time
        self._label = label
        self._lock = threading.Lock()
        self._state: str = "below"   # "below" | "alerting"
        self._last_alert_ts: float = 0.0
        self._alerts_fired: int = 0

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @property
    def alerts_fired(self) -> int:
        with self._lock:
            return self._alerts_fired

    def check(self) -> bool:
        """Read the cost provider and fire if we cross the threshold.

        Returns True iff a notification was fired during this call.
        """
        current = float(self._cost_provider())
        now = self._clock()
        with self._lock:
            if current < self._threshold:
                # Below threshold — drop back to "below" so the next
                # crossing will fire again. Do NOT clear the cooldown.
                self._state = "below"
                return False
            # current >= threshold
            if self._state == "alerting" and (now - self._last_alert_ts) < self._cooldown:
                return False
            self._state = "alerting"
            self._last_alert_ts = now
            self._alerts_fired += 1
        # Outside the lock so the (potentially blocking) notifier
        # doesn't hold the watcher state.
        message = (
            f"[cost-alert] {self._label} daily spend ${current:.4f} "
            f"crossed threshold ${self._threshold:.4f}."
        )
        try:
            self._notify(message)
        except Exception as exc:  # noqa: BLE001 — never let notifier crash the watcher
            logger.error("CostAlertWatcher notifier failed: %s", exc)
        return True

    def reset(self) -> None:
        """Drop the alerting state — call at UTC-midnight rollover."""
        with self._lock:
            self._state = "below"
            self._last_alert_ts = 0.0


# ---------------------------------------------------------------------------
# Notifier adapters
# ---------------------------------------------------------------------------


class LoggingNotifier:
    """Writes the alert to the standard logger."""
    def __init__(self, level: int = logging.WARNING):
        self._level = level

    def __call__(self, message: str) -> None:
        logger.log(self._level, message)


class DiscordWebhookNotifier:
    """POSTs the alert to a Discord webhook URL.

    Synchronous POST with a 5-second timeout. The watcher catches
    notifier exceptions so a Discord outage will not wedge the watcher.
    """

    def __init__(self, webhook_url: str, *, username: str = "MIA-Markets-Cost-Alert",
                 timeout_seconds: float = 5.0):
        if not webhook_url:
            raise ValueError("webhook_url is required")
        self._url = webhook_url
        self._username = username
        self._timeout = float(timeout_seconds)

    def __call__(self, message: str) -> None:
        payload = json.dumps({
            "content": message,
            "username": self._username,
        }).encode("utf-8")
        req = urllib.request.Request(
            self._url, data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                if resp.status >= 400:
                    logger.warning(
                        "Discord webhook responded HTTP %s", resp.status,
                    )
        except urllib.error.URLError as exc:
            logger.warning("Discord webhook POST failed: %s", exc)


__all__ = [
    "CostAlertWatcher",
    "DiscordWebhookNotifier",
    "LoggingNotifier",
    "DEFAULT_THRESHOLD_USD",
    "DEFAULT_COOLDOWN_SECONDS",
]
