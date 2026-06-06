"""Chantier 4 — Couche 1 — adversarial input filter (doc §4.1).

The user message is screened against the 4 adversarial buckets BEFORE any LLM
call. A match means the question is risky (jailbreak / trade request / persona
hijack / personalised financial advice) and the chatbot answers directly with a
pedagogical refusal template — Haiku is never invoked for that turn.

Order matters: buckets are checked in the order declared in
``ADVERSARIAL_PATTERNS_BY_CATEGORY`` (jailbreak first, as the most
security-critical), and the FIRST matching bucket wins so the reported category
is deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.intelligence.chatbot.constants import (
    ADVERSARIAL_PATTERNS_BY_CATEGORY,
    normalize_text,
)


@dataclass(frozen=True)
class AdversarialCheckResult:
    """Outcome of screening a single user message.

    Attributes:
        triggered: True when a risky pattern matched (block before LLM).
        category: the winning bucket name (``jailbreak`` / ``trade_request`` /
            ``persona_hijack`` / ``financial_advice``) or None when clean.
        matched_pattern: the ``re.Pattern.pattern`` source string that matched,
            or None when clean. Kept for logging / auditability.
    """

    triggered: bool
    category: Optional[str] = None
    matched_pattern: Optional[str] = None


class AdversarialFilter:
    """Stateless screener for the 4 adversarial-pattern buckets (Couche 1)."""

    def __init__(self) -> None:
        # Snapshot the ordered mapping at construction so the check is a pure
        # function of the input. ``ADVERSARIAL_PATTERNS_BY_CATEGORY`` is already
        # ordered (jailbreak first) — dict preserves insertion order.
        self._buckets = ADVERSARIAL_PATTERNS_BY_CATEGORY

    def check(self, user_message: str) -> AdversarialCheckResult:
        """Return the first matching bucket, or a clean result.

        Empty / whitespace-only messages are treated as clean (nothing to
        block — downstream validation handles emptiness).
        """
        if not user_message or not user_message.strip():
            return AdversarialCheckResult(triggered=False)

        normalized = normalize_text(user_message)

        for category, patterns in self._buckets.items():
            for pattern in patterns:
                if pattern.search(normalized):
                    return AdversarialCheckResult(
                        triggered=True,
                        category=category,
                        matched_pattern=pattern.pattern,
                    )

        return AdversarialCheckResult(triggered=False)


__all__ = ["AdversarialCheckResult", "AdversarialFilter"]
