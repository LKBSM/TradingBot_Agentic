"""The 14-day annual guarantee — LEG-1.

The terms say, word for word:

    « À l'annuel, nous offrons une garantie de 14 jours à compter du paiement. »

This module is the rule behind that sentence, kept PURE so it can be tested
without Stripe, without a database and without a clock: given the plan, the
payment date and "now", is the customer entitled to their money back?

Two deliberate boundaries:

* **Annual only.** The monthly cadence is not covered — the terms give monthly
  customers access until the end of the period they paid for instead, which is
  why refusing here must point them at cancellation, not at a dead end.
* **From the PAYMENT.** Not from sign-up, not from the period start. A renewal
  payment opens a fresh 14 days, which is the honest reading of the sentence and
  the one that favours the customer.

Quebec's Consumer Protection Act prevails over all of this (terms §8). A refusal
returned by this module is never the last word — it is our commercial guarantee
declining, not a statement about the customer's legal rights.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from src.billing.pricing import PLAN_ANNUAL

#: Days of the commercial guarantee, counted from the payment.
GUARANTEE_DAYS = 14

#: Seconds in the guarantee window.
GUARANTEE_SECONDS = GUARANTEE_DAYS * 24 * 60 * 60


@dataclass(frozen=True)
class GuaranteeDecision:
    """Whether a refund is owed, and — when it is not — why."""

    eligible: bool
    #: Machine-readable reason when ``eligible`` is False.
    reason: Optional[str] = None
    #: Seconds left in the window (0 once elapsed). Informative, never negative.
    seconds_remaining: int = 0

    @property
    def days_remaining(self) -> int:
        """Whole days left, rounded UP: 1 second left is still 'today'."""
        if self.seconds_remaining <= 0:
            return 0
        return -(-self.seconds_remaining // (24 * 60 * 60))


#: Reasons a refund is declined. Each maps to a message in the API layer.
REASON_NO_SUBSCRIPTION = "no_subscription"
REASON_NOT_ANNUAL = "not_annual"
REASON_NO_PAYMENT = "no_payment_found"
REASON_WINDOW_ELAPSED = "window_elapsed"
REASON_ALREADY_REFUNDED = "already_refunded"


def evaluate(
    *,
    plan_key: Optional[str],
    paid_at: Optional[float],
    now: Optional[float] = None,
    already_refunded: bool = False,
) -> GuaranteeDecision:
    """Decide whether the 14-day guarantee covers this payment.

    ``plan_key`` is the resolved cadence (``"ANNUAL"`` / ``"MONTHLY"``),
    ``paid_at`` the Unix timestamp Stripe marked the invoice paid.
    """
    if already_refunded:
        return GuaranteeDecision(False, REASON_ALREADY_REFUNDED)
    if not plan_key:
        return GuaranteeDecision(False, REASON_NO_SUBSCRIPTION)
    if plan_key.upper() != PLAN_ANNUAL:
        return GuaranteeDecision(False, REASON_NOT_ANNUAL)
    if not paid_at:
        return GuaranteeDecision(False, REASON_NO_PAYMENT)

    moment = time.time() if now is None else now
    elapsed = moment - float(paid_at)
    # A payment dated in the future (clock skew) is treated as "just now" rather
    # than as an error: skew must never cost a customer their guarantee.
    remaining = GUARANTEE_SECONDS - max(0.0, elapsed)
    if remaining <= 0:
        return GuaranteeDecision(False, REASON_WINDOW_ELAPSED, 0)
    return GuaranteeDecision(True, None, int(remaining))
