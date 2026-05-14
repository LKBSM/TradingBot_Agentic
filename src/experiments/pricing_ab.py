"""Pricing A/B experiment framework — Sprint COMM-2B.5.

In-process sticky-bucket assignment + outcome tracking for pricing
A/B tests. Two experiments are pre-defined; the framework supports
adding arbitrary additional ones.

The two pre-defined experiments target the assumptions called out in
eval_27 (Pricing): trial length and annual bundling.

  EXP_TRIAL_LENGTH        7-day vs 14-day trial on LITE.
  EXP_ANNUAL_BUNDLE       PRO+ monthly vs annual -20% .

Sticky bucket
-------------
Per-user assignment is deterministic on (experiment_id, user_id) —
``hashlib.sha256`` over the concatenation, mod the number of arms.
A user always sees the same arm across sessions; clearing cookies
doesn't re-randomise. This avoids confounding by "users who happen to
re-open during the test window".

Outcome tracking
----------------
``record_outcome(user_id, experiment_id, outcome, value=)`` appends to
an in-memory log + optional persistent backend. The default backend
is in-memory; production wires SQLite via the SignalStore pattern.

Stats
-----
``summarise(experiment_id)`` returns per-arm n + conversion rate (or
mean value), with a chi-square / t-test against the control arm.
"""

from __future__ import annotations

import hashlib
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class ExperimentArm:
    key: str
    description: str
    payload: dict  # arbitrary config (e.g. {"trial_days": 7})


@dataclass(frozen=True)
class Experiment:
    experiment_id: str
    description: str
    arms: tuple[ExperimentArm, ...]
    is_active: bool = True

    def assign(self, user_id: str) -> ExperimentArm:
        if not self.arms:
            raise ValueError(f"experiment {self.experiment_id} has no arms")
        h = hashlib.sha256(
            f"{self.experiment_id}|{user_id}".encode("utf-8")
        ).hexdigest()
        idx = int(h[:8], 16) % len(self.arms)
        return self.arms[idx]


@dataclass
class Outcome:
    user_id: str
    experiment_id: str
    arm_key: str
    outcome: str        # e.g. "trial_started", "converted_paid", "churned"
    value: float = 0.0  # optional revenue / R-multiple etc.
    ts: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Built-in experiments (eval_27 hypotheses)
# ---------------------------------------------------------------------------


EXP_TRIAL_LENGTH = Experiment(
    experiment_id="trial_length_v1",
    description="LITE 7-day vs 14-day trial without card",
    arms=(
        ExperimentArm("control_14d", "14-day trial (default)", {"trial_days": 14}),
        ExperimentArm("treat_7d",    "7-day trial",            {"trial_days": 7}),
    ),
)

EXP_ANNUAL_BUNDLE = Experiment(
    experiment_id="annual_bundle_v1",
    description="PRO+ monthly only vs PRO+ annual -20% offer at checkout",
    arms=(
        ExperimentArm("control_monthly", "Monthly only", {"annual_discount": 0.0}),
        ExperimentArm("treat_annual_20", "Annual -20% offered", {"annual_discount": 0.20}),
    ),
)


# ---------------------------------------------------------------------------
# Registry + outcome log
# ---------------------------------------------------------------------------


class ExperimentRegistry:
    def __init__(self):
        self._experiments: dict[str, Experiment] = {}
        self._outcomes: list[Outcome] = []
        self._lock = threading.Lock()

    def register(self, exp: Experiment) -> None:
        with self._lock:
            self._experiments[exp.experiment_id] = exp

    def assign(self, user_id: str, experiment_id: str) -> Optional[ExperimentArm]:
        with self._lock:
            exp = self._experiments.get(experiment_id)
        if exp is None or not exp.is_active:
            return None
        return exp.assign(user_id)

    def record_outcome(
        self,
        user_id: str,
        experiment_id: str,
        outcome: str,
        *,
        value: float = 0.0,
    ) -> Optional[Outcome]:
        arm = self.assign(user_id, experiment_id)
        if arm is None:
            return None
        rec = Outcome(
            user_id=user_id,
            experiment_id=experiment_id,
            arm_key=arm.key,
            outcome=outcome,
            value=value,
        )
        with self._lock:
            self._outcomes.append(rec)
        return rec

    def summarise(self, experiment_id: str) -> dict:
        """Per-arm conversion + value stats with simple χ² vs control."""
        with self._lock:
            exp = self._experiments.get(experiment_id)
            outcomes = [o for o in self._outcomes if o.experiment_id == experiment_id]
        if exp is None:
            return {"error": "unknown experiment"}

        per_arm: dict[str, dict] = {a.key: {"n": 0, "converted": 0, "value_sum": 0.0} for a in exp.arms}
        for o in outcomes:
            slot = per_arm.setdefault(
                o.arm_key, {"n": 0, "converted": 0, "value_sum": 0.0}
            )
            slot["n"] += 1
            if o.outcome.startswith("converted"):
                slot["converted"] += 1
            slot["value_sum"] += o.value

        for arm_key, slot in per_arm.items():
            n = slot["n"]
            slot["conversion_rate"] = round(slot["converted"] / n, 4) if n else 0.0
            slot["mean_value"] = round(slot["value_sum"] / n, 4) if n else 0.0

        # χ² vs control (control = first arm). 2x2 table per treatment arm.
        if exp.arms:
            ctrl_key = exp.arms[0].key
            ctrl = per_arm.get(ctrl_key, {"n": 0, "converted": 0})
            for arm_key, slot in per_arm.items():
                if arm_key == ctrl_key:
                    slot["chi2_p_vs_control"] = None
                    continue
                slot["chi2_p_vs_control"] = _chi2_p(
                    a_n=ctrl["n"], a_k=ctrl["converted"],
                    b_n=slot["n"], b_k=slot["converted"],
                )

        return {
            "experiment_id": experiment_id,
            "description": exp.description,
            "is_active": exp.is_active,
            "arms": per_arm,
        }


def _chi2_p(*, a_n: int, a_k: int, b_n: int, b_k: int) -> Optional[float]:
    """Two-proportion χ² test, return p-value or None when degenerate."""
    if a_n == 0 or b_n == 0:
        return None
    if (a_k + b_k) == 0 or (a_k + b_k) == (a_n + b_n):
        return None

    obs = [[a_k, a_n - a_k], [b_k, b_n - b_k]]
    row_tot = [a_n, b_n]
    col_tot = [a_k + b_k, (a_n - a_k) + (b_n - b_k)]
    grand = a_n + b_n
    chi2 = 0.0
    for i in range(2):
        for j in range(2):
            exp = row_tot[i] * col_tot[j] / grand
            if exp == 0:
                continue
            chi2 += (obs[i][j] - exp) ** 2 / exp
    # χ² (1 df) → p-value via complementary error function approximation.
    # For 1 df: p = erfc(sqrt(chi2 / 2)).
    p = math.erfc(math.sqrt(chi2 / 2.0))
    return round(p, 4)


# Module-level default registry, pre-seeded with the eval_27 experiments.
DEFAULT_REGISTRY = ExperimentRegistry()
DEFAULT_REGISTRY.register(EXP_TRIAL_LENGTH)
DEFAULT_REGISTRY.register(EXP_ANNUAL_BUNDLE)


__all__ = [
    "DEFAULT_REGISTRY",
    "EXP_ANNUAL_BUNDLE",
    "EXP_TRIAL_LENGTH",
    "Experiment",
    "ExperimentArm",
    "ExperimentRegistry",
    "Outcome",
]
