"""Tests for the COMM-2B.5 pricing A/B framework."""

from __future__ import annotations

import pytest

from src.experiments.pricing_ab import (
    EXP_ANNUAL_BUNDLE,
    EXP_TRIAL_LENGTH,
    Experiment,
    ExperimentArm,
    ExperimentRegistry,
)


def test_assignment_is_deterministic():
    arm_a = EXP_TRIAL_LENGTH.assign("user-42")
    arm_b = EXP_TRIAL_LENGTH.assign("user-42")
    assert arm_a is arm_b


def test_different_users_can_get_different_arms():
    seen = set()
    for i in range(200):
        seen.add(EXP_TRIAL_LENGTH.assign(f"user-{i}").key)
    # Both arms hit with high probability across 200 random users.
    assert len(seen) == 2


def test_empty_arms_raises():
    exp = Experiment(experiment_id="empty", description="", arms=())
    with pytest.raises(ValueError):
        exp.assign("u1")


def test_registry_register_and_assign():
    reg = ExperimentRegistry()
    reg.register(EXP_TRIAL_LENGTH)
    assert reg.assign("u1", EXP_TRIAL_LENGTH.experiment_id) is not None


def test_assign_unknown_returns_none():
    reg = ExperimentRegistry()
    assert reg.assign("u1", "nope") is None


def test_record_outcome_uses_arm_assignment():
    reg = ExperimentRegistry()
    reg.register(EXP_TRIAL_LENGTH)
    for i in range(50):
        reg.record_outcome(f"u-{i}", EXP_TRIAL_LENGTH.experiment_id, "trial_started")
    summary = reg.summarise(EXP_TRIAL_LENGTH.experiment_id)
    # Both arms should have at least one observation.
    for arm_key, slot in summary["arms"].items():
        assert slot["n"] > 0


def test_summarise_reports_conversion_rate():
    reg = ExperimentRegistry()
    reg.register(EXP_TRIAL_LENGTH)
    # All 50 users in arm A convert; the framework should pick that up.
    for i in range(50):
        arm = reg.assign(f"u-{i}", EXP_TRIAL_LENGTH.experiment_id)
        if arm.key == "control_14d":
            reg.record_outcome(f"u-{i}", EXP_TRIAL_LENGTH.experiment_id, "converted_paid", value=29.0)
        else:
            reg.record_outcome(f"u-{i}", EXP_TRIAL_LENGTH.experiment_id, "trial_started")
    s = reg.summarise(EXP_TRIAL_LENGTH.experiment_id)
    ctrl = s["arms"]["control_14d"]
    treat = s["arms"]["treat_7d"]
    assert ctrl["conversion_rate"] == 1.0
    assert treat["conversion_rate"] == 0.0


def test_chi2_p_value_finite_when_treatment_differs():
    reg = ExperimentRegistry()
    reg.register(EXP_TRIAL_LENGTH)
    # Skew so the χ² has signal
    for i in range(100):
        arm = reg.assign(f"u-{i}", EXP_TRIAL_LENGTH.experiment_id)
        if arm.key == "control_14d":
            reg.record_outcome(f"u-{i}", EXP_TRIAL_LENGTH.experiment_id, "converted_paid")
        else:
            reg.record_outcome(f"u-{i}", EXP_TRIAL_LENGTH.experiment_id, "trial_started")
    s = reg.summarise(EXP_TRIAL_LENGTH.experiment_id)
    treat = s["arms"]["treat_7d"]
    assert isinstance(treat["chi2_p_vs_control"], float)
    assert 0 <= treat["chi2_p_vs_control"] <= 1.0


def test_both_eval_27_experiments_registered_in_default():
    from src.experiments.pricing_ab import DEFAULT_REGISTRY

    assert DEFAULT_REGISTRY.assign("u1", EXP_TRIAL_LENGTH.experiment_id) is not None
    assert DEFAULT_REGISTRY.assign("u1", EXP_ANNUAL_BUNDLE.experiment_id) is not None


def test_arm_payload_carries_config():
    arm = EXP_TRIAL_LENGTH.assign("user-1")
    assert "trial_days" in arm.payload
    assert arm.payload["trial_days"] in {7, 14}
