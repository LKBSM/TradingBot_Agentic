"""Tests for the regime-aware gate (Pilier 3).

Verifies bipower-variation jump detection on synthetic data, the BOCPD
integration responds to regime shifts, and the gate's 3-state decision
logic emits the right code at the right time.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.intelligence.regime_gate import (
    RegimeDecision,
    RegimeGate,
    RegimeGateOutput,
    bipower_variation,
    jump_ratio,
    realized_variance,
    run_regime_gate,
)


# =============================================================================
# Bipower variation primitives
# =============================================================================


def test_realized_variance_simple():
    r = np.array([0.01, -0.01, 0.02, -0.02, 0.01])
    rv = realized_variance(r, window=3)
    # last value: 0.02^2 + (-0.02)^2 + 0.01^2 = 0.0009
    assert np.isclose(rv[-1], 0.0009)
    # warmup is NaN
    assert np.isnan(rv[0])
    assert np.isnan(rv[1])


def test_bipower_variation_matches_rv_no_jumps():
    """Without jumps, BV ≈ RV for large samples (mu_1 correction applied)."""
    rng = np.random.default_rng(0)
    r = rng.normal(0.0, 0.005, 1000)
    window = 200
    rv = realized_variance(r, window=window)
    bv = bipower_variation(r, window=window)
    # On large clean Gaussian samples, BV/RV should be within ~10%.
    mask = np.isfinite(rv) & np.isfinite(bv) & (rv > 0)
    ratio = (bv[mask] / rv[mask]).mean()
    assert 0.85 < ratio < 1.15, f"Mean BV/RV ratio {ratio:.3f} too far from 1"


def test_jump_ratio_detects_synthetic_jump():
    """Inject one massive return into an otherwise quiet series and verify
    jump_ratio exceeds the BLOCK threshold near it."""
    rng = np.random.default_rng(1)
    r = rng.normal(0.0, 0.001, 300)
    jump_idx = 200
    r[jump_idx] = 0.10  # 100σ jump

    jr = jump_ratio(r, window=96)
    assert np.nanmax(jr[jump_idx:]) > 0.5


def test_jump_ratio_warmup_nans():
    r = np.zeros(50)
    jr = jump_ratio(r, window=96)
    # Insufficient data → all NaN
    assert np.isnan(jr).all()


# =============================================================================
# RegimeGate logic
# =============================================================================


def test_gate_emits_trade_on_quiet_series():
    rng = np.random.default_rng(2)
    gate = RegimeGate(bipower_window=96)
    # Warmup with quiet returns
    history = list(rng.normal(0.0, 0.001, 200))
    last_decision = None
    for r in history[-50:]:
        out = gate.update(log_return=r, recent_returns=np.asarray(history))
        last_decision = out.decision
    assert last_decision is RegimeDecision.TRADE


def test_gate_blocks_on_jump():
    rng = np.random.default_rng(3)
    gate = RegimeGate(bipower_window=96, jump_threshold_block=0.40)
    history = list(rng.normal(0.0, 0.001, 200))
    for r in history:
        gate.update(log_return=r, recent_returns=np.asarray(history))

    # Now inject a large jump
    big_jump = 0.08
    history.append(big_jump)
    out = gate.update(log_return=big_jump, recent_returns=np.asarray(history))
    assert out.decision in (RegimeDecision.REDUCE, RegimeDecision.BLOCK)
    assert out.jump_ratio > 0.25


def test_gate_blocks_on_changepoint_in_mean():
    """Run the gate through a clean run, then a sudden mean-shift. The
    BOCPD posterior should fire."""
    rng = np.random.default_rng(4)
    gate = RegimeGate(bipower_window=96)

    # Stable regime
    stable = rng.normal(0.0, 0.001, 300)
    for r in stable:
        gate.update(log_return=r)

    # Hard mean shift — much larger returns
    shifted = rng.normal(0.05, 0.001, 5)
    decisions = []
    for r in shifted:
        out = gate.update(log_return=r)
        decisions.append(out.decision)

    # At least one of the first few post-shift bars should be REDUCE or BLOCK
    assert any(d is not RegimeDecision.TRADE for d in decisions[:5])


def test_gate_output_serialises():
    gate = RegimeGate()
    out = gate.update(log_return=0.001)
    d = out.to_dict()
    assert "decision" in d
    assert "cp_prob" in d
    assert "jump_ratio" in d
    assert d["decision"] in ("TRADE", "REDUCE", "BLOCK")


def test_gate_reset_clears_state():
    gate = RegimeGate()
    # Inject jumpy data to push BOCPD off the prior
    for r in np.random.default_rng(5).normal(0.1, 0.01, 50):
        gate.update(log_return=float(r))
    gate.reset()
    # After reset, the run-length probability mass should sit at r=0.
    assert np.isclose(gate._bocpd_state.run_length_probs[0], 1.0)


# =============================================================================
# Batch helper
# =============================================================================


def test_run_regime_gate_shapes():
    rng = np.random.default_rng(6)
    r = rng.normal(0.0, 0.001, 400)
    decisions, cp_probs, j_ratios = run_regime_gate(r, bipower_window=96)
    assert decisions.shape == r.shape
    assert cp_probs.shape == r.shape
    assert j_ratios.shape == r.shape
    assert decisions.dtype == np.int8
    # All decision codes should be valid
    assert set(np.unique(decisions)).issubset({0, 1, 2})


def test_run_regime_gate_detects_injected_shift():
    """End-to-end batch run on a synthetic series with one jump should
    produce at least a few non-TRADE decisions."""
    rng = np.random.default_rng(7)
    r = rng.normal(0.0, 0.001, 500)
    r[300] = 0.10  # injected jump
    decisions, _, _ = run_regime_gate(r, bipower_window=96)
    n_blocked = int((decisions[295:320] != 0).sum())
    assert n_blocked >= 1
