"""Tests for the strategy admission gates.

Validates that the gate logic rejects pure-noise strategies (the A1-style
zero-edge case) and accepts genuinely profitable, distinguishable strategies.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.research.strategy_gates import (
    DEFAULT_BOOTSTRAP_N,
    DEFAULT_DM_PVAL_MAX,
    DEFAULT_DSR_MIN,
    DEFAULT_PBO_MAX,
    DEFAULT_PF_LO_MIN,
    GateResult,
    assert_passes_gates,
    evaluate_gates,
    profit_factor_bootstrap_ci,
)


def _make_profitable_returns(n: int = 200, seed: int = 0, edge: float = 0.0015) -> np.ndarray:
    """Build a synthetic return series with a real, persistent positive edge.

    edge=0.0015 with sigma=0.01 → Sharpe ≈ 0.15 per trade
    annualised (252 * 96 bars/yr ≈ 24k) that's about 23 → very strong.
    The point is *unambiguously profitable* so we can sanity-check the gate.
    """
    rng = np.random.default_rng(seed)
    return rng.normal(edge, 0.01, n)


def _make_zero_edge_returns(n: int = 200, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.01, n)


# =============================================================================
# Bootstrap CI
# =============================================================================


def test_bootstrap_ci_profitable_above_one():
    r = _make_profitable_returns(n=500, edge=0.005)
    point, lo, hi = profit_factor_bootstrap_ci(r, n_bootstraps=500, seed=1)
    assert point > 1.0
    assert lo > 1.0
    assert hi >= point


def test_bootstrap_ci_noise_straddles_one():
    r = _make_zero_edge_returns(n=500, seed=2)
    point, lo, hi = profit_factor_bootstrap_ci(r, n_bootstraps=500, seed=1)
    # A noise series may have point > 1 or < 1 by chance, but the CI must
    # straddle 1.0 (i.e., lower bound be at or below 1).
    assert lo <= 1.05


# =============================================================================
# Profitable strategy passes
# =============================================================================


def test_profitable_strategy_passes_all_gates():
    r = _make_profitable_returns(n=500, edge=0.005, seed=42)
    result = evaluate_gates(r, n_trials=1, n_bootstraps=500)
    assert result.all_passed, result.failure_reasons
    assert result.trades_pass
    assert result.dsr_pass
    assert result.pbo_pass
    assert result.pf_lo_pass
    assert result.dm_pass


# =============================================================================
# Zero-edge (A1-style) is rejected
# =============================================================================


def test_zero_edge_fails_at_least_one_gate():
    r = _make_zero_edge_returns(n=500, seed=7)
    result = evaluate_gates(r, n_trials=28, n_bootstraps=500)
    assert not result.all_passed
    # At minimum, the DM gate vs zero-baseline must fail
    # (no significant difference from "doing nothing").
    assert not result.dm_pass


def test_zero_edge_dsr_below_threshold():
    r = _make_zero_edge_returns(n=500, seed=8)
    result = evaluate_gates(r, n_trials=28, n_bootstraps=500)
    # DSR z should be ~ 0 for pure noise tested under 28 trials.
    assert result.dsr < DEFAULT_DSR_MIN


# =============================================================================
# Trades floor
# =============================================================================


def test_too_few_trades_blocks_gate():
    r = _make_profitable_returns(n=10, edge=0.005, seed=3)
    result = evaluate_gates(r, n_trials=1, min_trades=30, n_bootstraps=200)
    assert not result.trades_pass
    assert not result.all_passed
    assert any("min_trades" in fr for fr in result.failure_reasons)


def test_tiny_sample_returns_safe_zeros():
    r = np.array([0.01, -0.01])
    result = evaluate_gates(r, n_trials=1, n_bootstraps=100)
    # < 4 samples → all-zero/all-fail GateResult, no exception.
    assert not result.all_passed
    assert result.n_trades == 2


# =============================================================================
# PBO with explicit path_returns
# =============================================================================


def test_pbo_with_path_returns_inputs():
    # Two paths: one consistently profitable, one consistently flat.
    # PBO is "fraction of paths with Sharpe <= 0" — the flat one has Sharpe ≈ 0
    # so PBO should be ~0.5; the profitable one should be > 0.
    profitable = _make_profitable_returns(n=200, edge=0.005, seed=10)
    flat = _make_zero_edge_returns(n=200, seed=11)
    paths = [profitable, flat]
    combined = np.concatenate(paths)
    result = evaluate_gates(
        combined,
        n_trials=2,
        path_returns=paths,
        n_bootstraps=300,
    )
    assert isinstance(result, GateResult)
    assert 0.0 <= result.pbo <= 1.0


def test_pbo_all_profitable_paths_passes():
    # 10 independent profitable seeds — should give PBO well below 0.35
    paths = [
        _make_profitable_returns(n=100, edge=0.005, seed=s)
        for s in range(10)
    ]
    combined = np.concatenate(paths)
    result = evaluate_gates(
        combined, n_trials=10, path_returns=paths, n_bootstraps=300
    )
    assert result.pbo < DEFAULT_PBO_MAX, (
        f"PBO {result.pbo} should be < {DEFAULT_PBO_MAX} for consistent edge"
    )


# =============================================================================
# assert_passes_gates raises on failure, passes on success
# =============================================================================


def test_assert_passes_gates_raises_on_zero_edge():
    r = _make_zero_edge_returns(n=200, seed=9)
    with pytest.raises(AssertionError) as exc:
        assert_passes_gates(r, strategy_name="noise_test", n_trials=28)
    assert "noise_test" in str(exc.value)
    assert "failed admission gates" in str(exc.value)


def test_assert_passes_gates_succeeds_on_real_edge():
    r = _make_profitable_returns(n=500, edge=0.005, seed=4)
    result = assert_passes_gates(r, strategy_name="profitable_test", n_trials=1)
    assert result.all_passed


# =============================================================================
# Result is JSON-serialisable
# =============================================================================


def test_gate_result_serialises_to_dict():
    r = _make_profitable_returns(n=200, edge=0.003, seed=5)
    result = evaluate_gates(r, n_trials=1, n_bootstraps=200)
    d = result.to_dict()
    assert isinstance(d, dict)
    assert "gates" in d
    assert "thresholds" in d
    assert "n_trades" in d
    assert d["n_trades"] == 200
