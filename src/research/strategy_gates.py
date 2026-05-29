"""Quant strategy admission gates.

Single source of truth for the CPCV + DSR + PBO + Diebold-Mariano +
profit-factor lower-CI thresholds that every new strategy must clear before
it is allowed to graduate from research into production code.

Why this module exists
----------------------
A1 verdict 2026-05-01 (DSR=0.000, PBO=0.500) was caught only because we
*chose* to apply CPCV + DSR + PBO + DM. The institutional quant
transformation plan (reports/institutional_quant_transformation_plan.md §0,
§6 Sprint 0) makes those checks MANDATORY for any new strategy or signal
component. This module is the contract every evaluation script reuses.

Threshold rationale (defaults)
------------------------------
- DSR >= 1.5
    Bailey & López de Prado 2014. Below 1.5 the deflated Sharpe is
    statistically indistinguishable from "model with no edge tested under N
    trials". A1's stack scored 0.0 → that is the value any unbiased coin
    flip would yield on 28 paths.
- PBO <= 0.35
    Bailey-Borwein-LdP-Zhu 2014. PBO=0.5 is "best-IS strategy is below
    median OOS half the time" = pure noise. 0.35 leaves 15% safety margin
    over the noise floor while remaining attainable.
- PF lower CI 95% > 1.00
    Profit factor bootstrap lower bound must exceed 1.0; that is the
    *minimum* commercial-grade survival criterion (`reports/decision_matrix_
    2026_04_30.md` showed 0/4 strategies currently clear it).
- Diebold-Mariano p < 0.05 vs constant baseline
    The strategy must produce returns that are *statistically distinguishable*
    from the constant-zero baseline. Without this, you cannot reject "noise"
    even if the point estimate of Sharpe is positive.

Optional gates (per-strategy):
- Sharpe net OOS > some target (set by caller, default disabled)
- Min trades (default 30; under that, point estimates are unreliable)

Usage
-----
>>> from src.research.strategy_gates import evaluate_gates, GateResult
>>> result = evaluate_gates(
...     returns=path_returns_array,    # 1D returns per trade or per bar
...     n_trials=28,                   # CPCV paths, or hyperparam grid size
...     baseline_returns=np.zeros_like(path_returns_array),
...     n_bootstraps=1000,
... )
>>> assert result.all_passed, result.failure_reasons
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.research.cpcv_harness import (
    deflated_sharpe_ratio,
    diebold_mariano,
    profit_factor,
    sharpe_ratio,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Default thresholds (do not weaken without team sign-off)
# =============================================================================

DEFAULT_DSR_MIN = 1.5
DEFAULT_PBO_MAX = 0.35
DEFAULT_PF_LO_MIN = 1.00
DEFAULT_DM_PVAL_MAX = 0.05
DEFAULT_MIN_TRADES = 30
DEFAULT_BOOTSTRAP_N = 1000
DEFAULT_BOOTSTRAP_SEED = 42


# =============================================================================
# Result container
# =============================================================================


@dataclass
class GateResult:
    """Outcome of running the strategy admission gates."""

    # The raw metrics
    n_trades: int
    sharpe: float
    profit_factor: float
    profit_factor_lo: float
    profit_factor_hi: float
    dsr: float
    pbo: float
    dm_stat: float
    dm_pvalue: float

    # Per-gate pass flags
    trades_pass: bool
    dsr_pass: bool
    pbo_pass: bool
    pf_lo_pass: bool
    dm_pass: bool

    # The thresholds used (so they appear in serialised output)
    thresholds: dict = field(default_factory=dict)
    failure_reasons: list = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return (
            self.trades_pass
            and self.dsr_pass
            and self.pbo_pass
            and self.pf_lo_pass
            and self.dm_pass
        )

    def to_dict(self) -> dict:
        return {
            "passed": self.all_passed,
            "n_trades": self.n_trades,
            "sharpe": self.sharpe,
            "profit_factor": self.profit_factor,
            "profit_factor_lo": self.profit_factor_lo,
            "profit_factor_hi": self.profit_factor_hi,
            "dsr": self.dsr,
            "pbo": self.pbo,
            "dm_stat": self.dm_stat,
            "dm_pvalue": self.dm_pvalue,
            "gates": {
                "trades": self.trades_pass,
                "dsr": self.dsr_pass,
                "pbo": self.pbo_pass,
                "pf_lo": self.pf_lo_pass,
                "dm": self.dm_pass,
            },
            "thresholds": self.thresholds,
            "failure_reasons": self.failure_reasons,
        }


# =============================================================================
# Profit-factor bootstrap CI
# =============================================================================


def profit_factor_bootstrap_ci(
    returns: np.ndarray,
    n_bootstraps: int = DEFAULT_BOOTSTRAP_N,
    alpha: float = 0.05,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    """Return (point_pf, lower_ci, upper_ci) for the profit factor.

    Uses bootstrap resampling of trades with replacement. With 1000 draws and
    n>=30 trades, the lower-95 CI is a reasonable proxy for the "stress test"
    PF that a portfolio could expect under regime shifts.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    point = profit_factor(r)
    if n < 5:
        return point, 0.0, float("inf")

    rng = np.random.default_rng(seed)
    pfs = np.empty(n_bootstraps, dtype=float)
    for i in range(n_bootstraps):
        sample = rng.choice(r, size=n, replace=True)
        pfs[i] = profit_factor(sample)
    pfs = pfs[np.isfinite(pfs)]
    if len(pfs) == 0:
        return point, 0.0, float("inf")
    lo = float(np.quantile(pfs, alpha / 2))
    hi = float(np.quantile(pfs, 1 - alpha / 2))
    return point, lo, hi


# =============================================================================
# Main API
# =============================================================================


def evaluate_gates(
    returns: np.ndarray,
    n_trials: int = 1,
    baseline_returns: Optional[np.ndarray] = None,
    path_returns: Optional[list] = None,
    dsr_min: float = DEFAULT_DSR_MIN,
    pbo_max: float = DEFAULT_PBO_MAX,
    pf_lo_min: float = DEFAULT_PF_LO_MIN,
    dm_pvalue_max: float = DEFAULT_DM_PVAL_MAX,
    min_trades: int = DEFAULT_MIN_TRADES,
    n_bootstraps: int = DEFAULT_BOOTSTRAP_N,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> GateResult:
    """Run all admission gates on a strategy's return series.

    Parameters
    ----------
    returns : np.ndarray
        1-D array of per-trade or per-bar returns (in fractional units).
        At least `min_trades` non-NaN entries required for the gates to fire.
    n_trials : int, default 1
        Number of strategies/hyperparam-configs/CPCV-paths tried. Inflates the
        DSR penalty. Use 28 for a standard CPCV sweep, or your grid size for
        an explicit hyperparam search.
    baseline_returns : np.ndarray, optional
        Return series of the comparison strategy for the Diebold-Mariano test.
        Defaults to a zero-vector (= "no trading" baseline).
    path_returns : list[np.ndarray], optional
        Per-CPCV-path returns for the PBO calculation. If omitted, PBO is
        computed via the median-split proxy on the input `returns`.

    Returns
    -------
    GateResult
        Container with per-gate flags, raw metrics, and any failure reasons.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n_trades = len(r)

    thresholds = {
        "dsr_min": dsr_min,
        "pbo_max": pbo_max,
        "pf_lo_min": pf_lo_min,
        "dm_pvalue_max": dm_pvalue_max,
        "min_trades": min_trades,
        "n_trials": n_trials,
    }
    failures: list = []

    trades_pass = n_trades >= min_trades
    if not trades_pass:
        failures.append(
            f"min_trades: have {n_trades}, need >= {min_trades}"
        )

    if n_trades < 4:
        # Below this, every metric is undefined / unstable.
        return GateResult(
            n_trades=n_trades,
            sharpe=0.0,
            profit_factor=0.0,
            profit_factor_lo=0.0,
            profit_factor_hi=0.0,
            dsr=0.0,
            pbo=1.0,
            dm_stat=0.0,
            dm_pvalue=1.0,
            trades_pass=trades_pass,
            dsr_pass=False,
            pbo_pass=False,
            pf_lo_pass=False,
            dm_pass=False,
            thresholds=thresholds,
            failure_reasons=failures + ["sample too small to evaluate (< 4)"],
        )

    sr = sharpe_ratio(r)
    pf_point, pf_lo, pf_hi = profit_factor_bootstrap_ci(
        r, n_bootstraps=n_bootstraps, seed=bootstrap_seed
    )

    # DSR returns a probability in [0,1] in the current implementation;
    # the gate expects "value >= 1.5" which translates to a z-score on the
    # Sharpe. We compute the equivalent z-score for the gate to match the
    # intuition documented in the plan.
    dsr_prob = deflated_sharpe_ratio(r, n_trials=n_trials)
    # Convert prob -> z-score-equivalent via the inverse normal CDF.
    # A DSR-prob of 0.93 ≈ z=1.5; 0.99 ≈ z=2.33.
    from scipy import stats as _stats
    if dsr_prob >= 1.0:
        dsr_z = 8.0
    elif dsr_prob <= 0.0:
        dsr_z = -8.0
    else:
        dsr_z = float(_stats.norm.ppf(dsr_prob))

    # PBO — single-strategy proxy: fraction of paths/chunks with non-positive
    # Sharpe. A genuine edge should keep most chunks above zero; pure noise
    # gives 50/50. This is NOT the multi-strategy Bailey-Borwein-LdP-Zhu PBO
    # (which requires hyperparam grid). It is the analogue for single-strategy
    # validation where what we test is "is the edge consistent across folds".
    if path_returns is not None and len(path_returns) >= 2:
        sharpes = np.array([sharpe_ratio(np.asarray(p)) for p in path_returns])
    else:
        # Fallback: split the series into 10 contiguous chunks and compute
        # per-chunk Sharpe. Requires >=40 samples to make this meaningful.
        n_chunks = min(10, max(2, n_trades // 20))
        chunk_size = n_trades // n_chunks
        if chunk_size < 4:
            pbo = 1.0
            sharpes = np.array([])
        else:
            chunks = [
                r[i * chunk_size : (i + 1) * chunk_size]
                for i in range(n_chunks)
            ]
            sharpes = np.array([sharpe_ratio(c) for c in chunks])

    if len(sharpes) > 0:
        pbo = float((sharpes <= 0).mean())

    # Diebold-Mariano vs baseline (default: zero-baseline = "no trading")
    if baseline_returns is None:
        b = np.zeros_like(r)
    else:
        b = np.asarray(baseline_returns, dtype=float)
        b = b[np.isfinite(b)]
        if len(b) != len(r):
            # Resize or zero-pad — keep equal length
            if len(b) >= len(r):
                b = b[: len(r)]
            else:
                b = np.concatenate([b, np.zeros(len(r) - len(b))])

    # DM compares forecast *errors*; we want to compare PnL. We flip the
    # sign convention so "lower error" = "higher return". Treating returns
    # as negative errors works as a proxy.
    dm_stat, dm_pvalue = diebold_mariano(-r, -b, h=1)

    dsr_pass = dsr_z >= dsr_min
    pbo_pass = pbo <= pbo_max
    pf_lo_pass = pf_lo > pf_lo_min
    dm_pass = dm_pvalue < dm_pvalue_max and r.mean() > b.mean()

    if not dsr_pass:
        failures.append(f"DSR z-score {dsr_z:.3f} < {dsr_min} (prob={dsr_prob:.3f})")
    if not pbo_pass:
        failures.append(f"PBO {pbo:.3f} > {pbo_max}")
    if not pf_lo_pass:
        failures.append(
            f"PF lower-CI {pf_lo:.3f} <= {pf_lo_min} (point={pf_point:.3f})"
        )
    if not dm_pass:
        failures.append(
            f"DM p-value {dm_pvalue:.3f} >= {dm_pvalue_max} "
            f"(or mean return {r.mean():.6f} <= baseline {b.mean():.6f})"
        )

    return GateResult(
        n_trades=n_trades,
        sharpe=sr,
        profit_factor=pf_point,
        profit_factor_lo=pf_lo,
        profit_factor_hi=pf_hi,
        dsr=dsr_z,
        pbo=pbo,
        dm_stat=dm_stat,
        dm_pvalue=dm_pvalue,
        trades_pass=trades_pass,
        dsr_pass=dsr_pass,
        pbo_pass=pbo_pass,
        pf_lo_pass=pf_lo_pass,
        dm_pass=dm_pass,
        thresholds=thresholds,
        failure_reasons=failures,
    )


def assert_passes_gates(
    returns: np.ndarray,
    strategy_name: str = "unnamed",
    **kwargs,
) -> GateResult:
    """Strict version that raises AssertionError on failure.

    Intended for use in CI / unit-test contexts where a failing strategy
    must HARD-FAIL the build, e.g.:

    >>> def test_event_driven_macro_gates():
    ...     returns = run_strategy_oos()
    ...     assert_passes_gates(returns, "event_driven_macro", n_trials=28)
    """
    result = evaluate_gates(returns, **kwargs)
    if not result.all_passed:
        raise AssertionError(
            f"strategy '{strategy_name}' failed admission gates:\n  - "
            + "\n  - ".join(result.failure_reasons)
            + f"\nFull metrics: {result.to_dict()}"
        )
    return result
