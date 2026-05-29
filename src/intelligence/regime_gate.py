"""Regime-aware gate combining BOCPD changepoint probability with HAR-RV-J
bipower jumps.

Pilier 3 of the institutional quant transformation plan (see
``reports/institutional_quant_transformation_plan.md`` §4 Pilier 3).

Goal
----
Block new trade entries (but never exits) when the market is in a
"regime-shift" or "jump-heavy" state. Two independent detectors vote:

1. **BOCPD posterior** (Adams & MacKay 2007, ``src/intelligence/bocpd.py``).
   When ``P(changepoint at t) > regime_block_threshold`` (default 0.30),
   we are likely in or about to enter a structural shift. The hysteresis
   confluence score is still valid for *exits*, but new entries are too
   risky to take.
2. **Bipower-variation jump indicator** (Barndorff-Nielsen & Shephard
   2004). Realised variance RV decomposes into a continuous component
   (BV) and a jump component J = max(0, RV - BV). When the jump share
   J/RV exceeds ``jump_block_threshold`` (default 0.40) in a recent
   window, the market is "jump-heavy" — discretionary blocks should
   step aside until continuous volatility re-dominates.

The gate emits a 3-state decision (TRADE / REDUCE / BLOCK) plus the
underlying scalars so the narrative engine can explain *why* a trade
was blocked.

Why this is the right place to add it
-------------------------------------
The HMM regime detector in ``volatility_forecaster.py`` already gives a
3-state classification (low/normal/high vol) but its lag is 5-10 bars
because HMM Viterbi smooths posteriors over long histories. BOCPD's
Bayesian online updates respond in 1-2 bars to genuine shifts. The two
are complementary: HMM tells you "we are *in* a high-vol regime",
BOCPD tells you "we are *entering* a new regime *now*".

References
----------
- Adams, R.P. & MacKay, D.J.C. (2007). *Bayesian Online Changepoint
  Detection*. arXiv:0710.3742.
- Barndorff-Nielsen, O.E. & Shephard, N. (2004). *Power and Bipower
  Variation with Stochastic Volatility and Jumps*. JoFEM 2 (1), 1-37.
- Corsi, F. (2009). *A Simple Approximate Long-Memory Model of Realized
  Volatility*. JoFEM 7 (2), 174-196.
- Tsaknaki, I., Lillo, F., Mazzarisi, P. (2024). *Online score-driven
  CPD for financial time series*. Quant Finance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np

from src.intelligence.bocpd import BOCPDState, bocpd_step, DEFAULT_HAZARD_INV

logger = logging.getLogger(__name__)


# =============================================================================
# Public enums and result containers
# =============================================================================


class RegimeDecision(str, Enum):
    """3-state decision emitted by the gate for each bar."""

    TRADE = "TRADE"      # No regime risk detected; entries allowed
    REDUCE = "REDUCE"    # Mild regime stress; size halved (advisory)
    BLOCK = "BLOCK"      # New entries forbidden; only exits allowed


@dataclass
class RegimeGateOutput:
    """Per-bar output of the regime gate."""

    decision: RegimeDecision
    cp_prob: float           # BOCPD changepoint posterior
    jump_ratio: float        # J/RV from bipower variation (0..1)
    reason: str              # Human-readable explanation
    expected_run_length: float = 0.0  # E[r_t] under BOCPD posterior

    def to_dict(self) -> dict:
        return {
            "decision": self.decision.value,
            "cp_prob": round(self.cp_prob, 4),
            "jump_ratio": round(self.jump_ratio, 4),
            "expected_run_length": round(self.expected_run_length, 1),
            "reason": self.reason,
        }


# =============================================================================
# Bipower variation (Barndorff-Nielsen & Shephard 2004)
# =============================================================================


_MU_1 = np.sqrt(2.0 / np.pi)  # E|N(0,1)|


def realized_variance(returns: np.ndarray, window: int = 96) -> np.ndarray:
    """Rolling realised variance over ``window`` log-returns.

    RV_t = Σ_{i=t-window+1}^{t} r_i^2
    """
    r = np.asarray(returns, dtype=float)
    r2 = r ** 2
    out = np.full_like(r, np.nan)
    if len(r) < window:
        return out
    cum = np.cumsum(np.insert(r2, 0, 0.0))
    out[window - 1 :] = cum[window:] - cum[:-window]
    return out


def bipower_variation(returns: np.ndarray, window: int = 96) -> np.ndarray:
    """Rolling bipower variation (the jump-robust analogue of RV).

    BV_t = (1 / mu_1^2) * Σ_{i=t-window+2}^{t} |r_{i-1}| * |r_i|

    Under the null of no jumps, BV → RV in probability. The gap RV - BV
    is the jump contribution.
    """
    r = np.asarray(returns, dtype=float)
    n = len(r)
    out = np.full(n, np.nan)
    if n < window:
        return out
    abs_r = np.abs(r)
    products = abs_r[1:] * abs_r[:-1]  # length n-1
    # We want rolling sum of `products` over a window of size (window-1) and
    # to anchor the result at index `i` corresponding to position window-1.
    win = window - 1
    if win <= 0:
        return out
    cum = np.cumsum(np.insert(products, 0, 0.0))
    rolling = cum[win:] - cum[:-win]  # length n - win
    bv = (1.0 / (_MU_1 ** 2)) * rolling
    # `bv[k]` is the bipower variation using r[k+1-win-1+1 ... k+1], so it
    # corresponds to position `k + win` of the original `returns` array.
    out[win:] = bv
    return out


def jump_ratio(returns: np.ndarray, window: int = 96) -> np.ndarray:
    """Per-bar jump share J/RV in [0, 1] using bipower variation.

    Returns NaN where insufficient data; otherwise clamped to [0, 1].
    """
    rv = realized_variance(returns, window)
    bv = bipower_variation(returns, window)
    with np.errstate(invalid="ignore", divide="ignore"):
        j = np.maximum(0.0, rv - bv)
        ratio = np.where(rv > 0, j / rv, np.nan)
    ratio = np.clip(ratio, 0.0, 1.0)
    return ratio


# =============================================================================
# Main gate
# =============================================================================


class RegimeGate:
    """Online regime-shift gate combining BOCPD + bipower jumps.

    Designed to be called once per bar inside the scanner loop (after the
    ConfluenceDetector has produced a candidate signal). The decision can
    be applied as a hard block on new entries while exits flow through the
    existing SignalStateMachine unchanged.

    Usage
    -----
    >>> gate = RegimeGate()
    >>> for bar in stream:
    ...     out = gate.update(log_return=bar.log_return, recent_returns=window)
    ...     if out.decision is RegimeDecision.BLOCK:
    ...         skip_new_entries()
    """

    def __init__(
        self,
        bocpd_threshold_block: float = 0.10,
        bocpd_threshold_reduce: float = 0.05,
        jump_threshold_block: float = 0.40,
        jump_threshold_reduce: float = 0.25,
        bocpd_hazard_inv: float = DEFAULT_HAZARD_INV,
        bocpd_max_run_length: int = 300,
        bocpd_prior_kappa: float = 1.0,
        bocpd_prior_alpha: float = 2.0,
        bocpd_prior_beta: float = 1e-6,
        bipower_window: int = 96,
    ) -> None:
        self.bocpd_threshold_block = float(bocpd_threshold_block)
        self.bocpd_threshold_reduce = float(bocpd_threshold_reduce)
        self.jump_threshold_block = float(jump_threshold_block)
        self.jump_threshold_reduce = float(jump_threshold_reduce)
        self.bipower_window = int(bipower_window)

        self._bocpd_state = BOCPDState(
            max_run_length=int(bocpd_max_run_length),
            hazard_inv=float(bocpd_hazard_inv),
            mu_0=0.0,
            kappa_0=float(bocpd_prior_kappa),
            alpha_0=float(bocpd_prior_alpha),
            beta_0=float(bocpd_prior_beta),
        )

    def update(
        self,
        log_return: float,
        recent_returns: Optional[np.ndarray] = None,
    ) -> RegimeGateOutput:
        """Process one bar.

        Parameters
        ----------
        log_return
            The latest log-return (close-to-close on the active timeframe).
        recent_returns
            A 1-D array containing AT LEAST ``bipower_window`` of the most
            recent log-returns INCLUDING the current bar. If omitted, the
            jump ratio is treated as 0 (no jump info available).

        Returns
        -------
        RegimeGateOutput
        """
        # BOCPD posterior — runs even on the first observation
        cp_prob = float(bocpd_step(self._bocpd_state, float(log_return)))

        # Bipower jump ratio for the current bar
        if recent_returns is not None and len(recent_returns) >= self.bipower_window:
            window = np.asarray(recent_returns[-self.bipower_window :], dtype=float)
            ratio_series = jump_ratio(window, window=self.bipower_window)
            j_ratio = float(ratio_series[-1]) if np.isfinite(ratio_series[-1]) else 0.0
        else:
            j_ratio = 0.0

        # Decision logic
        block = (
            cp_prob >= self.bocpd_threshold_block
            or j_ratio >= self.jump_threshold_block
        )
        reduce = (
            cp_prob >= self.bocpd_threshold_reduce
            or j_ratio >= self.jump_threshold_reduce
        )

        if block:
            decision = RegimeDecision.BLOCK
            triggers = []
            if cp_prob >= self.bocpd_threshold_block:
                triggers.append(f"cp_prob={cp_prob:.3f}>={self.bocpd_threshold_block:.2f}")
            if j_ratio >= self.jump_threshold_block:
                triggers.append(f"jump_ratio={j_ratio:.3f}>={self.jump_threshold_block:.2f}")
            reason = "BLOCK: " + " & ".join(triggers)
        elif reduce:
            decision = RegimeDecision.REDUCE
            reason = (
                f"REDUCE: cp_prob={cp_prob:.3f}, jump_ratio={j_ratio:.3f}"
            )
        else:
            decision = RegimeDecision.TRADE
            reason = (
                f"TRADE: cp_prob={cp_prob:.3f}, jump_ratio={j_ratio:.3f}"
            )

        # Expected run length is a useful narrative hook
        rs = np.arange(self._bocpd_state.max_run_length, dtype=np.float64)
        expected_rl = float((self._bocpd_state.run_length_probs * rs).sum())

        return RegimeGateOutput(
            decision=decision,
            cp_prob=cp_prob,
            jump_ratio=j_ratio,
            reason=reason,
            expected_run_length=expected_rl,
        )

    def reset(self) -> None:
        """Reset the BOCPD state (used between symbol sessions or on
        scanner cold-start)."""
        self._bocpd_state = BOCPDState(
            max_run_length=self._bocpd_state.max_run_length,
            hazard_inv=self._bocpd_state.hazard_inv,
            mu_0=self._bocpd_state.mu_0,
            kappa_0=self._bocpd_state.kappa_0,
            alpha_0=self._bocpd_state.alpha_0,
            beta_0=self._bocpd_state.beta_0,
        )


# =============================================================================
# Batch helper for backtest replays
# =============================================================================


def run_regime_gate(
    log_returns: np.ndarray,
    bipower_window: int = 96,
    **gate_kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised batch evaluation for replays.

    Returns three arrays of equal length:
    - decisions  : int codes 0=TRADE, 1=REDUCE, 2=BLOCK
    - cp_probs   : BOCPD posterior at each bar
    - jump_ratios: bipower jump share at each bar (NaN before warmup)
    """
    r = np.asarray(log_returns, dtype=float)
    n = len(r)

    decisions = np.zeros(n, dtype=np.int8)
    cp_probs = np.zeros(n, dtype=np.float64)
    j_ratios = np.full(n, np.nan, dtype=np.float64)

    gate = RegimeGate(bipower_window=bipower_window, **gate_kwargs)

    for i in range(n):
        recent = r[max(0, i - bipower_window + 1) : i + 1]
        out = gate.update(log_return=float(r[i]), recent_returns=recent)
        cp_probs[i] = out.cp_prob
        j_ratios[i] = out.jump_ratio
        if out.decision is RegimeDecision.BLOCK:
            decisions[i] = 2
        elif out.decision is RegimeDecision.REDUCE:
            decisions[i] = 1
        else:
            decisions[i] = 0

    return decisions, cp_probs, j_ratios
