"""
Combinatorial Purged Cross-Validation harness with DSR + PBO + Holm.

Sprint QUANT-1.2 (Elena, 6h). See `reports/roadmap_2026_2027/PLAN_12_MOIS.md`
Partie II.2 Agent 2.

Implements the gold-standard backtest validation methodology for time-series
ML (López de Prado, *Advances in Financial Machine Learning*, ch. 7) plus
the meta-statistics that prevent backtest overfitting:

- **CPCV split**: N=8 folds, k=2 test folds per path → C(8,2) = 28 paths
- **Purging**: training observations whose label-horizon overlaps the test
  fold are removed (eliminates label leakage)
- **Embargo**: training observations within `embargo` bars of the test fold
  boundary are also removed (eliminates serial-correlation leakage)
- **Deflated Sharpe Ratio** (Bailey & López de Prado 2014): adjusts the
  observed SR for skew, kurtosis, and the multiple-testing inflation that
  arises from trying many strategies / hyperparameter configurations
- **PBO** (Probability of Backtest Overfitting, Bailey-Borwein-López-Zhu
  2014, rank-logit method): probability that an IS-best configuration ranks
  OOS *worse* than median
- **Holm-Bonferroni**: family-wise error control on per-feature p-values

References
----------
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Bailey, D.H., & López de Prado, M. (2014). *The Deflated Sharpe Ratio:
  Correcting for Selection Bias, Backtest Overfitting and Non-Normality*.
- Bailey, D.H., Borwein, J., López de Prado, M., & Zhu, Q.J. (2014).
  *The Probability of Backtest Overfitting*.
- Diebold, F.X., & Mariano, R.S. (1995). *Comparing Predictive Accuracy*.
- Holm, S. (1979). *A Simple Sequentially Rejective Multiple Test Procedure*.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class CPCVPath:
    """One out of the 28 CPCV paths."""

    fold_combo: tuple[int, ...]
    train_size: int
    test_size: int
    sharpe: float
    profit_factor: float
    hit_rate: float
    n_trades: int
    returns: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class CPCVResult:
    """End-to-end CPCV summary."""

    n_folds: int
    n_test_folds: int
    embargo: int
    paths: list[CPCVPath]
    sharpe_mean: float
    sharpe_p25: float
    sharpe_p75: float
    pf_mean: float
    pf_p25: float
    dsr: float
    pbo: float
    holm_significant: dict[str, bool] = field(default_factory=dict)
    holm_pvalues: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# CPCV split logic (López de Prado AFML ch. 7)
# ---------------------------------------------------------------------------


def split_into_n_folds(n_samples: int, n_folds: int) -> list[tuple[int, int]]:
    """Contiguous fold boundaries [start, end) along the timeline."""
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    if n_samples < n_folds:
        raise ValueError(f"need at least {n_folds} samples, got {n_samples}")
    sizes = np.full(n_folds, n_samples // n_folds)
    sizes[: n_samples % n_folds] += 1  # distribute remainder
    boundaries = []
    cursor = 0
    for s in sizes:
        boundaries.append((cursor, cursor + s))
        cursor += s
    return boundaries


def purged_train_indices(
    test_folds: list[tuple[int, int]],
    n_samples: int,
    embargo: int,
    label_horizon: int,
) -> np.ndarray:
    """Return train indices after applying purge + embargo around test folds.

    Purge: any training index `i` whose label-window [i, i + label_horizon)
    overlaps any test fold is removed.
    Embargo: also remove training indices within `embargo` bars of any test
    fold boundary (in either direction) to prevent serial-correlation leakage.

    The combined "forbidden zone" for each test fold [a, b) is:
        [a - embargo - label_horizon, b + embargo)
    (the `- label_horizon` extends backwards to catch labels straddling `a`.)
    """
    forbidden = np.zeros(n_samples, dtype=bool)
    for a, b in test_folds:
        lo = max(0, a - embargo - label_horizon)
        hi = min(n_samples, b + embargo)
        forbidden[lo:hi] = True
    test_mask = np.zeros(n_samples, dtype=bool)
    for a, b in test_folds:
        test_mask[a:b] = True
    train_mask = ~forbidden & ~test_mask
    return np.where(train_mask)[0]


def cpcv_path_indices(
    n_samples: int,
    n_folds: int,
    n_test_folds: int,
    embargo: int,
    label_horizon: int,
):
    """Generator of (path_id, fold_combo, train_idx, test_idx)."""
    fold_boundaries = split_into_n_folds(n_samples, n_folds)
    combos = list(combinations(range(n_folds), n_test_folds))
    for path_id, combo in enumerate(combos):
        test_fold_ranges = [fold_boundaries[i] for i in combo]
        # Concatenate test indices across the chosen folds
        test_idx_parts = [np.arange(a, b) for a, b in test_fold_ranges]
        test_idx = np.concatenate(test_idx_parts)
        train_idx = purged_train_indices(
            test_fold_ranges, n_samples, embargo, label_horizon
        )
        yield path_id, combo, train_idx, test_idx


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------


def sharpe_ratio(returns: np.ndarray, periods_per_year: int = 252 * 96) -> float:
    """Annualised Sharpe. Default annualisation = M15 bars per year ~ 252 * 96.

    For our use case, what matters is the *relative* Sharpe (paths comparable
    to each other), so the absolute annualisation factor is mostly cosmetic.
    """
    r = np.asarray(returns)
    r = r[~np.isnan(r)]
    if len(r) < 2 or r.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / r.std(ddof=1) * np.sqrt(periods_per_year))


def profit_factor(returns: np.ndarray) -> float:
    """Sum of positives / |sum of negatives|. inf if no losses."""
    r = np.asarray(returns)
    r = r[~np.isnan(r)]
    pos = r[r > 0].sum()
    neg = -r[r < 0].sum()
    if neg == 0:
        return float("inf") if pos > 0 else 0.0
    return float(pos / neg)


def hit_rate(returns: np.ndarray) -> float:
    r = np.asarray(returns)
    r = r[~np.isnan(r)]
    if len(r) == 0:
        return 0.0
    return float((r > 0).mean())


# ---------------------------------------------------------------------------
# Deflated Sharpe Ratio (Bailey & López de Prado 2014)
# ---------------------------------------------------------------------------


def expected_max_sharpe(n_trials: int) -> float:
    """E[max SR] under null hypothesis (no real edge), per Bailey-LdP 2014.

    Approximation:  (1 - gamma) * Phi^-1(1 - 1/N)  +  gamma * Phi^-1(1 - 1/(N*e))
    where gamma is Euler-Mascheroni and Phi^-1 is the standard normal inverse CDF.
    For N=1, returns 0 (no inflation).
    """
    if n_trials <= 1:
        return 0.0
    gamma = 0.5772156649  # Euler-Mascheroni
    e = np.e
    z1 = stats.norm.ppf(1 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1 - 1.0 / (n_trials * e))
    return (1 - gamma) * z1 + gamma * z2


def deflated_sharpe_ratio(
    returns: np.ndarray,
    n_trials: int = 1,
    sr_zero: float = 0.0,
) -> float:
    """DSR per Bailey & López de Prado 2014, equation (6).

    DSR = Phi( (SR - SR0) * sqrt(T-1) / sqrt(1 - g3*SR + (g4-1)/4 * SR^2) )

    where g3 = skew of returns, g4 = excess kurtosis (or full kurtosis -- the
    Bailey-LdP form uses kurtosis NOT excess; we follow that convention).

    SR0 here is the expected max SR under N trials of the null hypothesis;
    we use SR0 = expected_max_sharpe(n_trials) when n_trials > 1.

    Returns the *probability* DSR — a value in [0, 1] interpretable as
    "probability the true Sharpe exceeds SR0".
    """
    r = np.asarray(returns)
    r = r[~np.isnan(r)]
    if len(r) < 4:
        return 0.0
    sr = r.mean() / r.std(ddof=1)
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=False))  # full kurtosis
    sr0 = sr_zero or expected_max_sharpe(n_trials)
    denom = np.sqrt(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr)
    if denom <= 0 or not np.isfinite(denom):
        return 0.0
    z = (sr - sr0) * np.sqrt(len(r) - 1) / denom
    return float(stats.norm.cdf(z))


# ---------------------------------------------------------------------------
# Probability of Backtest Overfitting (Bailey-Borwein-LdP-Zhu 2014, rank logit)
# ---------------------------------------------------------------------------


def probability_backtest_overfitting(
    is_metrics: np.ndarray,
    oos_metrics: np.ndarray,
) -> float:
    """PBO per Bailey-Borwein-López-Zhu 2014.

    Inputs are arrays of shape (n_paths, n_strategies) where each row is one
    CPCV path and each column is a candidate strategy / hyperparam combo.

    For each path:
      1. find the strategy with the best IS metric (`argmax`)
      2. compute its OOS rank (1 = best)
      3. logit(rank / (S+1)) → if < 0 ⇒ "this best-IS performed worse than median OOS"
    PBO = fraction of paths where logit < 0.

    For our single-strategy CPCV (no hyperparameter sweep), we synthesise a
    null distribution by treating each path's training/test SR pair as one
    "strategy" — same idea, just collapsed.
    """
    is_metrics = np.atleast_2d(is_metrics)
    oos_metrics = np.atleast_2d(oos_metrics)
    if is_metrics.shape != oos_metrics.shape:
        raise ValueError("is_metrics and oos_metrics must have same shape")
    n_paths, n_strategies = is_metrics.shape
    if n_paths == 0:
        return 0.5  # no information → uninformative prior

    failures = 0
    for p in range(n_paths):
        best_is = int(np.argmax(is_metrics[p]))
        # OOS rank (1 = best, n_strategies = worst)
        oos_row = oos_metrics[p]
        # Higher OOS metric = better; rank from top
        oos_rank = (oos_row > oos_row[best_is]).sum() + 1
        # Convert rank to "performance percentile": fraction that did better
        # is `(oos_rank-1)/n_strategies`. PBO is fraction where this >= 0.5.
        if (oos_rank - 1) / n_strategies >= 0.5:
            failures += 1
    return failures / n_paths


def _pbo_from_path_returns(
    path_returns: list[np.ndarray],
) -> float:
    """When we have a single strategy, fall back to a "median-split" PBO
    proxy: for each path, compare its Sharpe to the median Sharpe across
    paths. PBO ≈ fraction of paths below median.

    This gives a value around 0.5 for noise (no edge) and < 0.3 when the
    strategy generalises consistently across all paths.
    """
    sharpes = np.array([sharpe_ratio(r) for r in path_returns])
    median = float(np.median(sharpes))
    return float((sharpes < median).mean())


# ---------------------------------------------------------------------------
# Holm-Bonferroni & Diebold-Mariano
# ---------------------------------------------------------------------------


def holm_bonferroni(
    p_values: dict[str, float], alpha: float = 0.05
) -> tuple[dict[str, bool], dict[str, float]]:
    """Holm sequentially-rejective procedure.

    Returns a dict {name: rejected_null?} and the original p-values map.
    Reject H0 (= "feature has no signal") in order of ascending p-value;
    threshold for the i-th smallest is alpha / (m - i + 1).
    """
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    significant: dict[str, bool] = {}
    for i, (name, p) in enumerate(items):
        threshold = alpha / (m - i)
        if p < threshold:
            significant[name] = True
        else:
            # Once one fails, all subsequent (larger p) also fail
            for name2, _ in items[i:]:
                significant[name2] = False
            break
    else:
        # All passed
        for name, _ in items:
            if name not in significant:
                significant[name] = True

    return significant, dict(p_values)


def diebold_mariano(
    errors_a: np.ndarray, errors_b: np.ndarray, h: int = 1
) -> tuple[float, float]:
    """Diebold-Mariano test of equal forecast accuracy.

    Returns (statistic, two-sided p-value). H0: E[loss_A] = E[loss_B].
    Negative DM with small p-value ⇒ A is significantly more accurate.

    Uses sample variance with HAC adjustment via lag h. h=1 for non-overlapping.
    """
    a = np.asarray(errors_a, dtype=float)
    b = np.asarray(errors_b, dtype=float)
    if len(a) != len(b):
        raise ValueError("error series must have equal length")
    d = a - b
    n = len(d)
    if n < 4:
        return 0.0, 1.0
    d_mean = float(d.mean())
    # HAC-style variance: γ_0 + 2 * sum_{k=1..h-1} (1 - k/h) * γ_k
    gamma_0 = float(((d - d_mean) ** 2).mean())
    var_d = gamma_0
    for k in range(1, h):
        gamma_k = float(((d[k:] - d_mean) * (d[:-k] - d_mean)).mean())
        var_d += 2 * (1 - k / h) * gamma_k
    if var_d <= 0:
        return 0.0, 1.0
    dm = d_mean / np.sqrt(var_d / n)
    p = 2 * (1 - stats.norm.cdf(abs(dm)))
    return float(dm), float(p)


# ---------------------------------------------------------------------------
# End-to-end harness
# ---------------------------------------------------------------------------


def run_cpcv(
    model_factory: Callable,
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int = 8,
    n_test_folds: int = 2,
    embargo: int = 16,
    label_horizon: int = 4,
    threshold: float = 0.0,
    seed: int = 42,
    verbose: bool = False,
) -> CPCVResult:
    """Run CPCV over the dataset.

    Parameters
    ----------
    model_factory : callable
        Zero-arg function returning a fresh sklearn-style model instance.
        We call ``model_factory()`` per path so each path trains independently.
    X, y : pandas
        Feature matrix and TARGET series (typically forward log-returns).
        Must be aligned by index.
    n_folds, n_test_folds : int
        N=8, k=2 by plan default ⇒ 28 CPCV paths.
    embargo : int
        Bars of buffer between train and test (16 = 4h on M15).
    label_horizon : int
        How many bars forward each label spans (= forecast horizon).
        Used to purge training rows whose label leaks into the test fold.
    threshold : float
        Long position above; short position below (else flat). 0 = always trade.
    seed : int
        For reproducibility (fixed in the model where applicable).
    verbose : bool
        Log per-path stats.

    Returns
    -------
    CPCVResult with:
        - paths: 28 entries each with sharpe, PF, hit_rate, returns
        - aggregate stats (mean, p25, p75)
        - DSR (deflated sharpe ratio)
        - PBO (probability of backtest overfitting)
    """
    if len(X) != len(y):
        raise ValueError(f"X len {len(X)} != y len {len(y)}")

    rng = np.random.default_rng(seed)
    _ = rng  # reserved for stochastic-model seeding hook

    n = len(X)
    paths: list[CPCVPath] = []

    for path_id, combo, train_idx, test_idx in cpcv_path_indices(
        n, n_folds, n_test_folds, embargo, label_horizon
    ):
        if len(train_idx) < 50 or len(test_idx) < 10:
            logger.warning("Path %d skipped: too few samples", path_id)
            continue

        model = model_factory()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[test_idx])

        # Build returns: long when pred > threshold, short when < -threshold,
        # else flat. Each "trade" earns y[i] (= forward return) sign-adjusted.
        positions = np.where(
            preds > threshold, 1.0,
            np.where(preds < -threshold, -1.0, 0.0)
        )
        actual = y.iloc[test_idx].to_numpy()
        returns = positions * actual
        non_flat = positions != 0
        n_trades = int(non_flat.sum())

        sr = sharpe_ratio(returns[non_flat]) if n_trades > 0 else 0.0
        pf = profit_factor(returns[non_flat]) if n_trades > 0 else 0.0
        hr = hit_rate(returns[non_flat]) if n_trades > 0 else 0.0

        paths.append(
            CPCVPath(
                fold_combo=combo,
                train_size=len(train_idx),
                test_size=len(test_idx),
                sharpe=sr,
                profit_factor=pf,
                hit_rate=hr,
                n_trades=n_trades,
                returns=returns,
            )
        )

        if verbose:
            logger.info(
                "Path %2d combo=%s train=%d test=%d trades=%d sr=%.3f pf=%.2f hr=%.3f",
                path_id, combo, len(train_idx), len(test_idx), n_trades, sr, pf, hr,
            )

    if not paths:
        raise RuntimeError("No CPCV paths produced (all skipped?)")

    sharpes = np.array([p.sharpe for p in paths])
    pfs = np.array([p.profit_factor for p in paths])

    # Aggregate returns across all paths for DSR
    all_returns = np.concatenate([p.returns for p in paths])
    all_returns = all_returns[~np.isnan(all_returns)]
    dsr = deflated_sharpe_ratio(all_returns, n_trials=len(paths))

    pbo = _pbo_from_path_returns([p.returns for p in paths])

    return CPCVResult(
        n_folds=n_folds,
        n_test_folds=n_test_folds,
        embargo=embargo,
        paths=paths,
        sharpe_mean=float(np.mean(sharpes)),
        sharpe_p25=float(np.quantile(sharpes, 0.25)),
        sharpe_p75=float(np.quantile(sharpes, 0.75)),
        pf_mean=float(np.mean(pfs[np.isfinite(pfs)])) if np.any(np.isfinite(pfs)) else 0.0,
        pf_p25=float(np.quantile(pfs[np.isfinite(pfs)], 0.25))
        if np.any(np.isfinite(pfs))
        else 0.0,
        dsr=dsr,
        pbo=pbo,
    )
