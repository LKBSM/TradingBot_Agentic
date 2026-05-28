"""Coverage tests for ``src/intelligence/scoring/logistic_l1.py``.

Sprint 1 — S1.6 — push scoring/ coverage closer to the 85 % gate by
exercising LogisticL1Scorer end-to-end on synthetic data.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.intelligence.scoring.logistic_l1 import (
    DEFAULT_COMPONENT_NAMES,
    LogisticL1Scorer,
)


def _synthetic_xy(n: int = 200, seed: int = 42):
    rng = np.random.default_rng(seed)
    # 8 features matching the default component names
    X = rng.uniform(0.0, 1.0, size=(n, len(DEFAULT_COMPONENT_NAMES)))
    # Outcome correlated with sum of features 0+4 (smc_structure + regime)
    logit = (X[:, 0] + X[:, 4]) * 2.0 + rng.normal(0, 0.3, n)
    y = (logit > np.median(logit)).astype(int)
    return X, y


def test_fit_then_predict_proba_shape():
    X, y = _synthetic_xy()
    scorer = LogisticL1Scorer().fit(X, y)
    proba = scorer.predict_proba(X[:5])
    assert proba.shape == (5, 2)
    # Probabilities sum to 1
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(5), atol=1e-6)


def test_predict_p_win_returns_class_1_probabilities():
    X, y = _synthetic_xy()
    scorer = LogisticL1Scorer().fit(X, y)
    p_win = scorer.predict_p_win(X[:5])
    assert p_win.shape == (5,)
    assert ((p_win >= 0.0) & (p_win <= 1.0)).all()


def test_predict_on_unfitted_raises():
    scorer = LogisticL1Scorer()
    with pytest.raises(RuntimeError):
        scorer.predict_proba(np.zeros((1, 8)))


def test_coefficients_unfitted_raises():
    with pytest.raises(RuntimeError):
        LogisticL1Scorer().coefficients()


def test_coefficients_returns_dict_keyed_by_component_names():
    X, y = _synthetic_xy()
    scorer = LogisticL1Scorer().fit(X, y)
    coefs = scorer.coefficients()
    assert isinstance(coefs, dict)
    assert set(coefs.keys()) == set(DEFAULT_COMPONENT_NAMES)


def test_non_zero_components_filters_l1_zeros():
    X, y = _synthetic_xy()
    # Very strong L1 → most coefficients pinned to 0
    scorer = LogisticL1Scorer(C=0.01).fit(X, y)
    nz = scorer.non_zero_components()
    assert isinstance(nz, list)
    # All listed components must be a subset of the full feature set
    assert set(nz).issubset(set(DEFAULT_COMPONENT_NAMES))


def test_fit_rejects_1d_input():
    scorer = LogisticL1Scorer()
    with pytest.raises(ValueError):
        scorer.fit(np.zeros(10), np.zeros(10))


def test_predict_proba_accepts_1d_input():
    X, y = _synthetic_xy()
    scorer = LogisticL1Scorer().fit(X, y)
    # 1D feature vector — wrapper reshapes to (1, 8)
    p = scorer.predict_proba(X[0])
    assert p.shape == (1, 2)


def test_misaligned_feature_count_warns_but_fits(caplog):
    X = np.random.default_rng(0).uniform(size=(50, 4))
    y = np.random.default_rng(1).integers(0, 2, size=50)
    with caplog.at_level("WARNING"):
        scorer = LogisticL1Scorer().fit(X, y)
    # Logged the misalignment warning
    assert any("alignment may be off" in m for m in caplog.messages)
    # But it still produced a fitted model
    p = scorer.predict_proba(X[:3])
    assert p.shape == (3, 2)
