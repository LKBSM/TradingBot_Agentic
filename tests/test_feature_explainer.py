"""Tests for the QUANT-2B.4 SHAP feature explainer."""

from __future__ import annotations

import numpy as np
import pytest

from src.intelligence.feature_explainer import FeatureDriver, FeatureExplainer


@pytest.fixture(scope="module")
def trained_booster():
    """Tiny LightGBM regressor on synthetic data.

    Two features matter (x0, x1), three are noise. The explainer
    should consistently rank x0+x1 above the noise features.
    """
    import lightgbm as lgb

    rng = np.random.default_rng(0)
    n = 500
    x0 = rng.uniform(-1, 1, n)
    x1 = rng.uniform(-1, 1, n)
    noise = rng.normal(0, 0.01, (n, 3))
    X = np.column_stack([x0, x1, noise])
    y = 2 * x0 + 3 * x1 + rng.normal(0, 0.1, n)
    train_data = lgb.Dataset(X, label=y, feature_name=["x0", "x1", "n0", "n1", "n2"])
    booster = lgb.train(
        {"objective": "regression", "verbosity": -1, "num_leaves": 7},
        train_data,
        num_boost_round=50,
    )
    return booster, X


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


def test_constructor_rejects_none():
    with pytest.raises(ValueError):
        FeatureExplainer(None)


def test_feature_names_derived_from_booster(trained_booster):
    booster, _ = trained_booster
    e = FeatureExplainer(booster)
    assert e._feature_names == ["x0", "x1", "n0", "n1", "n2"]


def test_feature_names_can_be_overridden(trained_booster):
    booster, _ = trained_booster
    custom = ["a", "b", "c", "d", "e"]
    e = FeatureExplainer(booster, feature_names=custom)
    assert e._feature_names == custom


# ---------------------------------------------------------------------------
# contributions shape
# ---------------------------------------------------------------------------


def test_contributions_returns_n_rows_by_features_plus_bias(trained_booster):
    booster, X = trained_booster
    e = FeatureExplainer(booster)
    c = e.contributions(X[:5])
    assert c.shape == (5, 6)  # 5 features + 1 bias column


def test_contributions_accepts_1d_input(trained_booster):
    booster, X = trained_booster
    e = FeatureExplainer(booster)
    c = e.contributions(X[0])
    # 1D input → still returns 2D (one row).
    assert c.ndim == 2 and c.shape[0] == 1


# ---------------------------------------------------------------------------
# top_drivers ranking
# ---------------------------------------------------------------------------


def test_top_drivers_returns_k_items(trained_booster):
    booster, X = trained_booster
    e = FeatureExplainer(booster)
    drivers = e.top_drivers(X[0:1], k=3)
    assert len(drivers) == 3
    assert all(isinstance(d, FeatureDriver) for d in drivers)


def test_top_drivers_are_x0_or_x1(trained_booster):
    """The known signal features should dominate top-2 across many rows."""
    booster, X = trained_booster
    e = FeatureExplainer(booster)
    hits = 0
    for i in range(50):
        drivers = e.top_drivers(X, k=2, row=i)
        top_names = {d.feature for d in drivers}
        if top_names.issubset({"x0", "x1"}):
            hits += 1
    # We tolerate the model picking up some noise occasionally on
    # individual rows, but the *vast* majority should be x0/x1.
    assert hits > 35


def test_direction_field_aligns_with_contribution_sign(trained_booster):
    booster, X = trained_booster
    e = FeatureExplainer(booster)
    for d in e.top_drivers(X, k=5, row=0):
        if d.contribution >= 0:
            assert d.direction == "up"
        else:
            assert d.direction == "down"


# ---------------------------------------------------------------------------
# explain_signal — JSON-friendly summary
# ---------------------------------------------------------------------------


def test_explain_signal_shape(trained_booster):
    booster, X = trained_booster
    e = FeatureExplainer(booster)
    out = e.explain_signal(X[0:1], k=3)
    assert set(out.keys()) == {"drivers", "bias", "feature_count", "k"}
    assert len(out["drivers"]) == 3
    assert out["feature_count"] == 5


def test_explain_signal_is_json_serialisable(trained_booster):
    import json

    booster, X = trained_booster
    e = FeatureExplainer(booster)
    json.dumps(e.explain_signal(X[0:1], k=3))


# ---------------------------------------------------------------------------
# Sklearn wrapper compatibility
# ---------------------------------------------------------------------------


def test_works_with_lgbm_sklearn_wrapper():
    import lightgbm as lgb

    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (300, 3))
    y = 2 * X[:, 0] + rng.normal(0, 0.1, 300)
    model = lgb.LGBMRegressor(n_estimators=30, num_leaves=7, verbosity=-1)
    model.fit(X, y)

    e = FeatureExplainer(model, feature_names=["a", "b", "c"])
    drivers = e.top_drivers(X[0:1], k=2)
    assert len(drivers) == 2
