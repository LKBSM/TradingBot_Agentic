"""Tests for the REGIME-2B.1 3-state HMM regime classifier."""

from __future__ import annotations

import numpy as np
import pytest

from src.intelligence.regime_classifier import (
    LABEL_HIGH_VOL_STRESS,
    LABEL_LOW_VOL_RANGING,
    LABEL_LOW_VOL_TRENDING,
    RegimeClassifier,
)


def _synthetic_three_regimes(rng_seed=0, n_per=500):
    """Generate returns from three known regimes — variance + drift well-separated."""
    rng = np.random.default_rng(rng_seed)
    trending = rng.normal(loc=0.002, scale=0.001, size=n_per)   # small vol, drift+
    ranging  = rng.normal(loc=0.0,   scale=0.001, size=n_per)   # small vol, zero drift
    stress   = rng.normal(loc=0.0,   scale=0.01,  size=n_per)   # large vol
    return np.concatenate([trending, ranging, stress])


# ---------------------------------------------------------------------------
# Fit + predict basics
# ---------------------------------------------------------------------------


def test_fit_then_predict_returns_correct_length():
    r = _synthetic_three_regimes()
    c = RegimeClassifier().fit(r)
    states = c.predict(r)
    assert len(states) == len(r)
    assert set(states).issubset({0, 1, 2})


def test_fit_requires_minimum_samples():
    with pytest.raises(ValueError):
        RegimeClassifier().fit(np.zeros(10))


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        RegimeClassifier().predict(np.zeros(50))


# ---------------------------------------------------------------------------
# Label semantics — labels are deterministic by content, not by EM init
# ---------------------------------------------------------------------------


def test_three_labels_present():
    c = RegimeClassifier().fit(_synthetic_three_regimes())
    labels = set(c.state_labels().values())
    assert labels == {
        LABEL_LOW_VOL_TRENDING,
        LABEL_LOW_VOL_RANGING,
        LABEL_HIGH_VOL_STRESS,
    }


def test_stress_state_classifies_high_vol_samples():
    """A purely-stress series should map mostly to high_vol_stress."""
    rng = np.random.default_rng(0)
    stress_only = rng.normal(loc=0.0, scale=0.01, size=500)
    full = _synthetic_three_regimes()
    c = RegimeClassifier().fit(full)
    preds = c.predict_with_confidence(stress_only)
    labels = [p.label for p in preds]
    n_stress = sum(1 for l in labels if l == LABEL_HIGH_VOL_STRESS)
    # We tolerate some noise from the HMM, but the stress label should dominate.
    assert n_stress / len(labels) > 0.7


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------


def test_predict_with_confidence_returns_valid_probs():
    c = RegimeClassifier().fit(_synthetic_three_regimes())
    preds = c.predict_with_confidence(_synthetic_three_regimes())
    for p in preds:
        assert 0.0 <= p.confidence <= 1.0
        assert p.label in {
            LABEL_LOW_VOL_TRENDING,
            LABEL_LOW_VOL_RANGING,
            LABEL_HIGH_VOL_STRESS,
        }


# ---------------------------------------------------------------------------
# Reproducibility — same seed, same labels
# ---------------------------------------------------------------------------


def test_same_seed_produces_same_label_mapping():
    r = _synthetic_three_regimes()
    a = RegimeClassifier(random_state=42).fit(r)
    b = RegimeClassifier(random_state=42).fit(r)
    assert a.state_labels() == b.state_labels()


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_save_load_roundtrip_preserves_predictions(tmp_path):
    r = _synthetic_three_regimes()
    c = RegimeClassifier().fit(r)
    preds_before = c.predict(r)

    p = tmp_path / "regime.pkl"
    c.save(p)
    c2 = RegimeClassifier.load(p)
    preds_after = c2.predict(r)

    assert np.array_equal(preds_before, preds_after)
    assert c.state_labels() == c2.state_labels()


def test_save_before_fit_raises():
    with pytest.raises(RuntimeError):
        RegimeClassifier().save("/tmp/nope")
