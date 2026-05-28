"""Coverage tests for ``src/intelligence/scoring/lgbm_scoring_engine.py``.

Sprint 1 — S1.6 — exercise the run-time scoring engine end-to-end with a
mock regressor model. Goal: bring legacy-but-still-imported module
coverage above the 30 % floor.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.intelligence.scoring.lgbm_scoring_engine import (
    ConvictionReadout,
    LGBMScoringEngine,
    DEFAULT_LABEL_BINS,
    DEFAULT_LABELS,
)


class _MockModel:
    """Minimal stub matching the predict interface the engine expects."""
    feature_names = ("f0", "f1", "f2", "f3")

    def __init__(self, value: float = 0.01):
        self._value = value

    def predict(self, X):
        return np.array([self._value] * len(X))


# ---------------------------------------------------------------------------
# ConvictionReadout
# ---------------------------------------------------------------------------

def test_conviction_readout_to_dict_includes_all_fields():
    cr = ConvictionReadout(
        p_win=0.62, score_0_100=62.0, label="moderate",
        interval_lower=45.0, interval_upper=78.0, alpha=0.10,
        edge_claim=False,
    )
    d = cr.to_dict()
    assert d["p_win"] == 0.62
    assert d["conviction_0_100"] == 62.0
    assert d["conviction_label"] == "moderate"
    assert d["conviction_interval"]["lower"] == 45.0
    assert d["conviction_interval"]["upper"] == 78.0
    assert d["edge_claim"] is False


def test_conviction_readout_to_dict_handles_no_interval():
    cr = ConvictionReadout(
        p_win=0.5, score_0_100=50.0, label="weak",
        interval_lower=None, interval_upper=None,
    )
    d = cr.to_dict()
    assert d["conviction_interval"]["lower"] is None
    assert d["conviction_interval"]["upper"] is None


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def test_from_pickle_round_trip(tmp_path):
    pkl_path = tmp_path / "model.pkl"
    with pkl_path.open("wb") as f:
        pickle.dump(_MockModel(), f)
    engine = LGBMScoringEngine.from_pickle(pkl_path)
    assert engine.model is not None
    # feature_names lifted from the wrapped model
    assert engine.feature_names == ("f0", "f1", "f2", "f3")


def test_from_pickle_missing_file_raises(tmp_path):
    missing = tmp_path / "nope.pkl"
    with pytest.raises(FileNotFoundError):
        LGBMScoringEngine.from_pickle(missing)


# ---------------------------------------------------------------------------
# label_for
# ---------------------------------------------------------------------------

def test_label_for_with_default_bins():
    engine = LGBMScoringEngine(model=_MockModel())
    assert engine.label_for(10.0) == "weak"
    assert engine.label_for(50.0) == "moderate"
    assert engine.label_for(70.0) == "strong"
    assert engine.label_for(95.0) == "institutional"


# ---------------------------------------------------------------------------
# Score path
# ---------------------------------------------------------------------------

def test_score_dict_input_returns_readout():
    engine = LGBMScoringEngine(model=_MockModel(value=0.005))
    out = engine.score({"f0": 1.0, "f1": 2.0, "f2": 0.5, "f3": 0.0})
    assert isinstance(out, ConvictionReadout)
    assert 0.0 <= out.p_win <= 1.0
    assert 0.0 <= out.score_0_100 <= 100.0
    assert out.label in DEFAULT_LABELS


def test_score_array_input_returns_readout():
    engine = LGBMScoringEngine(model=_MockModel(value=0.02))
    out = engine.score(np.array([1.0, 2.0, 0.5, 0.0]))
    assert isinstance(out, ConvictionReadout)


def test_score_dataframe_input_returns_readout():
    engine = LGBMScoringEngine(model=_MockModel(value=-0.01))
    df = pd.DataFrame([{"f0": 1.0, "f1": 2.0, "f2": 0.0, "f3": 0.5}])
    out = engine.score(df)
    # Negative pred → P(win) close to 0 via sigmoid
    assert out.p_win < 0.5


def test_score_without_model_raises():
    engine = LGBMScoringEngine(model=None)
    with pytest.raises(RuntimeError):
        engine.score({"f0": 1.0})


# ---------------------------------------------------------------------------
# Bin calibration
# ---------------------------------------------------------------------------

def test_calibrate_bins_updates_thresholds():
    engine = LGBMScoringEngine(model=_MockModel())
    rng = np.random.default_rng(0)
    preds = rng.uniform(-0.01, 0.02, size=200)
    engine.calibrate_bins(preds)
    # Bins should have been replaced with quantile-derived boundaries
    assert engine.label_bins != DEFAULT_LABEL_BINS
    assert len(engine.label_bins) == 5
    # Strictly increasing-ish (within numerical noise)
    for a, b in zip(engine.label_bins[:-1], engine.label_bins[1:]):
        assert a <= b


def test_calibrate_bins_noop_when_too_few_samples():
    engine = LGBMScoringEngine(model=_MockModel())
    engine.calibrate_bins(np.array([0.01, 0.02, 0.03]))
    # Defaults still in place
    assert engine.label_bins == DEFAULT_LABEL_BINS
