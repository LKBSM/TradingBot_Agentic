"""Tests for scripts/train_calibrated_conviction.py — end-to-end smoke."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Module under test (script lives in scripts/)
sys.path.insert(0, str(ROOT / "scripts"))
import train_calibrated_conviction as train_mod  # noqa: E402

from src.intelligence.scoring.calibrated_conviction import (  # noqa: E402
    CalibratedConvictionPipeline,
)


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------


def test_synthetic_replay_shape():
    df = train_mod.synthetic_replay(n_signals=200, seed=1)
    assert len(df) == 200
    for col in train_mod.FEATURE_COLUMNS:
        assert col in df.columns
    assert "outcome" in df.columns
    assert df["outcome"].isin([0, 1]).all()
    assert "pnl_r_multiple" in df.columns


def test_synthetic_replay_has_some_signal():
    """Synthetic features should be at least weakly correlated with outcome."""
    df = train_mod.synthetic_replay(n_signals=500, seed=42)
    # The sum of features should correlate with outcome
    feature_sum = df[list(train_mod.FEATURE_COLUMNS)].sum(axis=1)
    corr = float(np.corrcoef(feature_sum, df["outcome"])[0, 1])
    assert corr > 0.1  # ≥ 10% Pearson


# ---------------------------------------------------------------------------
# Chronological split
# ---------------------------------------------------------------------------


def test_split_chronological_proportions():
    df = train_mod.synthetic_replay(n_signals=1000, seed=0)
    train, val = train_mod.split_chronological(df, val_fraction=0.30)
    assert len(train) == 700
    assert len(val) == 300


def test_split_chronological_rejects_bad_fraction():
    df = train_mod.synthetic_replay(n_signals=100)
    with pytest.raises(ValueError):
        train_mod.split_chronological(df, val_fraction=0.04)
    with pytest.raises(ValueError):
        train_mod.split_chronological(df, val_fraction=0.6)


# ---------------------------------------------------------------------------
# Full training pipeline
# ---------------------------------------------------------------------------


def test_train_returns_fitted_pipeline():
    df = train_mod.synthetic_replay(n_signals=500, seed=7)
    pipeline = train_mod.train(df, val_fraction=0.30, alpha=0.10,
                                lgbm_kwargs={"n_estimators": 50})
    assert isinstance(pipeline, CalibratedConvictionPipeline)
    # All three stages should be fitted
    assert pipeline.lgbm is not None
    assert pipeline.isotonic is not None
    assert pipeline.conformal is not None


def test_train_then_score_one_returns_non_fallback():
    df = train_mod.synthetic_replay(n_signals=500, seed=7)
    pipeline = train_mod.train(df, val_fraction=0.30, alpha=0.10,
                                lgbm_kwargs={"n_estimators": 50})
    features = df[list(train_mod.FEATURE_COLUMNS)].iloc[-1].to_numpy(dtype=float)
    cc = pipeline.score_one(features)
    assert cc.is_fallback is False
    assert 0 <= cc.conviction_0_100 <= 100


def test_train_rejects_too_little_data():
    df = train_mod.synthetic_replay(n_signals=60)
    # 60 * 0.3 = 18 val rows < 30 minimum
    with pytest.raises(ValueError, match="Insufficient data"):
        train_mod.train(df, val_fraction=0.30, alpha=0.10)


# ---------------------------------------------------------------------------
# Round-trip save / load
# ---------------------------------------------------------------------------


def test_save_and_load_round_trip(tmp_path):
    df = train_mod.synthetic_replay(n_signals=400, seed=11)
    pipeline = train_mod.train(df, val_fraction=0.30,
                                lgbm_kwargs={"n_estimators": 50})
    out = tmp_path / "pipe.pkl"
    train_mod.save_pipeline(pipeline, out)
    assert out.exists()

    loaded = train_mod.load_calibrated_pipeline(out)
    assert isinstance(loaded, CalibratedConvictionPipeline)

    # Loaded pipeline produces same scoring for same features
    features = df[list(train_mod.FEATURE_COLUMNS)].iloc[0].to_numpy(dtype=float)
    cc1 = pipeline.score_one(features)
    cc2 = loaded.score_one(features)
    assert cc1.conviction_0_100 == cc2.conviction_0_100
    assert cc1.p_win_raw == pytest.approx(cc2.p_win_raw, abs=1e-9)


def test_load_missing_file_returns_unfitted_fallback(tmp_path):
    loaded = train_mod.load_calibrated_pipeline(tmp_path / "nonexistent.pkl")
    assert isinstance(loaded, CalibratedConvictionPipeline)
    cc = loaded.score_one(np.zeros(8, dtype=float))
    assert cc.is_fallback is True


def test_load_corrupted_file_returns_unfitted_fallback(tmp_path):
    bad = tmp_path / "corrupted.pkl"
    bad.write_bytes(b"not a pickle")
    loaded = train_mod.load_calibrated_pipeline(bad)
    cc = loaded.score_one(np.zeros(8, dtype=float))
    assert cc.is_fallback is True


# ---------------------------------------------------------------------------
# Replay CSV loader
# ---------------------------------------------------------------------------


def test_load_replay_validates_outcome_column(tmp_path):
    csv = tmp_path / "bad.csv"
    pd.DataFrame({"score_bos": [1, 2, 3]}).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="outcome"):
        train_mod.load_replay(csv)


def test_load_replay_fills_missing_feature_columns(tmp_path, caplog):
    """Missing score_* columns are tolerated (filled with zeros) and logged."""
    csv = tmp_path / "partial.csv"
    pd.DataFrame({
        "score_bos": [1, 2, 3],
        "outcome": [0, 1, 1],
    }).to_csv(csv, index=False)
    with caplog.at_level("WARNING"):
        df = train_mod.load_replay(csv)
    for col in train_mod.FEATURE_COLUMNS:
        assert col in df.columns


def test_load_replay_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        train_mod.load_replay(tmp_path / "missing.csv")
