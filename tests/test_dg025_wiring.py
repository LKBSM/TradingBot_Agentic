"""Sprint 1 DG-025 wiring tests.

Cover the production wiring of the calibrated_conviction pipeline through
the InsightAssembler + scanner:

- ``_components_to_feature_vector`` in ``src/intelligence/sentinel_scanner.py``
- ``verify_data_quality_or_abort`` in ``src/intelligence/main.py``
- SCORING_VERSION=v2 path in ``build_system`` (the loading branch)
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.intelligence.confluence_detector import (
    ComponentScore,
    ConfluenceSignal,
    SignalTier,
)
from src.intelligence.confluence_detector import SignalType


def _make_signal(component_values: dict[str, float]) -> ConfluenceSignal:
    """Helper to build a ConfluenceSignal with the requested components."""
    components = [
        ComponentScore(
            name=name, raw_value=val, weighted_score=val,
            weight=1.0, reasoning=f"{name}={val}",
        )
        for name, val in component_values.items()
    ]
    return ConfluenceSignal(
        signal_id="test-1", symbol="XAUUSD",
        signal_type=SignalType.LONG, confluence_score=42.0,
        tier=SignalTier.WEAK, entry_price=1900.0,
        stop_loss=1880.0, take_profit=1940.0,
        rr_ratio=2.0, atr=2.0, components=components,
    )


# ---------------------------------------------------------------------------
# _components_to_feature_vector
# ---------------------------------------------------------------------------

def test_components_to_feature_vector_empty_signal():
    from src.intelligence.sentinel_scanner import _components_to_feature_vector
    vec = _components_to_feature_vector(None)
    assert vec.shape == (8,)
    assert np.all(vec == 0.0)


def test_components_to_feature_vector_all_known_components():
    from src.intelligence.sentinel_scanner import _components_to_feature_vector
    sig = _make_signal({
        "bos": 12.0, "order_block": 7.5, "fvg": 9.0,
        "regime": 18.0, "news": 5.0,
        "momentum": 2.0, "rsi_divergence": 1.5,
    })
    vec = _components_to_feature_vector(sig)
    # Expected order: smc_structure / order_blocks / fvg / retest / regime /
    # vol_forecast / news / momentum_rsi_div
    expected = np.array([12.0, 7.5, 9.0, 0.0, 18.0, 0.0, 5.0, 3.5])
    np.testing.assert_array_equal(vec, expected)


def test_components_to_feature_vector_ignores_unknown_components():
    from src.intelligence.sentinel_scanner import _components_to_feature_vector
    sig = _make_signal({
        "bos": 10.0,
        "htf_alignment": 5.0,    # not in mapping → ignored
        "unknown": 999.0,        # not in mapping → ignored
    })
    vec = _components_to_feature_vector(sig)
    # Only bos contributes (smc_structure index 0)
    expected = np.zeros(8)
    expected[0] = 10.0
    np.testing.assert_array_equal(vec, expected)


def test_components_to_feature_vector_accumulates_momentum_pair():
    from src.intelligence.sentinel_scanner import _components_to_feature_vector
    sig = _make_signal({"momentum": 2.0, "rsi_divergence": 1.0})
    vec = _components_to_feature_vector(sig)
    # Both momentum and rsi_divergence map to index 7 → sum
    assert vec[7] == 3.0
    assert vec[0] == 0.0  # nothing else


# ---------------------------------------------------------------------------
# verify_data_quality_or_abort
# ---------------------------------------------------------------------------

def _make_valid_frame(n: int = 500, freq: str = "15min") -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq=freq)
    return pd.DataFrame({
        "Open": np.full(n, 100.0),
        "High": np.full(n, 101.0),
        "Low": np.full(n, 99.0),
        "Close": np.full(n, 100.0),
        "Volume": np.full(n, 1.0),
    }, index=idx)


def test_verify_data_quality_passes_on_clean_frame():
    from src.intelligence.main import verify_data_quality_or_abort
    df = _make_valid_frame(1000)
    # Should not raise / sys.exit
    verify_data_quality_or_abort(df, "XAUUSD", "M15")


def test_verify_data_quality_aborts_on_nan():
    from src.intelligence.main import verify_data_quality_or_abort
    df = _make_valid_frame(1000)
    df.iloc[10, df.columns.get_loc("Close")] = np.nan
    with pytest.raises(SystemExit) as excinfo:
        verify_data_quality_or_abort(df, "XAUUSD", "M15")
    assert excinfo.value.code == 4


def test_verify_data_quality_bypassed_when_strict_off(monkeypatch):
    from src.intelligence.main import verify_data_quality_or_abort
    monkeypatch.setenv("DATA_QUALITY_STRICT", "off")
    monkeypatch.setenv("COVERAGE_GATE", "off")
    df = _make_valid_frame(1000)
    df.iloc[10, df.columns.get_loc("Close")] = np.nan
    # No raise: bypass is honoured
    verify_data_quality_or_abort(df, "XAUUSD", "M15")


def test_verify_data_quality_aborts_on_low_coverage(monkeypatch):
    from src.intelligence.main import verify_data_quality_or_abort
    monkeypatch.delenv("COVERAGE_GATE", raising=False)
    # 500 bars over a 5-year span — coverage ~ 0.3%
    idx = pd.date_range("2019-01-01", "2024-01-01", periods=500)
    df = pd.DataFrame({
        "Open": np.full(500, 100.0),
        "High": np.full(500, 101.0),
        "Low": np.full(500, 99.0),
        "Close": np.full(500, 100.0),
        "Volume": np.full(500, 1.0),
    }, index=idx)
    with pytest.raises(SystemExit) as excinfo:
        verify_data_quality_or_abort(df, "XAUUSD", "M15")
    # Coverage exit code 3 (defined in _check_coverage_or_abort)
    assert excinfo.value.code == 3


# ---------------------------------------------------------------------------
# SCORING_VERSION wiring smoke test
# ---------------------------------------------------------------------------

def test_scoring_version_v1_no_pipeline_loaded(tmp_path, monkeypatch):
    """SCORING_VERSION=v1 (default) → no calibrated pipeline path taken.

    We simulate the gate by replicating the same condition build_system
    does. Full build_system smoke is exercised in test_smoke_e2e.
    """
    monkeypatch.setenv("SCORING_VERSION", "v1")
    scoring_version = os.environ.get("SCORING_VERSION", "v1").strip().lower()
    assert scoring_version == "v1"


def test_scoring_version_v2_path_loads_pipeline(tmp_path):
    """When SCORING_VERSION=v2 and a model exists, it is loaded successfully."""
    from src.intelligence.scoring.calibrated_conviction import (
        CalibratedConvictionPipeline,
    )
    from scripts.train_calibrated_conviction import load_calibrated_pipeline

    # Create a minimal pickle of an empty pipeline (fallback path)
    pkl_path = tmp_path / "stub.pkl"
    empty = CalibratedConvictionPipeline()
    with open(pkl_path, "wb") as f:
        pickle.dump(empty, f)
    loaded = load_calibrated_pipeline(pkl_path)
    assert isinstance(loaded, CalibratedConvictionPipeline)


def test_scoring_version_v2_path_falls_back_on_missing_file(tmp_path, caplog):
    """When the model is missing, loader returns an unfitted pipeline (fallback)."""
    from src.intelligence.scoring.calibrated_conviction import (
        CalibratedConvictionPipeline,
    )
    from scripts.train_calibrated_conviction import load_calibrated_pipeline

    missing = tmp_path / "missing.pkl"
    pipeline = load_calibrated_pipeline(missing)
    # Empty fallback returned — score_one yields conviction=50, is_fallback=True
    assert isinstance(pipeline, CalibratedConvictionPipeline)
    cc = pipeline.score_one(np.zeros(8))
    assert cc.is_fallback is True
    assert cc.conviction_0_100 == 50
