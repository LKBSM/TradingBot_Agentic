"""Tests for src/research/a1_train.py (QUANT-1.3).

Smoke tests: StackedA1Model fit/predict + A1Verdict decision logic.
The full a1_verdict pipeline is tested via the smoke run (commit history
shows the actual KPI numbers from the production matrix).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.a1_train import A1Verdict, StackedA1Model


# ---------------------------------------------------------------------------
# StackedA1Model
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_xy():
    rng = np.random.default_rng(0)
    n = 1500
    X = pd.DataFrame(
        {
            # 7 price features
            "r_1": rng.normal(0, 0.001, n),
            "r_4": rng.normal(0, 0.002, n),
            "r_16": rng.normal(0, 0.004, n),
            "atr_14_pct": rng.uniform(0.001, 0.01, n),
            "rsi_14": rng.uniform(20, 80, n),
            "macd_signal_diff": rng.normal(0, 0.5, n),
            "atr_ratio_14_50": rng.uniform(0.8, 1.5, n),
            # 7 macro
            "dgs10": rng.uniform(2, 5, n),
            "breakeven_10y": rng.uniform(1, 3, n),
            "dtwexbgs": rng.uniform(95, 110, n),
            "vix": rng.uniform(10, 30, n),
            "t10y2y": rng.uniform(-1, 2, n),
            "cot_mm_net_pct_z52": rng.normal(0, 1, n),
            "cot_producer_net_z52": rng.normal(0, 1, n),
            # 5 calendar+intra
            "bar_minute_of_day": rng.integers(0, 1440, n),
            "dow": rng.integers(0, 7, n),
            "is_lunch_hour": rng.integers(0, 2, n),
            "min_to_next_red_news": rng.uniform(0, 1440, n),
            "min_since_last_red_news": rng.uniform(0, 1440, n),
        }
    )
    y = pd.Series(rng.normal(0, 0.005, n))
    return X, y


def test_stacked_model_fit_predict_shape(synthetic_xy):
    X, y = synthetic_xy
    model = StackedA1Model().fit(X, y)
    preds = model.predict(X.iloc[:100])
    assert preds.shape == (100,)
    assert np.all(np.isfinite(preds))


def test_stacked_model_uses_all_three_groups(synthetic_xy):
    X, y = synthetic_xy
    model = StackedA1Model().fit(X, y)
    assert set(model.level1_models.keys()) == {"price_only", "macro", "calendar_intra"}
    assert model.meta_model is not None


def test_stacked_model_feature_importance_per_group(synthetic_xy):
    X, y = synthetic_xy
    model = StackedA1Model().fit(X, y)
    fi = model.feature_importance_per_group()
    assert "price_only" in fi
    assert "macro" in fi
    assert "calendar_intra" in fi
    # Each group's features get a non-negative gain importance
    for group_dict in fi.values():
        for feat, imp in group_dict.items():
            assert imp >= 0


# ---------------------------------------------------------------------------
# A1Verdict decision logic
# ---------------------------------------------------------------------------


def test_a1_verdict_passes_only_when_all_thresholds_met():
    v_pass = A1Verdict(
        dsr=0.999,
        pbo=0.2,
        cpcv_pf_mean=1.30,
        cpcv_pf_p25=1.10,
        cpcv_sharpe_mean=2.0,
        holm_significant_count=5,
    )
    assert v_pass.passes
    assert v_pass.decision == "GO_2A"


def test_a1_verdict_fails_when_any_threshold_misses():
    # Only DSR fails
    v = A1Verdict(
        dsr=0.5,
        pbo=0.2,
        cpcv_pf_mean=1.30,
        cpcv_pf_p25=1.10,
        cpcv_sharpe_mean=2.0,
        holm_significant_count=5,
    )
    assert not v.passes


def test_a1_verdict_decision_2b_plus_for_marginal_case():
    """DSR=0.75, PBO=0.35, PF=1.10 → marginal but worth selective borrow."""
    v = A1Verdict(
        dsr=0.75,
        pbo=0.35,
        cpcv_pf_mean=1.10,
        cpcv_pf_p25=1.0,
        cpcv_sharpe_mean=0.8,
        holm_significant_count=2,
    )
    assert v.decision == "GO_2B_PLUS"


def test_a1_verdict_decision_2b_for_clear_failure():
    """Very low DSR, PF ≈ 1 → clear no-edge → 2B."""
    v = A1Verdict(
        dsr=0.0,
        pbo=0.5,
        cpcv_pf_mean=1.0,
        cpcv_pf_p25=0.95,
        cpcv_sharpe_mean=0.0,
        holm_significant_count=0,
    )
    assert v.decision == "GO_2B"


def test_a1_verdict_actual_phase1_outcome():
    """The actual numbers from the 2026-05-01 production run.

    Documents the real verdict in code so a future run can compare against
    the historical baseline.
    """
    v = A1Verdict(
        dsr=0.0,
        pbo=0.5,
        cpcv_pf_mean=1.008,
        cpcv_pf_p25=0.994,
        cpcv_sharpe_mean=0.384,
        holm_significant_count=19,
    )
    assert not v.passes
    assert v.decision == "GO_2B"
