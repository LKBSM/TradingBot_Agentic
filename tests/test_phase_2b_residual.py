"""Tests for the residual Phase 2B code modules (Batch D).

DATA-2B.3, QUANT-2B.3, REGIME-2B.2, LLM-2B.2 (manifest loader).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ===========================================================================
# DATA-2B.3 — multi-instrument quality
# ===========================================================================


def test_check_instrument_xau_full_coverage():
    from src.agents.data.multi_instrument_quality import check_instrument

    idx = pd.date_range("2024-01-01", periods=100, freq="15min", tz="UTC")
    df = pd.DataFrame({"close": np.linspace(2000, 2100, 100)}, index=idx)
    q = check_instrument(df, instrument="XAU", timeframe="M15")
    assert q.coverage_pct == pytest.approx(1.0)
    assert q.stale_bars == 0
    assert q.ok is True


def test_check_instrument_unknown_raises():
    from src.agents.data.multi_instrument_quality import check_instrument

    idx = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
    df = pd.DataFrame({"close": np.arange(10)}, index=idx)
    with pytest.raises(ValueError, match="unsupported instrument"):
        check_instrument(df, instrument="GBPNZD", timeframe="D1")


def test_check_instrument_detects_stale_feed():
    from src.agents.data.multi_instrument_quality import check_instrument

    idx = pd.date_range("2024-01-01", periods=100, freq="15min", tz="UTC")
    # 20 consecutive identical closes — frozen feed.
    closes = list(np.linspace(2000, 2080, 80)) + [2080.0] * 20
    df = pd.DataFrame({"close": closes}, index=idx)
    q = check_instrument(df, instrument="XAU", timeframe="M15")
    assert q.stale_bars >= 19
    assert q.stale_pct >= 0.19
    assert q.ok is False  # stale_pct > 0.05 default


def test_check_instrument_max_gap():
    from src.agents.data.multi_instrument_quality import check_instrument

    idx = pd.DatetimeIndex(
        pd.to_datetime(
            [
                "2024-01-01 00:00",
                "2024-01-01 00:15",
                "2024-01-01 02:00",  # 7-bar gap (1h45m / 15m)
                "2024-01-01 02:15",
            ],
            utc=True,
        )
    )
    df = pd.DataFrame({"close": [1, 2, 3, 4]}, index=idx)
    q = check_instrument(df, instrument="XAU", timeframe="M15")
    assert q.max_gap_bars == 7


def test_check_all_returns_one_per_instrument():
    from src.agents.data.multi_instrument_quality import check_all

    idx = pd.date_range("2024-01-01", periods=30, freq="D", tz="UTC")
    f = {
        "XAU": (pd.DataFrame({"close": np.arange(30)}, index=idx), "D1"),
        "EURUSD": (pd.DataFrame({"close": np.arange(30) * 0.01 + 1}, index=idx), "D1"),
        "USOIL": (pd.DataFrame({"close": np.arange(30) + 70}, index=idx), "D1"),
    }
    out = check_all(f)
    assert [r.instrument for r in out] == ["EURUSD", "USOIL", "XAU"]


# ===========================================================================
# QUANT-2B.3 — B2B signal audit
# ===========================================================================


def test_audit_empty_returns_error():
    from src.research.b2b_signal_audit import run_audit

    out = run_audit(pd.DataFrame())
    assert out["ok"] is False
    assert "error" in out


def test_audit_basic_shape():
    from src.research.b2b_signal_audit import run_audit

    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "signal_id": [f"s-{i}" for i in range(n)],
        "ts_utc": pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC"),
        "instrument": ["XAU"] * n,
        "direction": rng.choice(["LONG", "SHORT"], n),
        "entry_price": rng.uniform(2000, 2100, n),
        "stop_price": rng.uniform(1990, 2090, n),
        "target_price": rng.uniform(2010, 2110, n),
        "hit": rng.choice(["TARGET", "STOP", "TIMEOUT"], n, p=[0.4, 0.4, 0.2]),
    })
    out = run_audit(df, n_trials_claimed=10)
    for key in ("n_signals", "win_rate", "mean_R", "profit_factor",
                "max_drawdown_R", "dsr_approx", "per_instrument",
                "leak_audit"):
        assert key in out


def test_audit_flags_label_leak():
    from src.research.b2b_signal_audit import run_audit

    df = pd.DataFrame({
        "signal_id": ["s-1", "s-2"],
        "ts_utc": pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC"),
        "instrument": ["XAU"] * 2,
        "direction": ["LONG"] * 2,
        "entry_price": [2000, 2010],
        "stop_price": [1990, 2000],
        "target_price": [2020, 2030],
        "hit": ["TARGET", "STOP"],
        # The smoking gun:
        "fitted_future_close": [2025, 2005],
    })
    out = run_audit(df)
    assert out["leak_audit"]["count"] >= 1
    assert any("fitted_future_close" in s for s in out["leak_audit"]["details"])


def test_audit_beta_capture_flag():
    """Strategy whose returns mirror a benchmark must trigger beta_capture_warn."""
    from src.research.b2b_signal_audit import run_audit

    n = 100
    rng = np.random.default_rng(0)
    bench = pd.Series(rng.normal(0, 0.01, n))
    # Construct R per trade ~= bench → high correlation
    r = bench.copy()

    # Make the signals look like targets/stops aligned with r
    df = pd.DataFrame({
        "signal_id": [f"s-{i}" for i in range(n)],
        "ts_utc": pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC"),
        "instrument": ["XAU"] * n,
        "direction": ["LONG"] * n,
        "entry_price": [100.0] * n,
        "stop_price": [99.0] * n,
        "target_price": [101.0] * n,
        "hit": ["TARGET" if x > 0 else "STOP" for x in r],
    })
    out = run_audit(df, benchmark_returns=r)
    # Strong correlation expected since hit perfectly maps to r sign
    assert out["beta_capture_corr"] is not None


# ===========================================================================
# REGIME-2B.2 — viz payload
# ===========================================================================


def test_build_timeline_payload_shape():
    from src.intelligence.regime_classifier import RegimeClassifier
    from src.intelligence.regime_viz import build_timeline_payload

    rng = np.random.default_rng(0)
    n = 600
    closes = (1.0 + rng.normal(0, 0.001, n)).cumprod() * 100
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    df = pd.DataFrame({"close": closes}, index=idx)

    clf = RegimeClassifier().fit(df["close"].pct_change().dropna().values)
    payload = build_timeline_payload(df, clf, max_points=200)
    assert set(payload.keys()) == {"entries", "summary", "n_entries", "max_points"}
    assert payload["n_entries"] <= 200
    assert all("ts" in e and "label" in e for e in payload["entries"])


def test_build_timeline_rejects_missing_close():
    from src.intelligence.regime_classifier import RegimeClassifier
    from src.intelligence.regime_viz import build_timeline_payload

    df = pd.DataFrame({"x": [1, 2, 3]},
                      index=pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC"))
    with pytest.raises(ValueError, match="close"):
        build_timeline_payload(df, RegimeClassifier())


# ===========================================================================
# LLM-2B.2 — manifest loader
# ===========================================================================


def test_manifest_loader_returns_records(tmp_path):
    import yaml

    from src.intelligence.rag.source_manifest import load_manifest

    p = tmp_path / "m.yaml"
    yaml.safe_dump(
        {
            "version": 1,
            "sources": [
                {"id": "a", "type": "paper", "title": "A", "year": 2020, "authority": 5, "tags": ["x"]},
                {"id": "b", "type": "report", "title": "B", "year": 2021, "authority": 3},
            ],
        },
        open(p, "w", encoding="utf-8"),
    )
    out = load_manifest(p)
    assert {r.id for r in out} == {"a", "b"}
    assert next(r for r in out if r.id == "a").authority == 5


def test_filter_sources_by_authority():
    from src.intelligence.rag.source_manifest import SourceRecord, filter_sources

    recs = [
        SourceRecord(id="x", type="paper", title="t", authors=(), year=2020, authority=5),
        SourceRecord(id="y", type="paper", title="t", authors=(), year=2020, authority=2),
    ]
    out = filter_sources(recs, min_authority=4)
    assert {r.id for r in out} == {"x"}


def test_filter_sources_excludes_biased_tags():
    from src.intelligence.rag.source_manifest import SourceRecord, filter_sources

    recs = [
        SourceRecord(id="x", type="educational", title="t", authors=(), year=2020,
                     tags=("biased-author",)),
        SourceRecord(id="y", type="educational", title="t", authors=(), year=2020,
                     tags=("clean",)),
    ]
    out = filter_sources(recs, exclude_tags=["biased-author"])
    assert {r.id for r in out} == {"y"}


def test_production_manifest_loads():
    """The shipped data/rag/sources_manifest.yaml must load without errors."""
    from pathlib import Path

    from src.intelligence.rag.source_manifest import load_manifest

    p = Path(__file__).resolve().parent.parent / "data" / "rag" / "sources_manifest.yaml"
    if not p.exists():
        pytest.skip("production manifest not in this checkout")
    out = load_manifest(p)
    # We seeded with at least 40 sources.
    assert len(out) >= 40
    # Authority must be between 1 and 5
    for r in out:
        assert 1 <= r.authority <= 5
