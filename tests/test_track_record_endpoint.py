"""DG-142 — public track-record endpoint tests."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.routes.track_record import (
    DEFAULT_BOOTSTRAP_N,
    _bootstrap_pf_ci,
    _equity_curve,
    _hit_rate,
    _load_trades,
    _profit_factor,
)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def test_profit_factor_on_balanced_trades():
    pf = _profit_factor([1.0, 1.0, -1.0])
    assert pf == pytest.approx(2.0)


def test_profit_factor_zero_losses_returns_inf():
    pf = _profit_factor([1.0, 2.0])
    assert pf == float("inf")


def test_profit_factor_empty_returns_one():
    assert _profit_factor([]) == 1.0


def test_hit_rate_count_wins():
    assert _hit_rate([1.0, -1.0, 2.0]) == pytest.approx(2 / 3)


def test_equity_curve_cumulative():
    eq = _equity_curve([1.0, 1.0, -1.0])
    assert eq == [1.0, 2.0, 1.0]


def test_equity_curve_downsamples_large_input():
    pnls = [0.1] * 1000
    eq = _equity_curve(pnls, max_points=10)
    assert len(eq) <= 11  # max_points + last anchor
    # Down-sampled monotonic in this synthetic case
    assert eq[-1] == pytest.approx(100.0, rel=1e-3)


# ---------------------------------------------------------------------------
# Bootstrap CI is deterministic with a fixed seed
# ---------------------------------------------------------------------------

def test_bootstrap_pf_ci_is_deterministic_under_seed():
    pnls = [1.0, -0.5, 1.2, -0.8, 1.5, -0.3, 0.9]
    lo1, hi1 = _bootstrap_pf_ci(pnls, n_bootstraps=200, seed=42)
    lo2, hi2 = _bootstrap_pf_ci(pnls, n_bootstraps=200, seed=42)
    assert lo1 == lo2
    assert hi1 == hi2


def test_bootstrap_pf_ci_brackets_the_point_estimate():
    pnls = [1.0, -0.5, 1.2, -0.8, 1.5, -0.3, 0.9, 1.1, -0.6, 0.7]
    point = _profit_factor(pnls)
    lo, hi = _bootstrap_pf_ci(pnls, n_bootstraps=500, seed=1)
    assert lo <= point <= hi


def test_bootstrap_pf_ci_with_tiny_sample_returns_wide_interval():
    lo, hi = _bootstrap_pf_ci([1.0])
    assert lo == 0.0 and hi == float("inf")


# ---------------------------------------------------------------------------
# Trades loader
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_trades_csv(tmp_path: Path) -> Path:
    """Write a tiny synthetic backtest CSV that matches the production schema."""
    p = tmp_path / "trades.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "signal_id", "direction", "entry_bar", "exit_bar",
            "entry_price", "exit_price", "stop_loss", "take_profit",
            "confluence_score", "exit_reason", "bars_held",
            "pnl_price", "pnl_price_raw", "r_multiple", "initial_risk",
        ])
        # 10 trades — enough for the bootstrap CI to produce finite bounds.
        rows = [
            ("sig-1", "LONG", "2024-01-01 12:00:00", "2024-01-01 13:00:00",
             100.0, 102.0, 99.0, 103.0, 75.0, "tp_hit", 4, 2.0, 2.0, 2.0, 1.0),
            ("sig-2", "SHORT", "2024-01-02 12:00:00", "2024-01-02 13:00:00",
             100.0, 99.5, 101.0, 98.0, 70.0, "regime_shifted", 4, 0.5, 0.5, 0.5, 1.0),
            ("sig-3", "LONG", "2024-01-03 12:00:00", "2024-01-03 13:00:00",
             100.0, 99.0, 99.0, 103.0, 65.0, "sl_hit", 2, -1.0, -1.0, -1.0, 1.0),
            ("sig-4", "LONG", "2024-01-04 12:00:00", "2024-01-04 13:00:00",
             100.0, 101.5, 99.0, 103.0, 72.0, "tp_hit", 3, 1.5, 1.5, 1.5, 1.0),
            ("sig-5", "SHORT", "2024-01-05 12:00:00", "2024-01-05 13:00:00",
             100.0, 100.6, 101.0, 98.0, 68.0, "sl_hit", 2, -0.6, -0.6, -0.6, 1.0),
            ("sig-6", "LONG", "2024-01-06 12:00:00", "2024-01-06 13:00:00",
             100.0, 102.5, 99.0, 103.0, 78.0, "tp_hit", 5, 2.5, 2.5, 2.5, 1.0),
            ("sig-7", "LONG", "2024-01-07 12:00:00", "2024-01-07 13:00:00",
             100.0, 99.3, 99.0, 103.0, 60.0, "sl_hit", 3, -0.7, -0.7, -0.7, 1.0),
            ("sig-8", "SHORT", "2024-01-08 12:00:00", "2024-01-08 13:00:00",
             100.0, 98.0, 101.0, 98.0, 76.0, "tp_hit", 4, 2.0, 2.0, 2.0, 1.0),
            ("sig-9", "LONG", "2024-01-09 12:00:00", "2024-01-09 13:00:00",
             100.0, 100.8, 99.0, 103.0, 64.0, "early_exit", 6, 0.8, 0.8, 0.8, 1.0),
            ("sig-10", "SHORT", "2024-01-10 12:00:00", "2024-01-10 13:00:00",
             100.0, 100.4, 101.0, 98.0, 66.0, "sl_hit", 2, -0.4, -0.4, -0.4, 1.0),
        ]
        for r in rows:
            w.writerow(r)
    return p


def test_load_trades_returns_pnls(temp_trades_csv):
    snap = _load_trades(temp_trades_csv)
    assert snap is not None
    assert snap.n_trades == 10
    assert snap.pnls[:3] == [2.0, 0.5, -1.0]
    assert "2024-01-01" in snap.backtest_window


def test_load_trades_missing_file_returns_none(tmp_path):
    assert _load_trades(tmp_path / "missing.csv") is None


# ---------------------------------------------------------------------------
# Endpoint integration
# ---------------------------------------------------------------------------

@pytest.fixture
def authed_client(monkeypatch, temp_trades_csv, tmp_path):
    """TestClient bound to the synthetic CSV via env override."""
    monkeypatch.setenv("TRACK_RECORD_CSV", str(temp_trades_csv))
    monkeypatch.setenv("SENTINEL_TESTING_MODE", "1")
    # Avoid the global default SignalStore writing into the real /data dir
    monkeypatch.setenv("SIGNAL_DB_PATH", str(tmp_path / "signals.db"))
    app = create_app()
    return TestClient(app)


def test_endpoint_returns_pf_and_ci(authed_client):
    resp = authed_client.get("/api/v1/track-record")
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_trades"] == 10
    # PF = (2.0+0.5+1.5+2.5+2.0+0.8) / (1.0+0.6+0.7+0.4) ≈ 9.3 / 2.7 ≈ 3.44
    assert body["profit_factor"] == pytest.approx(3.44, abs=0.05)
    lo, hi = body["profit_factor_ci95"]
    assert lo is not None and hi is not None
    assert lo <= body["profit_factor"] <= hi
    assert body["bootstrap"]["n_iterations"] == DEFAULT_BOOTSTRAP_N
    # Honesty
    assert body["edge_claim"] is False or body["edge_claim"] is True  # bool
    assert "disclaimer" in body
    assert isinstance(body["equity_curve_r_multiples"], list)
    assert len(body["equity_curve_r_multiples"]) == 10


def test_endpoint_placeholder_when_csv_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("TRACK_RECORD_CSV", str(tmp_path / "nope.csv"))
    monkeypatch.setenv("SENTINEL_TESTING_MODE", "1")
    monkeypatch.setenv("SIGNAL_DB_PATH", str(tmp_path / "signals.db"))
    app = create_app()
    client = TestClient(app)
    resp = client.get("/api/v1/track-record")
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_trades"] == 0
    assert body["edge_claim"] is False
    assert "validation" in body["backtest_window"].lower()


def test_endpoint_is_public_no_auth_needed(authed_client):
    """No X-API-Key required — marketing page embeds this."""
    resp = authed_client.get("/api/v1/track-record")
    assert resp.status_code == 200
