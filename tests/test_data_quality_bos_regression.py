"""Régression test data quality — Sprint 0 Batch 0.4.

Garde-fou contre le retour à un CSV à coverage faible (< 95 %).

Contexte
--------
Le CSV `data/XAU_15MIN_2019_2025.csv` (63 % coverage session active) cause
le bug "BOS event firing sur 100 % des bars" car les discontinuités (gaps)
font que la détection BOS croit voir un break à chaque bar.

Le CSV primaire courant `data/XAU_15MIN_2019_2026.csv` (98.72 % coverage)
produit un firing rate BOS raisonnable (< 5 %).

Ce test :
1. Charge le CSV primaire actuel.
2. Lance le SmartMoneyEngine sur un échantillon de 2 000 bars.
3. Vérifie que `BOS_EVENT != 0` firing rate est dans [0.5 %, 10 %].
4. Le test tombe rouge si on revient sur le CSV cassé OU si la logique
   BOS dérive vers du sur-firing/sous-firing.

Décision tranchée : `audits/2026-Q2/sprint_0_decisions.md` (décision A).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


PRIMARY_XAU_CSV = Path(__file__).resolve().parent.parent / "data" / "XAU_15MIN_2019_2026.csv"


@pytest.fixture(scope="module")
def xau_sample() -> pd.DataFrame:
    """Load first 2000 bars of the primary XAU CSV."""
    if not PRIMARY_XAU_CSV.exists():
        pytest.skip(f"Primary XAU CSV missing: {PRIMARY_XAU_CSV}")
    df = pd.read_csv(PRIMARY_XAU_CSV, nrows=2000, parse_dates=["Date"]).set_index("Date")
    df.rename(columns={c: c.capitalize() for c in df.columns if c.lower() in {"open", "high", "low", "close", "volume"}}, inplace=True)
    if "Volume" not in df.columns:
        df["Volume"] = 0
    return df


def test_primary_csv_coverage_is_sufficient(xau_sample: pd.DataFrame) -> None:
    """Sanity check: the loaded sample is non-trivial."""
    assert len(xau_sample) >= 1800, "Primary CSV should yield ≥ 1800 bars in 2000-row head"
    expected_cols = {"Open", "High", "Low", "Close"}
    assert expected_cols.issubset(set(xau_sample.columns)), f"Missing OHLC columns: {expected_cols - set(xau_sample.columns)}"


def test_BOS_EVENT_firing_rate_is_reasonable(xau_sample: pd.DataFrame) -> None:
    """BOS event firing rate must be in [0.5 %, 10 %].

    < 0.5 % → détecteur trop strict, manque les breaks (régression logique).
    > 10 %  → détecteur sur-fire, probablement data quality issue (CSV broken).

    Cible empirique sur XAU 2019-2026 (98.72 % coverage) : ~1-3 %.
    Cible empirique sur XAU 2019-2025 (63 % broken)       : ~100 % ← exclu.
    """
    from src.environment.strategy_features import SmartMoneyEngine

    engine = SmartMoneyEngine(data=xau_sample.copy(), config={}, verbose=False)
    enriched = engine.analyze()

    assert "BOS_EVENT" in enriched.columns, "SmartMoneyEngine must emit `BOS_EVENT` column"

    n_bars = len(enriched)
    n_events = int((enriched["BOS_EVENT"] != 0).sum())
    firing_rate_pct = 100.0 * n_events / n_bars

    assert firing_rate_pct >= 0.5, (
        f"BOS event firing rate too low: {firing_rate_pct:.2f}% on {n_bars} bars. "
        f"Detector may be over-strict — investigate the BOS detection logic."
    )
    assert firing_rate_pct <= 10.0, (
        f"BOS event firing rate too high: {firing_rate_pct:.2f}% on {n_bars} bars. "
        f"This is the symptom of using a low-coverage CSV (e.g. XAU_15MIN_2019_2025.csv at 63%). "
        f"Verify that config.py and scripts/audit_backtest.py point to "
        f"XAU_15MIN_2019_2026.csv (98.72% coverage). "
        f"See audits/2026-Q2/data_layer_pre_flight.md (Decision A)."
    )


def test_config_points_to_primary_csv() -> None:
    """`config.py` must point HISTORICAL_DATA_FILE at the high-coverage CSV.

    This is the prod fallback path used by `src.core.config_loader`.
    """
    import config as project_config

    historical = Path(project_config.HISTORICAL_DATA_FILE).name
    assert historical == "XAU_15MIN_2019_2026.csv", (
        f"config.HISTORICAL_DATA_FILE points to '{historical}' — expected "
        f"'XAU_15MIN_2019_2026.csv' (decision A in audits/2026-Q2/sprint_0_decisions.md). "
        f"Reverting to lower-coverage feeds (notably 2019_2025 at 63%) re-introduces "
        f"the BOS-100%-bars data quality bug."
    )
