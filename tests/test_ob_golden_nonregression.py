"""Golden non-regression test — OB detection is IDENTICAL before/after the
rejection-diagnostics instrumentation (mission 2026-07-02, exigence dure #2).

The snapshot ``tests/fixtures/ob_golden/golden_obs.json`` was generated on the
UNMODIFIED engine (main @ 793c9a0) by ``scripts/audit/gen_ob_golden_fixtures.py``
over 6 deterministic candle fixtures (XAUUSD/EURUSD × M15/H1/H4, 500 bars each,
committed CSVs). This test re-runs the live code on the same fixtures and
asserts the full OB set — BOTH levels:

  * engine level  — every bar where ``_add_smc_order_blocks`` marked an OB
    (side, zone bounds, strength);
  * surfaced level — the ``collect_zones`` output the product actually shows
    (direction, bounds, importance, status, tested, timestamps, rank).

Any accept/reject drift, zone-geometry drift, lifecycle drift or ordering drift
fails loudly. DO NOT regenerate the snapshot to make this test pass without an
explicit founder decision — the whole point is that instrumentation must not
move detection by a single zone.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "ob_golden"
GOLDEN_PATH = FIXTURE_DIR / "golden_obs.json"

COMBOS = [
    ("XAUUSD", "M15"),
    ("XAUUSD", "H1"),
    ("XAUUSD", "H4"),
    ("EURUSD", "M15"),
    ("EURUSD", "H1"),
    ("EURUSD", "H4"),
]


def _r(value, ndigits: int = 10):
    """Stable float rounding so the JSON snapshot is platform-independent."""
    if value is None:
        return None
    return round(float(value), ndigits)


def load_fixture_frame(instrument: str, timeframe: str) -> pd.DataFrame:
    path = FIXTURE_DIR / f"{instrument}_{timeframe}.csv"
    df = pd.read_csv(path, index_col="ts", parse_dates=["ts"])
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def snapshot_combo(df: pd.DataFrame) -> dict:
    """Run the LIVE engine + zone collector on a fixture frame and normalise
    the complete OB set into a JSON-stable dict.

    Shared by the generator script and this test so the snapshot format has a
    single definition.
    """
    from src.intelligence.market_reading_mappers import collect_zones
    from src.intelligence.smart_money import SmartMoneyEngine

    engine = SmartMoneyEngine(data=df.copy(), config={}, verbose=False)
    enriched = engine.analyze(compute_divergence=False)

    engine_obs: list[dict] = []
    for k in range(len(enriched)):
        row = enriched.iloc[k]
        for side, hi_col, lo_col in (
            ("bullish", "BULLISH_OB_HIGH", "BULLISH_OB_LOW"),
            ("bearish", "BEARISH_OB_HIGH", "BEARISH_OB_LOW"),
        ):
            hv, lv = row[hi_col], row[lo_col]
            if pd.isna(hv) or pd.isna(lv):
                continue
            engine_obs.append({
                "bar_index": k,
                "ts": enriched.index[k].isoformat(),
                "side": side,
                "ob_high": _r(hv),
                "ob_low": _r(lv),
                "strength": _r(row["OB_STRENGTH_NORM"]),
            })

    zones = collect_zones(enriched, idx=len(enriched) - 1)
    surfaced = [
        {
            "rank": rank,
            "direction": z["direction"],
            "level_high": _r(z["level_high"]),
            "level_low": _r(z["level_low"]),
            "importance": z["importance"],
            "status": z["status"],
            "tested": bool(z["tested"]),
            "created_at": z["created_at"].isoformat() if z["created_at"] else None,
            "mitigated_at": z["mitigated_at"].isoformat() if z["mitigated_at"] else None,
            "strength": _r(z["_strength"]),
            "bar_index": z["_k"],
        }
        for rank, z in enumerate(zones["order_blocks"])
    ]

    return {"engine_obs": engine_obs, "surfaced_order_blocks": surfaced}


def snapshot_all() -> dict:
    out: dict = {}
    for instrument, timeframe in COMBOS:
        df = load_fixture_frame(instrument, timeframe)
        out[f"{instrument}_{timeframe}"] = snapshot_combo(df)
    return out


@pytest.mark.parametrize("instrument,timeframe", COMBOS)
def test_ob_set_identical_to_golden(instrument: str, timeframe: str) -> None:
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    key = f"{instrument}_{timeframe}"
    assert key in golden, f"golden snapshot missing combo {key} — regenerate deliberately"

    live = snapshot_combo(load_fixture_frame(instrument, timeframe))
    expected = golden[key]

    assert live["engine_obs"] == expected["engine_obs"], (
        f"{key}: engine-level OB set drifted from golden snapshot "
        f"({len(live['engine_obs'])} vs {len(expected['engine_obs'])} zones)"
    )
    assert live["surfaced_order_blocks"] == expected["surfaced_order_blocks"], (
        f"{key}: surfaced OB list drifted from golden snapshot"
    )


def test_golden_snapshot_is_non_trivial() -> None:
    """Guard against a vacuous golden file: every combo must actually contain
    OBs at both levels, otherwise the non-regression proof proves nothing."""
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    assert set(golden) == {f"{i}_{t}" for i, t in COMBOS}
    for key, snap in golden.items():
        assert len(snap["engine_obs"]) >= 5, f"{key}: too few engine OBs for a meaningful golden"
        assert len(snap["surfaced_order_blocks"]) >= 1, f"{key}: no surfaced OB in golden"
