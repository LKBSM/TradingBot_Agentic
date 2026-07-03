"""Generate the OB golden fixtures + snapshot (mission diagnostics-rejet-OB).

Run ONCE on the unmodified engine (main @ 793c9a0), output committed:

    python -m scripts.audit.gen_ob_golden_fixtures

* Candle fixtures — 6 deterministic seeded random-walk OHLCV series
  (XAUUSD/EURUSD × M15/H1/H4, 500 bars, tz-aware index). Only created if the
  CSV does not exist yet: the committed CSVs are the frozen source of truth,
  the seed only mattered at creation time.
* Golden snapshot — ``tests/fixtures/ob_golden/golden_obs.json`` via the SAME
  ``snapshot_combo`` the test uses (single definition of the format).

Re-running against modified code will OVERWRITE the golden — never do that to
silence the non-regression test without an explicit founder decision.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tests.test_ob_golden_nonregression import (  # noqa: E402
    COMBOS,
    FIXTURE_DIR,
    GOLDEN_PATH,
    snapshot_all,
)

BARS = 500
TF_FREQ = {"M15": "15min", "H1": "1h", "H4": "4h"}
BASE_PRICE = {"XAUUSD": 4100.0, "EURUSD": 1.08}
STEP_SCALE = {
    ("XAUUSD", "M15"): 3.0, ("XAUUSD", "H1"): 6.0, ("XAUUSD", "H4"): 12.0,
    ("EURUSD", "M15"): 0.0008, ("EURUSD", "H1"): 0.0016, ("EURUSD", "H4"): 0.0032,
}


def make_fixture(instrument: str, timeframe: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    scale = STEP_SCALE[(instrument, timeframe)]
    steps = rng.normal(0.0, scale, BARS)
    # Mild regime drift so structure (fractals, BOS) actually forms.
    drift = np.sin(np.linspace(0, 6 * np.pi, BARS)) * scale * 0.4
    closes = BASE_PRICE[instrument] + np.cumsum(steps + drift)
    opens = np.concatenate([[BASE_PRICE[instrument]], closes[:-1]])
    wick = np.abs(rng.normal(0.0, scale * 0.6, BARS))
    highs = np.maximum(opens, closes) + wick
    lows = np.minimum(opens, closes) - wick
    volume = np.abs(rng.normal(1000.0, 250.0, BARS))
    index = pd.date_range(end="2026-06-30 20:00:00+00:00", periods=BARS,
                          freq=TF_FREQ[timeframe], name="ts")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volume},
        index=index,
    )


def main() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    for i, (instrument, timeframe) in enumerate(COMBOS):
        path = FIXTURE_DIR / f"{instrument}_{timeframe}.csv"
        if path.exists():
            print(f"kept existing fixture {path.name}")
            continue
        df = make_fixture(instrument, timeframe, seed=20260702 + i)
        df.to_csv(path, float_format="%.10f")
        print(f"wrote fixture {path.name} ({len(df)} bars)")

    golden = snapshot_all()
    GOLDEN_PATH.write_text(
        json.dumps(golden, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    for key, snap in golden.items():
        print(f"{key}: {len(snap['engine_obs'])} engine OBs, "
              f"{len(snap['surfaced_order_blocks'])} surfaced")
    print(f"wrote {GOLDEN_PATH}")


if __name__ == "__main__":
    main()
