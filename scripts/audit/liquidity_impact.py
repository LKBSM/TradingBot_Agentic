"""Impact measurement — external liquidity pools across the 6 reference combos.

Combos = {XAUUSD, EURUSD} × {M15, H1, H4} (the standard audit set). For each
combo we resample the 15-min source to the target timeframe, run the real
SmartMoneyEngine, then sample `collect_liquidity_pools` at evenly-spaced read
points across the history and aggregate:

  * pools per read (mean), and the count by kind (EQH/EQL/range),
  * % external vs internal,
  * % intact / swept / broken.

Reproducible; reads only the existing CSVs (path via --data-dir). Writes a JSON
summary the audit report cites. NOTHING is mutated — pure measurement.

Usage:
    python scripts/audit/liquidity_impact.py \
        --data-dir C:/MyPythonProjects/TradingBOT_Agentic/data \
        --out docs/audits/liquidity_impact_data.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src.environment.strategy_features import SmartMoneyEngine
from src.intelligence.market_reading_mappers import collect_liquidity_pools
from src.intelligence.volatility_forecaster import resample_ohlcv

SOURCES = {
    "XAUUSD": "XAU_15MIN_2019_2026.csv",
    "EURUSD": "EURUSD_15MIN_2019_2025.csv",
}
TIMEFRAMES = ["M15", "H1", "H4"]
LOOKBACK = 200          # bars the engine window / pocket scan uses per read
N_READS = 60            # evenly-spaced read points per combo


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df.columns = [c.lower() for c in df.columns]
    return df[["open", "high", "low", "close", "volume"]]


def _measure_combo(df15: pd.DataFrame, tf: str) -> dict:
    df = df15 if tf == "M15" else resample_ohlcv(df15, "M15", tf)
    df = df.dropna()
    engine = SmartMoneyEngine(data=df, config={}, verbose=False)
    enriched = engine.analyze(compute_divergence=False)
    cfg = engine.config

    n = len(enriched)
    if n < LOOKBACK + 5:
        return {"reads": 0}

    read_idxs = np.linspace(LOOKBACK, n - 1, num=min(N_READS, n - LOOKBACK), dtype=int)
    per_read_counts: list[int] = []
    kind_counter: Counter = Counter()
    status_counter: Counter = Counter()
    external_counter: Counter = Counter()

    for ridx in read_idxs:
        pools = collect_liquidity_pools(
            enriched,
            idx=int(ridx),
            eq_tolerance_atr=cfg.EQ_TOLERANCE_ATR,
            eq_tolerance_pips_floor=cfg.EQ_TOLERANCE_PIPS_FLOOR,
            eq_min_touches=cfg.EQ_MIN_TOUCHES,
            lookback=cfg.LIQ_LOOKBACK,
        )
        per_read_counts.append(len(pools))
        for p in pools:
            kind_counter[p["kind"]] += 1
            status_counter[p["status"]] += 1
            external_counter["external" if p["is_external"] else "internal"] += 1

    total = sum(status_counter.values())

    def _pct(c: Counter, key: str) -> float:
        return round(100.0 * c.get(key, 0) / total, 1) if total else 0.0

    return {
        "bars": int(n),
        "reads": int(len(read_idxs)),
        "pools_total": int(total),
        "pools_per_read_mean": round(float(np.mean(per_read_counts)), 2) if per_read_counts else 0.0,
        "pools_per_read_max": int(np.max(per_read_counts)) if per_read_counts else 0,
        "by_kind": dict(kind_counter),
        "pct_external": _pct(external_counter, "external"),
        "pct_internal": _pct(external_counter, "internal"),
        "pct_intact": _pct(status_counter, "intact"),
        "pct_swept": _pct(status_counter, "swept"),
        "pct_broken": _pct(status_counter, "broken"),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)

    results: dict[str, dict] = {}
    for instrument, fname in SOURCES.items():
        src = args.data_dir / fname
        if not src.exists():
            print(f"[skip] {instrument}: {src} missing")
            continue
        df15 = _load(src)
        for tf in TIMEFRAMES:
            combo = f"{instrument}_{tf}"
            print(f"[run] {combo} ...", flush=True)
            results[combo] = _measure_combo(df15, tf)
            r = results[combo]
            print(f"      {r.get('pools_total', 0)} pools over {r.get('reads', 0)} reads "
                  f"| mean/read={r.get('pools_per_read_mean', 0)} "
                  f"| swept={r.get('pct_swept', 0)}% broken={r.get('pct_broken', 0)}% "
                  f"intact={r.get('pct_intact', 0)}% | external={r.get('pct_external', 0)}%")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[done] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
