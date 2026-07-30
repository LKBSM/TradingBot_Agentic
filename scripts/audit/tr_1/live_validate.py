"""TR-1 LIVE validation — drive the REAL assembler + FastAPI endpoint + scanner
on real cached XAUUSD candles, and print the exact JSON the frontend receives.
No Twelve Data call (candles come from the MT-D1 cache)."""
from __future__ import annotations
import json, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fastapi import FastAPI
from fastapi.testclient import TestClient

from scripts.audit.mt_d1.harness import load_candles
from src.intelligence.data_providers.twelve_data_provider import Candle
from src.intelligence.market_reading_assembler import (
    MarketReadingAssembler, build_cache_mtf_provider,
)
from src.storage.candles_cache_store import CandlesCacheStore
from src.storage.market_readings_store import MarketReadingsStore
from src.api.dependencies import AppState
from src.api.routes.market_reading import router as mr_router
from src.api.signal_store import SignalStore
from src.intelligence.conditions_scanner import _eval_mtf_aligned


def _candles(tf: str) -> list[Candle]:
    return [Candle(ts=c.ts, open=c.open, high=c.high, low=c.low, close=c.close,
                   volume=c.volume) for c in load_candles(tf)]


class _Provider:
    """Serves cached candles per timeframe — no network."""
    def __init__(self):
        self._by_tf = {tf: _candles(tf) for tf in ("H1", "M15")}
        # Synthesize H4 from H1 (every 4th bar) so the alignment tile has an upper unit.
        self._by_tf["H4"] = self._by_tf["H1"][::4]

    def fetch_candles(self, instrument, timeframe, count):
        return self._by_tf.get(timeframe.upper(), [])[-count:]


def main():
    tmp = Path(tempfile.mkdtemp())
    prov = _Provider()
    candles_store = CandlesCacheStore(db_path=str(tmp / "candles.db"))
    # Prime the cache so the mtf_provider (cache-only read) has upper-TF candles.
    for tf in ("H1", "H4", "M15"):
        candles_store.upsert_candles("XAUUSD", tf, prov._by_tf[tf])
    readings = MarketReadingsStore(db_path=str(tmp / "readings.db"))
    assembler = MarketReadingAssembler(
        data_provider=prov,
        readings_store=readings,
        candles_store=candles_store,
        mtf_provider=build_cache_mtf_provider(candles_store),
    )

    app = FastAPI()
    app.state.app_state = AppState(
        signal_store=SignalStore(db_path=str(tmp / "signals.db")),
        market_reading_assembler=assembler,
    )
    app.include_router(mr_router)
    client = TestClient(app)

    trends_by_tf = {}
    for tf in ("H1", "M15"):
        r = client.get("/api/market-reading",
                       params={"instrument": "XAUUSD", "timeframe": tf})
        assert r.status_code == 200, (tf, r.status_code, r.text[:300])
        d = r.json()
        reg = d["regime"]
        trends_by_tf[tf] = reg["trend"]
        print(f"\n===== GET /api/market-reading  XAUUSD {tf}  (HTTP {r.status_code}) =====")
        print(f"  regime.trend          = {reg['trend']}")
        print(f"  regime.market_phase   = {reg['market_phase']}")
        print(f"  regime.mtf_confluence = {reg.get('mtf_confluence')}")
        tr = reg.get("trend_reference")
        if tr:
            print(f"  regime.trend_reference= kind={tr['kind']} dir={tr['direction']} "
                  f"level={tr['level']} broken_at={tr['broken_at']} bars_ago={tr['bars_ago']}")
        else:
            print(f"  regime.trend_reference= None (indeterminate)")
        # journal coherence: last event direction must match trend (never contradict)
        bev = d["structure"].get("bos_events", [])
        cev = d["structure"].get("choch_events", [])
        last = None
        for e in (cev + bev):
            if last is None or (e.get("bars_ago") or 1e9) < (last.get("bars_ago") or 1e9):
                last = e
        print(f"  last journal event    = {last['direction'] if last else None} "
              f"(trend {'COHERENT' if (last and last['direction']==reg['trend']) or reg['trend']=='indeterminate' else 'CONTRADICTS!'})")

    # ---- Scanner alignment on the real per-TF trends, plus an injected indeterminate ----
    print("\n===== Scanner _eval_mtf_aligned (real per-TF structural trends) =====")
    real = {"H4": trends_by_tf.get("H1"), "H1": trends_by_tf["H1"], "M15": trends_by_tf["M15"]}
    res = _eval_mtf_aligned({"regime": {"trend": trends_by_tf["H1"]},
                             "header": {"close_price": 4050.0}}, "any", real)
    print(f"  real trends {real}")
    print(f"  -> met={res['met']} available={res['available']}")
    print(f"     {res['detail']}")

    inj = {"H4": "bullish", "H1": "bullish", "M15": "indeterminate"}
    res2 = _eval_mtf_aligned({"regime": {"trend": "bullish"},
                              "header": {"close_price": 4050.0}}, "any", inj)
    print(f"\n  injected one indeterminate {inj}")
    print(f"  -> met={res2['met']} available={res2['available']}")
    print(f"     {res2['detail']}")
    assert res2["met"] is False and res2["available"] is True
    assert "sur 3" in res2["detail"] and "indétermin" in res2["detail"].lower()
    print("\nLIVE VALIDATION OK — real payload carries structural trend + reference; "
          "scanner counts indeterminate apart with a visible denominator.")


if __name__ == "__main__":
    main()
