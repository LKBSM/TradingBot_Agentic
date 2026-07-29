"""MT-D1 audit harness — READ-ONLY on the engine.

Loads cached XAUUSD candles, runs the REAL engine (build_enriched_frame /
SmartMoneyEngine) exactly as production does, and exposes the enriched frame +
event extraction. Nothing here alters a detection rule; the only "variations"
(section 7) are done either through the public SMCConfig knob (FRACTAL_WINDOW)
or in a clearly-separate standalone re-implementation (variant_detect.py) used
for the report only — the repo engine is never modified.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

CACHE = Path("docs/audits/ECHANTILLON-DETECTION-2026-07-29/_cache")


@dataclass
class Candle:
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


def _parse_ts(s: str) -> datetime:
    # Twelve Data returns "YYYY-MM-DD HH:MM:SS" (UTC for XAU/USD). Treat as UTC.
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def load_candles(tf: str) -> list[Candle]:
    path = CACHE / f"XAUUSD_{tf}.jsonl"
    out: list[Candle] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        v = json.loads(line)
        out.append(Candle(
            ts=_parse_ts(v["datetime"]),
            open=float(v["open"]), high=float(v["high"]),
            low=float(v["low"]), close=float(v["close"]),
            volume=float(v.get("volume", 0.0) or 0.0),
        ))
    out.sort(key=lambda c: c.ts)
    return out


def enrich(candles: list[Candle], fractal_window: Optional[int] = None) -> pd.DataFrame:
    """Run the REAL SmartMoneyEngine. fractal_window=None → product default (2)."""
    from src.intelligence.smart_money import SmartMoneyEngine

    df = pd.DataFrame(
        [{"open": c.open, "high": c.high, "low": c.low, "close": c.close,
          "volume": c.volume} for c in candles],
        index=[c.ts for c in candles],
    )
    df.index.name = "ts"
    config = {} if fractal_window is None else {"FRACTAL_WINDOW": int(fractal_window)}
    engine = SmartMoneyEngine(data=df, config=config, verbose=False)
    return engine.analyze(compute_divergence=False)


def extract_events(enriched: pd.DataFrame) -> list[dict]:
    """All discrete BOS/CHOCH events over the frame, oldest→newest.

    Mirrors collect_structure_events' source columns (BOS_EVENT, CHOCH_SIGNAL,
    BOS_BREAK_LEVEL) WITHOUT the max-per-type cap, so we can see the full
    detected journal (the "deep" view) vs. the capped surface separately.
    """
    ev = []
    idx = list(enriched.index)
    be = enriched["BOS_EVENT"].values
    ch = enriched["CHOCH_SIGNAL"].values
    lvl = enriched["BOS_BREAK_LEVEL"].values
    close = enriched["close"].values
    for k in range(len(enriched)):
        for col, val, kind in (("BOS_EVENT", be[k], "BOS"), ("CHOCH_SIGNAL", ch[k], "CHOCH")):
            if pd.isna(val) or val == 0:
                continue
            level = lvl[k] if not pd.isna(lvl[k]) else close[k]
            ev.append({
                "k": k,
                "ts": idx[k],
                "type": kind,
                "direction": "bullish" if val > 0 else "bearish",
                "level": float(level),
            })
    ev.sort(key=lambda e: e["k"])
    return ev


def surface_events(enriched: pd.DataFrame, window_bars: int = 500, cap: int = 8) -> dict:
    """Reproduce what the LIVE assembler surfaces: detection over the last
    `window_bars` bars, then most-recent-first capped at `cap` per type
    (MAX_STRUCTURE_EVENTS). This is exactly the product's on-screen journal.
    """
    win = enriched.iloc[-window_bars:] if len(enriched) > window_bars else enriched
    ev = extract_events(win)
    bos = [e for e in ev if e["type"] == "BOS"]
    choch = [e for e in ev if e["type"] == "CHOCH"]
    bos_surf = sorted(bos, key=lambda e: -e["k"])[:cap]
    choch_surf = sorted(choch, key=lambda e: -e["k"])[:cap]
    return {
        "window_bars": len(win),
        "bos_in_window": len(bos),
        "choch_in_window": len(choch),
        "bos_surfaced": len(bos_surf),
        "choch_surfaced": len(choch_surf),
        "bos_surf": bos_surf,
        "choch_surf": choch_surf,
        "last_event": max(ev, key=lambda e: e["k"]) if ev else None,
    }
