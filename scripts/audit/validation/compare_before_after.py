"""Comparaison AVANT/APRÈS des corrections de niveaux (F1/F2/F3/F6/P4).

AVANT = /tmp/marketreadings_BEFORE.json (dataset pré-fixes, sauvegardé).
APRÈS = data/validation/marketreadings_2026_06_06.json (régénéré post-fixes).

Vérifie aussi la VÉRITÉ MOTEUR indépendante (BOS_BREAK_LEVEL ffill ré-extrait de
l'engine) pour 5 échantillons, afin de montrer AVANT vs APRÈS vs vrai moteur.

Sortie : impression console + data/validation/before_after_comparison.json
Usage : python scripts/audit/validation/compare_before_after.py /tmp/marketreadings_BEFORE.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.intelligence.smart_money import SmartMoneyEngine  # noqa: E402
from src.intelligence.market_reading_mappers import realized_levels  # noqa: E402

AFTER = ROOT / "data" / "validation" / "marketreadings_2026_06_06.json"
OUT = ROOT / "data" / "validation" / "before_after_comparison.json"

CSVS = {
    "XAUUSD": ROOT / "data" / "XAU_15MIN_2019_2026.csv",
    "EURUSD": ROOT / "data" / "EURUSD_15MIN_2019_2025.csv",
}


def _bos(reading):
    return reading["structure"].get("bos")


def _load_engine_truth(instrument: str, bar_open_ts: str) -> float | None:
    """Re-extract the engine's forward-filled BOS_BREAK_LEVEL at a given H1 bar."""
    from src.intelligence.volatility_forecaster import resample_ohlcv
    df = pd.read_csv(CSVS[instrument])
    df = df.rename(columns={c: c.capitalize() for c in df.columns})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df.columns = [c.lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]]
    df.index.name = "timestamp"
    h1 = resample_ohlcv(df.reset_index(), "M15", "H1")
    h1["timestamp"] = pd.to_datetime(h1["timestamp"])
    h1 = h1.set_index("timestamp").sort_index()
    eng = SmartMoneyEngine(data=h1[["open", "high", "low", "close", "volume"]],
                           config={}, verbose=False)
    enr = eng.analyze().reindex(h1.index)
    target = pd.Timestamp(bar_open_ts.replace("+00:00", "").replace("Z", ""))
    if target.tzinfo is not None:
        target = target.tz_localize(None)
    pos = h1.index.get_indexer([target], method="nearest")[0]
    lv = realized_levels(enr, idx=pos)
    return lv.get("BOS_BREAK_LEVEL_LAST")


def main(before_path: str):
    before = {r["candle_id"]: r for r in json.loads(Path(before_path).read_text(encoding="utf-8"))["readings"]}
    after_data = json.loads(AFTER.read_text(encoding="utf-8"))
    after = {r["candle_id"]: r for r in after_data["readings"]}

    # Agrégats
    bos_tag_before = sum(1 for r in before.values()
                         if any(t.startswith("bos_recent") for t in r["market_reading"]["conditions"]["tags"]))
    bos_tag_after = sum(1 for r in after.values()
                        if any(t.startswith("bos_recent") for t in r["market_reading"]["conditions"]["tags"]))
    src_before = {}
    src_after = {}
    for r in before.values():
        src_before[r["description_source"]] = src_before.get(r["description_source"], 0) + 1
    for r in after.values():
        src_after[r["description_source"]] = src_after.get(r["description_source"], 0) + 1

    # BOS level: combien retombaient sur close_price (proxy) AVANT vs APRÈS
    def bos_eq_close(reg):
        cnt = 0
        for r in reg.values():
            b = _bos(r["market_reading"])
            if b and abs(b["level"] - r["close_price"]) < 1e-9:
                cnt += 1
        return cnt

    # 5 échantillons BOS (ceux qui ont un bos APRÈS, pour montrer le vrai niveau)
    samples = []
    for cid, ra in sorted(after.items()):
        ba = _bos(ra["market_reading"])
        if ba is None:
            continue
        rb = before.get(cid)
        bb = _bos(rb["market_reading"]) if rb else None
        truth = _load_engine_truth(ra["instrument"], ra["bar_open_ts"])
        samples.append({
            "candle_id": cid, "instrument": ra["instrument"],
            "bar_open_ts": ra["bar_open_ts"], "close_price": ra["close_price"],
            "bos_level_before": bb["level"] if bb else None,
            "bos_level_after": ba["level"],
            "bos_level_engine_truth": truth,
            "matches_truth": (truth is not None and abs(ba["level"] - truth) < 1e-6),
        })
        if len(samples) >= 5:
            break

    # OB/FVG sample (montrer proxy AVANT vs zone reelle APRÈS)
    ob_fvg_samples = []
    for cid, ra in sorted(after.items()):
        s = ra["market_reading"]["structure"]
        if not s.get("order_blocks") and not s.get("fair_value_gaps"):
            continue
        rb = before.get(cid)
        sb = rb["market_reading"]["structure"] if rb else {}
        rec = {"candle_id": cid, "instrument": ra["instrument"], "close_price": ra["close_price"]}
        if s.get("order_blocks"):
            ob_a = s["order_blocks"][0]
            ob_b = sb.get("order_blocks", [{}])[0] if sb.get("order_blocks") else {}
            rec["ob_before"] = [ob_b.get("level_low"), ob_b.get("level_high")]
            rec["ob_after"] = [ob_a["level_low"], ob_a["level_high"]]
        if s.get("fair_value_gaps"):
            fa = s["fair_value_gaps"][0]
            fb = sb.get("fair_value_gaps", [{}])[0] if sb.get("fair_value_gaps") else {}
            rec["fvg_before"] = [fb.get("level_low"), fb.get("level_high")]
            rec["fvg_after"] = [fa["level_low"], fa["level_high"]]
        ob_fvg_samples.append(rec)
        if len(ob_fvg_samples) >= 5:
            break

    result = {
        "aggregates": {
            "bos_recent_tag_count": {"before": bos_tag_before, "after": bos_tag_after, "of": 60},
            "bos_level_equals_close_proxy": {"before": bos_eq_close(before), "after": bos_eq_close(after)},
            "description_source": {"before": src_before, "after": src_after},
        },
        "bos_level_samples": samples,
        "ob_fvg_samples": ob_fvg_samples,
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 70)
    print("COMPARAISON AVANT / APRÈS — corrections niveaux")
    print("=" * 70)
    print(f"Tag bos_recent présent : AVANT {bos_tag_before}/60 → APRÈS {bos_tag_after}/60")
    print(f"BOS.level == close_price (proxy faux) : AVANT {bos_eq_close(before)} → APRÈS {bos_eq_close(after)}")
    print(f"description_source AVANT : {src_before}")
    print(f"description_source APRÈS : {src_after}")
    print("\n--- 5 échantillons BOS : AVANT vs APRÈS vs VÉRITÉ MOTEUR ---")
    for s in samples:
        print(f"  #{s['candle_id']:2d} {s['instrument']} close={s['close_price']:.5f} | "
              f"avant={s['bos_level_before']} | après={s['bos_level_after']:.5f} | "
              f"moteur={s['bos_level_engine_truth']:.5f} | match={s['matches_truth']}")
    print("\n--- 5 échantillons OB/FVG : proxy AVANT vs zone réelle APRÈS ---")
    for r in ob_fvg_samples:
        if "ob_after" in r:
            print(f"  #{r['candle_id']:2d} {r['instrument']} OB  avant={r['ob_before']} → après={r['ob_after']}")
        if "fvg_after" in r:
            print(f"  #{r['candle_id']:2d} {r['instrument']} FVG avant={r['fvg_before']} → après={r['fvg_after']}")
    print(f"\n→ {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/marketreadings_BEFORE.json")
