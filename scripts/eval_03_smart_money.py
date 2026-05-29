"""Audit quantitatif du Smart Money Engine (Prompt 03).

Mesure sur XAU M15 2019-2024 (97.6% couverture) :
  - Fréquence d'émission BOS/CHOCH/OB/FVG par an, par direction
  - Taux de succès du retest (AWAITING -> ARMED) vs invalidation vs timeout
  - Qualité forward post-BOS (prix +5/+20/+50 bars)
  - Asymétrie long/short par année
  - Latence de détection (bars entre swing et émission)
  - Matrice de confusion retest armé -> continuation >= 1*ATR / échec / ambigu

Sortie : reports/eval_03/eval_03_stats.json + eval_03_summary.md
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from src.environment.strategy_features import SmartMoneyEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("eval_03")

CSV_PATH = Path("data/XAU_15MIN_2019_2024.csv")
OUT_DIR = Path("reports/eval_03")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Horizons de forward check (en bars M15 => 5 = 75min, 20 = 5h, 50 = 12.5h)
FORWARD_HORIZONS = [5, 20, 50]


def load_ohlcv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for cand in ("timestamp", "Date", "date", "datetime", "time"):
        if cand in df.columns:
            df[cand] = pd.to_datetime(df[cand])
            df = df.set_index(cand)
            break
    rename = {}
    for col in df.columns:
        lc = col.lower()
        if lc in ("open", "high", "low", "close", "volume"):
            rename[col] = lc.capitalize()
    if rename:
        df = df.rename(columns=rename)
    if "Volume" not in df.columns:
        df["Volume"] = 0.0
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df = df[~df.index.duplicated(keep="first")].sort_index()
    log.info("Loaded %d bars (%s -> %s)", len(df), df.index.min(), df.index.max())
    return df


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    engine = SmartMoneyEngine(
        data=df,
        config={
            "RSI_WINDOW": 14,
            "MACD_FAST": 12,
            "MACD_SLOW": 26,
            "MACD_SIGNAL": 9,
            "BB_WINDOW": 20,
            "ATR_WINDOW": 14,
            "FRACTAL_WINDOW": 2,
            "FVG_THRESHOLD": 0.1,
            "OB_REQUIRE_FVG": False,
        },
    )
    enriched = engine.analyze()
    log.info("Enriched: %d bars kept (after cleaning NaN)", len(enriched))
    return enriched


def forward_stats(df: pd.DataFrame, idx: int, atr: float, direction: int, horizon: int):
    """Retourne (max_favorable_excursion, max_adverse_excursion) en multiples d'ATR."""
    end = min(idx + horizon + 1, len(df))
    closes = df["close"].iloc[idx + 1:end].values
    highs = df["high"].iloc[idx + 1:end].values
    lows = df["low"].iloc[idx + 1:end].values
    if len(closes) == 0 or atr <= 0:
        return 0.0, 0.0
    entry = float(df["close"].iloc[idx])
    if direction == 1:  # long
        mfe = (highs.max() - entry) / atr
        mae = (entry - lows.min()) / atr
    else:
        mfe = (entry - lows.min()) / atr
        mae = (highs.max() - entry) / atr
    return float(mfe), float(mae)


def audit(df: pd.DataFrame) -> Dict:
    bos_event = df["BOS_EVENT"].values
    choch = df["CHOCH_SIGNAL"].values
    retest_state = df["BOS_RETEST_STATE"].values
    retest_armed = df["BOS_RETEST_ARMED"].values
    fvg = df["FVG_SIGNAL"].values
    ob_up = df["BULLISH_OB_HIGH"].notna().values
    ob_dn = df["BEARISH_OB_HIGH"].notna().values
    atr = df["ATR"].values
    years = df.index.year.values
    n = len(df)

    # --- Compteurs globaux ---
    stats = {
        "n_bars": int(n),
        "date_start": str(df.index.min()),
        "date_end": str(df.index.max()),
        "counts_global": {
            "bos_up": int((bos_event == 1).sum()),
            "bos_down": int((bos_event == -1).sum()),
            "choch_up": int((choch == 1).sum()),
            "choch_down": int((choch == -1).sum()),
            "fvg_up": int((fvg == 1).sum()),
            "fvg_down": int((fvg == -1).sum()),
            "ob_bullish": int(ob_up.sum()),
            "ob_bearish": int(ob_dn.sum()),
            "retest_armed_up_bars": int((retest_armed == 1).sum()),
            "retest_armed_down_bars": int((retest_armed == -1).sum()),
        },
    }
    # Frequences
    stats["freq_per_1k_bars"] = {
        k: round(v * 1000.0 / n, 2) for k, v in stats["counts_global"].items()
    }

    # --- Par année : événements + asymétrie forward ---
    per_year = defaultdict(lambda: {
        "bars": 0,
        "bos_up": 0, "bos_down": 0,
        "choch_up": 0, "choch_down": 0,
        "fvg_up": 0, "fvg_down": 0,
        "ob_up": 0, "ob_down": 0,
        "retest_armed_up_events": 0, "retest_armed_down_events": 0,
    })
    for i in range(n):
        y = int(years[i])
        per_year[y]["bars"] += 1
        if bos_event[i] == 1: per_year[y]["bos_up"] += 1
        elif bos_event[i] == -1: per_year[y]["bos_down"] += 1
        if choch[i] == 1: per_year[y]["choch_up"] += 1
        elif choch[i] == -1: per_year[y]["choch_down"] += 1
        if fvg[i] == 1: per_year[y]["fvg_up"] += 1
        elif fvg[i] == -1: per_year[y]["fvg_down"] += 1
        if ob_up[i]: per_year[y]["ob_up"] += 1
        if ob_dn[i]: per_year[y]["ob_down"] += 1
    stats["per_year_counts"] = {str(k): v for k, v in sorted(per_year.items())}

    # --- Retest state machine : AWAITING -> ARMED vs invalidation ---
    # Reconstruction par parcours : un BOS event entre en state = ±1 (awaiting)
    # On compte les transitions : armed_success, invalidation, timeout
    awaiting_to_armed = 0
    awaiting_invalidated = 0  # state passe de ±1 à 0 sans atteindre ±2
    armed_events = {"up": 0, "down": 0}  # premier bar où state = ±2
    prev = 0
    current_bos_dir = 0
    waiting_was_seen = False
    for i in range(n):
        cur = retest_state[i]
        if cur in (1, -1) and prev == 0:
            # entrée en awaiting
            current_bos_dir = int(cur)
            waiting_was_seen = True
        elif cur in (2, -2) and prev in (1, -1):
            awaiting_to_armed += 1
            if cur == 2: armed_events["up"] += 1
            else: armed_events["down"] += 1
        elif cur == 0 and prev in (1, -1) and waiting_was_seen:
            awaiting_invalidated += 1
            waiting_was_seen = False
        if cur == 0:
            waiting_was_seen = False
            current_bos_dir = 0
        prev = cur

    total_bos = stats["counts_global"]["bos_up"] + stats["counts_global"]["bos_down"]
    stats["retest_transitions"] = {
        "total_bos_events": total_bos,
        "awaiting_to_armed": awaiting_to_armed,
        "awaiting_invalidated_or_timeout": awaiting_invalidated,
        "retest_success_rate": round(awaiting_to_armed / max(1, total_bos), 3),
        "armed_events_up": armed_events["up"],
        "armed_events_down": armed_events["down"],
    }

    # --- Forward excursion post-BOS event (entrée naïve au close du break) ---
    bos_fwd = {h: {"up": {"mfe": [], "mae": []}, "down": {"mfe": [], "mae": []}}
               for h in FORWARD_HORIZONS}
    for i in range(n):
        if bos_event[i] == 0: continue
        a = float(atr[i])
        if a <= 0: continue
        d = 1 if bos_event[i] == 1 else -1
        side = "up" if d == 1 else "down"
        for h in FORWARD_HORIZONS:
            mfe, mae = forward_stats(df, i, a, d, h)
            bos_fwd[h][side]["mfe"].append(mfe)
            bos_fwd[h][side]["mae"].append(mae)

    def _summ(arr):
        if not arr: return None
        return {
            "n": len(arr),
            "mean": round(float(np.mean(arr)), 3),
            "median": round(float(np.median(arr)), 3),
            "p25": round(float(np.percentile(arr, 25)), 3),
            "p75": round(float(np.percentile(arr, 75)), 3),
        }

    stats["forward_after_bos"] = {}
    for h in FORWARD_HORIZONS:
        stats["forward_after_bos"][f"h_{h}"] = {
            "up_mfe_atr":  _summ(bos_fwd[h]["up"]["mfe"]),
            "up_mae_atr":  _summ(bos_fwd[h]["up"]["mae"]),
            "down_mfe_atr":_summ(bos_fwd[h]["down"]["mfe"]),
            "down_mae_atr":_summ(bos_fwd[h]["down"]["mae"]),
        }

    # --- Forward post-retest armed (premier bar où retest_armed devient ±1) ---
    armed_fwd = {h: {"up": {"mfe": [], "mae": []}, "down": {"mfe": [], "mae": []}}
                 for h in FORWARD_HORIZONS}
    prev_armed = 0
    for i in range(n):
        cur = int(retest_armed[i])
        # on ne compte que la transition d'enclenchement (0 -> ±1)
        if cur != 0 and prev_armed == 0:
            a = float(atr[i])
            if a > 0:
                d = 1 if cur == 1 else -1
                side = "up" if d == 1 else "down"
                for h in FORWARD_HORIZONS:
                    mfe, mae = forward_stats(df, i, a, d, h)
                    armed_fwd[h][side]["mfe"].append(mfe)
                    armed_fwd[h][side]["mae"].append(mae)
        prev_armed = cur

    stats["forward_after_retest_armed"] = {}
    for h in FORWARD_HORIZONS:
        stats["forward_after_retest_armed"][f"h_{h}"] = {
            "up_mfe_atr":  _summ(armed_fwd[h]["up"]["mfe"]),
            "up_mae_atr":  _summ(armed_fwd[h]["up"]["mae"]),
            "down_mfe_atr":_summ(armed_fwd[h]["down"]["mfe"]),
            "down_mae_atr":_summ(armed_fwd[h]["down"]["mae"]),
        }

    # --- Matrice de confusion : armed -> issue "gagnant" (MFE >= 2 ATR
    # avant MAE >= 1 ATR) vs "perdant" (MAE >= 1 ATR en premier) vs ambigu ---
    # approche : on scanne jusqu'à 50 bars post armed et on regarde ce qui
    # arrive en premier.
    outcomes_armed = {"up": defaultdict(int), "down": defaultdict(int)}
    prev_armed = 0
    for i in range(n):
        cur = int(retest_armed[i])
        if cur != 0 and prev_armed == 0:
            a = float(atr[i])
            if a > 0:
                d = 1 if cur == 1 else -1
                side = "up" if d == 1 else "down"
                entry = float(df["close"].iloc[i])
                tp = entry + d * 2.0 * a
                sl = entry - d * 1.0 * a
                outcome = "ambiguous"
                for k in range(i + 1, min(i + 51, n)):
                    h = float(df["high"].iloc[k]); l = float(df["low"].iloc[k])
                    if d == 1:
                        hit_tp = h >= tp
                        hit_sl = l <= sl
                    else:
                        hit_tp = l <= tp
                        hit_sl = h >= sl
                    if hit_tp and hit_sl:
                        # même bar : prend le pire (SL) — conservateur
                        outcome = "loss"; break
                    if hit_sl:
                        outcome = "loss"; break
                    if hit_tp:
                        outcome = "win"; break
                outcomes_armed[side][outcome] += 1
        prev_armed = cur
    stats["armed_outcomes_2R_1R"] = {
        "up": dict(outcomes_armed["up"]),
        "down": dict(outcomes_armed["down"]),
    }

    # --- Latence fractal -> BOS event : combien de bars séparent le fractal
    # (swing pivot confirmé) du bar où bos_event != 0 qui l'utilise ? ---
    # On mesure simplement la distance entre un bos_event et le dernier
    # up_fractal/down_fractal non-NaN observé avant lui.
    up_fr = df["UP_FRACTAL"].values
    dn_fr = df["DOWN_FRACTAL"].values
    last_up = -1; last_dn = -1
    latencies_up = []; latencies_down = []
    for i in range(n):
        if not np.isnan(up_fr[i]): last_up = i
        if not np.isnan(dn_fr[i]): last_dn = i
        if bos_event[i] == 1 and last_up >= 0:
            latencies_up.append(i - last_up)
        elif bos_event[i] == -1 and last_dn >= 0:
            latencies_down.append(i - last_dn)
    stats["detection_latency_bars"] = {
        "bos_up_bars_since_fractal":   _summ(latencies_up),
        "bos_down_bars_since_fractal": _summ(latencies_down),
    }

    return stats


def write_summary(stats: Dict, path: Path) -> None:
    lines = []
    g = stats["counts_global"]
    rt = stats["retest_transitions"]
    lines.append("# Eval 03 — Smart Money Engine — Stats\n")
    lines.append(f"- Période : {stats['date_start']} → {stats['date_end']}")
    lines.append(f"- Bars totaux : {stats['n_bars']:,}\n")
    lines.append("## Compteurs d'événements\n")
    lines.append("| Evénement | Up/Bull | Down/Bear | /1k bars up | /1k bars down |")
    lines.append("|-----------|---------|-----------|-------------|---------------|")
    f = stats["freq_per_1k_bars"]
    lines.append(f"| BOS | {g['bos_up']} | {g['bos_down']} | {f['bos_up']} | {f['bos_down']} |")
    lines.append(f"| CHOCH | {g['choch_up']} | {g['choch_down']} | {f['choch_up']} | {f['choch_down']} |")
    lines.append(f"| FVG | {g['fvg_up']} | {g['fvg_down']} | {f['fvg_up']} | {f['fvg_down']} |")
    lines.append(f"| Order Block | {g['ob_bullish']} | {g['ob_bearish']} | {f['ob_bullish']} | {f['ob_bearish']} |")
    lines.append("")
    lines.append("## Retest state machine")
    lines.append(f"- Total BOS events : **{rt['total_bos_events']}**")
    lines.append(f"- Awaiting → Armed : **{rt['awaiting_to_armed']}** (taux succès {rt['retest_success_rate']*100:.1f}%)")
    lines.append(f"- Invalidés/timeout : {rt['awaiting_invalidated_or_timeout']}")
    lines.append(f"- Armed up / down : {rt['armed_events_up']} / {rt['armed_events_down']}\n")
    lines.append("## Outcomes armed (TP=2R, SL=1R, fenêtre 50 bars)")
    up_o = stats["armed_outcomes_2R_1R"]["up"]
    dn_o = stats["armed_outcomes_2R_1R"]["down"]
    def wr(o):
        w, l = o.get("win", 0), o.get("loss", 0); tot = w + l
        return f"{w}W / {l}L / {o.get('ambiguous',0)}amb — win rate {100*w/max(1,tot):.1f}%"
    lines.append(f"- LONG armed : {wr(up_o)}")
    lines.append(f"- SHORT armed : {wr(dn_o)}\n")
    lines.append("## Par année — BOS / Armed")
    lines.append("| Année | Bars | BOS↑ | BOS↓ | CHOCH↑ | CHOCH↓ | OB↑ | OB↓ | FVG↑ | FVG↓ |")
    lines.append("|-------|------|------|------|--------|--------|-----|-----|------|------|")
    for y, v in stats["per_year_counts"].items():
        lines.append(f"| {y} | {v['bars']:,} | {v['bos_up']} | {v['bos_down']} | {v['choch_up']} | {v['choch_down']} | {v['ob_up']} | {v['ob_down']} | {v['fvg_up']} | {v['fvg_down']} |")
    lines.append("")
    lines.append("## Forward post-BOS (close du break, MFE/MAE en ATR)")
    for h, b in stats["forward_after_bos"].items():
        lines.append(f"### {h}")
        lines.append("| Dir | n | mean MFE | median MFE | mean MAE | median MAE |")
        lines.append("|-----|---|----------|------------|----------|------------|")
        for side in ("up", "down"):
            mfe = b[f"{side}_mfe_atr"]; mae = b[f"{side}_mae_atr"]
            if mfe and mae:
                lines.append(f"| {side} | {mfe['n']} | {mfe['mean']} | {mfe['median']} | {mae['mean']} | {mae['median']} |")
    lines.append("")
    lines.append("## Latence détection (bars fractal → BOS event)")
    lu = stats["detection_latency_bars"]["bos_up_bars_since_fractal"]
    ld = stats["detection_latency_bars"]["bos_down_bars_since_fractal"]
    if lu: lines.append(f"- BOS↑ : median **{lu['median']}** bars, p25 {lu['p25']}, p75 {lu['p75']} (n={lu['n']})")
    if ld: lines.append(f"- BOS↓ : median **{ld['median']}** bars, p25 {ld['p25']}, p75 {ld['p75']} (n={ld['n']})")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    df = load_ohlcv(CSV_PATH)
    enriched = enrich(df)
    stats = audit(enriched)
    json_path = OUT_DIR / "eval_03_stats.json"
    md_path = OUT_DIR / "eval_03_summary.md"
    json_path.write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")
    write_summary(stats, md_path)
    log.info("Wrote %s and %s", json_path, md_path)


if __name__ == "__main__":
    main()
