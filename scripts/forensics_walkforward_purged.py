"""
Forensic L2 - Walk-forward purge
=================================
Train  : 2019-01-01 -> 2022-12-31  (52 mois)
Embargo: 2023-01-01 -> 2023-01-05
Test   : 2023-01-06 -> 2024-12-31  (24 mois)
Embargo: 2025-01-01 -> 2025-01-05
Holdout: 2025-01-06 -> 2026-04-29  (16 mois)

Le scoring 8-composants est rule-based et n'a aucun parametre fitte.
Le HMM n'est pas utilise ici (HAR-RV uniquement). Le calendrier news
contient des events historiques avec timestamps reels => pas de fuite.

Donc "walk-forward purge" = simuler sur 3 segments separes, comparer
PF/Sharpe/MaxDD pour detecter degradation IS->OOS.

Important : le SL/TP scaling depend de la volatilite ROLLING (HAR-RV),
qui se recalcule a chaque barre sans re-fit global. OK.

Aussi : on ajoute un sweep timeout sur le HOLDOUT pour valider que le
findding L1 (PF monotone croissant) tient OOS.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import quant_audit_2026_04_30 as base

OUT = ROOT / "reports" / "forensics"
OUT.mkdir(exist_ok=True, parents=True)

# ----------------------------------------------------------------------------
# 1. Charger + features (1 fois)
# ----------------------------------------------------------------------------
print("[L2] Chargement + features...")
df, cal = base.load_data()
df["atr"] = base.compute_atr(df, base.CFG["atr_period"])
df = base.detect_swings(df, lookback=5)
df = base.detect_bos(df)
df = base.detect_choch(df)
df = base.detect_order_blocks(df)
df = base.detect_fvg(df, atr_mult=0.4)
df = base.detect_retest(df, atr_tol=0.25)
df = base.detect_rsi_divergence(df, lookback=14)
df = base.detect_regime(df)
df = base.detect_news_blackout(df, cal)
df = base.har_rv_forecast(df)
df = base.compute_confluence_score(df)
print(f"[L2] features OK : {len(df)} barres")

# ----------------------------------------------------------------------------
# 2. Bootstrap PF
# ----------------------------------------------------------------------------
def bootstrap_pf(pnl, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(pnl)
    if n < 10:
        return float("nan"), float("nan")
    pfs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        s = pnl[idx]
        gw = s[s > 0].sum(); gl = -s[s < 0].sum()
        pfs[i] = gw / gl if gl > 0 else np.nan
    pfs = pfs[~np.isnan(pfs)]
    if len(pfs) == 0:
        return float("nan"), float("nan")
    return float(np.percentile(pfs, 2.5)), float(np.percentile(pfs, 97.5))

# ----------------------------------------------------------------------------
# 3. Definir les segments
# ----------------------------------------------------------------------------
SEGMENTS = [
    {"name": "Train",   "start": "2019-01-01", "end": "2022-12-31"},
    {"name": "Embargo1","start": "2023-01-01", "end": "2023-01-05"},
    {"name": "Test",    "start": "2023-01-06", "end": "2024-12-31"},
    {"name": "Embargo2","start": "2025-01-01", "end": "2025-01-05"},
    {"name": "Holdout", "start": "2025-01-06", "end": "2026-04-29"},
]

# ----------------------------------------------------------------------------
# 4. Run par segment - re-utiliser simulate avec filtre par dates
# Le simulate() de base parcourt tout df. On peut tronquer df par segment.
# Mais la sequence d'etats StateMachine se reset entre segments, ce qui est
# correct : on ne porte pas l'etat d'un segment a l'autre.
# ----------------------------------------------------------------------------
def run_segment(df_full, name, start, end, cfg_overrides=None):
    df_seg = df_full[(df_full.index >= start) & (df_full.index <= end)].copy()
    if cfg_overrides:
        for k, v in cfg_overrides.items():
            base.CFG[k] = v
    np.random.seed(base.CFG["random_seed"])
    trades, equity = base.simulate(df_seg)
    metrics = base.compute_metrics(trades, equity)
    pnl_arr = np.array([t.pnl_usd for t in trades]) if trades else np.array([])
    ci_lo, ci_hi = bootstrap_pf(pnl_arr, n_boot=2000, seed=42)
    return {
        "segment": name,
        "start": start,
        "end": end,
        "n_bars": len(df_seg),
        "n_trades": metrics.get("n_trades", 0),
        "wr": metrics.get("win_rate", float("nan")),
        "pf": metrics.get("profit_factor", float("nan")),
        "pf_ci95_lo": ci_lo,
        "pf_ci95_hi": ci_hi,
        "sharpe": metrics.get("sharpe", float("nan")),
        "sortino": metrics.get("sortino", float("nan")),
        "max_dd_pct": metrics.get("max_dd_pct", float("nan")),
        "max_dd_usd": metrics.get("max_dd_usd", float("nan")),
        "expectancy_r": metrics.get("expectancy_r", float("nan")),
        "return_pct": metrics.get("return_pct", float("nan")),
        "final_capital": metrics.get("final_capital", float("nan")),
    }, trades

# Reset CFG max_lifetime_bars a 24 (defaut original) pour ce livrable
base.CFG["max_lifetime_bars"] = 24

# ----------------------------------------------------------------------------
# 5. Run principal : Train, Test, Holdout
# ----------------------------------------------------------------------------
print("\n[L2] Walk-forward purge avec timeout=24 (config production actuelle)")
results_main = []
trades_main = {}
for seg in SEGMENTS:
    if seg["name"].startswith("Embargo"):
        continue  # skip embargo periods
    print(f"\n[L2] segment {seg['name']} {seg['start']} -> {seg['end']}")
    row, trades = run_segment(df, seg["name"], seg["start"], seg["end"])
    results_main.append(row)
    trades_main[seg["name"]] = trades
    print(f"  n_trades={row['n_trades']}, PF={row['pf']:.4f} CI=[{row['pf_ci95_lo']:.3f}, {row['pf_ci95_hi']:.3f}]")
    print(f"  Sharpe={row['sharpe']:.3f}, MaxDD={row['max_dd_pct']*100:.1f}%, return={row['return_pct']*100:.1f}%")

df_main = pd.DataFrame(results_main)
df_main.to_csv(OUT / "L2_walkforward_main.csv", index=False)

# ----------------------------------------------------------------------------
# 6. Sub-decomposition Test (2023-2024) par annee
# ----------------------------------------------------------------------------
print("\n[L2] Sub-decomposition annuelle des segments")
sub_yearly = []
for seg in [{"name": "Train_2019", "start": "2019-01-01", "end": "2019-12-31"},
            {"name": "Train_2020", "start": "2020-01-01", "end": "2020-12-31"},
            {"name": "Train_2021", "start": "2021-01-01", "end": "2021-12-31"},
            {"name": "Train_2022", "start": "2022-01-01", "end": "2022-12-31"},
            {"name": "Test_2023",  "start": "2023-01-06", "end": "2023-12-31"},
            {"name": "Test_2024",  "start": "2024-01-01", "end": "2024-12-31"},
            {"name": "Holdout_2025", "start": "2025-01-06", "end": "2025-12-31"},
            {"name": "Holdout_2026", "start": "2026-01-01", "end": "2026-04-29"}]:
    row, _ = run_segment(df, seg["name"], seg["start"], seg["end"])
    sub_yearly.append(row)
    print(f"  {seg['name']}: n={row['n_trades']}, PF={row['pf']:.3f}, MaxDD={row['max_dd_pct']*100:.1f}%")

df_yearly = pd.DataFrame(sub_yearly)
df_yearly.to_csv(OUT / "L2_walkforward_yearly.csv", index=False)

# ----------------------------------------------------------------------------
# 7. Verifier que finding L1 (timeout monotone croissant) tient OOS
# ----------------------------------------------------------------------------
print("\n[L2] Verification finding L1 sur Holdout uniquement")
holdout_sweep = []
for tmo in [12, 18, 24, 30, 36, 48, 64]:
    base.CFG["max_lifetime_bars"] = tmo
    row, _ = run_segment(df, f"Holdout_tmo{tmo}", "2025-01-06", "2026-04-29")
    row["timeout_bars"] = tmo
    holdout_sweep.append(row)
    print(f"  Holdout tmo={tmo}: n={row['n_trades']}, PF={row['pf']:.3f} CI=[{row['pf_ci95_lo']:.3f}, {row['pf_ci95_hi']:.3f}]")

df_holdout_sweep = pd.DataFrame(holdout_sweep)
df_holdout_sweep.to_csv(OUT / "L2_holdout_timeout_sweep.csv", index=False)

# Reset
base.CFG["max_lifetime_bars"] = 24

# ----------------------------------------------------------------------------
# 8. Verdicts
# ----------------------------------------------------------------------------
pf_train  = next(r["pf"] for r in results_main if r["segment"] == "Train")
pf_test   = next(r["pf"] for r in results_main if r["segment"] == "Test")
pf_holdout = next(r["pf"] for r in results_main if r["segment"] == "Holdout")

if pf_train > pf_test > pf_holdout:
    verdict_degradation = "DEGRADATION MONOTONE (signe d'overfit ou regime change)"
elif pf_holdout > pf_test > pf_train:
    verdict_degradation = "AMELIORATION MONOTONE (regime change favorable, pas d'overfit)"
else:
    verdict_degradation = f"Pas de monotonie : Train={pf_train:.3f}, Test={pf_test:.3f}, Holdout={pf_holdout:.3f}"

# Test si holdout franchit 1.20 cible commerciale
holdout_above_120 = pf_holdout >= 1.20
holdout_ci_excludes_120 = next(r["pf_ci95_lo"] for r in results_main if r["segment"] == "Holdout") > 1.20

print(f"\n[L2] Verdict degradation: {verdict_degradation}")
print(f"[L2] Holdout PF >= 1.20 ? {holdout_above_120}, CI lo > 1.20 ? {holdout_ci_excludes_120}")

# ----------------------------------------------------------------------------
# 9. Save summary
# ----------------------------------------------------------------------------
summary = {
    "segments_main": results_main,
    "segments_yearly": sub_yearly,
    "holdout_timeout_sweep": holdout_sweep,
    "pf_train": float(pf_train),
    "pf_test": float(pf_test),
    "pf_holdout": float(pf_holdout),
    "verdict_degradation": verdict_degradation,
    "holdout_pf_above_120": bool(holdout_above_120),
    "holdout_ci_lo_above_120": bool(holdout_ci_excludes_120),
}
with open(OUT / "L2_walkforward.json", "w") as f:
    json.dump(summary, f, indent=2, default=float)

print(f"\n[L2] DONE. Files: L2_walkforward_main.csv, L2_walkforward_yearly.csv, L2_holdout_timeout_sweep.csv, L2_walkforward.json")
print("\n[L2] Tableau principal:")
print(df_main.to_string(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else x))
