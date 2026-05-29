"""
Forensic L1 - Timeout sweep
============================
Re-run le backtest avec max_lifetime_bars in {12, 18, 24, 30, 36, 48, 64}.
Tout le reste identique a quant_audit_2026_04_30.py.

Note IMPORTANTE de coupling:
  sl_dist = sl_atr_mult * vol * sqrt(max_lifetime_bars)
  Le SL et TP sont COUPLES au timeout via sqrt(timeout). Donc cette
  sweep n'isole pas le timeout pur, elle reflete la coupling actuelle
  de production. C'est la metrique pertinente pour decision Sprint 1.

Usage : python scripts/forensics_timeout_sweep.py
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
# 1. Charger donnees + computer features UNE FOIS (independant du timeout)
# ----------------------------------------------------------------------------
print("[L1] Chargement + features...")
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
print(f"[L1] features OK : {len(df)} barres")

# ----------------------------------------------------------------------------
# 2. Bootstrap PF
# ----------------------------------------------------------------------------
def bootstrap_pf(pnl, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(pnl)
    pfs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        s = pnl[idx]
        gw = s[s > 0].sum(); gl = -s[s < 0].sum()
        pfs[i] = gw / gl if gl > 0 else np.nan
    pfs = pfs[~np.isnan(pfs)]
    return float(np.percentile(pfs, 2.5)), float(np.percentile(pfs, 97.5))

# ----------------------------------------------------------------------------
# 3. Sweep
# ----------------------------------------------------------------------------
TIMEOUTS = [12, 18, 24, 30, 36, 48, 64]
results = []
trades_per_timeout = {}

for tmo in TIMEOUTS:
    print(f"\n[L1] timeout = {tmo} bars...")
    base.CFG["max_lifetime_bars"] = tmo
    # IMPORTANT: re-seed le rng pour reproductibilite identique entre runs
    np.random.seed(base.CFG["random_seed"])
    trades, equity = base.simulate(df)
    metrics = base.compute_metrics(trades, equity)
    pnl_arr = np.array([t.pnl_usd for t in trades])
    ci_lo, ci_hi = bootstrap_pf(pnl_arr, n_boot=2000, seed=42)

    row = {
        "timeout_bars": tmo,
        "n_trades": metrics["n_trades"],
        "wr": metrics["win_rate"],
        "pf": metrics["profit_factor"],
        "pf_ci95_lo": ci_lo,
        "pf_ci95_hi": ci_hi,
        "sharpe": metrics["sharpe"],
        "sortino": metrics["sortino"],
        "max_dd_pct": metrics["max_dd_pct"],
        "max_dd_usd": metrics["max_dd_usd"],
        "expectancy_r": metrics["expectancy_r"],
        "final_capital": metrics["final_capital"],
        "return_pct": metrics["return_pct"],
        "n_timeout": sum(1 for t in trades if t.exit_reason == "timeout"),
        "n_sl": sum(1 for t in trades if t.exit_reason == "sl"),
        "n_tp": sum(1 for t in trades if t.exit_reason == "tp"),
        "n_opp": sum(1 for t in trades if t.exit_reason == "opposite"),
    }
    results.append(row)
    trades_per_timeout[tmo] = trades
    print(f"  PF={row['pf']:.4f} CI=[{ci_lo:.3f},{ci_hi:.3f}] n={row['n_trades']} dd={row['max_dd_pct']*100:.1f}%")

df_results = pd.DataFrame(results)
df_results.to_csv(OUT / "L1_timeout_sweep.csv", index=False)

# ----------------------------------------------------------------------------
# 4. Detection pic suspect a 24
# ----------------------------------------------------------------------------
pf_24 = df_results.loc[df_results.timeout_bars == 24, "pf"].iloc[0]
others = df_results[df_results.timeout_bars != 24]["pf"].values
above_24 = (others > pf_24).sum()
print(f"\n[L1] PF a timeout=24 : {pf_24:.4f}")
print(f"[L1] # autres timeouts avec PF > 24 : {above_24}/6")
if above_24 == 0:
    verdict_pic = "PIC SUSPECT a 24 -> drapeau rouge overfitting"
elif above_24 <= 2:
    verdict_pic = "Pic leger a 24 -> overfit possible mais pas certain"
else:
    verdict_pic = "Pas de pic - 24 n'est pas optimal -> pas d'overfitting du timeout"
print(f"[L1] {verdict_pic}")

# Test monotonie : difference entre PF max et PF min sur la sweep
pf_range = df_results.pf.max() - df_results.pf.min()
print(f"[L1] PF range sur sweep : {pf_range:.4f}")
if pf_range < 0.05:
    verdict_flat = "PLAT - le timeout n'est pas un parametre sensible"
elif pf_range < 0.15:
    verdict_flat = "Sensitivite moderee"
else:
    verdict_flat = "FORTE sensitivite - couplage SL/TP via sqrt(timeout) domine"
print(f"[L1] {verdict_flat}")

# ----------------------------------------------------------------------------
# 5. Save JSON summary
# ----------------------------------------------------------------------------
summary = {
    "timeouts_tested": TIMEOUTS,
    "results": results,
    "verdict_pic_at_24": verdict_pic,
    "verdict_flatness": verdict_flat,
    "pf_range": float(pf_range),
    "pf_at_24": float(pf_24),
    "n_timeouts_above_24": int(above_24),
    "note_coupling": (
        "sl_dist = sl_atr_mult * vol * sqrt(max_lifetime_bars). "
        "Le sweep mesure l'effet COUPLE timeout x SL_size, pas timeout pur."
    ),
}
with open(OUT / "L1_timeout_sweep.json", "w") as f:
    json.dump(summary, f, indent=2, default=float)

print(f"\n[L1] Resultats sauves dans {OUT / 'L1_timeout_sweep.csv'}")
print(df_results.to_string(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else x))
