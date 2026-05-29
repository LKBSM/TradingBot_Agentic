"""
Forensic L3 - Decomposition Long/Short pre/post 2024
=====================================================
A partir du ledger existant audit_2026_04_30_trades.csv.
Calcule pour chaque cellule (periode x side) :
  n, WR, PF, PnL, Sharpe (mensuel ann), MaxDD, bootstrap CI 95% sur PF
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OUT = REPORTS / "forensics"
OUT.mkdir(exist_ok=True, parents=True)

trades = pd.read_csv(REPORTS / "audit_2026_04_30_trades.csv", parse_dates=["ts_in", "ts_out"])

def pf(pnl):
    g = pnl[pnl > 0].sum(); l = -pnl[pnl < 0].sum()
    return g / l if l > 0 else float("nan")

def bootstrap_pf(pnl, n_boot=5000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(pnl)
    if n < 10:
        return float("nan"), float("nan")
    pfs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        s = pnl[idx]
        gw = s[s > 0].sum(); gl = -s[s < 0].sum()
        if gl > 0: pfs.append(gw/gl)
    if not pfs: return float("nan"), float("nan")
    return float(np.percentile(pfs, 2.5)), float(np.percentile(pfs, 97.5))

def metrics_for(sub):
    if len(sub) == 0:
        return None
    pnl = sub.pnl_usd.values
    pf_v = pf(pnl)
    ci_lo, ci_hi = bootstrap_pf(pnl, 5000)
    wr = (pnl > 0).mean()
    avg_r = sub.r_realized.mean()
    # Equity simulee a partir des trades sortis dans l'ordre temporel
    sub_sorted = sub.sort_values("ts_out").reset_index(drop=True)
    cap0 = 10000.0
    eq = cap0 + sub_sorted.pnl_usd.cumsum()
    # Sharpe mensuel
    sub_sorted["ym"] = sub_sorted.ts_out.dt.to_period("M")
    monthly = sub_sorted.groupby("ym")["pnl_usd"].sum()
    eq_monthly = (cap0 + monthly.cumsum())
    eq_ret = eq_monthly.pct_change().dropna()
    if len(eq_ret) > 1 and eq_ret.std() > 0:
        sharpe = eq_ret.mean() / eq_ret.std() * np.sqrt(12)
    else:
        sharpe = float("nan")
    # MaxDD
    running_max = eq.cummax()
    dd_pct = ((eq - running_max) / running_max).min()
    # Coverage period
    period_days = (sub.ts_out.max() - sub.ts_in.min()).days if len(sub) > 0 else 0
    return {
        "n": int(len(sub)),
        "wr": float(wr),
        "pf": float(pf_v),
        "pf_ci_lo": ci_lo,
        "pf_ci_hi": ci_hi,
        "pnl_usd": float(sub.pnl_usd.sum()),
        "avg_r": float(avg_r),
        "sharpe_monthly_ann": float(sharpe),
        "max_dd_pct": float(dd_pct),
        "period_days": int(period_days),
    }

# ----------------------------------------------------------------------------
PERIODES = [
    ("2019-2023", "2019-01-01", "2023-12-31"),
    ("2024-2026", "2024-01-01", "2026-12-31"),
    ("FULL",      "2019-01-01", "2026-12-31"),
]
SIDES = [(1, "Long"), (-1, "Short")]

results = []
for pname, pstart, pend in PERIODES:
    for side, sname in SIDES:
        sub = trades[(trades.ts_in >= pstart) & (trades.ts_in <= pend) & (trades.side == side)]
        m = metrics_for(sub)
        if m is None:
            continue
        m["periode"] = pname
        m["side"] = sname
        results.append(m)
    # Combine
    sub_both = trades[(trades.ts_in >= pstart) & (trades.ts_in <= pend)]
    m = metrics_for(sub_both)
    m["periode"] = pname
    m["side"] = "BOTH"
    results.append(m)

df_res = pd.DataFrame(results)
cols_order = ["periode", "side", "n", "wr", "pf", "pf_ci_lo", "pf_ci_hi", "pnl_usd",
              "avg_r", "sharpe_monthly_ann", "max_dd_pct"]
df_res = df_res[cols_order]
df_res.to_csv(OUT / "L3_decomp_side.csv", index=False)
print("\n[L3] Decomposition complete:")
print(df_res.to_string(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else x))

# ----------------------------------------------------------------------------
# Verdicts explicites
# ----------------------------------------------------------------------------
print("\n[L3] VERDICTS:")

# Q1: long-only deguise sur 2024-2026 ?
long_2024 = df_res[(df_res.periode == "2024-2026") & (df_res.side == "Long")].iloc[0]
short_2024 = df_res[(df_res.periode == "2024-2026") & (df_res.side == "Short")].iloc[0]
ratio_pnl = long_2024.pnl_usd / abs(short_2024.pnl_usd) if short_2024.pnl_usd != 0 else float("inf")
print(f"\nQ1 - long-only deguise sur 2024-2026 ?")
print(f"  Long 2024-2026  : n={long_2024.n}, PF={long_2024.pf:.3f}, PnL={long_2024.pnl_usd:.0f}")
print(f"  Short 2024-2026 : n={short_2024.n}, PF={short_2024.pf:.3f}, PnL={short_2024.pnl_usd:.0f}")
print(f"  Ratio PnL Long/|Short| = {ratio_pnl:.2f}")
if long_2024.pf > 1.20 and short_2024.pf < 1.05:
    verdict_q1 = "OUI - long-only deguise (Long PF>1.20, Short PF<1.05)"
elif long_2024.pf > 1.20 and short_2024.pf > 1.05:
    verdict_q1 = "NON - les deux cotes contribuent"
else:
    verdict_q1 = f"Ambigu - Long PF={long_2024.pf:.2f}, Short PF={short_2024.pf:.2f}"
print(f"  -> {verdict_q1}")

# Q2: module short cassé en absolu ou seulement en bull ?
short_full = df_res[(df_res.periode == "FULL") & (df_res.side == "Short")].iloc[0]
short_pre = df_res[(df_res.periode == "2019-2023") & (df_res.side == "Short")].iloc[0]
print(f"\nQ2 - module Short casse ?")
print(f"  Short 2019-2023 : PF={short_pre.pf:.3f} CI=[{short_pre.pf_ci_lo:.3f},{short_pre.pf_ci_hi:.3f}]")
print(f"  Short 2024-2026 : PF={short_2024.pf:.3f} CI=[{short_2024.pf_ci_lo:.3f},{short_2024.pf_ci_hi:.3f}]")
print(f"  Short FULL      : PF={short_full.pf:.3f} CI=[{short_full.pf_ci_lo:.3f},{short_full.pf_ci_hi:.3f}]")
if short_pre.pf < 0.7 and short_2024.pf < 1.0:
    verdict_q2 = "Module Short structurellement casse (PF<1 dans tous regimes)"
elif short_pre.pf < 0.8 and short_2024.pf > 1.0:
    verdict_q2 = "Short defaillant en bear/range, OK en bull (anormal pour XAU)"
else:
    verdict_q2 = f"Mixte - Short pre={short_pre.pf:.2f}, post={short_2024.pf:.2f}"
print(f"  -> {verdict_q2}")

# Q3: long-only sur 2019-2023 (bear/range) - PF ?
long_pre = df_res[(df_res.periode == "2019-2023") & (df_res.side == "Long")].iloc[0]
print(f"\nQ3 - long-only sur 2019-2023 ?")
print(f"  Long 2019-2023  : PF={long_pre.pf:.3f} CI=[{long_pre.pf_ci_lo:.3f},{long_pre.pf_ci_hi:.3f}]")
print(f"  PnL Long 2019-2023 : {long_pre.pnl_usd:.0f} USD sur n={long_pre.n} trades")
if long_pre.pf >= 1.0:
    verdict_q3 = "Long-only 2019-2023 PROFITABLE - edge directionnel reel"
elif long_pre.pf >= 0.85:
    verdict_q3 = "Long-only 2019-2023 quasi-breakeven - edge faible"
else:
    verdict_q3 = "Long-only 2019-2023 NON-profitable - pas d'edge directionnel"
print(f"  -> {verdict_q3}")

# Save final
out_summary = {
    "table": df_res.to_dict(orient="records"),
    "verdict_q1_long_only_disguise_2024_2026": verdict_q1,
    "verdict_q2_short_module": verdict_q2,
    "verdict_q3_long_only_2019_2023": verdict_q3,
}
with open(OUT / "L3_decomp_side.json", "w") as f:
    json.dump(out_summary, f, indent=2, default=float)
print(f"\n[L3] DONE. Files: L3_decomp_side.csv, L3_decomp_side.json")
