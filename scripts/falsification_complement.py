"""
Complement falsification - Sprint 3 MTF, longs-only full sample, PBO, beta full
"""
import json
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

REPORTS = Path("C:/MyPythonProjects/TradingBOT_Agentic/reports")
DATA = Path("C:/MyPythonProjects/TradingBOT_Agentic/data")
OUT = REPORTS / "falsification"

trades = pd.read_csv(REPORTS / "audit_2026_04_30_trades.csv", parse_dates=["ts_in", "ts_out"])

def pf(pnl):
    gw = pnl[pnl > 0].sum()
    gl = -pnl[pnl < 0].sum()
    return gw / gl if gl > 0 else np.nan

# ----------------------------------------------------------------------------
# 1. Beta vs alpha sur le sample COMPLET (pas seulement sub)
print("=== Beta vs alpha full sample ===")
xau = pd.read_csv(DATA / "XAU_15MIN_2019_2026.csv")
xau["Date"] = pd.to_datetime(xau["Date"])
xau = xau.set_index("Date").sort_index()
trades_sorted = trades.sort_values("ts_out").reset_index(drop=True)
trades_sorted["cum_pnl"] = trades_sorted["pnl_usd"].cumsum()

eq_daily = trades_sorted.set_index("ts_out")["cum_pnl"].resample("1D").last().ffill()
px_daily = xau["Close"].resample("1D").last().ffill()
aligned = pd.concat([eq_daily, px_daily], axis=1, join="inner").dropna()
aligned.columns = ["equity", "xau_px"]

# Full
r_lvl_full = np.corrcoef(aligned["equity"], aligned["xau_px"])[0,1]
aligned["eq_ret"] = aligned["equity"].diff()
aligned["px_ret"] = aligned["xau_px"].pct_change()
rr = aligned[["eq_ret", "px_ret"]].dropna()
r_ret_full = np.corrcoef(rr["eq_ret"], rr["px_ret"])[0,1]
print(f"FULL  corr levels: {r_lvl_full:.4f}, corr returns: {r_ret_full:.4f}")

# Pre 2024
pre = aligned.loc[:"2023-12-31"].dropna()
r_lvl_pre = np.corrcoef(pre["equity"], pre["xau_px"])[0,1]
print(f"PRE   corr levels: {r_lvl_pre:.4f}")

# Sub 2024+ (cumul reset a partir de 2024)
sub_eq = aligned.loc["2024-01-01":].copy()
# Reset: equity relative au point 2024-01-01
if len(sub_eq) > 0:
    sub_eq["equity_reset"] = sub_eq["equity"] - sub_eq["equity"].iloc[0]
    r_lvl_sub_reset = np.corrcoef(sub_eq["equity_reset"], sub_eq["xau_px"])[0,1]
    print(f"SUB(2024+) corr levels (reset): {r_lvl_sub_reset:.4f}")

# ----------------------------------------------------------------------------
# 2. Longs-only / Shorts-only sur full sample
print("\n=== Side decomp full sample ===")
for side, name in [(1, "Long"), (-1, "Short")]:
    s = trades[trades.side == side]
    print(f"  {name}: n={len(s)}, PnL={s.pnl_usd.sum():.0f}, PF={pf(s.pnl_usd.values):.4f}, WR={s.pnl_usd.gt(0).mean():.3f}")

# ----------------------------------------------------------------------------
# 3. Sprint 3 MTF - simulation naive
# Hypothese: MTF naif coupe 50% trades. Question: proportion gagnants/perdants coupes
# Si MTF aligne avec direction du HTF moving avg (ex EMA50 H4):
#   - Longs gardes ssi close > EMA50_H4
#   - Shorts gardes ssi close < EMA50_H4
print("\n=== Stress Sprint 3 MTF ===")

# H4 EMA50 = 16 barres M15 EMA puis aligner sur trade ts_in
# Calcul EMA50 H4 sur close
xau_h4 = xau["Close"].resample("4h").last().dropna()
ema50_h4 = xau_h4.ewm(span=50, adjust=False).mean()
# Reindex sur M15 (forward fill - on prend la valeur du H4 cloturé le plus récent)
ema50_h4_15 = ema50_h4.reindex(xau.index, method="ffill")

# Aligner aux trades
trades_x = trades.set_index("ts_in").join(ema50_h4_15.rename("ema50_h4"))
trades_x["close_at_entry"] = trades_x["entry"]
trades_x["mtf_ok"] = (
    ((trades_x.side == 1) & (trades_x.close_at_entry > trades_x.ema50_h4))
    | ((trades_x.side == -1) & (trades_x.close_at_entry < trades_x.ema50_h4))
)
mtf_ok = trades_x["mtf_ok"].fillna(False)
print(f"  Trades align MTF (close vs EMA50_H4): {mtf_ok.sum()}/{len(trades_x)} = {mtf_ok.mean():.3f}")
filtered = trades_x[mtf_ok].reset_index()
removed = trades_x[~mtf_ok].reset_index()
print(f"  PF retained: {pf(filtered.pnl_usd.values):.4f}")
print(f"  PF removed:  {pf(removed.pnl_usd.values):.4f}")
print(f"  PnL retained: {filtered.pnl_usd.sum():.0f}")
print(f"  PnL removed:  {removed.pnl_usd.sum():.0f}")

# Sprint 3 sub 2024-2026
sub24 = trades_x[trades_x.index >= "2024-01-01"]
mtf_ok_sub = sub24["mtf_ok"].fillna(False)
filt24 = sub24[mtf_ok_sub]
rem24 = sub24[~mtf_ok_sub]
print(f"  --- 2024-2026 only ---")
print(f"  Filtered: n={len(filt24)}, PF={pf(filt24.pnl_usd.values):.4f}, PnL={filt24.pnl_usd.sum():.0f}")
print(f"  Removed:  n={len(rem24)}, PF={pf(rem24.pnl_usd.values):.4f}, PnL={rem24.pnl_usd.sum():.0f}")

# ----------------------------------------------------------------------------
# 4. Profit factor par annee + bootstrap CI
print("\n=== PF par annee avec CI bootstrap ===")
for y in range(2019, 2027):
    yt = trades[trades.ts_in.dt.year == y]
    if len(yt) < 50:
        print(f"  {y}: n={len(yt)} (insuffisant)")
        continue
    pf_y = pf(yt.pnl_usd.values)
    # Bootstrap quick
    rng = np.random.default_rng(42)
    n = len(yt)
    pfs = []
    for _ in range(2000):
        idx = rng.integers(0, n, n)
        s = yt.pnl_usd.values[idx]
        gw = s[s > 0].sum(); gl = -s[s < 0].sum()
        if gl > 0: pfs.append(gw/gl)
    ci = np.percentile(pfs, [2.5, 97.5])
    print(f"  {y}: n={n}, PF={pf_y:.3f}, CI95=[{ci[0]:.3f}, {ci[1]:.3f}]")

# ----------------------------------------------------------------------------
# 5. PBO (Probability of Backtest Overfitting) - approximation
# PBO = P(median rank de la strat sur OOS < N/2 | sur IS strat etait rank max)
# Bailey & Lopez de Prado 2015 "Probability of backtest overfitting"
# Sans avoir N strategies candidates a comparer, on peut estimer pour
# 1 seule strategie via signal-to-noise ratio et longueur sample
# Approche: split temporal trades en 2 (CSCV simplifiee 2 splits)
print("\n=== PBO simplifie (CSCV 2 partitions) ===")
trades_sorted_ts = trades.sort_values("ts_in").reset_index(drop=True)
half = len(trades_sorted_ts) // 2
A = trades_sorted_ts.iloc[:half]
B = trades_sorted_ts.iloc[half:]
pf_A = pf(A.pnl_usd.values)
pf_B = pf(B.pnl_usd.values)
print(f"  Split temporel A (first half): n={len(A)}, PF={pf_A:.4f}")
print(f"  Split temporel B (second half): n={len(B)}, PF={pf_B:.4f}")
print(f"  -> divergence PF entre les deux moities = {abs(pf_A - pf_B):.4f}")

# Simulation: 8 features booleennes + score - PBO theorique pour cette config
# Avec N_features=8, T=2363 trades, ratio T/N = 295. Suffisant pour LightGBM ?
# Rule of thumb Lopez de Prado: T/N >= 50 acceptable, >= 100 conf, >= 200 robuste
# Mais avec features faibles (R^2_indiv ~ 0), le bruit domine
print(f"\n  Ratio trades/features = {2363/8:.0f} (>= 100 = OK pour GBM)")
print(f"  Mais R^2 OLS multivar = 0.0007 -> SNR effectif minuscule")

# ----------------------------------------------------------------------------
# 6. White Reality Check / Hansen SPA approche simplifiee
# "Le sub-segment 2024-2026 est-il distinguable du bruit ?"
# H0: le PF cumule sur n'importe quelle fenetre 28-mois ressemble a celui observe
print("\n=== White-Reality-Check approximation: PF rolling 28-mois ===")
trades_sorted_ts["month"] = trades_sorted_ts["ts_in"].dt.to_period("M").astype(str)
month_pnl = trades_sorted_ts.groupby("month")["pnl_usd"].sum()
month_pnl.index = pd.to_datetime(month_pnl.index)
month_pnl = month_pnl.sort_index()

# rolling PF sur 28 mois (sur fenetres avec PnL accumule)
window = 28
pfs_window = []
for i in range(0, len(month_pnl) - window + 1):
    sub_idx = month_pnl.index[i:i+window]
    # Trades dans cette fenetre
    start = sub_idx[0]
    end = sub_idx[-1] + pd.offsets.MonthEnd(0)
    sub_trades = trades_sorted_ts[(trades_sorted_ts.ts_in >= start) & (trades_sorted_ts.ts_in <= end)]
    if len(sub_trades) < 200: continue
    p = pf(sub_trades.pnl_usd.values)
    pfs_window.append((str(start.date()), len(sub_trades), p))

dfw = pd.DataFrame(pfs_window, columns=["start", "n", "pf"])
print(dfw.tail(10).to_string(index=False))
print(f"\n  Max PF rolling 28-mois: {dfw.pf.max():.4f} (start={dfw.loc[dfw.pf.idxmax(), 'start']})")
print(f"  PF sub 2024-2026 = 1.275 -> rang centile {(dfw.pf <= 1.275).mean()*100:.1f}%")

# ----------------------------------------------------------------------------
# 7. Test "drop CHOCH+RSI_div" - estimation PF post-S1
# Replique le filtrage sample existant (toutes lignes sont deja les trades selectionnes)
# Le rapport propose: retirer les composants du score -> moins de trades passent le seuil 65
# Sur ce ledger on ne peut pas savoir lesquels SERAIENT pris si on retire ces composants
# car le seuil shift
# Mais on peut calculer: scores ajustes = score actuel - 12.5*(CHOCH + RSI_div)
trades["score_no_choch"] = trades.score_in - 12.5 * trades.c_choch
trades["score_no_rsi"] = trades.score_in - 12.5 * trades.c_rsi_div
trades["score_no_both"] = trades.score_in - 12.5 * (trades.c_choch + trades.c_rsi_div)

# Si seuil reste 65, mais avec moins de poids dispo (8->6 composants = max 75)
# il faut adapter - rapport propose "75/(6*12.5) = 100% des 6 composants restants"
# Ie threshold_normalise = 65/100 = 0.65, sur 6 composants -> need >= 4
six_comps = ["c_bos", "c_ob", "c_fvg", "c_retest", "c_regime", "c_news_ok"]
trades["count_6"] = trades[six_comps].sum(axis=1)
# Need >= 4 of 6 for entry (preserve relative threshold)
for thresh in [4, 5, 6]:
    keep = trades[trades.count_6 >= thresh]
    pf_k = pf(keep.pnl_usd.values)
    print(f"  Drop CHOCH+RSI_div, keep when sum(6 comps)>={thresh}: n={len(keep)}, PF={pf_k:.4f}, PnL={keep.pnl_usd.sum():.0f}")

# ----------------------------------------------------------------------------
# 8. P(simulated PF >= 1.20 sur 2024-2026 par chance, sous H0 = mean R = 0)
# Approche: bootstrap 5000 reechantillonnages des R post-2024 mais avec mean = 0
print("\n=== P(PF_sub >= 1.20 | shuffled trades 2024+) ===")
sub_post = trades[trades.ts_in >= "2024-01-01"]
pnl_post = sub_post.pnl_usd.values
# Recentrer (mean=0) puis bootstrap PF
pnl_centered = pnl_post - pnl_post.mean()
rng = np.random.default_rng(7)
n_boot = 5000
pfs_h0 = []
for _ in range(n_boot):
    idx = rng.integers(0, len(pnl_centered), len(pnl_centered))
    s = pnl_centered[idx]
    gw = s[s > 0].sum(); gl = -s[s < 0].sum()
    if gl > 0: pfs_h0.append(gw/gl)
pfs_h0 = np.array(pfs_h0)
print(f"  PF sub observe = {pf(pnl_post):.4f}")
print(f"  Distribution PF sous H0 mean=0: median={np.median(pfs_h0):.4f}, p99={np.percentile(pfs_h0, 99):.4f}")
print(f"  P(PF_H0 >= 1.20) = {(pfs_h0 >= 1.20).mean():.4f}  -> (faible = rejet H0)")
print(f"  P(PF_H0 >= PF_obs) = {(pfs_h0 >= pf(pnl_post)).mean():.4f}")
