"""
Falsification quant senior - audit_2026_04_30
Reviewer: PhD finance/stats, 12 ans buy-side systematique
Job: tuer le rapport, pas le confirmer
"""
import json
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

REPORTS = Path("C:/MyPythonProjects/TradingBOT_Agentic/reports")
DATA = Path("C:/MyPythonProjects/TradingBOT_Agentic/data")
OUT = REPORTS / "falsification"
OUT.mkdir(exist_ok=True, parents=True)

RNG = np.random.default_rng(42)

# Charger ledger + summary
trades = pd.read_csv(REPORTS / "audit_2026_04_30_trades.csv", parse_dates=["ts_in", "ts_out"])
with open(REPORTS / "audit_2026_04_30_summary.json", "r") as f:
    summary = json.load(f)

print(f"=== Ledger: {len(trades)} trades, {trades['ts_in'].min()} -> {trades['ts_in'].max()} ===")
print(f"Cols: {list(trades.columns)}")
print(f"PF observe summary={summary['metrics']['profit_factor']:.4f}")
gw = trades.loc[trades.pnl_usd > 0, "pnl_usd"].sum()
gl = -trades.loc[trades.pnl_usd < 0, "pnl_usd"].sum()
print(f"PF recompute   ={gw/gl:.4f} (gw={gw:.0f}, gl={gl:.0f})")

# =============================================================================
# LIVRABLE 1 - FALSIFICATION STATISTIQUE
# =============================================================================
print("\n" + "="*80)
print("LIVRABLE 1 - FALSIFICATION STATISTIQUE")
print("="*80)

def bootstrap_pf(pnl, n_boot=5000, seed=42):
    """CI 95% bootstrap pour profit factor."""
    rng = np.random.default_rng(seed)
    n = len(pnl)
    pfs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        sample = pnl[idx]
        gw = sample[sample > 0].sum()
        gl = -sample[sample < 0].sum()
        pfs[i] = gw / gl if gl > 0 else np.nan
    pfs = pfs[~np.isnan(pfs)]
    return np.percentile(pfs, [2.5, 50, 97.5]), pfs

# ---- A. Bootstrap PF
pnl_all = trades["pnl_usd"].values
ci_pf, boot_pf = bootstrap_pf(pnl_all, 5000)
pf_obs = gw / gl
prob_pf_gte_1 = (boot_pf >= 1.0).mean()
print(f"\nA) Bootstrap PF 95% CI: [{ci_pf[0]:.4f}, {ci_pf[2]:.4f}], median={ci_pf[1]:.4f}")
print(f"   PF observe = {pf_obs:.4f}")
print(f"   P(PF_resample >= 1.0) = {prob_pf_gte_1:.4f}")
print(f"   CI exclut 1.0 ? {ci_pf[2] < 1.0}")

# ---- B. Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014)
# DSR = norm.cdf( (SR_obs - E[max SR_N]) * sqrt(T-1) / sqrt(1 - skew*SR_obs + ((kurt-1)/4)*SR_obs^2) )
# E[max SR_N] approx = sqrt(2*ln(N)) - (gamma_em + ln(ln(N)))/(2*sqrt(2*ln(N)))
# Where gamma_em = 0.5772 (Euler-Mascheroni)
def deflated_sharpe(sr_obs, T, n_trials, skew, kurt):
    gamma = 0.5772156649
    z = np.sqrt(2*np.log(n_trials))
    e_max_sr = z - (gamma + np.log(np.log(n_trials))) / (2*z)
    num = (sr_obs - e_max_sr) * np.sqrt(T - 1)
    denom = np.sqrt(1 - skew * sr_obs + ((kurt - 1)/4) * sr_obs**2)
    return stats.norm.cdf(num / denom), e_max_sr

# Construire serie mensuelle de PnL
trades["month"] = trades["ts_in"].dt.to_period("M")
monthly = trades.groupby("month")["pnl_usd"].sum()
mean_m = monthly.mean()
std_m = monthly.std(ddof=1)
sr_m = mean_m / std_m
sr_ann = sr_m * np.sqrt(12)
skew_m = stats.skew(monthly.values)
kurt_m = stats.kurtosis(monthly.values, fisher=False)  # non-excess
T_m = len(monthly)
print(f"\nB) Sharpe mensuel annualise = {sr_ann:.4f} (T={T_m} mois)")
print(f"   Skewness mensuel: {skew_m:.3f}, Kurtosis: {kurt_m:.3f}")

for N_trials in [50, 100, 200]:
    dsr, e_max = deflated_sharpe(sr_m, T_m, N_trials, skew_m, kurt_m)
    print(f"   DSR (N_trials={N_trials}, monthly): p-val={dsr:.4f}, E[max SR mensuel]={e_max:.3f}")

# Sub-segment 2024-2026
sub = trades[trades["ts_in"] >= "2024-01-01"].copy()
sub["month"] = sub["ts_in"].dt.to_period("M")
m2 = sub.groupby("month")["pnl_usd"].sum()
sr_m2 = m2.mean() / m2.std(ddof=1)
sr_ann_sub = sr_m2 * np.sqrt(12)
sk2 = stats.skew(m2.values)
ku2 = stats.kurtosis(m2.values, fisher=False)
print(f"\n   Sub-segment 2024-2026 ({len(sub)} trades, {len(m2)} mois)")
print(f"   Sharpe mensuel annualise sub = {sr_ann_sub:.4f}")
print(f"   Skewness sub: {sk2:.3f}, Kurtosis sub: {ku2:.3f}")
for N_trials in [50, 100, 200]:
    dsr2, _ = deflated_sharpe(sr_m2, len(m2), N_trials, sk2, ku2)
    print(f"   DSR sub (N_trials={N_trials}, monthly): p-val={dsr2:.4f}")

# ---- C. Test de Chow rupture 2024-01-01
# H0: meme moyenne pre/post sur trade-level R
trades["pre2024"] = trades["ts_in"] < "2024-01-01"
pre = trades.loc[trades.pre2024, "r_realized"].values
post = trades.loc[~trades.pre2024, "r_realized"].values

# Test de Welch sur la moyenne (Chow simplifie sans regresseurs autres que constante)
t_w, p_w = stats.ttest_ind(pre, post, equal_var=False)
# Test de Levene sur la variance
f_l, p_l = stats.levene(pre, post)
# Mann-Whitney
u, p_mw = stats.mannwhitneyu(pre, post, alternative="two-sided")

print(f"\nC) Test rupture 2024-01-01 (R par trade)")
print(f"   n_pre={len(pre)}, n_post={len(post)}")
print(f"   mean_pre={pre.mean():.4f}, mean_post={post.mean():.4f}")
print(f"   var_pre={pre.var(ddof=1):.4f}, var_post={post.var(ddof=1):.4f}")
print(f"   Welch t-test: t={t_w:.3f}, p={p_w:.4f}")
print(f"   Levene var:   F={f_l:.3f}, p={p_l:.4f}")
print(f"   Mann-Whitney: U={u:.0f}, p={p_mw:.4f}")

# ---- D. Significativite par composant (bootstrap CI sur edge_R)
def bootstrap_edge(on_R, off_R, n_boot=5000, seed=42):
    rng = np.random.default_rng(seed)
    n_on, n_off = len(on_R), len(off_R)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        s_on = rng.choice(on_R, n_on, replace=True)
        s_off = rng.choice(off_R, n_off, replace=True)
        diffs[i] = s_on.mean() - s_off.mean()
    return np.percentile(diffs, [2.5, 50, 97.5]), diffs

components = ["c_bos", "c_choch", "c_ob", "c_fvg", "c_retest", "c_rsi_div", "c_regime", "c_news_ok"]
edges = []
print(f"\nD) Edge par composant (bootstrap 5000)")
for c in components:
    on_R = trades.loc[trades[c] == 1, "r_realized"].values
    off_R = trades.loc[trades[c] == 0, "r_realized"].values
    if len(off_R) < 10:
        edges.append((c, len(on_R), len(off_R), np.nan, np.nan, np.nan, np.nan, np.nan))
        continue
    ci, diffs = bootstrap_edge(on_R, off_R, 5000)
    edge = on_R.mean() - off_R.mean()
    # p-value bilateral via centrage bootstrap
    p_val = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    # t-test Welch
    t_c, p_c = stats.ttest_ind(on_R, off_R, equal_var=False)
    edges.append((c, len(on_R), len(off_R), edge, ci[0], ci[2], p_val, p_c))

edges_df = pd.DataFrame(edges, columns=["component", "n_on", "n_off", "edge_R", "ci_lo", "ci_hi", "p_boot", "p_welch"])
print(edges_df.to_string(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) and not np.isnan(x) else "NA"))

# Correction Holm-Bonferroni
valid = edges_df.dropna(subset=["p_welch"]).copy()
valid_sorted = valid.sort_values("p_welch")
m = len(valid_sorted)
holm_alpha = []
holm_reject = []
for i, p in enumerate(valid_sorted["p_welch"].values):
    alpha_i = 0.05 / (m - i)
    holm_alpha.append(alpha_i)
    holm_reject.append(p < alpha_i)
valid_sorted["holm_alpha"] = holm_alpha
valid_sorted["reject_holm"] = holm_reject
# Bonferroni simple
valid_sorted["bonf_alpha"] = 0.05 / m
valid_sorted["reject_bonf"] = valid_sorted["p_welch"] < valid_sorted["bonf_alpha"]
print(f"\n   Correction multiple (m={m}):")
print(valid_sorted[["component", "edge_R", "p_welch", "holm_alpha", "reject_holm", "reject_bonf"]]
      .to_string(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else x))

n_survive_bonf = valid_sorted["reject_bonf"].sum()
n_survive_holm = valid_sorted["reject_holm"].sum()
print(f"   Survivants Bonferroni: {n_survive_bonf}/{m}")
print(f"   Survivants Holm:        {n_survive_holm}/{m}")

# Save pour plus tard
edges_df.to_csv(OUT / "L1_D_edges.csv", index=False)
valid_sorted.to_csv(OUT / "L1_D_edges_holm.csv", index=False)

# =============================================================================
# LIVRABLE 2 - FORENSICS METHODOLOGIE
# =============================================================================
print("\n" + "="*80)
print("LIVRABLE 2 - FORENSICS")
print("="*80)

# B. Calendrier news - probabilite mathematique de blackout
# Si une release tombe a HH:MM uniformement dans une heure, et le blackout est
# +/-15 min, et les barres M15 sont [HH:00, HH:15), [HH:15, HH:30), etc.
# Une release a HH:MM impacte les barres dont le ts_in (ouverture barre) est
# dans la fenetre [HH:MM-15, HH:MM+15]. Une barre M15 est touchee ssi
# son ts_in est dans cette fenetre.
# Avec releases majoritairement a HH:00 ou HH:30 (NFP/CPI/FOMC):
# - Release a 13:30 (NFP) -> fenetre [13:15, 13:45]
#   Barres dont ts_in dans [13:15, 13:45] = la barre 13:15 et la barre 13:30 (2 barres)
# - Release a 14:00 (FOMC) -> fenetre [13:45, 14:15] -> barre 13:45 et 14:00 (2 barres)

# Sur 875 events sur 7 ans avec ~2 barres par event = ~1750 barres en blackout
# Le ledger en compte 2. Probleme: soit le blackout n'est jamais arme,
# soit il filtre les ENTREES mais pas les barres (une entree tombe rarement
# pile sur une barre de blackout car les entrees viennent de signal triggered).

# Calculons combien de trades ont ts_in dans une fenetre +/-15 du calendrier
cal_path = DATA / "economic_calendar_HIGH_IMPACT_2019_2025.csv"
if cal_path.exists():
    cal = pd.read_csv(cal_path, parse_dates=["timestamp"]) if "timestamp" in pd.read_csv(cal_path, nrows=1).columns else None
    # Auto-detect column
    raw = pd.read_csv(cal_path, nrows=2)
    print(f"\nB) Calendar columns: {list(raw.columns)}")
    # Try common column names
    ts_col = None
    for c in raw.columns:
        if "time" in c.lower() or "date" in c.lower() or "timestamp" in c.lower():
            ts_col = c
            break
    if ts_col:
        cal = pd.read_csv(cal_path, parse_dates=[ts_col])
        cal = cal.rename(columns={ts_col: "ts"})
        # Trades en blackout +/-15
        trades_ts = trades["ts_in"].values
        cal_ts = cal["ts"].values
        n_blackout_15 = 0
        n_blackout_60 = 0
        # Pour chaque trade, check si ts_in dans fenetre
        cal_sorted = np.sort(cal_ts)
        for tts in trades_ts:
            # binary search nearest event
            idx = np.searchsorted(cal_sorted, tts)
            cands = []
            if idx > 0: cands.append(cal_sorted[idx-1])
            if idx < len(cal_sorted): cands.append(cal_sorted[idx])
            if cands:
                deltas_min = [abs((tts - c)/np.timedelta64(1, 'm')) for c in cands]
                d = min(deltas_min)
                if d <= 15: n_blackout_15 += 1
                if d <= 60: n_blackout_60 += 1
        print(f"   Events: {len(cal)}, Trades: {len(trades)}")
        print(f"   Trades dans +/-15min d'un event: {n_blackout_15}")
        print(f"   Trades dans +/-60min d'un event: {n_blackout_60}")
        print(f"   Theorique: avec 875 events sur 365*7=2555 jours, prob qu'un trade")
        print(f"   tombe dans +/-15min ~ 875*30min/(2555*1440min) = {875*30/(2555*1440):.4f} = {875*30/(2555*1440)*100:.2f}%")
        print(f"   Soit attendu = {875*30/(2555*1440)*len(trades):.0f} trades en theorie (uniforme)")

# C. Lookahead regime SMA200 + slope
# Indice: si regime_label correle plus avec returns POST-entry qu'avec returns PRE-entry
# Charger XAU prix
print("\nC) Lookahead regime SMA200")
xau_file = DATA / "XAU_15MIN_2019_2026.csv"
xau = pd.read_csv(xau_file)
# Auto-detect time col
print(f"   XAU columns: {list(xau.columns)[:8]}...")
ts_c = None
for c in xau.columns:
    cl = c.lower()
    if cl in ("time", "timestamp", "datetime", "date"):
        ts_c = c; break
if ts_c:
    xau[ts_c] = pd.to_datetime(xau[ts_c])
    xau = xau.set_index(ts_c).sort_index()
    # close col
    close_c = next((c for c in xau.columns if "close" in c.lower()), None)
    if close_c:
        # Construire regime label "as observed at entry"
        # SMA200 sur close, slope = sma[t] - sma[t-50] (50 barres glissantes)
        xau["sma200"] = xau[close_c].rolling(200, min_periods=200).mean()
        xau["sma200_lag50"] = xau["sma200"].shift(50)
        xau["slope"] = xau["sma200"] - xau["sma200_lag50"]
        # Regime
        def lab(c, sma, slope):
            if pd.isna(sma) or pd.isna(slope): return "na"
            if c > sma and slope > 0: return "bull"
            if c < sma and slope < 0: return "bear"
            return "range"
        xau["regime"] = [lab(c, s, sl) for c, s, sl in zip(xau[close_c], xau["sma200"], xau["slope"])]
        # Future return 50 bars post-entry
        xau["fwd_50"] = xau[close_c].shift(-50) / xau[close_c] - 1
        # Past return 50 bars
        xau["bwd_50"] = xau[close_c] / xau[close_c].shift(50) - 1

        # Merge avec trades sur ts_in (eviter collision regime)
        xau_join = xau[["regime", "fwd_50", "bwd_50"]].rename(columns={"regime": "regime_recompute"})
        m = trades.set_index("ts_in").join(xau_join, how="left")
        m["regime_audit"] = m["regime_recompute"].fillna("na")

        # Coherence avec regime du ledger
        cross = pd.crosstab(m["regime"].fillna("na"), m["regime_audit"])
        print(f"   Crosstab regime ledger vs recompute (count):")
        print(cross.to_string())

        # Correlation regime_label_at_entry × return POST-entry
        m["bull_dummy"] = (m["regime_audit"] == "bull").astype(int)
        m["bear_dummy"] = (m["regime_audit"] == "bear").astype(int)
        # Pearson sur fwd_50 et bwd_50
        for lab in ["bull_dummy", "bear_dummy"]:
            mm = m.dropna(subset=[lab, "fwd_50", "bwd_50"])
            r_fwd = np.corrcoef(mm[lab], mm["fwd_50"])[0,1]
            r_bwd = np.corrcoef(mm[lab], mm["bwd_50"])[0,1]
            print(f"   {lab}: corr(label, fwd_50bars)={r_fwd:.4f}, corr(label, bwd_50bars)={r_bwd:.4f}")
            print(f"      ratio |fwd|/|bwd| = {abs(r_fwd)/max(abs(r_bwd),1e-9):.3f} (>>1 = lookahead suspect)")

# D. ATR-based SL/TP - dispersion
print("\nD) Verification ATR sur SL/TP")
trades["sl_dist"] = (trades["entry"] - trades["sl"]).abs()
trades["tp_dist"] = (trades["tp"] - trades["entry"]).abs()
trades["sl_tp_ratio"] = trades["sl_dist"] / trades["tp_dist"]
print(f"   sl_tp_ratio - mean={trades.sl_tp_ratio.mean():.4f}, std={trades.sl_tp_ratio.std():.4f}")
print(f"   sl_tp_ratio - p1={trades.sl_tp_ratio.quantile(0.01):.4f}, p99={trades.sl_tp_ratio.quantile(0.99):.4f}")
print(f"   Si SL/TP = 1.5/2.5 ATR fixe, ratio attendu = 0.6 (constant)")
print(f"   ratio = 0.6 ? mean={(trades.sl_tp_ratio - 0.6).abs().mean():.6f} (proche 0 = propre)")

# E. Timeout - pas de replay possible, on marque comme TODO
print("\nE) Timeout overfit test - non testable depuis ledger seul")
print("   Le ledger ne contient pas les trades exits a 12/18/36/48 barres - serait a re-runner")
print("   exit_reason='timeout' sur {} trades ({:.1%}) suggere parametre tres impactant".format(
    (trades.exit_reason == "timeout").sum(), (trades.exit_reason == "timeout").mean()))

# =============================================================================
# LIVRABLE 4 - STRESS TEST ROADMAP
# =============================================================================
print("\n" + "="*80)
print("LIVRABLE 4 - STRESS TEST ROADMAP")
print("="*80)

# A. Correlation CHOCH - RSI div sur le ledger
choch_x_rsi = pd.crosstab(trades.c_choch, trades.c_rsi_div)
phi_choch_rsi = np.corrcoef(trades.c_choch, trades.c_rsi_div)[0,1]
print(f"\nA) Crosstab CHOCH x RSI_div:")
print(choch_x_rsi.to_string())
print(f"   Phi (Pearson dichotomiques) = {phi_choch_rsi:.4f}")

# Correlation matrix entre tous composants
corr_comp = trades[components].corr()
print(f"\n   Matrice correlation composants:")
print(corr_comp.to_string(float_format=lambda x: f"{x:.3f}"))

# Edges combines - effet de retirer CHOCH ET RSI_div ensemble
# Estimation: edge_R agrege quand on filtre dehors
# scenario: garder uniquement trades ou choch=0 ET rsi_div=0
mask_drop = (trades.c_choch == 0) & (trades.c_rsi_div == 0)
sub_no = trades[mask_drop]
gw_no = sub_no.loc[sub_no.pnl_usd > 0, "pnl_usd"].sum()
gl_no = -sub_no.loc[sub_no.pnl_usd < 0, "pnl_usd"].sum()
pf_no = gw_no / gl_no if gl_no > 0 else np.nan
print(f"\n   Scenario 'CHOCH=0 ET RSI_div=0' (garder ces trades): n={len(sub_no)}, PF={pf_no:.4f}")

# Inversement: que se passe-t-il si on RETIRE les composants comme features
# (i.e. on ne change pas le filtre - on regarde l'edge_R sur l'ensemble)
# Le rapport propose: drop these from score -> threshold = (8-2)*12.5 = 75
# Donc on garderait les trades ou les 6 autres sont a 1 -> score >= 75 ancien -> 75/(6*12.5)=100%
# Ca change la selection
# Stress test: garde les trades ou bos+ob+fvg+retest+regime+news=1 (ignorer choch & rsi_div)
six_on = (trades[["c_bos", "c_ob", "c_fvg", "c_retest", "c_regime", "c_news_ok"]].sum(axis=1) >= 5)
sub_six = trades[six_on]
gw_s = sub_six.loc[sub_six.pnl_usd > 0, "pnl_usd"].sum()
gl_s = -sub_six.loc[sub_six.pnl_usd < 0, "pnl_usd"].sum()
pf_s = gw_s / gl_s if gl_s > 0 else np.nan
print(f"   Scenario 'au moins 5/6 des autres': n={len(sub_six)}, PF={pf_s:.4f}")

# B. Borne theorique R^2 d'un classifier sur ces features
# R^2 cumule = (eta_quad correlation^2 sommees) avec terme de redondance
# point-biserial ~ Pearson sur dichotomique vs continue
print(f"\nB) Borne R^2 GBM (point-biserial sur R)")
r_pb = []
for c in components:
    on_R = trades.loc[trades[c] == 1, "r_realized"].values
    off_R = trades.loc[trades[c] == 0, "r_realized"].values
    if len(off_R) < 10: continue
    pb, p = stats.pointbiserialr(trades[c], trades["r_realized"])
    r_pb.append((c, pb, pb**2))
r_pb_df = pd.DataFrame(r_pb, columns=["component", "r_pb", "r2_indiv"])
print(r_pb_df.to_string(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else x))
print(f"   Somme R^2 individuels (borne sup naive) = {r_pb_df['r2_indiv'].sum():.4f}")
print(f"   Pearson(score, R) global = {np.corrcoef(trades.score_in, trades.r_realized)[0,1]:.4f}")
print(f"   R^2 global score = {np.corrcoef(trades.score_in, trades.r_realized)[0,1]**2:.4f}")

# Multivariate R^2 (regression OLS sur composants)
X = trades[components].values
y = trades["r_realized"].values
# OLS via numpy
Xc = np.column_stack([np.ones(len(X)), X])
coef, *_ = np.linalg.lstsq(Xc, y, rcond=None)
yhat = Xc @ coef
ss_res = ((y - yhat)**2).sum()
ss_tot = ((y - y.mean())**2).sum()
r2_multi = 1 - ss_res/ss_tot
print(f"   R^2 OLS multivar (8 composants) = {r2_multi:.4f}")
print(f"   -> borne sup pour GBM en lineaire. Avec interactions ~ 1.5-2x = ~{r2_multi*1.5:.4f}")

# =============================================================================
# LIVRABLE 5 - SURVIVANCE 2024-2026
# =============================================================================
print("\n" + "="*80)
print("LIVRABLE 5 - SURVIVANCE 2024-2026")
print("="*80)

sub = trades[trades["ts_in"] >= "2024-01-01"].copy()

# A. Decomposition par side
for side, name in [(1, "Long"), (-1, "Short")]:
    s = sub[sub.side == side]
    g_w = s.loc[s.pnl_usd > 0, "pnl_usd"].sum()
    g_l = -s.loc[s.pnl_usd < 0, "pnl_usd"].sum()
    pf = g_w / g_l if g_l > 0 else np.nan
    print(f"   {name}: n={len(s)}, PnL={s.pnl_usd.sum():.0f}, PF={pf:.4f}, WR={s.pnl_usd.gt(0).mean():.3f}")

# B. Beta vs alpha: equity curve sub vs XAU spot
print(f"\nB) Beta vs alpha 2024-2026")
sub_sorted = sub.sort_values("ts_out").reset_index(drop=True)
sub_sorted["cum_pnl"] = sub_sorted["pnl_usd"].cumsum()
# XAU price aligned on trade ts_out
if 'xau' in dir() and close_c is not None:
    # daily resample
    eq_daily = sub_sorted.set_index("ts_out")["cum_pnl"].resample("1D").last().ffill()
    px_daily = xau[close_c].resample("1D").last().ffill()
    aligned = pd.concat([eq_daily, px_daily], axis=1, join="inner").dropna()
    aligned.columns = ["equity", "xau_px"]
    # filtrer dates 2024-2026
    aligned = aligned.loc["2024-01-01":]
    # corr daily levels
    r_lvl = np.corrcoef(aligned["equity"], aligned["xau_px"])[0,1]
    # corr daily returns
    aligned["eq_ret"] = aligned["equity"].diff()
    aligned["px_ret"] = aligned["xau_px"].pct_change()
    rr = aligned[["eq_ret", "px_ret"]].dropna()
    r_ret = np.corrcoef(rr["eq_ret"], rr["px_ret"])[0,1]
    # beta OLS
    cov = np.cov(rr["eq_ret"], rr["px_ret"])
    beta = cov[0,1] / cov[1,1]
    print(f"   Corr daily levels (equity vs XAU spot) = {r_lvl:.4f}")
    print(f"   Corr daily returns                       = {r_ret:.4f}")
    print(f"   Beta equity vs XAU returns               = {beta:.4f}")

# C. Test contrefactuel: shorts only sur 2024-2026
shorts_sub = sub[sub.side == -1]
gw_s = shorts_sub.loc[shorts_sub.pnl_usd > 0, "pnl_usd"].sum()
gl_s = -shorts_sub.loc[shorts_sub.pnl_usd < 0, "pnl_usd"].sum()
pf_short_sub = gw_s / gl_s if gl_s > 0 else np.nan
print(f"\nC) Shorts only 2024-2026: n={len(shorts_sub)}, PF={pf_short_sub:.4f}, PnL={shorts_sub.pnl_usd.sum():.0f}")
ci_short, _ = bootstrap_pf(shorts_sub.pnl_usd.values, 5000)
print(f"   Bootstrap 95% CI sur PF shorts 2024-2026: [{ci_short[0]:.4f}, {ci_short[2]:.4f}]")

# Bootstrap PF sub-segment 2024-2026 complet
ci_sub, boot_sub = bootstrap_pf(sub.pnl_usd.values, 5000)
print(f"   Bootstrap 95% CI sur PF 2024-2026 complet: [{ci_sub[0]:.4f}, {ci_sub[2]:.4f}]")
prob_sub_above_120 = (boot_sub >= 1.20).mean()
print(f"   P(PF_sub >= 1.20 | bootstrap) = {prob_sub_above_120:.4f}")

# Score predictivity 2024-2026
r_score_sub = np.corrcoef(sub.score_in, sub.r_realized)[0,1]
print(f"   Pearson(score, R) 2024-2026 = {r_score_sub:.4f}")

# =============================================================================
# Save consolide
# =============================================================================
out_summary = {
    "L1_A_bootstrap_pf_overall": {
        "ci_lo": float(ci_pf[0]), "ci_md": float(ci_pf[1]), "ci_hi": float(ci_pf[2]),
        "pf_obs": float(pf_obs),
        "p_pf_gte_1": float(prob_pf_gte_1),
    },
    "L1_B_DSR_overall": {
        "sharpe_monthly_ann": float(sr_ann),
        "skew": float(skew_m), "kurt": float(kurt_m),
        "T_months": int(T_m),
    },
    "L1_B_DSR_subsegment_2024_2026": {
        "sharpe_monthly_ann": float(sr_ann_sub),
        "skew": float(sk2), "kurt": float(ku2),
        "T_months": int(len(m2)),
    },
    "L1_C_Chow_2024": {
        "n_pre": int(len(pre)), "n_post": int(len(post)),
        "mean_pre": float(pre.mean()), "mean_post": float(post.mean()),
        "var_pre": float(pre.var(ddof=1)), "var_post": float(post.var(ddof=1)),
        "welch_t": float(t_w), "welch_p": float(p_w),
        "levene_F": float(f_l), "levene_p": float(p_l),
        "mannwhitney_p": float(p_mw),
    },
    "L1_D_holm_survivors": int(n_survive_holm),
    "L1_D_bonf_survivors": int(n_survive_bonf),
    "L4_A_phi_choch_rsi": float(phi_choch_rsi),
    "L4_B_R2_OLS_multi": float(r2_multi),
    "L5_C_pf_short_2024_2026": float(pf_short_sub),
    "L5_C_ci_short": {"lo": float(ci_short[0]), "hi": float(ci_short[2])},
    "L5_pf_2024_2026_ci": {"lo": float(ci_sub[0]), "hi": float(ci_sub[2])},
    "L5_p_pf_sub_gte_120": float(prob_sub_above_120),
}

with open(OUT / "falsification_results.json", "w") as f:
    json.dump(out_summary, f, indent=2)

print("\n\n=== SAVED to", OUT, "===")
