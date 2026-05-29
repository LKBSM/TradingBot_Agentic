"""
Action 2 - Comparatif XAU H1
=============================
Re-aggrege XAU M15 -> H1 et applique le pipeline SMC tel quel.

Adaptations time-based :
  - ATR_14 sur H1 (au lieu M15)
  - SMA200 sur H1 (= 8.3 jours, vs 50h en M15)
  - max_lifetime = 24 H1 bars (= 1 jour) - garde la valeur "24" de la version
    pre-quick-win pour comparabilite conceptuelle (le timeout est en BARS, pas en heures)
  - News blackout : +/-60 min (au lieu +/-15)
  - Cout identique : spread 0.30, slippage 0.10-0.20, commission 7 USD/lot RT
  - Risk 1%/trade, capital 10k, seed=42

NOTE: pas de re-tuning des seuils du score (test out-of-domain honnete).
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

OUT_REPORTS = ROOT / "reports"
OUT_REPORTS.mkdir(exist_ok=True)

# ----------------------------------------------------------------------------
# 1. Adapter CFG pour XAU H1
# ----------------------------------------------------------------------------
CFG_H1 = dict(base.CFG)  # copie
CFG_H1["max_lifetime_bars"] = 24                # 24 H1 bars = 1 journee
CFG_H1["blackout_minutes_before"] = 60
CFG_H1["blackout_minutes_after"] = 60
# Cooldown / lockout : adapter (l'audit utilise 8 bars en M15 = 2h, on garde 8 H1 bars = 8h ?)
# Choix conservateur : garder les memes valeurs en BARS pour test out-of-domain honnete
# (le user a dit : "Ne pas re-tuner les seuils sur les nouveaux assets")
# Donc cooldown_bars=8, lockout=12 restent identiques

# Patch CFG global du module base AVANT toute simulation
for k, v in CFG_H1.items():
    base.CFG[k] = v

# Reset le seed
np.random.seed(base.CFG["random_seed"])

# ----------------------------------------------------------------------------
# 2. Charger M15 et resampler en H1
# ----------------------------------------------------------------------------
print("[H1] Chargement XAU M15 + resample H1...")
df_m15 = pd.read_csv(base.CFG["data_file"], parse_dates=["Date"])
df_m15 = df_m15.rename(columns={"Date": "ts"}).set_index("ts").sort_index()
df_m15.columns = [c.lower() for c in df_m15.columns]
df_m15 = df_m15[(df_m15.index >= base.CFG["start"]) & (df_m15.index <= base.CFG["end"])].copy()
df_m15 = df_m15[~df_m15.index.duplicated(keep="first")]
print(f"[H1] M15 chargee : {len(df_m15):,} barres")

# Resample 1h - prendre OHLCV agg classique
df = pd.DataFrame({
    "open": df_m15["open"].resample("1h").first(),
    "high": df_m15["high"].resample("1h").max(),
    "low": df_m15["low"].resample("1h").min(),
    "close": df_m15["close"].resample("1h").last(),
    "volume": df_m15["volume"].resample("1h").sum() if "volume" in df_m15.columns else 0,
}).dropna(subset=["open", "high", "low", "close"])
print(f"[H1] H1 generee : {len(df):,} barres de {df.index.min()} a {df.index.max()}")

# Charger calendrier
print("[H1] Chargement calendrier news...")
cal = pd.read_csv(base.CFG["calendar_file"], parse_dates=["Date"])
cal = cal.rename(columns={"Date": "ts"})
cal = cal[cal["Currency"].isin(["USD", "EUR"])].sort_values("ts").reset_index(drop=True)
print(f"[H1] {len(cal):,} events high-impact")

# ----------------------------------------------------------------------------
# 3. Computer features avec helpers du base script (period_indicator agnostic)
# ----------------------------------------------------------------------------
print("[H1] Computing features (ATR, swings, BOS, CHOCH, OB, FVG, retest, RSI div, regime, news, HAR-RV)...")
df["atr"] = base.compute_atr(df, base.CFG["atr_period"])
df = base.detect_swings(df, lookback=5)
df = base.detect_bos(df)
df = base.detect_choch(df)
df = base.detect_order_blocks(df)
df = base.detect_fvg(df, atr_mult=0.4)
df = base.detect_retest(df, atr_tol=0.25)
df = base.detect_rsi_divergence(df, lookback=14)
df = base.detect_regime(df)

# News blackout - le helper base.detect_news_blackout iter sur les events ;
# on doit l'adapter car la fenetre minutes ne change pas (on bumpe juste les ms)
# La fonction base.detect_news_blackout lit CFG["blackout_minutes_*"], donc ok
df = base.detect_news_blackout(df, cal)

# HAR-RV - le helper assume 96 bars/jour (M15). On doit adapter pour H1 (24/jour)
# Patch local : recalculer manuellement
print("[H1] HAR-RV adapte pour H1 (24 bars/jour)...")
bars_per_day = 24
log_ret = np.log(df["close"] / df["close"].shift())
rv = (log_ret ** 2).rolling(bars_per_day).sum()
rv_d = rv
rv_w = rv.rolling(5 * bars_per_day).mean()
rv_m = rv.rolling(22 * bars_per_day).mean()
rv_forecast = 0.50 * rv_d + 0.30 * rv_w + 0.20 * rv_m
annual_vol = np.sqrt(rv_forecast * 252)
df["vol_forecast_pct"] = annual_vol
df["vol_15m_usd"] = df["close"] * np.sqrt(rv_forecast / bars_per_day)  # nom historique
# (le simulator utilise vol_15m_usd, fallback ATR si NaN - on garde le nom)

df = base.compute_confluence_score(df)
print(f"[H1] features OK : {len(df):,} bars")

# ----------------------------------------------------------------------------
# 4. Simulate (re-utilise base.simulate)
# ----------------------------------------------------------------------------
print("[H1] Simulating...")
np.random.seed(base.CFG["random_seed"])
trades, equity = base.simulate(df)
print(f"[H1] {len(trades):,} trades")

metrics = base.compute_metrics(trades, equity)
bh = base.buy_hold_metrics(df)
pearson = base.score_pearson(trades)
bk = base.regime_breakdown(trades)

# ----------------------------------------------------------------------------
# 5. Bootstrap CI
# ----------------------------------------------------------------------------
def bootstrap_pf(pnl, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(pnl)
    if n < 10: return float("nan"), float("nan")
    pfs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        s = pnl[idx]
        gw = s[s > 0].sum(); gl = -s[s < 0].sum()
        if gl > 0: pfs.append(gw/gl)
    if not pfs: return float("nan"), float("nan")
    return float(np.percentile(pfs, 2.5)), float(np.percentile(pfs, 97.5))

pnl_arr = np.array([t.pnl_usd for t in trades]) if trades else np.array([])
ci_lo, ci_hi = bootstrap_pf(pnl_arr, 2000)

# ----------------------------------------------------------------------------
# 6. Decompositions
# ----------------------------------------------------------------------------
def metrics_subset(filter_fn, label):
    sub = [t for t in trades if filter_fn(t)]
    if not sub: return None
    pnl = np.array([t.pnl_usd for t in sub])
    g = pnl[pnl > 0].sum(); l = -pnl[pnl < 0].sum()
    pf_v = g / l if l > 0 else float("nan")
    cl, ch = bootstrap_pf(pnl, 2000)
    return {
        "label": label, "n": len(sub),
        "wr": float((pnl > 0).mean()),
        "pf": float(pf_v),
        "ci_lo": cl, "ci_hi": ch,
        "pnl_usd": float(pnl.sum()),
        "avg_r": float(np.array([t.r_realized for t in sub]).mean()),
    }

# Annee
yearly = []
for y in range(2019, 2027):
    m = metrics_subset(lambda t,y=y: t.ts_in.year == y, f"{y}")
    if m: yearly.append(m)
df_yearly = pd.DataFrame(yearly)

# Side
m_long = metrics_subset(lambda t: t.side == 1, "Long")
m_short = metrics_subset(lambda t: t.side == -1, "Short")

# Regime
m_bull = metrics_subset(lambda t: t.regime == "bull", "Bull")
m_bear = metrics_subset(lambda t: t.regime == "bear", "Bear")
m_range = metrics_subset(lambda t: t.regime == "range", "Range")

# Pre/post 2024
m_pre = metrics_subset(lambda t: t.ts_in < pd.Timestamp("2024-01-01"), "Pre_2024")
m_post = metrics_subset(lambda t: t.ts_in >= pd.Timestamp("2024-01-01"), "Post_2024")

# ----------------------------------------------------------------------------
# 7. Save
# ----------------------------------------------------------------------------
trades_df = pd.DataFrame([{
    "id": t.id, "ts_in": t.ts_in, "ts_out": t.ts_out, "side": t.side,
    "entry": t.entry, "sl": t.sl, "tp": t.tp, "exit_price": t.exit_price,
    "exit_reason": t.exit_reason, "lots": t.lots, "pnl_usd": t.pnl_usd,
    "r_realized": t.r_realized, "score_in": t.score_in, "bars_held": t.bars_held,
    "regime": t.regime, "session": t.session,
    **{f"c_{k}": v for k, v in t.components.items()},
} for t in trades])
trades_df.to_csv(OUT_REPORTS / "comparatif_xau_h1_trades.csv", index=False)

# Equity
eq_df = equity.reset_index().rename(columns={"ts": "ts", 0: "equity"})
eq_df.columns = ["ts", "equity"]
eq_df.to_csv(OUT_REPORTS / "comparatif_xau_h1_equity.csv", index=False)

summary = {
    "asset": "XAU/USD",
    "timeframe": "H1",
    "config": {k: str(v) if isinstance(v, Path) else v for k, v in base.CFG.items()},
    "n_bars": len(df),
    "metrics": {**metrics, "pf_ci_lo": ci_lo, "pf_ci_hi": ci_hi},
    "buy_hold": bh,
    "pearson_score_vs_R": pearson,
    "score_distribution": {
        "p50": float(df["score_max"].quantile(0.5)),
        "p75": float(df["score_max"].quantile(0.75)),
        "p90": float(df["score_max"].quantile(0.9)),
        "p99": float(df["score_max"].quantile(0.99)),
        "max": float(df["score_max"].max()),
    },
    "yearly": yearly,
    "side": {"long": m_long, "short": m_short},
    "regime": {"bull": m_bull, "bear": m_bear, "range": m_range},
    "pre_post_2024": {"pre": m_pre, "post": m_post},
}
with open(OUT_REPORTS / "comparatif_xau_h1_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=float)

# ----------------------------------------------------------------------------
# 8. Print recap
# ----------------------------------------------------------------------------
print("\n" + "="*70)
print("XAU H1 - METRIQUES")
print("="*70)
print(f"  n_trades : {metrics['n_trades']}")
print(f"  WR       : {metrics['win_rate']:.4f}")
print(f"  PF       : {metrics['profit_factor']:.4f}  CI95=[{ci_lo:.3f}, {ci_hi:.3f}]")
print(f"  Sharpe   : {metrics['sharpe']:.4f}")
print(f"  Sortino  : {metrics['sortino']:.4f}")
print(f"  MaxDD    : {metrics['max_dd_pct']*100:.2f}%")
print(f"  Return   : {metrics['return_pct']*100:.2f}%")
print(f"  Pearson(score, R) : {pearson:.4f}")
print(f"  Buy & Hold        : {bh['buyhold_return_pct']*100:.2f}%")
if df_yearly.shape[0] > 0:
    print("\nPar annee:")
    print(df_yearly[["label","n","wr","pf","ci_lo","ci_hi","pnl_usd"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else x))
if m_long: print(f"\nLong  : n={m_long['n']}, PF={m_long['pf']:.3f}, CI=[{m_long['ci_lo']:.3f},{m_long['ci_hi']:.3f}]")
if m_short: print(f"Short : n={m_short['n']}, PF={m_short['pf']:.3f}, CI=[{m_short['ci_lo']:.3f},{m_short['ci_hi']:.3f}]")
if m_bull: print(f"Bull regime : n={m_bull['n']}, PF={m_bull['pf']:.3f}")
if m_bear: print(f"Bear regime : n={m_bear['n']}, PF={m_bear['pf']:.3f}")
if m_range: print(f"Range regime: n={m_range['n']}, PF={m_range['pf']:.3f}")
if m_pre: print(f"Pre 2024: n={m_pre['n']}, PF={m_pre['pf']:.3f}")
if m_post: print(f"Post 2024: n={m_post['n']}, PF={m_post['pf']:.3f}")

print(f"\n[H1] DONE. Files: comparatif_xau_h1_*.csv/.json")
