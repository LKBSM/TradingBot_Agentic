"""
Action 3 - Comparatif EURUSD M15
=================================
Pipeline SMC actuel applique sur EURUSD M15 2019-2025 (donnee dispo).
Test out-of-domain honnete : aucun re-tuning des seuils ; on applique
la config XAU M15 v2 (timeout=64) telle quelle, avec adaptation des
unites monetaires FX.

Adaptations FX :
  - 1 lot = 100 000 EUR (vs 100 oz pour XAU)
  - spread 0.5 pip = 0.00005 (price units)
  - slippage 0.1-0.3 pip = 0.00001 a 0.00003
  - commission 7 USD/lot RT
  - clamp sl_dist 0.0010 - 0.0100 (10 - 100 pips, vs 2-30 USD pour XAU)
  - news blackout +/- 60 min (au lieu +/-15)
  - max_lifetime_bars = 64 (consistent quick-win)

Pas de 2026 dans le data EURUSD - sample est 2019-01-02 -> 2025-12-31.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import quant_audit_2026_04_30 as base

OUT_REPORTS = ROOT / "reports"

# ----------------------------------------------------------------------------
# CFG FX
# ----------------------------------------------------------------------------
CFG_FX = {
    "data_file": ROOT / "data" / "EURUSD_15MIN_2019_2025.csv",
    "calendar_file": ROOT / "data" / "economic_calendar_HIGH_IMPACT_2019_2025.csv",
    "start": "2019-01-01",
    "end": "2026-04-30",
    # Microstructure FX
    "spread_price": 0.00005,     # 0.5 pip
    "slippage_min": 0.00001,
    "slippage_max": 0.00003,
    "commission_per_lot_rt": 7.0,
    "lot_size_units": 100_000.0,  # 1 standard lot EUR
    # Capital / risk
    "initial_capital": 10_000.0,
    "risk_per_trade_pct": 0.01,
    "max_lot": 5.0,
    # Volatilite
    "atr_period": 14,
    "sl_atr_mult": 1.5,
    "tp_atr_mult": 2.5,
    # State machine - pris du XAU M15 (out-of-domain, pas re-tune)
    "enter_threshold": 65,
    "exit_threshold": 40,
    "cooldown_bars": 8,
    "lockout_bars_opposite": 12,
    "max_lifetime_bars": 64,    # quick-win
    # News
    "blackout_minutes_before": 60,
    "blackout_minutes_after": 60,
    # Score weights identiques
    "weights": {
        "bos": 12.5, "choch": 12.5, "ob": 12.5, "fvg": 12.5,
        "retest": 12.5, "rsi_div": 12.5, "regime": 12.5, "news_ok": 12.5,
    },
    # SL clamp en price units pour EURUSD (10 a 100 pips)
    "sl_dist_min": 0.0010,
    "sl_dist_max": 0.0100,
    "random_seed": 42,
}

np.random.seed(CFG_FX["random_seed"])

# ----------------------------------------------------------------------------
# 1. Patch CFG global pour features (composants utilisent base.CFG)
# ----------------------------------------------------------------------------
for k in ["enter_threshold", "exit_threshold", "cooldown_bars", "lockout_bars_opposite",
          "max_lifetime_bars", "blackout_minutes_before", "blackout_minutes_after",
          "weights", "atr_period", "sl_atr_mult", "tp_atr_mult", "random_seed"]:
    base.CFG[k] = CFG_FX[k]

# ----------------------------------------------------------------------------
# 2. Charger donnees EURUSD
# ----------------------------------------------------------------------------
print("[EUR] Chargement EURUSD M15...")
df = pd.read_csv(CFG_FX["data_file"], parse_dates=["Date"])
df = df.rename(columns={"Date": "ts"}).set_index("ts").sort_index()
df.columns = [c.lower() for c in df.columns]
df = df[(df.index >= CFG_FX["start"]) & (df.index <= CFG_FX["end"])].copy()
df = df[~df.index.duplicated(keep="first")]
print(f"[EUR] {len(df):,} barres de {df.index.min()} a {df.index.max()}")

cal = pd.read_csv(CFG_FX["calendar_file"], parse_dates=["Date"])
cal = cal.rename(columns={"Date": "ts"})
cal = cal[cal["Currency"].isin(["USD", "EUR"])].sort_values("ts").reset_index(drop=True)
print(f"[EUR] {len(cal):,} events high-impact")

# ----------------------------------------------------------------------------
# 3. Features (reutiliser helpers base, qui sont price-unit-agnostic)
# ----------------------------------------------------------------------------
print("[EUR] Computing features...")
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

# HAR-RV : 96 bars/jour (M15)
print("[EUR] HAR-RV (96 bars/jour)...")
bars_per_day = 96
log_ret = np.log(df["close"] / df["close"].shift())
rv = (log_ret ** 2).rolling(bars_per_day).sum()
rv_d = rv
rv_w = rv.rolling(5 * bars_per_day).mean()
rv_m = rv.rolling(22 * bars_per_day).mean()
rv_forecast = 0.50 * rv_d + 0.30 * rv_w + 0.20 * rv_m
df["vol_forecast_pct"] = np.sqrt(rv_forecast * 252)
df["vol_15m"] = df["close"] * np.sqrt(rv_forecast / bars_per_day)

df = base.compute_confluence_score(df)
print(f"[EUR] features OK : {len(df):,} bars")

# ----------------------------------------------------------------------------
# 4. Simulateur FX - re-implementation locale (clamp + lot_size adaptes)
# ----------------------------------------------------------------------------
@dataclass
class TradeFX:
    id: int
    ts_in: pd.Timestamp
    ts_out: Optional[pd.Timestamp]
    side: int
    entry: float
    sl: float
    tp: float
    exit_price: Optional[float]
    exit_reason: Optional[str]
    lots: float
    pnl_usd: float
    r_realized: float
    score_in: float
    bars_held: int
    regime: str
    session: str
    components: dict = field(default_factory=dict)


def simulate_fx(df):
    sm = base.StateMachine()
    trades = []
    open_trade = None
    bar_open_idx = -1
    capital = CFG_FX["initial_capital"]
    equity = pd.Series(index=df.index, dtype=float)
    equity.iloc[0] = capital

    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    score_bulls = df["score_bull"].values
    score_bears = df["score_bear"].values
    vol = df["vol_15m"].fillna(df["atr"]).values
    atrs = df["atr"].values
    timestamps = df.index
    rng = np.random.default_rng(CFG_FX["random_seed"])

    for i in range(50, len(df)):
        ts = timestamps[i]
        c = closes[i]
        h = highs[i]
        l = lows[i]

        if open_trade is not None:
            sl = open_trade.sl
            tp = open_trade.tp
            side = open_trade.side
            exit_price = None
            exit_reason = None
            if side == 1:
                if l <= sl:
                    exit_price, exit_reason = sl, "sl"
                elif h >= tp:
                    exit_price, exit_reason = tp, "tp"
            else:
                if h >= sl:
                    exit_price, exit_reason = sl, "sl"
                elif l <= tp:
                    exit_price, exit_reason = tp, "tp"
            if exit_price is None and (i - bar_open_idx) >= CFG_FX["max_lifetime_bars"]:
                exit_price, exit_reason = c, "timeout"
            if exit_price is None:
                opp = score_bears[i] if side == 1 else score_bulls[i]
                if opp >= CFG_FX["enter_threshold"]:
                    exit_price, exit_reason = c, "opposite"
            if exit_price is not None:
                slip = rng.uniform(CFG_FX["slippage_min"], CFG_FX["slippage_max"])
                if side == 1:
                    fill = exit_price - slip - CFG_FX["spread_price"] / 2
                else:
                    fill = exit_price + slip + CFG_FX["spread_price"] / 2
                pnl_per_unit = (fill - open_trade.entry) * side
                pnl_usd = pnl_per_unit * open_trade.lots * CFG_FX["lot_size_units"]
                pnl_usd -= CFG_FX["commission_per_lot_rt"] * open_trade.lots
                capital += pnl_usd
                sl_distance = abs(open_trade.entry - open_trade.sl)
                r = pnl_per_unit / sl_distance if sl_distance > 0 else 0.0
                open_trade.ts_out = ts
                open_trade.exit_price = fill
                open_trade.exit_reason = exit_reason
                open_trade.pnl_usd = pnl_usd
                open_trade.r_realized = r
                open_trade.bars_held = i - bar_open_idx
                trades.append(open_trade)
                sm.register_exit(i, side)
                open_trade = None

        if open_trade is None:
            decision = sm.step(i, score_bulls[i], score_bears[i])
            if decision != 0:
                v = vol[i] if not np.isnan(vol[i]) else atrs[i]
                if np.isnan(v) or v <= 0:
                    equity.iloc[i] = capital
                    continue
                sl_dist = CFG_FX["sl_atr_mult"] * v * np.sqrt(CFG_FX["max_lifetime_bars"])
                sl_dist = max(CFG_FX["sl_dist_min"], min(sl_dist, CFG_FX["sl_dist_max"]))
                tp_dist = CFG_FX["tp_atr_mult"] / CFG_FX["sl_atr_mult"] * sl_dist

                slip = rng.uniform(CFG_FX["slippage_min"], CFG_FX["slippage_max"])
                if decision == 1:
                    entry = c + slip + CFG_FX["spread_price"] / 2
                    sl = entry - sl_dist
                    tp = entry + tp_dist
                else:
                    entry = c - slip - CFG_FX["spread_price"] / 2
                    sl = entry + sl_dist
                    tp = entry - tp_dist

                risk_usd = capital * CFG_FX["risk_per_trade_pct"]
                lots = risk_usd / (sl_dist * CFG_FX["lot_size_units"])
                lots = round(min(max(lots, 0.01), CFG_FX["max_lot"]), 2)

                if df["regime_bull"].iloc[i]: reg = "bull"
                elif df["regime_bear"].iloc[i]: reg = "bear"
                else: reg = "range"
                hour = ts.hour
                if 0 <= hour < 7: sess = "asia"
                elif 7 <= hour < 13: sess = "london"
                elif 13 <= hour < 21: sess = "ny"
                else: sess = "off"

                comps = {
                    "bos": int(df["bos_bull_active" if decision == 1 else "bos_bear_active"].iloc[i]),
                    "choch": int(df["choch_bull_active" if decision == 1 else "choch_bear_active"].iloc[i]),
                    "ob": int(df["ob_bull_active" if decision == 1 else "ob_bear_active"].iloc[i]),
                    "fvg": int(df["fvg_bull_active" if decision == 1 else "fvg_bear_active"].iloc[i]),
                    "retest": int(df["retest_bull" if decision == 1 else "retest_bear"].iloc[i]),
                    "rsi_div": int(df["rsi_div_bull_active" if decision == 1 else "rsi_div_bear_active"].iloc[i]),
                    "regime": int(df["regime_bull" if decision == 1 else "regime_bear"].iloc[i]),
                    "news_ok": int(df["news_ok"].iloc[i]),
                }

                open_trade = TradeFX(
                    id=len(trades) + 1, ts_in=ts, ts_out=None, side=decision,
                    entry=entry, sl=sl, tp=tp, exit_price=None, exit_reason=None,
                    lots=lots, pnl_usd=0.0, r_realized=0.0,
                    score_in=score_bulls[i] if decision == 1 else score_bears[i],
                    bars_held=0, regime=reg, session=sess, components=comps,
                )
                bar_open_idx = i

        equity.iloc[i] = capital

    equity = equity.ffill().fillna(CFG_FX["initial_capital"])
    return trades, equity

print("[EUR] Simulating...")
trades, equity = simulate_fx(df)
print(f"[EUR] {len(trades):,} trades")

# ----------------------------------------------------------------------------
# 5. Metrics + decompositions
# ----------------------------------------------------------------------------
def pf_fn(pnl):
    g = pnl[pnl > 0].sum(); l = -pnl[pnl < 0].sum()
    return g / l if l > 0 else float("nan")

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

# Standard metrics manuel pour eviter Trade vs TradeFX issues
pnls = np.array([t.pnl_usd for t in trades])
rs = np.array([t.r_realized for t in trades])
n = len(trades)
wr = (pnls > 0).mean()
gw = pnls[pnls > 0].sum()
gl = -pnls[pnls < 0].sum()
pf = gw / gl if gl > 0 else float("nan")
ci_lo, ci_hi = bootstrap_pf(pnls, 2000)
final_capital = equity.iloc[-1]
return_pct = final_capital / CFG_FX["initial_capital"] - 1
eq_monthly = equity.resample("ME").last().pct_change().dropna()
sharpe = (eq_monthly.mean()/eq_monthly.std()*np.sqrt(12)) if len(eq_monthly)>1 and eq_monthly.std()>0 else float("nan")
downside = eq_monthly[eq_monthly < 0]
sortino = (eq_monthly.mean()/downside.std()*np.sqrt(12)) if len(downside)>1 and downside.std()>0 else float("nan")
running_max = equity.cummax()
dd = (equity - running_max) / running_max
max_dd_pct = dd.min()
calmar = return_pct / abs(max_dd_pct) if max_dd_pct < 0 else float("nan")

# Pearson(score, R)
scores_arr = np.array([t.score_in for t in trades])
pearson = float(np.corrcoef(scores_arr, rs)[0,1]) if len(scores_arr) > 30 and scores_arr.std() > 0 else float("nan")

# Buy & hold
bh_in = df["close"].iloc[0]
bh_out = df["close"].iloc[-1]
bh_ret = (bh_out - bh_in) / bh_in

print(f"\n=== EURUSD M15 ===")
print(f"  n_trades={n}, WR={wr:.4f}, PF={pf:.4f} CI=[{ci_lo:.3f},{ci_hi:.3f}]")
print(f"  Sharpe={sharpe:.3f}, Sortino={sortino:.3f}, MaxDD={max_dd_pct*100:.2f}%")
print(f"  Capital final={final_capital:.0f}, Return={return_pct*100:.2f}%")
print(f"  Pearson(score, R) = {pearson:.4f}")
print(f"  Buy & Hold = {bh_ret*100:.2f}%")

# Decompositions
def metrics_subset(filter_fn, label):
    sub = [t for t in trades if filter_fn(t)]
    if not sub: return None
    p = np.array([t.pnl_usd for t in sub])
    return {
        "label": label, "n": len(sub),
        "wr": float((p > 0).mean()),
        "pf": float(pf_fn(p)),
        "ci_lo": bootstrap_pf(p, 2000)[0],
        "ci_hi": bootstrap_pf(p, 2000)[1],
        "pnl_usd": float(p.sum()),
        "avg_r": float(np.array([t.r_realized for t in sub]).mean()),
    }

yearly = []
for y in range(2019, 2026):
    m = metrics_subset(lambda t,y=y: t.ts_in.year == y, f"{y}")
    if m: yearly.append(m)

m_long = metrics_subset(lambda t: t.side == 1, "Long")
m_short = metrics_subset(lambda t: t.side == -1, "Short")
m_bull = metrics_subset(lambda t: t.regime == "bull", "Bull")
m_bear = metrics_subset(lambda t: t.regime == "bear", "Bear")
m_range = metrics_subset(lambda t: t.regime == "range", "Range")
m_pre = metrics_subset(lambda t: t.ts_in < pd.Timestamp("2024-01-01"), "Pre_2024")
m_post = metrics_subset(lambda t: t.ts_in >= pd.Timestamp("2024-01-01"), "Post_2024")

print(f"\nLong : n={m_long['n']}, PF={m_long['pf']:.3f} CI=[{m_long['ci_lo']:.3f},{m_long['ci_hi']:.3f}]")
print(f"Short: n={m_short['n']}, PF={m_short['pf']:.3f} CI=[{m_short['ci_lo']:.3f},{m_short['ci_hi']:.3f}]")
print(f"Bull : n={m_bull['n']}, PF={m_bull['pf']:.3f}")
print(f"Bear : n={m_bear['n']}, PF={m_bear['pf']:.3f}")
print(f"Range: n={m_range['n']}, PF={m_range['pf']:.3f}")
print(f"Pre  2024: n={m_pre['n']}, PF={m_pre['pf']:.3f}")
print(f"Post 2024: n={m_post['n']}, PF={m_post['pf']:.3f}")
print("\nPar annee:")
for y in yearly:
    print(f"  {y['label']}: n={y['n']}, PF={y['pf']:.3f}, CI=[{y['ci_lo']:.3f},{y['ci_hi']:.3f}], PnL={y['pnl_usd']:.0f}")

# ----------------------------------------------------------------------------
# 6. Save
# ----------------------------------------------------------------------------
trades_df = pd.DataFrame([{
    "id": t.id, "ts_in": t.ts_in, "ts_out": t.ts_out, "side": t.side,
    "entry": t.entry, "sl": t.sl, "tp": t.tp, "exit_price": t.exit_price,
    "exit_reason": t.exit_reason, "lots": t.lots, "pnl_usd": t.pnl_usd,
    "r_realized": t.r_realized, "score_in": t.score_in, "bars_held": t.bars_held,
    "regime": t.regime, "session": t.session,
    **{f"c_{k}": v for k, v in t.components.items()},
} for t in trades])
trades_df.to_csv(OUT_REPORTS / "comparatif_eurusd_m15_trades.csv", index=False)

eq_df = pd.DataFrame({"ts": equity.index, "equity": equity.values})
eq_df.to_csv(OUT_REPORTS / "comparatif_eurusd_m15_equity.csv", index=False)

summary = {
    "asset": "EURUSD",
    "timeframe": "M15",
    "config": {k: str(v) if isinstance(v, Path) else v for k, v in CFG_FX.items()},
    "data_period": [str(df.index.min()), str(df.index.max())],
    "n_bars": len(df),
    "metrics": {
        "n_trades": int(n),
        "win_rate": float(wr),
        "profit_factor": float(pf),
        "pf_ci_lo": ci_lo,
        "pf_ci_hi": ci_hi,
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "calmar": float(calmar),
        "max_dd_pct": float(max_dd_pct),
        "final_capital": float(final_capital),
        "return_pct": float(return_pct),
        "expectancy_r": float(rs.mean()),
        "gross_win": float(gw),
        "gross_loss": float(gl),
    },
    "buy_hold": {"buyhold_return_pct": float(bh_ret), "px_in": float(bh_in), "px_out": float(bh_out)},
    "pearson_score_vs_R": pearson,
    "score_distribution": {
        "p50": float(df["score_max"].quantile(0.5)),
        "p90": float(df["score_max"].quantile(0.9)),
        "max": float(df["score_max"].max()),
    },
    "yearly": yearly,
    "side": {"long": m_long, "short": m_short},
    "regime": {"bull": m_bull, "bear": m_bear, "range": m_range},
    "pre_post_2024": {"pre": m_pre, "post": m_post},
}
with open(OUT_REPORTS / "comparatif_eurusd_m15_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=float)

print(f"\n[EUR] DONE. Files: comparatif_eurusd_m15_*.csv/.json")
