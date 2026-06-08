"""
Baseline NR4 (Crabel volatility breakout) - L4
==============================================
Strategie non-SMC pour comparaison apples-to-apples vs Smart Sentinel SMC.

Spec :
  - NR4 : range[t] = min(range[t..t-3])
  - ATR expansion : ATR_14[t] / ATR_14[t-20] > 1.2
  - Signal long  : close[t] > high(NR4_bar) ET expansion=True
  - Signal short : close[t] < low(NR4_bar) ET expansion=True
  - SL = 1.0 * ATR_14, TP = 2.0 * ATR_14
  - Max lifetime : 16 bars
  - Cooldown : 8 bars
  - Filtre session : exclure 21h-00h UTC (trade 00-21h)
  - Filtre news : +/-60 min events high-impact
  - Microstructure : spread 0.30, slippage 0.10-0.20, commission 7 USD/lot RT
  - Capital 10 000 USD, risk 1%/trade, seed=42

Reference : Crabel (1990) "Day Trading with Short Term Price Patterns and ORB"
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
# CFG NR4
# ----------------------------------------------------------------------------
CFG_NR4 = {
    "data_file": base.DATA_DIR / "XAU_15MIN_2019_2026.csv",
    "calendar_file": base.DATA_DIR / "economic_calendar_HIGH_IMPACT_2019_2025.csv",
    "start": "2019-01-01",
    "end": "2026-04-30",
    "spread_usd": 0.30,
    "slippage_min": 0.10,
    "slippage_max": 0.20,
    "commission_per_lot_rt": 7.0,
    "lot_size_oz": 100.0,
    "initial_capital": 10_000.0,
    "risk_per_trade_pct": 0.01,
    "max_lot": 5.0,
    "atr_period": 14,
    "sl_atr_mult": 1.0,
    "tp_atr_mult": 2.0,
    "max_lifetime_bars": 16,
    "cooldown_bars": 8,
    "atr_expansion_lookback": 20,
    "atr_expansion_threshold": 1.2,
    "session_exclude_hours": [21, 22, 23],  # exclure 21h-00h UTC
    "blackout_minutes_before": 60,
    "blackout_minutes_after": 60,
    "random_seed": 42,
}

np.random.seed(CFG_NR4["random_seed"])

# ----------------------------------------------------------------------------
# 1. Charger donnees
# ----------------------------------------------------------------------------
print("[NR4] Loading data...")
df = pd.read_csv(CFG_NR4["data_file"], parse_dates=["Date"])
df = df.rename(columns={"Date": "ts"}).set_index("ts").sort_index()
df.columns = [c.lower() for c in df.columns]
df = df[(df.index >= CFG_NR4["start"]) & (df.index <= CFG_NR4["end"])].copy()
df = df[~df.index.duplicated(keep="first")]
print(f"[NR4] {len(df):,} barres de {df.index.min()} a {df.index.max()}")

cal = pd.read_csv(CFG_NR4["calendar_file"], parse_dates=["Date"])
cal = cal.rename(columns={"Date": "ts"})
cal = cal[cal["Currency"].isin(["USD", "EUR"])].sort_values("ts").reset_index(drop=True)
print(f"[NR4] {len(cal):,} events high-impact USD/EUR")

# ----------------------------------------------------------------------------
# 2. Features NR4
# ----------------------------------------------------------------------------
print("[NR4] Computing features...")
df["range"] = df["high"] - df["low"]
df["atr"] = base.compute_atr(df, CFG_NR4["atr_period"])

# NR4 : range[t] est min des 4 derniers (incluant t)
df["min_range_4"] = df["range"].rolling(4).min()
df["is_nr4"] = df["range"] == df["min_range_4"]

# ATR expansion
df["atr_lag20"] = df["atr"].shift(CFG_NR4["atr_expansion_lookback"])
df["atr_ratio"] = df["atr"] / df["atr_lag20"]
df["atr_expanding"] = df["atr_ratio"] > CFG_NR4["atr_expansion_threshold"]

# Pour le breakout, on veut le high/low de LA NR4 bar (qui est dans le passe immediate)
# Approche : detect la NR4 sur t-1 (la bougie precedente etait NR4),
# puis trigger sur t (close > high_NR4 ou close < low_NR4) avec ATR expansion
# C'est la lecture habituelle du Crabel signal
df["nr4_high"] = df["high"].where(df["is_nr4"]).ffill()
df["nr4_low"] = df["low"].where(df["is_nr4"]).ffill()

# Reference NR4 = le plus recent NR4 dans les ~10 dernieres bars (avant t)
# On lock la fenetre : NR4 valide pour 8 barres apres son apparition
df["nr4_age"] = (~df["is_nr4"]).astype(int)
# rolling cumsum reset on each NR4 - approximation: just shift the high/low after detection
# Plus simple: utiliser le dernier NR4 ffill avec un cap a 8 barres (sinon stale)
nr4_idx_arr = np.where(df["is_nr4"].values)[0]
last_nr4_idx = np.full(len(df), -1, dtype=int)
for i, ix in enumerate(nr4_idx_arr):
    end = nr4_idx_arr[i+1] if i+1 < len(nr4_idx_arr) else len(df)
    last_nr4_idx[ix:end] = ix

age = np.arange(len(df)) - last_nr4_idx
df["nr4_age_bars"] = age
df["nr4_valid"] = (age > 0) & (age <= 8)  # NR4 reference est dans les 8 dernieres bars
# Reset des high/low si stale
df.loc[~df["nr4_valid"], "nr4_high"] = np.nan
df.loc[~df["nr4_valid"], "nr4_low"] = np.nan

# Signal triggering (apres confirmation breakout intra-bar)
df["signal_long"] = (
    (df["close"] > df["nr4_high"])
    & df["atr_expanding"]
    & df["nr4_valid"]
)
df["signal_short"] = (
    (df["close"] < df["nr4_low"])
    & df["atr_expanding"]
    & df["nr4_valid"]
)

# News blackout +/-60 min
print("[NR4] Computing news blackout +/-60 min...")
blackout = pd.Series(False, index=df.index)
before = pd.Timedelta(minutes=CFG_NR4["blackout_minutes_before"])
after = pd.Timedelta(minutes=CFG_NR4["blackout_minutes_after"])
for ts in cal["ts"]:
    start = ts - before
    end = ts + after
    idx = (df.index >= start) & (df.index <= end)
    blackout.iloc[np.where(idx)[0]] = True
df["news_blackout"] = blackout
df["news_ok"] = ~df["news_blackout"]

# Session filter
df["hour"] = df.index.hour
df["session_ok"] = ~df["hour"].isin(CFG_NR4["session_exclude_hours"])

# Combined entry filter
df["entry_ok"] = df["news_ok"] & df["session_ok"]

n_signals_long_raw = df["signal_long"].sum()
n_signals_short_raw = df["signal_short"].sum()
n_signals_long_filtered = (df["signal_long"] & df["entry_ok"]).sum()
n_signals_short_filtered = (df["signal_short"] & df["entry_ok"]).sum()
print(f"[NR4] Signals brut: long={n_signals_long_raw}, short={n_signals_short_raw}")
print(f"[NR4] Signals apres filtres: long={n_signals_long_filtered}, short={n_signals_short_filtered}")

# ----------------------------------------------------------------------------
# 3. Simulateur (reutilise structure audit script)
# ----------------------------------------------------------------------------
class TradeNR4:
    __slots__ = ("id", "ts_in", "ts_out", "side", "entry", "sl", "tp",
                 "exit_price", "exit_reason", "lots", "pnl_usd", "r_realized",
                 "bars_held", "session", "year")
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def simulate_nr4(df):
    trades = []
    open_trade = None
    bar_open_idx = -1
    last_exit_idx = -10_000
    capital = CFG_NR4["initial_capital"]
    equity = pd.Series(index=df.index, dtype=float)
    equity.iloc[0] = capital

    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    atrs = df["atr"].values
    sig_long = df["signal_long"].values
    sig_short = df["signal_short"].values
    entry_ok = df["entry_ok"].values
    timestamps = df.index
    rng = np.random.default_rng(CFG_NR4["random_seed"])

    for i in range(50, len(df)):
        ts = timestamps[i]
        c = closes[i]
        h = highs[i]
        l = lows[i]
        atr = atrs[i]
        if np.isnan(atr) or atr <= 0:
            equity.iloc[i] = capital
            continue

        # 1. Sortie
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
            if exit_price is None and (i - bar_open_idx) >= CFG_NR4["max_lifetime_bars"]:
                exit_price = c
                exit_reason = "timeout"
            if exit_price is not None:
                slip = rng.uniform(CFG_NR4["slippage_min"], CFG_NR4["slippage_max"])
                if side == 1:
                    fill = exit_price - slip - CFG_NR4["spread_usd"] / 2
                else:
                    fill = exit_price + slip + CFG_NR4["spread_usd"] / 2
                pnl_per_oz = (fill - open_trade.entry) * side
                pnl_usd = pnl_per_oz * open_trade.lots * CFG_NR4["lot_size_oz"]
                pnl_usd -= CFG_NR4["commission_per_lot_rt"] * open_trade.lots
                capital += pnl_usd
                sl_dist = abs(open_trade.entry - open_trade.sl)
                r = pnl_per_oz / sl_dist if sl_dist > 0 else 0.0
                open_trade.ts_out = ts
                open_trade.exit_price = fill
                open_trade.exit_reason = exit_reason
                open_trade.pnl_usd = pnl_usd
                open_trade.r_realized = r
                open_trade.bars_held = i - bar_open_idx
                trades.append(open_trade)
                last_exit_idx = i
                open_trade = None

        # 2. Entree (cooldown)
        if open_trade is None and (i - last_exit_idx) >= CFG_NR4["cooldown_bars"]:
            decision = 0
            if entry_ok[i]:
                if sig_long[i]:
                    decision = 1
                elif sig_short[i]:
                    decision = -1
            if decision != 0:
                slip = rng.uniform(CFG_NR4["slippage_min"], CFG_NR4["slippage_max"])
                if decision == 1:
                    entry = c + slip + CFG_NR4["spread_usd"] / 2
                    sl = entry - CFG_NR4["sl_atr_mult"] * atr
                    tp = entry + CFG_NR4["tp_atr_mult"] * atr
                else:
                    entry = c - slip - CFG_NR4["spread_usd"] / 2
                    sl = entry + CFG_NR4["sl_atr_mult"] * atr
                    tp = entry - CFG_NR4["tp_atr_mult"] * atr
                sl_dist = abs(entry - sl)
                if sl_dist <= 0:
                    equity.iloc[i] = capital
                    continue
                risk_usd = capital * CFG_NR4["risk_per_trade_pct"]
                lots = risk_usd / (sl_dist * CFG_NR4["lot_size_oz"])
                lots = round(min(max(lots, 0.01), CFG_NR4["max_lot"]), 2)
                hour = ts.hour
                if 0 <= hour < 7: sess = "asia"
                elif 7 <= hour < 13: sess = "london"
                elif 13 <= hour < 21: sess = "ny"
                else: sess = "off"
                open_trade = TradeNR4(
                    id=len(trades) + 1, ts_in=ts, ts_out=None, side=decision,
                    entry=entry, sl=sl, tp=tp, exit_price=None, exit_reason=None,
                    lots=lots, pnl_usd=0.0, r_realized=0.0, bars_held=0,
                    session=sess, year=ts.year
                )
                bar_open_idx = i

        equity.iloc[i] = capital

    equity = equity.ffill().fillna(CFG_NR4["initial_capital"])
    return trades, equity

# ----------------------------------------------------------------------------
# 4. Run
# ----------------------------------------------------------------------------
print("[NR4] Simulating...")
trades, equity = simulate_nr4(df)
print(f"[NR4] {len(trades)} trades generated")

# Construire DataFrame
trades_df = pd.DataFrame([{
    "id": t.id, "ts_in": t.ts_in, "ts_out": t.ts_out, "side": t.side,
    "entry": t.entry, "sl": t.sl, "tp": t.tp, "exit_price": t.exit_price,
    "exit_reason": t.exit_reason, "lots": t.lots, "pnl_usd": t.pnl_usd,
    "r_realized": t.r_realized, "bars_held": t.bars_held, "session": t.session,
    "year": t.year,
} for t in trades])
trades_df.to_csv(OUT / "L4_baseline_nr4_trades.csv", index=False)

# ----------------------------------------------------------------------------
# 5. Metrics
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

def metrics(trades_list, equity_series, label="ALL"):
    if not trades_list:
        return {"label": label, "n_trades": 0}
    pnl = np.array([t.pnl_usd for t in trades_list])
    wr = (pnl > 0).mean()
    pf_v = pf_fn(pnl)
    ci_lo, ci_hi = bootstrap_pf(pnl, 2000)
    eq_monthly = equity_series.resample("ME").last().pct_change().dropna()
    sharpe = (eq_monthly.mean()/eq_monthly.std()*np.sqrt(12)) if len(eq_monthly)>1 and eq_monthly.std()>0 else float("nan")
    if len(eq_monthly)>1:
        downside = eq_monthly[eq_monthly < 0]
        sortino = (eq_monthly.mean()/downside.std()*np.sqrt(12)) if len(downside)>1 and downside.std()>0 else float("nan")
    else:
        sortino = float("nan")
    running_max = equity_series.cummax()
    dd = (equity_series - running_max) / running_max
    return {
        "label": label,
        "n_trades": len(trades_list),
        "wr": float(wr),
        "pf": float(pf_v),
        "pf_ci_lo": ci_lo,
        "pf_ci_hi": ci_hi,
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_dd_pct": float(dd.min()),
        "expectancy_r": float(np.array([t.r_realized for t in trades_list]).mean()),
        "final_capital": float(equity_series.iloc[-1]),
        "return_pct": float(equity_series.iloc[-1]/CFG_NR4["initial_capital"]-1),
        "n_tp": int(sum(1 for t in trades_list if t.exit_reason == "tp")),
        "n_sl": int(sum(1 for t in trades_list if t.exit_reason == "sl")),
        "n_timeout": int(sum(1 for t in trades_list if t.exit_reason == "timeout")),
    }

m_full = metrics(trades, equity, "FULL_2019_2026")
print("\n[NR4] Metriques globales:")
for k, v in m_full.items():
    if isinstance(v, float):
        print(f"  {k}: {v:.4f}")
    else:
        print(f"  {k}: {v}")

# Decomposition
def metrics_subset(filter_fn, label, equity_full):
    sub_trades = [t for t in trades if filter_fn(t)]
    if not sub_trades:
        return {"label": label, "n_trades": 0}
    sub_pnl = np.array([t.pnl_usd for t in sub_trades])
    cap0 = CFG_NR4["initial_capital"]
    eq = pd.Series(cap0 + np.cumsum(sub_pnl), index=[t.ts_out for t in sub_trades])
    eq = eq.resample("D").last().ffill()
    return metrics(sub_trades, eq, label)

# Par annee
yearly = []
for y in range(2019, 2027):
    m = metrics_subset(lambda t,y=y: t.ts_in.year == y, f"year_{y}", equity)
    yearly.append(m)
df_yearly = pd.DataFrame(yearly)

# Periode pre/post 2024
pre = metrics_subset(lambda t: t.ts_in < pd.Timestamp("2024-01-01"), "2019-2023", equity)
post = metrics_subset(lambda t: t.ts_in >= pd.Timestamp("2024-01-01"), "2024-2026", equity)

# Par side
m_long = metrics_subset(lambda t: t.side == 1, "Long_full", equity)
m_short = metrics_subset(lambda t: t.side == -1, "Short_full", equity)

# Bear / Bull / Range - use SMA200 + slope from base script
df_regime = df.copy()
sma200 = df_regime["close"].rolling(200).mean()
slope = sma200.diff(20)
df_regime["regime"] = np.where(
    (df_regime["close"] > sma200) & (slope > 0), "bull",
    np.where((df_regime["close"] < sma200) & (slope < 0), "bear", "range")
)
trade_regime = {t.ts_in: df_regime.loc[t.ts_in, "regime"] if t.ts_in in df_regime.index else "na" for t in trades}
m_bull = metrics_subset(lambda t: trade_regime.get(t.ts_in) == "bull", "regime_bull", equity)
m_bear = metrics_subset(lambda t: trade_regime.get(t.ts_in) == "bear", "regime_bear", equity)
m_range = metrics_subset(lambda t: trade_regime.get(t.ts_in) == "range", "regime_range", equity)

# ----------------------------------------------------------------------------
# 6. Comparaison vs SMC
# ----------------------------------------------------------------------------
# Charger SMC
smc_summary_path = ROOT / "reports" / "audit_2026_04_30_summary.json"
with open(smc_summary_path) as f:
    smc = json.load(f)
smc_full = {
    "label": "SMC_FULL",
    "n_trades": smc["metrics"]["n_trades"],
    "wr": smc["metrics"]["win_rate"],
    "pf": smc["metrics"]["profit_factor"],
    "sharpe": smc["metrics"]["sharpe"],
    "sortino": smc["metrics"]["sortino"],
    "max_dd_pct": smc["metrics"]["max_dd_pct"],
    "expectancy_r": smc["metrics"]["expectancy_r"],
    "return_pct": smc["metrics"]["return_pct"],
}

print("\n[NR4] Comparaison apples-to-apples NR4 vs SMC:")
comp_df = pd.DataFrame([m_full, smc_full])
print(comp_df[["label","n_trades","wr","pf","sharpe","sortino","max_dd_pct","expectancy_r","return_pct"]]
      .to_string(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else x))

# ----------------------------------------------------------------------------
# 7. Save
# ----------------------------------------------------------------------------
out = {
    "config": {k: str(v) if isinstance(v, Path) else v for k, v in CFG_NR4.items()},
    "metrics_full": m_full,
    "metrics_yearly": yearly,
    "metrics_pre_2024": pre,
    "metrics_post_2024": post,
    "metrics_long": m_long,
    "metrics_short": m_short,
    "metrics_bull": m_bull,
    "metrics_bear": m_bear,
    "metrics_range": m_range,
    "smc_full_for_comparison": smc_full,
}
with open(OUT / "L4_baseline_nr4.json", "w") as f:
    json.dump(out, f, indent=2, default=float)

print("\n[NR4] Tableau annuel:")
print(df_yearly[["label","n_trades","wr","pf","pf_ci_lo","pf_ci_hi","max_dd_pct"]]
      .to_string(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else x))

print("\n[NR4] DONE. Files: L4_baseline_nr4_trades.csv, L4_baseline_nr4.json")
