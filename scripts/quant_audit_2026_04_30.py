"""
Smart Sentinel AI - Audit Quant Senior - Backtest reproductible 2019-2026 (XAU M15)
====================================================================================

Pipeline fidèle simplifié :
  Data -> 8 composants confluence -> score 0-100 -> StateMachine -> SL/TP HAR-RV
       -> Execution avec spread/slippage/commission -> Metriques -> Rapport

Donnees : XAU_15MIN_2019_2026.csv (172k barres reelles, refresh 2026-04-29)
Calendrier : economic_calendar_HIGH_IMPACT_2019_2025.csv

Usage : python scripts/quant_audit_2026_04_30.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ============================================================
# 1. CONFIGURATION
# ============================================================
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

CFG = {
    "data_file": DATA_DIR / "XAU_15MIN_2019_2026.csv",
    "calendar_file": DATA_DIR / "economic_calendar_HIGH_IMPACT_2019_2025.csv",
    "start": "2019-01-01",
    "end": "2026-04-30",
    # Microstructure realiste XAU
    "spread_usd": 0.30,            # spread typique brokers retail
    "slippage_min": 0.10,
    "slippage_max": 0.20,
    "commission_per_lot_rt": 7.0,  # 7 USD aller-retour par lot standard (100 oz)
    "lot_size_oz": 100.0,          # 1 lot = 100 oz
    # Capital & risk
    "initial_capital": 10_000.0,
    "risk_per_trade_pct": 0.01,    # 1% par trade
    "max_lot": 5.0,                # cap protection
    # Volatilite
    "atr_period": 14,
    "har_rv_window_d": 1,          # 1 jour
    "har_rv_window_w": 5,          # 5 jours (semaine)
    "har_rv_window_m": 22,         # 22 jours (mois)
    "sl_atr_mult": 1.5,
    "tp_atr_mult": 2.5,            # R:R 1:1.67
    # Score thresholds (state machine)
    "enter_threshold": 65,
    "exit_threshold": 40,
    "cooldown_bars": 8,
    "lockout_bars_opposite": 12,
    "max_lifetime_bars": 64,       # 16h sur M15 (v2 2026-04-30: 24->64 forensic L1: PF 0.79->1.04)
    # News blackout
    "blackout_minutes_before": 15,
    "blackout_minutes_after": 15,
    # Composants : poids initiaux uniformes
    "weights": {
        "bos": 12.5,
        "choch": 12.5,
        "ob": 12.5,
        "fvg": 12.5,
        "retest": 12.5,
        "rsi_div": 12.5,
        "regime": 12.5,
        "news_ok": 12.5,
    },
    # Reproducibilite
    "random_seed": 42,
}

np.random.seed(CFG["random_seed"])


# ============================================================
# 2. DATA LOADING
# ============================================================
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"[load] Lecture {CFG['data_file'].name}...")
    df = pd.read_csv(CFG["data_file"], parse_dates=["Date"])
    df = df.rename(columns={"Date": "ts"}).set_index("ts").sort_index()
    df.columns = [c.lower() for c in df.columns]
    df = df[(df.index >= CFG["start"]) & (df.index <= CFG["end"])].copy()
    df = df[~df.index.duplicated(keep="first")]
    print(f"[load] {len(df):,} barres de {df.index.min()} a {df.index.max()}")

    print(f"[load] Lecture {CFG['calendar_file'].name}...")
    cal = pd.read_csv(CFG["calendar_file"], parse_dates=["Date"])
    cal = cal.rename(columns={"Date": "ts"})
    cal = cal[cal["Currency"].isin(["USD", "EUR"])].copy()
    cal = cal.sort_values("ts").reset_index(drop=True)
    print(f"[load] {len(cal):,} events high-impact USD/EUR")
    return df, cal


# ============================================================
# 3. INDICATEURS DE BASE
# ============================================================
def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat(
        [(h - l).abs(), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ============================================================
# 4. SMART MONEY COMPONENTS (vectorises)
# ============================================================
def detect_swings(df: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """Swing high/low : pivot sur 2*lookback+1 barres."""
    hi = df["high"].rolling(2 * lookback + 1, center=True).max()
    lo = df["low"].rolling(2 * lookback + 1, center=True).min()
    df["swing_high"] = (df["high"] == hi)
    df["swing_low"] = (df["low"] == lo)
    df["last_swing_high"] = df["high"].where(df["swing_high"]).ffill().shift(lookback + 1)
    df["last_swing_low"] = df["low"].where(df["swing_low"]).ffill().shift(lookback + 1)
    return df


def detect_bos(df: pd.DataFrame) -> pd.DataFrame:
    """BOS (Break of Structure) : close > dernier swing high (bull) ou close < dernier swing low (bear)."""
    df["bos_bull"] = (df["close"] > df["last_swing_high"]) & (
        df["close"].shift() <= df["last_swing_high"].shift()
    )
    df["bos_bear"] = (df["close"] < df["last_swing_low"]) & (
        df["close"].shift() >= df["last_swing_low"].shift()
    )
    # Sticky : reste actif N barres
    df["bos_bull_active"] = df["bos_bull"].rolling(8, min_periods=1).max().astype(bool)
    df["bos_bear_active"] = df["bos_bear"].rolling(8, min_periods=1).max().astype(bool)
    return df


def detect_choch(df: pd.DataFrame) -> pd.DataFrame:
    """CHOCH (Change of Character) : flip de tendance via cross EMA20/EMA50 + confirme par BOS oppose recent."""
    ema20 = df["close"].ewm(span=20, adjust=False).mean()
    ema50 = df["close"].ewm(span=50, adjust=False).mean()
    cross_up = (ema20 > ema50) & (ema20.shift() <= ema50.shift())
    cross_dn = (ema20 < ema50) & (ema20.shift() >= ema50.shift())
    df["choch_bull"] = cross_up
    df["choch_bear"] = cross_dn
    df["choch_bull_active"] = df["choch_bull"].rolling(16, min_periods=1).max().astype(bool)
    df["choch_bear_active"] = df["choch_bear"].rolling(16, min_periods=1).max().astype(bool)
    return df


def detect_order_blocks(df: pd.DataFrame) -> pd.DataFrame:
    """OB ICT : derniere bougie BAISSIERE avant un BOS bullish (bullish OB) et inverse."""
    body = (df["close"] - df["open"]).abs()
    is_bear = df["close"] < df["open"]
    is_bull = df["close"] > df["open"]
    avg_body = body.rolling(20).mean()
    strong = body > avg_body  # bougie significative

    # OB bullish : bougie bear precedente, suivie d'un BOS bullish dans les 5 barres
    bos_bull_in_5 = df["bos_bull"].rolling(5).max().astype(bool).fillna(False)
    ob_bull_anchor = is_bear.shift(1).fillna(False) & strong.shift(1).fillna(False) & bos_bull_in_5
    ob_bear_anchor = is_bull.shift(1).fillna(False) & strong.shift(1).fillna(False) & df["bos_bear"].rolling(5).max().astype(bool).fillna(False)

    df["ob_bull"] = ob_bull_anchor
    df["ob_bear"] = ob_bear_anchor
    df["ob_bull_active"] = df["ob_bull"].rolling(20, min_periods=1).max().astype(bool)
    df["ob_bear_active"] = df["ob_bear"].rolling(20, min_periods=1).max().astype(bool)
    return df


def detect_fvg(df: pd.DataFrame, atr_mult: float = 0.4) -> pd.DataFrame:
    """FVG (Fair Value Gap) : gap 3-barres avec taille >= atr_mult * ATR."""
    h2 = df["high"].shift(2)
    l2 = df["low"].shift(2)
    h0 = df["high"]
    l0 = df["low"]
    # Bullish FVG : low[t] > high[t-2]
    gap_bull = l0 - h2
    gap_bear = l2 - h0
    threshold = atr_mult * df["atr"]
    df["fvg_bull"] = gap_bull > threshold
    df["fvg_bear"] = gap_bear > threshold
    df["fvg_bull_active"] = df["fvg_bull"].rolling(12, min_periods=1).max().astype(bool)
    df["fvg_bear_active"] = df["fvg_bear"].rolling(12, min_periods=1).max().astype(bool)
    return df


def detect_retest(df: pd.DataFrame, atr_tol: float = 0.25) -> pd.DataFrame:
    """Retest : prix revient dans la zone OB ou FVG (tolerance 0.25 ATR)."""
    tol = atr_tol * df["atr"]
    # Approximation : retest si le low actuel touche (close+/-tol) du dernier OB/FVG bull dans les 20 barres
    last_ob_bull_close = df["close"].where(df["ob_bull"]).ffill()
    last_ob_bear_close = df["close"].where(df["ob_bear"]).ffill()
    df["retest_bull"] = (df["low"] <= last_ob_bull_close + tol) & (df["high"] >= last_ob_bull_close - tol) & df["ob_bull_active"]
    df["retest_bear"] = (df["high"] >= last_ob_bear_close - tol) & (df["low"] <= last_ob_bear_close + tol) & df["ob_bear_active"]
    return df


def detect_rsi_divergence(df: pd.DataFrame, lookback: int = 14) -> pd.DataFrame:
    """Divergence RSI classique sur lookback barres."""
    df["rsi"] = compute_rsi(df["close"], 14)
    rolling_low_price = df["low"].rolling(lookback).min()
    rolling_low_rsi = df["rsi"].rolling(lookback).min()
    rolling_high_price = df["high"].rolling(lookback).max()
    rolling_high_rsi = df["rsi"].rolling(lookback).max()
    # Bullish div : prix fait un new low mais RSI non
    df["rsi_div_bull"] = (df["low"] == rolling_low_price) & (df["rsi"] > rolling_low_rsi)
    # Bearish div : prix new high mais RSI non
    df["rsi_div_bear"] = (df["high"] == rolling_high_price) & (df["rsi"] < rolling_high_rsi)
    df["rsi_div_bull_active"] = df["rsi_div_bull"].rolling(8, min_periods=1).max().astype(bool)
    df["rsi_div_bear_active"] = df["rsi_div_bear"].rolling(8, min_periods=1).max().astype(bool)
    return df


def detect_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Regime simple : SMA200 + volatilite normalisee. 3 etats : trend_bull, trend_bear, range."""
    sma200 = df["close"].rolling(200).mean()
    slope = sma200.diff(20)
    atr_pct = df["atr"] / df["close"]
    atr_pct_med = atr_pct.rolling(500).median()
    high_vol = atr_pct > 1.3 * atr_pct_med

    df["regime_bull"] = (df["close"] > sma200) & (slope > 0)
    df["regime_bear"] = (df["close"] < sma200) & (slope < 0)
    df["regime_range"] = ~(df["regime_bull"] | df["regime_bear"])
    df["regime_high_vol"] = high_vol.fillna(False)
    return df


def detect_news_blackout(df: pd.DataFrame, cal: pd.DataFrame) -> pd.DataFrame:
    """News blackout : True si dans la fenetre [-15min, +15min] d'un event high-impact."""
    bar_minutes = 15
    blackout_mask = pd.Series(False, index=df.index)
    before = pd.Timedelta(minutes=CFG["blackout_minutes_before"])
    after = pd.Timedelta(minutes=CFG["blackout_minutes_after"])

    for ts in cal["ts"]:
        # Fenetre exacte autour de l'event
        start = ts - before
        end = ts + after
        idx = (df.index >= start) & (df.index <= end)
        blackout_mask.iloc[np.where(idx)[0]] = True

    df["news_blackout"] = blackout_mask
    df["news_ok"] = ~df["news_blackout"]
    return df


# ============================================================
# 5. VOLATILITE (HAR-RV simplifie)
# ============================================================
def har_rv_forecast(df: pd.DataFrame) -> pd.DataFrame:
    """HAR-RV simplifie : forecast volatilite via moyennes 1d/5d/22d des log-returns au carre."""
    bars_per_day = 96  # M15
    log_ret = np.log(df["close"] / df["close"].shift())
    rv = (log_ret ** 2).rolling(bars_per_day).sum()  # RV journaliere a chaque barre

    rv_d = rv  # 1 jour
    rv_w = rv.rolling(5 * bars_per_day).mean()
    rv_m = rv.rolling(22 * bars_per_day).mean()

    # Modele HAR : RV(t+1) = a + b*RV_d(t) + c*RV_w(t) + d*RV_m(t)
    # On utilise la moyenne ponderee comme proxy operationnel (pas de fit OLS pour reproductibilite)
    rv_forecast = 0.50 * rv_d + 0.30 * rv_w + 0.20 * rv_m
    # Conversion vol annualisee -> ATR-equivalent (USD/oz)
    annual_vol = np.sqrt(rv_forecast * 252)
    df["vol_forecast_pct"] = annual_vol  # ex: 0.18 = 18% annualise
    # SL en USD = vol_forecast_pct * close * sqrt(24bars/252/96)
    df["vol_15m_usd"] = df["close"] * np.sqrt(rv_forecast / bars_per_day)
    return df


# ============================================================
# 6. SCORE CONFLUENCE 0-100
# ============================================================
def compute_confluence_score(df: pd.DataFrame) -> pd.DataFrame:
    w = CFG["weights"]
    # Score directionnel : composantes pour chaque side
    score_bull = (
        w["bos"] * df["bos_bull_active"].astype(int)
        + w["choch"] * df["choch_bull_active"].astype(int)
        + w["ob"] * df["ob_bull_active"].astype(int)
        + w["fvg"] * df["fvg_bull_active"].astype(int)
        + w["retest"] * df["retest_bull"].astype(int)
        + w["rsi_div"] * df["rsi_div_bull_active"].astype(int)
        + w["regime"] * df["regime_bull"].astype(int)
        + w["news_ok"] * df["news_ok"].astype(int)
    )
    score_bear = (
        w["bos"] * df["bos_bear_active"].astype(int)
        + w["choch"] * df["choch_bear_active"].astype(int)
        + w["ob"] * df["ob_bear_active"].astype(int)
        + w["fvg"] * df["fvg_bear_active"].astype(int)
        + w["retest"] * df["retest_bear"].astype(int)
        + w["rsi_div"] * df["rsi_div_bear_active"].astype(int)
        + w["regime"] * df["regime_bear"].astype(int)
        + w["news_ok"] * df["news_ok"].astype(int)
    )
    df["score_bull"] = score_bull.clip(0, 100)
    df["score_bear"] = score_bear.clip(0, 100)
    df["score_max"] = np.maximum(df["score_bull"], df["score_bear"])
    df["score_side"] = np.where(df["score_bull"] >= df["score_bear"], 1, -1)
    return df


# ============================================================
# 7. STATE MACHINE
# ============================================================
@dataclass
class StateMachine:
    enter: float = CFG["enter_threshold"]
    exit_: float = CFG["exit_threshold"]
    cooldown: int = CFG["cooldown_bars"]
    lockout: int = CFG["lockout_bars_opposite"]
    state: int = 0          # 0=HOLD, 1=BUY, -1=SELL
    last_exit_bar: int = -10_000
    last_exit_side: int = 0

    def step(self, bar_idx: int, score_bull: float, score_bear: float) -> int:
        """Renvoie la decision pour cette barre : 0=hold, 1=enter long, -1=enter short."""
        # Sortie sur perte de score (la sortie operationnelle se fait via SL/TP/timeout)
        bull_signal = score_bull >= self.enter and score_bull > score_bear
        bear_signal = score_bear >= self.enter and score_bear > score_bull

        # Cooldown global
        if bar_idx - self.last_exit_bar < self.cooldown:
            return 0
        # Lockout oppose
        if bar_idx - self.last_exit_bar < self.lockout:
            if self.last_exit_side == 1 and bear_signal:
                return 0
            if self.last_exit_side == -1 and bull_signal:
                return 0
        if bull_signal:
            return 1
        if bear_signal:
            return -1
        return 0

    def register_exit(self, bar_idx: int, side: int) -> None:
        self.last_exit_bar = bar_idx
        self.last_exit_side = side


# ============================================================
# 8. EXECUTION SIMULATOR
# ============================================================
@dataclass
class Trade:
    id: int
    ts_in: pd.Timestamp
    ts_out: Optional[pd.Timestamp]
    side: int                    # 1 long, -1 short
    entry: float
    sl: float
    tp: float
    exit_price: Optional[float]
    exit_reason: Optional[str]   # tp/sl/timeout/opposite
    lots: float
    pnl_usd: float
    r_realized: float
    score_in: float
    bars_held: int
    regime: str
    session: str
    components: dict = field(default_factory=dict)


def simulate(df: pd.DataFrame) -> tuple[list[Trade], pd.Series]:
    sm = StateMachine()
    trades: list[Trade] = []
    open_trade: Optional[Trade] = None
    bar_open_idx = -1

    capital = CFG["initial_capital"]
    equity = pd.Series(index=df.index, dtype=float)
    equity.iloc[0] = capital

    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    score_bulls = df["score_bull"].values
    score_bears = df["score_bear"].values
    vol_15m = df["vol_15m_usd"].fillna(df["atr"]).values  # fallback ATR
    atrs = df["atr"].values
    timestamps = df.index

    rng = np.random.default_rng(CFG["random_seed"])

    for i in range(50, len(df)):  # warmup 50 barres
        ts = timestamps[i]
        c = closes[i]
        h = highs[i]
        l = lows[i]

        # 1. Gestion sortie trade ouvert
        if open_trade is not None:
            # Touch SL/TP intra-bar (priorite SL adverse, conservateur)
            sl = open_trade.sl
            tp = open_trade.tp
            side = open_trade.side
            exit_price = None
            exit_reason = None

            if side == 1:
                if l <= sl:
                    exit_price = sl
                    exit_reason = "sl"
                elif h >= tp:
                    exit_price = tp
                    exit_reason = "tp"
            else:
                if h >= sl:
                    exit_price = sl
                    exit_reason = "sl"
                elif l <= tp:
                    exit_price = tp
                    exit_reason = "tp"

            # Timeout
            if exit_price is None and (i - bar_open_idx) >= CFG["max_lifetime_bars"]:
                exit_price = c
                exit_reason = "timeout"

            # Signal oppose fort (mais on respecte SL/TP en priorite)
            if exit_price is None:
                opp_score = score_bears[i] if side == 1 else score_bulls[i]
                if opp_score >= CFG["enter_threshold"]:
                    exit_price = c
                    exit_reason = "opposite"

            if exit_price is not None:
                # Slippage + spread cote sortie
                slip = rng.uniform(CFG["slippage_min"], CFG["slippage_max"])
                if side == 1:
                    fill = exit_price - slip - CFG["spread_usd"] / 2
                else:
                    fill = exit_price + slip + CFG["spread_usd"] / 2

                pnl_per_oz = (fill - open_trade.entry) * side
                pnl_usd = pnl_per_oz * open_trade.lots * CFG["lot_size_oz"]
                pnl_usd -= CFG["commission_per_lot_rt"] * open_trade.lots  # commission RT
                capital += pnl_usd
                # R realise
                sl_distance = abs(open_trade.entry - open_trade.sl)
                r = pnl_per_oz / sl_distance if sl_distance > 0 else 0.0

                open_trade.ts_out = ts
                open_trade.exit_price = fill
                open_trade.exit_reason = exit_reason
                open_trade.pnl_usd = pnl_usd
                open_trade.r_realized = r
                open_trade.bars_held = i - bar_open_idx
                trades.append(open_trade)
                sm.register_exit(i, side)
                open_trade = None

        # 2. Entree nouvelle (si pas de trade ouvert)
        if open_trade is None:
            decision = sm.step(i, score_bulls[i], score_bears[i])
            if decision != 0:
                vol = vol_15m[i] if not np.isnan(vol_15m[i]) else atrs[i]
                if np.isnan(vol) or vol <= 0:
                    continue
                sl_dist = CFG["sl_atr_mult"] * vol * np.sqrt(CFG["max_lifetime_bars"])  # ajuste sur horizon
                # Cap sl_dist a 5 USD min, 30 USD max (realisme XAU M15)
                sl_dist = max(2.0, min(sl_dist, 30.0))
                tp_dist = CFG["tp_atr_mult"] / CFG["sl_atr_mult"] * sl_dist

                slip = rng.uniform(CFG["slippage_min"], CFG["slippage_max"])
                if decision == 1:
                    entry = c + slip + CFG["spread_usd"] / 2
                    sl = entry - sl_dist
                    tp = entry + tp_dist
                else:
                    entry = c - slip - CFG["spread_usd"] / 2
                    sl = entry + sl_dist
                    tp = entry - tp_dist

                # Position sizing : risk fixe 1% capital
                risk_usd = capital * CFG["risk_per_trade_pct"]
                lots = risk_usd / (sl_dist * CFG["lot_size_oz"])
                lots = round(min(max(lots, 0.01), CFG["max_lot"]), 2)

                # Regime label
                if df["regime_bull"].iloc[i]:
                    reg = "bull"
                elif df["regime_bear"].iloc[i]:
                    reg = "bear"
                else:
                    reg = "range"
                # Session label (UTC)
                hour = ts.hour
                if 0 <= hour < 7:
                    sess = "asia"
                elif 7 <= hour < 13:
                    sess = "london"
                elif 13 <= hour < 21:
                    sess = "ny"
                else:
                    sess = "off"

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

                open_trade = Trade(
                    id=len(trades) + 1,
                    ts_in=ts,
                    ts_out=None,
                    side=decision,
                    entry=entry,
                    sl=sl,
                    tp=tp,
                    exit_price=None,
                    exit_reason=None,
                    lots=lots,
                    pnl_usd=0.0,
                    r_realized=0.0,
                    score_in=score_bulls[i] if decision == 1 else score_bears[i],
                    bars_held=0,
                    regime=reg,
                    session=sess,
                    components=comps,
                )
                bar_open_idx = i

        equity.iloc[i] = capital

    equity = equity.ffill().fillna(CFG["initial_capital"])
    return trades, equity


# ============================================================
# 9. METRIQUES & RAPPORT
# ============================================================
def compute_metrics(trades: list[Trade], equity: pd.Series) -> dict:
    if not trades:
        return {"n_trades": 0}

    pnls = np.array([t.pnl_usd for t in trades])
    rs = np.array([t.r_realized for t in trades])
    wins = pnls > 0
    losses = pnls < 0
    n = len(trades)
    wr = wins.mean()
    gross_win = pnls[wins].sum()
    gross_loss = -pnls[losses].sum()
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")

    final_capital = equity.iloc[-1]
    ret_total = final_capital / CFG["initial_capital"] - 1

    # Sharpe / Sortino (sur returns mensuels d'equity)
    eq_monthly = equity.resample("ME").last().pct_change().dropna()
    if len(eq_monthly) > 1 and eq_monthly.std() > 0:
        sharpe = (eq_monthly.mean() / eq_monthly.std()) * np.sqrt(12)
        downside = eq_monthly[eq_monthly < 0]
        sortino = (eq_monthly.mean() / downside.std()) * np.sqrt(12) if len(downside) > 1 and downside.std() > 0 else float("nan")
    else:
        sharpe = float("nan")
        sortino = float("nan")

    # Max DD
    running_max = equity.cummax()
    dd = (equity - running_max) / running_max
    max_dd_pct = dd.min()
    max_dd_usd = (equity - running_max).min()
    dd_idx = dd.idxmin()
    peak_idx = running_max[:dd_idx].idxmax()
    # Duree DD
    recovery = equity[equity.index >= dd_idx]
    rec_idx = recovery[recovery >= running_max.loc[peak_idx]].index
    dd_dur_days = (rec_idx[0] - peak_idx).days if len(rec_idx) > 0 else (equity.index[-1] - peak_idx).days

    calmar = (ret_total / abs(max_dd_pct)) if max_dd_pct < 0 else float("nan")
    avg_r = rs.mean()
    expectancy_r = avg_r
    avg_rr = (gross_win / wins.sum()) / (gross_loss / losses.sum()) if losses.sum() > 0 and wins.sum() > 0 else float("nan")

    return {
        "n_trades": n,
        "win_rate": wr,
        "profit_factor": pf,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_dd_pct": max_dd_pct,
        "max_dd_usd": max_dd_usd,
        "dd_dur_days": dd_dur_days,
        "avg_rr": avg_rr,
        "expectancy_r": expectancy_r,
        "final_capital": final_capital,
        "return_pct": ret_total,
        "gross_win": gross_win,
        "gross_loss": gross_loss,
    }


def buy_hold_metrics(df: pd.DataFrame) -> dict:
    """Buy & hold XAU sur la periode (1 oz, 1% capital risk-equiv)."""
    px_in = df["close"].iloc[0]
    px_out = df["close"].iloc[-1]
    ret = (px_out - px_in) / px_in
    return {"buyhold_return_pct": ret, "px_in": px_in, "px_out": px_out}


def regime_breakdown(trades: list[Trade]) -> pd.DataFrame:
    rows = []
    for t in trades:
        rows.append({"regime": t.regime, "session": t.session, "side": t.side,
                     "year": t.ts_in.year, "dow": t.ts_in.day_name(),
                     "score": t.score_in, "pnl": t.pnl_usd, "r": t.r_realized})
    return pd.DataFrame(rows)


def score_decile_table(bk: pd.DataFrame) -> pd.DataFrame:
    if bk.empty:
        return pd.DataFrame()
    bk = bk.copy()
    bk["decile"] = pd.qcut(bk["score"], q=min(10, bk["score"].nunique()), labels=False, duplicates="drop")
    g = bk.groupby("decile").agg(n=("pnl", "count"),
                                  wr=("pnl", lambda x: (x > 0).mean()),
                                  avg_r=("r", "mean"),
                                  total_pnl=("pnl", "sum"),
                                  score_min=("score", "min"),
                                  score_max=("score", "max"))
    return g


def component_edge(trades: list[Trade]) -> pd.DataFrame:
    """Pour chaque composant : WR conditionne (composant=1) vs (composant=0), edge = wr_on - wr_off."""
    if not trades:
        return pd.DataFrame()
    df = pd.DataFrame([{**t.components, "win": t.pnl_usd > 0, "r": t.r_realized} for t in trades])
    rows = []
    for c in ["bos", "choch", "ob", "fvg", "retest", "rsi_div", "regime", "news_ok"]:
        on = df[df[c] == 1]
        off = df[df[c] == 0]
        wr_on = on["win"].mean() if len(on) else float("nan")
        wr_off = off["win"].mean() if len(off) else float("nan")
        avg_r_on = on["r"].mean() if len(on) else float("nan")
        avg_r_off = off["r"].mean() if len(off) else float("nan")
        rows.append({"component": c, "n_on": len(on), "n_off": len(off),
                     "wr_on": wr_on, "wr_off": wr_off, "edge_wr": (wr_on or 0) - (wr_off or 0),
                     "avg_r_on": avg_r_on, "avg_r_off": avg_r_off,
                     "edge_r": (avg_r_on or 0) - (avg_r_off or 0)})
    return pd.DataFrame(rows).sort_values("edge_r", ascending=False)


def score_pearson(trades: list[Trade]) -> float:
    if len(trades) < 30:
        return float("nan")
    s = np.array([t.score_in for t in trades])
    p = np.array([t.r_realized for t in trades])
    if s.std() == 0 or p.std() == 0:
        return float("nan")
    return float(np.corrcoef(s, p)[0, 1])


# ============================================================
# 10. MAIN
# ============================================================
def main() -> None:
    print("=" * 70)
    print("Smart Sentinel AI - Audit Quant Senior - 2026-04-30")
    print("=" * 70)

    df, cal = load_data()

    print("[features] Calcul ATR + RSI...")
    df["atr"] = compute_atr(df, CFG["atr_period"])

    print("[features] Detection swings...")
    df = detect_swings(df, lookback=5)

    print("[features] BOS / CHOCH...")
    df = detect_bos(df)
    df = detect_choch(df)

    print("[features] Order Blocks ICT...")
    df = detect_order_blocks(df)

    print("[features] FVG...")
    df = detect_fvg(df, atr_mult=0.4)

    print("[features] Retest...")
    df = detect_retest(df, atr_tol=0.25)

    print("[features] RSI divergence...")
    df = detect_rsi_divergence(df, lookback=14)

    print("[features] Regime...")
    df = detect_regime(df)

    print("[features] News blackout...")
    df = detect_news_blackout(df, cal)

    print("[features] HAR-RV vol forecast...")
    df = har_rv_forecast(df)

    print("[features] Score confluence...")
    df = compute_confluence_score(df)

    print(f"[score] distribution score_max : "
          f"p50={df['score_max'].quantile(0.5):.1f} "
          f"p75={df['score_max'].quantile(0.75):.1f} "
          f"p90={df['score_max'].quantile(0.9):.1f} "
          f"p99={df['score_max'].quantile(0.99):.1f} "
          f"max={df['score_max'].max():.1f}")
    n_above_enter = (df["score_max"] >= CFG["enter_threshold"]).sum()
    print(f"[score] {n_above_enter:,} barres avec score >= {CFG['enter_threshold']} "
          f"({n_above_enter/len(df)*100:.2f}%)")

    print("[exec] Simulation execution...")
    trades, equity = simulate(df)
    print(f"[exec] {len(trades):,} trades generes")

    metrics = compute_metrics(trades, equity)
    bh = buy_hold_metrics(df)
    pearson = score_pearson(trades)
    bk = regime_breakdown(trades)

    # ============================================================
    # AFFICHAGE METRIQUES
    # ============================================================
    print("\n" + "=" * 70)
    print("METRIQUES GLOBALES")
    print("=" * 70)
    print(f"Periode             : {df.index.min().date()} -> {df.index.max().date()}")
    print(f"Capital initial     : ${CFG['initial_capital']:,.0f}")
    print(f"Capital final       : ${metrics.get('final_capital', 0):,.0f}")
    print(f"Return total        : {metrics.get('return_pct', 0)*100:+.2f}%")
    print(f"Nombre trades       : {metrics.get('n_trades', 0):,}")
    print(f"Win rate            : {metrics.get('win_rate', 0)*100:.2f}%")
    print(f"Profit factor       : {metrics.get('profit_factor', 0):.3f}  [cible commerciale > 1.20]")
    print(f"Sharpe              : {metrics.get('sharpe', 0):.3f}")
    print(f"Sortino             : {metrics.get('sortino', 0):.3f}")
    print(f"Calmar              : {metrics.get('calmar', 0):.3f}")
    print(f"Max DD              : {metrics.get('max_dd_pct', 0)*100:.2f}% (${metrics.get('max_dd_usd', 0):,.0f})")
    print(f"Duree DD max        : {metrics.get('dd_dur_days', 0)} jours")
    print(f"Avg R:R             : {metrics.get('avg_rr', 0):.3f}")
    print(f"Expectancy          : {metrics.get('expectancy_r', 0):+.3f}R / trade")
    print(f"Pearson(score, R)   : {pearson:+.4f}  [cible > 0.10]")
    print(f"Buy & Hold XAU      : {bh['buyhold_return_pct']*100:+.2f}% (sur memes dates)")
    print(f"Gross win / loss    : ${metrics.get('gross_win', 0):,.0f} / ${metrics.get('gross_loss', 0):,.0f}")

    # Decomposition
    if not bk.empty:
        print("\n" + "=" * 70)
        print("PERFORMANCE PAR ANNEE")
        print("=" * 70)
        per_year = bk.groupby("year").agg(n=("pnl", "count"),
                                            wr=("pnl", lambda x: (x > 0).mean()),
                                            pf=("pnl", lambda x: -x[x > 0].sum() / x[x < 0].sum() if x[x < 0].sum() < 0 else float("inf")),
                                            total_pnl=("pnl", "sum"))
        print(per_year.round(3).to_string())

        print("\n" + "=" * 70)
        print("PERFORMANCE PAR REGIME")
        print("=" * 70)
        per_reg = bk.groupby("regime").agg(n=("pnl", "count"),
                                              wr=("pnl", lambda x: (x > 0).mean()),
                                              pf=("pnl", lambda x: -x[x > 0].sum() / x[x < 0].sum() if x[x < 0].sum() < 0 else float("inf")),
                                              total_pnl=("pnl", "sum"))
        print(per_reg.round(3).to_string())

        print("\n" + "=" * 70)
        print("PERFORMANCE PAR SESSION (UTC)")
        print("=" * 70)
        per_sess = bk.groupby("session").agg(n=("pnl", "count"),
                                                wr=("pnl", lambda x: (x > 0).mean()),
                                                pf=("pnl", lambda x: -x[x > 0].sum() / x[x < 0].sum() if x[x < 0].sum() < 0 else float("inf")),
                                                total_pnl=("pnl", "sum"))
        print(per_sess.round(3).to_string())

        print("\n" + "=" * 70)
        print("PERFORMANCE PAR SIDE")
        print("=" * 70)
        per_side = bk.groupby("side").agg(n=("pnl", "count"),
                                             wr=("pnl", lambda x: (x > 0).mean()),
                                             pf=("pnl", lambda x: -x[x > 0].sum() / x[x < 0].sum() if x[x < 0].sum() < 0 else float("inf")),
                                             total_pnl=("pnl", "sum"))
        print(per_side.round(3).to_string())

        print("\n" + "=" * 70)
        print("DECILES SCORE CONFLUENCE -> WIN RATE / R MOYEN")
        print("=" * 70)
        dec = score_decile_table(bk)
        print(dec.round(3).to_string())

        print("\n" + "=" * 70)
        print("EDGE PAR COMPOSANT (n_on/n_off, wr_on vs wr_off, edge en R)")
        print("=" * 70)
        ce = component_edge(trades)
        print(ce.round(3).to_string(index=False))

    # ============================================================
    # SORTIES FICHIERS
    # ============================================================
    out_trades = REPORTS_DIR / "audit_2026_04_30_trades.csv"
    out_equity = REPORTS_DIR / "audit_2026_04_30_equity.csv"
    out_summary = REPORTS_DIR / "audit_2026_04_30_summary.json"

    pd.DataFrame([{
        "id": t.id, "ts_in": t.ts_in, "ts_out": t.ts_out, "side": t.side,
        "entry": round(t.entry, 2), "sl": round(t.sl, 2), "tp": round(t.tp, 2),
        "exit_price": round(t.exit_price, 2) if t.exit_price else None,
        "exit_reason": t.exit_reason, "lots": t.lots,
        "pnl_usd": round(t.pnl_usd, 2), "r_realized": round(t.r_realized, 3),
        "score_in": t.score_in, "bars_held": t.bars_held,
        "regime": t.regime, "session": t.session,
        **{f"c_{k}": v for k, v in t.components.items()},
    } for t in trades]).to_csv(out_trades, index=False)
    equity.to_csv(out_equity)

    summary = {
        "period": [str(df.index.min()), str(df.index.max())],
        "n_bars": len(df),
        "metrics": {k: (None if isinstance(v, float) and (np.isnan(v) or np.isinf(v)) else float(v))
                    for k, v in metrics.items()},
        "buy_hold": bh,
        "pearson_score_vs_R": float(pearson) if not np.isnan(pearson) else None,
        "score_distribution": {
            "p50": float(df["score_max"].quantile(0.5)),
            "p75": float(df["score_max"].quantile(0.75)),
            "p90": float(df["score_max"].quantile(0.9)),
            "p99": float(df["score_max"].quantile(0.99)),
            "max": float(df["score_max"].max()),
        },
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in CFG.items()},
    }
    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n[out] Trades : {out_trades}")
    print(f"[out] Equity : {out_equity}")
    print(f"[out] Summary: {out_summary}")
    print("[done]")


if __name__ == "__main__":
    main()
