"""Historical replay + PnL evaluation for the signal state machine.

Answers the three questions that must be answered before launch:

1. **Does the state machine produce signals at a reasonable cadence?**
   (too few = bored clients, too many = noise)
2. **Is the advice positive on historical data?**
   (win-rate, profit-factor, expectancy, drawdown)
3. **Which exit reasons dominate?**
   (lets us tune — e.g., 80% time-expired → thresholds too wide)

The harness is deterministic: given the same OHLCV + config, it produces
identical results every run. That makes it safe to use for threshold
sweeps and A/B comparisons.

Design
------
Pipeline mirrors live scanner but offline and batched:

    OHLCV df ──► SmartMoneyEngine (vectorized, one-shot)
             └► ConfluenceDetector (per-bar)
             └► SignalStateMachine (per-bar, via BarInput)
             └► transition stream
             └► trade pairing (entry ↔ exit)
             └► PnL + metrics

The LLM narrative engine, semantic cache, and news agent are **bypassed**
because they affect the client UX, not the commercial validity of the
signal series. Volatility forecasting is also bypassed (naive ATR used
instead) to keep the harness fast; the state machine's behaviour is
unchanged because it only reads ``vol_regime`` from the bar input, and
we supply that from a lightweight in-harness classifier.

Regime classifier
-----------------
Replaces :class:`MarketRegimeAgent` for replay only. Uses an SMA-slope
heuristic that's good enough to give ConfluenceDetector a non-neutral
regime score without the full agent's compute cost. Feeding the full
agent would be faithful to live but 100× slower over 100k bars — we
trade some realism for a testable batch harness.
"""

from __future__ import annotations

import logging
import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.intelligence.confluence_detector import (
    ConfluenceDetector,
    ConfluenceSignal,
    SignalType,
)
from src.intelligence.signal_state_machine import (
    BarInput,
    Direction,
    ExitReason,
    PublicState,
    SignalStateMachine,
    StateMachineConfig,
    StateTransition,
)

logger = logging.getLogger(__name__)


# =============================================================================
# SIMPLE REGIME CLASSIFIER (replay-only)
# =============================================================================


@dataclass(frozen=True)
class _SimpleRegime:
    """Duck-typed stand-in for :class:`RegimeAnalysis` — only the fields
    that :meth:`ConfluenceDetector._score_regime` actually reads."""
    regime: str
    confidence: float
    trend_direction: int
    trend_strength: float


def classify_regime_series(
    df: pd.DataFrame,
    fast_window: int = 20,
    slow_window: int = 50,
    uptrend_bps: float = 30.0,
    downtrend_bps: float = -30.0,
    close_col: str = "close",
) -> pd.Series:
    """Vectorised regime tag per bar — ``uptrend`` / ``downtrend`` / ``ranging``.

    ``uptrend_bps`` is the percentage slope (in basis-points) of the fast
    SMA over the slow SMA required to call an uptrend. 30 bps = 0.30% on
    XAUUSD ≈ about $6 at $2000. Reasonable for 15-min bars.
    """
    close = df[close_col] if close_col in df.columns else df["Close"]
    sma_fast = close.rolling(fast_window, min_periods=fast_window).mean()
    sma_slow = close.rolling(slow_window, min_periods=slow_window).mean()
    slope_bps = (sma_fast - sma_slow) / sma_slow * 10000.0
    out = pd.Series(["ranging"] * len(df), index=df.index, dtype=object)
    out[slope_bps >= uptrend_bps] = "uptrend"
    out[slope_bps <= downtrend_bps] = "downtrend"
    out[slope_bps.isna()] = None
    return out


def _regime_for(tag: Optional[str], slope_bps: float) -> Optional[_SimpleRegime]:
    if not tag:
        return None
    strength = min(1.0, abs(slope_bps) / 200.0)  # 200bps slope → full strength
    if tag == "uptrend":
        return _SimpleRegime(regime="uptrend", confidence=0.7,
                             trend_direction=1, trend_strength=strength)
    if tag == "downtrend":
        return _SimpleRegime(regime="downtrend", confidence=0.7,
                             trend_direction=-1, trend_strength=strength)
    return _SimpleRegime(regime="ranging", confidence=0.5,
                         trend_direction=0, trend_strength=0.2)


def classify_vol_regime_series(
    df: pd.DataFrame,
    atr_col: str = "ATR",
    low_quantile: float = 0.25,
    high_quantile: float = 0.95,
) -> pd.Series:
    """Map ATR into ``low`` / ``normal`` / ``high`` buckets using sample quantiles.

    Uses expanding-window quantiles so regime at bar *i* only depends on
    ATR values available up to *i* (no look-ahead bias).

    ``high`` is reserved for the top 5% — those bars force the state
    machine to exit via REGIME_SHIFTED. If we set this too low (e.g., 80%)
    20% of bars become forced-exits, which turns every trade into a
    one-bar round-trip.
    """
    atr = df[atr_col] if atr_col in df.columns else pd.Series(np.nan, index=df.index)
    q_low = atr.expanding(min_periods=100).quantile(low_quantile)
    q_high = atr.expanding(min_periods=100).quantile(high_quantile)
    out = pd.Series(["normal"] * len(df), index=df.index, dtype=object)
    out[atr < q_low] = "low"
    out[atr > q_high] = "high"
    out[atr.isna()] = None
    return out


# =============================================================================
# TRADE & RESULTS DATACLASSES
# =============================================================================


@dataclass(frozen=True)
class TradeRecord:
    """One entry→exit round-trip. Aligned with one state-machine cycle."""

    signal_id: str
    direction: str                     # "LONG" | "SHORT"
    entry_bar: str
    exit_bar: str
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    confluence_score: float
    exit_reason: str
    bars_held: int
    pnl_price: float                   # in quote-currency (e.g., USD for XAU)
    r_multiple: float                  # pnl / initial_risk
    initial_risk: float                # |entry - stop_loss|

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "direction": self.direction,
            "entry_bar": self.entry_bar,
            "exit_bar": self.exit_bar,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "confluence_score": self.confluence_score,
            "exit_reason": self.exit_reason,
            "bars_held": self.bars_held,
            "pnl_price": round(self.pnl_price, 4),
            "r_multiple": round(self.r_multiple, 3),
            "initial_risk": round(self.initial_risk, 4),
        }


@dataclass
class ReplayResults:
    """Aggregate metrics + per-trade records for a replay run."""

    # Configuration echo
    symbol: str
    timeframe: str
    bars_processed: int
    date_range: Tuple[str, str]          # (first_bar, last_bar)
    state_machine_config: Dict[str, Any]

    # Trade-level stats
    trades: List[TradeRecord] = field(default_factory=list)
    open_trade_bars: int = 0             # if replay ends mid-signal

    # Aggregate metrics (filled by _compute_metrics)
    total_trades: int = 0
    wins: int = 0                        # target_reached
    losses: int = 0                      # invalidated
    other_exits: int = 0                 # time / score / regime / opposing
    win_rate: float = 0.0
    loss_rate: float = 0.0
    exits_by_reason: Dict[str, int] = field(default_factory=dict)

    avg_r: float = 0.0
    median_r: float = 0.0
    total_r: float = 0.0
    best_r: float = 0.0
    worst_r: float = 0.0

    gross_win_r: float = 0.0
    gross_loss_r: float = 0.0
    profit_factor: float = 0.0
    expectancy_r: float = 0.0            # avg_r per trade (same as avg_r; named for clarity)

    max_drawdown_r: float = 0.0
    max_consecutive_losses: int = 0
    sharpe_per_trade: float = 0.0        # mean / stdev of R-series (unannualised)

    avg_bars_held: float = 0.0
    median_bars_held: float = 0.0
    signals_per_day: float = 0.0

    # State-machine telemetry
    arms_started: int = 0
    arms_confirmed: int = 0
    arms_aborted: int = 0
    confirmation_rate: Optional[float] = None
    avg_signal_lifetime_bars_machine: float = 0.0

    # Diagnostics: help tune thresholds on fresh data
    bars_with_bos: int = 0
    signals_produced_by_detector: int = 0
    score_percentiles: Dict[str, float] = field(default_factory=dict)
    score_max: float = 0.0

    def to_dict(self, include_trades: bool = True) -> Dict[str, Any]:
        payload = {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "bars_processed": self.bars_processed,
            "date_range": list(self.date_range),
            "state_machine_config": self.state_machine_config,
            "open_trade_bars": self.open_trade_bars,
            "summary": {
                "total_trades": self.total_trades,
                "wins": self.wins,
                "losses": self.losses,
                "other_exits": self.other_exits,
                "win_rate": round(self.win_rate, 4),
                "loss_rate": round(self.loss_rate, 4),
                "exits_by_reason": dict(self.exits_by_reason),
                "avg_r": round(self.avg_r, 3),
                "median_r": round(self.median_r, 3),
                "total_r": round(self.total_r, 3),
                "best_r": round(self.best_r, 3),
                "worst_r": round(self.worst_r, 3),
                "gross_win_r": round(self.gross_win_r, 3),
                "gross_loss_r": round(self.gross_loss_r, 3),
                "profit_factor": round(self.profit_factor, 3),
                "expectancy_r": round(self.expectancy_r, 3),
                "max_drawdown_r": round(self.max_drawdown_r, 3),
                "max_consecutive_losses": self.max_consecutive_losses,
                "sharpe_per_trade": round(self.sharpe_per_trade, 3),
                "avg_bars_held": round(self.avg_bars_held, 2),
                "median_bars_held": round(self.median_bars_held, 2),
                "signals_per_day": round(self.signals_per_day, 3),
                "arms_started": self.arms_started,
                "arms_confirmed": self.arms_confirmed,
                "arms_aborted": self.arms_aborted,
                "confirmation_rate": (
                    round(self.confirmation_rate, 3)
                    if self.confirmation_rate is not None else None
                ),
                "avg_signal_lifetime_bars_machine": round(
                    self.avg_signal_lifetime_bars_machine, 2
                ),
                "bars_with_bos": self.bars_with_bos,
                "signals_produced_by_detector": self.signals_produced_by_detector,
                "score_max": round(self.score_max, 2),
                "score_percentiles": {
                    k: round(v, 2) for k, v in self.score_percentiles.items()
                },
            },
        }
        if include_trades:
            payload["trades"] = [t.to_dict() for t in self.trades]
        return payload

    def pretty(self) -> str:
        """Human-readable summary block — what you paste into a report."""
        s = self
        lines = [
            f"=== Replay report: {s.symbol} {s.timeframe} ===",
            f"Window        : {s.date_range[0]} -> {s.date_range[1]} "
            f"({s.bars_processed:,} bars)",
            f"Trades        : {s.total_trades}  "
            f"(wins={s.wins}, losses={s.losses}, other={s.other_exits})",
            f"Cadence       : {s.signals_per_day:.2f} signals/day   "
            f"avg lifetime {s.avg_bars_held:.1f} bars "
            f"(machine {s.avg_signal_lifetime_bars_machine:.1f})",
            f"Confirm rate  : "
            f"{(s.confirmation_rate * 100 if s.confirmation_rate else 0):.1f}% "
            f"({s.arms_confirmed}/{s.arms_started} arms confirmed, "
            f"{s.arms_aborted} aborted)",
            "",
            f"Win rate      : {s.win_rate * 100:.1f}%   "
            f"Loss rate: {s.loss_rate * 100:.1f}%",
            f"Expectancy    : {s.expectancy_r:+.3f} R / trade  "
            f"(total {s.total_r:+.2f} R)",
            f"Avg / median R: {s.avg_r:+.3f} / {s.median_r:+.3f}  "
            f"best {s.best_r:+.2f}  worst {s.worst_r:+.2f}",
            f"Profit factor : {s.profit_factor:.2f}  "
            f"(wins {s.gross_win_r:+.2f} R vs losses {s.gross_loss_r:+.2f} R)",
            f"Max drawdown  : {s.max_drawdown_r:.2f} R   "
            f"Max consec losses: {s.max_consecutive_losses}",
            f"Sharpe / trade: {s.sharpe_per_trade:+.3f}",
            "",
            "Exits by reason:",
        ]
        for reason, n in sorted(
            s.exits_by_reason.items(), key=lambda kv: -kv[1]
        ):
            pct = 100.0 * n / s.total_trades if s.total_trades else 0.0
            lines.append(f"  {reason:<18s} {n:>5d}  ({pct:5.1f}%)")
        lines.append("")
        lines.append(
            f"Diagnostic   : BOS bars={s.bars_with_bos:,}  "
            f"detector signals={s.signals_produced_by_detector:,}  "
            f"max score={s.score_max:.1f}"
        )
        if s.score_percentiles:
            pct_str = "  ".join(
                f"{k}={v:.1f}" for k, v in s.score_percentiles.items()
            )
            lines.append(f"Score pct    : {pct_str}")
        return "\n".join(lines)


# =============================================================================
# THE REPLAY ENGINE
# =============================================================================


class SignalReplay:
    """Feed a historical OHLCV frame through the full signal pipeline.

    Usage::

        replay = SignalReplay(
            symbol="XAUUSD", timeframe="M15",
            state_machine_config=StateMachineConfig(symbol="XAUUSD"),
        )
        results = replay.run(enriched_df)
        print(results.pretty())
    """

    def __init__(
        self,
        symbol: str = "XAUUSD",
        timeframe: str = "M15",
        state_machine_config: Optional[StateMachineConfig] = None,
        confluence_detector: Optional[ConfluenceDetector] = None,
        use_regime: bool = True,
        use_vol_regime: bool = True,
        warmup_bars: int = 50,
        detector_min_score: Optional[float] = None,
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.state_machine_config = state_machine_config or StateMachineConfig(symbol=symbol)
        # Replay lets the state machine be the sole gating layer. If no custom
        # detector is supplied we default the detector's own filter to a low
        # floor (or the caller's override) so the state machine sees the raw
        # score distribution rather than getting a pre-filtered stream.
        if confluence_detector is not None:
            self.confluence = confluence_detector
        else:
            effective_min = (
                float(detector_min_score)
                if detector_min_score is not None
                else min(
                    self.state_machine_config.exit_threshold,
                    self.state_machine_config.enter_threshold,
                )
            )
            self.confluence = ConfluenceDetector(symbol=symbol, min_score=effective_min)
        self.use_regime = use_regime
        self.use_vol_regime = use_vol_regime
        self.warmup_bars = max(20, int(warmup_bars))

    # ------------------------------------------------------------------ #
    # PUBLIC
    # ------------------------------------------------------------------ #

    def run(self, enriched_df: pd.DataFrame) -> ReplayResults:
        """Execute the replay.

        ``enriched_df`` must already contain SMC features (BOS_SIGNAL,
        FVG_SIGNAL, OB_STRENGTH_NORM, RSI, MACD_Diff, ATR) plus OHLCV in
        lower-case columns (the :class:`SmartMoneyEngine` output format).
        """
        if enriched_df is None or len(enriched_df) == 0:
            raise ValueError("enriched_df is empty")
        required = {"high", "low", "close", "BOS_SIGNAL", "ATR"}
        missing = required - set(enriched_df.columns)
        if missing:
            raise ValueError(f"enriched_df missing required columns: {sorted(missing)}")

        # Precompute regime / vol-regime series (O(N) each, vectorised)
        regime_tags = (
            classify_regime_series(enriched_df)
            if self.use_regime else pd.Series([None] * len(enriched_df), index=enriched_df.index)
        )
        close = enriched_df["close"] if "close" in enriched_df.columns else enriched_df["Close"]
        slow = close.rolling(50, min_periods=50).mean()
        fast = close.rolling(20, min_periods=20).mean()
        slope_bps = (fast - slow) / slow * 10000.0

        vol_tags = (
            classify_vol_regime_series(enriched_df)
            if self.use_vol_regime else pd.Series([None] * len(enriched_df), index=enriched_df.index)
        )

        # Build a fresh state machine for the replay
        sm = SignalStateMachine(self.state_machine_config)
        transitions: List[StateTransition] = []
        bars_processed = 0
        score_samples: List[float] = []  # diagnostic: distribution of confluence scores
        bars_with_bos = 0

        # Iterate bars
        for i in range(self.warmup_bars, len(enriched_df)):
            row = enriched_df.iloc[i]
            bar_ts = str(enriched_df.index[i])

            close_v = float(row.get("close", row.get("Close", 0.0)))
            high_v = float(row.get("high", row.get("High", close_v)))
            low_v = float(row.get("low", row.get("Low", close_v)))
            atr = float(row.get("ATR", 0.0) or 0.0)
            if atr <= 0 or not math.isfinite(atr):
                continue  # Skip bars where ATR is invalid
            if not (0 < low_v <= close_v <= high_v):
                continue  # Skip malformed bars

            smc_features = {
                "BOS_SIGNAL": float(row.get("BOS_SIGNAL", 0) or 0),
                "FVG_SIGNAL": float(row.get("FVG_SIGNAL", 0) or 0),
                "OB_STRENGTH_NORM": float(row.get("OB_STRENGTH_NORM", 0) or 0),
                "RSI": float(row.get("RSI", 50) or 50),
                "MACD_Diff": float(row.get("MACD_Diff", 0) or 0),
                "CHOCH_SIGNAL": float(row.get("CHOCH_SIGNAL", 0) or 0),
                "CHOCH_DIVERGENCE": float(row.get("CHOCH_DIVERGENCE", 0) or 0),
                "FVG_SIZE_NORM": float(row.get("FVG_SIZE_NORM", 0) or 0),
            }

            regime = _regime_for(regime_tags.iloc[i], float(slope_bps.iloc[i] or 0))
            vol_regime = vol_tags.iloc[i] if isinstance(vol_tags.iloc[i], str) else None

            if smc_features["BOS_SIGNAL"] != 0.0:
                bars_with_bos += 1

            signal: Optional[ConfluenceSignal] = None
            try:
                signal = self.confluence.analyze(
                    smc_features=smc_features,
                    regime=regime,
                    news=None,
                    price=close_v,
                    atr=atr,
                    volume=None,
                    volume_ma=None,
                    bar_timestamp=bar_ts,
                    vol_forecast=None,
                )
            except Exception as e:
                logger.debug("Confluence analyse failed at bar %s: %s", bar_ts, e)
                signal = None
            if signal is not None:
                score_samples.append(signal.confluence_score)

            try:
                bar = BarInput(
                    bar_timestamp=bar_ts,
                    high=high_v, low=low_v, close=close_v,
                    signal=signal, vol_regime=vol_regime,
                    structure_broken=False,
                )
            except ValueError:
                continue  # Defensive — malformed bar already filtered above

            _, transition = sm.on_bar(bar)
            bars_processed += 1
            if transition is not None:
                transitions.append(transition)

        # Build trades + metrics
        trades, open_trade_bars = self._pair_trades(transitions, sm)
        results = ReplayResults(
            symbol=self.symbol,
            timeframe=self.timeframe,
            bars_processed=bars_processed,
            date_range=(
                str(enriched_df.index[self.warmup_bars]) if bars_processed else "",
                str(enriched_df.index[-1]) if bars_processed else "",
            ),
            state_machine_config={
                "enter_threshold": self.state_machine_config.enter_threshold,
                "exit_threshold": self.state_machine_config.exit_threshold,
                "confirm_bars": self.state_machine_config.confirm_bars,
                "cooldown_bars": self.state_machine_config.cooldown_bars,
                "max_signal_age_bars": self.state_machine_config.max_signal_age_bars,
                "silent_bars_before_score_exit": (
                    self.state_machine_config.silent_bars_before_score_exit
                ),
                "high_vol_forces_exit": self.state_machine_config.high_vol_forces_exit,
            },
            trades=trades,
            open_trade_bars=open_trade_bars,
            bars_with_bos=bars_with_bos,
            signals_produced_by_detector=len(score_samples),
        )
        if score_samples:
            arr = np.asarray(score_samples)
            results.score_max = float(arr.max())
            results.score_percentiles = {
                "p50": float(np.percentile(arr, 50)),
                "p75": float(np.percentile(arr, 75)),
                "p90": float(np.percentile(arr, 90)),
                "p95": float(np.percentile(arr, 95)),
                "p99": float(np.percentile(arr, 99)),
            }
        self._compute_metrics(results, sm, enriched_df)
        return results

    # ------------------------------------------------------------------ #
    # INTERNALS
    # ------------------------------------------------------------------ #

    @staticmethod
    def _pair_trades(
        transitions: List[StateTransition],
        sm: SignalStateMachine,
    ) -> Tuple[List[TradeRecord], int]:
        """Pair entry and exit transitions into :class:`TradeRecord` trades.

        Invariant: transitions alternate entry (HOLD→BUY/SELL) and exit
        (BUY/SELL→HOLD). A replay that ends mid-signal contributes to
        ``open_trade_bars`` but is not counted as a closed trade.
        """
        trades: List[TradeRecord] = []
        open_bars = 0
        i = 0
        n = len(transitions)
        while i < n:
            t = transitions[i]
            if t.to_state is not PublicState.HOLD:
                # Entry — find the next exit
                entry = t
                if i + 1 >= n:
                    # No pairing — open at end of replay
                    open_bars = sm.snapshot().bars_in_state
                    break
                exit_t = transitions[i + 1]
                if exit_t.to_state is not PublicState.HOLD or exit_t.exit_reason is None:
                    # Unexpected sequence — skip this entry defensively
                    logger.warning(
                        "Unpaired entry transition at %s (next is %s)",
                        entry.at_bar, exit_t.to_state,
                    )
                    i += 1
                    continue
                trades.append(_build_trade(entry, exit_t))
                i += 2
            else:
                # Stray exit without entry — shouldn't happen, skip
                logger.warning("Stray exit transition at %s — skipping", t.at_bar)
                i += 1
        return trades, open_bars

    @staticmethod
    def _compute_metrics(
        results: ReplayResults,
        sm: SignalStateMachine,
        df: pd.DataFrame,
    ) -> None:
        r_series = [t.r_multiple for t in results.trades]
        bars_held = [t.bars_held for t in results.trades]
        results.total_trades = len(results.trades)

        # Exit reason distribution + win/loss buckets
        counts: Dict[str, int] = {}
        for t in results.trades:
            counts[t.exit_reason] = counts.get(t.exit_reason, 0) + 1
        results.exits_by_reason = counts
        results.wins = counts.get(ExitReason.TARGET_REACHED.value, 0)
        results.losses = counts.get(ExitReason.INVALIDATED.value, 0)
        results.other_exits = results.total_trades - results.wins - results.losses
        if results.total_trades:
            results.win_rate = results.wins / results.total_trades
            results.loss_rate = results.losses / results.total_trades

        if r_series:
            results.avg_r = float(np.mean(r_series))
            results.median_r = float(np.median(r_series))
            results.total_r = float(np.sum(r_series))
            results.best_r = float(max(r_series))
            results.worst_r = float(min(r_series))
            results.expectancy_r = results.avg_r
            wins_r = [r for r in r_series if r > 0]
            losses_r = [r for r in r_series if r < 0]
            results.gross_win_r = float(sum(wins_r)) if wins_r else 0.0
            results.gross_loss_r = float(sum(losses_r)) if losses_r else 0.0
            denom = abs(results.gross_loss_r)
            results.profit_factor = (
                results.gross_win_r / denom if denom > 0 else (
                    float("inf") if results.gross_win_r > 0 else 0.0
                )
            )
            results.max_drawdown_r = _max_drawdown_r(r_series)
            results.max_consecutive_losses = _max_consecutive_losses(r_series)
            if len(r_series) > 1 and statistics.stdev(r_series) > 0:
                results.sharpe_per_trade = (
                    statistics.mean(r_series) / statistics.stdev(r_series)
                )
        if bars_held:
            results.avg_bars_held = float(np.mean(bars_held))
            results.median_bars_held = float(np.median(bars_held))

        # Signals/day from date range
        try:
            start = pd.to_datetime(results.date_range[0])
            end = pd.to_datetime(results.date_range[1])
            days = max(1.0, (end - start).total_seconds() / 86400.0)
            results.signals_per_day = results.total_trades / days
        except Exception:
            results.signals_per_day = 0.0

        # Machine-level telemetry
        stats = sm.get_stats()
        results.arms_started = int(stats.get("arms_started", 0))
        results.arms_confirmed = int(stats.get("arms_confirmed", 0))
        results.arms_aborted = int(stats.get("arms_aborted", 0))
        results.confirmation_rate = stats.get("confirmation_rate")
        results.avg_signal_lifetime_bars_machine = float(
            stats.get("avg_signal_lifetime_bars", 0.0) or 0.0
        )


# =============================================================================
# HELPERS (module-level so they're testable)
# =============================================================================


def _build_trade(entry: StateTransition, exit_t: StateTransition) -> TradeRecord:
    """Assemble a TradeRecord from a paired (entry, exit) transition."""
    sig = entry.active_signal
    direction = entry.direction.value if entry.direction else "LONG"

    def _get(name: str, default: Any = 0.0) -> Any:
        if sig is None:
            return default
        if isinstance(sig, dict):
            return sig.get(name, default)
        return getattr(sig, name, default)

    entry_price = float(entry.entry_price or _get("entry_price", 0.0))
    exit_price = float(exit_t.exit_price or 0.0)
    stop_loss = float(_get("stop_loss", entry_price))
    take_profit = float(_get("take_profit", entry_price))
    confluence = float(_get("confluence_score", 0.0))

    initial_risk = abs(entry_price - stop_loss)
    if direction == "LONG":
        pnl = exit_price - entry_price
    else:
        pnl = entry_price - exit_price
    r_mult = pnl / initial_risk if initial_risk > 0 else 0.0

    bars_held = _count_bars_between(entry.at_bar, exit_t.at_bar)

    return TradeRecord(
        signal_id=str(_get("signal_id", "")),
        direction=direction,
        entry_bar=entry.at_bar,
        exit_bar=exit_t.at_bar,
        entry_price=entry_price,
        exit_price=exit_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        confluence_score=confluence,
        exit_reason=exit_t.exit_reason.value if exit_t.exit_reason else "unknown",
        bars_held=bars_held,
        pnl_price=pnl,
        r_multiple=r_mult,
        initial_risk=initial_risk,
    )


def _count_bars_between(entry_ts: str, exit_ts: str) -> int:
    """Approximate bars-held by timestamp diff assuming 15-minute bars.

    If timestamps aren't parseable, fall back to 1. This is a display
    metric; it doesn't drive state-machine decisions.
    """
    try:
        a = pd.to_datetime(entry_ts)
        b = pd.to_datetime(exit_ts)
        delta_min = (b - a).total_seconds() / 60.0
        # Nearest multiple of 15 minutes, minimum 1
        return max(1, int(round(delta_min / 15.0)))
    except Exception:
        return 1


def _max_drawdown_r(r_series: List[float]) -> float:
    """Peak-to-trough drawdown on the cumulative R-curve."""
    if not r_series:
        return 0.0
    cum = np.cumsum(r_series)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    return float(dd.max())


def _max_consecutive_losses(r_series: List[float]) -> int:
    """Longest run of R <= 0."""
    best = run = 0
    for r in r_series:
        if r <= 0:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best
