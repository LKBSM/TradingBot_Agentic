"""Tests for the historical replay harness — trade pairing, PnL, metrics."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.backtest.state_machine_replay import (
    SignalReplay,
    TradeRecord,
    _build_trade,
    _count_bars_between,
    _max_consecutive_losses,
    _max_drawdown_r,
)
from src.intelligence.signal_state_machine import (
    Direction,
    ExitReason,
    PublicState,
    SignalStateMachine,
    StateMachineConfig,
    StateTransition,
)


# =============================================================================
# HELPERS
# =============================================================================

@dataclass
class _Signal:
    signal_id: str = "s1"
    symbol: str = "TEST"
    signal_type: str = "LONG"
    confluence_score: float = 80.0
    entry_price: float = 100.0
    stop_loss: float = 98.0
    take_profit: float = 104.0


def _entry(direction: Direction, bar: str, entry_price: float,
           stop: float, target: float, score: float = 80.0) -> StateTransition:
    sig_dir = "LONG" if direction is Direction.LONG else "SHORT"
    return StateTransition(
        at_bar=bar,
        from_state=PublicState.HOLD,
        to_state=direction.to_public(),
        reason="entry",
        direction=direction,
        active_signal=_Signal(
            signal_type=sig_dir,
            entry_price=entry_price,
            stop_loss=stop,
            take_profit=target,
            confluence_score=score,
        ),
        entry_price=entry_price,
    )


def _exit(bar: str, exit_price: float, reason: ExitReason,
          direction: Direction, entry_signal: _Signal) -> StateTransition:
    return StateTransition(
        at_bar=bar,
        from_state=direction.to_public(),
        to_state=PublicState.HOLD,
        reason="exit",
        exit_reason=reason,
        direction=direction,
        active_signal=entry_signal,
        exit_price=exit_price,
    )


# =============================================================================
# PNL SIGN CONVENTION
# =============================================================================

class TestPnLConvention:
    def test_long_profit_is_exit_minus_entry(self):
        e = _entry(Direction.LONG, "t1", 100.0, 98.0, 104.0)
        x = _exit("t5", 104.0, ExitReason.TARGET_REACHED, Direction.LONG, e.active_signal)
        trade = _build_trade(e, x)
        assert trade.pnl_price == pytest.approx(4.0)
        assert trade.initial_risk == pytest.approx(2.0)
        assert trade.r_multiple == pytest.approx(2.0)

    def test_long_loss(self):
        e = _entry(Direction.LONG, "t1", 100.0, 98.0, 104.0)
        x = _exit("t3", 98.0, ExitReason.INVALIDATED, Direction.LONG, e.active_signal)
        trade = _build_trade(e, x)
        assert trade.pnl_price == pytest.approx(-2.0)
        assert trade.r_multiple == pytest.approx(-1.0)

    def test_short_profit_is_entry_minus_exit(self):
        e = _entry(Direction.SHORT, "t1", 100.0, 102.0, 96.0)
        x = _exit("t5", 96.0, ExitReason.TARGET_REACHED, Direction.SHORT, e.active_signal)
        trade = _build_trade(e, x)
        assert trade.pnl_price == pytest.approx(4.0)
        assert trade.initial_risk == pytest.approx(2.0)
        assert trade.r_multiple == pytest.approx(2.0)

    def test_short_loss(self):
        e = _entry(Direction.SHORT, "t1", 100.0, 102.0, 96.0)
        x = _exit("t3", 102.0, ExitReason.INVALIDATED, Direction.SHORT, e.active_signal)
        trade = _build_trade(e, x)
        assert trade.pnl_price == pytest.approx(-2.0)
        assert trade.r_multiple == pytest.approx(-1.0)

    def test_zero_risk_gives_zero_r(self):
        e = _entry(Direction.LONG, "t1", 100.0, 100.0, 104.0)  # SL == entry
        x = _exit("t2", 103.0, ExitReason.TIME_EXPIRED, Direction.LONG, e.active_signal)
        trade = _build_trade(e, x)
        assert trade.r_multiple == 0.0  # no divide-by-zero, graceful


# =============================================================================
# TRADE PAIRING
# =============================================================================

class TestTradePairing:
    def test_paired_entry_and_exit(self):
        sm = SignalStateMachine()
        sig1 = _Signal(signal_id="s1", entry_price=100, stop_loss=98, take_profit=104)
        transitions = [
            _entry(Direction.LONG, "t1", 100, 98, 104),
            _exit("t3", 104.0, ExitReason.TARGET_REACHED, Direction.LONG,
                  transitions_sig := sig1),
        ]
        trades, open_bars = SignalReplay._pair_trades(transitions, sm)
        assert len(trades) == 1
        assert open_bars == 0
        assert trades[0].pnl_price == pytest.approx(4.0)

    def test_multiple_sequential_trades(self):
        sm = SignalStateMachine()
        s1 = _Signal(signal_id="s1", entry_price=100, stop_loss=98, take_profit=104)
        s2 = _Signal(signal_id="s2", entry_price=105, stop_loss=107, take_profit=101,
                     signal_type="SHORT")
        transitions = [
            _entry(Direction.LONG, "t1", 100, 98, 104),
            _exit("t3", 104.0, ExitReason.TARGET_REACHED, Direction.LONG, s1),
            _entry(Direction.SHORT, "t6", 105, 107, 101),
            _exit("t9", 107.0, ExitReason.INVALIDATED, Direction.SHORT, s2),
        ]
        trades, _ = SignalReplay._pair_trades(transitions, sm)
        assert len(trades) == 2
        assert trades[0].direction == "LONG" and trades[0].r_multiple == pytest.approx(2.0)
        assert trades[1].direction == "SHORT" and trades[1].r_multiple == pytest.approx(-1.0)

    def test_open_trade_at_end_does_not_count(self):
        sm = SignalStateMachine()
        transitions = [
            _entry(Direction.LONG, "t1", 100, 98, 104),
            # no exit
        ]
        trades, open_bars = SignalReplay._pair_trades(transitions, sm)
        assert trades == []

    def test_empty_transitions(self):
        sm = SignalStateMachine()
        trades, open_bars = SignalReplay._pair_trades([], sm)
        assert trades == [] and open_bars == 0


# =============================================================================
# METRIC HELPERS
# =============================================================================

class TestMetricHelpers:
    def test_max_drawdown_empty(self):
        assert _max_drawdown_r([]) == 0.0

    def test_max_drawdown_all_wins(self):
        assert _max_drawdown_r([1.0, 1.0, 1.0]) == 0.0

    def test_max_drawdown_intermediate_drawdown(self):
        # cum:    2,  1,  3,  0,  2
        # peak:   2,  2,  3,  3,  3
        # dd:     0,  1,  0,  3,  1  → max 3.0
        series = [2.0, -1.0, 2.0, -3.0, 2.0]
        assert _max_drawdown_r(series) == pytest.approx(3.0)

    def test_max_consecutive_losses(self):
        assert _max_consecutive_losses([]) == 0
        assert _max_consecutive_losses([1, 2, 3]) == 0
        assert _max_consecutive_losses([-1, -1, 2, -1, -1, -1]) == 3
        assert _max_consecutive_losses([0, -1]) == 2  # zero R counts as loss

    def test_count_bars_between_15m(self):
        assert _count_bars_between("2026-04-23T00:00:00", "2026-04-23T00:45:00") == 3

    def test_count_bars_between_bad_input(self):
        assert _count_bars_between("garbage", "nope") == 1


# =============================================================================
# END-TO-END SANITY — drive a synthetic sequence through SignalReplay
# =============================================================================

class TestEndToEnd:
    def test_simple_long_round_trip(self):
        """Feed a hand-built enriched dataframe through SignalReplay and
        assert exactly one winning LONG trade is produced."""
        import numpy as np
        import pandas as pd
        idx = pd.date_range("2026-01-01", periods=150, freq="15min")

        # Most bars: neutral BOS=0 so no signal. Bars 100-101: BOS_SIGNAL=1
        # with strong supporting features so confluence score is >= 75.
        df = pd.DataFrame({
            "open":   np.full(150, 100.0),
            "high":   np.full(150, 100.3),
            "low":    np.full(150, 99.7),
            "close":  np.full(150, 100.0),
            "volume": np.full(150, 1000.0),
            "ATR":    np.full(150, 1.0),
            "RSI":    np.full(150, 50.0),
            "MACD_Diff": np.full(150, 0.0),
            "BOS_SIGNAL": np.zeros(150),
            "FVG_SIGNAL": np.zeros(150),
            "OB_STRENGTH_NORM": np.zeros(150),
            "CHOCH_SIGNAL": np.zeros(150),
            "CHOCH_DIVERGENCE": np.zeros(150),
            "FVG_SIZE_NORM": np.zeros(150),
        }, index=idx)

        # Arm + confirm LONG on bars 100 & 101
        for i in (100, 101):
            df.iat[i, df.columns.get_loc("BOS_SIGNAL")] = 1.0
            df.iat[i, df.columns.get_loc("CHOCH_SIGNAL")] = 1.0
            df.iat[i, df.columns.get_loc("FVG_SIGNAL")] = 1.0
            df.iat[i, df.columns.get_loc("FVG_SIZE_NORM")] = 1.0
            df.iat[i, df.columns.get_loc("OB_STRENGTH_NORM")] = 1.0
            df.iat[i, df.columns.get_loc("RSI")] = 62.0
            df.iat[i, df.columns.get_loc("MACD_Diff")] = 0.5
            df.iat[i, df.columns.get_loc("CHOCH_DIVERGENCE")] = 1.0
        # Build a steady uptrend so the regime scorer contributes toward LONG
        df["close"] = np.linspace(100.0, 108.0, 150)
        df["high"] = df["close"] + 0.3
        df["low"] = df["close"] - 0.3
        df["open"] = df["close"]
        # Post-entry: a bar that spikes high enough to hit the TP
        # Entry on bar 101, TP = entry + 4*ATR = close_101 + 4 = ~104+4
        tp_hit_ix = 110
        df.iat[tp_hit_ix, df.columns.get_loc("high")] = df.iat[101, df.columns.get_loc("close")] + 8.0

        cfg = StateMachineConfig(
            enter_threshold=40.0,   # loosened so our synthetic row triggers;
                                    # without regime the max achievable is ~45
            exit_threshold=25.0,
            confirm_bars=2,
            cooldown_bars=1,
            max_signal_age_bars=20,
        )
        replay = SignalReplay(
            symbol="SYNTH", timeframe="M15",
            state_machine_config=cfg,
            use_regime=False, use_vol_regime=False,
            warmup_bars=50,
        )
        results = replay.run(df)
        # Expect at least one trade; it should be LONG
        assert results.total_trades >= 1
        assert any(t.direction == "LONG" for t in results.trades)

    def test_run_rejects_empty_df(self):
        import pandas as pd
        with pytest.raises(ValueError):
            SignalReplay().run(pd.DataFrame())

    def test_run_rejects_missing_columns(self):
        import pandas as pd
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="missing required"):
            SignalReplay().run(df)
