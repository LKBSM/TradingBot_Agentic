"""Tests for the backtest-only news provider."""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from src.backtest.news_replay import BacktestNewsProvider, _BlackoutEvent


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    csv = tmp_path / "cal.csv"
    pd.DataFrame([
        {"Date": "2024-01-05 13:30:00", "Currency": "USD",
         "Event": "Non-Farm Payrolls", "Impact": "HIGH",
         "Actual": "", "Forecast": "", "Previous": ""},
        {"Date": "2024-01-05 15:00:00", "Currency": "USD",
         "Event": "ISM Services PMI", "Impact": "HIGH",
         "Actual": "", "Forecast": "", "Previous": ""},
        # Medium impact — should be dropped
        {"Date": "2024-01-05 14:00:00", "Currency": "USD",
         "Event": "Factory Orders", "Impact": "MEDIUM",
         "Actual": "", "Forecast": "", "Previous": ""},
        # Wrong currency for XAUUSD — should be dropped
        {"Date": "2024-01-05 16:00:00", "Currency": "JPY",
         "Event": "BoJ Rate", "Impact": "HIGH",
         "Actual": "", "Forecast": "", "Previous": ""},
    ]).to_csv(csv, index=False)
    return csv


class TestBacktestNewsProvider:

    def test_loads_only_high_impact_and_affecting_currencies(self, sample_csv):
        p = BacktestNewsProvider.from_csv(sample_csv, symbol="XAUUSD")
        assert len(p.events) == 2  # MEDIUM + JPY filtered out
        assert all(e.currency == "USD" for e in p.events)

    def test_events_sorted_by_timestamp(self, sample_csv):
        p = BacktestNewsProvider.from_csv(sample_csv, symbol="XAUUSD")
        ts = [e.ts for e in p.events]
        assert ts == sorted(ts)

    def test_inside_blackout_returns_block(self, sample_csv):
        p = BacktestNewsProvider.from_csv(sample_csv, symbol="XAUUSD")
        # NFP at 13:30. ±30 min blackout.
        a = p.at("2024-01-05 13:25:00")
        assert a is not None
        assert a.decision.value == "block"

    def test_outside_blackout_returns_none(self, sample_csv):
        p = BacktestNewsProvider.from_csv(sample_csv, symbol="XAUUSD")
        # NFP at 13:30, next event at 15:00. 14:15 is outside both windows.
        a = p.at("2024-01-05 14:15:00")
        assert a is None

    def test_window_boundaries(self, sample_csv):
        p = BacktestNewsProvider.from_csv(
            sample_csv, symbol="XAUUSD",
            block_before_min=30, block_after_min=30,
        )
        # Exactly 30 min before NFP → blocked
        assert p.at("2024-01-05 13:00:00") is not None
        # 31 min before → not blocked
        assert p.at("2024-01-05 12:59:00") is None
        # Exactly 30 min after → blocked
        assert p.at("2024-01-05 14:00:00") is not None
        # 31 min after (and before the next event's window opens at 14:30)
        assert p.at("2024-01-05 14:01:00") is None

    def test_accepts_pandas_timestamp(self, sample_csv):
        p = BacktestNewsProvider.from_csv(sample_csv, symbol="XAUUSD")
        a = p.at(pd.Timestamp("2024-01-05 13:30:00"))
        assert a is not None

    def test_callable_interface(self, sample_csv):
        p = BacktestNewsProvider.from_csv(sample_csv, symbol="XAUUSD")
        a = p("2024-01-05 13:30:00")
        assert a is not None

    def test_missing_csv_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            BacktestNewsProvider.from_csv(tmp_path / "missing.csv")

    def test_empty_events_returns_none(self):
        p = BacktestNewsProvider(events=[])
        assert p.at("2024-01-05 13:30:00") is None

    def test_affecting_currencies_override(self, sample_csv):
        # Override so JPY counts — the JPY BoJ event should now load.
        p = BacktestNewsProvider.from_csv(
            sample_csv, symbol="XAUUSD",
            affecting_currencies=["JPY"],
        )
        assert len(p.events) == 1
        assert p.events[0].currency == "JPY"

    def test_assessment_is_blocking_for_detector(self, sample_csv):
        """The returned assessment must trigger ConfluenceDetector's
        ``_is_news_blocked`` gate — we verify the field name contract."""
        from src.intelligence.confluence_detector import ConfluenceDetector

        p = BacktestNewsProvider.from_csv(sample_csv, symbol="XAUUSD")
        a = p.at("2024-01-05 13:30:00")
        assert a is not None
        assert ConfluenceDetector._is_news_blocked(a) is True
