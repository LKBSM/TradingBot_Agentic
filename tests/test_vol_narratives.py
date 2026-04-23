"""Tests for Sprint 3: Volatility Context in Narratives.

Tests cover:
  - LLM signal CSV includes vol fields when present
  - LLM signal CSV omits vol fields when absent (backward compat)
  - System prompt mentions volatility regime
  - Telegram message includes volatility line
  - Telegram STRATEGIST gets confidence interval
  - Telegram FREE does not get confidence interval
"""

from dataclasses import dataclass, field
from typing import List, Optional
from unittest.mock import MagicMock

import pytest

from src.intelligence.llm_narrative_engine import (
    LLMNarrativeEngine,
    SMC_SYSTEM_PROMPT,
)
from src.delivery.telegram_notifier import TelegramNotifier


# =============================================================================
# MOCK SIGNAL
# =============================================================================

@dataclass
class MockComponent:
    name: str = "bos"
    weighted_score: float = 12.0
    weight: float = 15.0


@dataclass
class MockSignal:
    """Mimics ConfluenceSignal with vol fields."""
    symbol: str = "XAUUSD"
    signal_type: str = "LONG"
    confluence_score: float = 72.5
    tier: str = "STRONG"
    entry_price: float = 2000.0
    stop_loss: float = 1992.0
    take_profit: float = 2016.0
    rr_ratio: float = 2.0
    atr: float = 3.0
    components: List[MockComponent] = field(
        default_factory=lambda: [MockComponent()]
    )
    # Vol fields
    vol_forecast_atr: Optional[float] = None
    vol_regime: Optional[str] = None
    vol_confidence_lower: Optional[float] = None
    vol_confidence_upper: Optional[float] = None


# =============================================================================
# LLM NARRATIVE ENGINE — SIGNAL CSV TESTS
# =============================================================================

class TestSignalCSVVolFields:
    def test_csv_includes_vol_fields_when_present(self):
        """Vol fields should appear in CSV when signal has them."""
        signal = MockSignal(
            vol_forecast_atr=4.5,
            vol_regime="high",
            vol_confidence_lower=2.7,
            vol_confidence_upper=6.3,
        )
        csv = LLMNarrativeEngine._signal_to_csv(signal)

        assert "vol_regime=high" in csv
        assert "vol_forecast=4.50" in csv
        assert "vol_naive=3.00" in csv
        assert "vol_ci=[2.70,6.30]" in csv

    def test_csv_omits_vol_fields_when_absent(self):
        """Backward compat: no vol fields in CSV when signal lacks them."""
        signal = MockSignal()
        csv = LLMNarrativeEngine._signal_to_csv(signal)

        assert "vol_regime" not in csv
        assert "vol_forecast" not in csv

    def test_csv_omits_ci_when_bounds_missing(self):
        """Vol regime/forecast present but no CI bounds."""
        signal = MockSignal(
            vol_forecast_atr=3.5,
            vol_regime="normal",
        )
        csv = LLMNarrativeEngine._signal_to_csv(signal)

        assert "vol_regime=normal" in csv
        assert "vol_forecast=3.50" in csv
        assert "vol_ci" not in csv

    def test_csv_base_fields_unchanged(self):
        """Base signal fields should remain in CSV regardless of vol."""
        signal = MockSignal(vol_forecast_atr=4.0, vol_regime="low")
        csv = LLMNarrativeEngine._signal_to_csv(signal)

        assert "sym=XAUUSD" in csv
        assert "dir=LONG" in csv
        assert "score=72.5" in csv
        assert "entry=2000.00" in csv
        assert "sl=1992.00" in csv
        assert "tp=2016.00" in csv
        assert "rr=2.00" in csv
        assert "atr=3.00" in csv
        assert "components=" in csv

    def test_csv_vol_regime_values(self):
        """All regime values should serialize correctly."""
        for regime in ("low", "normal", "high"):
            signal = MockSignal(vol_forecast_atr=3.0, vol_regime=regime)
            csv = LLMNarrativeEngine._signal_to_csv(signal)
            assert f"vol_regime={regime}" in csv


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

class TestSystemPromptVolContext:
    def test_system_prompt_mentions_vol_regime(self):
        """System prompt should guide LLM on volatility regime handling."""
        assert "vol_regime=low" in SMC_SYSTEM_PROMPT
        assert "vol_regime=normal" in SMC_SYSTEM_PROMPT
        assert "vol_regime=high" in SMC_SYSTEM_PROMPT

    def test_system_prompt_mentions_wider_stops(self):
        """High-vol regime guidance should mention wider stops."""
        assert "Wider stops" in SMC_SYSTEM_PROMPT or "wider stops" in SMC_SYSTEM_PROMPT

    def test_system_prompt_mentions_forecast_vs_naive(self):
        """Prompt should mention forecast vs naive ATR comparison."""
        assert "vol_forecast" in SMC_SYSTEM_PROMPT
        assert "vol_naive" in SMC_SYSTEM_PROMPT


# =============================================================================
# TELEGRAM NOTIFIER — VOLATILITY LINE
# =============================================================================

class TestTelegramVolLine:
    def test_message_includes_vol_line(self):
        """Telegram message should show volatility regime and forecast."""
        signal = MockSignal(
            vol_forecast_atr=4.5,
            vol_regime="high",
        )
        msg = TelegramNotifier.format_signal_message(signal, tier="FREE")

        assert "Volatility:" in msg
        assert "High" in msg
        assert "4.50" in msg

    def test_message_no_vol_line_when_absent(self):
        """No volatility line when signal has no vol data."""
        signal = MockSignal()
        msg = TelegramNotifier.format_signal_message(signal, tier="FREE")
        assert "Volatility:" not in msg

    def test_vol_regime_emoji_low(self):
        signal = MockSignal(vol_forecast_atr=2.0, vol_regime="low")
        msg = TelegramNotifier.format_signal_message(signal, tier="FREE")
        assert "\U0001f7e2" in msg  # Green circle

    def test_vol_regime_emoji_normal(self):
        signal = MockSignal(vol_forecast_atr=3.0, vol_regime="normal")
        msg = TelegramNotifier.format_signal_message(signal, tier="FREE")
        assert "\U0001f7e1" in msg  # Yellow circle

    def test_vol_regime_emoji_high(self):
        signal = MockSignal(vol_forecast_atr=5.0, vol_regime="high")
        msg = TelegramNotifier.format_signal_message(signal, tier="FREE")
        assert "\U0001f534" in msg  # Red circle (vol) — note: signal also has red for SHORT

    def test_strategist_gets_confidence_interval(self):
        """STRATEGIST tier should see 95% CI."""
        signal = MockSignal(
            vol_forecast_atr=4.0,
            vol_regime="normal",
            vol_confidence_lower=2.4,
            vol_confidence_upper=5.6,
        )
        msg = TelegramNotifier.format_signal_message(signal, tier="STRATEGIST")

        assert "95% CI" in msg
        assert "2.40" in msg
        assert "5.60" in msg

    def test_institutional_gets_confidence_interval(self):
        """INSTITUTIONAL tier should also see 95% CI."""
        signal = MockSignal(
            vol_forecast_atr=4.0,
            vol_regime="normal",
            vol_confidence_lower=2.4,
            vol_confidence_upper=5.6,
        )
        msg = TelegramNotifier.format_signal_message(signal, tier="INSTITUTIONAL")
        assert "95% CI" in msg

    def test_free_no_confidence_interval(self):
        """FREE tier should NOT see CI even if data is present."""
        signal = MockSignal(
            vol_forecast_atr=4.0,
            vol_regime="normal",
            vol_confidence_lower=2.4,
            vol_confidence_upper=5.6,
        )
        msg = TelegramNotifier.format_signal_message(signal, tier="FREE")
        assert "95% CI" not in msg

    def test_analyst_no_confidence_interval(self):
        """ANALYST tier should NOT see CI."""
        signal = MockSignal(
            vol_forecast_atr=4.0,
            vol_regime="normal",
            vol_confidence_lower=2.4,
            vol_confidence_upper=5.6,
        )
        msg = TelegramNotifier.format_signal_message(signal, tier="ANALYST")
        assert "95% CI" not in msg

    def test_vol_line_appears_before_validation(self):
        """Vol line should appear between R:R and validation sections."""
        signal = MockSignal(
            vol_forecast_atr=4.0,
            vol_regime="high",
        )
        narrative_data = {"validation_reason": "Confirmed by price action"}
        msg = TelegramNotifier.format_signal_message(
            signal, narrative_data=narrative_data, tier="ANALYST"
        )

        vol_pos = msg.index("Volatility:")
        val_pos = msg.index("Validation:")
        assert vol_pos < val_pos, "Vol line should appear before validation"

    def test_full_message_structure_with_vol(self):
        """Full message with all sections including vol."""
        signal = MockSignal(
            vol_forecast_atr=5.0,
            vol_regime="high",
            vol_confidence_lower=3.0,
            vol_confidence_upper=7.0,
        )
        narrative_data = {
            "validation_reason": "Strong BOS + FVG alignment",
            "full_narrative": "Market shows institutional accumulation...",
        }
        msg = TelegramNotifier.format_signal_message(
            signal, narrative_data=narrative_data, tier="STRATEGIST"
        )

        # Verify all sections present
        assert "Smart Sentinel Signal" in msg
        assert "Direction:" in msg
        assert "Entry:" in msg
        assert "Volatility:" in msg
        assert "High" in msg
        assert "95% CI" in msg
        assert "Validation:" in msg
        assert "Analysis:" in msg
        assert "Not financial advice" in msg
