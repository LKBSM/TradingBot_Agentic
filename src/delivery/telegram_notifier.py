"""Telegram Bot API integration for push signal delivery.

Formats messages per tier:
  FREE: Entry/SL/TP only + upgrade prompt
  ANALYST: + Haiku validation reason
  STRATEGIST+: + Full narrative (Market Setup / Confluences / Risk)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """
    Sends trading signals to Telegram users via Bot API.

    Uses python-telegram-bot for async delivery, but exposes
    a synchronous ``send_signal()`` for pipeline integration.
    """

    def __init__(
        self,
        bot_token: Optional[str] = None,
        default_chat_id: Optional[str] = None,
    ):
        self._bot_token = bot_token
        self._default_chat_id = default_chat_id
        self._bot: Any = None
        self._messages_sent = 0

        if bot_token:
            self._init_bot()

    def _init_bot(self) -> None:
        """Initialize Telegram bot client."""
        try:
            import telegram
            self._bot = telegram.Bot(token=self._bot_token)
            logger.info("Telegram bot initialized")
        except ImportError:
            logger.warning(
                "python-telegram-bot not installed. "
                "Install with: pip install python-telegram-bot"
            )
            self._bot = None

    # ------------------------------------------------------------------ #
    # MESSAGE FORMATTING
    # ------------------------------------------------------------------ #

    @staticmethod
    def format_signal_message(
        signal: Any,
        narrative_data: Optional[Dict[str, Any]] = None,
        tier: str = "FREE",
    ) -> str:
        """
        Format a signal into a Telegram-friendly message.

        Args:
            signal: ConfluenceSignal object.
            narrative_data: Dict from SignalNarrative.to_dict().
            tier: User tier (FREE, ANALYST, STRATEGIST, INSTITUTIONAL).

        Returns:
            Formatted message string with Telegram MarkdownV2.
        """
        signal_type = getattr(signal, "signal_type", "UNKNOWN")
        if hasattr(signal_type, "value"):
            signal_type = signal_type.value

        emoji = "\U0001f7e2" if signal_type == "LONG" else "\U0001f534"  # Green/Red circle
        tier_value = getattr(signal, "tier", "UNKNOWN")
        if hasattr(tier_value, "value"):
            tier_value = tier_value.value

        lines = [
            f"{emoji} *Smart Sentinel Signal*",
            "",
            f"*Direction:* {signal_type}",
            f"*Symbol:* {getattr(signal, 'symbol', 'XAUUSD')}",
            f"*Score:* {getattr(signal, 'confluence_score', 0):.0f}/100 ({tier_value})",
            "",
            f"*Entry:* {getattr(signal, 'entry_price', 0):.2f}",
            f"*Stop Loss:* {getattr(signal, 'stop_loss', 0):.2f}",
            f"*Take Profit:* {getattr(signal, 'take_profit', 0):.2f}",
            f"*R:R Ratio:* {getattr(signal, 'rr_ratio', 0):.1f}:1",
        ]

        # Volatility context (all tiers if available)
        vol_regime = getattr(signal, "vol_regime", None)
        vol_forecast_atr = getattr(signal, "vol_forecast_atr", None)
        if vol_regime is not None and vol_forecast_atr is not None:
            regime_emoji = {
                "low": "\U0001f7e2",     # Green circle
                "normal": "\U0001f7e1",  # Yellow circle
                "high": "\U0001f534",    # Red circle
            }.get(vol_regime, "\u26aa")  # White circle fallback
            lines.append("")
            lines.append(
                f"{regime_emoji} *Volatility:* {vol_regime.capitalize()} "
                f"(ATR forecast: {vol_forecast_atr:.2f})"
            )
            # STRATEGIST+ gets confidence interval
            if tier in ("STRATEGIST", "INSTITUTIONAL"):
                vol_lower = getattr(signal, "vol_confidence_lower", None)
                vol_upper = getattr(signal, "vol_confidence_upper", None)
                if vol_lower is not None and vol_upper is not None:
                    lines.append(
                        f"  _95% CI: [{vol_lower:.2f} — {vol_upper:.2f}]_"
                    )

        # ANALYST tier: add validation reason
        if tier in ("ANALYST", "STRATEGIST", "INSTITUTIONAL") and narrative_data:
            reason = narrative_data.get("validation_reason", "")
            if reason:
                lines.append("")
                lines.append(f"\U0001f9e0 *Validation:* {reason}")

        # STRATEGIST/INSTITUTIONAL: add full narrative
        if tier in ("STRATEGIST", "INSTITUTIONAL") and narrative_data:
            narrative = narrative_data.get("full_narrative", "")
            if narrative:
                # Truncate to Telegram's 4096 char limit
                max_narrative = 2500
                if len(narrative) > max_narrative:
                    narrative = narrative[:max_narrative] + "..."
                lines.append("")
                lines.append("\U0001f4ca *Analysis:*")
                lines.append(narrative)

        # FREE tier: upgrade prompt
        if tier == "FREE":
            lines.append("")
            lines.append("\U0001f512 _Upgrade to Analyst ($49/mo) for AI validation_")

        lines.append("")
        lines.append("_Smart Sentinel AI — Not financial advice_")

        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # SEND
    # ------------------------------------------------------------------ #

    def send_signal(
        self,
        signal: Any,
        narrative_data: Optional[Dict[str, Any]] = None,
        chat_id: Optional[str] = None,
        tier: str = "FREE",
    ) -> bool:
        """
        Send a signal to a Telegram chat.

        Args:
            signal: ConfluenceSignal object.
            narrative_data: Dict from SignalNarrative.to_dict().
            chat_id: Target Telegram chat ID. Uses default if None.
            tier: User tier for message formatting.

        Returns:
            True if sent successfully.
        """
        target = chat_id or self._default_chat_id
        if not target:
            logger.warning("No chat_id provided and no default set")
            return False

        message = self.format_signal_message(signal, narrative_data, tier)

        if self._bot is None:
            logger.warning("Telegram bot not initialized — message not sent")
            return False

        try:
            self._bot.send_message(
                chat_id=target,
                text=message,
                parse_mode="Markdown",
            )
            self._messages_sent += 1
            logger.info("Signal sent to Telegram chat %s", target)
            return True
        except Exception as e:
            logger.error("Telegram send failed: %s", e)
            return False

    def send_to_multiple(
        self,
        signal: Any,
        narrative_data: Optional[Dict[str, Any]],
        recipients: List[Dict[str, str]],
    ) -> int:
        """
        Send signal to multiple users with tier-appropriate formatting.

        Args:
            signal: ConfluenceSignal.
            narrative_data: Narrative dict.
            recipients: List of {"chat_id": str, "tier": str}.

        Returns:
            Number of successful sends.
        """
        sent = 0
        for r in recipients:
            if self.send_signal(signal, narrative_data, r["chat_id"], r.get("tier", "FREE")):
                sent += 1
        return sent

    def get_stats(self) -> Dict[str, Any]:
        return {
            "messages_sent": self._messages_sent,
            "bot_initialized": self._bot is not None,
        }
