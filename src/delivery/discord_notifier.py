"""Discord webhook integration for Smart Sentinel AI signal delivery.

Mirrors TelegramNotifier's interface (`send_signal`, `send_exit`) but uses
Discord webhooks — no bot account needed, just a webhook URL from a server
channel's Integrations settings.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

COLOR_LONG = 0x2ECC71
COLOR_SHORT = 0xE74C3C
COLOR_EXIT_WIN = 0x27AE60
COLOR_EXIT_LOSS = 0xC0392B
COLOR_EXIT_NEUTRAL = 0x95A5A6
COLOR_INFO = 0x3498DB

VOL_EMOJI = {"low": "\U0001f7e2", "normal": "\U0001f7e1", "high": "\U0001f534"}


class DiscordNotifier:
    """Send Sentinel signals to Discord via webhook.

    The webhook URL looks like:
        https://discord.com/api/webhooks/<id>/<token>

    Usage:
        notifier = DiscordNotifier(webhook_url=os.environ["DISCORD_WEBHOOK_URL"])
        notifier.send_signal(signal, narrative_data)
    """

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        username: str = "Smart Sentinel AI",
        timeout: float = 10.0,
    ):
        self._webhook_url = webhook_url
        self._username = username
        self._timeout = timeout
        self._messages_sent = 0
        self._failures = 0

    def is_configured(self) -> bool:
        return bool(self._webhook_url)

    def send_raw(self, content: str) -> bool:
        """Send a plain text message (useful for startup pings / admin alerts)."""
        return self._post({"content": content, "username": self._username})

    def send_signal(
        self,
        signal: Any,
        narrative_data: Optional[Dict[str, Any]] = None,
        tier: str = "INSTITUTIONAL",
    ) -> bool:
        """Send a ConfluenceSignal to Discord as a rich embed."""
        if not self.is_configured():
            logger.warning("Discord webhook not configured — signal not sent")
            return False

        embed = self._build_signal_embed(signal, narrative_data, tier)
        return self._post({"username": self._username, "embeds": [embed]})

    def send_exit(
        self,
        signal: Any,
        exit_reason: str,
        exit_price: Optional[float] = None,
    ) -> bool:
        """Send an exit event (TP hit, SL hit, lifetime, manual, etc.)."""
        if not self.is_configured():
            return False

        embed = self._build_exit_embed(signal, exit_reason, exit_price)
        return self._post({"username": self._username, "embeds": [embed]})

    def get_stats(self) -> Dict[str, Any]:
        return {
            "messages_sent": self._messages_sent,
            "failures": self._failures,
            "configured": self.is_configured(),
        }

    def _build_signal_embed(
        self,
        signal: Any,
        narrative_data: Optional[Dict[str, Any]],
        tier: str,
    ) -> Dict[str, Any]:
        signal_type = getattr(signal, "signal_type", "UNKNOWN")
        if hasattr(signal_type, "value"):
            signal_type = signal_type.value
        symbol = getattr(signal, "symbol", "XAUUSD")
        score = float(getattr(signal, "confluence_score", 0) or 0)
        tier_value = getattr(signal, "tier", tier)
        if hasattr(tier_value, "value"):
            tier_value = tier_value.value

        direction_emoji = "\U0001f7e2" if signal_type == "LONG" else "\U0001f534"
        color = COLOR_LONG if signal_type == "LONG" else COLOR_SHORT

        fields = [
            {"name": "Direction", "value": f"**{signal_type}**", "inline": True},
            {"name": "Symbol", "value": symbol, "inline": True},
            {"name": "Score", "value": f"{score:.0f}/100 ({tier_value})", "inline": True},
            {"name": "Entry", "value": f"{float(getattr(signal, 'entry_price', 0)):.2f}", "inline": True},
            {"name": "Stop Loss", "value": f"{float(getattr(signal, 'stop_loss', 0)):.2f}", "inline": True},
            {"name": "Take Profit", "value": f"{float(getattr(signal, 'take_profit', 0)):.2f}", "inline": True},
            {"name": "R:R Ratio", "value": f"{float(getattr(signal, 'rr_ratio', 0)):.1f} : 1", "inline": True},
        ]

        vol_regime = getattr(signal, "vol_regime", None)
        vol_forecast_atr = getattr(signal, "vol_forecast_atr", None)
        if vol_regime is not None and vol_forecast_atr is not None:
            emoji = VOL_EMOJI.get(vol_regime, "⚪")
            vol_value = f"{emoji} {str(vol_regime).capitalize()} (ATR {float(vol_forecast_atr):.2f})"
            vol_lower = getattr(signal, "vol_confidence_lower", None)
            vol_upper = getattr(signal, "vol_confidence_upper", None)
            if vol_lower is not None and vol_upper is not None:
                vol_value += f"\n95% CI: [{float(vol_lower):.2f}, {float(vol_upper):.2f}]"
            fields.append({"name": "Volatility", "value": vol_value, "inline": False})

        description_parts = []
        if narrative_data:
            reason = narrative_data.get("validation_reason")
            if reason:
                description_parts.append(f"**Validation:** {reason}")
            narrative = narrative_data.get("full_narrative")
            if narrative:
                # Discord embed description limit is 4096; keep margin for other fields
                max_len = 2000
                if len(narrative) > max_len:
                    narrative = narrative[:max_len] + "..."
                description_parts.append(f"\n**Analysis:**\n{narrative}")

        embed: Dict[str, Any] = {
            "title": f"{direction_emoji} Smart Sentinel Signal — {signal_type} {symbol}",
            "color": color,
            "fields": fields,
            "footer": {"text": "Smart Sentinel AI — Not financial advice"},
        }

        if description_parts:
            embed["description"] = "\n".join(description_parts)

        ts = getattr(signal, "timestamp", None)
        if ts is not None:
            iso = self._as_iso8601(ts)
            if iso:
                embed["timestamp"] = iso

        return embed

    def _build_exit_embed(
        self,
        signal: Any,
        exit_reason: str,
        exit_price: Optional[float],
    ) -> Dict[str, Any]:
        symbol = getattr(signal, "symbol", "XAUUSD")
        signal_type = getattr(signal, "signal_type", "?")
        if hasattr(signal_type, "value"):
            signal_type = signal_type.value
        entry = float(getattr(signal, "entry_price", 0) or 0)

        reason_l = (exit_reason or "").lower()
        if "tp" in reason_l or "take_profit" in reason_l or "win" in reason_l:
            color = COLOR_EXIT_WIN
            title_icon = "✅"
        elif "sl" in reason_l or "stop" in reason_l or "loss" in reason_l:
            color = COLOR_EXIT_LOSS
            title_icon = "❌"
        else:
            color = COLOR_EXIT_NEUTRAL
            title_icon = "⚪"

        fields = [
            {"name": "Symbol", "value": symbol, "inline": True},
            {"name": "Direction", "value": signal_type, "inline": True},
            {"name": "Reason", "value": str(exit_reason).replace("_", " ").title(), "inline": True},
            {"name": "Entry", "value": f"{entry:.2f}", "inline": True},
        ]

        if exit_price is not None:
            exit_val = float(exit_price)
            fields.append({"name": "Exit", "value": f"{exit_val:.2f}", "inline": True})
            if entry:
                pnl_pct = ((exit_val - entry) / entry) * 100.0
                if signal_type == "SHORT":
                    pnl_pct = -pnl_pct
                fields.append({"name": "P&L", "value": f"{pnl_pct:+.2f}%", "inline": True})

        return {
            "title": f"{title_icon} Position Closed — {symbol} {signal_type}",
            "color": color,
            "fields": fields,
            "footer": {"text": "Smart Sentinel AI"},
        }

    @staticmethod
    def _as_iso8601(ts: Any) -> Optional[str]:
        try:
            if hasattr(ts, "isoformat"):
                return ts.isoformat()
            return str(ts)
        except Exception:
            return None

    def _post(self, payload: Dict[str, Any]) -> bool:
        try:
            import requests
        except ImportError:
            logger.error("requests library not installed — Discord delivery disabled")
            self._failures += 1
            return False

        try:
            response = requests.post(
                self._webhook_url,
                json=payload,
                timeout=self._timeout,
            )
            if 200 <= response.status_code < 300:
                self._messages_sent += 1
                return True
            self._failures += 1
            logger.warning(
                "Discord webhook returned %d: %s",
                response.status_code,
                response.text[:200],
            )
            return False
        except Exception as e:
            self._failures += 1
            logger.error("Discord webhook post failed: %s", e)
            return False
