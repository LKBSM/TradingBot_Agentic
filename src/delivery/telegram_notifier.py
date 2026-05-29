"""Telegram Bot API integration for push delivery of algorithmic analyses.

Formats messages per tier:
  FREE: Entry/SL/TP only + upgrade prompt
  ANALYST: + Haiku validation reason
  STRATEGIST+: + Full narrative (Market Setup / Confluences / Risk)

User-facing language follows UE Directive 2024/2811 (finfluencers, March 2026):
  - "signaux"   -> "analyses algorithmiques"
  - "BUY/SELL"  -> "BULLISH SETUP / BEARISH SETUP"
  - Mandatory risk disclaimer footer (multi-language).
"""

from __future__ import annotations

import html
import logging
import random
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from src.api.disclaimers import get_footer

logger = logging.getLogger(__name__)


# DG-054 — retry / dedup configuration. Tuned for Telegram's published
# rate limits (30 messages/sec/bot global, 1 message/sec/chat) so a
# burst on a popular chat decays gracefully instead of getting the bot
# soft-banned. Numbers are conservative defaults; the constructor lets
# the operator override them.
DEFAULT_MAX_RETRIES = 4
DEFAULT_BASE_BACKOFF_S = 0.5
DEFAULT_MAX_BACKOFF_S = 30.0
DEFAULT_DEDUP_TTL_S = 24 * 3600  # one calendar day — covers the broadcast window
DEFAULT_DEDUP_MAX_ENTRIES = 50_000


# UE 2024/2811: trading-direction language must be framed as a market read,
# not an instruction. We translate the engine's internal LONG/SHORT to a
# setup label and avoid the imperative "BUY"/"SELL" outright.
DIRECTION_LABEL = {
    "LONG": "BULLISH SETUP",
    "SHORT": "BEARISH SETUP",
}


def _escape_html(s: Any) -> str:
    """Escape arbitrary value for Telegram HTML parse_mode.

    Telegram HTML mode only requires &, <, > to be escaped — much cleaner
    than MarkdownV2's 18 special chars. Eval 13 finding: legacy ``Markdown``
    (deprecated 2018) silently dropped messages whose narrative contained
    ``RSI_divergence`` (underscore unbalanced) or ``[`` from LLM output.
    """
    if s is None:
        return ""
    return html.escape(str(s), quote=False)


class _DedupStore:
    """In-memory (chat_id, signal_id) → expiry-time dedup store.

    Why in-memory: the typical broadcast cycle is < 24h and we just need
    to ensure no double-fire when the same signal is re-evaluated by the
    scheduler. For multi-worker prod the store can be swapped for SQLite
    or Redis without touching the notifier's public surface.
    """

    def __init__(self, *, ttl_seconds: float = DEFAULT_DEDUP_TTL_S,
                 max_entries: int = DEFAULT_DEDUP_MAX_ENTRIES):
        self._ttl = float(ttl_seconds)
        self._max = int(max_entries)
        self._lock = threading.Lock()
        self._seen: Dict[Tuple[str, str], float] = {}

    def __len__(self) -> int:
        with self._lock:
            return len(self._seen)

    def _sweep_locked(self, now: float) -> None:
        expired = [k for k, exp in self._seen.items() if exp <= now]
        for k in expired:
            del self._seen[k]

    def check_and_record(self, chat_id: str, signal_id: str,
                         *, now: Optional[float] = None) -> bool:
        """Return True if (chat_id, signal_id) is fresh and was just recorded.

        Returns False if the pair has been seen within ``ttl_seconds`` —
        i.e. the caller should skip the send to prevent flood.
        """
        if not chat_id or not signal_id:
            # Refuse to dedupe on missing keys to avoid silently swallowing
            # messages when the caller forgot to populate signal_id.
            return True
        key = (str(chat_id), str(signal_id))
        if now is None:
            now = time.time()
        with self._lock:
            self._sweep_locked(now)
            existing = self._seen.get(key)
            if existing is not None and existing > now:
                return False
            if len(self._seen) >= self._max:
                oldest = min(self._seen.items(), key=lambda kv: kv[1])[0]
                del self._seen[oldest]
            self._seen[key] = now + self._ttl
            return True

    def reset(self) -> None:
        with self._lock:
            self._seen.clear()


def _backoff_with_jitter(attempt: int, *, base: float, ceiling: float) -> float:
    """Exponential backoff with full jitter (AWS pattern)."""
    cap = min(ceiling, base * (2 ** attempt))
    return random.uniform(0.0, cap)


_RETRYABLE_SUBSTRINGS = (
    "timed out", "timeout", "temporarily unavailable",
    "rate limit", "too many requests", "retry after",
    "bad gateway", "service unavailable", "gateway timeout",
)


def _is_retryable(exc: BaseException) -> bool:
    """Decide whether a Telegram error should be retried.

    Retryable: explicit ``RetryAfter`` / ``TimedOut`` / ``NetworkError``
    from python-telegram-bot, or generic 429 / 5xx hints in the
    exception's string repr (covers third-party wrappers and tests
    that raise plain :class:`Exception`).
    """
    # python-telegram-bot exposes RetryAfter / TimedOut / NetworkError;
    # we duck-type by class name to avoid a hard import dep in tests.
    name = exc.__class__.__name__.lower()
    if name in {"retryafter", "timedout", "networkerror", "slowmode"}:
        return True
    msg = str(exc).lower()
    return any(sub in msg for sub in _RETRYABLE_SUBSTRINGS)


def _retry_after_seconds(exc: BaseException) -> Optional[float]:
    """Extract the retry hint Telegram sent us, if any.

    ``python-telegram-bot``'s ``RetryAfter`` exception carries a
    ``retry_after`` attribute (seconds). When present, we honour it
    rather than using exponential backoff so we don't add unnecessary
    delay.
    """
    hint = getattr(exc, "retry_after", None)
    if hint is None:
        return None
    try:
        return float(hint)
    except (TypeError, ValueError):
        return None


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
        lang_store: Any = None,
        default_lang: str = "en",
        *,
        dedup_store: Optional[_DedupStore] = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        base_backoff_s: float = DEFAULT_BASE_BACKOFF_S,
        max_backoff_s: float = DEFAULT_MAX_BACKOFF_S,
        sleep_fn: Any = None,
    ):
        """
        Args:
            bot_token: Telegram bot token (optional in tests).
            default_chat_id: Fallback chat id when ``send_signal`` is called
                without one.
            lang_store: Optional :class:`TelegramLangStore` (or any object with
                a ``get(chat_id) -> Optional[str]`` method) used to look up
                each recipient's language. Populated by the bot's ``/start``
                handler from the Telegram ``User.language_code`` field.
            default_lang: Disclaimer language used when the resolver returns
                ``None`` (typical for chats that never ran ``/start``).
            dedup_store: DG-054 — when provided, (chat_id, signal_id) pairs
                already delivered within the store's TTL window are skipped.
                Default = a fresh in-memory store with 24h TTL.
            max_retries / base_backoff_s / max_backoff_s: DG-054 exponential
                backoff config for transient errors (429 + 5xx). Use
                ``max_retries=0`` to disable.
            sleep_fn: Test seam — defaults to ``time.sleep``. Replace in
                unit tests so backoff doesn't actually block.
        """
        self._bot_token = bot_token
        self._default_chat_id = default_chat_id
        self._lang_store = lang_store
        self._default_lang = default_lang
        self._bot: Any = None
        self._messages_sent = 0
        self._dedup_store = dedup_store if dedup_store is not None else _DedupStore()
        self._max_retries = int(max_retries)
        self._base_backoff = float(base_backoff_s)
        self._max_backoff = float(max_backoff_s)
        self._sleep_fn = sleep_fn if sleep_fn is not None else time.sleep
        # DG-054 telemetry — surfaced by ``get_stats()`` so operators
        # see retries + dedup hits even when nothing fails outright.
        self._retries_total = 0
        self._dedup_skips = 0
        self._send_failures = 0

        if bot_token:
            self._init_bot()

    def _resolve_lang(self, chat_id: Optional[str], explicit: Optional[str]) -> str:
        """Pick a language for a chat: explicit > store lookup > default."""
        if explicit:
            return explicit
        if self._lang_store is not None and chat_id is not None:
            looked_up = self._lang_store.get(chat_id)
            if looked_up:
                return looked_up
        return self._default_lang

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
        lang: str = "en",
    ) -> str:
        """
        Format an algorithmic analysis into a Telegram-friendly message.

        Args:
            signal: ConfluenceSignal object.
            narrative_data: Dict from SignalNarrative.to_dict().
            tier: User tier (FREE, ANALYST, STRATEGIST, INSTITUTIONAL).
            lang: ISO-639-1 code for the disclaimer footer (fr/en/de/es).

        Returns:
            Formatted message string using Telegram HTML parse_mode (safer
            than legacy Markdown — only ``& < >`` need escaping, while LLM
            narratives commonly include ``_ * [ ]`` that broke the legacy
            Markdown parser silently).
        """
        signal_type = getattr(signal, "signal_type", "UNKNOWN")
        if hasattr(signal_type, "value"):
            signal_type = signal_type.value

        emoji = "\U0001f7e2" if signal_type == "LONG" else "\U0001f534"  # Green/Red circle
        direction_label = DIRECTION_LABEL.get(signal_type, signal_type)
        tier_value = getattr(signal, "tier", "UNKNOWN")
        if hasattr(tier_value, "value"):
            tier_value = tier_value.value

        symbol = _escape_html(getattr(signal, "symbol", "XAUUSD"))
        lines = [
            f"{emoji} <b>Smart Sentinel — Algorithmic Analysis</b>",
            "",
            f"<b>Setup:</b> {_escape_html(direction_label)}",
            f"<b>Symbol:</b> {symbol}",
            f"<b>Score:</b> {getattr(signal, 'confluence_score', 0):.0f}/100 ({_escape_html(tier_value)})",
            "",
            f"<b>Entry zone:</b> {getattr(signal, 'entry_price', 0):.2f}",
            f"<b>Invalidation:</b> {getattr(signal, 'stop_loss', 0):.2f}",
            f"<b>Target:</b> {getattr(signal, 'take_profit', 0):.2f}",
            f"<b>R:R Ratio:</b> {getattr(signal, 'rr_ratio', 0):.1f}:1",
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
                f"{regime_emoji} <b>Volatility:</b> {_escape_html(vol_regime.capitalize())} "
                f"(ATR forecast: {vol_forecast_atr:.2f})"
            )
            # STRATEGIST+ gets confidence interval
            if tier in ("STRATEGIST", "INSTITUTIONAL"):
                vol_lower = getattr(signal, "vol_confidence_lower", None)
                vol_upper = getattr(signal, "vol_confidence_upper", None)
                if vol_lower is not None and vol_upper is not None:
                    lines.append(
                        f"  <i>95% CI: [{vol_lower:.2f} — {vol_upper:.2f}]</i>"
                    )

        # ANALYST tier: add validation reason
        if tier in ("ANALYST", "STRATEGIST", "INSTITUTIONAL") and narrative_data:
            reason = narrative_data.get("validation_reason", "")
            if reason:
                lines.append("")
                lines.append(f"\U0001f9e0 <b>Validation:</b> {_escape_html(reason)}")

        # STRATEGIST/INSTITUTIONAL: add full narrative
        if tier in ("STRATEGIST", "INSTITUTIONAL") and narrative_data:
            narrative = narrative_data.get("full_narrative", "")
            if narrative:
                # Truncate to Telegram's 4096 char limit
                max_narrative = 2500
                if len(narrative) > max_narrative:
                    narrative = narrative[:max_narrative] + "..."
                lines.append("")
                lines.append("\U0001f4ca <b>Analysis:</b>")
                lines.append(_escape_html(narrative))

        # FREE tier: upgrade prompt
        if tier == "FREE":
            lines.append("")
            lines.append("\U0001f512 <i>Upgrade to Analyst for AI validation</i>")

        lines.append("")
        lines.append(f"<i>{_escape_html(get_footer(lang))}</i>")

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
        lang: Optional[str] = None,
    ) -> bool:
        """
        Send an algorithmic analysis to a Telegram chat.

        Args:
            signal: ConfluenceSignal object.
            narrative_data: Dict from SignalNarrative.to_dict().
            chat_id: Target Telegram chat ID. Uses default if None.
            tier: User tier for message formatting.
            lang: ISO-639-1 code for the disclaimer footer. When omitted,
                the notifier looks up the chat's language via the configured
                ``lang_store`` (Telegram ``User.language_code``) and falls
                back to ``default_lang``.

        Returns:
            True if sent successfully.
        """
        target = chat_id or self._default_chat_id
        if not target:
            logger.warning("No chat_id provided and no default set")
            return False

        # DG-054 dedup gate. ``ConfluenceSignal`` carries ``signal_id`` as
        # canonical key; we fall back to ``getattr`` to stay compatible
        # with stubs in tests. A missing id disables dedup for this call.
        signal_id = getattr(signal, "signal_id", None) or (
            signal.get("signal_id") if isinstance(signal, dict) else None
        )
        if signal_id and not self._dedup_store.check_and_record(str(target), str(signal_id)):
            self._dedup_skips += 1
            logger.info(
                "Telegram dedup: skipping (chat_id=%s, signal_id=%s) — already delivered",
                target, signal_id,
            )
            return False

        resolved_lang = self._resolve_lang(target, lang)
        message = self.format_signal_message(signal, narrative_data, tier, lang=resolved_lang)

        if self._bot is None:
            logger.warning("Telegram bot not initialized — message not sent")
            return False

        return self._send_with_retry(target, message)

    def _send_with_retry(self, target: str, message: str) -> bool:
        """Send with exp-backoff retry. Returns True on first success.

        Retryable errors:
          - explicit RetryAfter from python-telegram-bot (respects ``.retry_after``);
          - TimedOut, NetworkError (transient transport faults);
          - generic 429 / 5xx surfaced via ``message`` substring.
        """
        attempt = 0
        last_exc: Optional[BaseException] = None
        while True:
            try:
                self._bot.send_message(
                    chat_id=target,
                    text=message,
                    parse_mode="HTML",
                    disable_web_page_preview=True,
                )
                self._messages_sent += 1
                logger.info("Signal sent to Telegram chat %s (attempts=%d)", target, attempt + 1)
                return True
            except Exception as exc:  # noqa: BLE001 — Telegram error tree is wide
                last_exc = exc
                if attempt >= self._max_retries or not _is_retryable(exc):
                    self._send_failures += 1
                    logger.error(
                        "Telegram send failed after %d attempt(s): %s", attempt + 1, exc,
                    )
                    return False
                wait_s = _retry_after_seconds(exc)
                if wait_s is None:
                    wait_s = _backoff_with_jitter(
                        attempt, base=self._base_backoff, ceiling=self._max_backoff,
                    )
                logger.warning(
                    "Telegram transient error (attempt=%d, sleeping %.2fs): %s",
                    attempt + 1, wait_s, exc,
                )
                self._retries_total += 1
                self._sleep_fn(wait_s)
                attempt += 1
        # Unreachable, but mypy/linters appreciate it.
        return False  # pragma: no cover

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
            if self.send_signal(
                signal,
                narrative_data,
                r["chat_id"],
                r.get("tier", "FREE"),
                lang=r.get("lang"),  # None -> resolve via lang_store
            ):
                sent += 1
        return sent

    def get_stats(self) -> Dict[str, Any]:
        return {
            "messages_sent": self._messages_sent,
            "bot_initialized": self._bot is not None,
            "retries_total": self._retries_total,
            "dedup_skips": self._dedup_skips,
            "send_failures": self._send_failures,
            "dedup_store_size": len(self._dedup_store),
        }
