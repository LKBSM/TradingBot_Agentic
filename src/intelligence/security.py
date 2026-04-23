"""Security hardening for Smart Sentinel AI.

Provides:
  - Input validation for API parameters
  - Rate limiter (in-memory, per-key sliding window)
  - Secure configuration loader (no secrets in logs)
  - Request sanitization
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# =============================================================================
# INPUT VALIDATION
# =============================================================================

# Valid symbol pattern: 2-10 uppercase alphanumeric chars
SYMBOL_PATTERN = re.compile(r"^[A-Z0-9]{2,10}$")

# Valid timeframe pattern
VALID_TIMEFRAMES: Set[str] = {"M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1"}

# Allowed sort fields
VALID_SORT_FIELDS: Set[str] = {
    "created_at", "confluence_score", "rr_ratio", "symbol",
}


def validate_symbol(symbol: str) -> str:
    """Validate and sanitize instrument symbol.

    Raises ValueError if invalid.
    """
    symbol = symbol.strip().upper()
    if not SYMBOL_PATTERN.match(symbol):
        raise ValueError(
            f"Invalid symbol: '{symbol}'. Must be 2-10 uppercase alphanumeric chars."
        )
    return symbol


def validate_timeframe(timeframe: str) -> str:
    """Validate timeframe string.

    Raises ValueError if invalid.
    """
    timeframe = timeframe.strip().upper()
    if timeframe not in VALID_TIMEFRAMES:
        raise ValueError(
            f"Invalid timeframe: '{timeframe}'. Must be one of: {sorted(VALID_TIMEFRAMES)}"
        )
    return timeframe


def validate_pagination(page: int, page_size: int, max_page_size: int = 100) -> tuple[int, int]:
    """Validate and clamp pagination parameters.

    Returns (page, page_size) clamped to valid ranges.
    """
    page = max(1, int(page))
    page_size = max(1, min(int(page_size), max_page_size))
    return page, page_size


def validate_score_range(min_score: float, max_score: float) -> tuple[float, float]:
    """Validate confluence score range.

    Returns clamped (min_score, max_score).
    """
    min_score = max(0.0, min(100.0, float(min_score)))
    max_score = max(0.0, min(100.0, float(max_score)))
    if min_score > max_score:
        min_score, max_score = max_score, min_score
    return min_score, max_score


def sanitize_string(s: str, max_length: int = 500) -> str:
    """Strip and truncate a user-provided string. Remove control chars."""
    s = s.strip()[:max_length]
    # Remove control characters (except newlines and tabs)
    s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", s)
    return s


# =============================================================================
# IN-MEMORY RATE LIMITER (Sliding Window)
# =============================================================================

class RateLimiter:
    """In-memory sliding window rate limiter.

    Thread-safe. Uses per-key deques for O(1) amortized cleanup.

    Usage:
        limiter = RateLimiter(max_requests=100, window_seconds=60)
        if not limiter.allow("user_123"):
            raise RateLimitExceeded()
    """

    def __init__(self, max_requests: int = 100, window_seconds: float = 60.0):
        self._max_requests = max_requests
        self._window = window_seconds
        self._lock = threading.Lock()
        self._buckets: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_requests + 10))

    def allow(self, key: str) -> bool:
        """Check if request from key is allowed.

        Returns True if under rate limit, False if exceeded.
        """
        now = time.time()
        cutoff = now - self._window

        with self._lock:
            bucket = self._buckets[key]

            # Evict expired entries
            while bucket and bucket[0] < cutoff:
                bucket.popleft()

            if len(bucket) >= self._max_requests:
                return False

            bucket.append(now)
            return True

    def remaining(self, key: str) -> int:
        """Return remaining requests in the current window."""
        now = time.time()
        cutoff = now - self._window

        with self._lock:
            bucket = self._buckets[key]
            while bucket and bucket[0] < cutoff:
                bucket.popleft()
            return max(0, self._max_requests - len(bucket))

    def reset(self, key: str) -> None:
        """Reset rate limit for a specific key."""
        with self._lock:
            if key in self._buckets:
                self._buckets[key].clear()

    def cleanup(self) -> int:
        """Remove empty buckets to free memory. Returns number removed."""
        now = time.time()
        cutoff = now - self._window
        removed = 0

        with self._lock:
            empty_keys = []
            for key, bucket in self._buckets.items():
                while bucket and bucket[0] < cutoff:
                    bucket.popleft()
                if not bucket:
                    empty_keys.append(key)

            for key in empty_keys:
                del self._buckets[key]
                removed += 1

        return removed

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "active_keys": len(self._buckets),
                "max_requests": self._max_requests,
                "window_seconds": self._window,
            }


# =============================================================================
# SECURE CONFIGURATION
# =============================================================================

@dataclass
class SecureConfig:
    """Loads configuration from environment variables with validation.

    Secrets are masked in string representations and logs.
    """
    # API
    api_port: int = 8000
    api_host: str = "0.0.0.0"
    log_level: str = "INFO"

    # Symbols
    symbols: List[str] = None  # type: ignore[assignment]

    # Paths
    data_dir: str = "./data"
    signal_db_path: str = "./data/signals.db"
    calendar_path: Optional[str] = None

    # Volatility
    vol_mode: str = "hybrid"

    # Secrets (masked in __repr__)
    anthropic_api_key: Optional[str] = None
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None

    # Rate limits
    rate_limit_per_minute: int = 100
    rate_limit_narrative_per_minute: int = 20

    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ["XAUUSD"]

    @classmethod
    def from_env(cls) -> "SecureConfig":
        """Load from environment variables with validation."""
        symbols_str = os.environ.get("SYMBOLS", "XAUUSD")
        symbols = [validate_symbol(s) for s in symbols_str.split(",") if s.strip()]

        vol_mode = os.environ.get("VOL_MODE", "hybrid")
        if vol_mode not in ("har", "lgbm", "hybrid"):
            raise ValueError(f"Invalid VOL_MODE: {vol_mode}. Use: har, lgbm, hybrid")

        port = int(os.environ.get("API_PORT", "8000"))
        if not (1024 <= port <= 65535):
            raise ValueError(f"Invalid API_PORT: {port}. Must be 1024-65535.")

        return cls(
            api_port=port,
            api_host=os.environ.get("API_HOST", "0.0.0.0"),
            log_level=os.environ.get("LOG_LEVEL", "INFO").upper(),
            symbols=symbols,
            data_dir=os.environ.get("DATA_DIR", "./data"),
            signal_db_path=os.environ.get("SIGNAL_DB_PATH", "./data/signals.db"),
            calendar_path=os.environ.get("CALENDAR_PATH"),
            vol_mode=vol_mode,
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
            telegram_bot_token=os.environ.get("TELEGRAM_BOT_TOKEN"),
            telegram_chat_id=os.environ.get("TELEGRAM_CHAT_ID"),
            rate_limit_per_minute=int(os.environ.get("RATE_LIMIT_PER_MIN", "100")),
            rate_limit_narrative_per_minute=int(os.environ.get("RATE_LIMIT_NARRATIVE_PER_MIN", "20")),
        )

    def __repr__(self) -> str:
        """Mask secrets in representation."""
        return (
            f"SecureConfig(symbols={self.symbols}, vol_mode={self.vol_mode}, "
            f"api_port={self.api_port}, "
            f"anthropic_key={'****' if self.anthropic_api_key else 'None'}, "
            f"telegram_token={'****' if self.telegram_bot_token else 'None'})"
        )

    def validate(self) -> List[str]:
        """Validate configuration, return list of warnings."""
        warnings = []

        if not self.anthropic_api_key:
            warnings.append("ANTHROPIC_API_KEY not set — LLM narratives will use fallback")
        elif not self.anthropic_api_key.startswith("sk-ant-"):
            warnings.append("ANTHROPIC_API_KEY doesn't start with 'sk-ant-' — may be invalid")

        if not self.symbols:
            warnings.append("No symbols configured")

        if not os.path.isdir(self.data_dir):
            warnings.append(f"DATA_DIR does not exist: {self.data_dir}")

        return warnings
