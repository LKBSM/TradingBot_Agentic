"""Tests for Sprint 10: Security Hardening & Performance.

Tests cover:
  - Input validation: symbol, timeframe, pagination, score range
  - String sanitization
  - Rate limiter: allow, deny, remaining, reset, cleanup
  - Rate limiter thread safety
  - SecureConfig from_env, validation, secret masking
"""

import os
import threading
import time
from unittest.mock import patch

import pytest

from src.intelligence.security import (
    RateLimiter,
    SecureConfig,
    sanitize_string,
    validate_pagination,
    validate_score_range,
    validate_symbol,
    validate_timeframe,
)


# =============================================================================
# INPUT VALIDATION
# =============================================================================

class TestValidateSymbol:
    def test_valid_symbol(self):
        assert validate_symbol("XAUUSD") == "XAUUSD"

    def test_lowercase_normalized(self):
        assert validate_symbol("eurusd") == "EURUSD"

    def test_whitespace_stripped(self):
        assert validate_symbol("  BTCUSD  ") == "BTCUSD"

    def test_too_short_raises(self):
        with pytest.raises(ValueError, match="Invalid symbol"):
            validate_symbol("X")

    def test_too_long_raises(self):
        with pytest.raises(ValueError, match="Invalid symbol"):
            validate_symbol("AVERYLONGSYMBOL")

    def test_special_chars_raise(self):
        with pytest.raises(ValueError, match="Invalid symbol"):
            validate_symbol("XAU/USD")

    def test_sql_injection_blocked(self):
        with pytest.raises(ValueError):
            validate_symbol("'; DROP TABLE--")

    def test_numeric_allowed(self):
        assert validate_symbol("US500") == "US500"


class TestValidateTimeframe:
    def test_valid_timeframes(self):
        for tf in ("M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1"):
            assert validate_timeframe(tf) == tf

    def test_lowercase_normalized(self):
        assert validate_timeframe("m15") == "M15"

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Invalid timeframe"):
            validate_timeframe("M7")


class TestValidatePagination:
    def test_normal_values(self):
        assert validate_pagination(1, 50) == (1, 50)

    def test_negative_page_clamped(self):
        page, _ = validate_pagination(-5, 50)
        assert page == 1

    def test_oversized_page_size_clamped(self):
        _, size = validate_pagination(1, 999)
        assert size == 100  # max_page_size

    def test_zero_page_size_clamped(self):
        _, size = validate_pagination(1, 0)
        assert size == 1


class TestValidateScoreRange:
    def test_normal_range(self):
        assert validate_score_range(40.0, 80.0) == (40.0, 80.0)

    def test_inverted_swapped(self):
        assert validate_score_range(80.0, 40.0) == (40.0, 80.0)

    def test_clamped_to_bounds(self):
        assert validate_score_range(-10.0, 150.0) == (0.0, 100.0)


class TestSanitizeString:
    def test_normal_string(self):
        assert sanitize_string("hello world") == "hello world"

    def test_whitespace_stripped(self):
        assert sanitize_string("  hello  ") == "hello"

    def test_truncated(self):
        result = sanitize_string("x" * 1000, max_length=100)
        assert len(result) == 100

    def test_control_chars_removed(self):
        result = sanitize_string("hello\x00\x01world")
        assert result == "helloworld"

    def test_newlines_preserved(self):
        result = sanitize_string("line1\nline2")
        assert "\n" in result


# =============================================================================
# RATE LIMITER
# =============================================================================

class TestRateLimiter:
    def test_allow_under_limit(self):
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        for _ in range(5):
            assert limiter.allow("user1") is True

    def test_deny_over_limit(self):
        limiter = RateLimiter(max_requests=3, window_seconds=60)
        for _ in range(3):
            limiter.allow("user1")
        assert limiter.allow("user1") is False

    def test_per_key_isolation(self):
        limiter = RateLimiter(max_requests=2, window_seconds=60)
        limiter.allow("user1")
        limiter.allow("user1")
        # user1 is exhausted
        assert limiter.allow("user1") is False
        # user2 should still be allowed
        assert limiter.allow("user2") is True

    def test_remaining(self):
        limiter = RateLimiter(max_requests=10, window_seconds=60)
        limiter.allow("user1")
        limiter.allow("user1")
        assert limiter.remaining("user1") == 8

    def test_window_expiry(self):
        limiter = RateLimiter(max_requests=2, window_seconds=0.1)
        limiter.allow("user1")
        limiter.allow("user1")
        assert limiter.allow("user1") is False

        time.sleep(0.15)
        assert limiter.allow("user1") is True

    def test_reset(self):
        limiter = RateLimiter(max_requests=2, window_seconds=60)
        limiter.allow("user1")
        limiter.allow("user1")
        assert limiter.allow("user1") is False

        limiter.reset("user1")
        assert limiter.allow("user1") is True

    def test_cleanup(self):
        limiter = RateLimiter(max_requests=10, window_seconds=0.05)
        limiter.allow("user1")
        limiter.allow("user2")
        time.sleep(0.1)

        removed = limiter.cleanup()
        assert removed == 2

    def test_stats(self):
        limiter = RateLimiter(max_requests=50, window_seconds=30)
        limiter.allow("user1")
        stats = limiter.get_stats()
        assert stats["active_keys"] == 1
        assert stats["max_requests"] == 50

    def test_thread_safety(self):
        limiter = RateLimiter(max_requests=100, window_seconds=60)
        results = []

        def worker():
            allowed = limiter.allow("shared_key")
            results.append(allowed)

        threads = [threading.Thread(target=worker) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should succeed (50 < 100)
        assert all(results)
        assert limiter.remaining("shared_key") == 50


# =============================================================================
# SECURE CONFIG
# =============================================================================

class TestSecureConfig:
    def test_default_config(self):
        config = SecureConfig()
        assert config.symbols == ["XAUUSD"]
        assert config.vol_mode == "hybrid"
        assert config.api_port == 8000

    def test_from_env(self):
        env = {
            "SYMBOLS": "XAUUSD,EURUSD",
            "VOL_MODE": "har",
            "API_PORT": "9000",
            "ANTHROPIC_API_KEY": "sk-ant-test123",
            "LOG_LEVEL": "debug",
        }
        with patch.dict(os.environ, env, clear=False):
            config = SecureConfig.from_env()

        assert config.symbols == ["XAUUSD", "EURUSD"]
        assert config.vol_mode == "har"
        assert config.api_port == 9000
        assert config.log_level == "DEBUG"

    def test_invalid_vol_mode_raises(self):
        with patch.dict(os.environ, {"VOL_MODE": "invalid"}, clear=False):
            with pytest.raises(ValueError, match="Invalid VOL_MODE"):
                SecureConfig.from_env()

    def test_invalid_port_raises(self):
        with patch.dict(os.environ, {"API_PORT": "80"}, clear=False):
            with pytest.raises(ValueError, match="Invalid API_PORT"):
                SecureConfig.from_env()

    def test_secrets_masked_in_repr(self):
        config = SecureConfig(
            anthropic_api_key="sk-ant-secret",
            telegram_bot_token="123:ABC",
        )
        repr_str = repr(config)
        assert "sk-ant-secret" not in repr_str
        assert "123:ABC" not in repr_str
        assert "****" in repr_str

    def test_secrets_none_in_repr(self):
        config = SecureConfig()
        repr_str = repr(config)
        assert "None" in repr_str

    def test_validate_no_warnings(self):
        config = SecureConfig(
            anthropic_api_key="sk-ant-valid",
            symbols=["XAUUSD"],
            data_dir=".",  # Current dir exists
        )
        warnings = config.validate()
        assert len(warnings) == 0

    def test_validate_missing_key_warning(self):
        config = SecureConfig(anthropic_api_key=None)
        warnings = config.validate()
        assert any("ANTHROPIC_API_KEY" in w for w in warnings)

    def test_validate_bad_key_prefix_warning(self):
        config = SecureConfig(anthropic_api_key="bad-key-format")
        warnings = config.validate()
        assert any("sk-ant-" in w for w in warnings)

    def test_validate_missing_data_dir_warning(self):
        config = SecureConfig(data_dir="/nonexistent/path")
        warnings = config.validate()
        assert any("DATA_DIR" in w for w in warnings)

    def test_invalid_symbol_in_env(self):
        """Invalid symbol in SYMBOLS env var should raise."""
        with patch.dict(os.environ, {"SYMBOLS": "XAU/USD"}, clear=False):
            with pytest.raises(ValueError, match="Invalid symbol"):
                SecureConfig.from_env()
