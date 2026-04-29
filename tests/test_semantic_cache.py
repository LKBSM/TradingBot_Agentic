"""Tests for SemanticCache — Sprint 3 of Smart Sentinel AI."""

import os
import time
import pytest
from dataclasses import dataclass, field
from typing import List

from src.intelligence.semantic_cache import SemanticCache


# ============================================================================
# MOCK SIGNAL
# ============================================================================

@dataclass
class MockComponent:
    name: str
    weighted_score: float


@dataclass
class MockSignal:
    symbol: str = "XAUUSD"
    bar_timestamp: str = "2025-06-15T14:30:00"
    components: List[MockComponent] = field(default_factory=lambda: [
        MockComponent("BOS", 15.0),
        MockComponent("FVG", 15.0),
        MockComponent("Regime", 20.0),
    ])


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test_cache.db")


@pytest.fixture
def cache(db_path):
    return SemanticCache(db_path=db_path, ttl_seconds=3600)


@pytest.fixture
def short_ttl_cache(db_path):
    return SemanticCache(db_path=db_path, ttl_seconds=1)


@pytest.fixture
def signal():
    return MockSignal()


@pytest.fixture
def narrative_data():
    return {
        "tier": "NARRATOR",
        "is_valid": True,
        "validation_reason": "Strong setup",
        "full_narrative": "Gold is testing resistance...",
        "cost_usd": 0.015,
    }


# ============================================================================
# TESTS: CACHE HIT / MISS
# ============================================================================

class TestCacheHitMiss:
    def test_miss_on_empty_cache(self, cache, signal):
        key = SemanticCache.generate_cache_key(signal)
        result = cache.get(key)
        assert result is None

    def test_hit_after_put(self, cache, signal, narrative_data):
        key = SemanticCache.generate_cache_key(signal)
        cache.put(key, narrative_data)
        result = cache.get(key)
        assert result is not None
        assert result["is_valid"] is True
        assert result["full_narrative"] == "Gold is testing resistance..."

    def test_different_bar_same_signal_is_cache_HIT(self, cache, narrative_data):
        """Bar timestamp is intentionally NOT part of the key — two bars with the
        same symbol/dir/tier/components must collide, otherwise hit rate stays at 0."""
        signal_1 = MockSignal(bar_timestamp="2025-06-15T14:30:00")
        signal_2 = MockSignal(bar_timestamp="2025-06-15T14:45:00")

        key_1 = SemanticCache.generate_cache_key(signal_1)
        key_2 = SemanticCache.generate_cache_key(signal_2)

        assert key_1 == key_2  # Same signal shape → same key, regardless of bar

        cache.put(key_1, narrative_data)
        assert cache.get(key_2) is not None  # Hit on the "next" bar

    def test_same_signal_same_key(self, signal):
        key_1 = SemanticCache.generate_cache_key(signal)
        key_2 = SemanticCache.generate_cache_key(signal)
        assert key_1 == key_2


# ============================================================================
# TESTS: TTL EXPIRATION
# ============================================================================

class TestTTLExpiration:
    def test_expired_entry_returns_none(self, short_ttl_cache, signal, narrative_data):
        key = SemanticCache.generate_cache_key(signal)
        short_ttl_cache.put(key, narrative_data)

        # Should be available immediately
        assert short_ttl_cache.get(key) is not None

        # Wait for expiration
        time.sleep(1.5)
        assert short_ttl_cache.get(key) is None

    def test_cleanup_removes_expired(self, short_ttl_cache, signal, narrative_data):
        key = SemanticCache.generate_cache_key(signal)
        short_ttl_cache.put(key, narrative_data)

        assert short_ttl_cache.size() == 1
        time.sleep(1.5)

        deleted = short_ttl_cache.cleanup_expired()
        assert deleted == 1
        assert short_ttl_cache.size() == 0


# ============================================================================
# TESTS: HIT COUNT TRACKING
# ============================================================================

class TestHitTracking:
    def test_hit_count_increments(self, cache, signal, narrative_data):
        key = SemanticCache.generate_cache_key(signal)
        cache.put(key, narrative_data)

        cache.get(key)
        cache.get(key)
        cache.get(key)

        stats = cache.get_stats()
        assert stats["hits"] == 3
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 1.0

    def test_miss_tracked(self, cache, signal):
        key = SemanticCache.generate_cache_key(signal)
        cache.get(key)  # Miss

        stats = cache.get_stats()
        assert stats["misses"] == 1
        assert stats["hits"] == 0
        assert stats["hit_rate"] == 0.0

    def test_mixed_stats(self, cache, signal, narrative_data):
        key = SemanticCache.generate_cache_key(signal)
        cache.get(key)  # Miss
        cache.put(key, narrative_data)
        cache.get(key)  # Hit
        cache.get(key)  # Hit

        stats = cache.get_stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == pytest.approx(0.667, abs=0.01)


# ============================================================================
# TESTS: CACHE KEY DETERMINISM
# ============================================================================

class TestCacheKeyDeterminism:
    def test_key_is_16_hex_chars(self, signal):
        key = SemanticCache.generate_cache_key(signal)
        assert len(key) == 16
        assert all(c in "0123456789abcdef" for c in key)

    def test_different_symbols_different_keys(self):
        sig_gold = MockSignal(symbol="XAUUSD")
        sig_silver = MockSignal(symbol="XAGUSD")
        assert SemanticCache.generate_cache_key(sig_gold) != SemanticCache.generate_cache_key(sig_silver)

    def test_different_scores_different_keys(self):
        # 20 and 10 are one full 10-pt bucket apart → different keys.
        sig_1 = MockSignal(components=[MockComponent("BOS", 20.0)])
        sig_2 = MockSignal(components=[MockComponent("BOS", 10.0)])
        assert SemanticCache.generate_cache_key(sig_1) != SemanticCache.generate_cache_key(sig_2)

    def test_close_scores_collide_via_bucketing(self):
        """Scores within the 10-pt bucket window should collide (fuzzy match)."""
        sig_1 = MockSignal(components=[MockComponent("BOS", 11.0)])
        sig_2 = MockSignal(components=[MockComponent("BOS", 13.0)])
        # Both round to 10 with step=10 → identical key
        assert SemanticCache.generate_cache_key(sig_1) == SemanticCache.generate_cache_key(sig_2)

    def test_component_order_does_not_matter(self):
        """Components are sorted by name before hashing — order is irrelevant."""
        sig_1 = MockSignal(components=[
            MockComponent("BOS", 15.0),
            MockComponent("FVG", 15.0),
        ])
        sig_2 = MockSignal(components=[
            MockComponent("FVG", 15.0),
            MockComponent("BOS", 15.0),
        ])
        assert SemanticCache.generate_cache_key(sig_1) == SemanticCache.generate_cache_key(sig_2)

    def test_direction_changes_key(self):
        @dataclass
        class _Side:
            value: str = "LONG"

        @dataclass
        class _SignalWithDir:
            symbol: str = "XAUUSD"
            signal_type: _Side = field(default_factory=_Side)
            tier: str = "STANDARD"
            components: List[MockComponent] = field(default_factory=lambda: [
                MockComponent("BOS", 15.0),
            ])

        long_sig = _SignalWithDir()
        short_sig = _SignalWithDir(signal_type=_Side(value="SHORT"))
        assert (
            SemanticCache.generate_cache_key(long_sig)
            != SemanticCache.generate_cache_key(short_sig)
        )

    def test_tier_changes_key(self):
        sig_premium = MockSignal()
        sig_premium.tier = "PREMIUM"
        sig_basic = MockSignal()
        sig_basic.tier = "BASIC"
        assert (
            SemanticCache.generate_cache_key(sig_premium)
            != SemanticCache.generate_cache_key(sig_basic)
        )


# ============================================================================
# TESTS: SIZE
# ============================================================================

class TestSize:
    def test_empty_cache_size_zero(self, cache):
        assert cache.size() == 0

    def test_size_after_puts(self, cache, narrative_data):
        cache.put("key1", narrative_data)
        cache.put("key2", narrative_data)
        assert cache.size() == 2

    def test_put_overwrites_same_key(self, cache, narrative_data):
        cache.put("key1", narrative_data)
        cache.put("key1", {"updated": True})
        assert cache.size() == 1
        result = cache.get("key1")
        assert result["updated"] is True
