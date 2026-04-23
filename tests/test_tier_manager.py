"""Tests for UserTierManager — Sprint 5 of Smart Sentinel AI."""

import time
import pytest

from src.api.tier_manager import (
    UserTierManager,
    UserTier,
    TIER_CONFIG,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test_users.db")


@pytest.fixture
def manager(db_path):
    return UserTierManager(db_path=db_path)


# ============================================================================
# TESTS: TIER CONFIG
# ============================================================================

class TestTierConfig:
    def test_free_tier_defaults(self):
        cfg = TIER_CONFIG[UserTier.FREE]
        assert cfg["price_usd"] == 0
        assert cfg["api_calls_per_day"] == 10
        assert cfg["narrative_depth"] == "VISUAL"
        assert cfg["telegram"] is False

    def test_analyst_tier(self):
        cfg = TIER_CONFIG[UserTier.ANALYST]
        assert cfg["price_usd"] == 49
        assert cfg["api_calls_per_day"] == 100
        assert cfg["narrative_depth"] == "VALIDATOR"
        assert cfg["telegram"] is True

    def test_strategist_tier(self):
        cfg = TIER_CONFIG[UserTier.STRATEGIST]
        assert cfg["price_usd"] == 99
        assert cfg["api_calls_per_day"] == 500
        assert cfg["narrative_depth"] == "NARRATOR"

    def test_institutional_tier(self):
        cfg = TIER_CONFIG[UserTier.INSTITUTIONAL]
        assert cfg["price_usd"] == 149
        assert cfg["api_calls_per_day"] == 2000
        assert cfg["chat"] is True
        assert cfg["webhooks"] is True

    def test_all_tiers_defined(self):
        for tier in UserTier:
            assert tier in TIER_CONFIG


# ============================================================================
# TESTS: USER CRUD
# ============================================================================

class TestUserCRUD:
    def test_create_user(self, manager):
        user = manager.create_user("test@example.com")
        assert user["user_id"] == 1
        assert user["email"] == "test@example.com"
        assert user["tier"] == "FREE"

    def test_create_user_with_tier(self, manager):
        user = manager.create_user("pro@example.com", tier=UserTier.STRATEGIST)
        assert user["tier"] == "STRATEGIST"

    def test_get_user(self, manager):
        created = manager.create_user("test@example.com")
        user = manager.get_user(created["user_id"])
        assert user is not None
        assert user["email"] == "test@example.com"

    def test_get_nonexistent_user(self, manager):
        assert manager.get_user(999) is None

    def test_duplicate_email_rejected(self, manager):
        manager.create_user("dup@example.com")
        with pytest.raises(Exception):  # sqlite3.IntegrityError
            manager.create_user("dup@example.com")

    def test_update_tier(self, manager):
        user = manager.create_user("test@example.com")
        success = manager.update_tier(user["user_id"], UserTier.ANALYST)
        assert success is True

        updated = manager.get_user(user["user_id"])
        assert updated["tier"] == "ANALYST"

    def test_link_api_key(self, manager):
        user = manager.create_user("test@example.com")
        success = manager.link_api_key(user["user_id"], api_key_id=42)
        assert success is True

        found = manager.get_user_by_api_key(42)
        assert found is not None
        assert found["email"] == "test@example.com"

    def test_get_user_by_api_key_not_found(self, manager):
        assert manager.get_user_by_api_key(999) is None

    def test_link_telegram(self, manager):
        user = manager.create_user("test@example.com")
        manager.link_telegram(user["user_id"], "123456789")

        updated = manager.get_user(user["user_id"])
        assert updated["telegram_chat_id"] == "123456789"


# ============================================================================
# TESTS: RATE LIMITING
# ============================================================================

class TestRateLimiting:
    def test_under_limit_allowed(self, manager):
        user = manager.create_user("test@example.com")
        # FREE tier: 10 calls/day
        for _ in range(5):
            manager.record_usage(user["user_id"], "/api/v1/signals")

        assert manager.check_rate_limit(user["user_id"]) is True

    def test_over_limit_blocked(self, manager):
        user = manager.create_user("test@example.com")
        # FREE tier: 10 calls/day
        for _ in range(10):
            manager.record_usage(user["user_id"], "/api/v1/signals")

        assert manager.check_rate_limit(user["user_id"]) is False

    def test_higher_tier_higher_limit(self, manager):
        user = manager.create_user("test@example.com", tier=UserTier.ANALYST)
        # ANALYST: 100 calls/day
        for _ in range(50):
            manager.record_usage(user["user_id"], "/api/v1/signals")

        assert manager.check_rate_limit(user["user_id"]) is True

    def test_nonexistent_user_blocked(self, manager):
        assert manager.check_rate_limit(999) is False

    def test_daily_usage_count(self, manager):
        user = manager.create_user("test@example.com")
        manager.record_usage(user["user_id"], "/api/v1/signals")
        manager.record_usage(user["user_id"], "/api/v1/signals")
        manager.record_usage(user["user_id"], "/api/v1/narratives")

        assert manager.get_daily_usage(user["user_id"]) == 3


# ============================================================================
# TESTS: TIER QUERIES
# ============================================================================

class TestTierQueries:
    def test_get_tier_config(self):
        cfg = UserTierManager.get_tier_config(UserTier.STRATEGIST)
        assert cfg["narrative_depth"] == "NARRATOR"

    def test_get_narrative_tier(self):
        assert UserTierManager.get_narrative_tier(UserTier.FREE) == "VISUAL"
        assert UserTierManager.get_narrative_tier(UserTier.ANALYST) == "VALIDATOR"
        assert UserTierManager.get_narrative_tier(UserTier.STRATEGIST) == "NARRATOR"

    def test_list_users_all(self, manager):
        manager.create_user("a@x.com")
        manager.create_user("b@x.com", tier=UserTier.ANALYST)
        assert len(manager.list_users()) == 2

    def test_list_users_filtered(self, manager):
        manager.create_user("a@x.com")
        manager.create_user("b@x.com", tier=UserTier.ANALYST)
        analysts = manager.list_users(tier=UserTier.ANALYST)
        assert len(analysts) == 1
        assert analysts[0]["email"] == "b@x.com"
