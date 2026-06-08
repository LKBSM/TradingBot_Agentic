"""Tests for the ForexFactory news pipeline (Chantier 3).

Covers: mocked fetch, currency filter, impact filter, schema mapping,
cache hit/miss (TTL), and the critical niveau-1.5 guarantee (100 news
combinations → zero forbidden token in any potential_effect_description).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.intelligence.market_reading_mappers import contains_forbidden_tokens
from src.intelligence.market_reading_schema import NewsJustPublished, NewsUpcoming
from src.intelligence.news_pipeline import NewsPipeline, normalize_ff_event
from src.storage import NewsCacheStore

NOW = datetime(2026, 5, 29, 14, 0, tzinfo=timezone.utc)


def ff_event(
    title: str,
    country: str,
    dt: datetime,
    impact: str = "high",
    actual: str = "",
    forecast: str = "",
    previous: str = "",
) -> dict:
    """Build one raw ForexFactory JSON event."""
    return {
        "title": title,
        "country": country,
        "date": dt.isoformat(),
        "impact": impact,
        "actual": actual,
        "forecast": forecast,
        "previous": previous,
    }


@pytest.fixture
def cache(tmp_path):
    return NewsCacheStore(db_path=str(tmp_path / "news_cache.db"))


def _pipeline(cache, raw_events, ttl_seconds=120):
    """Pipeline with an in-memory fetcher returning ``raw_events``."""
    return NewsPipeline(
        cache_store=cache,
        fetch_fn=lambda: list(raw_events),
        ttl_seconds=ttl_seconds,
    )


# ===================================================================== #
# Normalization
# ===================================================================== #
class TestNormalization:
    def test_basic_event_normalized(self):
        ev = ff_event("US NFP", "USD", NOW + timedelta(minutes=30), impact="high")
        norm = normalize_ff_event(ev)
        assert norm is not None
        assert norm.currency == "USD"
        assert norm.impact == "high"
        assert norm.scheduled_at.tzinfo is not None

    def test_low_impact_dropped(self):
        assert normalize_ff_event(ff_event("Minor", "USD", NOW, impact="low")) is None

    def test_holiday_dropped(self):
        assert normalize_ff_event(ff_event("Bank Holiday", "USD", NOW, impact="holiday")) is None

    def test_malformed_dropped(self):
        assert normalize_ff_event({"title": "", "country": "USD", "date": ""}) is None

    def test_numeric_parsing_suffixes(self):
        ev = ff_event("NFP", "USD", NOW, actual="180K", forecast="3.2%", previous="-0.1")
        norm = normalize_ff_event(ev)
        assert norm.actual == 180000.0
        assert norm.forecast == 3.2
        assert norm.previous == -0.1


# ===================================================================== #
# Currency filter
# ===================================================================== #
class TestCurrencyFilter:
    def test_default_filter_keeps_usd_eur_drops_gbp(self, cache):
        raw = [
            ff_event("US NFP", "USD", NOW + timedelta(minutes=30)),
            ff_event("EU CPI", "EUR", NOW + timedelta(minutes=60)),
            ff_event("UK GDP", "GBP", NOW + timedelta(minutes=90)),
        ]
        pipe = _pipeline(cache, raw)
        upcoming = pipe.get_upcoming(now=NOW)
        currencies = {u.currency for u in upcoming}
        assert currencies == {"USD", "EUR"}

    def test_none_filter_keeps_all(self, cache):
        raw = [
            ff_event("US NFP", "USD", NOW + timedelta(minutes=30)),
            ff_event("UK GDP", "GBP", NOW + timedelta(minutes=90)),
        ]
        pipe = _pipeline(cache, raw)
        upcoming = pipe.get_upcoming(currency_filter=None, now=NOW)
        assert {"USD", "GBP"}.issubset({u.currency for u in upcoming})


# ===================================================================== #
# Impact filter
# ===================================================================== #
class TestImpactFilter:
    def test_low_and_holiday_never_surface(self, cache):
        raw = [
            ff_event("US NFP", "USD", NOW + timedelta(minutes=30), impact="high"),
            ff_event("US PMI", "USD", NOW + timedelta(minutes=45), impact="medium"),
            ff_event("Minor data", "USD", NOW + timedelta(minutes=60), impact="low"),
            ff_event("Holiday", "USD", NOW + timedelta(minutes=75), impact="holiday"),
        ]
        pipe = _pipeline(cache, raw)
        upcoming = pipe.get_upcoming(now=NOW)
        impacts = {u.impact for u in upcoming}
        assert impacts == {"high", "medium"}
        assert len(upcoming) == 2


# ===================================================================== #
# Mapping to schema
# ===================================================================== #
class TestMapping:
    def test_upcoming_fields(self, cache):
        raw = [ff_event("US NFP", "USD", NOW + timedelta(minutes=30), impact="high")]
        pipe = _pipeline(cache, raw)
        upcoming = pipe.get_upcoming(now=NOW)
        assert len(upcoming) == 1
        u = upcoming[0]
        assert isinstance(u, NewsUpcoming)
        assert u.event == "US NFP"
        assert u.time_to_event_min == 30
        assert u.currency == "USD"
        assert u.potential_effect_description

    def test_lookahead_window_excludes_far_events(self, cache):
        raw = [
            ff_event("Soon", "USD", NOW + timedelta(minutes=30)),
            ff_event("Far", "USD", NOW + timedelta(minutes=600)),
        ]
        pipe = _pipeline(cache, raw)
        upcoming = pipe.get_upcoming(lookahead_minutes=240, now=NOW)
        assert [u.event for u in upcoming] == ["Soon"]

    def test_just_published_fields_and_surprise(self, cache):
        raw = [
            ff_event(
                "US NFP", "USD", NOW - timedelta(minutes=15),
                impact="high", actual="200K", forecast="180K", previous="175K",
            )
        ]
        pipe = _pipeline(cache, raw)
        published = pipe.get_just_published(now=NOW)
        assert len(published) == 1
        p = published[0]
        assert isinstance(p, NewsJustPublished)
        assert p.surprise_direction == "beat"  # 200K > 180K
        assert p.actual == 200000.0

    def test_just_published_miss(self, cache):
        raw = [
            ff_event(
                "US CPI", "USD", NOW - timedelta(minutes=10),
                impact="high", actual="2.9%", forecast="3.2%",
            )
        ]
        pipe = _pipeline(cache, raw)
        p = pipe.get_just_published(now=NOW)[0]
        assert p.surprise_direction == "miss"

    def test_just_published_lookback_excludes_old(self, cache):
        raw = [ff_event("Old", "USD", NOW - timedelta(minutes=120), impact="high")]
        pipe = _pipeline(cache, raw)
        assert pipe.get_just_published(lookback_minutes=60, now=NOW) == []


# ===================================================================== #
# Cache hit / miss (TTL)
# ===================================================================== #
class TestCacheTTL:
    def test_within_ttl_does_not_refetch(self, cache):
        calls = {"n": 0}

        def counting_fetch():
            calls["n"] += 1
            return [ff_event("US NFP", "USD", NOW + timedelta(minutes=30))]

        pipe = NewsPipeline(cache_store=cache, fetch_fn=counting_fetch, ttl_seconds=120)
        pipe.get_upcoming(now=NOW)
        pipe.get_upcoming(now=NOW + timedelta(seconds=60))  # still within TTL
        assert calls["n"] == 1

    def test_beyond_ttl_refetches(self, cache):
        calls = {"n": 0}

        def counting_fetch():
            calls["n"] += 1
            return [ff_event("US NFP", "USD", NOW + timedelta(minutes=30))]

        pipe = NewsPipeline(cache_store=cache, fetch_fn=counting_fetch, ttl_seconds=120)
        pipe.get_upcoming(now=NOW)
        pipe.get_upcoming(now=NOW + timedelta(seconds=200))  # past TTL
        assert calls["n"] == 2

    def test_fetch_failure_keeps_stale_cache(self, cache):
        # Seed once successfully.
        good = _pipeline(cache, [ff_event("US NFP", "USD", NOW + timedelta(minutes=30))])
        good.get_upcoming(now=NOW)

        def failing_fetch():
            raise RuntimeError("network down")

        pipe = NewsPipeline(cache_store=cache, fetch_fn=failing_fetch, ttl_seconds=0)
        # TTL=0 forces a refresh attempt; fetch fails but stale cache still served.
        upcoming = pipe.get_upcoming(now=NOW)
        assert len(upcoming) == 1


# ===================================================================== #
# CRITICAL — niveau 1.5 strict: 100 combinations, zero forbidden token
# ===================================================================== #
class TestNiveau15Strict:
    def test_100_combinations_zero_forbidden_token(self, cache):
        currencies = ["USD", "EUR", "GBP", "JPY", "CAD"]
        impacts = ["high", "medium"]
        # Realistic + a couple of adversarial titles embedding forbidden words,
        # to prove the _ensure_clean guard scrubs even feed-injected tokens.
        titles = [
            "Non-Farm Payrolls", "CPI y/y", "ECB Rate Decision", "GDP q/q",
            "Unemployment Rate", "Retail Sales m/m", "PMI Manufacturing",
            "Fed Chair Powell Speaks", "Trade Balance", "Core PCE",
            "achète massif rumeur",   # adversarial: contains "achète"
            "moment évite la séance",  # adversarial: contains "évite"
        ]
        raw = []
        idx = 0
        for cur in currencies:
            for imp in impacts:
                for title in titles:
                    # Spread events around NOW (both upcoming and just published).
                    offset = (idx % 21) - 10  # minutes -10..+10
                    dt = NOW + timedelta(minutes=offset * 5)
                    raw.append(ff_event(
                        title, cur, dt, impact=imp,
                        actual="200K", forecast="180K",
                    ))
                    idx += 1
        assert len(raw) >= 100

        pipe = NewsPipeline(
            cache_store=cache, fetch_fn=lambda: list(raw), ttl_seconds=120
        )
        upcoming = pipe.get_upcoming(
            currency_filter=None, lookahead_minutes=240, now=NOW
        )
        published = pipe.get_just_published(
            currency_filter=None, lookback_minutes=240, now=NOW
        )
        all_descriptions = (
            [u.potential_effect_description for u in upcoming]
            + [p.potential_effect_description for p in published]
        )
        assert len(all_descriptions) >= 100, (
            f"expected ≥100 generated descriptions, got {len(all_descriptions)}"
        )
        offenders = [
            (d, contains_forbidden_tokens(d))
            for d in all_descriptions
            if contains_forbidden_tokens(d) is not None
        ]
        assert offenders == [], f"forbidden tokens leaked: {offenders[:5]}"
