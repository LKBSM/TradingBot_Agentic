# =============================================================================
# SPRINT 3 INTEGRATION TESTS - Real-time & Data Sources
# =============================================================================
"""
Tests for Sprint 3 features:
- WebSocket news feeds
- Multi-source news aggregation
- Data source adapters (RSS, Twitter, FedWatch, COT)
- Multi-asset support
- Correlation tracking

Run tests:
    python -m pytest tests/test_sprint3_realtime.py -v
"""

import sys
import os
import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# =============================================================================
# TEST: NEWS SOURCE ADAPTERS
# =============================================================================

class TestBaseNewsAdapter:
    """Tests for base news adapter functionality."""

    def test_news_article_creation(self):
        """Test NewsArticle dataclass creation."""
        from src.agents.news.sources.base_adapter import (
            NewsArticle,
            ArticleSource,
            ArticleCategory
        )

        article = NewsArticle(
            article_id="test123",
            source_name="TestSource",
            source_type=ArticleSource.RSS,
            title="Fed Raises Rates",
            content="The Federal Reserve raised rates by 25 basis points.",
            category=ArticleCategory.CENTRAL_BANK,
            importance="HIGH"
        )

        assert article.article_id == "test123"
        assert article.source_type == ArticleSource.RSS
        assert article.category == ArticleCategory.CENTRAL_BANK
        assert article.importance == "HIGH"

    def test_article_id_generation(self):
        """Test unique ID generation for articles."""
        from src.agents.news.sources.base_adapter import NewsArticle

        id1 = NewsArticle.generate_id("Source1", "Title1", "url1")
        id2 = NewsArticle.generate_id("Source1", "Title1", "url1")
        id3 = NewsArticle.generate_id("Source2", "Title1", "url1")

        # Same inputs should generate same ID
        assert id1 == id2

        # Different inputs should generate different IDs
        assert id1 != id3

    def test_article_to_dict(self):
        """Test article serialization to dictionary."""
        from src.agents.news.sources.base_adapter import (
            NewsArticle,
            ArticleSource,
            ArticleCategory
        )

        article = NewsArticle(
            article_id="test123",
            source_name="TestSource",
            source_type=ArticleSource.API,
            title="Test Title",
            category=ArticleCategory.ECONOMIC
        )

        data = article.to_dict()

        assert data['article_id'] == "test123"
        assert data['source_type'] == "API"
        assert data['category'] == "ECONOMIC"
        assert 'published_at' in data


class TestRSSAdapter:
    """Tests for RSS adapter."""

    def test_rss_adapter_initialization(self):
        """Test RSS adapter can be initialized."""
        from src.agents.news.sources.rss_adapter import RSSAdapter, create_rss_adapter

        adapter = create_rss_adapter(
            feed_url="https://example.com/rss",
            name="TestFeed"
        )

        assert adapter is not None
        assert adapter.config.name == "TestFeed"

    def test_rss_adapter_status(self):
        """Test RSS adapter status reporting."""
        from src.agents.news.sources.rss_adapter import create_rss_adapter

        adapter = create_rss_adapter(
            feed_url="https://example.com/rss",
            name="TestFeed"
        )

        status = adapter.get_status()

        assert 'name' in status
        assert 'enabled' in status
        assert status['name'] == "TestFeed"


class TestTwitterAdapter:
    """Tests for Twitter adapter."""

    def test_twitter_adapter_initialization(self):
        """Test Twitter adapter can be initialized."""
        from src.agents.news.sources.twitter_adapter import (
            TwitterAdapter,
            create_twitter_adapter
        )

        adapter = create_twitter_adapter(bearer_token="test_token")

        assert adapter is not None
        assert adapter.twitter_config.bearer_token == "test_token"

    def test_twitter_search_query_building(self):
        """Test Twitter search query construction."""
        from src.agents.news.sources.twitter_adapter import TwitterAdapter

        adapter = TwitterAdapter(bearer_token="test")
        query = adapter._build_search_query()

        # Should contain keywords and filters
        assert len(query) > 0
        assert "-is:retweet" in query

    def test_twitter_categorization(self):
        """Test tweet categorization logic."""
        from src.agents.news.sources.twitter_adapter import TwitterAdapter
        from src.agents.news.sources.base_adapter import ArticleCategory

        adapter = TwitterAdapter()

        # Fed-related tweet
        category = adapter._categorize_tweet("Fed raises rates by 25bps")
        assert category == ArticleCategory.CENTRAL_BANK

        # Economic data tweet
        category = adapter._categorize_tweet("CPI comes in at 3.2%")
        assert category == ArticleCategory.ECONOMIC

        # General market tweet
        category = adapter._categorize_tweet("Markets rally on earnings")
        assert category == ArticleCategory.MARKET


class TestFedWatchAdapter:
    """Tests for FedWatch adapter."""

    def test_fedwatch_adapter_initialization(self):
        """Test FedWatch adapter can be initialized."""
        from src.agents.news.sources.fed_watch_adapter import (
            FedWatchAdapter,
            create_fed_watch_adapter
        )

        adapter = create_fed_watch_adapter()

        assert adapter is not None
        assert adapter.fed_config.name == "FedWatch"

    def test_fedwatch_mock_probabilities(self):
        """Test mock probability generation."""
        from src.agents.news.sources.fed_watch_adapter import FedWatchAdapter

        adapter = FedWatchAdapter()
        probabilities = adapter._generate_mock_probabilities()

        # Should have probabilities for upcoming meetings
        assert len(probabilities) > 0

        # Each should have required fields
        for meeting_key, prob in probabilities.items():
            assert prob.meeting_date is not None
            assert 0 <= prob.hike_probability <= 1
            assert 0 <= prob.cut_probability <= 1
            assert 0 <= prob.hold_probability <= 1
            # Probabilities should sum to approximately 1
            total = prob.hike_probability + prob.cut_probability + prob.hold_probability
            assert 0.95 <= total <= 1.05

    def test_fedwatch_status(self):
        """Test FedWatch status reporting."""
        from src.agents.news.sources.fed_watch_adapter import FedWatchAdapter

        adapter = FedWatchAdapter()
        status = adapter.get_status()

        assert 'updates_fetched' in status
        assert 'current_rate_bps' in status


class TestCOTAdapter:
    """Tests for CFTC COT adapter."""

    def test_cot_adapter_initialization(self):
        """Test COT adapter can be initialized."""
        from src.agents.news.sources.cot_adapter import COTAdapter, create_cot_adapter

        adapter = create_cot_adapter()

        assert adapter is not None
        assert adapter.cot_config.name == "CFTC_COT"

    def test_cot_mock_positions(self):
        """Test mock position generation."""
        from src.agents.news.sources.cot_adapter import COTAdapter

        adapter = COTAdapter()
        positions = adapter._generate_mock_positions()

        # Should have gold position at minimum
        assert len(positions) > 0
        assert any('GOLD' in key for key in positions.keys())

        # Check position structure
        for instrument, pos in positions.items():
            assert pos.commercial_long >= 0
            assert pos.commercial_short >= 0
            assert pos.non_commercial_long >= 0
            assert pos.non_commercial_short >= 0
            assert pos.open_interest > 0

    def test_cot_spec_index_calculation(self):
        """Test speculator index calculation."""
        from src.agents.news.sources.cot_adapter import COTAdapter, COTPosition

        adapter = COTAdapter()

        # Create a test position
        position = COTPosition(
            report_date=datetime.now(),
            instrument="TEST",
            cftc_code="123",
            commercial_long=100000,
            commercial_short=150000,
            commercial_net=-50000,
            non_commercial_long=200000,
            non_commercial_short=50000,
            non_commercial_net=150000,
            non_reportable_long=50000,
            non_reportable_short=50000,
            non_reportable_net=0,
            open_interest=400000,
            open_interest_change=5000
        )

        spec_index = adapter._calculate_spec_index(position)

        # Should be between 0 and 100
        assert 0 <= spec_index <= 100


# =============================================================================
# TEST: NEWS AGGREGATOR
# =============================================================================

class TestNewsAggregator:
    """Tests for multi-source news aggregator."""

    def test_aggregator_initialization(self):
        """Test aggregator can be initialized."""
        from src.agents.news.aggregator import (
            NewsAggregator,
            create_news_aggregator
        )

        aggregator = create_news_aggregator()

        assert aggregator is not None
        assert aggregator.config is not None

    def test_source_registration(self):
        """Test source registration."""
        from src.agents.news.aggregator import (
            NewsAggregator,
            SourcePriority
        )
        from src.agents.news.sources.rss_adapter import create_rss_adapter

        aggregator = NewsAggregator()
        adapter = create_rss_adapter("https://example.com/rss", "Test")

        result = aggregator.register_source(
            adapter,
            "test_rss",
            SourcePriority.SECONDARY
        )

        assert result is True
        assert "test_rss" in aggregator._sources

    def test_duplicate_source_registration(self):
        """Test that duplicate source names are rejected."""
        from src.agents.news.aggregator import NewsAggregator, SourcePriority
        from src.agents.news.sources.rss_adapter import create_rss_adapter

        aggregator = NewsAggregator()
        adapter = create_rss_adapter("https://example.com/rss", "Test")

        aggregator.register_source(adapter, "test_rss", SourcePriority.SECONDARY)
        result = aggregator.register_source(adapter, "test_rss", SourcePriority.PRIMARY)

        assert result is False  # Should reject duplicate

    def test_aggregator_status(self):
        """Test aggregator status reporting."""
        from src.agents.news.aggregator import NewsAggregator

        aggregator = NewsAggregator()
        status = aggregator.get_status()

        assert 'is_running' in status
        assert 'total_sources' in status
        assert 'total_articles' in status


# =============================================================================
# TEST: WEBSOCKET FEED
# =============================================================================

class TestWebSocketFeed:
    """Tests for WebSocket news feed."""

    def test_websocket_config(self):
        """Test WebSocket configuration."""
        from src.agents.news.websocket_feed import WebSocketConfig

        config = WebSocketConfig(
            url="wss://news.example.com/ws",
            name="TestWS"
        )

        assert config.url == "wss://news.example.com/ws"
        assert config.name == "TestWS"
        assert config.auto_reconnect is True

    def test_websocket_feed_initialization(self):
        """Test WebSocket feed can be initialized."""
        from src.agents.news.websocket_feed import (
            WebSocketNewsFeed,
            WebSocketConfig,
            ConnectionState
        )

        config = WebSocketConfig(url="wss://test.com/ws", name="Test")
        feed = WebSocketNewsFeed(config)

        assert feed.state == ConnectionState.DISCONNECTED
        assert feed.is_connected is False

    def test_websocket_message_parsing(self):
        """Test WebSocket message parsing."""
        from src.agents.news.websocket_feed import (
            WebSocketNewsFeed,
            WebSocketConfig
        )
        import json

        config = WebSocketConfig(url="wss://test.com/ws", name="Test")
        feed = WebSocketNewsFeed(config)

        # Test valid article message
        message = json.dumps({
            'type': 'article',
            'id': 'test123',
            'title': 'Fed Decision',
            'content': 'Fed holds rates',
            'source': 'Reuters',
            'importance': 'HIGH'
        })

        article = feed._parse_message(message)

        assert article is not None
        assert article.title == 'Fed Decision'
        assert article.importance == 'HIGH'

    def test_websocket_heartbeat_ignored(self):
        """Test that heartbeat messages are ignored."""
        from src.agents.news.websocket_feed import (
            WebSocketNewsFeed,
            WebSocketConfig
        )
        import json

        config = WebSocketConfig(url="wss://test.com/ws", name="Test")
        feed = WebSocketNewsFeed(config)

        # Heartbeat message
        message = json.dumps({'type': 'heartbeat'})
        article = feed._parse_message(message)

        assert article is None

    def test_websocket_status(self):
        """Test WebSocket feed status."""
        from src.agents.news.websocket_feed import (
            WebSocketNewsFeed,
            WebSocketConfig
        )

        config = WebSocketConfig(url="wss://test.com/ws", name="TestFeed")
        feed = WebSocketNewsFeed(config)

        status = feed.get_status()

        assert status['name'] == 'TestFeed'
        assert status['is_connected'] is False
        assert 'messages_received' in status


# =============================================================================
# TEST: MULTI-ASSET SUPPORT
# =============================================================================

class TestAssetConfig:
    """Tests for asset configuration."""

    def test_get_asset_config(self):
        """Test retrieving asset configuration."""
        from src.multi_asset.asset_config import get_asset_config, AssetClass

        config = get_asset_config('XAUUSD')

        assert config is not None
        assert config.symbol == 'XAUUSD'
        assert config.asset_class == AssetClass.COMMODITY_METAL

    def test_get_all_assets(self):
        """Test retrieving all supported assets."""
        from src.multi_asset.asset_config import get_all_assets

        assets = get_all_assets()

        assert 'XAUUSD' in assets
        assert 'EURUSD' in assets
        assert len(assets) >= 5

    def test_get_assets_by_class(self):
        """Test filtering assets by class."""
        from src.multi_asset.asset_config import get_assets_by_class, AssetClass

        forex_assets = get_assets_by_class(AssetClass.FOREX_MAJOR)

        assert 'EURUSD' in forex_assets
        assert 'GBPUSD' in forex_assets
        assert 'XAUUSD' not in forex_assets  # Gold is commodity

    def test_asset_config_structure(self):
        """Test asset config has required fields."""
        from src.multi_asset.asset_config import get_asset_config

        config = get_asset_config('EURUSD')

        assert config.pip_value > 0
        assert config.lot_size > 0
        assert config.typical_spread_pips >= 0
        assert len(config.high_impact_events) > 0


class TestMultiAssetManager:
    """Tests for multi-asset manager."""

    def test_manager_initialization(self):
        """Test manager can be initialized."""
        from src.multi_asset.asset_manager import (
            MultiAssetManager,
            create_multi_asset_manager
        )

        manager = create_multi_asset_manager(assets=['XAUUSD', 'EURUSD'])

        assert manager is not None
        assert 'XAUUSD' in manager._assets
        assert 'EURUSD' in manager._assets

    def test_add_remove_asset(self):
        """Test adding and removing assets."""
        from src.multi_asset.asset_manager import MultiAssetManager

        manager = MultiAssetManager(assets=['XAUUSD'])

        # Add asset
        result = manager.add_asset('EURUSD')
        assert result is True
        assert 'EURUSD' in manager._assets

        # Remove asset
        result = manager.remove_asset('EURUSD')
        assert result is True
        assert 'EURUSD' not in manager._assets

    def test_price_update(self):
        """Test price updates."""
        from src.multi_asset.asset_manager import MultiAssetManager

        manager = MultiAssetManager(assets=['XAUUSD'])

        manager.update_price('XAUUSD', bid=1950.50, ask=1951.00)

        state = manager.get_state('XAUUSD')
        assert state is not None
        assert state.bid == 1950.50
        assert state.ask == 1951.00
        assert abs(state.last_price - 1950.75) < 0.01

    def test_can_open_position_checks(self):
        """Test position opening validation."""
        from src.multi_asset.asset_manager import MultiAssetManager

        manager = MultiAssetManager(assets=['XAUUSD'])
        manager.update_price('XAUUSD', bid=1950.00, ask=1950.50)

        # Valid position
        can_open, reason = manager.can_open_position('XAUUSD', 1, 0.5)
        assert can_open is True

        # Invalid - unknown asset
        can_open, reason = manager.can_open_position('INVALID', 1, 0.5)
        assert can_open is False
        assert 'Unknown' in reason

    def test_portfolio_status(self):
        """Test portfolio status reporting."""
        from src.multi_asset.asset_manager import MultiAssetManager

        manager = MultiAssetManager(assets=['XAUUSD', 'EURUSD'])

        status = manager.get_portfolio_status()

        assert 'managed_assets' in status
        assert 'total_exposure_usd' in status
        assert 'daily_pnl' in status
        assert 'limits' in status


class TestCorrelationTracker:
    """Tests for correlation tracking."""

    def test_tracker_initialization(self):
        """Test tracker can be initialized."""
        from src.multi_asset.correlation_tracker import (
            CorrelationTracker,
            create_correlation_tracker
        )

        tracker = create_correlation_tracker(assets=['XAUUSD', 'EURUSD'])

        assert tracker is not None
        assert 'XAUUSD' in tracker._prices
        assert 'EURUSD' in tracker._prices

    def test_price_history_tracking(self):
        """Test that prices are tracked over time."""
        from src.multi_asset.correlation_tracker import CorrelationTracker

        tracker = CorrelationTracker()
        tracker.add_asset('XAUUSD')

        # Add multiple prices
        for i in range(10):
            tracker.update_price('XAUUSD', 1950 + i)

        assert len(tracker._prices['XAUUSD']) == 10

    def test_correlation_calculation(self):
        """Test correlation calculation."""
        from src.multi_asset.correlation_tracker import CorrelationTracker

        tracker = CorrelationTracker()
        tracker.add_asset('ASSET1')
        tracker.add_asset('ASSET2')

        # Add correlated prices (both going up)
        np.random.seed(42)
        for i in range(50):
            base = 100 + i
            tracker.update_price('ASSET1', base + np.random.normal(0, 1))
            tracker.update_price('ASSET2', base + np.random.normal(0, 1))

        corr = tracker.get_correlation('ASSET1', 'ASSET2')

        # Should be positively correlated
        assert corr > 0.5

    def test_self_correlation(self):
        """Test correlation of asset with itself is 1."""
        from src.multi_asset.correlation_tracker import CorrelationTracker

        tracker = CorrelationTracker()
        tracker.add_asset('XAUUSD')

        corr = tracker.get_correlation('XAUUSD', 'XAUUSD')

        assert corr == 1.0

    def test_exposure_multiplier(self):
        """Test exposure multiplier calculation."""
        from src.multi_asset.correlation_tracker import CorrelationTracker

        tracker = CorrelationTracker()

        # No existing positions should give multiplier of 1.0
        multiplier = tracker.get_exposure_multiplier('XAUUSD', {})
        assert multiplier == 1.0

        # With correlated positions, multiplier should be reduced
        # (This test is simplified since we need price history)
        multiplier = tracker.get_exposure_multiplier('XAUUSD', {'EURUSD': 1.0})
        assert 0 < multiplier <= 1.0


# =============================================================================
# TEST: DATA SOURCES CONFIG
# =============================================================================

class TestDataSourcesConfig:
    """Tests for data sources configuration."""

    def test_config_initialization(self):
        """Test configuration initialization."""
        from src.agents.data_sources_config import DataSourcesConfig

        config = DataSourcesConfig()

        assert config.news_api.enabled is True
        assert config.finnhub.enabled is True
        assert len(config.primary_assets) > 0

    def test_get_enabled_sources(self):
        """Test getting list of enabled sources."""
        from src.agents.data_sources_config import DataSourcesConfig

        config = DataSourcesConfig()
        enabled = config.get_enabled_sources()

        assert isinstance(enabled, list)
        # Some sources should be enabled by default
        assert len(enabled) > 0

    def test_rate_limit_config(self):
        """Test rate limit configuration."""
        from src.agents.data_sources_config import DataSourcesConfig

        config = DataSourcesConfig()

        # Check news API rate limits
        assert config.news_api.rate_limit.requests_per_minute > 0
        assert config.news_api.rate_limit.requests_per_day > 0

    def test_api_key_retrieval(self):
        """Test API key retrieval function."""
        from src.agents.data_sources_config import get_api_key
        import os

        # Set a test environment variable
        os.environ['TEST_API_KEY'] = 'test_value_123'

        key = get_api_key('TEST_API_KEY')
        assert key == 'test_value_123'

        # Clean up
        del os.environ['TEST_API_KEY']

        # Non-existent key should return empty string
        key = get_api_key('NON_EXISTENT_KEY')
        assert key == ''


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestSprint3Integration:
    """Integration tests for Sprint 3 features."""

    def test_full_news_pipeline(self):
        """Test complete news pipeline: adapter -> aggregator."""
        from src.agents.news.aggregator import NewsAggregator, SourcePriority
        from src.agents.news.sources.rss_adapter import create_rss_adapter

        # Create aggregator
        aggregator = NewsAggregator()

        # Register sources
        rss = create_rss_adapter("https://example.com/rss", "TestRSS")
        aggregator.register_source(rss, "test_rss", SourcePriority.SECONDARY)

        # Check status
        status = aggregator.get_status()
        assert status['total_sources'] == 1

    def test_multi_asset_with_correlation(self):
        """Test multi-asset manager with correlation tracking."""
        from src.multi_asset.asset_manager import MultiAssetManager
        from src.multi_asset.correlation_tracker import CorrelationTracker

        # Create components
        manager = MultiAssetManager(assets=['XAUUSD', 'EURUSD'])
        tracker = CorrelationTracker()

        # Add assets to tracker
        tracker.add_asset('XAUUSD')
        tracker.add_asset('EURUSD')

        # Update prices
        manager.update_price('XAUUSD', bid=1950.00, ask=1950.50)
        manager.update_price('EURUSD', bid=1.0850, ask=1.0852)

        tracker.update_price('XAUUSD', 1950.25)
        tracker.update_price('EURUSD', 1.0851)

        # Both should have state
        gold_state = manager.get_state('XAUUSD')
        eur_state = manager.get_state('EURUSD')

        assert gold_state is not None
        assert eur_state is not None

    def test_all_adapters_importable(self):
        """Test that all adapters can be imported."""
        from src.agents.news.sources import (
            BaseNewsAdapter,
            NewsArticle,
            RSSAdapter,
            TwitterAdapter,
            FedWatchAdapter,
            COTAdapter
        )

        # All should be importable (even if None due to missing deps)
        assert NewsArticle is not None
        assert BaseNewsAdapter is not None

    def test_sprint3_module_imports(self):
        """Test that all Sprint 3 modules can be imported."""
        # News module
        from src.agents import news
        assert news is not None

        # Multi-asset module
        from src import multi_asset
        assert multi_asset is not None


# =============================================================================
# ASYNC TESTS
# =============================================================================

class TestAsyncFunctionality:
    """Tests for async functionality."""

    @pytest.mark.asyncio
    async def test_async_adapter_fetch(self):
        """Test async fetch on adapters."""
        from src.agents.news.sources.fed_watch_adapter import FedWatchAdapter

        adapter = FedWatchAdapter()

        # This will use mock data since no API key
        articles = await adapter.fetch_async()

        # Should return articles (even if mock)
        assert isinstance(articles, list)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
