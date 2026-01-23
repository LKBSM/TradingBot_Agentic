# =============================================================================
# NEWS MODULE - News Analysis Components for Forex Trading v3.0
# =============================================================================
# This module provides news-related functionality for the trading system:
#
# === EXISTING (Sprint 1-2) ===
#   - Economic calendar parsing (ForexFactory, Investing.com)
#   - News headline fetching (NewsAPI, Central Bank RSS)
#   - Sentiment analysis (rule-based keyword matching)
#   - Impact classification (HIGH/MEDIUM/LOW)
#
# === SPRINT 3: Real-time & Multi-Source ===
#   - WebSocket real-time news feeds (<1 second latency)
#   - Multi-source news aggregator (RSS + WebSocket + API)
#   - Source adapters (RSS, Twitter, FedWatch, COT)
#   - Event-driven architecture with EventBus integration
#
# =============================================================================

import logging

logger = logging.getLogger(__name__)

# =============================================================================
# CORE COMPONENTS (Always Available)
# =============================================================================

from .sentiment import SentimentAnalyzer
from .economic_calendar import EconomicCalendarFetcher, EconomicEvent
from .fetchers import NewsHeadlineFetcher, NewsItem

# =============================================================================
# SPRINT 3: REAL-TIME COMPONENTS
# =============================================================================

# WebSocket feed
WebSocketNewsFeed = None
WebSocketConfig = None
create_websocket_feed = None
ConnectionState = None

try:
    from .websocket_feed import (
        WebSocketNewsFeed,
        WebSocketConfig,
        create_websocket_feed,
        ConnectionState
    )
except ImportError as e:
    logger.debug(f"WebSocket feed not available: {e}")

# News aggregator
NewsAggregator = None
AggregatorConfig = None
SourcePriority = None
create_news_aggregator = None

try:
    from .aggregator import (
        NewsAggregator,
        AggregatorConfig,
        SourcePriority,
        create_news_aggregator
    )
except ImportError as e:
    logger.debug(f"News aggregator not available: {e}")

# =============================================================================
# SOURCE ADAPTERS
# =============================================================================

# Base adapter
BaseNewsAdapter = None
NewsArticle = None
AdapterConfig = None
ArticleSource = None
ArticleCategory = None

try:
    from .sources import (
        BaseNewsAdapter,
        NewsArticle,
        AdapterConfig,
        ArticleSource,
        ArticleCategory
    )
except ImportError as e:
    logger.debug(f"Base adapter not available: {e}")

# RSS Adapter
RSSAdapter = None
create_rss_adapter = None

try:
    from .sources import RSSAdapter, create_rss_adapter
except ImportError as e:
    logger.debug(f"RSS adapter not available: {e}")

# Twitter Adapter
TwitterAdapter = None
create_twitter_adapter = None

try:
    from .sources import TwitterAdapter, create_twitter_adapter
except ImportError as e:
    logger.debug(f"Twitter adapter not available: {e}")

# FedWatch Adapter (CME Rate Probabilities)
FedWatchAdapter = None
create_fed_watch_adapter = None
RateProbability = None

try:
    from .sources import FedWatchAdapter, create_fed_watch_adapter, RateProbability
except ImportError as e:
    logger.debug(f"FedWatch adapter not available: {e}")

# COT Adapter (CFTC Commitment of Traders)
COTAdapter = None
create_cot_adapter = None
COTPosition = None

try:
    from .sources import COTAdapter, create_cot_adapter, COTPosition
except ImportError as e:
    logger.debug(f"COT adapter not available: {e}")

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Core (Sprint 1-2)
    'SentimentAnalyzer',
    'EconomicCalendarFetcher',
    'EconomicEvent',
    'NewsHeadlineFetcher',
    'NewsItem',

    # Sprint 3: Real-time
    'WebSocketNewsFeed',
    'WebSocketConfig',
    'create_websocket_feed',
    'ConnectionState',
    'NewsAggregator',
    'AggregatorConfig',
    'SourcePriority',
    'create_news_aggregator',

    # Source Adapters
    'BaseNewsAdapter',
    'NewsArticle',
    'AdapterConfig',
    'ArticleSource',
    'ArticleCategory',
    'RSSAdapter',
    'create_rss_adapter',
    'TwitterAdapter',
    'create_twitter_adapter',
    'FedWatchAdapter',
    'create_fed_watch_adapter',
    'RateProbability',
    'COTAdapter',
    'create_cot_adapter',
    'COTPosition',
]

__version__ = '3.0.0'  # Sprint 3: Real-time & Multi-Source
