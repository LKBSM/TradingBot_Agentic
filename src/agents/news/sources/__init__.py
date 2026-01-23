# =============================================================================
# NEWS SOURCES MODULE - Data Source Adapters for Sprint 3
# =============================================================================
"""
This module provides adapters for various news data sources:
- RSS feeds (financial news sites)
- Twitter/X API (social sentiment)
- CME FedWatch (interest rate probabilities)
- CFTC COT Reports (institutional positioning)

Each adapter implements a common interface for fetching and normalizing data.
"""

import logging

logger = logging.getLogger(__name__)

# Base classes (always available)
from .base_adapter import (
    BaseNewsAdapter,
    NewsArticle,
    AdapterConfig,
    ArticleSource,
    ArticleCategory
)

# RSS Adapter
from .rss_adapter import RSSAdapter, create_rss_adapter

# Twitter Adapter
TwitterAdapter = None
create_twitter_adapter = None
try:
    from .twitter_adapter import TwitterAdapter, create_twitter_adapter
except ImportError as e:
    logger.debug(f"Twitter adapter not available: {e}")

# FedWatch Adapter
FedWatchAdapter = None
create_fed_watch_adapter = None
RateProbability = None
try:
    from .fed_watch_adapter import (
        FedWatchAdapter,
        create_fed_watch_adapter,
        RateProbability
    )
except ImportError as e:
    logger.debug(f"FedWatch adapter not available: {e}")

# COT Adapter
COTAdapter = None
create_cot_adapter = None
COTPosition = None
try:
    from .cot_adapter import (
        COTAdapter,
        create_cot_adapter,
        COTPosition
    )
except ImportError as e:
    logger.debug(f"COT adapter not available: {e}")


__all__ = [
    # Base classes
    'BaseNewsAdapter',
    'NewsArticle',
    'AdapterConfig',
    'ArticleSource',
    'ArticleCategory',
    # RSS
    'RSSAdapter',
    'create_rss_adapter',
    # Twitter
    'TwitterAdapter',
    'create_twitter_adapter',
    # FedWatch
    'FedWatchAdapter',
    'create_fed_watch_adapter',
    'RateProbability',
    # COT
    'COTAdapter',
    'create_cot_adapter',
    'COTPosition',
]
