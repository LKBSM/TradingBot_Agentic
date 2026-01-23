# =============================================================================
# RSS NEWS ADAPTER - RSS Feed Adapter for Financial News
# =============================================================================
"""
Adapter for fetching news from RSS feeds.

Supports feeds from:
- Reuters
- Bloomberg
- Financial Times
- Central Bank news
- Forex Factory

Uses feedparser for parsing and aiohttp for async fetching.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
import logging
import asyncio

from .base_adapter import (
    BaseNewsAdapter,
    NewsArticle,
    AdapterConfig,
    ArticleSource,
    ArticleCategory
)

logger = logging.getLogger(__name__)

# Try to import feedparser (optional dependency)
try:
    import feedparser
    HAS_FEEDPARSER = True
except ImportError:
    HAS_FEEDPARSER = False
    logger.warning("feedparser not installed, RSS adapter will be limited")

# Try to import aiohttp (optional dependency)
try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False
    logger.warning("aiohttp not installed, async RSS fetching will be limited")


# Default RSS feeds for financial news
DEFAULT_FEEDS = {
    'reuters_markets': {
        'url': 'https://www.reutersagency.com/feed/?best-topics=business-finance',
        'category': ArticleCategory.MARKET,
        'importance': 'HIGH'
    },
    'forex_factory': {
        'url': 'https://www.forexfactory.com/ffcal_week_this.xml',
        'category': ArticleCategory.ECONOMIC,
        'importance': 'HIGH'
    },
    'investing_news': {
        'url': 'https://www.investing.com/rss/news.rss',
        'category': ArticleCategory.MARKET,
        'importance': 'MEDIUM'
    },
}


@dataclass
class RSSFeedConfig:
    """Configuration for a single RSS feed."""
    url: str
    name: str = ""
    category: ArticleCategory = ArticleCategory.UNKNOWN
    importance: str = "MEDIUM"
    enabled: bool = True


@dataclass
class RSSAdapterConfig(AdapterConfig):
    """Configuration for RSS adapter."""
    feeds: List[RSSFeedConfig] = field(default_factory=list)
    fetch_timeout_sec: float = 30.0


class RSSAdapter(BaseNewsAdapter):
    """
    Adapter for fetching news from RSS feeds.

    Supports both synchronous and asynchronous fetching.
    """

    def __init__(
        self,
        feeds: Optional[List[RSSFeedConfig]] = None,
        config: Optional[RSSAdapterConfig] = None
    ):
        """
        Initialize RSS adapter.

        Args:
            feeds: List of RSS feed configurations
            config: Adapter configuration
        """
        super().__init__(config or RSSAdapterConfig(name="RSS"))

        self.feeds = feeds or []

        # Add default feeds if none specified
        if not self.feeds:
            for name, feed_info in DEFAULT_FEEDS.items():
                self.feeds.append(RSSFeedConfig(
                    url=feed_info['url'],
                    name=name,
                    category=feed_info['category'],
                    importance=feed_info['importance']
                ))

    def fetch(self) -> List[NewsArticle]:
        """
        Fetch news from all RSS feeds (synchronous).

        Returns:
            List of NewsArticle objects
        """
        if not HAS_FEEDPARSER:
            self._logger.error("feedparser not installed, cannot fetch RSS")
            return []

        if self._should_rate_limit():
            self._logger.warning("Rate limited, skipping fetch")
            return list(self._cache.values())

        articles = []

        for feed_config in self.feeds:
            if not feed_config.enabled:
                continue

            try:
                feed_articles = self._fetch_feed(feed_config)
                articles.extend(feed_articles)
            except Exception as e:
                self._logger.error(f"Error fetching {feed_config.name}: {e}")

        self._record_request()

        # Update cache
        for article in articles:
            self._cache[article.article_id] = article

        return self._filter_articles(articles)

    async def fetch_async(self) -> List[NewsArticle]:
        """
        Fetch news from all RSS feeds (asynchronous).

        Returns:
            List of NewsArticle objects
        """
        if not HAS_AIOHTTP or not HAS_FEEDPARSER:
            # Fall back to sync
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.fetch)

        if self._should_rate_limit():
            self._logger.warning("Rate limited, skipping fetch")
            return list(self._cache.values())

        articles = []
        tasks = []

        for feed_config in self.feeds:
            if feed_config.enabled:
                tasks.append(self._fetch_feed_async(feed_config))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, list):
                    articles.extend(result)
                elif isinstance(result, Exception):
                    self._logger.error(f"Feed fetch error: {result}")

        self._record_request()

        # Update cache
        for article in articles:
            self._cache[article.article_id] = article

        return self._filter_articles(articles)

    def _fetch_feed(self, feed_config: RSSFeedConfig) -> List[NewsArticle]:
        """Fetch a single RSS feed (synchronous)."""
        feed = feedparser.parse(feed_config.url)

        if feed.bozo:
            self._logger.warning(f"Feed parse warning for {feed_config.name}: {feed.bozo_exception}")

        articles = []
        for entry in feed.entries:
            article = self._parse_entry(entry, feed_config)
            if article:
                articles.append(article)

        self._logger.info(f"Fetched {len(articles)} articles from {feed_config.name}")
        return articles

    async def _fetch_feed_async(self, feed_config: RSSFeedConfig) -> List[NewsArticle]:
        """Fetch a single RSS feed (asynchronous)."""
        config = self.config if isinstance(self.config, RSSAdapterConfig) else RSSAdapterConfig()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    feed_config.url,
                    timeout=aiohttp.ClientTimeout(total=config.fetch_timeout_sec)
                ) as response:
                    content = await response.text()
                    feed = feedparser.parse(content)

                    articles = []
                    for entry in feed.entries:
                        article = self._parse_entry(entry, feed_config)
                        if article:
                            articles.append(article)

                    self._logger.info(f"Fetched {len(articles)} articles from {feed_config.name}")
                    return articles

        except Exception as e:
            self._logger.error(f"Async fetch error for {feed_config.name}: {e}")
            return []

    def _parse_entry(
        self,
        entry: Any,
        feed_config: RSSFeedConfig
    ) -> Optional[NewsArticle]:
        """Parse a single RSS feed entry into a NewsArticle."""
        try:
            # Extract title
            title = entry.get('title', '').strip()
            if not title:
                return None

            # Extract URL
            url = entry.get('link', '')

            # Generate ID
            article_id = NewsArticle.generate_id(
                feed_config.name,
                title,
                url
            )

            # Extract content/summary
            content = ""
            summary = ""

            if 'content' in entry:
                content = entry.content[0].get('value', '') if entry.content else ""
            if 'summary' in entry:
                summary = entry.get('summary', '')

            # Extract published date
            published_at = datetime.now()
            if 'published_parsed' in entry and entry.published_parsed:
                try:
                    import time
                    published_at = datetime.fromtimestamp(
                        time.mktime(entry.published_parsed)
                    )
                except Exception:
                    pass

            # Detect assets mentioned
            assets = self._detect_assets(title + " " + summary)

            # Extract keywords from categories/tags
            keywords = []
            if 'tags' in entry:
                keywords = [tag.get('term', '') for tag in entry.tags if tag.get('term')]

            return NewsArticle(
                article_id=article_id,
                source_name=feed_config.name,
                source_type=ArticleSource.RSS,
                title=title,
                content=content,
                summary=summary,
                url=url,
                published_at=published_at,
                category=feed_config.category,
                assets=assets,
                keywords=keywords,
                importance=feed_config.importance
            )

        except Exception as e:
            self._logger.debug(f"Error parsing entry: {e}")
            return None

    def _detect_assets(self, text: str) -> List[str]:
        """Detect asset mentions in text."""
        assets = []
        text_upper = text.upper()

        # Forex pairs
        forex_pairs = [
            'XAUUSD', 'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD',
            'USDCAD', 'USDCHF', 'NZDUSD', 'EUR/USD', 'GBP/USD',
            'XAU/USD', 'USD/JPY', 'AUD/USD', 'USD/CAD'
        ]
        for pair in forex_pairs:
            if pair in text_upper or pair.replace('/', '') in text_upper:
                normalized = pair.replace('/', '')
                if normalized not in assets:
                    assets.append(normalized)

        # Commodities
        if 'GOLD' in text_upper or 'PRECIOUS' in text_upper:
            if 'XAUUSD' not in assets:
                assets.append('XAUUSD')
        if 'SILVER' in text_upper:
            assets.append('XAGUSD')
        if 'OIL' in text_upper or 'CRUDE' in text_upper:
            assets.append('USOIL')

        return assets

    def add_feed(self, feed_config: RSSFeedConfig) -> None:
        """Add a new RSS feed."""
        self.feeds.append(feed_config)
        self._logger.info(f"Added RSS feed: {feed_config.name}")

    def remove_feed(self, name: str) -> bool:
        """Remove an RSS feed by name."""
        for i, feed in enumerate(self.feeds):
            if feed.name == name:
                del self.feeds[i]
                self._logger.info(f"Removed RSS feed: {name}")
                return True
        return False

    def get_feeds(self) -> List[Dict[str, Any]]:
        """Get list of configured feeds."""
        return [
            {
                'name': f.name,
                'url': f.url,
                'category': f.category.name,
                'importance': f.importance,
                'enabled': f.enabled
            }
            for f in self.feeds
        ]


def create_rss_adapter(
    feed_url: str = "",
    name: str = "RSS",
    category: ArticleCategory = ArticleCategory.MARKET,
    importance: str = "MEDIUM"
) -> RSSAdapter:
    """
    Factory function to create an RSS adapter.

    Args:
        feed_url: RSS feed URL (if empty, uses default feeds)
        name: Feed name
        category: Article category
        importance: Article importance level

    Returns:
        Configured RSSAdapter instance
    """
    feeds = []

    if feed_url:
        feeds.append(RSSFeedConfig(
            url=feed_url,
            name=name,
            category=category,
            importance=importance
        ))

    config = RSSAdapterConfig(name=name)
    return RSSAdapter(feeds=feeds if feeds else None, config=config)
