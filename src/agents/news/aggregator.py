# =============================================================================
# NEWS AGGREGATOR - Multi-Source News Aggregation for Sprint 3
# =============================================================================
"""
Aggregates news from multiple sources into a unified stream.

Sprint 3 Feature: Combines RSS, WebSocket, Twitter, and other sources
into a single real-time news feed for the trading system.

Features:
- Priority-based source ranking
- Cross-source deduplication
- Unified sentiment analysis
- Real-time event publishing via EventBus
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, AsyncIterator, Callable
from enum import Enum, auto
import logging
import asyncio
import hashlib
from collections import OrderedDict

logger = logging.getLogger(__name__)

from .sources.base_adapter import BaseNewsAdapter, NewsArticle, ArticleCategory
from .websocket_feed import WebSocketNewsFeed


class SourcePriority(Enum):
    """Priority for news sources."""
    REALTIME = 1      # WebSocket, direct feeds (highest priority)
    PRIMARY = 2       # Major news agencies (Reuters, Bloomberg)
    SECONDARY = 3     # Secondary sources (financial blogs)
    SOCIAL = 4        # Social media (Twitter, Reddit)
    ARCHIVE = 5       # Archival/delayed sources (lowest priority)


@dataclass
class AggregatorConfig:
    """Configuration for news aggregator."""
    # Deduplication
    dedupe_window_hours: int = 24
    similarity_threshold: float = 0.8       # 0-1, for fuzzy deduplication

    # Rate limiting
    max_articles_per_minute: int = 100
    burst_limit: int = 50                   # Max articles in single batch

    # Filtering
    min_importance: str = "LOW"             # Minimum importance level
    asset_filter: List[str] = field(default_factory=list)  # Empty = all assets

    # Sentiment analysis
    auto_analyze_sentiment: bool = True
    sentiment_cache_ttl_min: int = 60

    # EventBus integration
    publish_to_event_bus: bool = True


@dataclass
class SourceRegistration:
    """Registration for a news source."""
    source: Any                             # BaseNewsAdapter or WebSocketNewsFeed
    name: str
    priority: SourcePriority
    enabled: bool = True
    last_fetch: Optional[datetime] = None
    articles_fetched: int = 0
    errors: int = 0


class NewsAggregator:
    """
    Multi-source news aggregator.

    Combines news from multiple sources (RSS, WebSocket, API) into
    a unified stream with deduplication and priority ranking.
    """

    def __init__(
        self,
        config: Optional[AggregatorConfig] = None,
        event_bus: Optional[Any] = None
    ):
        """
        Initialize news aggregator.

        Args:
            config: Aggregator configuration
            event_bus: Optional EventBus for publishing news events
        """
        self.config = config or AggregatorConfig()
        self._event_bus = event_bus
        self._logger = logging.getLogger("news.aggregator")

        # Source registry
        self._sources: Dict[str, SourceRegistration] = {}
        self._sources_by_priority: Dict[SourcePriority, List[str]] = {
            p: [] for p in SourcePriority
        }

        # Deduplication
        self._seen_articles: OrderedDict[str, datetime] = OrderedDict()
        self._title_hashes: OrderedDict[str, str] = OrderedDict()  # For fuzzy dedup

        # Rate limiting
        self._article_times: List[datetime] = []

        # Sentiment cache
        self._sentiment_cache: Dict[str, Dict[str, Any]] = {}

        # Statistics
        self._total_articles: int = 0
        self._dedupe_count: int = 0
        self._start_time: Optional[datetime] = None

        # Running state
        self._is_running: bool = False
        self._stream_task: Optional[asyncio.Task] = None

    def register_source(
        self,
        source: Any,
        name: str,
        priority: SourcePriority = SourcePriority.SECONDARY
    ) -> bool:
        """
        Register a news source.

        Args:
            source: BaseNewsAdapter or WebSocketNewsFeed instance
            name: Unique name for the source
            priority: Source priority level

        Returns:
            True if registration successful
        """
        if name in self._sources:
            self._logger.warning(f"Source {name} already registered")
            return False

        registration = SourceRegistration(
            source=source,
            name=name,
            priority=priority
        )

        self._sources[name] = registration
        self._sources_by_priority[priority].append(name)

        self._logger.info(f"Registered source: {name} (priority: {priority.name})")
        return True

    def unregister_source(self, name: str) -> bool:
        """
        Unregister a news source.

        Args:
            name: Source name

        Returns:
            True if successfully unregistered
        """
        if name not in self._sources:
            return False

        reg = self._sources.pop(name)
        if name in self._sources_by_priority[reg.priority]:
            self._sources_by_priority[reg.priority].remove(name)

        self._logger.info(f"Unregistered source: {name}")
        return True

    async def fetch_all(self) -> List[NewsArticle]:
        """
        Fetch from all registered sources.

        Returns:
            List of deduplicated, prioritized articles
        """
        all_articles: List[tuple] = []  # (priority, article)

        for priority in sorted(SourcePriority, key=lambda x: x.value):
            source_names = self._sources_by_priority.get(priority, [])

            for name in source_names:
                reg = self._sources.get(name)
                if not reg or not reg.enabled:
                    continue

                try:
                    # Fetch from source
                    if isinstance(reg.source, WebSocketNewsFeed):
                        # WebSocket sources are streamed, not fetched
                        continue

                    if hasattr(reg.source, 'fetch_async'):
                        articles = await reg.source.fetch_async()
                    elif hasattr(reg.source, 'fetch'):
                        loop = asyncio.get_event_loop()
                        articles = await loop.run_in_executor(None, reg.source.fetch)
                    else:
                        continue

                    reg.last_fetch = datetime.now()
                    reg.articles_fetched += len(articles)

                    for article in articles:
                        all_articles.append((priority.value, article))

                except Exception as e:
                    self._logger.error(f"Error fetching from {name}: {e}")
                    reg.errors += 1

        # Sort by priority (lower value = higher priority)
        all_articles.sort(key=lambda x: x[0])

        # Deduplicate and filter
        result = []
        for _, article in all_articles:
            if self._should_include(article):
                result.append(article)

        self._total_articles += len(result)
        return result

    async def stream(self) -> AsyncIterator[NewsArticle]:
        """
        Stream news from all sources in real-time.

        Combines RSS polling with WebSocket streaming.

        Yields:
            NewsArticle objects as they arrive
        """
        self._is_running = True
        self._start_time = datetime.now()

        # Start WebSocket streams
        ws_tasks = []
        for name, reg in self._sources.items():
            if isinstance(reg.source, WebSocketNewsFeed) and reg.enabled:
                task = asyncio.create_task(
                    self._stream_websocket(name, reg)
                )
                ws_tasks.append(task)

        # Combined queue for all sources
        queue: asyncio.Queue = asyncio.Queue()

        # Start polling task for RSS/API sources
        poll_task = asyncio.create_task(self._poll_sources(queue))

        # Forward WebSocket articles to queue
        async def ws_to_queue(name: str, reg: SourceRegistration):
            try:
                async for article in reg.source.stream():
                    if self._should_include(article):
                        await queue.put(article)
            except Exception as e:
                self._logger.error(f"WebSocket stream error for {name}: {e}")

        for name, reg in self._sources.items():
            if isinstance(reg.source, WebSocketNewsFeed) and reg.enabled:
                asyncio.create_task(ws_to_queue(name, reg))

        try:
            while self._is_running:
                try:
                    article = await asyncio.wait_for(queue.get(), timeout=1.0)

                    # Analyze sentiment if configured
                    if self.config.auto_analyze_sentiment:
                        article = await self._analyze_sentiment(article)

                    # Publish to EventBus
                    if self.config.publish_to_event_bus and self._event_bus:
                        await self._publish_article(article)

                    yield article

                except asyncio.TimeoutError:
                    continue

        finally:
            self._is_running = False
            poll_task.cancel()
            for task in ws_tasks:
                task.cancel()

    async def _stream_websocket(
        self,
        name: str,
        reg: SourceRegistration
    ) -> AsyncIterator[NewsArticle]:
        """Stream from a WebSocket source."""
        try:
            async for article in reg.source.stream():
                reg.articles_fetched += 1
                yield article
        except Exception as e:
            self._logger.error(f"WebSocket {name} error: {e}")
            reg.errors += 1

    async def _poll_sources(self, queue: asyncio.Queue) -> None:
        """Poll RSS/API sources periodically."""
        while self._is_running:
            try:
                articles = await self.fetch_all()
                for article in articles:
                    await queue.put(article)
            except Exception as e:
                self._logger.error(f"Poll error: {e}")

            # Wait before next poll (configurable per source in future)
            await asyncio.sleep(60)

    def _should_include(self, article: NewsArticle) -> bool:
        """Check if article should be included (not duplicate, passes filters)."""
        # Check deduplication
        if self._is_duplicate(article):
            self._dedupe_count += 1
            return False

        # Check rate limit
        if not self._check_rate_limit():
            return False

        # Check importance filter
        importance_levels = ['LOW', 'MEDIUM', 'HIGH']
        min_level = importance_levels.index(self.config.min_importance)
        article_level = importance_levels.index(article.importance) if article.importance in importance_levels else 0
        if article_level < min_level:
            return False

        # Check asset filter
        if self.config.asset_filter:
            if not any(asset in article.assets for asset in self.config.asset_filter):
                # Also check title/content
                text = f"{article.title} {article.content}".upper()
                if not any(asset.upper() in text for asset in self.config.asset_filter):
                    return False

        # Mark as seen
        self._mark_seen(article)
        return True

    def _is_duplicate(self, article: NewsArticle) -> bool:
        """Check if article is a duplicate."""
        # Exact ID match
        if article.article_id in self._seen_articles:
            return True

        # Fuzzy title match
        title_hash = hashlib.md5(article.title.lower().encode()).hexdigest()[:8]
        if title_hash in self._title_hashes:
            return True

        return False

    def _mark_seen(self, article: NewsArticle) -> None:
        """Mark article as seen for deduplication."""
        now = datetime.now()

        self._seen_articles[article.article_id] = now
        title_hash = hashlib.md5(article.title.lower().encode()).hexdigest()[:8]
        self._title_hashes[title_hash] = article.article_id

        # Clean old entries
        self._cleanup_seen()

    def _cleanup_seen(self) -> None:
        """Remove old entries from deduplication cache."""
        cutoff = datetime.now() - timedelta(hours=self.config.dedupe_window_hours)

        # Clean seen articles
        while self._seen_articles:
            oldest_id, oldest_time = next(iter(self._seen_articles.items()))
            if oldest_time < cutoff:
                del self._seen_articles[oldest_id]
            else:
                break

        # Clean title hashes
        while self._title_hashes:
            oldest_hash, oldest_id = next(iter(self._title_hashes.items()))
            if oldest_id not in self._seen_articles:
                del self._title_hashes[oldest_hash]
            else:
                break

    def _check_rate_limit(self) -> bool:
        """Check if within rate limits."""
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)

        # Clean old times
        self._article_times = [t for t in self._article_times if t > minute_ago]

        # Check limit
        if len(self._article_times) >= self.config.max_articles_per_minute:
            return False

        self._article_times.append(now)
        return True

    async def _analyze_sentiment(self, article: NewsArticle) -> NewsArticle:
        """Analyze sentiment for an article."""
        try:
            from src.agents import create_sentiment_analyzer
            if create_sentiment_analyzer is None:
                return article

            # Check cache
            if article.article_id in self._sentiment_cache:
                cached = self._sentiment_cache[article.article_id]
                if (datetime.now() - cached['time']).total_seconds() < self.config.sentiment_cache_ttl_min * 60:
                    article.sentiment_score = cached['score']
                    article.sentiment_label = cached['label']
                    return article

            # Analyze
            analyzer = create_sentiment_analyzer()
            text = article.title + " " + (article.summary or article.content[:500])
            result = analyzer.analyze(text)

            article.sentiment_score = result.score
            article.sentiment_label = result.category.name if hasattr(result.category, 'name') else str(result.category)

            # Cache result
            self._sentiment_cache[article.article_id] = {
                'score': result.score,
                'label': article.sentiment_label,
                'time': datetime.now()
            }

            return article

        except Exception as e:
            self._logger.debug(f"Sentiment analysis error: {e}")
            return article

    async def _publish_article(self, article: NewsArticle) -> None:
        """Publish article to EventBus."""
        try:
            from src.agents.events import EventType, AgentEvent

            event = AgentEvent(
                event_type=EventType.REALTIME_NEWS,
                source_agent="NewsAggregator",
                payload=article.to_dict()
            )

            if hasattr(self._event_bus, 'publish_async'):
                await self._event_bus.publish_async(EventType.REALTIME_NEWS, event)
            elif hasattr(self._event_bus, 'publish'):
                self._event_bus.publish(EventType.REALTIME_NEWS, event)

        except Exception as e:
            self._logger.debug(f"EventBus publish error: {e}")

    def stop(self) -> None:
        """Stop the aggregator."""
        self._is_running = False

    def get_status(self) -> Dict[str, Any]:
        """Get aggregator status."""
        source_status = {}
        for name, reg in self._sources.items():
            source_status[name] = {
                'priority': reg.priority.name,
                'enabled': reg.enabled,
                'articles_fetched': reg.articles_fetched,
                'errors': reg.errors,
                'last_fetch': reg.last_fetch.isoformat() if reg.last_fetch else None
            }

        return {
            'is_running': self._is_running,
            'total_sources': len(self._sources),
            'total_articles': self._total_articles,
            'dedupe_count': self._dedupe_count,
            'seen_cache_size': len(self._seen_articles),
            'uptime_sec': (
                (datetime.now() - self._start_time).total_seconds()
                if self._start_time else 0
            ),
            'sources': source_status
        }


def create_news_aggregator(
    event_bus: Optional[Any] = None,
    asset_filter: Optional[List[str]] = None
) -> NewsAggregator:
    """
    Factory function to create a news aggregator.

    Args:
        event_bus: Optional EventBus for publishing events
        asset_filter: Optional list of assets to filter for

    Returns:
        Configured NewsAggregator instance
    """
    config = AggregatorConfig(
        asset_filter=asset_filter or ['XAUUSD']
    )

    return NewsAggregator(config=config, event_bus=event_bus)
