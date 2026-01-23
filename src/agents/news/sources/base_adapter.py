# =============================================================================
# BASE NEWS ADAPTER - Abstract Foundation for Data Source Adapters
# =============================================================================
"""
Abstract base class for all news data source adapters.

Each adapter must implement:
- fetch(): Fetch latest news from the source
- fetch_async(): Async version for real-time feeds
- normalize(): Convert source-specific format to NewsArticle

Design Principles:
- Consistent interface across all sources
- Async-first for real-time performance
- Built-in rate limiting and error handling
- Automatic retry with exponential backoff
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, AsyncIterator
from enum import Enum, auto
import logging
import asyncio
import hashlib

logger = logging.getLogger(__name__)


class ArticleSource(Enum):
    """Source type for a news article."""
    RSS = auto()
    TWITTER = auto()
    API = auto()
    WEBSOCKET = auto()
    MANUAL = auto()


class ArticleCategory(Enum):
    """Category of news article."""
    ECONOMIC = auto()       # Economic data releases
    CENTRAL_BANK = auto()   # Central bank news
    GEOPOLITICAL = auto()   # Geopolitical events
    MARKET = auto()         # General market news
    TECHNICAL = auto()      # Technical analysis
    UNKNOWN = auto()


@dataclass
class NewsArticle:
    """
    Normalized news article from any source.

    All adapters produce this format for consistency.
    """
    # Identity
    article_id: str                         # Unique identifier (hash of content)
    source_name: str                        # e.g., "Reuters", "Bloomberg"
    source_type: ArticleSource              # RSS, Twitter, API, etc.

    # Content
    title: str
    content: str = ""                       # Full text (if available)
    summary: str = ""                       # Short summary
    url: str = ""                           # Link to original

    # Metadata
    published_at: datetime = field(default_factory=datetime.now)
    fetched_at: datetime = field(default_factory=datetime.now)
    category: ArticleCategory = ArticleCategory.UNKNOWN

    # Relevance
    assets: List[str] = field(default_factory=list)  # Related assets (XAUUSD, etc.)
    keywords: List[str] = field(default_factory=list)
    importance: str = "LOW"                 # HIGH, MEDIUM, LOW

    # Analysis (filled by sentiment analyzer)
    sentiment_score: Optional[float] = None
    sentiment_label: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'article_id': self.article_id,
            'source_name': self.source_name,
            'source_type': self.source_type.name,
            'title': self.title,
            'content': self.content[:500] if self.content else "",
            'summary': self.summary,
            'url': self.url,
            'published_at': self.published_at.isoformat(),
            'fetched_at': self.fetched_at.isoformat(),
            'category': self.category.name,
            'assets': self.assets,
            'keywords': self.keywords,
            'importance': self.importance,
            'sentiment_score': self.sentiment_score,
            'sentiment_label': self.sentiment_label
        }

    @staticmethod
    def generate_id(source: str, title: str, url: str = "") -> str:
        """Generate unique ID from content."""
        content = f"{source}:{title}:{url}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class AdapterConfig:
    """Configuration for a news adapter."""
    enabled: bool = True
    name: str = "Unknown"

    # Rate limiting
    max_requests_per_minute: int = 30
    min_request_interval_sec: float = 2.0

    # Retry settings
    max_retries: int = 3
    retry_backoff_base: float = 2.0         # Exponential backoff base

    # Filtering
    max_age_hours: int = 24                 # Ignore articles older than this
    keywords: List[str] = field(default_factory=list)  # Filter keywords
    excluded_keywords: List[str] = field(default_factory=list)

    # Cache
    cache_ttl_minutes: int = 5
    dedupe_window_hours: int = 24           # Deduplicate within this window


class BaseNewsAdapter(ABC):
    """
    Abstract base class for news data source adapters.

    Subclasses must implement fetch() and fetch_async() methods.
    """

    def __init__(self, config: Optional[AdapterConfig] = None):
        """
        Initialize the adapter.

        Args:
            config: Adapter configuration
        """
        self.config = config or AdapterConfig()
        self._logger = logging.getLogger(f"news.{self.config.name}")

        # Rate limiting
        self._last_request_time: Optional[datetime] = None
        self._request_count: int = 0
        self._request_count_reset: datetime = datetime.now()

        # Cache
        self._cache: Dict[str, NewsArticle] = {}
        self._cache_time: Optional[datetime] = None

        # Deduplication
        self._seen_ids: Dict[str, datetime] = {}

    @abstractmethod
    def fetch(self) -> List[NewsArticle]:
        """
        Fetch latest news from the source (synchronous).

        Returns:
            List of NewsArticle objects
        """
        pass

    @abstractmethod
    async def fetch_async(self) -> List[NewsArticle]:
        """
        Fetch latest news from the source (asynchronous).

        Returns:
            List of NewsArticle objects
        """
        pass

    async def stream(self) -> AsyncIterator[NewsArticle]:
        """
        Stream news articles as they arrive (for WebSocket sources).

        Default implementation polls fetch_async() periodically.
        Override for true streaming sources.

        Yields:
            NewsArticle objects as they arrive
        """
        while True:
            try:
                articles = await self.fetch_async()
                for article in articles:
                    if not self._is_duplicate(article):
                        self._mark_seen(article)
                        yield article
            except Exception as e:
                self._logger.error(f"Error in stream: {e}")

            # Wait before next poll
            await asyncio.sleep(self.config.min_request_interval_sec)

    def _is_duplicate(self, article: NewsArticle) -> bool:
        """Check if article was already seen."""
        if article.article_id in self._seen_ids:
            return True
        return False

    def _mark_seen(self, article: NewsArticle) -> None:
        """Mark article as seen for deduplication."""
        self._seen_ids[article.article_id] = datetime.now()

        # Clean old entries
        cutoff = datetime.now() - timedelta(hours=self.config.dedupe_window_hours)
        self._seen_ids = {
            k: v for k, v in self._seen_ids.items()
            if v > cutoff
        }

    def _should_rate_limit(self) -> bool:
        """Check if we should rate limit requests."""
        now = datetime.now()

        # Reset counter every minute
        if (now - self._request_count_reset).total_seconds() >= 60:
            self._request_count = 0
            self._request_count_reset = now

        # Check requests per minute
        if self._request_count >= self.config.max_requests_per_minute:
            return True

        # Check minimum interval
        if self._last_request_time:
            elapsed = (now - self._last_request_time).total_seconds()
            if elapsed < self.config.min_request_interval_sec:
                return True

        return False

    def _record_request(self) -> None:
        """Record that a request was made."""
        self._last_request_time = datetime.now()
        self._request_count += 1

    def _filter_articles(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Apply filtering rules to articles."""
        filtered = []
        cutoff = datetime.now() - timedelta(hours=self.config.max_age_hours)

        for article in articles:
            # Age filter
            if article.published_at < cutoff:
                continue

            # Keyword filter (if specified)
            if self.config.keywords:
                text = f"{article.title} {article.content}".lower()
                if not any(kw.lower() in text for kw in self.config.keywords):
                    continue

            # Exclusion filter
            if self.config.excluded_keywords:
                text = f"{article.title} {article.content}".lower()
                if any(kw.lower() in text for kw in self.config.excluded_keywords):
                    continue

            filtered.append(article)

        return filtered

    def get_status(self) -> Dict[str, Any]:
        """Get adapter status."""
        return {
            'name': self.config.name,
            'enabled': self.config.enabled,
            'request_count': self._request_count,
            'cache_size': len(self._cache),
            'seen_ids_count': len(self._seen_ids),
            'last_request': self._last_request_time.isoformat() if self._last_request_time else None
        }
