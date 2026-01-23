# =============================================================================
# NEWS FETCHERS - Headlines from NewsAPI and Central Bank RSS
# =============================================================================
# Fetches forex-related news from free sources:
#   - NewsAPI (free tier: 100 requests/day)
#   - Central Bank RSS feeds (unlimited)
#
# Features:
#   - Rate limiting to stay within free tier limits
#   - Caching to reduce API calls
#   - Keyword filtering for forex relevance
#
# =============================================================================

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import asyncio
import aiohttp
import logging
import os
import xml.etree.ElementTree as ET
from collections import deque
import time
import re

logger = logging.getLogger(__name__)


@dataclass
class NewsItem:
    """
    Represents a news headline.

    Attributes:
        headline: News headline text
        body: Full article body (may be truncated)
        source: News source name
        published_at: Publication timestamp
        sentiment_score: Pre-calculated sentiment (-1 to +1)
        related_currencies: Currencies mentioned
        keywords: Detected keywords
        url: Link to full article
    """
    headline: str
    source: str
    published_at: datetime
    body: str = ""
    sentiment_score: float = 0.0
    related_currencies: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    url: Optional[str] = None
    news_id: str = ""

    def __post_init__(self):
        if not self.news_id:
            self.news_id = f"news_{hash(self.headline)}_{self.published_at.strftime('%Y%m%d%H%M')}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'news_id': self.news_id,
            'headline': self.headline,
            'body': self.body[:500] if self.body else "",  # Truncate
            'source': self.source,
            'published_at': self.published_at.isoformat(),
            'sentiment_score': self.sentiment_score,
            'related_currencies': self.related_currencies,
            'keywords': self.keywords,
            'url': self.url
        }

    def age_hours(self) -> float:
        """Get age of news item in hours."""
        delta = datetime.now() - self.published_at
        return delta.total_seconds() / 3600


class NewsHeadlineFetcher:
    """
    Fetches forex-related news from free APIs.

    Sources:
        - NewsAPI (free tier: 100 requests/day)
        - Central Bank RSS feeds (unlimited)

    Features:
        - Rate limiting
        - Keyword filtering
        - Source credibility weighting
    """

    # === FOREX-RELATED KEYWORDS ===
    FOREX_KEYWORDS = [
        'federal reserve', 'fed', 'fomc', 'interest rate', 'rate hike', 'rate cut',
        'inflation', 'cpi', 'pce', 'gdp', 'unemployment', 'jobs report', 'nfp',
        'central bank', 'monetary policy', 'quantitative easing', 'qe', 'tapering',
        'dollar', 'euro', 'yen', 'pound', 'sterling', 'forex', 'currency',
        'gold', 'xau', 'precious metal', 'safe haven',
        'ecb', 'boe', 'boj', 'rba', 'snb',
        'treasury', 'bond yield', 'yield curve',
        'recession', 'economic growth', 'trade war', 'tariff',
        'powell', 'lagarde', 'bailey', 'ueda',
    ]

    # === CENTRAL BANK RSS FEEDS ===
    CENTRAL_BANK_FEEDS = {
        'federal_reserve': {
            'url': 'https://www.federalreserve.gov/feeds/press_monetary.xml',
            'currency': 'USD',
            'name': 'Federal Reserve'
        },
        'ecb': {
            'url': 'https://www.ecb.europa.eu/rss/press.html',
            'currency': 'EUR',
            'name': 'European Central Bank'
        },
        # Note: Some feeds may require different parsing
    }

    # === SOURCE CREDIBILITY (for weighting) ===
    SOURCE_CREDIBILITY = {
        'federal reserve': 1.0,
        'ecb': 1.0,
        'boe': 1.0,
        'reuters': 0.95,
        'bloomberg': 0.95,
        'wsj': 0.9,
        'financial times': 0.9,
        'cnbc': 0.8,
        'marketwatch': 0.75,
        'default': 0.5
    }

    def __init__(
        self,
        newsapi_key: Optional[str] = None,
        cache_minutes: int = 15,
        max_requests_per_day: int = 90  # Leave buffer for free tier
    ):
        """
        Initialize the news fetcher.

        Args:
            newsapi_key: NewsAPI key (optional, uses env var if not provided)
            cache_minutes: How long to cache news
            max_requests_per_day: Max NewsAPI requests per day
        """
        self._newsapi_key = newsapi_key or os.environ.get('NEWSAPI_KEY')
        self._cache_minutes = cache_minutes
        self._max_requests_per_day = max_requests_per_day

        # Cache
        self._cached_news: List[NewsItem] = []
        self._last_fetch: Optional[datetime] = None
        self._cache_valid_until: Optional[datetime] = None

        # Rate limiting for NewsAPI
        self._newsapi_requests_today = 0
        self._newsapi_day_start: Optional[datetime] = None

        logger.info(
            f"NewsHeadlineFetcher initialized "
            f"(NewsAPI key: {'present' if self._newsapi_key else 'missing'}, "
            f"cache: {cache_minutes}min)"
        )

    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid."""
        if not self._cache_valid_until:
            return False
        return datetime.now() < self._cache_valid_until

    def _can_use_newsapi(self) -> bool:
        """Check if we can make a NewsAPI request."""
        if not self._newsapi_key:
            return False

        # Reset counter at midnight
        now = datetime.now()
        if self._newsapi_day_start is None or now.date() > self._newsapi_day_start.date():
            self._newsapi_requests_today = 0
            self._newsapi_day_start = now

        return self._newsapi_requests_today < self._max_requests_per_day

    def _is_forex_relevant(self, text: str) -> bool:
        """Check if text is relevant to forex trading."""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.FOREX_KEYWORDS)

    def _detect_currencies(self, text: str) -> List[str]:
        """Detect currency mentions in text."""
        currencies = []
        text_lower = text.lower()

        currency_patterns = {
            'USD': ['dollar', 'usd', 'fed', 'federal reserve', 'us economy'],
            'EUR': ['euro', 'eur', 'ecb', 'eurozone'],
            'GBP': ['pound', 'gbp', 'sterling', 'boe', 'bank of england'],
            'JPY': ['yen', 'jpy', 'boj', 'japan'],
            'XAU': ['gold', 'xau', 'precious metal', 'bullion'],
            'CHF': ['franc', 'chf', 'swiss', 'snb'],
        }

        for currency, patterns in currency_patterns.items():
            if any(p in text_lower for p in patterns):
                currencies.append(currency)

        return currencies

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract forex-related keywords from text."""
        text_lower = text.lower()
        return [kw for kw in self.FOREX_KEYWORDS if kw in text_lower]

    def get_source_credibility(self, source: str) -> float:
        """Get credibility weight for a news source."""
        source_lower = source.lower()

        for known_source, credibility in self.SOURCE_CREDIBILITY.items():
            if known_source in source_lower:
                return credibility

        return self.SOURCE_CREDIBILITY['default']

    async def fetch_news_async(
        self,
        max_age_hours: int = 24,
        currencies: Optional[List[str]] = None
    ) -> List[NewsItem]:
        """
        Fetch latest forex news asynchronously.

        Args:
            max_age_hours: Maximum age of news to return
            currencies: Filter by currencies (None = all)

        Returns:
            List of NewsItem objects
        """
        # Check cache
        if self._is_cache_valid():
            news = self._filter_news(self._cached_news, max_age_hours, currencies)
            logger.debug(f"Returning {len(news)} cached news items")
            return news

        all_news = []

        # Fetch from NewsAPI if available
        if self._can_use_newsapi():
            try:
                newsapi_items = await self._fetch_newsapi_async()
                all_news.extend(newsapi_items)
                self._newsapi_requests_today += 1
                logger.info(f"Fetched {len(newsapi_items)} items from NewsAPI")
            except Exception as e:
                logger.error(f"NewsAPI fetch failed: {e}")

        # Fetch from central bank RSS feeds
        try:
            rss_items = await self._fetch_central_bank_rss_async()
            all_news.extend(rss_items)
            logger.info(f"Fetched {len(rss_items)} items from central bank RSS")
        except Exception as e:
            logger.error(f"RSS fetch failed: {e}")

        # Update cache
        if all_news:
            self._cached_news = all_news
            self._last_fetch = datetime.now()
            self._cache_valid_until = datetime.now() + timedelta(minutes=self._cache_minutes)

        return self._filter_news(all_news, max_age_hours, currencies)

    def fetch_news(
        self,
        max_age_hours: int = 24,
        currencies: Optional[List[str]] = None
    ) -> List[NewsItem]:
        """
        Synchronous wrapper for fetch_news_async.

        Args:
            max_age_hours: Maximum age of news to return
            currencies: Filter by currencies

        Returns:
            List of NewsItem objects
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(
                        asyncio.run,
                        self.fetch_news_async(max_age_hours, currencies)
                    )
                    return future.result(timeout=30)
            else:
                return loop.run_until_complete(
                    self.fetch_news_async(max_age_hours, currencies)
                )
        except Exception as e:
            logger.error(f"Sync news fetch failed: {e}")
            return self._filter_news(self._cached_news, max_age_hours, currencies)

    async def _fetch_newsapi_async(self) -> List[NewsItem]:
        """Fetch news from NewsAPI."""
        news_items = []

        if not self._newsapi_key:
            return news_items

        url = "https://newsapi.org/v2/everything"
        params = {
            'q': 'forex OR "federal reserve" OR "interest rate" OR inflation',
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': 20,
            'apiKey': self._newsapi_key
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=10) as response:
                    if response.status != 200:
                        logger.warning(f"NewsAPI returned status {response.status}")
                        return news_items

                    data = await response.json()

                    for article in data.get('articles', []):
                        # Filter for forex relevance
                        headline = article.get('title', '')
                        description = article.get('description', '')

                        if not self._is_forex_relevant(headline + ' ' + description):
                            continue

                        published_str = article.get('publishedAt', '')
                        try:
                            published_at = datetime.fromisoformat(
                                published_str.replace('Z', '+00:00')
                            ).replace(tzinfo=None)
                        except:
                            published_at = datetime.now()

                        news_item = NewsItem(
                            headline=headline,
                            body=description or '',
                            source=article.get('source', {}).get('name', 'Unknown'),
                            published_at=published_at,
                            url=article.get('url'),
                            related_currencies=self._detect_currencies(headline + ' ' + description),
                            keywords=self._extract_keywords(headline + ' ' + description)
                        )
                        news_items.append(news_item)

        except asyncio.TimeoutError:
            logger.error("NewsAPI request timed out")
        except aiohttp.ClientError as e:
            logger.error(f"NewsAPI client error: {e}")

        return news_items

    async def _fetch_central_bank_rss_async(self) -> List[NewsItem]:
        """Fetch news from central bank RSS feeds."""
        news_items = []

        async with aiohttp.ClientSession() as session:
            for feed_id, feed_info in self.CENTRAL_BANK_FEEDS.items():
                try:
                    async with session.get(
                        feed_info['url'],
                        timeout=10
                    ) as response:
                        if response.status != 200:
                            continue

                        content = await response.text()
                        items = self._parse_rss(
                            content,
                            feed_info['name'],
                            feed_info['currency']
                        )
                        news_items.extend(items)

                except Exception as e:
                    logger.warning(f"Failed to fetch {feed_id}: {e}")

        return news_items

    def _parse_rss(
        self,
        content: str,
        source_name: str,
        currency: str
    ) -> List[NewsItem]:
        """Parse RSS feed content."""
        news_items = []

        try:
            root = ET.fromstring(content)

            # Handle different RSS formats
            items = root.findall('.//item') or root.findall('.//{http://www.w3.org/2005/Atom}entry')

            for item in items[:10]:  # Limit to 10 items per feed
                title = (
                    item.findtext('title') or
                    item.findtext('{http://www.w3.org/2005/Atom}title') or
                    ''
                )
                description = (
                    item.findtext('description') or
                    item.findtext('{http://www.w3.org/2005/Atom}summary') or
                    ''
                )
                link = (
                    item.findtext('link') or
                    item.find('{http://www.w3.org/2005/Atom}link')
                )
                if hasattr(link, 'get'):
                    link = link.get('href')

                pub_date_str = (
                    item.findtext('pubDate') or
                    item.findtext('{http://www.w3.org/2005/Atom}published') or
                    item.findtext('{http://www.w3.org/2005/Atom}updated')
                )

                try:
                    # Try to parse various date formats
                    published_at = self._parse_date(pub_date_str)
                except:
                    published_at = datetime.now()

                if title:
                    news_item = NewsItem(
                        headline=title.strip(),
                        body=self._clean_html(description),
                        source=source_name,
                        published_at=published_at,
                        url=link if isinstance(link, str) else None,
                        related_currencies=[currency],
                        keywords=self._extract_keywords(title + ' ' + description)
                    )
                    news_items.append(news_item)

        except ET.ParseError as e:
            logger.warning(f"RSS parse error for {source_name}: {e}")

        return news_items

    def _parse_date(self, date_str: Optional[str]) -> datetime:
        """Parse various date formats from RSS feeds."""
        if not date_str:
            return datetime.now()

        # Try common formats
        formats = [
            '%a, %d %b %Y %H:%M:%S %z',     # RFC 822
            '%a, %d %b %Y %H:%M:%S GMT',
            '%Y-%m-%dT%H:%M:%S%z',          # ISO 8601
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%d %H:%M:%S',
        ]

        for fmt in formats:
            try:
                dt = datetime.strptime(date_str.strip(), fmt)
                return dt.replace(tzinfo=None)
            except ValueError:
                continue

        return datetime.now()

    def _clean_html(self, text: str) -> str:
        """Remove HTML tags from text."""
        if not text:
            return ""
        clean = re.sub(r'<[^>]+>', '', text)
        return clean.strip()

    def _filter_news(
        self,
        news: List[NewsItem],
        max_age_hours: int,
        currencies: Optional[List[str]] = None
    ) -> List[NewsItem]:
        """Filter news by age and currency."""
        cutoff = datetime.now() - timedelta(hours=max_age_hours)

        filtered = [n for n in news if n.published_at >= cutoff]

        if currencies:
            filtered = [
                n for n in filtered
                if any(c in n.related_currencies for c in currencies)
            ]

        # Sort by publication date (newest first)
        filtered.sort(key=lambda x: x.published_at, reverse=True)

        return filtered

    def get_api_usage(self) -> Dict[str, Any]:
        """Get NewsAPI usage statistics."""
        return {
            'requests_today': self._newsapi_requests_today,
            'max_per_day': self._max_requests_per_day,
            'remaining': self._max_requests_per_day - self._newsapi_requests_today,
            'api_key_present': bool(self._newsapi_key)
        }

    def clear_cache(self) -> None:
        """Clear the news cache."""
        self._cached_news = []
        self._last_fetch = None
        self._cache_valid_until = None
        logger.info("News cache cleared")
