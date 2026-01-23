# =============================================================================
# TWITTER/X ADAPTER - Real-time Social Media News for Sprint 3
# =============================================================================
"""
Twitter/X API adapter for real-time financial news from social media.

Sprint 3 Feature: Monitors key financial accounts and keywords for
breaking news that affects forex/commodities markets.

Key accounts monitored:
- @federalreserve - Fed announcements
- @ecb - ECB announcements
- @Reuters, @Bloomberg - Breaking news
- @ForexLive - Forex analysis

Requires Twitter API v2 access (Basic tier minimum).

Usage:
    adapter = TwitterAdapter(api_key="...", api_secret="...")
    articles = await adapter.fetch_async()
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, AsyncIterator
import logging
import asyncio
import re

from .base_adapter import (
    BaseNewsAdapter,
    AdapterConfig,
    NewsArticle,
    ArticleSource,
    ArticleCategory
)

logger = logging.getLogger(__name__)

# Try to import httpx for async HTTP (optional)
try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False
    logger.warning("httpx not installed, Twitter adapter will use sync requests")

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


@dataclass
class TwitterAdapterConfig(AdapterConfig):
    """Configuration for Twitter adapter."""
    name: str = "Twitter"

    # API credentials
    api_key: str = ""
    api_secret: str = ""
    bearer_token: str = ""

    # API settings
    base_url: str = "https://api.twitter.com/2"
    max_results: int = 100

    # Accounts to monitor (handles without @)
    priority_accounts: List[str] = field(default_factory=lambda: [
        "federalreserve",
        "ecb",
        "bankofengland",
        "Reuters",
        "Bloomberg",
        "ForexLive",
        "zaborit",
        "faborit"
    ])

    # Keywords to track
    keywords: List[str] = field(default_factory=lambda: [
        "XAUUSD", "gold", "forex",
        "Fed", "FOMC", "interest rate",
        "inflation", "CPI", "NFP", "jobs report",
        "ECB", "BOE", "monetary policy",
        "USD", "EUR", "GBP"
    ])

    # Filters
    min_followers: int = 10000  # Ignore low-follower accounts
    exclude_retweets: bool = True
    languages: List[str] = field(default_factory=lambda: ["en"])


class TwitterAdapter(BaseNewsAdapter):
    """
    Twitter/X API adapter for financial news.

    Uses Twitter API v2 to fetch tweets from priority accounts
    and track relevant keywords.
    """

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        bearer_token: str = "",
        config: Optional[TwitterAdapterConfig] = None
    ):
        """
        Initialize Twitter adapter.

        Args:
            api_key: Twitter API key
            api_secret: Twitter API secret
            bearer_token: OAuth 2.0 bearer token (preferred)
            config: Adapter configuration
        """
        self.twitter_config = config or TwitterAdapterConfig()
        super().__init__(self.twitter_config)

        # Set credentials
        self.twitter_config.api_key = api_key or self.twitter_config.api_key
        self.twitter_config.api_secret = api_secret or self.twitter_config.api_secret
        self.twitter_config.bearer_token = bearer_token or self.twitter_config.bearer_token

        # HTTP client
        self._session: Optional[Any] = None

        # Account ID cache
        self._account_ids: Dict[str, str] = {}

        # Statistics
        self._tweets_fetched: int = 0
        self._api_calls: int = 0

    def _get_headers(self) -> Dict[str, str]:
        """Get authorization headers."""
        return {
            "Authorization": f"Bearer {self.twitter_config.bearer_token}",
            "Content-Type": "application/json"
        }

    async def _get_user_id(self, username: str) -> Optional[str]:
        """Get Twitter user ID from username."""
        if username in self._account_ids:
            return self._account_ids[username]

        url = f"{self.twitter_config.base_url}/users/by/username/{username}"

        try:
            if HAS_HTTPX:
                async with httpx.AsyncClient() as client:
                    response = await client.get(url, headers=self._get_headers())
                    if response.status_code == 200:
                        data = response.json()
                        user_id = data.get('data', {}).get('id')
                        if user_id:
                            self._account_ids[username] = user_id
                            return user_id
        except Exception as e:
            self._logger.debug(f"Error getting user ID for {username}: {e}")

        return None

    def fetch(self) -> List[NewsArticle]:
        """
        Fetch tweets synchronously.

        Returns:
            List of NewsArticle objects from tweets
        """
        if not HAS_REQUESTS:
            self._logger.error("requests library not installed")
            return []

        if not self.twitter_config.bearer_token:
            self._logger.warning("Twitter bearer token not configured")
            return []

        if self._should_rate_limit():
            self._logger.debug("Rate limited, skipping fetch")
            return []

        articles = []

        try:
            # Search for keywords
            query = self._build_search_query()
            url = f"{self.twitter_config.base_url}/tweets/search/recent"
            params = {
                "query": query,
                "max_results": min(self.twitter_config.max_results, 100),
                "tweet.fields": "created_at,author_id,public_metrics,entities",
                "expansions": "author_id",
                "user.fields": "name,username,verified,public_metrics"
            }

            response = requests.get(
                url,
                headers=self._get_headers(),
                params=params,
                timeout=30
            )

            self._api_calls += 1
            self._record_request()

            if response.status_code == 200:
                data = response.json()
                articles = self._parse_response(data)
            elif response.status_code == 429:
                self._logger.warning("Twitter API rate limited")
            else:
                self._logger.error(f"Twitter API error: {response.status_code}")

        except Exception as e:
            self._logger.error(f"Twitter fetch error: {e}")

        self._tweets_fetched += len(articles)
        return self._filter_articles(articles)

    async def fetch_async(self) -> List[NewsArticle]:
        """
        Fetch tweets asynchronously.

        Returns:
            List of NewsArticle objects from tweets
        """
        if not HAS_HTTPX:
            # Fallback to sync
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.fetch)

        if not self.twitter_config.bearer_token:
            self._logger.warning("Twitter bearer token not configured")
            return []

        if self._should_rate_limit():
            self._logger.debug("Rate limited, skipping fetch")
            return []

        articles = []

        try:
            query = self._build_search_query()
            url = f"{self.twitter_config.base_url}/tweets/search/recent"
            params = {
                "query": query,
                "max_results": min(self.twitter_config.max_results, 100),
                "tweet.fields": "created_at,author_id,public_metrics,entities",
                "expansions": "author_id",
                "user.fields": "name,username,verified,public_metrics"
            }

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    headers=self._get_headers(),
                    params=params,
                    timeout=30.0
                )

                self._api_calls += 1
                self._record_request()

                if response.status_code == 200:
                    data = response.json()
                    articles = self._parse_response(data)
                elif response.status_code == 429:
                    self._logger.warning("Twitter API rate limited")
                else:
                    self._logger.error(f"Twitter API error: {response.status_code}")

        except Exception as e:
            self._logger.error(f"Twitter async fetch error: {e}")

        self._tweets_fetched += len(articles)
        return self._filter_articles(articles)

    async def stream(self) -> AsyncIterator[NewsArticle]:
        """
        Stream tweets in real-time using filtered stream.

        Note: Requires elevated API access for streaming.
        Falls back to polling if streaming not available.

        Yields:
            NewsArticle objects as tweets arrive
        """
        # Try to use filtered stream (requires elevated access)
        stream_url = f"{self.twitter_config.base_url}/tweets/search/stream"

        if HAS_HTTPX and self.twitter_config.bearer_token:
            try:
                async with httpx.AsyncClient(timeout=None) as client:
                    async with client.stream(
                        "GET",
                        stream_url,
                        headers=self._get_headers()
                    ) as response:
                        if response.status_code == 200:
                            async for line in response.aiter_lines():
                                if line:
                                    article = self._parse_stream_line(line)
                                    if article and not self._is_duplicate(article):
                                        self._mark_seen(article)
                                        yield article
                        else:
                            self._logger.warning(
                                f"Streaming not available ({response.status_code}), falling back to polling"
                            )
                            # Fall through to polling
            except Exception as e:
                self._logger.warning(f"Stream error: {e}, falling back to polling")

        # Fallback to polling
        async for article in super().stream():
            yield article

    def _build_search_query(self) -> str:
        """Build Twitter search query from config."""
        parts = []

        # Add keywords (OR)
        if self.twitter_config.keywords:
            keyword_query = " OR ".join(f'"{kw}"' for kw in self.twitter_config.keywords[:10])
            parts.append(f"({keyword_query})")

        # Add priority accounts (from:)
        if self.twitter_config.priority_accounts:
            accounts = self.twitter_config.priority_accounts[:5]  # API limit
            account_query = " OR ".join(f"from:{acc}" for acc in accounts)
            parts.append(f"({account_query})")

        # Filters
        if self.twitter_config.exclude_retweets:
            parts.append("-is:retweet")

        if self.twitter_config.languages:
            lang = self.twitter_config.languages[0]
            parts.append(f"lang:{lang}")

        return " ".join(parts)

    def _parse_response(self, data: Dict[str, Any]) -> List[NewsArticle]:
        """Parse Twitter API response into NewsArticle objects."""
        articles = []

        tweets = data.get('data', [])
        users = {u['id']: u for u in data.get('includes', {}).get('users', [])}

        for tweet in tweets:
            try:
                author_id = tweet.get('author_id', '')
                author = users.get(author_id, {})

                # Skip low-follower accounts
                followers = author.get('public_metrics', {}).get('followers_count', 0)
                if followers < self.twitter_config.min_followers:
                    continue

                # Create article
                article = NewsArticle(
                    article_id=NewsArticle.generate_id("Twitter", tweet.get('id', ''), ""),
                    source_name=f"Twitter/@{author.get('username', 'unknown')}",
                    source_type=ArticleSource.TWITTER,
                    title=tweet.get('text', '')[:100],  # First 100 chars as title
                    content=tweet.get('text', ''),
                    url=f"https://twitter.com/{author.get('username', 'i')}/status/{tweet.get('id', '')}",
                    published_at=datetime.fromisoformat(
                        tweet.get('created_at', '').replace('Z', '+00:00')
                    ) if tweet.get('created_at') else datetime.now(),
                    category=self._categorize_tweet(tweet.get('text', '')),
                    assets=self._extract_assets(tweet.get('text', '')),
                    keywords=self._extract_keywords(tweet.get('text', '')),
                    importance=self._assess_importance(tweet, author)
                )

                articles.append(article)

            except Exception as e:
                self._logger.debug(f"Error parsing tweet: {e}")

        return articles

    def _parse_stream_line(self, line: str) -> Optional[NewsArticle]:
        """Parse a line from the streaming API."""
        try:
            import json
            data = json.loads(line)

            if 'data' in data:
                tweet = data['data']
                return NewsArticle(
                    article_id=NewsArticle.generate_id("Twitter", tweet.get('id', ''), ""),
                    source_name="Twitter/Stream",
                    source_type=ArticleSource.TWITTER,
                    title=tweet.get('text', '')[:100],
                    content=tweet.get('text', ''),
                    published_at=datetime.now(),
                    category=self._categorize_tweet(tweet.get('text', '')),
                    importance="MEDIUM"
                )
        except Exception as e:
            self._logger.debug(f"Error parsing stream line: {e}")

        return None

    def _categorize_tweet(self, text: str) -> ArticleCategory:
        """Categorize tweet based on content."""
        text_lower = text.lower()

        if any(kw in text_lower for kw in ['fed', 'fomc', 'powell', 'rate decision']):
            return ArticleCategory.CENTRAL_BANK
        elif any(kw in text_lower for kw in ['ecb', 'lagarde', 'boe', 'bailey']):
            return ArticleCategory.CENTRAL_BANK
        elif any(kw in text_lower for kw in ['cpi', 'nfp', 'gdp', 'jobs', 'employment', 'inflation']):
            return ArticleCategory.ECONOMIC
        elif any(kw in text_lower for kw in ['war', 'sanction', 'geopolitical', 'crisis']):
            return ArticleCategory.GEOPOLITICAL
        else:
            return ArticleCategory.MARKET

    def _extract_assets(self, text: str) -> List[str]:
        """Extract asset mentions from tweet."""
        assets = []
        text_upper = text.upper()

        asset_patterns = [
            'XAUUSD', 'GOLD', 'XAU',
            'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF',
            'EUR', 'USD', 'GBP', 'JPY', 'CHF',
            'DXY', 'DOLLAR'
        ]

        for asset in asset_patterns:
            if asset in text_upper:
                assets.append(asset)

        return list(set(assets))

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from tweet."""
        keywords = []
        text_lower = text.lower()

        keyword_patterns = [
            'rate hike', 'rate cut', 'hawkish', 'dovish',
            'inflation', 'deflation', 'recession',
            'bull', 'bear', 'breakout', 'support', 'resistance'
        ]

        for kw in keyword_patterns:
            if kw in text_lower:
                keywords.append(kw)

        # Extract hashtags
        hashtags = re.findall(r'#(\w+)', text)
        keywords.extend(hashtags[:5])

        return keywords

    def _assess_importance(self, tweet: Dict, author: Dict) -> str:
        """Assess tweet importance based on metrics and author."""
        # Check if from priority account
        username = author.get('username', '').lower()
        is_priority = username in [a.lower() for a in self.twitter_config.priority_accounts]

        # Check engagement
        metrics = tweet.get('public_metrics', {})
        retweets = metrics.get('retweet_count', 0)
        likes = metrics.get('like_count', 0)

        # Check if verified
        is_verified = author.get('verified', False)

        # Score
        score = 0
        if is_priority:
            score += 3
        if is_verified:
            score += 2
        if retweets > 100:
            score += 2
        elif retweets > 10:
            score += 1
        if likes > 500:
            score += 1

        if score >= 5:
            return "HIGH"
        elif score >= 2:
            return "MEDIUM"
        else:
            return "LOW"

    def get_status(self) -> Dict[str, Any]:
        """Get adapter status."""
        base_status = super().get_status()
        base_status.update({
            'tweets_fetched': self._tweets_fetched,
            'api_calls': self._api_calls,
            'monitored_accounts': len(self.twitter_config.priority_accounts),
            'tracked_keywords': len(self.twitter_config.keywords),
            'has_bearer_token': bool(self.twitter_config.bearer_token)
        })
        return base_status


def create_twitter_adapter(
    bearer_token: str = "",
    priority_accounts: Optional[List[str]] = None
) -> TwitterAdapter:
    """
    Factory function to create a Twitter adapter.

    Args:
        bearer_token: Twitter API bearer token
        priority_accounts: Optional list of accounts to monitor

    Returns:
        Configured TwitterAdapter instance
    """
    import os

    # Try to get token from environment
    token = bearer_token or os.environ.get('TWITTER_BEARER_TOKEN', '')

    config = TwitterAdapterConfig()
    if priority_accounts:
        config.priority_accounts = priority_accounts

    return TwitterAdapter(bearer_token=token, config=config)
