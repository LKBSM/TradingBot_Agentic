# =============================================================================
# DATA SOURCES CONFIGURATION - Sprint 3 Real-time Data Sources
# =============================================================================
"""
Configuration for all external data sources used by the trading bot.

This file defines:
- API endpoints and credentials
- Rate limits and retry policies
- Source priorities and fallback chains
- WebSocket connection settings

SECURITY NOTE:
- API keys should be stored in environment variables or .env file
- Never commit actual API keys to version control
- Use the get_api_key() helper to retrieve credentials safely

Usage:
    from src.agents.data_sources_config import DataSourcesConfig, get_api_key

    config = DataSourcesConfig()
    api_key = get_api_key('NEWS_API_KEY')
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum, auto
import os
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# SECURITY: API KEY MANAGEMENT
# =============================================================================

def get_api_key(key_name: str, default: str = "") -> str:
    """
    Safely retrieve an API key from environment variables.

    Args:
        key_name: Environment variable name
        default: Default value if not found

    Returns:
        API key value or default
    """
    value = os.environ.get(key_name, default)
    if not value and not default:
        logger.warning(f"API key {key_name} not found in environment")
    return value


def validate_api_keys() -> Dict[str, bool]:
    """
    Validate that required API keys are present.

    Returns:
        Dictionary mapping key names to availability status
    """
    required_keys = [
        'NEWS_API_KEY',
        'TWITTER_API_KEY',
        'TWITTER_API_SECRET',
        'ALPHA_VANTAGE_KEY',
        'FINNHUB_API_KEY',
    ]

    status = {}
    for key in required_keys:
        value = os.environ.get(key, '')
        status[key] = bool(value and len(value) > 10)

    return status


# =============================================================================
# ENUMS
# =============================================================================

class DataSourceType(Enum):
    """Types of data sources."""
    NEWS_API = auto()           # News API services
    RSS_FEED = auto()           # RSS/Atom feeds
    WEBSOCKET = auto()          # Real-time WebSocket
    REST_API = auto()           # REST API polling
    TWITTER = auto()            # Twitter/X API
    ECONOMIC = auto()           # Economic data (Fed, CFTC)


class DataSourceStatus(Enum):
    """Status of a data source."""
    ACTIVE = auto()             # Working normally
    DEGRADED = auto()           # Working with issues
    UNAVAILABLE = auto()        # Not working
    MAINTENANCE = auto()        # Scheduled maintenance
    RATE_LIMITED = auto()       # Hit rate limits


# =============================================================================
# SOURCE CONFIGURATIONS
# =============================================================================

@dataclass
class RateLimitConfig:
    """Rate limit configuration for an API."""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_limit: int = 10
    retry_after_sec: int = 60


@dataclass
class RetryConfig:
    """Retry configuration for failed requests."""
    max_retries: int = 3
    base_delay_sec: float = 1.0
    max_delay_sec: float = 60.0
    exponential_backoff: bool = True
    retry_on_status_codes: List[int] = field(default_factory=lambda: [429, 500, 502, 503, 504])


@dataclass
class NewsAPIConfig:
    """Configuration for NewsAPI.org."""
    enabled: bool = True
    api_key_env: str = "NEWS_API_KEY"
    base_url: str = "https://newsapi.org/v2"

    # Endpoints
    everything_endpoint: str = "/everything"
    top_headlines_endpoint: str = "/top-headlines"

    # Default parameters
    language: str = "en"
    sort_by: str = "publishedAt"
    page_size: int = 100

    # Rate limits (free tier)
    rate_limit: RateLimitConfig = field(default_factory=lambda: RateLimitConfig(
        requests_per_minute=10,
        requests_per_hour=100,
        requests_per_day=500
    ))

    retry: RetryConfig = field(default_factory=RetryConfig)


@dataclass
class FinnhubConfig:
    """Configuration for Finnhub API."""
    enabled: bool = True
    api_key_env: str = "FINNHUB_API_KEY"
    base_url: str = "https://finnhub.io/api/v1"
    websocket_url: str = "wss://ws.finnhub.io"

    # Endpoints
    news_endpoint: str = "/news"
    forex_endpoint: str = "/forex/candle"
    economic_calendar_endpoint: str = "/calendar/economic"

    # Rate limits (free tier)
    rate_limit: RateLimitConfig = field(default_factory=lambda: RateLimitConfig(
        requests_per_minute=60,
        requests_per_hour=3600
    ))

    retry: RetryConfig = field(default_factory=RetryConfig)


@dataclass
class TwitterConfig:
    """Configuration for Twitter/X API."""
    enabled: bool = False  # Disabled by default (requires paid API)
    api_key_env: str = "TWITTER_API_KEY"
    api_secret_env: str = "TWITTER_API_SECRET"
    bearer_token_env: str = "TWITTER_BEARER_TOKEN"

    base_url: str = "https://api.twitter.com/2"
    stream_url: str = "https://api.twitter.com/2/tweets/search/stream"

    # Search parameters
    max_results: int = 100

    # Accounts to follow for forex news
    forex_accounts: List[str] = field(default_factory=lambda: [
        "Reuters", "Bloomberg", "ForexLive", "Faborbit",
        "federalreserve", "ecaborit", "bankofengland"
    ])

    # Keywords to track
    track_keywords: List[str] = field(default_factory=lambda: [
        "XAUUSD", "gold", "forex", "Fed", "ECB", "interest rate",
        "inflation", "NFP", "CPI", "GDP"
    ])

    rate_limit: RateLimitConfig = field(default_factory=lambda: RateLimitConfig(
        requests_per_minute=300,
        requests_per_hour=18000
    ))

    retry: RetryConfig = field(default_factory=RetryConfig)


@dataclass
class RSSFeedConfig:
    """Configuration for RSS feeds."""
    enabled: bool = True

    # Default feeds
    feeds: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        'reuters_markets': {
            'url': 'https://www.reutersagency.com/feed/?best-topics=business-finance',
            'priority': 1,
            'category': 'MARKET'
        },
        'forex_factory': {
            'url': 'https://www.forexfactory.com/ffcal_week_this.xml',
            'priority': 1,
            'category': 'ECONOMIC'
        },
        'investing_news': {
            'url': 'https://www.investing.com/rss/news.rss',
            'priority': 2,
            'category': 'MARKET'
        },
        'ecb_press': {
            'url': 'https://www.ecb.europa.eu/rss/press.html',
            'priority': 1,
            'category': 'CENTRAL_BANK'
        },
        'fed_speeches': {
            'url': 'https://www.federalreserve.gov/feeds/speeches.xml',
            'priority': 1,
            'category': 'CENTRAL_BANK'
        }
    })

    # Polling interval in seconds
    poll_interval_sec: int = 60

    rate_limit: RateLimitConfig = field(default_factory=lambda: RateLimitConfig(
        requests_per_minute=30,
        requests_per_hour=1800
    ))


@dataclass
class CMEFedWatchConfig:
    """Configuration for CME FedWatch tool."""
    enabled: bool = True
    base_url: str = "https://www.cmegroup.com/markets/interest-rates/cme-fedwatch-tool.html"

    # Note: CME doesn't have public API, requires scraping or third-party
    data_provider: str = "finnhub"  # or "manual" for web scraping

    # Update interval (Fed meetings are infrequent)
    poll_interval_hours: int = 4


@dataclass
class CFTCCOTConfig:
    """Configuration for CFTC Commitment of Traders reports."""
    enabled: bool = True
    base_url: str = "https://www.cftc.gov/dea/options/financial_lof.htm"

    # Alternative data providers
    quandl_dataset: str = "CFTC/GC_F_ALL"  # Gold futures

    # COT reports are weekly (Tuesday data, Friday release)
    check_day: str = "Friday"
    check_hour: int = 15  # 3 PM EST


@dataclass
class AlphaVantageConfig:
    """Configuration for Alpha Vantage API."""
    enabled: bool = True
    api_key_env: str = "ALPHA_VANTAGE_KEY"
    base_url: str = "https://www.alphavantage.co/query"

    # Functions
    fx_daily: str = "FX_DAILY"
    fx_intraday: str = "FX_INTRADAY"
    news_sentiment: str = "NEWS_SENTIMENT"

    # Rate limits (free tier: 5 calls/minute, 500/day)
    rate_limit: RateLimitConfig = field(default_factory=lambda: RateLimitConfig(
        requests_per_minute=5,
        requests_per_hour=300,
        requests_per_day=500
    ))

    retry: RetryConfig = field(default_factory=RetryConfig)


# =============================================================================
# MAIN CONFIGURATION
# =============================================================================

@dataclass
class DataSourcesConfig:
    """
    Main configuration class for all data sources.

    Usage:
        config = DataSourcesConfig()
        if config.news_api.enabled:
            api_key = get_api_key(config.news_api.api_key_env)
    """
    # News sources
    news_api: NewsAPIConfig = field(default_factory=NewsAPIConfig)
    finnhub: FinnhubConfig = field(default_factory=FinnhubConfig)
    twitter: TwitterConfig = field(default_factory=TwitterConfig)
    rss_feeds: RSSFeedConfig = field(default_factory=RSSFeedConfig)

    # Economic data sources
    cme_fedwatch: CMEFedWatchConfig = field(default_factory=CMEFedWatchConfig)
    cftc_cot: CFTCCOTConfig = field(default_factory=CFTCCOTConfig)

    # Market data sources
    alpha_vantage: AlphaVantageConfig = field(default_factory=AlphaVantageConfig)

    # Global settings
    default_timeout_sec: float = 30.0
    max_concurrent_requests: int = 10
    cache_ttl_minutes: int = 5

    # Asset focus
    primary_assets: List[str] = field(default_factory=lambda: ['XAUUSD'])
    secondary_assets: List[str] = field(default_factory=lambda: [
        'EURUSD', 'GBPUSD', 'USDJPY', 'DXY'
    ])

    def get_enabled_sources(self) -> List[str]:
        """Get list of enabled data sources."""
        enabled = []
        if self.news_api.enabled:
            enabled.append('news_api')
        if self.finnhub.enabled:
            enabled.append('finnhub')
        if self.twitter.enabled:
            enabled.append('twitter')
        if self.rss_feeds.enabled:
            enabled.append('rss_feeds')
        if self.cme_fedwatch.enabled:
            enabled.append('cme_fedwatch')
        if self.cftc_cot.enabled:
            enabled.append('cftc_cot')
        if self.alpha_vantage.enabled:
            enabled.append('alpha_vantage')
        return enabled

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'enabled_sources': self.get_enabled_sources(),
            'primary_assets': self.primary_assets,
            'secondary_assets': self.secondary_assets,
            'default_timeout_sec': self.default_timeout_sec,
            'max_concurrent_requests': self.max_concurrent_requests,
            'cache_ttl_minutes': self.cache_ttl_minutes,
            'api_keys_status': validate_api_keys()
        }


# =============================================================================
# SAMPLE .ENV FILE TEMPLATE
# =============================================================================

ENV_TEMPLATE = """
# =============================================================================
# TRADING BOT - DATA SOURCES CONFIGURATION
# =============================================================================
# Copy this to .env file and fill in your API keys

# News API (https://newsapi.org)
NEWS_API_KEY=your_news_api_key_here

# Finnhub (https://finnhub.io)
FINNHUB_API_KEY=your_finnhub_key_here

# Twitter/X API (https://developer.twitter.com)
TWITTER_API_KEY=your_twitter_api_key
TWITTER_API_SECRET=your_twitter_api_secret
TWITTER_BEARER_TOKEN=your_twitter_bearer_token

# Alpha Vantage (https://www.alphavantage.co)
ALPHA_VANTAGE_KEY=your_alpha_vantage_key

# =============================================================================
# OPTIONAL: Premium Data Providers
# =============================================================================
# Bloomberg (requires terminal/API subscription)
# BLOOMBERG_API_KEY=

# Reuters Eikon (requires subscription)
# REUTERS_API_KEY=
"""


def generate_env_template() -> str:
    """Generate .env template file content."""
    return ENV_TEMPLATE.strip()


def check_configuration() -> Dict[str, Any]:
    """
    Check the current configuration status.

    Returns:
        Dictionary with configuration status and recommendations
    """
    config = DataSourcesConfig()
    api_status = validate_api_keys()

    issues = []
    recommendations = []

    # Check API keys
    for key, available in api_status.items():
        if not available:
            issues.append(f"Missing API key: {key}")

    # Check enabled sources vs available keys
    if config.news_api.enabled and not api_status.get('NEWS_API_KEY', False):
        recommendations.append("NEWS_API_KEY required for NewsAPI")

    if config.finnhub.enabled and not api_status.get('FINNHUB_API_KEY', False):
        recommendations.append("FINNHUB_API_KEY required for Finnhub")

    if config.twitter.enabled:
        twitter_keys = ['TWITTER_API_KEY', 'TWITTER_API_SECRET']
        if not all(api_status.get(k, False) for k in twitter_keys):
            recommendations.append("Twitter API keys required for Twitter source")

    return {
        'config': config.to_dict(),
        'api_status': api_status,
        'issues': issues,
        'recommendations': recommendations,
        'status': 'OK' if not issues else 'INCOMPLETE'
    }
