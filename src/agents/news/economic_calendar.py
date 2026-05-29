# =============================================================================
# ECONOMIC CALENDAR FETCHER - ForexFactory & Investing.com Integration
# =============================================================================
# Fetches economic event calendars from free sources:
#   - Primary: ForexFactory (web scraping with rate limiting)
#   - Fallback: Investing.com economic calendar
#
# Event Impact Classification:
#   HIGH: FOMC, NFP, CPI, GDP - Major market movers
#   MEDIUM: PMI, Retail Sales - Moderate impact
#   LOW: Minor economic indicators
#
# =============================================================================

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from enum import Enum
from pathlib import Path
import asyncio
import aiohttp
import csv
import hashlib
import logging
import re
from collections import deque
import time

logger = logging.getLogger(__name__)


def _to_float(val: Any) -> Optional[float]:
    """Best-effort parse of FF economic values like '250K', '3.2%', '1.5B'."""
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    mult = 1.0
    s_lower = s.lower().rstrip("%")
    if s_lower.endswith("k"):
        mult, s_lower = 1e3, s_lower[:-1]
    elif s_lower.endswith("m"):
        mult, s_lower = 1e6, s_lower[:-1]
    elif s_lower.endswith("b"):
        mult, s_lower = 1e9, s_lower[:-1]
    elif s_lower.endswith("t"):
        mult, s_lower = 1e12, s_lower[:-1]
    try:
        return float(s_lower) * mult
    except ValueError:
        return None


class NewsImpact(Enum):
    """Impact level of an economic event."""
    HIGH = "high"        # FOMC, NFP, CPI -> BLOCK trading
    MEDIUM = "medium"    # PMI, Retail Sales -> REDUCE position
    LOW = "low"          # Minor data -> Monitor only
    NONE = "none"        # No impact


@dataclass
class EconomicEvent:
    """
    Represents a scheduled economic event.

    Attributes:
        event_id: Unique identifier
        name: Event name (e.g., "Non-Farm Payrolls")
        currency: Affected currency (e.g., "USD")
        impact: Impact level (HIGH, MEDIUM, LOW)
        scheduled_time: When the event occurs
        actual_value: Released value (None if not yet released)
        forecast_value: Market consensus forecast
        previous_value: Previous period's value
    """
    event_id: str
    name: str
    currency: str
    impact: NewsImpact
    scheduled_time: datetime
    actual_value: Optional[float] = None
    forecast_value: Optional[float] = None
    previous_value: Optional[float] = None
    description: str = ""
    source: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'event_id': self.event_id,
            'name': self.name,
            'currency': self.currency,
            'impact': self.impact.value,
            'scheduled_time': self.scheduled_time.isoformat(),
            'actual_value': self.actual_value,
            'forecast_value': self.forecast_value,
            'previous_value': self.previous_value,
            'description': self.description,
            'source': self.source
        }

    def is_released(self) -> bool:
        """Check if the event data has been released."""
        return self.actual_value is not None

    def minutes_until(self) -> float:
        """Get minutes until the event occurs."""
        delta = self.scheduled_time - datetime.now()
        return delta.total_seconds() / 60

    def is_within_window(self, before_minutes: int, after_minutes: int) -> bool:
        """
        Check if current time is within the event window.

        Args:
            before_minutes: Minutes before event to start window
            after_minutes: Minutes after event to end window

        Returns:
            True if within window
        """
        now = datetime.now()
        window_start = self.scheduled_time - timedelta(minutes=before_minutes)
        window_end = self.scheduled_time + timedelta(minutes=after_minutes)
        return window_start <= now <= window_end


class EconomicCalendarFetcher:
    """
    Fetches economic calendar from free sources.

    Primary source: ForexFactory (web scraping)
    Fallback: Pre-defined schedule for major events

    Features:
        - Rate limiting (1 request per 60 seconds)
        - Caching (4-hour cache)
        - Impact classification
        - Currency filtering
    """

    # === HIGH-IMPACT EVENTS (Auto-classified) ===
    HIGH_IMPACT_EVENTS = {
        # US Events
        'non-farm payrolls', 'nfp', 'non farm payrolls',
        'fomc', 'federal reserve', 'fed rate decision', 'fed interest rate',
        'cpi', 'consumer price index', 'core cpi',
        'gdp', 'gross domestic product',
        'unemployment rate', 'unemployment claims',
        'retail sales', 'core retail sales',
        'fed chair', 'powell', 'fed speaks',
        'pce', 'pce price index', 'core pce',
        'ism manufacturing', 'ism services',

        # EU Events
        'ecb', 'ecb rate decision', 'lagarde',
        'german gdp', 'eurozone gdp', 'german cpi',

        # UK Events
        'boe', 'bank of england', 'bailey',
        'uk gdp', 'uk cpi', 'uk unemployment',

        # Other Major
        'boj', 'bank of japan',
        'rba', 'reserve bank of australia',
    }

    # === MEDIUM-IMPACT EVENTS ===
    MEDIUM_IMPACT_EVENTS = {
        'pmi', 'purchasing managers', 'manufacturing pmi', 'services pmi',
        'trade balance', 'industrial production',
        'consumer confidence', 'consumer sentiment',
        'housing starts', 'building permits',
        'durable goods', 'factory orders',
        'jobless claims', 'initial claims',
        'empire state', 'philly fed',
        'zew', 'ifo', 'gfk',
    }

    # === CURRENCY MAPPING ===
    CURRENCY_MAP = {
        'usd': 'USD', 'us': 'USD', 'united states': 'USD', 'american': 'USD',
        'eur': 'EUR', 'eurozone': 'EUR', 'german': 'EUR', 'european': 'EUR',
        'gbp': 'GBP', 'uk': 'GBP', 'british': 'GBP', 'sterling': 'GBP',
        'jpy': 'JPY', 'japan': 'JPY', 'japanese': 'JPY',
        'chf': 'CHF', 'swiss': 'CHF', 'switzerland': 'CHF',
        'aud': 'AUD', 'australia': 'AUD', 'australian': 'AUD',
        'cad': 'CAD', 'canada': 'CAD', 'canadian': 'CAD',
        'nzd': 'NZD', 'new zealand': 'NZD',
    }

    def __init__(
        self,
        cache_hours: int = 4,
        rate_limit_seconds: int = 60,
        user_agent: str = "TradingBot/1.0",
        csv_path: Optional[str] = None,
    ):
        """
        Initialize the calendar fetcher.

        Args:
            cache_hours: How long to cache calendar data
            rate_limit_seconds: Minimum seconds between fetches
            user_agent: User agent for HTTP requests
            csv_path: Optional path to a CSV calendar (columns:
                Date,Currency,Event,Impact,Actual,Forecast,Previous). When set,
                ``fetch_calendar()`` prefers this source over the ForexFactory
                HTML scraper — populate it with ``scripts/fetch_forexfactory_live.py``.
        """
        self._cache_hours = cache_hours
        self._rate_limit_seconds = rate_limit_seconds
        self._user_agent = user_agent
        self._csv_path: Optional[Path] = Path(csv_path) if csv_path else None

        # Cache
        self._cached_events: List[EconomicEvent] = []
        self._last_fetch: Optional[datetime] = None
        self._cache_valid_until: Optional[datetime] = None
        self._csv_mtime: Optional[float] = None

        # Rate limiting
        self._last_request_time: float = 0

        logger.info(
            f"EconomicCalendarFetcher initialized "
            f"(cache: {cache_hours}h, rate limit: {rate_limit_seconds}s, "
            f"csv: {self._csv_path or 'none'})"
        )

    def _classify_impact(self, event_name: str) -> NewsImpact:
        """
        Classify event impact based on name.

        Args:
            event_name: Name of the economic event

        Returns:
            NewsImpact level
        """
        name_lower = event_name.lower()

        # Check high-impact keywords
        for keyword in self.HIGH_IMPACT_EVENTS:
            if keyword in name_lower:
                return NewsImpact.HIGH

        # Check medium-impact keywords
        for keyword in self.MEDIUM_IMPACT_EVENTS:
            if keyword in name_lower:
                return NewsImpact.MEDIUM

        return NewsImpact.LOW

    def _detect_currency(self, event_name: str, default: str = "USD") -> str:
        """
        Detect currency from event name.

        Args:
            event_name: Name of the economic event
            default: Default currency if not detected

        Returns:
            Currency code (e.g., "USD")
        """
        name_lower = event_name.lower()

        for keyword, currency in self.CURRENCY_MAP.items():
            if keyword in name_lower:
                return currency

        return default

    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid."""
        if not self._cache_valid_until:
            return False
        return datetime.now() < self._cache_valid_until

    def _can_make_request(self) -> bool:
        """Check if rate limit allows a new request."""
        elapsed = time.time() - self._last_request_time
        return elapsed >= self._rate_limit_seconds

    async def fetch_calendar_async(
        self,
        days_ahead: int = 7,
        currencies: Optional[List[str]] = None
    ) -> List[EconomicEvent]:
        """
        Fetch economic calendar asynchronously.

        Source priority:
          1. CSV file (if ``csv_path`` was provided) — refreshed by
             ``scripts/fetch_forexfactory_live.py``, reliable and real-time.
          2. ForexFactory HTML scraping (fragile — placeholder timestamps).
          3. Built-in fallback schedule.

        Args:
            days_ahead: How many days ahead to fetch
            currencies: Filter by currencies (None = all)

        Returns:
            List of EconomicEvent objects
        """
        # Source 1: CSV file (preferred when configured)
        if self._csv_path is not None:
            events = self._fetch_from_csv(days_ahead)
            if events:
                return self._filter_events(events, currencies)
            # Only fall through if CSV is empty/missing — log clearly
            logger.warning(
                "CSV calendar %s unavailable or empty, falling back to scraping",
                self._csv_path,
            )

        # Check cache first
        if self._is_cache_valid():
            events = self._filter_events(self._cached_events, currencies)
            logger.debug(f"Returning {len(events)} cached events")
            return events

        # Check rate limit
        if not self._can_make_request():
            wait_time = self._rate_limit_seconds - (time.time() - self._last_request_time)
            logger.warning(f"Rate limited, returning cache. Wait {wait_time:.0f}s")
            return self._filter_events(self._cached_events, currencies)

        # Try to fetch from ForexFactory
        try:
            events = await self._fetch_forexfactory_async(days_ahead)
            if events:
                self._cached_events = events
                self._last_fetch = datetime.now()
                self._cache_valid_until = datetime.now() + timedelta(hours=self._cache_hours)
                self._last_request_time = time.time()
                logger.info(f"Fetched {len(events)} events from ForexFactory")
                return self._filter_events(events, currencies)
        except Exception as e:
            logger.error(f"ForexFactory fetch failed: {e}")

        # Fallback to built-in schedule
        events = self._get_builtin_events(days_ahead)
        logger.info(f"Using {len(events)} built-in events")
        return self._filter_events(events, currencies)

    def _fetch_from_csv(self, days_ahead: int) -> List[EconomicEvent]:
        """Load events from the configured CSV file.

        Expected columns: Date, Currency, Event, Impact, Actual, Forecast, Previous.
        Date is parsed as UTC-naive. Cache is invalidated automatically when the
        file mtime changes — so ``fetch_forexfactory_live.py`` refreshes take
        effect without restarting the bot.
        """
        if self._csv_path is None or not self._csv_path.exists():
            return []

        try:
            mtime = self._csv_path.stat().st_mtime
        except OSError:
            return []

        # Serve from cache if the file hasn't changed since last load.
        if self._csv_mtime == mtime and self._cached_events:
            pass  # fall through and re-apply the date window on cached list
        else:
            events: List[EconomicEvent] = []
            try:
                with self._csv_path.open("r", encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        date_str = (row.get("Date") or "").strip()
                        name = (row.get("Event") or "").strip()
                        currency = (row.get("Currency") or "").strip().upper()
                        if not (date_str and name and currency):
                            continue
                        try:
                            scheduled = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
                        except ValueError:
                            try:
                                scheduled = datetime.fromisoformat(date_str)
                            except ValueError:
                                continue
                        impact_raw = (row.get("Impact") or "").strip().lower()
                        impact = {
                            "high": NewsImpact.HIGH,
                            "medium": NewsImpact.MEDIUM,
                            "low": NewsImpact.LOW,
                        }.get(impact_raw, self._classify_impact(name))

                        eid = hashlib.md5(
                            f"{date_str}|{currency}|{name}".encode("utf-8")
                        ).hexdigest()[:12]

                        events.append(EconomicEvent(
                            event_id=f"csv_{eid}",
                            name=name,
                            currency=currency,
                            impact=impact,
                            scheduled_time=scheduled,
                            actual_value=_to_float(row.get("Actual")),
                            forecast_value=_to_float(row.get("Forecast")),
                            previous_value=_to_float(row.get("Previous")),
                            source="csv",
                        ))
            except OSError as e:
                logger.warning("CSV calendar read failed: %s", e)
                return []

            self._cached_events = events
            self._csv_mtime = mtime
            self._last_fetch = datetime.now()
            logger.info("Loaded %d events from CSV %s", len(events), self._csv_path)

        # Apply forward-looking window so we don't return historical events
        now = datetime.now()
        cutoff = now + timedelta(days=days_ahead)
        return [e for e in self._cached_events if now - timedelta(minutes=60) <= e.scheduled_time <= cutoff]

    def fetch_calendar(
        self,
        days_ahead: int = 7,
        currencies: Optional[List[str]] = None
    ) -> List[EconomicEvent]:
        """
        Synchronous wrapper for fetch_calendar_async.

        Args:
            days_ahead: How many days ahead to fetch
            currencies: Filter by currencies (None = all)

        Returns:
            List of EconomicEvent objects
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If already in async context, use existing loop
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(
                        asyncio.run,
                        self.fetch_calendar_async(days_ahead, currencies)
                    )
                    return future.result(timeout=30)
            else:
                return loop.run_until_complete(
                    self.fetch_calendar_async(days_ahead, currencies)
                )
        except Exception as e:
            logger.error(f"Sync fetch failed: {e}")
            return self._filter_events(
                self._get_builtin_events(days_ahead),
                currencies
            )

    async def _fetch_forexfactory_async(self, days_ahead: int) -> List[EconomicEvent]:
        """
        Fetch calendar from ForexFactory.

        Note: This uses web scraping. Be respectful of rate limits.
        """
        events = []
        url = "https://www.forexfactory.com/calendar"

        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    'User-Agent': self._user_agent,
                    'Accept': 'text/html,application/xhtml+xml'
                }

                async with session.get(url, headers=headers, timeout=10) as response:
                    if response.status != 200:
                        logger.warning(f"ForexFactory returned status {response.status}")
                        return events

                    html = await response.text()
                    events = self._parse_forexfactory_html(html, days_ahead)

        except asyncio.TimeoutError:
            logger.error("ForexFactory request timed out")
        except aiohttp.ClientError as e:
            logger.error(f"ForexFactory client error: {e}")

        return events

    def _parse_forexfactory_html(self, html: str, days_ahead: int) -> List[EconomicEvent]:
        """
        Parse ForexFactory HTML to extract events.

        This is a simplified parser - ForexFactory's HTML structure
        may change, so we use fallback data if parsing fails.
        """
        events = []

        # Simple regex-based extraction (ForexFactory uses specific CSS classes)
        # This is fragile and may need updates if FF changes their layout

        # Look for event rows
        event_pattern = re.compile(
            r'class="calendar__event[^"]*"[^>]*>.*?'
            r'class="calendar__cell calendar__currency[^"]*"[^>]*>([A-Z]{3})</td>.*?'
            r'class="calendar__event-title[^"]*"[^>]*>([^<]+)</span>',
            re.DOTALL
        )

        for match in event_pattern.finditer(html):
            currency = match.group(1).strip()
            event_name = match.group(2).strip()

            if event_name:
                impact = self._classify_impact(event_name)

                # We don't have exact times from this simple parse,
                # so we'll use the built-in schedule as primary
                event = EconomicEvent(
                    event_id=f"ff_{hash(event_name)}",
                    name=event_name,
                    currency=currency,
                    impact=impact,
                    scheduled_time=datetime.now() + timedelta(hours=1),  # Placeholder
                    source="forexfactory"
                )
                events.append(event)

        return events

    def _get_builtin_events(self, days_ahead: int) -> List[EconomicEvent]:
        """
        Get built-in economic events schedule.

        These are major recurring events that we know about in advance.
        Used as fallback when live fetching fails.
        """
        events = []
        now = datetime.now()
        end_date = now + timedelta(days=days_ahead)

        # Major recurring events (approximate schedules)
        recurring_events = [
            # US Events
            {
                'name': 'FOMC Rate Decision',
                'currency': 'USD',
                'impact': NewsImpact.HIGH,
                'description': 'Federal Reserve interest rate decision'
            },
            {
                'name': 'Non-Farm Payrolls',
                'currency': 'USD',
                'impact': NewsImpact.HIGH,
                'description': 'US employment report - first Friday of month'
            },
            {
                'name': 'CPI (Consumer Price Index)',
                'currency': 'USD',
                'impact': NewsImpact.HIGH,
                'description': 'US inflation data'
            },
            {
                'name': 'GDP (Gross Domestic Product)',
                'currency': 'USD',
                'impact': NewsImpact.HIGH,
                'description': 'US economic growth'
            },
            {
                'name': 'Core PCE Price Index',
                'currency': 'USD',
                'impact': NewsImpact.HIGH,
                'description': 'Fed preferred inflation measure'
            },

            # EU Events
            {
                'name': 'ECB Rate Decision',
                'currency': 'EUR',
                'impact': NewsImpact.HIGH,
                'description': 'European Central Bank rate decision'
            },

            # UK Events
            {
                'name': 'BOE Rate Decision',
                'currency': 'GBP',
                'impact': NewsImpact.HIGH,
                'description': 'Bank of England rate decision'
            },
        ]

        # Generate events for the requested period
        # In production, you'd use actual calendar data
        for i, event_data in enumerate(recurring_events):
            # Create a placeholder event (in real usage, fetch actual times)
            event = EconomicEvent(
                event_id=f"builtin_{i}_{now.strftime('%Y%m%d')}",
                name=event_data['name'],
                currency=event_data['currency'],
                impact=event_data['impact'],
                scheduled_time=now + timedelta(days=(i % days_ahead) + 1, hours=14, minutes=30),
                description=event_data['description'],
                source="builtin"
            )
            events.append(event)

        return events

    def _filter_events(
        self,
        events: List[EconomicEvent],
        currencies: Optional[List[str]] = None
    ) -> List[EconomicEvent]:
        """Filter events by currency."""
        if not currencies:
            return events

        return [e for e in events if e.currency in currencies]

    def get_upcoming_high_impact(
        self,
        hours_ahead: int = 24,
        currencies: Optional[List[str]] = None
    ) -> List[EconomicEvent]:
        """
        Get upcoming high-impact events.

        Args:
            hours_ahead: How many hours to look ahead
            currencies: Filter by currencies

        Returns:
            List of high-impact events within timeframe
        """
        events = self.fetch_calendar(days_ahead=1, currencies=currencies)
        cutoff = datetime.now() + timedelta(hours=hours_ahead)

        return [
            e for e in events
            if e.impact == NewsImpact.HIGH and e.scheduled_time <= cutoff
        ]

    def get_events_in_window(
        self,
        before_minutes: int = 30,
        after_minutes: int = 30,
        currencies: Optional[List[str]] = None
    ) -> List[EconomicEvent]:
        """
        Get events currently within a time window.

        Args:
            before_minutes: Minutes before event
            after_minutes: Minutes after event
            currencies: Filter by currencies

        Returns:
            Events currently within the window
        """
        events = self.fetch_calendar(days_ahead=1, currencies=currencies)

        return [
            e for e in events
            if e.is_within_window(before_minutes, after_minutes)
        ]

    def clear_cache(self) -> None:
        """Clear the event cache."""
        self._cached_events = []
        self._last_fetch = None
        self._cache_valid_until = None
        logger.info("Calendar cache cleared")
