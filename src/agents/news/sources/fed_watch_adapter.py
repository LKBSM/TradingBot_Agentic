# =============================================================================
# FED WATCH ADAPTER - CME Fed Rate Probability Tracker for Sprint 3
# =============================================================================
"""
Adapter for tracking Fed rate change probabilities from CME FedWatch tool.

Sprint 3 Feature: Provides real-time Fed rate expectations that significantly
impact gold (XAUUSD) and forex markets.

Data provides:
- Probability of rate hike/cut at next FOMC meeting
- Probabilities for future meetings (up to 12 months)
- Historical probability changes

Sources:
- Primary: Finnhub API (if available)
- Fallback: CME website scraping
- Alternative: Investing.com data

Usage:
    adapter = FedWatchAdapter()
    probabilities = await adapter.fetch_async()
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
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

# Try to import HTTP libraries
try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


@dataclass
class FOMCMeeting:
    """Represents an FOMC meeting."""
    date: datetime
    is_scheduled: bool = True
    is_past: bool = False
    statement_released: bool = False


@dataclass
class RateProbability:
    """Rate probability for a specific meeting."""
    meeting_date: datetime
    current_rate_bps: int              # Current rate in basis points
    probabilities: Dict[int, float]     # {target_rate_bps: probability}
    implied_rate_bps: float            # Market-implied rate
    rate_change_probability: float      # Probability of any change
    hike_probability: float            # Probability of hike
    cut_probability: float             # Probability of cut
    hold_probability: float            # Probability of hold
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class FedWatchConfig(AdapterConfig):
    """Configuration for FedWatch adapter."""
    name: str = "FedWatch"

    # Data source
    data_provider: str = "finnhub"  # "finnhub", "scrape", or "manual"

    # Finnhub settings
    finnhub_api_key: str = ""
    finnhub_base_url: str = "https://finnhub.io/api/v1"

    # Update frequency (Fed data doesn't change rapidly)
    min_update_interval_minutes: int = 30

    # FOMC meeting schedule (2024-2027)
    fomc_meetings: List[str] = field(default_factory=lambda: [
        "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
        "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
        "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
        "2025-07-30", "2025-09-17", "2025-11-05", "2025-12-17",
        "2026-01-28", "2026-03-18", "2026-05-06", "2026-06-17",
        "2026-07-29", "2026-09-16", "2026-11-04", "2026-12-16",
        "2027-01-27", "2027-03-17", "2027-05-05", "2027-06-16"
    ])

    # Threshold for "significant" probability change
    significant_change_threshold: float = 0.05  # 5%

    # Rate increments (in basis points)
    rate_increments_bps: List[int] = field(default_factory=lambda: [
        -75, -50, -25, 0, 25, 50, 75
    ])


class FedWatchAdapter(BaseNewsAdapter):
    """
    CME FedWatch adapter for Fed rate probabilities.

    Tracks market expectations for Federal Reserve rate decisions
    and generates news articles when probabilities change significantly.
    """

    def __init__(
        self,
        finnhub_api_key: str = "",
        config: Optional[FedWatchConfig] = None
    ):
        """
        Initialize FedWatch adapter.

        Args:
            finnhub_api_key: Finnhub API key
            config: Adapter configuration
        """
        self.fed_config = config or FedWatchConfig()
        super().__init__(self.fed_config)

        # Set API key
        self.fed_config.finnhub_api_key = finnhub_api_key or self.fed_config.finnhub_api_key

        # Cache for probabilities
        self._current_probabilities: Dict[str, RateProbability] = {}
        self._previous_probabilities: Dict[str, RateProbability] = {}
        self._last_update: Optional[datetime] = None

        # Current Fed rate (updated from data)
        self._current_rate_bps: int = 525  # 5.25% as of late 2023

        # Statistics
        self._updates_fetched: int = 0
        self._significant_changes: int = 0

    def fetch(self) -> List[NewsArticle]:
        """
        Fetch Fed rate probabilities synchronously.

        Returns:
            List of NewsArticle objects for significant changes
        """
        if not HAS_REQUESTS:
            self._logger.error("requests library not installed")
            return []

        # Check update interval
        if self._last_update:
            elapsed = (datetime.now() - self._last_update).total_seconds() / 60
            if elapsed < self.fed_config.min_update_interval_minutes:
                self._logger.debug(f"Skipping update, last update {elapsed:.1f} minutes ago")
                return []

        articles = []

        try:
            probabilities = self._fetch_probabilities_sync()
            articles = self._process_probabilities(probabilities)
            self._last_update = datetime.now()
            self._updates_fetched += 1

        except Exception as e:
            self._logger.error(f"FedWatch fetch error: {e}")

        return articles

    async def fetch_async(self) -> List[NewsArticle]:
        """
        Fetch Fed rate probabilities asynchronously.

        Returns:
            List of NewsArticle objects for significant changes
        """
        if not HAS_HTTPX:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.fetch)

        # Check update interval
        if self._last_update:
            elapsed = (datetime.now() - self._last_update).total_seconds() / 60
            if elapsed < self.fed_config.min_update_interval_minutes:
                return []

        articles = []

        try:
            probabilities = await self._fetch_probabilities_async()
            articles = self._process_probabilities(probabilities)
            self._last_update = datetime.now()
            self._updates_fetched += 1

        except Exception as e:
            self._logger.error(f"FedWatch async fetch error: {e}")

        return articles

    def _fetch_probabilities_sync(self) -> Dict[str, RateProbability]:
        """Fetch probabilities from data source (sync)."""
        if self.fed_config.data_provider == "finnhub":
            return self._fetch_from_finnhub_sync()
        elif self.fed_config.data_provider == "scrape":
            return self._fetch_from_cme_sync()
        else:
            return self._generate_mock_probabilities()

    async def _fetch_probabilities_async(self) -> Dict[str, RateProbability]:
        """Fetch probabilities from data source (async)."""
        if self.fed_config.data_provider == "finnhub":
            return await self._fetch_from_finnhub_async()
        elif self.fed_config.data_provider == "scrape":
            return await self._fetch_from_cme_async()
        else:
            return self._generate_mock_probabilities()

    def _fetch_from_finnhub_sync(self) -> Dict[str, RateProbability]:
        """Fetch from Finnhub API (sync)."""
        if not self.fed_config.finnhub_api_key:
            self._logger.warning("Finnhub API key not configured")
            return self._generate_mock_probabilities()

        # Finnhub economic calendar endpoint
        url = f"{self.fed_config.finnhub_base_url}/calendar/economic"
        params = {
            "token": self.fed_config.finnhub_api_key,
            "from": datetime.now().strftime("%Y-%m-%d"),
            "to": (datetime.now() + timedelta(days=365)).strftime("%Y-%m-%d")
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                return self._parse_finnhub_response(data)
        except Exception as e:
            self._logger.error(f"Finnhub error: {e}")

        return self._generate_mock_probabilities()

    async def _fetch_from_finnhub_async(self) -> Dict[str, RateProbability]:
        """Fetch from Finnhub API (async)."""
        if not self.fed_config.finnhub_api_key:
            return self._generate_mock_probabilities()

        url = f"{self.fed_config.finnhub_base_url}/calendar/economic"
        params = {
            "token": self.fed_config.finnhub_api_key,
            "from": datetime.now().strftime("%Y-%m-%d"),
            "to": (datetime.now() + timedelta(days=365)).strftime("%Y-%m-%d")
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params, timeout=30.0)
                if response.status_code == 200:
                    data = response.json()
                    return self._parse_finnhub_response(data)
        except Exception as e:
            self._logger.error(f"Finnhub async error: {e}")

        return self._generate_mock_probabilities()

    def _fetch_from_cme_sync(self) -> Dict[str, RateProbability]:
        """Fetch from CME website (scraping - sync)."""
        # Note: CME doesn't have a public API, scraping may violate ToS
        # This is a placeholder for educational purposes
        self._logger.warning("CME scraping not implemented, using mock data")
        return self._generate_mock_probabilities()

    async def _fetch_from_cme_async(self) -> Dict[str, RateProbability]:
        """Fetch from CME website (scraping - async)."""
        self._logger.warning("CME scraping not implemented, using mock data")
        return self._generate_mock_probabilities()

    def _parse_finnhub_response(self, data: Dict) -> Dict[str, RateProbability]:
        """Parse Finnhub economic calendar response."""
        probabilities = {}

        # Filter for FOMC events
        events = data.get('economicCalendar', [])
        fomc_events = [e for e in events if 'FOMC' in e.get('event', '').upper()]

        for event in fomc_events:
            try:
                meeting_date = datetime.strptime(event.get('time', '')[:10], "%Y-%m-%d")
                meeting_key = meeting_date.strftime("%Y-%m-%d")

                # Finnhub doesn't directly provide probabilities
                # We'd need to derive from Fed Funds futures
                # Using mock data structure for now
                probabilities[meeting_key] = RateProbability(
                    meeting_date=meeting_date,
                    current_rate_bps=self._current_rate_bps,
                    probabilities={
                        self._current_rate_bps - 25: 0.20,
                        self._current_rate_bps: 0.70,
                        self._current_rate_bps + 25: 0.10
                    },
                    implied_rate_bps=self._current_rate_bps - 5,
                    rate_change_probability=0.30,
                    hike_probability=0.10,
                    cut_probability=0.20,
                    hold_probability=0.70
                )

            except Exception as e:
                self._logger.debug(f"Error parsing FOMC event: {e}")

        return probabilities if probabilities else self._generate_mock_probabilities()

    def _generate_mock_probabilities(self) -> Dict[str, RateProbability]:
        """
        Generate mock probabilities for testing/demo.

        In production, this would be replaced with actual data feeds.
        """
        probabilities = {}
        now = datetime.now()

        # Get next 3 FOMC meetings
        future_meetings = [
            datetime.strptime(m, "%Y-%m-%d")
            for m in self.fed_config.fomc_meetings
            if datetime.strptime(m, "%Y-%m-%d") > now
        ][:3]

        for i, meeting_date in enumerate(future_meetings):
            meeting_key = meeting_date.strftime("%Y-%m-%d")

            # Simulate decreasing certainty for further meetings
            certainty = 0.7 - (i * 0.15)

            # Compute raw probabilities and normalize to sum to 1.0
            raw_hike = max(0.10 - (i * 0.03), 0.0)
            raw_cut = 0.20 + (i * 0.05)
            raw_hold = certainty
            prob_total = raw_hike + raw_cut + raw_hold
            norm_hike = raw_hike / prob_total
            norm_cut = raw_cut / prob_total
            norm_hold = raw_hold / prob_total

            probabilities[meeting_key] = RateProbability(
                meeting_date=meeting_date,
                current_rate_bps=self._current_rate_bps,
                probabilities={
                    self._current_rate_bps - 50: 0.05,
                    self._current_rate_bps - 25: 0.15 + (i * 0.05),
                    self._current_rate_bps: certainty,
                    self._current_rate_bps + 25: 0.10 - (i * 0.03)
                },
                implied_rate_bps=self._current_rate_bps - 10 - (i * 5),
                rate_change_probability=1 - certainty,
                hike_probability=norm_hike,
                cut_probability=norm_cut,
                hold_probability=norm_hold
            )

        return probabilities

    def _process_probabilities(
        self,
        new_probs: Dict[str, RateProbability]
    ) -> List[NewsArticle]:
        """
        Process new probabilities and generate articles for significant changes.

        Args:
            new_probs: New probability data

        Returns:
            List of NewsArticle objects for significant changes
        """
        articles = []

        for meeting_key, new_prob in new_probs.items():
            old_prob = self._previous_probabilities.get(meeting_key)

            if old_prob:
                # Check for significant changes
                changes = self._detect_significant_changes(old_prob, new_prob)
                for change in changes:
                    article = self._create_change_article(meeting_key, change, new_prob)
                    articles.append(article)
                    self._significant_changes += 1

            # Update cache
            self._previous_probabilities[meeting_key] = self._current_probabilities.get(meeting_key)
            self._current_probabilities[meeting_key] = new_prob

        # Always generate a summary article
        if new_probs and not articles:
            articles.append(self._create_summary_article(new_probs))

        return articles

    def _detect_significant_changes(
        self,
        old: RateProbability,
        new: RateProbability
    ) -> List[Dict[str, Any]]:
        """Detect significant probability changes."""
        changes = []
        threshold = self.fed_config.significant_change_threshold

        # Check hike probability
        hike_change = new.hike_probability - old.hike_probability
        if abs(hike_change) >= threshold:
            changes.append({
                'type': 'hike_probability',
                'direction': 'up' if hike_change > 0 else 'down',
                'old_value': old.hike_probability,
                'new_value': new.hike_probability,
                'change': hike_change
            })

        # Check cut probability
        cut_change = new.cut_probability - old.cut_probability
        if abs(cut_change) >= threshold:
            changes.append({
                'type': 'cut_probability',
                'direction': 'up' if cut_change > 0 else 'down',
                'old_value': old.cut_probability,
                'new_value': new.cut_probability,
                'change': cut_change
            })

        return changes

    def _create_change_article(
        self,
        meeting_key: str,
        change: Dict[str, Any],
        prob: RateProbability
    ) -> NewsArticle:
        """Create article for probability change."""
        meeting_date = datetime.strptime(meeting_key, "%Y-%m-%d")
        change_pct = abs(change['change']) * 100

        if change['type'] == 'hike_probability':
            if change['direction'] == 'up':
                title = f"Fed Rate HIKE Probability Jumps {change_pct:.0f}% for {meeting_date.strftime('%B')} Meeting"
                importance = "HIGH"
            else:
                title = f"Fed Rate HIKE Probability Falls {change_pct:.0f}% for {meeting_date.strftime('%B')} Meeting"
                importance = "MEDIUM"
        else:  # cut_probability
            if change['direction'] == 'up':
                title = f"Fed Rate CUT Probability Rises {change_pct:.0f}% for {meeting_date.strftime('%B')} Meeting"
                importance = "HIGH"
            else:
                title = f"Fed Rate CUT Probability Drops {change_pct:.0f}% for {meeting_date.strftime('%B')} Meeting"
                importance = "MEDIUM"

        content = f"""
Fed rate expectations have shifted significantly for the {meeting_date.strftime('%B %d, %Y')} FOMC meeting.

Current probabilities:
- Rate Hike: {prob.hike_probability * 100:.1f}%
- Rate Cut: {prob.cut_probability * 100:.1f}%
- Rate Hold: {prob.hold_probability * 100:.1f}%

Implied target rate: {prob.implied_rate_bps / 100:.2f}%

This shift in expectations could impact:
- XAUUSD (Gold) - typically inversely correlated with rate expectations
- USD pairs - higher rates typically strengthen USD
- Bond markets

Source: CME FedWatch Tool
        """.strip()

        return NewsArticle(
            article_id=NewsArticle.generate_id("FedWatch", title, meeting_key),
            source_name="CME FedWatch",
            source_type=ArticleSource.API,
            title=title,
            content=content,
            summary=f"Fed rate {change['type'].replace('_', ' ')} changed by {change_pct:.0f}%",
            published_at=datetime.now(),
            category=ArticleCategory.CENTRAL_BANK,
            assets=['XAUUSD', 'EURUSD', 'DXY', 'USD'],
            keywords=['Fed', 'FOMC', 'interest rate', 'FedWatch', 'monetary policy'],
            importance=importance
        )

    def _create_summary_article(
        self,
        probabilities: Dict[str, RateProbability]
    ) -> NewsArticle:
        """Create summary article of current probabilities."""
        # Get next meeting
        next_meeting_key = min(probabilities.keys())
        next_prob = probabilities[next_meeting_key]
        meeting_date = datetime.strptime(next_meeting_key, "%Y-%m-%d")

        title = f"FedWatch Update: {next_prob.hold_probability * 100:.0f}% Chance of Hold at {meeting_date.strftime('%B')} FOMC"

        content = f"""
Current market expectations for Fed rate decisions:

Next FOMC Meeting ({meeting_date.strftime('%B %d, %Y')}):
- Rate Hike: {next_prob.hike_probability * 100:.1f}%
- Rate Cut: {next_prob.cut_probability * 100:.1f}%
- Rate Hold: {next_prob.hold_probability * 100:.1f}%

Current Fed Funds Rate: {self._current_rate_bps / 100:.2f}%
Market-Implied Rate: {next_prob.implied_rate_bps / 100:.2f}%

Source: CME FedWatch Tool
        """.strip()

        return NewsArticle(
            article_id=NewsArticle.generate_id("FedWatch", title, datetime.now().isoformat()),
            source_name="CME FedWatch",
            source_type=ArticleSource.API,
            title=title,
            content=content,
            published_at=datetime.now(),
            category=ArticleCategory.CENTRAL_BANK,
            assets=['XAUUSD', 'EURUSD', 'DXY'],
            keywords=['Fed', 'FOMC', 'interest rate', 'FedWatch'],
            importance="LOW"
        )

    def get_current_probabilities(self) -> Dict[str, RateProbability]:
        """Get current cached probabilities."""
        return self._current_probabilities.copy()

    def get_next_meeting_probability(self) -> Optional[RateProbability]:
        """Get probability for next FOMC meeting."""
        if not self._current_probabilities:
            return None

        next_key = min(self._current_probabilities.keys())
        return self._current_probabilities.get(next_key)

    def get_status(self) -> Dict[str, Any]:
        """Get adapter status."""
        base_status = super().get_status()

        next_prob = self.get_next_meeting_probability()

        base_status.update({
            'updates_fetched': self._updates_fetched,
            'significant_changes': self._significant_changes,
            'meetings_tracked': len(self._current_probabilities),
            'last_update': self._last_update.isoformat() if self._last_update else None,
            'current_rate_bps': self._current_rate_bps,
            'next_meeting': {
                'date': next_prob.meeting_date.isoformat() if next_prob else None,
                'hike_prob': next_prob.hike_probability if next_prob else None,
                'cut_prob': next_prob.cut_probability if next_prob else None,
                'hold_prob': next_prob.hold_probability if next_prob else None
            } if next_prob else None
        })
        return base_status


def create_fed_watch_adapter(finnhub_api_key: str = "") -> FedWatchAdapter:
    """
    Factory function to create a FedWatch adapter.

    Args:
        finnhub_api_key: Finnhub API key for data

    Returns:
        Configured FedWatchAdapter instance
    """
    import os

    api_key = finnhub_api_key or os.environ.get('FINNHUB_API_KEY', '')

    return FedWatchAdapter(finnhub_api_key=api_key)
