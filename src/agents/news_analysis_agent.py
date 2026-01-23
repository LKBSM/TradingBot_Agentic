# =============================================================================
# NEWS ANALYSIS AGENT - Real-Time Forex News Monitoring
# =============================================================================
# This agent monitors economic calendars and news feeds to:
#   1. BLOCK trading during high-impact events (NFP, FOMC, CPI)
#   2. REDUCE position size during medium-impact events
#   3. Provide sentiment scores for observation space enhancement
#
# Integration Points:
#   - Subscribes to TRADE_PROPOSED events
#   - Returns NewsAssessment with BLOCK/REDUCE/ALLOW decision
#   - Works with TradingOrchestrator for coordinated decisions
#
# =============================================================================

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
import logging
import time
import uuid
from collections import deque

from src.agents.base_agent import BaseAgent, AgentCapability, AgentState
from src.agents.events import (
    AgentEvent, EventType, EventBus, DecisionType,
    TradeProposal, RiskLevel
)
from src.agents.config import AgentConfig
from src.agents.news.sentiment import SentimentAnalyzer, SentimentResult
from src.agents.news.economic_calendar import (
    EconomicCalendarFetcher, EconomicEvent, NewsImpact
)
from src.agents.news.fetchers import NewsHeadlineFetcher, NewsItem


# =============================================================================
# NEWS-SPECIFIC ENUMS AND DATA CLASSES
# =============================================================================


class NewsDecision(Enum):
    """Decision types for news-based trade filtering."""
    BLOCK = "block"     # Do NOT trade - high-impact event imminent
    REDUCE = "reduce"   # Trade with reduced position size
    ALLOW = "allow"     # Normal trading allowed


@dataclass
class NewsAssessment:
    """
    Result of news analysis for trading decision.

    This is the primary output of the NewsAnalysisAgent.
    Used by the TradingOrchestrator to filter/modify trades.
    """
    decision: NewsDecision              # BLOCK, REDUCE, or ALLOW
    current_impact_level: NewsImpact    # Current news environment impact
    sentiment_score: float              # -1 (bearish) to +1 (bullish)
    sentiment_confidence: float         # 0-1 confidence in sentiment
    blocking_events: List[EconomicEvent]  # Events causing blocks
    position_multiplier: float          # 1.0 = normal, 0.5 = reduced, 0.0 = blocked
    reasoning: List[str]                # Human-readable explanation
    valid_until: datetime               # When this assessment expires
    hours_to_next_high_impact: float    # Hours until next major event
    assessment_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    assessment_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'assessment_id': self.assessment_id,
            'decision': self.decision.value,
            'impact_level': self.current_impact_level.value,
            'sentiment_score': round(self.sentiment_score, 3),
            'sentiment_confidence': round(self.sentiment_confidence, 3),
            'blocking_events': [e.to_dict() for e in self.blocking_events],
            'position_multiplier': round(self.position_multiplier, 3),
            'reasoning': self.reasoning,
            'valid_until': self.valid_until.isoformat(),
            'hours_to_next_high_impact': round(self.hours_to_next_high_impact, 2),
            'assessment_time_ms': round(self.assessment_time_ms, 2),
            'timestamp': self.timestamp.isoformat()
        }

    def is_trading_allowed(self) -> bool:
        """Check if trading is allowed."""
        return self.decision != NewsDecision.BLOCK

    def get_summary(self) -> str:
        """Get a one-line summary."""
        icon = {
            NewsDecision.BLOCK: "XX",
            NewsDecision.REDUCE: "~~",
            NewsDecision.ALLOW: "OK"
        }
        return (
            f"[{icon.get(self.decision, '??')}] {self.decision.value.upper()} | "
            f"Impact: {self.current_impact_level.value} | "
            f"Sentiment: {self.sentiment_score:+.2f} | "
            f"Size: {self.position_multiplier:.0%}"
        )


@dataclass
class NewsAgentConfig(AgentConfig):
    """Configuration for NewsAnalysisAgent."""

    # === BLOCKING RULES ===
    high_impact_block_minutes_before: int = 30  # Block 30 min before event
    high_impact_block_minutes_after: int = 30   # Block 30 min after event
    medium_impact_reduce_factor: float = 0.5    # Reduce position by 50%
    low_impact_reduce_factor: float = 0.8       # Slight reduction for low impact

    # === API SETTINGS ===
    newsapi_key: Optional[str] = None           # NewsAPI key (env var fallback)
    calendar_fetch_interval_hours: float = 4.0  # How often to refresh calendar
    news_fetch_interval_minutes: int = 15       # How often to fetch news

    # === SENTIMENT SETTINGS ===
    sentiment_weight_decay_hours: float = 6.0   # News older than 6h has less weight
    min_sentiment_confidence: float = 0.3       # Ignore low-confidence sentiment
    sentiment_impact_on_sizing: float = 0.1     # Max sizing adjustment from sentiment

    # === CURRENCIES TO MONITOR ===
    monitored_currencies: List[str] = field(
        default_factory=lambda: ["USD", "EUR", "GBP", "JPY", "XAU"]
    )

    # === FEATURE FLAGS ===
    enable_sentiment_in_obs: bool = True        # Add sentiment to observation
    enable_calendar_blocking: bool = True       # Enable event blocking
    enable_news_headlines: bool = True          # Enable headline fetching

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        base = super().to_dict()
        base.update({
            'high_impact_block_minutes_before': self.high_impact_block_minutes_before,
            'high_impact_block_minutes_after': self.high_impact_block_minutes_after,
            'medium_impact_reduce_factor': self.medium_impact_reduce_factor,
            'low_impact_reduce_factor': self.low_impact_reduce_factor,
            'calendar_fetch_interval_hours': self.calendar_fetch_interval_hours,
            'news_fetch_interval_minutes': self.news_fetch_interval_minutes,
            'sentiment_weight_decay_hours': self.sentiment_weight_decay_hours,
            'min_sentiment_confidence': self.min_sentiment_confidence,
            'sentiment_impact_on_sizing': self.sentiment_impact_on_sizing,
            'monitored_currencies': self.monitored_currencies,
            'enable_sentiment_in_obs': self.enable_sentiment_in_obs,
            'enable_calendar_blocking': self.enable_calendar_blocking,
            'enable_news_headlines': self.enable_news_headlines,
        })
        return base


# =============================================================================
# NEWS ANALYSIS AGENT
# =============================================================================


class NewsAnalysisAgent(BaseAgent):
    """
    News Analysis Agent for Forex Trading.

    Monitors economic calendar and news feeds to:
    1. Block trading during high-impact events
    2. Reduce position sizes during medium-impact events
    3. Add sentiment scores to trading decisions

    Capabilities:
        - Economic calendar monitoring (ForexFactory)
        - News headline fetching (NewsAPI, Central Bank RSS)
        - Rule-based sentiment analysis
        - Impact classification (HIGH/MEDIUM/LOW)

    Integration:
        - Subscribes to TRADE_PROPOSED events
        - Returns NewsAssessment for orchestrator
        - Provides features for observation space
    """

    def __init__(
        self,
        config: Optional[NewsAgentConfig] = None,
        event_bus: Optional[EventBus] = None,
        name: str = "NewsAnalysisAgent"
    ):
        """
        Initialize the NewsAnalysisAgent.

        Args:
            config: Agent configuration
            event_bus: Event bus for communication
            name: Agent name
        """
        self._news_config = config or NewsAgentConfig()

        super().__init__(
            name=name,
            config=self._news_config.to_dict(),
            event_bus=event_bus
        )

        # === COMPONENTS (initialized in initialize()) ===
        self._calendar_fetcher: Optional[EconomicCalendarFetcher] = None
        self._news_fetcher: Optional[NewsHeadlineFetcher] = None
        self._sentiment_analyzer: Optional[SentimentAnalyzer] = None

        # === STATE ===
        self._upcoming_events: List[EconomicEvent] = []
        self._recent_news: deque = deque(maxlen=100)
        self._current_sentiment: Dict[str, SentimentResult] = {}  # Per currency
        self._last_calendar_update: Optional[datetime] = None
        self._last_news_update: Optional[datetime] = None

        # === BLOCKING STATE ===
        self._is_blocked: bool = False
        self._block_reason: Optional[str] = None
        self._block_until: Optional[datetime] = None
        self._current_blocking_events: List[EconomicEvent] = []

        # === STATISTICS ===
        self._total_assessments: int = 0
        self._blocks_count: int = 0
        self._reductions_count: int = 0
        self._allows_count: int = 0

        # === EVENT SUBSCRIPTIONS ===
        self._subscriptions = [EventType.TRADE_PROPOSED]

        self._logger = logging.getLogger(f"agent.{self.full_id}")

    # =========================================================================
    # LIFECYCLE METHODS
    # =========================================================================

    def initialize(self) -> bool:
        """
        Initialize agent resources.

        Creates fetchers, analyzers, and loads initial data.
        """
        self._logger.info("Initializing News Analysis Agent...")

        try:
            # Create sentiment analyzer
            self._sentiment_analyzer = SentimentAnalyzer()
            self._logger.info("  - Sentiment analyzer created")

            # Create calendar fetcher
            self._calendar_fetcher = EconomicCalendarFetcher(
                cache_hours=int(self._news_config.calendar_fetch_interval_hours)
            )
            self._logger.info("  - Economic calendar fetcher created")

            # Create news fetcher
            if self._news_config.enable_news_headlines:
                self._news_fetcher = NewsHeadlineFetcher(
                    newsapi_key=self._news_config.newsapi_key,
                    cache_minutes=self._news_config.news_fetch_interval_minutes
                )
                self._logger.info("  - News headline fetcher created")

            # Log configuration
            self._logger.info(
                f"  - Blocking: {self._news_config.high_impact_block_minutes_before}min before, "
                f"{self._news_config.high_impact_block_minutes_after}min after"
            )
            self._logger.info(
                f"  - Monitored currencies: {self._news_config.monitored_currencies}"
            )

            # Initial data fetch
            self._update_calendar()
            if self._news_config.enable_news_headlines:
                self._update_news()

            self._logger.info("News Analysis Agent initialized successfully")
            return True

        except Exception as e:
            self._logger.error(f"Initialization failed: {e}")
            return False

    def shutdown(self) -> bool:
        """Clean up agent resources."""
        self._logger.info("Shutting down News Analysis Agent...")

        # Log final statistics
        self._logger.info(f"  - Total assessments: {self._total_assessments}")
        self._logger.info(f"  - Blocks: {self._blocks_count}")
        self._logger.info(f"  - Reductions: {self._reductions_count}")
        self._logger.info(f"  - Allows: {self._allows_count}")

        return True

    def process_event(self, event: AgentEvent) -> Optional[AgentEvent]:
        """
        Process incoming events.

        Handles TRADE_PROPOSED events and returns news assessment.
        """
        if event.event_type == EventType.TRADE_PROPOSED:
            proposal = TradeProposal(**event.payload)
            assessment = self.evaluate_news_impact(proposal)

            return AgentEvent(
                event_type=EventType.NEWS_ALERT,
                source_agent=self.full_id,
                payload=assessment.to_dict(),
                correlation_id=proposal.proposal_id
            )

        return None

    def get_capabilities(self) -> List[AgentCapability]:
        """Declare agent capabilities."""
        return [AgentCapability.SENTIMENT_ANALYSIS]

    # =========================================================================
    # MAIN EVALUATION METHOD
    # =========================================================================

    def evaluate_news_impact(
        self,
        proposal: TradeProposal,
        current_time: Optional[datetime] = None
    ) -> NewsAssessment:
        """
        Evaluate current news environment for trading.

        This is the main evaluation method called by the orchestrator.

        Args:
            proposal: Trade proposal to evaluate
            current_time: Override current time (for testing)

        Returns:
            NewsAssessment with decision, position multiplier, and reasoning
        """
        start_time = time.time()
        self._total_assessments += 1
        current_time = current_time or datetime.now()

        reasoning: List[str] = []
        blocking_events: List[EconomicEvent] = []
        position_multiplier = 1.0

        # --- Check if agent is enabled ---
        if not self._news_config.enabled:
            reasoning.append("News Analysis Agent is disabled")
            return self._create_assessment(
                NewsDecision.ALLOW,
                NewsImpact.NONE,
                0.0, 0.0,
                [], 1.0,
                reasoning,
                start_time
            )

        # --- Refresh data if needed ---
        self._maybe_update_calendar()
        if self._news_config.enable_news_headlines:
            self._maybe_update_news()

        # --- Check for high-impact event blocking ---
        if self._news_config.enable_calendar_blocking:
            blocking_events, impact_level = self._check_event_blocking(current_time)

            if blocking_events:
                for event in blocking_events:
                    mins = event.minutes_until()
                    if mins > 0:
                        reasoning.append(
                            f"HIGH-IMPACT: {event.name} in {abs(mins):.0f} minutes"
                        )
                    else:
                        reasoning.append(
                            f"HIGH-IMPACT: {event.name} occurred {abs(mins):.0f} minutes ago"
                        )

                self._blocks_count += 1
                return self._create_assessment(
                    NewsDecision.BLOCK,
                    NewsImpact.HIGH,
                    0.0, 0.0,
                    blocking_events, 0.0,
                    reasoning,
                    start_time
                )

        # --- Check for medium-impact events (reduce position) ---
        medium_events = self._check_medium_impact_events(current_time)
        if medium_events:
            position_multiplier *= self._news_config.medium_impact_reduce_factor
            for event in medium_events:
                reasoning.append(f"MEDIUM-IMPACT: {event.name} - reducing position")
            self._reductions_count += 1

        # --- Calculate sentiment ---
        sentiment_score = 0.0
        sentiment_confidence = 0.0

        if self._news_config.enable_news_headlines and self._recent_news:
            currency = proposal.asset.split('/')[0] if '/' in proposal.asset else "USD"
            sentiment_result = self._calculate_aggregated_sentiment(currency)
            sentiment_score = sentiment_result.score
            sentiment_confidence = sentiment_result.confidence

            if sentiment_confidence >= self._news_config.min_sentiment_confidence:
                # Adjust position based on sentiment
                sentiment_adjustment = (
                    sentiment_score * self._news_config.sentiment_impact_on_sizing
                )
                # If opening long, bullish sentiment helps; bearish hurts
                if proposal.action in ['BUY', 'OPEN_LONG']:
                    position_multiplier *= (1 + sentiment_adjustment)
                elif proposal.action in ['SELL', 'OPEN_SHORT']:
                    position_multiplier *= (1 - sentiment_adjustment)

                reasoning.append(
                    f"Sentiment: {sentiment_result.direction} "
                    f"({sentiment_score:+.2f}, conf: {sentiment_confidence:.0%})"
                )

        # --- Determine final decision ---
        if position_multiplier < 0.9:
            decision = NewsDecision.REDUCE
            if not medium_events:
                self._reductions_count += 1
        else:
            decision = NewsDecision.ALLOW
            self._allows_count += 1

        # Ensure multiplier is in valid range
        position_multiplier = max(0.1, min(1.5, position_multiplier))

        # Determine current impact level
        if medium_events:
            impact_level = NewsImpact.MEDIUM
        elif self._upcoming_events:
            impact_level = NewsImpact.LOW
        else:
            impact_level = NewsImpact.NONE

        if not reasoning:
            reasoning.append("No significant news events detected")

        return self._create_assessment(
            decision,
            impact_level,
            sentiment_score,
            sentiment_confidence,
            blocking_events,
            position_multiplier,
            reasoning,
            start_time
        )

    # =========================================================================
    # INTERNAL METHODS
    # =========================================================================

    def _create_assessment(
        self,
        decision: NewsDecision,
        impact_level: NewsImpact,
        sentiment_score: float,
        sentiment_confidence: float,
        blocking_events: List[EconomicEvent],
        position_multiplier: float,
        reasoning: List[str],
        start_time: float
    ) -> NewsAssessment:
        """Create a NewsAssessment object."""
        elapsed_ms = (time.time() - start_time) * 1000

        # Calculate hours to next high-impact event
        hours_to_next = self._get_hours_to_next_high_impact()

        # Set validity period
        valid_until = datetime.now() + timedelta(minutes=5)

        self._metrics.decisions_made += 1
        self._metrics.last_activity = datetime.now()

        return NewsAssessment(
            decision=decision,
            current_impact_level=impact_level,
            sentiment_score=sentiment_score,
            sentiment_confidence=sentiment_confidence,
            blocking_events=blocking_events,
            position_multiplier=position_multiplier,
            reasoning=reasoning,
            valid_until=valid_until,
            hours_to_next_high_impact=hours_to_next,
            assessment_time_ms=elapsed_ms
        )

    def _check_event_blocking(
        self,
        current_time: datetime
    ) -> Tuple[List[EconomicEvent], NewsImpact]:
        """
        Check if any high-impact event requires trading to be blocked.

        Returns:
            Tuple of (blocking_events, highest_impact_level)
        """
        blocking_events = []
        highest_impact = NewsImpact.NONE

        for event in self._upcoming_events:
            if event.impact != NewsImpact.HIGH:
                continue

            # Check if within blocking window
            is_within = event.is_within_window(
                self._news_config.high_impact_block_minutes_before,
                self._news_config.high_impact_block_minutes_after
            )

            if is_within:
                blocking_events.append(event)
                highest_impact = NewsImpact.HIGH

        self._current_blocking_events = blocking_events
        self._is_blocked = len(blocking_events) > 0

        return blocking_events, highest_impact

    def _check_medium_impact_events(
        self,
        current_time: datetime
    ) -> List[EconomicEvent]:
        """Check for medium-impact events in progress."""
        medium_events = []

        for event in self._upcoming_events:
            if event.impact != NewsImpact.MEDIUM:
                continue

            # Use shorter window for medium impact
            is_within = event.is_within_window(15, 15)  # 15 min before/after

            if is_within:
                medium_events.append(event)

        return medium_events

    def _calculate_aggregated_sentiment(self, currency: str) -> SentimentResult:
        """
        Calculate aggregated sentiment from recent news.

        Applies time-decay weighting to older news.
        """
        if not self._sentiment_analyzer or not self._recent_news:
            return SentimentResult(
                score=0.0,
                confidence=0.0,
                direction="NEUTRAL",
                matched_keywords=[],
                currency_impact={}
            )

        # Get headlines with time decay
        headlines = []
        weights = []
        decay_hours = self._news_config.sentiment_weight_decay_hours

        for news_item in self._recent_news:
            age_hours = news_item.age_hours()
            weight = max(0.1, 1.0 - (age_hours / decay_hours))
            headlines.append(news_item.headline)
            weights.append(weight)

        # Analyze all headlines
        result = self._sentiment_analyzer.analyze_batch(headlines, currency)

        return result

    def _get_hours_to_next_high_impact(self) -> float:
        """Get hours until the next high-impact event."""
        now = datetime.now()

        for event in self._upcoming_events:
            if event.impact == NewsImpact.HIGH and event.scheduled_time > now:
                delta = event.scheduled_time - now
                return delta.total_seconds() / 3600

        return 999.0  # No upcoming high-impact events

    def _maybe_update_calendar(self) -> None:
        """Update calendar if refresh interval has passed."""
        if self._last_calendar_update is None:
            self._update_calendar()
            return

        elapsed = (datetime.now() - self._last_calendar_update).total_seconds() / 3600

        if elapsed >= self._news_config.calendar_fetch_interval_hours:
            self._update_calendar()

    def _update_calendar(self) -> None:
        """Fetch and update economic calendar."""
        if not self._calendar_fetcher:
            return

        try:
            events = self._calendar_fetcher.fetch_calendar(
                days_ahead=7,
                currencies=self._news_config.monitored_currencies
            )
            self._upcoming_events = events
            self._last_calendar_update = datetime.now()
            self._logger.debug(f"Updated calendar: {len(events)} events")
        except Exception as e:
            self._logger.error(f"Calendar update failed: {e}")

    def _maybe_update_news(self) -> None:
        """Update news if refresh interval has passed."""
        if self._last_news_update is None:
            self._update_news()
            return

        elapsed = (datetime.now() - self._last_news_update).total_seconds() / 60

        if elapsed >= self._news_config.news_fetch_interval_minutes:
            self._update_news()

    def _update_news(self) -> None:
        """Fetch and update news headlines."""
        if not self._news_fetcher:
            return

        try:
            news_items = self._news_fetcher.fetch_news(
                max_age_hours=24,
                currencies=self._news_config.monitored_currencies
            )
            for item in news_items:
                self._recent_news.append(item)
            self._last_news_update = datetime.now()
            self._logger.debug(f"Updated news: {len(news_items)} items")
        except Exception as e:
            self._logger.error(f"News update failed: {e}")

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def get_upcoming_events(self, hours: int = 24) -> List[EconomicEvent]:
        """Get economic events in the next N hours."""
        cutoff = datetime.now() + timedelta(hours=hours)
        return [e for e in self._upcoming_events if e.scheduled_time <= cutoff]

    def get_current_sentiment(self, currency: str = "USD") -> SentimentResult:
        """Get current sentiment for a currency."""
        return self._calculate_aggregated_sentiment(currency)

    def is_trading_blocked(self) -> Tuple[bool, Optional[str]]:
        """Check if trading is currently blocked."""
        if self._current_blocking_events:
            reasons = [e.name for e in self._current_blocking_events]
            return True, f"Blocked by: {', '.join(reasons)}"
        return False, None

    def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics."""
        total = self._total_assessments or 1
        return {
            'total_assessments': self._total_assessments,
            'blocks': self._blocks_count,
            'reductions': self._reductions_count,
            'allows': self._allows_count,
            'block_rate': f"{self._blocks_count / total * 100:.1f}%",
            'reduction_rate': f"{self._reductions_count / total * 100:.1f}%",
            'upcoming_events': len(self._upcoming_events),
            'recent_news_items': len(self._recent_news),
            'is_currently_blocked': self._is_blocked
        }

    def get_observation_features(self) -> Dict[str, float]:
        """
        Get features for observation space enhancement.

        Returns dict with features that can be added to the RL observation.
        """
        sentiment = self._calculate_aggregated_sentiment("USD")
        hours_to_event = self._get_hours_to_next_high_impact()

        return {
            'is_blocked': 1.0 if self._is_blocked else 0.0,
            'impact_high': 1.0 if any(e.impact == NewsImpact.HIGH for e in self._current_blocking_events) else 0.0,
            'impact_medium': 1.0 if self._check_medium_impact_events(datetime.now()) else 0.0,
            'impact_low': 1.0 if self._upcoming_events and not self._is_blocked else 0.0,
            'sentiment_score': sentiment.score,
            'sentiment_confidence': sentiment.confidence,
            'hours_to_event_normalized': min(1.0, hours_to_event / 24.0),
            'position_multiplier': 0.0 if self._is_blocked else 1.0
        }

    def get_news_dashboard(self) -> str:
        """Get a text-based dashboard for monitoring."""
        stats = self.get_statistics()
        blocked, reason = self.is_trading_blocked()
        sentiment = self._calculate_aggregated_sentiment("USD")

        status = "BLOCKED" if blocked else "ACTIVE"
        next_event = "None"
        if self._upcoming_events:
            next_high = next(
                (e for e in self._upcoming_events if e.impact == NewsImpact.HIGH),
                None
            )
            if next_high:
                mins = next_high.minutes_until()
                next_event = f"{next_high.name} ({mins:.0f}min)"

        return f"""
================================================================================
                        NEWS ANALYSIS DASHBOARD
================================================================================
 Status:           {status:12}
 Block Reason:     {reason or 'None'}
 Next High-Impact: {next_event}

 Sentiment:        {sentiment.direction:12} ({sentiment.score:+.2f})
 Confidence:       {sentiment.confidence:.0%}

 Statistics:
   Total Assessed: {stats['total_assessments']:>10}
   Blocked:        {stats['blocks']:>10} ({stats['block_rate']})
   Reduced:        {stats['reductions']:>10} ({stats['reduction_rate']})
   Allowed:        {stats['allows']:>10}

 Data:
   Upcoming Events: {stats['upcoming_events']}
   News Items:      {stats['recent_news_items']}
================================================================================
"""


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


def create_news_analysis_agent(
    newsapi_key: Optional[str] = None,
    block_minutes_before: int = 30,
    block_minutes_after: int = 30,
    monitored_currencies: Optional[List[str]] = None,
    event_bus: Optional[EventBus] = None
) -> NewsAnalysisAgent:
    """
    Factory function to create a configured NewsAnalysisAgent.

    Args:
        newsapi_key: NewsAPI key (uses env var if not provided)
        block_minutes_before: Minutes before high-impact to block
        block_minutes_after: Minutes after high-impact to block
        monitored_currencies: Currencies to monitor
        event_bus: Event bus for communication

    Returns:
        Configured NewsAnalysisAgent instance
    """
    config = NewsAgentConfig(
        newsapi_key=newsapi_key,
        high_impact_block_minutes_before=block_minutes_before,
        high_impact_block_minutes_after=block_minutes_after,
        monitored_currencies=monitored_currencies or ["USD", "EUR", "GBP", "XAU"]
    )

    agent = NewsAnalysisAgent(config=config, event_bus=event_bus)
    return agent
