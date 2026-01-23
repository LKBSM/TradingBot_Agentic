# =============================================================================
# SENTIMENT ANALYZER - Rule-Based Forex News Sentiment Analysis
# =============================================================================
# Fast, rule-based sentiment analysis optimized for Forex news.
# Uses keyword matching for speed (no external API calls in hot path).
#
# Sentiment Scale:
#   -1.0 = Very Bearish (strong sell signal)
#    0.0 = Neutral (no directional bias)
#   +1.0 = Very Bullish (strong buy signal)
#
# Currency Mapping:
#   - Positive news for USD = Bullish for USD/XXX pairs
#   - Negative news for USD = Bearish for USD/XXX pairs
#
# =============================================================================

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import re
import logging

logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    """Result of sentiment analysis on text."""
    score: float              # -1.0 to +1.0
    confidence: float         # 0.0 to 1.0
    direction: str            # "BULLISH", "BEARISH", "NEUTRAL"
    matched_keywords: List[str]
    currency_impact: Dict[str, float]  # Per-currency sentiment


class SentimentAnalyzer:
    """
    Rule-based sentiment analyzer for Forex news.

    Uses keyword matching with weighted scores for fast analysis.
    No external API calls - all processing is local.

    Features:
        - Bullish/Bearish keyword detection
        - Negation handling (e.g., "not" inverts sentiment)
        - Currency-specific impact mapping
        - Confidence scoring based on keyword density
    """

    # === BULLISH KEYWORDS (Positive for currency) ===
    BULLISH_KEYWORDS: Dict[str, float] = {
        # Central Bank - Hawkish
        'rate hike': 0.8,
        'rate increase': 0.8,
        'hawkish': 0.7,
        'tightening': 0.6,
        'quantitative tightening': 0.7,
        'qt': 0.5,
        'tapering': 0.5,

        # Economic Strength
        'strong growth': 0.6,
        'gdp growth': 0.5,
        'expansion': 0.5,
        'recovery': 0.4,
        'beat expectations': 0.6,
        'beats estimates': 0.6,
        'better than expected': 0.6,
        'above forecast': 0.5,
        'stronger than': 0.5,
        'outperform': 0.4,
        'surge': 0.5,
        'soar': 0.5,
        'jump': 0.4,
        'rally': 0.4,
        'gain': 0.3,
        'rise': 0.3,
        'increase': 0.3,
        'improve': 0.3,
        'robust': 0.4,
        'solid': 0.3,

        # Employment
        'job growth': 0.5,
        'employment increase': 0.5,
        'unemployment falls': 0.5,
        'hiring surge': 0.5,

        # Inflation (moderate is positive)
        'inflation target': 0.3,
        'price stability': 0.3,
    }

    # === BEARISH KEYWORDS (Negative for currency) ===
    BEARISH_KEYWORDS: Dict[str, float] = {
        # Central Bank - Dovish
        'rate cut': 0.8,
        'rate reduction': 0.8,
        'dovish': 0.7,
        'easing': 0.6,
        'quantitative easing': 0.7,
        'qe': 0.5,
        'stimulus': 0.5,
        'accommodation': 0.4,

        # Economic Weakness
        'recession': 0.8,
        'contraction': 0.7,
        'slowdown': 0.6,
        'decline': 0.5,
        'miss expectations': 0.6,
        'misses estimates': 0.6,
        'worse than expected': 0.6,
        'below forecast': 0.5,
        'weaker than': 0.5,
        'underperform': 0.4,
        'plunge': 0.6,
        'crash': 0.7,
        'tumble': 0.5,
        'drop': 0.4,
        'fall': 0.3,
        'decrease': 0.3,
        'deteriorate': 0.4,
        'weak': 0.4,
        'soft': 0.3,

        # Employment
        'job losses': 0.6,
        'unemployment rises': 0.5,
        'layoffs': 0.5,
        'hiring freeze': 0.4,

        # Crisis indicators
        'default': 0.8,
        'crisis': 0.7,
        'risk': 0.3,
        'uncertainty': 0.4,
        'concern': 0.3,
        'worry': 0.3,
        'fear': 0.4,
    }

    # === NEGATION WORDS (Invert sentiment) ===
    NEGATION_WORDS = {
        'not', 'no', "n't", 'never', 'neither', 'nor', 'none',
        'without', 'hardly', 'barely', 'unlikely'
    }

    # === CURRENCY KEYWORDS ===
    CURRENCY_KEYWORDS: Dict[str, List[str]] = {
        'USD': ['usd', 'dollar', 'fed', 'federal reserve', 'fomc', 'powell', 'us economy', 'american'],
        'EUR': ['eur', 'euro', 'ecb', 'european central bank', 'lagarde', 'eurozone', 'european'],
        'GBP': ['gbp', 'pound', 'sterling', 'boe', 'bank of england', 'bailey', 'uk economy', 'british'],
        'JPY': ['jpy', 'yen', 'boj', 'bank of japan', 'ueda', 'japanese'],
        'CHF': ['chf', 'franc', 'snb', 'swiss national bank', 'swiss'],
        'AUD': ['aud', 'aussie', 'rba', 'reserve bank of australia', 'australian'],
        'CAD': ['cad', 'loonie', 'boc', 'bank of canada', 'canadian'],
        'NZD': ['nzd', 'kiwi', 'rbnz', 'reserve bank of new zealand', 'new zealand'],
        'XAU': ['gold', 'xau', 'precious metal', 'bullion', 'safe haven'],
    }

    def __init__(
        self,
        custom_bullish: Optional[Dict[str, float]] = None,
        custom_bearish: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the sentiment analyzer.

        Args:
            custom_bullish: Additional bullish keywords with weights
            custom_bearish: Additional bearish keywords with weights
        """
        self._bullish = {**self.BULLISH_KEYWORDS, **(custom_bullish or {})}
        self._bearish = {**self.BEARISH_KEYWORDS, **(custom_bearish or {})}

        # Pre-compile regex patterns for efficiency
        self._bullish_patterns = self._compile_patterns(self._bullish)
        self._bearish_patterns = self._compile_patterns(self._bearish)
        self._negation_pattern = re.compile(
            r'\b(' + '|'.join(self.NEGATION_WORDS) + r')\b',
            re.IGNORECASE
        )

        logger.info(
            f"SentimentAnalyzer initialized with "
            f"{len(self._bullish)} bullish and {len(self._bearish)} bearish keywords"
        )

    def _compile_patterns(self, keywords: Dict[str, float]) -> List[Tuple[re.Pattern, float]]:
        """Compile keyword patterns for efficient matching."""
        patterns = []
        for keyword, weight in keywords.items():
            pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
            patterns.append((pattern, weight))
        return patterns

    def analyze(self, text: str, target_currency: Optional[str] = None) -> SentimentResult:
        """
        Analyze sentiment of text.

        Args:
            text: News headline or body text to analyze
            target_currency: Optional currency to focus on (e.g., "USD")

        Returns:
            SentimentResult with score, confidence, and matched keywords
        """
        if not text or not text.strip():
            return SentimentResult(
                score=0.0,
                confidence=0.0,
                direction="NEUTRAL",
                matched_keywords=[],
                currency_impact={}
            )

        text_lower = text.lower()

        # Find negation positions (words within 3 words of negation get inverted)
        negation_positions = set()
        for match in self._negation_pattern.finditer(text_lower):
            # Mark positions within window after negation
            start = match.start()
            negation_positions.add(start)

        # Score bullish keywords
        bullish_score = 0.0
        bullish_matches = []
        for pattern, weight in self._bullish_patterns:
            for match in pattern.finditer(text_lower):
                # Check if negated
                is_negated = self._is_negated(text_lower, match.start())
                adjusted_weight = -weight if is_negated else weight
                bullish_score += adjusted_weight
                bullish_matches.append(match.group())

        # Score bearish keywords
        bearish_score = 0.0
        bearish_matches = []
        for pattern, weight in self._bearish_patterns:
            for match in pattern.finditer(text_lower):
                is_negated = self._is_negated(text_lower, match.start())
                adjusted_weight = -weight if is_negated else weight
                bearish_score += adjusted_weight
                bearish_matches.append(match.group())

        # Calculate net sentiment
        net_score = bullish_score - bearish_score

        # Normalize to [-1, 1] using tanh-like scaling
        max_possible = max(abs(bullish_score) + abs(bearish_score), 1.0)
        normalized_score = max(-1.0, min(1.0, net_score / max_possible))

        # Calculate confidence based on keyword matches
        total_matches = len(bullish_matches) + len(bearish_matches)
        confidence = min(1.0, total_matches * 0.2)  # Cap at 1.0

        # Determine direction
        if normalized_score > 0.2:
            direction = "BULLISH"
        elif normalized_score < -0.2:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"

        # Detect currency impact
        currency_impact = self._detect_currency_impact(text_lower, normalized_score)

        # If target currency specified, adjust score
        if target_currency and target_currency in currency_impact:
            normalized_score = currency_impact[target_currency]

        return SentimentResult(
            score=round(normalized_score, 3),
            confidence=round(confidence, 3),
            direction=direction,
            matched_keywords=bullish_matches + bearish_matches,
            currency_impact=currency_impact
        )

    def _is_negated(self, text: str, position: int, window: int = 30) -> bool:
        """
        Check if a word at position is negated.

        Looks for negation words within `window` characters before the position.
        """
        start = max(0, position - window)
        preceding_text = text[start:position]
        return bool(self._negation_pattern.search(preceding_text))

    def _detect_currency_impact(
        self,
        text: str,
        base_sentiment: float
    ) -> Dict[str, float]:
        """
        Detect which currencies are mentioned and their sentiment impact.

        Returns dict mapping currency code to sentiment score.
        """
        currency_impact = {}

        for currency, keywords in self.CURRENCY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text:
                    # Currency mentioned - apply base sentiment
                    currency_impact[currency] = base_sentiment
                    break

        return currency_impact

    def analyze_batch(
        self,
        texts: List[str],
        target_currency: Optional[str] = None
    ) -> SentimentResult:
        """
        Analyze multiple texts and aggregate sentiment.

        Useful for analyzing multiple news items at once.

        Args:
            texts: List of news headlines/bodies
            target_currency: Optional currency focus

        Returns:
            Aggregated SentimentResult
        """
        if not texts:
            return SentimentResult(
                score=0.0,
                confidence=0.0,
                direction="NEUTRAL",
                matched_keywords=[],
                currency_impact={}
            )

        results = [self.analyze(text, target_currency) for text in texts]

        # Weighted average by confidence
        total_weight = sum(r.confidence for r in results) or 1.0
        weighted_score = sum(r.score * r.confidence for r in results) / total_weight

        # Aggregate confidence
        avg_confidence = sum(r.confidence for r in results) / len(results)

        # Collect all keywords
        all_keywords = []
        for r in results:
            all_keywords.extend(r.matched_keywords)

        # Aggregate currency impact
        currency_impact: Dict[str, List[float]] = {}
        for r in results:
            for currency, score in r.currency_impact.items():
                if currency not in currency_impact:
                    currency_impact[currency] = []
                currency_impact[currency].append(score)

        avg_currency_impact = {
            c: sum(scores) / len(scores)
            for c, scores in currency_impact.items()
        }

        # Determine direction
        if weighted_score > 0.2:
            direction = "BULLISH"
        elif weighted_score < -0.2:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"

        return SentimentResult(
            score=round(weighted_score, 3),
            confidence=round(avg_confidence, 3),
            direction=direction,
            matched_keywords=list(set(all_keywords)),  # Unique keywords
            currency_impact=avg_currency_impact
        )

    def get_keyword_stats(self) -> Dict[str, int]:
        """Get statistics about loaded keywords."""
        return {
            'bullish_keywords': len(self._bullish),
            'bearish_keywords': len(self._bearish),
            'negation_words': len(self.NEGATION_WORDS),
            'tracked_currencies': len(self.CURRENCY_KEYWORDS)
        }
