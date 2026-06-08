"""Position-multiplier is composed from regime + news inputs."""

from dataclasses import dataclass

from src.intelligence.confluence_detector import ConfluenceDetector


@dataclass
class _Regime:
    position_size_multiplier: float = 1.0
    regime: object = None
    confidence: float = 0.8
    trend_direction: object = None
    trend_strength: float = 0.5


@dataclass
class _News:
    position_multiplier: float = 1.0
    decision: object = None
    sentiment_score: float = 0.0
    sentiment_confidence: float = 0.5


def _features(direction: int = 1) -> dict:
    return {
        "BOS_SIGNAL": float(direction),
        "BOS_EVENT": float(direction),
        "FVG_SIGNAL": 0.0,
        "OB_STRENGTH_NORM": 0.0,
        "RSI": 50.0,
        "MACD_Diff": 0.0,
    }


def _analyse(detector, regime=None, news=None):
    return detector.analyze(
        smc_features=_features(),
        regime=regime,
        news=news,
        price=100.0,
        atr=1.0,
        volume=None,
        volume_ma=None,
    )


class TestPositionMultiplier:
    def setup_method(self):
        # require_retest=False isolates this test from the BOS retest gate;
        # we only care about the position-multiplier composition here.
        self.detector = ConfluenceDetector(min_score=0.0, require_retest=False)

    def test_defaults_to_full_size_when_no_agents(self):
        signal = _analyse(self.detector)
        assert signal is not None
        assert signal.position_multiplier == 1.0

    def test_regime_dampens_size(self):
        regime = _Regime(position_size_multiplier=0.5)
        signal = _analyse(self.detector, regime=regime)
        assert signal.position_multiplier == 0.5

    def test_news_block_zeros_size(self):
        regime = _Regime(position_size_multiplier=1.0)
        news = _News(position_multiplier=0.0)
        signal = _analyse(self.detector, regime=regime, news=news)
        assert signal.position_multiplier == 0.0

    def test_combined_multiplication(self):
        regime = _Regime(position_size_multiplier=0.75)
        news = _News(position_multiplier=0.5)
        signal = _analyse(self.detector, regime=regime, news=news)
        assert signal.position_multiplier == 0.375

    def test_reason_mentions_components(self):
        regime = _Regime(position_size_multiplier=0.5)
        news = _News(position_multiplier=0.8)
        signal = _analyse(self.detector, regime=regime, news=news)
        assert signal.position_reasoning is not None
        assert "0.50" in signal.position_reasoning
        assert "0.80" in signal.position_reasoning

    def test_clamped_to_ceiling(self):
        regime = _Regime(position_size_multiplier=1.5)
        news = _News(position_multiplier=1.5)
        signal = _analyse(self.detector, regime=regime, news=news)
        assert signal.position_multiplier <= 1.5
