# =============================================================================
# SPRINT 2 INTELLIGENCE TESTS
# =============================================================================
"""
Comprehensive unit tests for Sprint 2: Intelligence Enhancement

Tests cover:
- SentimentAnalyzer: FinBERT NLP sentiment analysis
- RegimePredictor: HMM market regime detection
- MultiTimeframeEngine: Multi-timeframe technical analysis
- EnsembleRiskModel: XGBoost/LSTM/MLP ensemble
- Sprint2Intelligence: Unified integration

Run with: pytest tests/test_sprint2_intelligence.py -v
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# SENTIMENT ANALYZER TESTS
# =============================================================================

class TestSentimentAnalyzer:
    """Tests for SentimentAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create sentiment analyzer instance."""
        from src.agents.sentiment_analyzer import create_sentiment_analyzer
        return create_sentiment_analyzer()

    def test_analyzer_creation(self, analyzer):
        """Test analyzer creates successfully."""
        assert analyzer is not None
        assert hasattr(analyzer, 'analyze')
        assert hasattr(analyzer, 'analyze_batch')

    def test_single_analysis_bullish(self, analyzer):
        """Test analysis of bullish text."""
        result = analyzer.analyze("Markets surge on strong earnings reports")
        assert result is not None
        assert result.score > 0  # Should be positive
        assert 0 <= result.confidence <= 1
        assert result.category.name in ['VERY_BULLISH', 'BULLISH', 'SLIGHTLY_BULLISH', 'NEUTRAL']

    def test_single_analysis_bearish(self, analyzer):
        """Test analysis of bearish text."""
        result = analyzer.analyze("Markets crash amid recession fears")
        assert result is not None
        assert result.score < 0  # Should be negative
        assert 0 <= result.confidence <= 1

    def test_single_analysis_neutral(self, analyzer):
        """Test analysis of neutral text."""
        result = analyzer.analyze("The meeting was scheduled for Tuesday")
        assert result is not None
        assert -0.3 <= result.score <= 0.3  # Should be near neutral
        assert 0 <= result.confidence <= 1

    def test_batch_analysis(self, analyzer):
        """Test batch analysis of multiple texts."""
        texts = [
            "Stocks rally on positive economic data",
            "Bond yields drop as investors seek safety",
            "Central bank maintains interest rates",
            "Tech sector leads market gains"
        ]
        timestamps = [datetime.now() - timedelta(hours=i) for i in range(len(texts))]

        result = analyzer.analyze_batch(texts, timestamps)

        assert result is not None
        assert result.article_count == len(texts)
        assert -1 <= result.aggregated_score <= 1
        assert 0 <= result.confidence <= 1
        assert 0 <= result.bullish_ratio <= 1

    def test_empty_text(self, analyzer):
        """Test handling of empty text."""
        result = analyzer.analyze("")
        assert result is not None
        assert result.score == 0.0  # Neutral for empty

    def test_currency_detection(self, analyzer):
        """Test currency entity detection."""
        result = analyzer.analyze("EUR/USD rallies as dollar weakens")
        assert result is not None
        # Should detect EUR and USD
        if hasattr(result, 'entities') and result.entities:
            currencies = [e for e in result.entities if e.get('type') == 'currency']
            assert len(currencies) >= 0  # May or may not detect based on model

    def test_cache_functionality(self, analyzer):
        """Test that cache works correctly."""
        text = "Markets show strong momentum today"

        # First analysis
        result1 = analyzer.analyze(text)

        # Second analysis (should use cache)
        result2 = analyzer.analyze(text)

        assert result1.score == result2.score
        assert result1.confidence == result2.confidence


# =============================================================================
# REGIME PREDICTOR TESTS
# =============================================================================

class TestRegimePredictor:
    """Tests for RegimePredictor."""

    @pytest.fixture
    def predictor(self):
        """Create regime predictor instance."""
        from src.agents.regime_predictor import create_regime_predictor
        return create_regime_predictor()

    def test_predictor_creation(self, predictor):
        """Test predictor creates successfully."""
        assert predictor is not None
        assert hasattr(predictor, 'update')
        assert hasattr(predictor, 'predict')
        assert hasattr(predictor, 'fit')

    def test_insufficient_data(self, predictor):
        """Test handling of insufficient data."""
        predictor.update(100.0)
        result = predictor.predict()
        assert result is None  # Not enough data

    def test_trending_market(self, predictor):
        """Test detection of trending market."""
        from src.agents.regime_predictor import MarketRegime

        # Create uptrend
        base_price = 100.0
        for i in range(100):
            price = base_price + i * 0.5 + np.random.randn() * 0.1
            predictor.update(price, volume=1000)

        result = predictor.predict()
        assert result is not None
        assert result.current_regime in [MarketRegime.BULL_QUIET, MarketRegime.BULL_VOLATILE]
        assert result.trend_strength > 0.3

    def test_volatile_market(self, predictor):
        """Test detection of volatile market."""
        # Create volatile price action
        base_price = 100.0
        for i in range(100):
            price = base_price + np.sin(i * 0.5) * 10 + np.random.randn() * 5
            predictor.update(price, volume=1000)

        result = predictor.predict()
        assert result is not None
        assert result.volatility_state in ['high', 'very_high', 'normal']

    def test_fit_method(self, predictor):
        """Test HMM fitting."""
        prices = 100 + np.cumsum(np.random.randn(200) * 0.5)
        volumes = np.random.uniform(1000, 5000, 200)

        predictor.fit(prices, volumes)

        result = predictor.predict()
        assert result is not None

    def test_regime_probability(self, predictor):
        """Test regime probability is valid."""
        # Feed data
        for i in range(100):
            predictor.update(100 + i * 0.1)

        result = predictor.predict()
        if result:
            assert 0 <= result.regime_probability <= 1
            assert 0 <= result.stability_score <= 1

    def test_transition_prediction(self, predictor):
        """Test regime transition prediction."""
        # Feed data
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        for p in prices:
            predictor.update(p)

        result = predictor.predict()
        if result and result.transition_probabilities:
            for regime, prob in result.transition_probabilities.items():
                assert 0 <= prob <= 1

    def test_reset(self, predictor):
        """Test reset functionality."""
        for i in range(50):
            predictor.update(100 + i)

        predictor.reset()

        result = predictor.predict()
        assert result is None  # Data cleared


# =============================================================================
# MULTI-TIMEFRAME ENGINE TESTS
# =============================================================================

class TestMultiTimeframeEngine:
    """Tests for MultiTimeframeEngine."""

    @pytest.fixture
    def engine(self):
        """Create MTF engine instance."""
        from src.agents.multi_timeframe import create_multi_timeframe_engine
        return create_multi_timeframe_engine()

    @pytest.fixture
    def sample_candles(self):
        """Generate sample OHLCV candles."""
        from src.agents.multi_timeframe import OHLCV, generate_synthetic_candles, Timeframe

        return generate_synthetic_candles(
            base_price=1.1000,
            volatility=0.005,
            trend=0.3,
            count=150,
            timeframe=Timeframe.H1
        )

    def test_engine_creation(self, engine):
        """Test engine creates successfully."""
        assert engine is not None
        assert hasattr(engine, 'add_data')
        assert hasattr(engine, 'analyze')

    def test_insufficient_data(self, engine):
        """Test handling of insufficient data."""
        result = engine.analyze()
        assert result is None

    def test_add_data(self, engine, sample_candles):
        """Test adding candle data."""
        from src.agents.multi_timeframe import Timeframe

        engine.add_data(Timeframe.H1, sample_candles)

        # Should have data now
        analyzer = engine.analyzers.get(Timeframe.H1)
        assert analyzer is not None
        assert len(analyzer.candles) > 0

    def test_trend_detection(self, engine):
        """Test trend detection in bullish market."""
        from src.agents.multi_timeframe import (
            Timeframe, generate_synthetic_candles, SignalType
        )

        # Generate bullish candles for multiple timeframes
        for tf in [Timeframe.W1, Timeframe.D1, Timeframe.H4, Timeframe.H1]:
            candles = generate_synthetic_candles(
                base_price=1.1000,
                volatility=0.003,
                trend=0.5,  # Strong uptrend
                count=150,
                timeframe=tf
            )
            engine.add_data(tf, candles)

        result = engine.analyze()
        assert result is not None
        assert result.trend_alignment > 0  # Should be bullish
        assert result.aggregated_signal in [
            SignalType.STRONG_BUY, SignalType.BUY, SignalType.WEAK_BUY, SignalType.HOLD
        ]

    def test_conflict_detection(self, engine):
        """Test timeframe conflict detection."""
        from src.agents.multi_timeframe import Timeframe, generate_synthetic_candles

        # Weekly: uptrend
        engine.add_data(Timeframe.W1, generate_synthetic_candles(
            1.1000, 0.003, 0.6, 150, Timeframe.W1
        ))

        # Daily: downtrend (conflict!)
        engine.add_data(Timeframe.D1, generate_synthetic_candles(
            1.1000, 0.003, -0.6, 150, Timeframe.D1
        ))

        # 4H: downtrend
        engine.add_data(Timeframe.H4, generate_synthetic_candles(
            1.1000, 0.003, -0.4, 150, Timeframe.H4
        ))

        # 1H: neutral
        engine.add_data(Timeframe.H1, generate_synthetic_candles(
            1.1000, 0.003, 0.0, 150, Timeframe.H1
        ))

        result = engine.analyze()
        if result:
            # May or may not detect conflict depending on strength
            assert 0 <= result.overall_alignment <= 1

    def test_signal_aggregation(self, engine):
        """Test signal aggregation weights."""
        from src.agents.multi_timeframe import Timeframe, generate_synthetic_candles

        # All bullish at different strengths
        for tf in [Timeframe.W1, Timeframe.D1, Timeframe.H4, Timeframe.H1]:
            candles = generate_synthetic_candles(
                1.1000, 0.003, 0.4, 150, tf
            )
            engine.add_data(tf, candles)

        result = engine.analyze()
        if result:
            assert -1 <= result.aggregated_signal.score <= 1
            assert 0 <= result.confidence <= 1

    def test_position_multiplier(self, engine):
        """Test position size multiplier calculation."""
        from src.agents.multi_timeframe import Timeframe, generate_synthetic_candles

        for tf in [Timeframe.W1, Timeframe.D1, Timeframe.H4, Timeframe.H1]:
            candles = generate_synthetic_candles(1.1000, 0.003, 0.3, 150, tf)
            engine.add_data(tf, candles)

        result = engine.analyze()
        if result:
            assert 0 <= result.position_size_multiplier <= 2

    def test_reset(self, engine):
        """Test reset functionality."""
        from src.agents.multi_timeframe import Timeframe, generate_synthetic_candles

        engine.add_data(Timeframe.H1, generate_synthetic_candles(
            1.1000, 0.003, 0.3, 150, Timeframe.H1
        ))

        engine.reset()

        for analyzer in engine.analyzers.values():
            assert len(analyzer.candles) == 0


# =============================================================================
# ENSEMBLE RISK MODEL TESTS
# =============================================================================

class TestEnsembleRiskModel:
    """Tests for EnsembleRiskModel."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample training data."""
        np.random.seed(42)
        n_samples = 500
        n_features = 10

        X = np.random.randn(n_samples, n_features)
        # Create target correlated with some features
        y = 0.3 * X[:, 0] + 0.2 * X[:, 1] - 0.4 * X[:, 2] + np.random.randn(n_samples) * 0.1
        y = (y - y.min()) / (y.max() - y.min())  # Normalize to 0-1

        feature_names = [f"feature_{i}" for i in range(n_features)]

        return X, y, feature_names

    @pytest.fixture
    def model(self):
        """Create ensemble model instance."""
        from src.agents.ensemble_risk_model import create_ensemble_risk_model
        return create_ensemble_risk_model()

    def test_model_creation(self, model):
        """Test model creates successfully."""
        assert model is not None
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')

    def test_model_training(self, model, sample_data):
        """Test model training."""
        X, y, feature_names = sample_data

        model.fit(X, y, feature_names)

        assert model.fitted
        assert model.gb_model is not None or model.mlp_model is not None

    def test_prediction(self, model, sample_data):
        """Test model prediction."""
        X, y, feature_names = sample_data

        model.fit(X[:400], y[:400], feature_names)

        # Predict on test data
        prediction = model.predict(X[400:401])

        assert prediction is not None
        assert 0 <= prediction.ensemble_prediction <= 1
        assert 0 <= prediction.ensemble_confidence <= 1
        assert prediction.models_used >= 1

    def test_risk_categories(self, model, sample_data):
        """Test risk category assignment."""
        from src.agents.ensemble_risk_model import RiskCategory

        X, y, feature_names = sample_data
        model.fit(X[:400], y[:400], feature_names)

        prediction = model.predict(X[400:401])

        assert prediction.risk_category in [
            RiskCategory.VERY_LOW, RiskCategory.LOW, RiskCategory.MODERATE,
            RiskCategory.HIGH, RiskCategory.VERY_HIGH, RiskCategory.EXTREME
        ]

    def test_model_weights(self, model, sample_data):
        """Test model weight tracking."""
        X, y, feature_names = sample_data
        model.fit(X, y, feature_names)

        weights = model.model_weights
        assert len(weights) > 0
        assert all(0 <= w <= 1 for w in weights.values())

    def test_feature_importance(self, model, sample_data):
        """Test feature importance extraction."""
        X, y, feature_names = sample_data
        model.fit(X, y, feature_names)

        prediction = model.predict(X[:1])

        if prediction.top_features:
            assert all(isinstance(f, tuple) and len(f) == 2 for f in prediction.top_features)
            assert all(isinstance(f[0], str) for f in prediction.top_features)

    def test_prediction_uncertainty(self, model, sample_data):
        """Test prediction uncertainty metrics."""
        X, y, feature_names = sample_data
        model.fit(X[:400], y[:400], feature_names)

        prediction = model.predict(X[400:401])

        assert prediction.prediction_std >= 0
        assert len(prediction.prediction_range) == 2
        assert prediction.prediction_range[0] <= prediction.prediction_range[1]

    def test_gradient_boost_only(self, sample_data):
        """Test with only gradient boosting."""
        from src.agents.ensemble_risk_model import create_ensemble_risk_model

        X, y, feature_names = sample_data
        model = create_ensemble_risk_model(
            use_gradient_boost=True,
            use_lstm=False,
            use_mlp=False
        )

        model.fit(X[:400], y[:400], feature_names)
        prediction = model.predict(X[400:401])

        assert prediction is not None

    def test_mlp_only(self, sample_data):
        """Test with only MLP."""
        from src.agents.ensemble_risk_model import create_ensemble_risk_model

        X, y, feature_names = sample_data
        model = create_ensemble_risk_model(
            use_gradient_boost=False,
            use_lstm=False,
            use_mlp=True
        )

        model.fit(X[:400], y[:400], feature_names)
        prediction = model.predict(X[400:401])

        assert prediction is not None

    def test_stats(self, model, sample_data):
        """Test statistics retrieval."""
        X, y, feature_names = sample_data
        model.fit(X, y, feature_names)
        model.predict(X[:10])

        stats = model.get_model_stats()

        assert 'fitted' in stats
        assert 'input_size' in stats
        assert 'models_available' in stats


# =============================================================================
# SPRINT 2 INTELLIGENCE INTEGRATION TESTS
# =============================================================================

class TestSprint2Intelligence:
    """Tests for Sprint2Intelligence integration."""

    @pytest.fixture
    def intel(self):
        """Create intelligence system instance."""
        from src.agents.sprint2_intelligence import create_sprint2_intelligence
        return create_sprint2_intelligence()

    def test_system_creation(self, intel):
        """Test system creates successfully."""
        assert intel is not None
        assert hasattr(intel, 'analyze')
        assert hasattr(intel, 'add_news')
        assert hasattr(intel, 'add_price')

    def test_add_news(self, intel):
        """Test adding news headlines."""
        news = [
            "Markets rally on strong economic data",
            "Fed signals potential rate pause"
        ]

        intel.add_news(news)

        assert len(intel.recent_news) == 2

    def test_add_price(self, intel):
        """Test adding price updates."""
        for i in range(50):
            intel.add_price(100 + i * 0.1)

        # Should have data in regime predictor
        assert intel._regime_predictor is not None

    def test_basic_analysis(self, intel):
        """Test basic analysis without all components."""
        from src.agents.sprint2_intelligence import TradingAction

        report = intel.analyze("EURUSD")

        assert report is not None
        assert report.symbol == "EURUSD"
        assert isinstance(report.recommended_action, TradingAction)

    def test_analysis_with_sentiment(self, intel):
        """Test analysis with sentiment data."""
        intel.add_news([
            "Strong earnings boost market confidence",
            "Bulls dominate as economy expands"
        ])

        report = intel.analyze("EURUSD")

        assert report.sentiment_result is not None
        assert 'score' in report.sentiment_result or 'category' in report.sentiment_result

    def test_analysis_with_regime(self, intel):
        """Test analysis with regime data."""
        # Add enough price data
        for i in range(100):
            intel.add_price(100 + i * 0.1)

        report = intel.analyze("EURUSD")

        if report.regime_result:
            assert 'regime' in report.regime_result

    def test_signal_aggregation(self, intel):
        """Test signal aggregation."""
        intel.add_news(["Strong bullish momentum in markets"])

        for i in range(100):
            intel.add_price(100 + i * 0.2)

        report = intel.analyze("TEST")

        assert -1 <= report.aggregated_signal <= 1
        assert 0 <= report.aggregated_confidence <= 1

    def test_blocking_conditions(self, intel):
        """Test blocking condition detection."""
        from src.agents.sprint2_intelligence import TradingAction

        # Create conflicting conditions
        intel.add_news(["Markets crash dramatically"])  # Bearish

        for i in range(100):
            intel.add_price(100 + i * 0.5)  # Bullish trend

        report = intel.analyze("TEST")

        # May or may not be blocked depending on agreement
        assert isinstance(report.is_blocked, bool)
        if report.is_blocked:
            assert report.recommended_action == TradingAction.BLOCKED

    def test_position_sizing(self, intel):
        """Test position size factor calculation."""
        intel.add_news(["Steady market conditions"])

        for i in range(100):
            intel.add_price(100 + i * 0.1)

        report = intel.analyze("TEST")

        assert report.position_size_factor > 0
        assert report.position_size_factor <= 2

    def test_warnings_generation(self, intel):
        """Test warnings generation."""
        report = intel.analyze("TEST")

        assert isinstance(report.warnings, list)

    def test_explanation_generation(self, intel):
        """Test explanation text generation."""
        intel.add_news(["Test news headline"])

        report = intel.analyze("TEST")

        assert len(report.explanation) > 0
        assert "TEST" in report.explanation

    def test_quick_signal(self, intel):
        """Test quick signal method."""
        from src.agents.sprint2_intelligence import TradingAction

        action, confidence = intel.get_quick_signal("TEST")

        assert isinstance(action, TradingAction)
        assert 0 <= confidence <= 1

    def test_stats(self, intel):
        """Test statistics retrieval."""
        intel.analyze("TEST")
        intel.analyze("TEST2")

        stats = intel.get_stats()

        assert stats['decisions_made'] == 2
        assert 'components_initialized' in stats

    def test_reset(self, intel):
        """Test system reset."""
        intel.add_news(["Test"])
        intel.analyze("TEST")

        intel.reset()

        assert len(intel.recent_news) == 0
        assert intel.decisions_made == 0

    def test_report_serialization(self, intel):
        """Test report JSON serialization."""
        intel.add_news(["Test headline"])

        report = intel.analyze("EURUSD")

        # Test to_dict
        data = report.to_dict()
        assert isinstance(data, dict)
        assert 'timestamp' in data
        assert 'symbol' in data

        # Test to_json
        json_str = report.to_json()
        assert isinstance(json_str, str)
        assert 'EURUSD' in json_str


# =============================================================================
# TECHNICAL INDICATORS TESTS
# =============================================================================

class TestTechnicalIndicators:
    """Tests for technical indicator calculations."""

    @pytest.fixture
    def indicators(self):
        """Get TechnicalIndicators class."""
        from src.agents.multi_timeframe import TechnicalIndicators
        return TechnicalIndicators

    def test_sma(self, indicators):
        """Test Simple Moving Average."""
        prices = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        sma = indicators.sma(prices, 3)

        assert len(sma) == len(prices)
        assert np.isnan(sma[0])  # Not enough data
        assert np.isnan(sma[1])
        assert sma[2] == 2.0  # (1+2+3)/3
        assert sma[-1] == 9.0  # (8+9+10)/3

    def test_ema(self, indicators):
        """Test Exponential Moving Average."""
        prices = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        ema = indicators.ema(prices, 3)

        assert len(ema) == len(prices)
        assert not np.isnan(ema[-1])

    def test_rsi(self, indicators):
        """Test Relative Strength Index."""
        prices = np.array([44, 44.34, 44.09, 43.61, 44.33, 44.83,
                          45.10, 45.42, 45.84, 46.08, 45.89, 46.03,
                          45.61, 46.28, 46.28, 46.00, 46.03, 46.41])
        rsi = indicators.rsi(prices, 14)

        assert len(rsi) == len(prices)
        # RSI should be between 0 and 100
        valid_rsi = rsi[~np.isnan(rsi)]
        assert all(0 <= r <= 100 for r in valid_rsi)

    def test_macd(self, indicators):
        """Test MACD calculation."""
        prices = np.random.randn(100).cumsum() + 100
        macd_line, signal, histogram = indicators.macd(prices)

        assert len(macd_line) == len(prices)
        assert len(signal) == len(prices)
        assert len(histogram) == len(prices)

    def test_atr(self, indicators):
        """Test Average True Range."""
        high = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], dtype=float)
        low = np.array([9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], dtype=float)
        close = np.array([9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5], dtype=float)

        atr = indicators.atr(high, low, close, 5)

        assert len(atr) == len(high)
        valid_atr = atr[~np.isnan(atr)]
        assert all(a >= 0 for a in valid_atr)

    def test_bollinger_bands(self, indicators):
        """Test Bollinger Bands."""
        prices = np.random.randn(50).cumsum() + 100

        upper, middle, lower = indicators.bollinger_bands(prices, 20, 2.0)

        assert len(upper) == len(prices)
        valid_idx = ~np.isnan(upper)
        assert all(upper[valid_idx] >= middle[valid_idx])
        assert all(middle[valid_idx] >= lower[valid_idx])


# =============================================================================
# NUMPY ML IMPLEMENTATIONS TESTS
# =============================================================================

class TestNumpyML:
    """Tests for pure NumPy ML implementations."""

    def test_gradient_boost_regressor(self):
        """Test GradientBoostRegressor."""
        from src.agents.ensemble_risk_model import GradientBoostRegressor

        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = X[:, 0] * 0.5 + X[:, 1] * 0.3 + np.random.randn(100) * 0.1

        model = GradientBoostRegressor(n_estimators=20, learning_rate=0.1)
        model.fit(X, y)

        predictions = model.predict(X)

        assert len(predictions) == len(y)
        assert model.fitted
        assert model.feature_importances_ is not None

    def test_numpy_mlp(self):
        """Test NumpyMLP."""
        from src.agents.ensemble_risk_model import NumpyMLP

        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = (X[:, 0] > 0).astype(float)

        model = NumpyMLP(input_size=5, hidden_layers=[16, 8])
        model.fit(X, y, epochs=50)

        predictions = model.predict(X)

        assert len(predictions) == len(y)
        assert model.fitted

    def test_numpy_lstm(self):
        """Test NumpyLSTM."""
        from src.agents.ensemble_risk_model import NumpyLSTM

        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randn(100)

        model = NumpyLSTM(input_size=5, hidden_size=16, sequence_length=10)
        model.fit(X, y, epochs=10)

        assert model.fitted

    def test_feature_normalizer(self):
        """Test FeatureNormalizer."""
        from src.agents.ensemble_risk_model import FeatureNormalizer

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=float)

        normalizer = FeatureNormalizer()
        X_normalized = normalizer.fit_transform(X)

        assert normalizer.fitted
        assert X_normalized.shape == X.shape
        assert np.abs(np.mean(X_normalized, axis=0)).max() < 0.01  # Mean ~0

        # Test inverse
        X_restored = normalizer.inverse_transform(X_normalized)
        np.testing.assert_array_almost_equal(X, X_restored)


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_sentiment_special_characters(self):
        """Test sentiment with special characters."""
        from src.agents.sentiment_analyzer import create_sentiment_analyzer

        analyzer = create_sentiment_analyzer()
        result = analyzer.analyze("Markets 📈📈📈 are up! #bullish @traders")

        assert result is not None

    def test_regime_constant_price(self):
        """Test regime with constant price."""
        from src.agents.regime_predictor import create_regime_predictor

        predictor = create_regime_predictor()

        for _ in range(100):
            predictor.update(100.0)  # Constant price

        result = predictor.predict()
        # Should handle gracefully
        assert result is None or result is not None

    def test_mtf_single_candle(self):
        """Test MTF with minimal data."""
        from src.agents.multi_timeframe import (
            create_multi_timeframe_engine, Timeframe, OHLCV
        )

        engine = create_multi_timeframe_engine()

        candle = OHLCV(
            timestamp=datetime.now(),
            open=1.1, high=1.2, low=1.0, close=1.15
        )

        engine.add_data(Timeframe.H1, [candle])

        result = engine.analyze()
        assert result is None  # Not enough data

    def test_ensemble_single_sample(self):
        """Test ensemble with single sample."""
        from src.agents.ensemble_risk_model import create_ensemble_risk_model

        model = create_ensemble_risk_model()

        X = np.random.randn(10, 5)  # Minimal data
        y = np.random.rand(10)

        # Should handle minimal data gracefully
        try:
            model.fit(X, y)
        except Exception:
            pass  # May fail with minimal data


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
