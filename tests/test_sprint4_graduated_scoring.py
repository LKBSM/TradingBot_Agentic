"""
Sprint 4: Graduated BOS/FVG Scoring Tests
Verifies that BOS and FVG scores are graduated based on quality,
not binary full-weight-or-zero.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.intelligence.confluence_detector import ConfluenceDetector, SignalType


class TestGraduatedBOSScoring:
    """BOS scoring should vary based on CHOCH confirmation."""

    def setup_method(self):
        self.detector = ConfluenceDetector()
        self.bos_weight = self.detector.weights["bos"]

    def test_bos_with_choch_gets_full_weight(self):
        """BOS + CHOCH confirmation should get 100% of weight."""
        smc = {"BOS_SIGNAL": 1.0, "CHOCH_SIGNAL": 1.0}
        result = self.detector._score_bos(smc, SignalType.LONG, atr=5.0)
        assert result.weighted_score == pytest.approx(self.bos_weight, abs=0.01)

    def test_bos_without_choch_gets_partial_weight(self):
        """BOS without CHOCH should get 60% of weight."""
        smc = {"BOS_SIGNAL": 1.0, "CHOCH_SIGNAL": 0.0}
        result = self.detector._score_bos(smc, SignalType.LONG, atr=5.0)
        expected = 0.6 * self.bos_weight
        assert result.weighted_score == pytest.approx(expected, abs=0.01)

    def test_bos_wrong_direction_gets_zero(self):
        """BOS opposing signal direction should get 0."""
        smc = {"BOS_SIGNAL": -1.0, "CHOCH_SIGNAL": 0.0}
        result = self.detector._score_bos(smc, SignalType.LONG, atr=5.0)
        assert result.weighted_score == 0.0

    def test_bearish_bos_with_choch(self):
        """Bearish BOS + CHOCH should get full weight for SHORT."""
        smc = {"BOS_SIGNAL": -1.0, "CHOCH_SIGNAL": -1.0}
        result = self.detector._score_bos(smc, SignalType.SHORT, atr=5.0)
        assert result.weighted_score == pytest.approx(self.bos_weight, abs=0.01)

    def test_bearish_bos_without_choch(self):
        """Bearish BOS without CHOCH should get 60%."""
        smc = {"BOS_SIGNAL": -1.0, "CHOCH_SIGNAL": 0.0}
        result = self.detector._score_bos(smc, SignalType.SHORT, atr=5.0)
        expected = 0.6 * self.bos_weight
        assert result.weighted_score == pytest.approx(expected, abs=0.01)

    def test_bos_reason_mentions_choch(self):
        """Reasoning string should distinguish CHOCH-confirmed from continuation."""
        smc_choch = {"BOS_SIGNAL": 1.0, "CHOCH_SIGNAL": 1.0}
        smc_cont = {"BOS_SIGNAL": 1.0, "CHOCH_SIGNAL": 0.0}

        r_choch = self.detector._score_bos(smc_choch, SignalType.LONG)
        r_cont = self.detector._score_bos(smc_cont, SignalType.LONG)

        assert "CHOCH" in r_choch.reasoning
        assert "continuation" in r_cont.reasoning


class TestGraduatedFVGScoring:
    """FVG scoring should vary based on gap size relative to ATR."""

    def setup_method(self):
        self.detector = ConfluenceDetector()
        self.fvg_weight = self.detector.weights["fvg"]

    def test_large_fvg_gets_full_weight(self):
        """FVG >= 1.0 ATR should get full weight."""
        smc = {"FVG_SIGNAL": 1.0, "FVG_SIZE_NORM": 1.5}
        result = self.detector._score_fvg(smc, SignalType.LONG, atr=5.0)
        assert result.weighted_score == pytest.approx(self.fvg_weight, abs=0.01)

    def test_medium_fvg_gets_partial_weight(self):
        """FVG of 0.5 ATR should get ~65% weight."""
        smc = {"FVG_SIGNAL": 1.0, "FVG_SIZE_NORM": 0.5}
        result = self.detector._score_fvg(smc, SignalType.LONG, atr=5.0)
        # quality = 0.3 + 0.7 * 0.5 = 0.65
        expected = 0.65 * self.fvg_weight
        assert result.weighted_score == pytest.approx(expected, abs=0.5)

    def test_small_fvg_gets_minimum_weight(self):
        """Very small FVG (0.1 ATR) should get ~37% weight."""
        smc = {"FVG_SIGNAL": 1.0, "FVG_SIZE_NORM": 0.1}
        result = self.detector._score_fvg(smc, SignalType.LONG, atr=5.0)
        # quality = 0.3 + 0.7 * 0.1 = 0.37
        expected = 0.37 * self.fvg_weight
        assert result.weighted_score == pytest.approx(expected, abs=0.5)

    def test_no_fvg_gets_zero(self):
        """No FVG should get 0 score."""
        smc = {"FVG_SIGNAL": 0.0, "FVG_SIZE_NORM": 0.0}
        result = self.detector._score_fvg(smc, SignalType.LONG, atr=5.0)
        assert result.weighted_score == 0.0

    def test_opposing_fvg_gets_zero(self):
        """FVG opposing signal direction should get 0."""
        smc = {"FVG_SIGNAL": -1.0, "FVG_SIZE_NORM": 1.0}
        result = self.detector._score_fvg(smc, SignalType.LONG, atr=5.0)
        assert result.weighted_score == 0.0

    def test_fvg_fallback_without_atr(self):
        """Without ATR data, FVG should get full weight (backward compatible)."""
        smc = {"FVG_SIGNAL": 1.0, "FVG_SIZE_NORM": 0.0}
        result = self.detector._score_fvg(smc, SignalType.LONG, atr=0.0)
        # FVG_SIZE_NORM=0 and atr=0 → fallback to full weight
        assert result.weighted_score == pytest.approx(self.fvg_weight, abs=0.01)

    def test_bearish_fvg_scoring(self):
        """Bearish FVG should score for SHORT direction."""
        smc = {"FVG_SIGNAL": -1.0, "FVG_SIZE_NORM": 0.8}
        result = self.detector._score_fvg(smc, SignalType.SHORT, atr=5.0)
        assert result.weighted_score > 0
        # quality = 0.3 + 0.7 * 0.8 = 0.86
        expected = 0.86 * self.fvg_weight
        assert result.weighted_score == pytest.approx(expected, abs=0.5)

    def test_fvg_reason_includes_quality(self):
        """Reasoning string should include quality percentage."""
        smc = {"FVG_SIGNAL": 1.0, "FVG_SIZE_NORM": 0.5}
        result = self.detector._score_fvg(smc, SignalType.LONG, atr=5.0)
        assert "quality" in result.reasoning.lower() or "%" in result.reasoning


class TestGraduatedScoringIntegration:
    """Integration tests with full confluence analysis."""

    def setup_method(self):
        self.detector = ConfluenceDetector(min_score=0)

    def test_higher_quality_signals_score_higher(self):
        """Signal with CHOCH + large FVG should outscore BOS-only + small FVG."""
        smc_high = {
            "BOS_SIGNAL": 1.0, "CHOCH_SIGNAL": 1.0,
            "FVG_SIGNAL": 1.0, "FVG_SIZE_NORM": 1.5,
            "OB_STRENGTH_NORM": 0.5, "RSI": 55.0, "MACD_Diff": 0.5,
        }
        smc_low = {
            "BOS_SIGNAL": 1.0, "CHOCH_SIGNAL": 0.0,
            "FVG_SIGNAL": 1.0, "FVG_SIZE_NORM": 0.1,
            "OB_STRENGTH_NORM": 0.1, "RSI": 52.0, "MACD_Diff": 0.1,
        }

        sig_high = self.detector.analyze(smc_high, regime=None, news=None, price=1900, atr=5.0)
        sig_low = self.detector.analyze(smc_low, regime=None, news=None, price=1900, atr=5.0)

        assert sig_high is not None and sig_low is not None
        assert sig_high.confluence_score > sig_low.confluence_score

    def test_graduated_scores_are_not_binary(self):
        """BOS and FVG component scores should not be exactly 0 or max_weight."""
        smc = {
            "BOS_SIGNAL": 1.0, "CHOCH_SIGNAL": 0.0,
            "FVG_SIGNAL": 1.0, "FVG_SIZE_NORM": 0.5,
            "OB_STRENGTH_NORM": 0.0, "RSI": 50.0, "MACD_Diff": 0.0,
        }

        signal = self.detector.analyze(smc, regime=None, news=None, price=1900, atr=5.0)
        assert signal is not None

        bos_comp = next(c for c in signal.components if c.name == "BOS")
        fvg_comp = next(c for c in signal.components if c.name == "FVG")

        assert 0 < bos_comp.weighted_score < bos_comp.weight
        assert 0 < fvg_comp.weighted_score < fvg_comp.weight
