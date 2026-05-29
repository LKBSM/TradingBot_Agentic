"""Tests for scripts/eval_05_narratives.py — narrative quality CI gate."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.eval_05_narratives import (  # noqa: E402
    JUDGE_SYSTEM_PROMPT,
    JudgeScore,
    judge_narrative,
    offline_score,
    parse_args,
    run,
    sample_signals,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def synthetic_ohlcv() -> pd.DataFrame:
    """200-bar synthetic XAU M15 frame, deterministic."""
    n = 200
    dates = pd.date_range("2025-06-01", periods=n, freq="15min")
    close = 2400 + (pd.Series(range(n)) * 0.5)
    return pd.DataFrame({
        "timestamp": dates,
        "open": close + 0.2,
        "high": close + 1.5,
        "low": close - 1.5,
        "close": close,
        "volume": 1000.0,
    })


# =============================================================================
# 1. JUDGE PROMPT SHAPE
# =============================================================================

class TestJudgePrompt:
    def test_judge_prompt_lists_all_five_dimensions(self):
        for dim in ("FAITHFULNESS", "SMC_FRAMEWORK", "RISK_FRAME", "TONE", "ACTIONABILITY"):
            assert dim in JUDGE_SYSTEM_PROMPT, f"Judge prompt missing dimension {dim}"

    def test_judge_prompt_demands_strict_json(self):
        assert "STRICT JSON" in JUDGE_SYSTEM_PROMPT or "strict JSON" in JUDGE_SYSTEM_PROMPT.lower()
        for key in ("faithfulness", "smc_framework", "risk_frame", "tone", "actionability", "total"):
            assert key in JUDGE_SYSTEM_PROMPT


# =============================================================================
# 2. JUDGE SCORE PARSING
# =============================================================================

class TestJudgeScoreParsing:
    def test_parses_well_formed_json(self):
        text = '{"faithfulness":5,"smc_framework":4,"risk_frame":4,"tone":5,"actionability":3,"total":21,"rationale":"good"}'
        score = JudgeScore.from_judge_text(text)
        assert score.total == 21
        assert score.faithfulness == 5
        assert "good" in score.rationale

    def test_recomputes_total_from_subscores(self):
        # Even if 'total' field is wrong, we trust the recomputed sum.
        text = '{"faithfulness":3,"smc_framework":3,"risk_frame":3,"tone":3,"actionability":3,"total":99,"rationale":"x"}'
        score = JudgeScore.from_judge_text(text)
        assert score.total == 15

    def test_unparseable_returns_zero(self):
        score = JudgeScore.from_judge_text("not json at all")
        assert score.total == 0
        assert "unparseable" in score.rationale.lower() or "json" in score.rationale.lower()


# =============================================================================
# 3. SAMPLE SIGNALS
# =============================================================================

class TestSampleSignals:
    def test_returns_requested_count(self, synthetic_ohlcv):
        sigs = sample_signals(synthetic_ohlcv, n=10, seed=7)
        assert len(sigs) == 10

    def test_signals_have_required_attributes(self, synthetic_ohlcv):
        sigs = sample_signals(synthetic_ohlcv, n=3, seed=7)
        for s in sigs:
            assert s.symbol == "XAUUSD"
            assert s.signal_type.value in ("LONG", "SHORT")
            assert s.entry_price > 0
            assert s.stop_loss > 0
            assert s.take_profit > 0
            assert len(s.components) >= 3

    def test_too_short_dataframe_returns_empty(self):
        small = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=10, freq="15min"),
            "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 100,
        })
        assert sample_signals(small, n=20) == []

    def test_seed_is_deterministic(self, synthetic_ohlcv):
        a = sample_signals(synthetic_ohlcv, n=5, seed=123)
        b = sample_signals(synthetic_ohlcv, n=5, seed=123)
        assert [s.signal_id for s in a] == [s.signal_id for s in b]


# =============================================================================
# 4. OFFLINE SCORER
# =============================================================================

class TestOfflineScorer:
    def test_empty_narrative_scores_low(self):
        # An empty narrative should score well below the quality threshold (18/25);
        # the heuristic gives ~6/25 from default-low subscores.
        score = offline_score("payload", "")
        assert score.total < 10
        assert score.tone == 0
        assert score.actionability < 4

    def test_marketing_tone_penalised(self):
        narrative = (
            "Incredible explosive must-trade setup! Gold is going to the moon!"
        )
        score = offline_score("payload", narrative)
        assert score.tone <= 2

    def test_quality_narrative_scores_well(self):
        narrative = (
            "Gold at 2400 with bullish BOS confirmed in strong_uptrend regime. "
            "Bullish FVG and Order Block support entry. SL at 2380 (2xATR), "
            "R:R 2:1, invalidates on close below 2380."
        )
        score = offline_score("payload", narrative)
        assert score.total >= 18, f"expected high score, got {score.total}"


# =============================================================================
# 5. CLI / END-TO-END (offline mode, fast)
# =============================================================================

class TestEndToEnd:
    def test_offline_run_writes_report(self, tmp_path, synthetic_ohlcv, monkeypatch):
        data_csv = tmp_path / "ohlcv.csv"
        synthetic_ohlcv.to_csv(data_csv, index=False)
        out_dir = tmp_path / "report"

        # Force offline + low threshold so the test isn't gated by quality
        argv = [
            "--data", str(data_csv),
            "--n", "5",
            "--min-samples", "3",
            "--threshold", "0",
            "--out", str(out_dir),
            "--offline",
        ]
        args = parse_args(argv)
        exit_code = run(args)

        # Threshold=0 should always pass
        assert exit_code == 0
        # Exactly one report should be written
        reports = list(out_dir.glob("narratives_*.json"))
        assert len(reports) == 1

        payload = json.loads(reports[0].read_text(encoding="utf-8"))
        assert "aggregate" in payload
        assert payload["aggregate"]["n"] == 5
        assert payload["aggregate"]["passed"] is True
        assert "samples" in payload and len(payload["samples"]) == 5

    def test_run_returns_2_when_signals_below_floor(self, tmp_path):
        # Tiny dataframe → can't produce 3 samples
        small = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=10, freq="15min"),
            "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 100,
        })
        data_csv = tmp_path / "tiny.csv"
        small.to_csv(data_csv, index=False)

        args = parse_args([
            "--data", str(data_csv),
            "--n", "5",
            "--min-samples", "3",
            "--out", str(tmp_path / "report"),
            "--offline",
        ])
        assert run(args) == 2


# =============================================================================
# 6. JUDGE NARRATIVE WITH MOCKED CLIENT
# =============================================================================

class TestJudgeNarrative:
    def test_judge_narrative_passes_payload_and_text(self):
        """Judge call should send both the CSV payload and the narrative text."""
        client = MagicMock()
        response = MagicMock()
        content_block = MagicMock()
        content_block.text = (
            '{"faithfulness":4,"smc_framework":4,"risk_frame":4,"tone":4,'
            '"actionability":4,"total":20,"rationale":"solid"}'
        )
        response.content = [content_block]
        client.messages.create.return_value = response

        score = judge_narrative(
            client, "claude-opus-4-7",
            payload="sym=XAUUSD,dir=LONG,score=82",
            narrative="Gold at 2400. BOS bullish. SL 2380, R:R 2:1.",
        )
        assert score.total == 20
        # Verify the call shape
        call = client.messages.create.call_args
        assert call.kwargs["model"] == "claude-opus-4-7"
        user_msg = call.kwargs["messages"][0]["content"]
        assert "sym=XAUUSD" in user_msg
        assert "BOS bullish" in user_msg

    def test_judge_swallows_api_errors(self):
        client = MagicMock()
        client.messages.create.side_effect = RuntimeError("API down")
        score = judge_narrative(client, "claude-opus-4-7", "payload", "narrative")
        # Should return a JudgeScore with the error captured, not raise.
        assert score.total == 0
        assert "judge error" in score.rationale.lower()

    def test_judge_empty_narrative_short_circuits(self):
        client = MagicMock()
        score = judge_narrative(client, "claude-opus-4-7", "payload", "")
        assert score.total == 0
        assert "empty" in score.rationale.lower()
        # Crucially, no API call should have been made.
        client.messages.create.assert_not_called()
