"""Tests for the LLM-2B.7/8 + INFRA-2B.2/6 + RISK-2B.1/2 final batch."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from src.intelligence.forward_test_paper import (
    PaperPosition,
    PaperTradingHarness,
)
from src.intelligence.llm_cost_policy import (
    MODEL_PRICING,
    cache_block_for,
    pick_model,
    should_batch,
)
from src.intelligence.narrative_quality import (
    NarrativeQualityTracker,
    NarrativeRecord,
)
from src.risk.compliance_checker import ComplianceChecker
from src.risk.transparency_log import TransparencyLog


# ===========================================================================
# LLM-2B.7 — NarrativeQualityTracker
# ===========================================================================


class _Clock:
    def __init__(self, t0=1_000_000.0): self.t = t0
    def __call__(self): return self.t
    def advance(self, dt): self.t += dt


def test_quality_tracker_empty_summary_safe():
    t = NarrativeQualityTracker(clock=_Clock())
    s = t.summary()
    assert s["n"] == 0
    assert s["worst_5"] == []


def test_quality_tracker_basic_aggregation():
    clk = _Clock()
    t = NarrativeQualityTracker(clock=clk)
    for i in range(10):
        t.record(NarrativeRecord(
            generation_id=f"g-{i}",
            ts=clk(),
            faithfulness=0.95 - i * 0.05,
            hallucination=(i == 0),
            cost_usd=0.01,
            language="fr" if i % 2 == 0 else "en",
            latency_ms=100.0 + i * 10,
        ))
    s = t.summary()
    assert s["n"] == 10
    assert s["hallucination_rate"] == 0.1
    assert s["cost_total_usd"] == pytest.approx(0.1)
    assert len(s["worst_5"]) == 5
    # worst_5 is sorted ascending by faithfulness
    faiths = [w["faithfulness"] for w in s["worst_5"]]
    assert faiths == sorted(faiths)
    assert set(s["by_language"].keys()) == {"fr", "en"}


def test_quality_tracker_purges_old_records():
    clk = _Clock()
    t = NarrativeQualityTracker(window_seconds=10, clock=clk)
    t.record(NarrativeRecord(generation_id="old", ts=clk()))
    clk.advance(100)
    t.record(NarrativeRecord(generation_id="new", ts=clk()))
    s = t.summary()
    assert s["n"] == 1


def test_quality_tracker_rejects_bad_args():
    with pytest.raises(ValueError):
        NarrativeQualityTracker(window_seconds=0)
    with pytest.raises(ValueError):
        NarrativeQualityTracker(capacity=0)


# ===========================================================================
# LLM-2B.8 — Cost policy
# ===========================================================================


def test_pick_model_free_tier_always_haiku():
    pick = pick_model(tier="FREE", task="narrative")
    assert pick.model == "claude-haiku-3-5"


def test_pick_model_pro_narrative_gets_sonnet():
    pick = pick_model(tier="PRO", task="narrative")
    assert pick.model == "claude-sonnet-4-6"


def test_pick_model_eval_gets_haiku_for_batch():
    pick = pick_model(tier="INSTITUTIONAL", task="eval")
    assert pick.model == "claude-haiku-4-5"


def test_user_override_to_opus_denied_for_lite():
    pick = pick_model(tier="LITE", task="narrative", user_override="claude-opus-4-7")
    assert pick.model != "claude-opus-4-7"


def test_user_override_to_opus_allowed_for_pro_plus():
    pick = pick_model(tier="PRO_PLUS", task="narrative", user_override="claude-opus-4-7")
    assert pick.model == "claude-opus-4-7"


def test_estimate_cost_returns_positive():
    pick = pick_model(tier="PRO", task="narrative")
    c = pick.estimate_cost(input_tokens=1000, output_tokens=500)
    assert c > 0


def test_cache_block_returns_none_for_short_prompt():
    assert cache_block_for("short") is None


def test_cache_block_returns_block_for_long_prompt():
    long_prompt = "x" * (1024 * 4 + 100)  # >= 1024 tokens
    block = cache_block_for(long_prompt)
    assert block is not None
    assert block["cache_control"] == {"type": "ephemeral"}


def test_should_batch_true_for_eval_context():
    assert should_batch("eval") is True
    assert should_batch("CI") is True
    assert should_batch("live") is False


# ===========================================================================
# INFRA-2B.2 — PaperTradingHarness
# ===========================================================================


def test_paper_harness_long_target_hit():
    h = PaperTradingHarness()
    pos = h.enter(direction="LONG", entry_price=100, stop_price=98, target_price=104)
    # mark below entry, no exit
    assert h.mark(pos.position_id, 99.5) is None
    # mark at target → close
    outcome = h.mark(pos.position_id, 104.5)
    assert outcome is not None
    assert outcome.hit == "TARGET"
    assert outcome.realised_r == pytest.approx(2.0)  # 4/2


def test_paper_harness_short_stop_hit():
    h = PaperTradingHarness()
    pos = h.enter(direction="SHORT", entry_price=100, stop_price=102, target_price=96)
    outcome = h.mark(pos.position_id, 102.5)
    assert outcome.hit == "STOP"
    assert outcome.realised_r == pytest.approx(-1.0)


def test_paper_harness_rejects_invalid_direction():
    h = PaperTradingHarness()
    with pytest.raises(ValueError):
        h.enter(direction="UP", entry_price=100, stop_price=99, target_price=101)


def test_paper_harness_rejects_invalid_long_levels():
    h = PaperTradingHarness()
    with pytest.raises(ValueError):
        h.enter(direction="LONG", entry_price=100, stop_price=101, target_price=102)


def test_paper_harness_equity_curve_grows():
    h = PaperTradingHarness()
    p1 = h.enter(direction="LONG", entry_price=100, stop_price=98, target_price=104)
    h.mark(p1.position_id, 104.5)
    p2 = h.enter(direction="LONG", entry_price=100, stop_price=98, target_price=104)
    h.mark(p2.position_id, 97.5)  # stopped
    curve = h.equity_curve()
    assert len(curve) == 3  # genesis + 2 closes
    assert curve[-1][1] == pytest.approx(1.0)  # +2R - 1R = +1R


def test_paper_harness_stats_shape():
    h = PaperTradingHarness()
    p = h.enter(direction="LONG", entry_price=100, stop_price=98, target_price=104)
    h.mark(p.position_id, 104.5)
    s = h.stats()
    assert s["n_trades"] == 1
    assert s["win_rate"] == 1.0
    assert "max_drawdown_R" in s


def test_paper_harness_export_includes_disclaimer():
    h = PaperTradingHarness()
    exp = h.export_for_publication()
    assert "Démonstration" in exp["disclaimer"]
    assert "Smart Sentinel" in exp["disclaimer"]
    # JSON-serialisable
    json.dumps(exp)


# ===========================================================================
# RISK-2B.1 — TransparencyLog
# ===========================================================================


def test_transparency_log_first_commit_genesis(tmp_path):
    log = TransparencyLog(tmp_path / "log.jsonl")
    entry = log.commit({"stats": {"n_trades": 1, "total_R": 0.5}})
    assert entry.seq == 1
    assert entry.prev_log_hash == "0" * 64


def test_transparency_log_chain_verifies(tmp_path):
    log = TransparencyLog(tmp_path / "log.jsonl")
    log.commit({"stats": {"n_trades": 1, "total_R": 0.5}})
    log.commit({"stats": {"n_trades": 2, "total_R": -0.2}})
    log.commit({"stats": {"n_trades": 3, "total_R": 0.4}})
    ok, broken, reason = log.verify()
    assert ok is True
    assert broken is None


def test_transparency_log_tamper_detected(tmp_path):
    p = tmp_path / "log.jsonl"
    log = TransparencyLog(p)
    log.commit({"stats": {"n_trades": 1, "total_R": 0.5}})
    log.commit({"stats": {"n_trades": 2, "total_R": -0.2}})

    # Tamper with the second line's curve_hash.
    lines = p.read_text(encoding="utf-8").splitlines()
    d = json.loads(lines[1])
    d["equity_curve_hash"] = "deadbeef"
    lines[1] = json.dumps(d)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")

    ok, broken, reason = log.verify()
    assert ok is False
    assert broken == 2


def test_transparency_log_size_grows(tmp_path):
    log = TransparencyLog(tmp_path / "log.jsonl")
    assert log.size == 0
    log.commit({"stats": {"n_trades": 1, "total_R": 0.1}})
    assert log.size == 1


# ===========================================================================
# RISK-2B.2 — ComplianceChecker
# ===========================================================================


def test_compliance_clean_text_passes():
    c = ComplianceChecker()
    r = c.check("Configuration modérée détectée — analyse éducative.", language="fr")
    assert r.ok is True
    assert r.violations == []


def test_compliance_catches_french_buy_token():
    c = ComplianceChecker()
    r = c.check("Vous devriez acheter maintenant.", language="fr")
    assert r.ok is False
    assert any(v.kind == "token" for v in r.violations)


def test_compliance_catches_percent_return_claim():
    c = ComplianceChecker()
    r = c.check("Notre méthode produit +50%/mois sur XAU.", language="fr")
    assert r.ok is False
    detail_set = {v.detail for v in r.violations}
    assert "explicit_return_claim" in detail_set


def test_compliance_catches_guarantee_phrase():
    c = ComplianceChecker()
    r = c.check("Nous garantissons des profits constants.", language="fr")
    assert r.ok is False


def test_compliance_catches_english_buy_signal():
    c = ComplianceChecker()
    r = c.check("This is a clear buy signal for XAU.", language="en")
    assert r.ok is False
    assert any(v.detail == "buy signal" for v in r.violations)


def test_compliance_llm_judge_wired():
    calls = []

    def fake_judge(text, language):
        calls.append((text, language))
        return {"violations": [{"kind": "llm", "detail": "implicit recommendation", "snippet": text[:30]}]}

    c = ComplianceChecker().with_llm_judge(fake_judge)
    r = c.check("Setup intéressant ce matin.", language="fr")
    assert r.ok is False
    assert any(v.kind == "llm" for v in r.violations)
    assert calls == [("Setup intéressant ce matin.", "fr")]


def test_compliance_empty_text_passes():
    r = ComplianceChecker().check("")
    assert r.ok is True


# ===========================================================================
# INFRA-2B.6 — Backup/restore round-trip
# ===========================================================================


def test_backup_dr_snapshot_and_restore_roundtrip(tmp_path):
    from scripts.backup_dr import cmd_restore, cmd_snapshot, cmd_verify

    src = tmp_path / "data"
    src.mkdir()
    (src / "a.db").write_bytes(b"hello world")
    (src / "b.db").write_bytes(b"second file")
    (src / "audit_snapshot_2026.jsonl").write_text("{}\n")

    backups = tmp_path / "backups"
    rc = cmd_snapshot(str(src), str(backups))
    assert rc == 0

    # Find the produced archive
    archives = list(backups.glob("data_*.tar.gz"))
    assert len(archives) == 1

    rc = cmd_verify(str(archives[0]))
    assert rc == 0

    restored = tmp_path / "restored"
    rc = cmd_restore(str(archives[0]), str(restored))
    assert rc == 0

    # Files match
    assert (restored / "a.db").read_bytes() == b"hello world"
    assert (restored / "audit_snapshot_2026.jsonl").read_text() == "{}\n"


def test_backup_dr_refuses_non_empty_target(tmp_path):
    from scripts.backup_dr import cmd_restore, cmd_snapshot

    src = tmp_path / "data"
    src.mkdir()
    (src / "a.db").write_bytes(b"x")
    backups = tmp_path / "backups"
    cmd_snapshot(str(src), str(backups))
    archive = next(backups.glob("data_*.tar.gz"))

    target = tmp_path / "restored"
    target.mkdir()
    (target / "existing_file.txt").write_text("squat")

    rc = cmd_restore(str(archive), str(target))
    assert rc == 2
