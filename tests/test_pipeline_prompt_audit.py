"""Tests for the LLM-2B.11 prompt-audit stamping on RAGResponse."""

from __future__ import annotations

from src.intelligence.prompt_registry import PromptRegistry, register_default_prompts
from src.intelligence.rag.chunking import Chunk
from src.intelligence.rag.embedders import HashEmbedder
from src.intelligence.rag.pipeline import RAGPipeline


def _seed_pipeline(*, language: str = "fr", registry=None) -> RAGPipeline:
    pipeline = RAGPipeline(
        embedder=HashEmbedder(dimension=32),
        language=language,
        prompt_registry=registry,
    )
    pipeline.ingest(
        [
            Chunk(
                text="Smart Sentinel AI vise XAU/USD M15.",
                source_id="src-1",
                chunk_index=0,
            ),
            Chunk(
                text="Le système opère sous EU 2024/2811.",
                source_id="src-2",
                chunk_index=0,
            ),
        ]
    )
    return pipeline


# ---------------------------------------------------------------------------
# When the registry is wired
# ---------------------------------------------------------------------------


def test_response_carries_prompt_audit_for_active_language():
    reg = register_default_prompts(PromptRegistry())
    pipeline = _seed_pipeline(language="fr", registry=reg)
    resp = pipeline.query("Quels actifs Smart Sentinel suit-il ?")
    assert resp.prompt_audit is not None
    assert resp.prompt_audit["template_id"] == "rag.system.fr"
    assert resp.prompt_audit["version"] == 1
    assert len(resp.prompt_audit["sha256"]) == 16
    # Body must NOT leak into the audit
    assert "body" not in resp.prompt_audit


def test_response_carries_correct_sha_per_language():
    reg = register_default_prompts(PromptRegistry())
    fr_resp = _seed_pipeline(language="fr", registry=reg).query("q")
    en_resp = _seed_pipeline(language="en", registry=reg).query("q")
    assert fr_resp.prompt_audit["sha256"] != en_resp.prompt_audit["sha256"]


def test_unknown_language_falls_back_to_english_audit():
    reg = register_default_prompts(PromptRegistry())
    pipeline = _seed_pipeline(language="ja", registry=reg)
    resp = pipeline.query("q")
    assert resp.prompt_audit is not None
    assert resp.prompt_audit["template_id"] == "rag.system.en"


# ---------------------------------------------------------------------------
# Legacy path — registry not wired
# ---------------------------------------------------------------------------


def test_no_registry_means_no_prompt_audit():
    pipeline = _seed_pipeline(language="fr", registry=None)
    resp = pipeline.query("q")
    assert resp.prompt_audit is None


def test_registry_without_default_prompts_returns_none():
    """A registry was created but no prompts loaded — graceful None."""
    empty = PromptRegistry()
    pipeline = _seed_pipeline(language="fr", registry=empty)
    resp = pipeline.query("q")
    assert resp.prompt_audit is None


# ---------------------------------------------------------------------------
# Audit cache-hit path also stamps
# ---------------------------------------------------------------------------


def test_cache_hit_path_also_stamps_prompt_audit():
    """The answer-cache short-circuit returns a separate RAGResponse;
    that path must also surface prompt_audit."""
    from src.intelligence.rag.cache import AnswerCache

    reg = register_default_prompts(PromptRegistry())
    pipeline = RAGPipeline(
        embedder=HashEmbedder(dimension=32),
        language="en",
        prompt_registry=reg,
        answer_cache=AnswerCache(max_size=8),
    )
    pipeline.ingest(
        [Chunk(text="hello world", source_id="s1", chunk_index=0)]
    )

    def fake_llm(system: str, user: str) -> str:
        return "stub answer"

    # Cold pass — populates cache.
    r1 = pipeline.query("hello", llm=fake_llm)
    assert r1.prompt_audit["template_id"] == "rag.system.en"

    # Hot pass — cache hit. prompt_audit must still be there.
    r2 = pipeline.query("hello", llm=fake_llm)
    assert r2.elapsed_seconds.get("cache_hit") == 1.0
    assert r2.prompt_audit == r1.prompt_audit
