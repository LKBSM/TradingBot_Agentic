"""Tests for LLM-2B.4 multi-language RAG prompt support.

Verifies:
- 4 system prompts present (FR/EN/DE/ES) with the same hard-rule structure
- Each prompt enforces UE 2024/2811 forbidden phrases in its language
- ``build_prompt_bundle`` dispatches correctly by language and falls back
  to EN on unknown tags
- Each prompt mandates source citations and "insufficient context" wording
"""

from __future__ import annotations

import pytest

from src.intelligence.rag.prompts import (
    SUPPORTED_LANGUAGES,
    SYSTEM_PROMPT_DE,
    SYSTEM_PROMPT_EN,
    SYSTEM_PROMPT_ES,
    SYSTEM_PROMPT_FR,
    SYSTEM_PROMPTS,
    build_prompt_bundle,
)


# ---------------------------------------------------------------------------
# Registry integrity
# ---------------------------------------------------------------------------


def test_supported_languages_covers_four():
    assert set(SUPPORTED_LANGUAGES) == {"fr", "en", "de", "es"}


def test_all_prompts_in_dispatch_table():
    assert SYSTEM_PROMPTS["fr"] is SYSTEM_PROMPT_FR
    assert SYSTEM_PROMPTS["en"] is SYSTEM_PROMPT_EN
    assert SYSTEM_PROMPTS["de"] is SYSTEM_PROMPT_DE
    assert SYSTEM_PROMPTS["es"] is SYSTEM_PROMPT_ES


def test_each_prompt_meaningful_length():
    for lang, prompt in SYSTEM_PROMPTS.items():
        assert len(prompt) > 800, f"{lang} prompt too short: {len(prompt)} chars"


# ---------------------------------------------------------------------------
# Anti-hallucination + compliance rules per language
# ---------------------------------------------------------------------------


def test_each_prompt_mentions_source_citation_token():
    """Every language must instruct the LLM to cite via [source:chunk_id]."""
    for lang, prompt in SYSTEM_PROMPTS.items():
        assert "[source:" in prompt, f"{lang} prompt missing [source:] template"


def test_each_prompt_has_insufficient_context_wording():
    """The 'do not guess, say insufficient' rule must be present in each lang."""
    expected_fragments = {
        "fr": "insuffisant",
        "en": "insufficient",
        "de": "unzureichend",
        "es": "insuficiente",
    }
    for lang, fragment in expected_fragments.items():
        assert fragment.lower() in SYSTEM_PROMPTS[lang].lower(), (
            f"{lang} prompt missing 'insufficient' equivalent"
        )


def test_each_prompt_blocks_calls_to_action():
    """UE 2024/2811: forbidden CTA phrases must be enumerated per language."""
    forbidden_per_lang = {
        "fr": ("achetez", "vendez"),
        "en": ("buy", "sell"),
        "de": ("kaufen", "verkaufen"),
        "es": ("compre", "venda"),
    }
    for lang, words in forbidden_per_lang.items():
        text = SYSTEM_PROMPTS[lang].lower()
        for w in words:
            assert w in text, f"{lang} prompt does not enumerate '{w}'"


def test_each_prompt_mentions_eu_finfluencer_regulation():
    """Compliance trace: every prompt cites EU 2024/2811."""
    for lang, prompt in SYSTEM_PROMPTS.items():
        assert "2024/2811" in prompt, f"{lang} prompt missing EU regulation citation"


def test_each_prompt_uses_neutral_setup_terminology():
    """Direction labels must be neutral (e.g. 'bullish setup', not 'buy').

    German uses the localised forms 'bullisch'/'bearisch' (with German
    adjective endings) rather than the English originals.
    """
    expected_labels = {
        "fr": ("haussier", "baissier"),
        "en": ("bullish", "bearish"),
        "de": ("bullish", "bearisch"),
        "es": ("alcista", "bajista"),
    }
    for lang, labels in expected_labels.items():
        text = SYSTEM_PROMPTS[lang].lower()
        for lbl in labels:
            assert lbl in text, f"{lang} prompt missing setup label '{lbl}'"


# ---------------------------------------------------------------------------
# build_prompt_bundle dispatch
# ---------------------------------------------------------------------------


def test_build_prompt_bundle_dispatches_de():
    chunks = [("a", "text", {})]
    bundle = build_prompt_bundle("Q", chunks, language="de")
    assert bundle.system is SYSTEM_PROMPT_DE


def test_build_prompt_bundle_dispatches_es():
    chunks = [("a", "text", {})]
    bundle = build_prompt_bundle("Q", chunks, language="es")
    assert bundle.system is SYSTEM_PROMPT_ES


def test_build_prompt_bundle_falls_back_to_en_on_unknown_lang():
    chunks = [("a", "text", {})]
    bundle = build_prompt_bundle("Q", chunks, language="zz")
    assert bundle.system is SYSTEM_PROMPT_EN


def test_build_prompt_bundle_default_is_french():
    """Backwards-compat: default language stays FR (existing callers)."""
    chunks = [("a", "text", {})]
    bundle = build_prompt_bundle("Q", chunks)
    assert bundle.system is SYSTEM_PROMPT_FR


@pytest.mark.parametrize("lang", ["fr", "en", "de", "es"])
def test_build_prompt_bundle_each_lang_returns_user_message(lang):
    chunks = [("a", "Some retrieved fact", {"type": "data", "label": "Source A"})]
    bundle = build_prompt_bundle("What is X?", chunks, language=lang)
    assert "What is X?" in bundle.user
    assert "[source:a]" in bundle.user
    assert bundle.cited_chunk_ids == ["a"]
