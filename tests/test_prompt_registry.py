"""Tests for the LLM-2B.10 frozen prompt template registry."""

from __future__ import annotations

import dataclasses

import pytest

from src.intelligence.prompt_registry import (
    PromptRecord,
    PromptRegistry,
    _fingerprint,
    register_default_prompts,
)


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------


def test_fingerprint_is_16_hex_chars():
    fp = _fingerprint("hello")
    assert len(fp) == 16
    int(fp, 16)


def test_fingerprint_changes_with_body():
    assert _fingerprint("a") != _fingerprint("b")


def test_fingerprint_stable_for_same_body():
    assert _fingerprint("same") == _fingerprint("same")


# ---------------------------------------------------------------------------
# register() validation
# ---------------------------------------------------------------------------


def test_register_rejects_empty_template_id():
    reg = PromptRegistry()
    with pytest.raises(ValueError):
        reg.register("", 1, "body")
    with pytest.raises(ValueError):
        reg.register("   ", 1, "body")


def test_register_rejects_version_below_one():
    reg = PromptRegistry()
    with pytest.raises(ValueError):
        reg.register("x", 0, "body")
    with pytest.raises(ValueError):
        reg.register("x", -1, "body")


def test_register_rejects_empty_body():
    reg = PromptRegistry()
    with pytest.raises(ValueError):
        reg.register("x", 1, "")
    with pytest.raises(ValueError):
        reg.register("x", 1, "   ")


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------


def test_register_then_get_returns_record():
    reg = PromptRegistry()
    rec = reg.register("x", 1, "hello world")
    assert isinstance(rec, PromptRecord)
    assert rec.template_id == "x"
    assert rec.version == 1
    assert rec.sha256 == _fingerprint("hello world")
    assert rec.body == "hello world"

    got = reg.get("x")
    assert got is rec


def test_unknown_template_id_raises_keyerror():
    reg = PromptRegistry()
    with pytest.raises(KeyError):
        reg.get("nope")


def test_unknown_version_raises_keyerror():
    reg = PromptRegistry()
    reg.register("x", 1, "body")
    with pytest.raises(KeyError):
        reg.get("x", version=99)


# ---------------------------------------------------------------------------
# Latest-version semantics
# ---------------------------------------------------------------------------


def test_get_without_version_returns_highest():
    reg = PromptRegistry()
    reg.register("x", 1, "v1 body")
    reg.register("x", 2, "v2 body")
    rec = reg.get("x")
    assert rec.version == 2
    assert rec.body == "v2 body"


def test_get_with_version_pins_specific():
    reg = PromptRegistry()
    reg.register("x", 1, "v1")
    reg.register("x", 3, "v3")
    reg.register("x", 2, "v2")
    assert reg.get("x", version=1).body == "v1"
    assert reg.get("x", version=2).body == "v2"
    assert reg.get("x").version == 3


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------


def test_prompt_record_is_frozen():
    reg = PromptRegistry()
    rec = reg.register("x", 1, "body")
    with pytest.raises(dataclasses.FrozenInstanceError):
        rec.body = "tampered"


def test_re_register_same_body_is_idempotent():
    reg = PromptRegistry()
    a = reg.register("x", 1, "same body")
    b = reg.register("x", 1, "same body")
    assert a is b
    assert reg.versions("x") == [1]


def test_re_register_different_body_raises_unless_replace():
    reg = PromptRegistry()
    reg.register("x", 1, "original body")
    with pytest.raises(ValueError) as exc:
        reg.register("x", 1, "DIFFERENT body")
    assert "already registered" in str(exc.value)
    # replace=True allows overwrite (for ops migrations)
    new = reg.register("x", 1, "DIFFERENT body", replace=True)
    assert reg.get("x", version=1).body == "DIFFERENT body"
    assert new.sha256 != _fingerprint("original body")


# ---------------------------------------------------------------------------
# Listings
# ---------------------------------------------------------------------------


def test_versions_returns_sorted():
    reg = PromptRegistry()
    reg.register("x", 3, "v3")
    reg.register("x", 1, "v1")
    reg.register("x", 2, "v2")
    assert reg.versions("x") == [1, 2, 3]


def test_versions_empty_for_unknown_id():
    reg = PromptRegistry()
    assert reg.versions("nope") == []


def test_template_ids_returns_sorted_unique():
    reg = PromptRegistry()
    reg.register("zeta", 1, "z")
    reg.register("alpha", 1, "a")
    reg.register("alpha", 2, "a2")
    assert reg.template_ids() == ["alpha", "zeta"]


def test_records_returns_all_sorted():
    reg = PromptRegistry()
    reg.register("b", 1, "b1")
    reg.register("a", 2, "a2")
    reg.register("a", 1, "a1")
    keys = [(r.template_id, r.version) for r in reg.records()]
    assert keys == [("a", 1), ("a", 2), ("b", 1)]


# ---------------------------------------------------------------------------
# Audit shape
# ---------------------------------------------------------------------------


def test_audit_dict_omits_body():
    reg = PromptRegistry()
    reg.register("x", 1, "secret prompt body")
    audit = reg.to_audit_dict("x")
    assert audit == {
        "template_id": "x",
        "version": 1,
        "sha256": _fingerprint("secret prompt body"),
    }
    assert "body" not in audit


# ---------------------------------------------------------------------------
# Membership operator
# ---------------------------------------------------------------------------


def test_contains_by_id_and_by_id_version():
    reg = PromptRegistry()
    reg.register("x", 1, "body")
    assert "x" in reg
    assert ("x", 1) in reg
    assert ("x", 2) not in reg
    assert "nope" not in reg
    assert 42 not in reg  # non-string non-tuple is False


# ---------------------------------------------------------------------------
# render()
# ---------------------------------------------------------------------------


def test_render_no_placeholders_passes_through():
    reg = PromptRegistry()
    rec = reg.register("x", 1, "plain body")
    assert rec.render() == "plain body"


def test_render_substitutes_placeholders():
    reg = PromptRegistry()
    rec = reg.register("x", 1, "Hello {name}, tier={tier}")
    assert rec.render(name="alice", tier="STRATEGIST") == (
        "Hello alice, tier=STRATEGIST"
    )


def test_render_missing_key_raises():
    reg = PromptRegistry()
    rec = reg.register("x", 1, "Hello {name}")
    with pytest.raises(KeyError):
        rec.render()


def test_render_does_not_mutate_body():
    reg = PromptRegistry()
    rec = reg.register("x", 1, "Hello {name}")
    rec.render(name="alice")
    # Original body untouched.
    assert rec.body == "Hello {name}"
    assert rec.sha256 == _fingerprint("Hello {name}")


# ---------------------------------------------------------------------------
# register_default_prompts — production seeding
# ---------------------------------------------------------------------------


def test_register_default_prompts_seeds_all_four_languages():
    reg = PromptRegistry()
    register_default_prompts(reg)
    assert sorted(reg.template_ids()) == [
        "rag.system.de",
        "rag.system.en",
        "rag.system.es",
        "rag.system.fr",
    ]
    for lang in ("fr", "en", "de", "es"):
        rec = reg.get(f"rag.system.{lang}")
        assert rec.version == 1
        assert rec.body.strip() != ""


def test_register_default_prompts_is_idempotent():
    reg = PromptRegistry()
    register_default_prompts(reg)
    register_default_prompts(reg)
    # Each language still v1, no duplicates / no exception.
    for lang in ("fr", "en", "de", "es"):
        assert reg.versions(f"rag.system.{lang}") == [1]
