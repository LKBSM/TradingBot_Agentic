"""PERF-3 — the prompt-cache prefix guard.

The audit (docs/audits/AUDIT-perf-3-efficacite.md, §B.1) found the caching
decision right by accident: ``cache_block_for`` checked a flat 1024-token
threshold, while Haiku 4.5 — the model both the chatbot and the scanner
translator run on — needs **4096**, the highest minimum of the whole range.
It also measured the system text alone, ignoring the tool definitions that
render BEFORE it (``tools -> system -> messages``) and are part of the SAME
cached prefix.

The production prefix cleared 4096 anyway, but by ~4%, with nothing guarding
it. A prefix that drops below the minimum does not raise: Anthropic returns
``cache_creation_input_tokens: 0`` and the bill quietly goes up. Three text
reduction missions have already touched this prompt (TXT-1, UI-3, MIA-5), so
the margin needs a test, not a comment.
"""

from __future__ import annotations

import pytest

from src.intelligence.llm_cost_policy import (
    CACHE_MIN_TOKENS_FLOOR,
    CACHE_MIN_TOKENS_STRICTEST,
    CHARS_PER_TOKEN,
    cache_block_for,
    cache_min_tokens_for,
)


# =========================================================================== #
# cache_min_tokens_for — the per-model minimum
# =========================================================================== #
class TestCacheMinTokensForModel:
    def test_haiku_4_5_requires_4096(self):
        """The whole point of PERF-3: Haiku 4.5 is NOT the flat 1024 the code
        used to assume — it is the strictest minimum Anthropic publishes."""
        assert cache_min_tokens_for("claude-haiku-4-5") == 4096

    def test_dated_snapshot_resolves_to_its_family(self):
        """Production passes a dated id, not the bare family name."""
        assert cache_min_tokens_for("claude-haiku-4-5-20251001") == 4096

    def test_newer_models_have_lower_minimums(self):
        """The minimum is not monotonic across generations — a newer, pricier
        model caches a SHORTER prefix. This is what makes 'move the translator
        to Sonnet to get caching' a real (if traffic-dependent) option."""
        assert cache_min_tokens_for("claude-sonnet-5") == 1024
        assert cache_min_tokens_for("claude-opus-5") == 512

    @pytest.mark.parametrize("model", [None, "", "some-model-we-never-heard-of"])
    def test_unknown_model_falls_back_to_the_strictest(self, model):
        """Never claim a prefix caches on an unverified assumption."""
        assert cache_min_tokens_for(model) == CACHE_MIN_TOKENS_STRICTEST


# =========================================================================== #
# cache_block_for — prefix accounting and the emit decision
# =========================================================================== #
class TestCacheBlockPrefixAccounting:
    def test_tool_definitions_count_towards_the_prefix(self):
        """A system block short on its own can still clear the floor once the
        tool definitions rendered before it are counted. Ignoring them was the
        second half of the defect."""
        short_system = "x" * (200 * CHARS_PER_TOKEN)  # ~200 tokens alone

        assert cache_block_for(short_system, model="claude-opus-5") is None

        with_tools = cache_block_for(
            short_system,
            model="claude-opus-5",
            prefix_chars=400 * CHARS_PER_TOKEN,  # ~400 tokens of tool schemas
        )
        assert with_tools is not None
        assert with_tools["cache_control"] == {"type": "ephemeral"}
        # The block carries ONLY its own text — prefix_chars is an accounting
        # input, never content.
        assert with_tools["text"] == short_system

    def test_empty_prompt_never_gets_a_breakpoint(self):
        assert cache_block_for("", model="claude-haiku-4-5") is None

    def test_below_the_universal_floor_returns_none(self):
        assert cache_block_for("short", model="claude-haiku-4-5") is None

    def test_between_floor_and_model_minimum_still_emits(self):
        """Deliberate asymmetry (see CACHE_MIN_TOKENS_FLOOR): marking a prefix
        that turns out too short costs nothing — Anthropic ignores the marker —
        whereas withholding a marker from a prefix that WOULD have cached costs
        ~90% of it on every turn. Since the char/token ratio is an estimate, we
        emit and log rather than suppress on a guess."""
        text = "x" * ((CACHE_MIN_TOKENS_FLOOR + 100) * CHARS_PER_TOKEN)
        block = cache_block_for(text, model="claude-haiku-4-5")
        assert block is not None, "a marker below the model minimum is free, not harmful"
        assert block["cache_control"] == {"type": "ephemeral"}


# =========================================================================== #
# The guard that matters: the real production prefix
# =========================================================================== #
class TestProductionChatbotPrefixStillCaches:
    def _measure(self):
        from src.intelligence.chatbot.chatbot import (
            DEFAULT_MODEL,
            SYSTEM_PROMPT_STATIC,
            TOOL_SCHEMAS,
            _schemas_chars,
        )

        tools_chars = _schemas_chars(TOOL_SCHEMAS)
        total_chars = tools_chars + len(SYSTEM_PROMPT_STATIC)
        return DEFAULT_MODEL, tools_chars, total_chars, total_chars // CHARS_PER_TOKEN

    def test_chatbot_cached_prefix_clears_its_model_minimum(self):
        """THE regression guard. If this fails, the chatbot's system prompt or
        tool set shrank past the point where Anthropic will cache it — the
        marker is still sent, still accepted, and silently does nothing.

        Do not 'fix' this by lowering the threshold. Either restore the prefix
        length, or accept the loss deliberately and move the caching strategy
        (a model with a lower minimum, per the audit §B.3)."""
        model, tools_chars, total_chars, est_tokens = self._measure()
        required = cache_min_tokens_for(model)

        assert est_tokens >= required, (
            f"the cached prefix no longer reaches {model}'s minimum: "
            f"~{est_tokens} tokens estimated ({total_chars} chars = "
            f"{tools_chars} of tool schemas + system prompt) vs {required} required. "
            "Anthropic will ignore the cache_control marker WITHOUT any error, "
            "so this test is the only thing that notices."
        )

    def test_tool_schemas_are_a_material_share_of_the_prefix(self):
        """Guards the accounting itself: were the tool definitions ever dropped
        from the estimate again, this documents how much would go missing."""
        _, tools_chars, total_chars, _ = self._measure()
        assert tools_chars > 0
        assert tools_chars / total_chars > 0.25, (
            "tool schemas are a large slice of the cached prefix — omitting them "
            "from the estimate is what made the old threshold check wrong"
        )

    def test_build_system_still_marks_the_static_block(self):
        """End-to-end: the wire request is unchanged by PERF-3 — first system
        block still carries the breakpoint, the variable signal block still
        does not."""
        from src.intelligence.chatbot.chatbot import Chatbot

        bot = Chatbot(
            anthropic_client=object(),
            summary_provider=None,
            assembler=None,
            adversarial_filter=None,
            output_filter=None,
        )
        blocks = bot._build_system({"instruments_tracked": []})

        assert isinstance(blocks, list) and blocks
        assert blocks[0].get("cache_control") == {"type": "ephemeral"}
        assert "cache_control" not in blocks[-1], (
            "the variable signal_summary must stay AFTER the breakpoint"
        )


# =========================================================================== #
# PERF-3 (B-1) — condensation of the get_market_reading tool result
#
# `structure` is 95.9% of a MarketReading payload, and the whole thing used to
# travel into the conversation at full price on every tool call (a tool result
# sits after the cache breakpoint and changes every turn, so it can never be
# cached). Measured on the real stored readings: 11k chars on average, 38 913 at
# the worst — one tool call could cost more than the entire cached system prompt.
#
# Capping is only acceptable because the true counts travel with it. These tests
# pin that honesty contract, not just the size win.
# =========================================================================== #
class TestToolResultCondensation:
    def _reading(self, n_bos=48, n_ob=12):
        return {
            "schema_version": "2.0.0",
            "header": {"instrument": "XAUUSD", "timeframe": "M15"},
            "regime": {"trend": "bullish"},
            "conditions": {"x": 1},
            "events": {"news_upcoming": []},
            "structure": {
                "current_bos": None,
                "bos_events": [{"direction": "bullish", "i": i} for i in range(n_bos)],
                "order_blocks": [
                    {"id": f"OB{i}", "status": "active" if i % 4 == 0 else "consumed"}
                    for i in range(n_ob)
                ],
                "liquidity_pools": [],
            },
        }

    def test_long_lists_are_capped(self):
        from src.intelligence.chatbot.chatbot import (
            _TOOL_STRUCTURE_LIMITS,
            _condense_reading_for_tool,
        )

        out = _condense_reading_for_tool(self._reading())
        assert len(out["structure"]["bos_events"]) == _TOOL_STRUCTURE_LIMITS["bos_events"]
        assert len(out["structure"]["order_blocks"]) == _TOOL_STRUCTURE_LIMITS["order_blocks"]

    def test_true_counts_travel_with_the_capped_lists(self):
        """The honesty contract: a capped list is never silently short. Without
        this the model would miscount when asked 'how many order blocks?'."""
        from src.intelligence.chatbot.chatbot import _condense_reading_for_tool

        out = _condense_reading_for_tool(self._reading(n_bos=48, n_ob=12))
        assert out["structure_totals"]["bos_events"] == 48
        assert out["structure_totals"]["order_blocks"] == 12
        assert "bos_events" in out["_truncated"]["fields"]
        assert "structure_totals" in out["_truncated"]["note"]

    def test_active_zones_are_never_dropped_for_stale_ones(self):
        """A cap must remove history, never a live level."""
        from src.intelligence.chatbot.chatbot import _condense_reading_for_tool

        out = _condense_reading_for_tool(self._reading(n_ob=40))
        kept = out["structure"]["order_blocks"]
        assert all(ob["status"] == "active" for ob in kept), (
            "active order blocks must sort ahead of consumed ones before capping"
        )

    def test_short_lists_are_untouched_and_no_truncation_claimed(self):
        from src.intelligence.chatbot.chatbot import _condense_reading_for_tool

        out = _condense_reading_for_tool(self._reading(n_bos=3, n_ob=2))
        assert len(out["structure"]["bos_events"]) == 3
        assert "_truncated" not in out

    def test_non_structure_blocks_pass_through_untouched(self):
        """Header/regime/conditions/events are ~4% of the payload — nothing to
        win, plenty of fidelity to lose."""
        from src.intelligence.chatbot.chatbot import _condense_reading_for_tool

        src = self._reading()
        out = _condense_reading_for_tool(src)
        for key in ("header", "regime", "conditions", "events", "schema_version"):
            assert out[key] == src[key]

    def test_scalar_structure_fields_survive(self):
        from src.intelligence.chatbot.chatbot import _condense_reading_for_tool

        out = _condense_reading_for_tool(self._reading())
        assert "current_bos" in out["structure"]

    def test_a_payload_without_structure_is_returned_as_is(self):
        from src.intelligence.chatbot.chatbot import _condense_reading_for_tool

        assert _condense_reading_for_tool({"header": {}}) == {"header": {}}

    def test_the_source_payload_is_not_mutated(self):
        """The API and the chart get the FULL reading — only what travels into
        the conversation is capped."""
        from src.intelligence.chatbot.chatbot import _condense_reading_for_tool

        src = self._reading()
        _condense_reading_for_tool(src)
        assert len(src["structure"]["bos_events"]) == 48
