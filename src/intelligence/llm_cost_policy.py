"""LLM cost optimisation policy — Sprint LLM-2B.8.

Centralises three cost-reduction levers that production Anthropic
LLM callers must apply:

1. **Tier-based model routing** — Haiku-first for cheap operations
   (Q&A simple, FREE tier), Sonnet for narrative, Opus only on
   explicit user opt-in. The pricing-aware ``pick_model(...)``
   returns ``{model, in_price, out_price, justification}``.

2. **Prompt caching metadata** — Anthropic offers a ~90% discount on
   cached reads of a prefix marked with ``cache_control: {type:
   "ephemeral"}``. ``cache_block_for(...)`` returns the right Anthropic
   message block. Two things it now gets right (PERF-3):

   - The **minimum cacheable prefix is per-model**, not a flat 1024.
     Haiku 4.5 — what the chatbot and the scanner translator both run
     on — needs **4096** tokens, the highest of the whole range. Below
     the minimum the marker is *silently ignored*: no error, just
     ``cache_creation_input_tokens: 0``.
   - The prefix is **tools + system**, not the system text alone. A
     request renders as ``tools -> system -> messages``, so a breakpoint
     on a system block caches the tool definitions with it. Callers pass
     what precedes the block via ``prefix_chars``.

3. **Batch API flag** — for offline eval runs Anthropic's batch
   endpoint is 50% off. ``should_batch(context)`` returns True for
   eval/CI/backfill workloads, False for live user requests.

Pricing as of 2026-01 (USD per 1M tokens, refresh manually):
    Haiku 3.5  : in $0.25  out $1.25
    Sonnet 4.6 : in $3     out $15
    Opus 4.7   : in $15    out $75
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# Token cost table — USD per 1M tokens. Updated manually when Anthropic
# changes pricing; CI lints the constant against the docs URL.
MODEL_PRICING: dict[str, dict] = {
    "claude-haiku-3-5":  {"in":  0.25, "out":  1.25},
    "claude-haiku-4-5":  {"in":  0.50, "out":  2.50},
    "claude-sonnet-4-6": {"in":  3.0,  "out": 15.0},
    "claude-opus-4-7":   {"in": 15.0,  "out": 75.0},
}

# Prompt caching — the minimum cacheable PREFIX, per model.
#
# Anthropic creates a cache entry only when the prefix reaching the
# breakpoint is at least this long. Below it the ``cache_control`` marker
# is SILENTLY ignored — no error is raised, the request simply comes back
# with ``cache_creation_input_tokens: 0``. The minimum is NOT monotonic
# across generations: the newest models have the lowest, and Haiku 4.5 has
# the highest of the entire range.
CACHE_MIN_TOKENS_BY_MODEL: dict[str, int] = {
    "claude-opus-5":     512,
    "claude-fable-5":    512,
    "claude-fable-5-1":  512,
    "claude-mythos-5":   512,
    "claude-mythos-5-1": 512,
    "claude-opus-4-8":   1024,
    "claude-sonnet-5":   1024,
    "claude-sonnet-4-6": 1024,
    "claude-sonnet-4-5": 1024,
    "claude-opus-4-7":   2048,
    "claude-haiku-3-5":  2048,
    "claude-opus-4-6":   4096,
    "claude-opus-4-5":   4096,
    "claude-haiku-4-5":  4096,
}

# What to require when the model is unknown or unnamed: the STRICTEST known
# minimum, so a guard/report never claims a prefix caches when it may not.
CACHE_MIN_TOKENS_STRICTEST = 4096

# The floor below which a breakpoint is pointless on EVERY current model.
#
# Why the emit decision uses this floor rather than the model's own minimum:
# the two failure modes are not symmetric. Marking a prefix that turns out to
# be too short costs nothing — Anthropic ignores the marker. NOT marking a
# prefix that would have cached costs ~90% of that prefix on every single
# turn. Since ``CHARS_PER_TOKEN`` is an estimate (and one that under-counts
# real French, ~3.3 chars/token), gating emission on the strict per-model
# minimum would suppress markers on prefixes that do in fact cache. So we
# emit whenever caching is possible at all, and *log* when the prefix is
# below the running model's own minimum — turning a silent miss into a
# visible one. ``usage.cache_read_input_tokens`` remains the ground truth.
CACHE_MIN_TOKENS_FLOOR = 512

# Backwards-compatible alias. Prefer ``cache_min_tokens_for(model)``.
CACHE_MIN_TOKENS = CACHE_MIN_TOKENS_STRICTEST

# Rough char-per-token heuristic (English/French). We don't ship the
# real tokenizer in this package — 4 chars/token is the documented
# Anthropic rule-of-thumb for prompt-size estimates. It UNDER-counts dense
# French (~3.3 chars/token), which is the safe direction for a guard: it
# warns early. ``client.messages.count_tokens`` is the exact answer.
CHARS_PER_TOKEN = 4


@dataclass(frozen=True)
class ModelPick:
    model: str
    in_price: float   # USD per 1M tokens
    out_price: float
    justification: str

    def estimate_cost(
        self, *, input_tokens: int, output_tokens: int
    ) -> float:
        return (
            (input_tokens / 1_000_000.0) * self.in_price
            + (output_tokens / 1_000_000.0) * self.out_price
        )


def pick_model(
    *,
    tier: str,
    task: str,
    user_override: Optional[str] = None,
) -> ModelPick:
    """Route a request to a model based on tier + task semantics.

    ``task`` is one of {"qa_simple", "narrative", "eval", "audit"}.
    ``tier`` is one of FREE/LITE/PRO/PRO_PLUS/B2B_BASIC/B2B_PRO.

    User overrides are honoured only if the user is on a tier that
    pays for the requested model (FREE can't force Opus).
    """
    t = tier.upper().replace("+", "_PLUS")

    # Hard floor: FREE always gets Haiku, regardless of task.
    if t == "FREE":
        return _pick("claude-haiku-3-5", "FREE tier — Haiku only")

    # Task-driven defaults
    if task == "qa_simple":
        chosen = "claude-haiku-4-5"
        reason = "Q&A simple — Haiku enough"
    elif task == "narrative":
        chosen = "claude-sonnet-4-6"
        reason = "narrative — Sonnet default"
    elif task == "eval":
        chosen = "claude-haiku-4-5"
        reason = "eval/CI — Haiku cheaper, batch-eligible"
    elif task == "audit":
        chosen = "claude-sonnet-4-6"
        reason = "audit — Sonnet for trace fidelity"
    else:
        chosen = "claude-haiku-4-5"
        reason = f"unknown task {task!r} — Haiku safe default"

    # Tier ceilings
    if t in {"LITE", "B2B_BASIC"}:
        # No Opus for LITE/B2B basic.
        if chosen == "claude-opus-4-7":
            chosen = "claude-sonnet-4-6"
            reason += " (Opus downgraded — LITE tier)"

    # User override — only if the tier can pay for it.
    if user_override is not None and user_override in MODEL_PRICING:
        if user_override == "claude-opus-4-7" and t not in {
            "PRO", "PRO_PLUS", "B2B_PRO", "INSTITUTIONAL"
        }:
            reason += f" (override {user_override} denied — tier {t})"
        else:
            chosen = user_override
            reason = f"user override → {user_override}"

    return _pick(chosen, reason)


def _pick(model: str, justification: str) -> ModelPick:
    p = MODEL_PRICING[model]
    return ModelPick(
        model=model,
        in_price=p["in"],
        out_price=p["out"],
        justification=justification,
    )


def cache_min_tokens_for(model: Optional[str]) -> int:
    """Minimum cacheable prefix, in tokens, for ``model``.

    Accepts a dated snapshot id (``claude-haiku-4-5-20251001``) as well as the
    bare family id. An unknown or missing model yields the strictest known
    minimum, so nothing ever *claims* to cache on an unverified assumption.
    """
    if not model:
        return CACHE_MIN_TOKENS_STRICTEST
    name = str(model).strip().lower()
    best: Optional[str] = None
    for key in CACHE_MIN_TOKENS_BY_MODEL:
        if name == key or name.startswith(key + "-"):
            # Longest match wins so "claude-fable-5-1" never resolves via
            # "claude-fable-5".
            if best is None or len(key) > len(best):
                best = key
    return CACHE_MIN_TOKENS_BY_MODEL[best] if best else CACHE_MIN_TOKENS_STRICTEST


#: Models already reported as too short to cache, so the warning below is
#: emitted once per (model, required) instead of on every single turn.
_CACHE_SHORTFALL_REPORTED: set = set()


def cache_block_for(
    system_prompt: str,
    *,
    model: Optional[str] = None,
    prefix_chars: int = 0,
) -> Optional[dict]:
    """Return an Anthropic ``content`` block carrying ``cache_control``.

    ``prefix_chars`` is the size of everything rendered BEFORE this block that
    belongs to the same cached prefix — in practice the serialised tool
    definitions, because a request renders as ``tools -> system -> messages``.
    Leaving them out under-measures the prefix (PERF-3): the chatbot's tool
    schemas alone are ~7.5 kB, roughly 44% of its cached prefix.

    Returns ``None`` only when the prefix cannot cache on ANY current model.
    When it clears that floor but falls short of ``model``'s own minimum, the
    block is still returned and the shortfall is logged once — see
    ``CACHE_MIN_TOKENS_FLOOR`` for why emitting is the safe side of that call.
    """
    if not system_prompt:
        return None
    est_tokens = (len(system_prompt) + max(0, int(prefix_chars))) // CHARS_PER_TOKEN
    if est_tokens < CACHE_MIN_TOKENS_FLOOR:
        return None

    required = cache_min_tokens_for(model)
    if est_tokens < required:
        marker = (model or "<unknown>", required)
        if marker not in _CACHE_SHORTFALL_REPORTED:
            _CACHE_SHORTFALL_REPORTED.add(marker)
            logger.warning(
                "prompt cache prefix looks too short for %s: ~%d tokens estimated "
                "(%d chars incl. %d of prefix) vs a %d-token minimum — the "
                "cache_control marker may be silently ignored. Confirm with "
                "usage.cache_read_input_tokens; count exactly with "
                "client.messages.count_tokens.",
                model or "<unknown model>", est_tokens,
                len(system_prompt) + max(0, int(prefix_chars)),
                max(0, int(prefix_chars)), required,
            )
    return {
        "type": "text",
        "text": system_prompt,
        "cache_control": {"type": "ephemeral"},
    }


def should_batch(context: str) -> bool:
    """True for offline/eval/CI workloads → use Anthropic batch API (-50%)."""
    return context.lower() in {
        "eval", "ci", "batch", "backfill", "regression", "research"
    }


__all__ = [
    "CACHE_MIN_TOKENS",
    "CACHE_MIN_TOKENS_BY_MODEL",
    "CACHE_MIN_TOKENS_FLOOR",
    "CACHE_MIN_TOKENS_STRICTEST",
    "cache_min_tokens_for",
    "CHARS_PER_TOKEN",
    "MODEL_PRICING",
    "ModelPick",
    "cache_block_for",
    "pick_model",
    "should_batch",
]
