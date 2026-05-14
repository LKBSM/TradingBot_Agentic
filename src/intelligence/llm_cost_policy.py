"""LLM cost optimisation policy — Sprint LLM-2B.8.

Centralises three cost-reduction levers that production Anthropic
LLM callers must apply:

1. **Tier-based model routing** — Haiku-first for cheap operations
   (Q&A simple, FREE tier), Sonnet for narrative, Opus only on
   explicit user opt-in. The pricing-aware ``pick_model(...)``
   returns ``{model, in_price, out_price, justification}``.

2. **Prompt caching metadata** — Anthropic offers a 90% discount on
   cached system-prompt reads if the system prompt is ≥ 1024 tokens
   and marked with ``cache_control: {type: "ephemeral"}``. ``cache_
   block_for(system_prompt)`` returns the right Anthropic message
   block or ``None`` if the prompt is too short to benefit.

3. **Batch API flag** — for offline eval runs Anthropic's batch
   endpoint is 50% off. ``should_batch(context)`` returns True for
   eval/CI/backfill workloads, False for live user requests.

Pricing as of 2026-01 (USD per 1M tokens, refresh manually):
    Haiku 3.5  : in $0.25  out $1.25
    Sonnet 4.6 : in $3     out $15
    Opus 4.7   : in $15    out $75
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# Token cost table — USD per 1M tokens. Updated manually when Anthropic
# changes pricing; CI lints the constant against the docs URL.
MODEL_PRICING: dict[str, dict] = {
    "claude-haiku-3-5":  {"in":  0.25, "out":  1.25},
    "claude-haiku-4-5":  {"in":  0.50, "out":  2.50},
    "claude-sonnet-4-6": {"in":  3.0,  "out": 15.0},
    "claude-opus-4-7":   {"in": 15.0,  "out": 75.0},
}

# Prompt caching threshold — Anthropic only caches blocks ≥ 1024 tokens.
# Below that, the cache_control header is rejected.
CACHE_MIN_TOKENS = 1024

# Rough char-per-token heuristic (English/French). We don't ship the
# real tokenizer in this package — 4 chars/token is the documented
# Anthropic rule-of-thumb for prompt-size estimates.
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


def cache_block_for(system_prompt: str) -> Optional[dict]:
    """Return an Anthropic ``content`` block with ``cache_control`` set,
    or ``None`` if the prompt is too short to benefit from caching.
    """
    if not system_prompt:
        return None
    est_tokens = len(system_prompt) // CHARS_PER_TOKEN
    if est_tokens < CACHE_MIN_TOKENS:
        return None
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
    "CHARS_PER_TOKEN",
    "MODEL_PRICING",
    "ModelPick",
    "cache_block_for",
    "pick_model",
    "should_batch",
]
