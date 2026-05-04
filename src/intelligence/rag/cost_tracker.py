"""LLM + embedding cost accounting — Sprint LLM-2B.8.

Two pricing concerns:

- **LLM tokens** (Anthropic Claude): per-model input/output rates per 1M
  tokens. The ``MODEL_PRICING`` table reflects 2026-Q2 list prices for
  the Claude family.
- **Embedding tokens** (Voyage AI): a single per-1M input rate, since
  embeddings have no output.

The tracker accumulates totals and per-tier breakdowns so we can:

- enforce daily quotas (FREE: hard cap; INSTITUTIONAL: soft cap with
  overage billing),
- surface ``cost_usd`` in API responses for transparency on B2B
  contracts,
- alert when monthly LLM cost-of-revenue crosses the kill criterion
  (>40% green, >60% red — see kill_criteria_board.md).

Thread-safe via a single ``threading.Lock``. Counters reset only on
explicit ``reset()`` — callers persist or aggregate elsewhere.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional


# Per-1M-token USD rates as of 2026-Q2.
# Source: anthropic.com/pricing + voyageai.com/pricing.
MODEL_PRICING: dict[str, dict[str, float]] = {
    # Anthropic Claude
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-opus-4-7": {"input": 15.00, "output": 75.00},
    # Test stubs — keep at 0 so tests can record without polluting totals.
    "stub": {"input": 0.0, "output": 0.0},
}

EMBEDDING_PRICING_PER_1M_TOKENS: dict[str, float] = {
    "voyage-3-large": 0.18,
    "voyage-3-lite": 0.02,
    "hash": 0.0,
}


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass
class LLMCallRecord:
    model: str
    tier: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


@dataclass
class EmbeddingCallRecord:
    model: str
    tier: str
    n_tokens: int
    cost_usd: float


@dataclass
class CostSummary:
    total_usd: float = 0.0
    llm_usd: float = 0.0
    embedding_usd: float = 0.0
    n_llm_calls: int = 0
    n_embedding_calls: int = 0
    by_tier_usd: dict[str, float] = field(default_factory=dict)
    by_model_usd: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------


class CostTracker:
    """Process-wide cost accumulator. Inject into routes for per-call recording."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._llm_records: list[LLMCallRecord] = []
        self._embedding_records: list[EmbeddingCallRecord] = []

    @staticmethod
    def _llm_cost(model: str, input_tokens: int, output_tokens: int) -> float:
        rates = MODEL_PRICING.get(model)
        if rates is None:
            # Unknown model: don't pretend to price. 0 is correct here
            # rather than an estimate, so the alert path stays trustworthy.
            return 0.0
        return (
            input_tokens / 1_000_000 * rates["input"]
            + output_tokens / 1_000_000 * rates["output"]
        )

    @staticmethod
    def _embedding_cost(model: str, n_tokens: int) -> float:
        rate = EMBEDDING_PRICING_PER_1M_TOKENS.get(model, 0.0)
        return n_tokens / 1_000_000 * rate

    def record_llm(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        tier: str = "unknown",
    ) -> LLMCallRecord:
        cost = self._llm_cost(model, input_tokens, output_tokens)
        record = LLMCallRecord(
            model=model,
            tier=tier,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=round(cost, 6),
        )
        with self._lock:
            self._llm_records.append(record)
        return record

    def record_embedding(
        self,
        model: str,
        n_tokens: int,
        tier: str = "unknown",
    ) -> EmbeddingCallRecord:
        cost = self._embedding_cost(model, n_tokens)
        record = EmbeddingCallRecord(
            model=model, tier=tier, n_tokens=n_tokens, cost_usd=round(cost, 6)
        )
        with self._lock:
            self._embedding_records.append(record)
        return record

    def summary(self) -> CostSummary:
        with self._lock:
            llm_usd = sum(r.cost_usd for r in self._llm_records)
            emb_usd = sum(r.cost_usd for r in self._embedding_records)
            by_tier: dict[str, float] = {}
            by_model: dict[str, float] = {}
            for r in self._llm_records:
                by_tier[r.tier] = by_tier.get(r.tier, 0.0) + r.cost_usd
                by_model[r.model] = by_model.get(r.model, 0.0) + r.cost_usd
            for r in self._embedding_records:
                by_tier[r.tier] = by_tier.get(r.tier, 0.0) + r.cost_usd
                by_model[r.model] = by_model.get(r.model, 0.0) + r.cost_usd
            return CostSummary(
                total_usd=round(llm_usd + emb_usd, 6),
                llm_usd=round(llm_usd, 6),
                embedding_usd=round(emb_usd, 6),
                n_llm_calls=len(self._llm_records),
                n_embedding_calls=len(self._embedding_records),
                by_tier_usd={k: round(v, 6) for k, v in by_tier.items()},
                by_model_usd={k: round(v, 6) for k, v in by_model.items()},
            )

    def reset(self) -> None:
        with self._lock:
            self._llm_records.clear()
            self._embedding_records.clear()


__all__ = [
    "CostSummary",
    "CostTracker",
    "EMBEDDING_PRICING_PER_1M_TOKENS",
    "EmbeddingCallRecord",
    "LLMCallRecord",
    "MODEL_PRICING",
]
