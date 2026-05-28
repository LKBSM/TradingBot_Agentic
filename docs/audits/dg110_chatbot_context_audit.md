# DG-110 — Chatbot context injection audit

**Date** : 2026-05-27
**Sprint** : 3 (Chatbot pilier solidifié)
**Reviewer** : Internal (auto-mode pass)
**Verdict** : ✅ **Context injection complete — no DG-110 finition required**

---

## Scope

The brief asks whether the production chatbot injects the **full InsightSignalV2 context** (8 components + uncertainty + structure + regime + vol + event + history) into the LLM prompt, or whether a `context_builder.py` finition is required.

## Method

Reviewed two file paths that bridge the InsightSignalV2 contract to the Anthropic Messages API:

1. `webapp/app/api/chat/route.ts` — SSE chat endpoint (Anthropic SDK).
2. `webapp/lib/chat/signal-summary.ts` — deterministic serializer used by (1).

Cross-referenced the serializer fields against `src/api/insight_signal_v2.py` (Pydantic v2.1.0) and `webapp/types/insight.ts` (TS mirror).

## Findings

`signal-summary.ts::buildSignalSummary()` walks **every documented section** of the InsightSignalV2 contract:

| Section | Fields serialized | Status |
|---|---|---|
| A. Identity | `instrument`, `timeframe`, `direction`, `conviction_0_100`, `conviction_label`, `created_at_utc`, `valid_until_utc` | ✅ |
| D. Uncertainty | `conformal_lower`, `conformal_upper`, `coverage_alpha`, `empirical_coverage`, `n_calibration` | ✅ |
| E. Structure | `bos_level`, `bos_event_age_bars`, `choch_present`, `fvg_zone`, `fvg_size_atr`, `ob_zone`, `ob_strength`, `retest_state`, `structural_invalidation` | ✅ |
| F. Regime | `hmm_label`, `hmm_posterior`, `bocpd_changepoint_prob`, `expected_run_length`, `jump_ratio`, `regime_gate_decision` | ✅ |
| G. Volatility | `regime`, `forecast_atr_pips`, `naive_atr_pips`, `forecast_vs_naive_pct`, `confidence_interval_pips`, `is_fallback` | ✅ |
| H. Event | `news_blackout_active`, `next_event_label`, `next_event_in_minutes`, `sentiment_score`, `sentiment_confidence`, `session` | ✅ |
| I. 8 components | `breakdown_components[*].name`, `contribution`, `weight_max`, `reasoning` | ✅ |
| J. Historical | `profit_factor`, `profit_factor_ci95`, `hit_rate_observed`, `similar_setups_n`, `empirical_coverage`, `backtest_window` | ✅ |
| Narrative | `narrative_short` | ✅ |

The serializer also applies `cache_control: { type: 'ephemeral' }` to the signal context block, so the Anthropic prompt cache hits on repeated questions about the same insight — keeping marginal cost on the cached portion ~10% of base.

## Aspirational fields intentionally omitted

`liquidity_zone_upper`, `liquidity_zone_lower`, and (when null) `historical_stats` are gracefully skipped rather than printed as "null". This is the right behaviour — surfacing "null" to the LLM invites the LLM to comment on its own absence.

## Gaps vs. Sprint 3 brief

The brief lists "8 composantes + uncertainty + structure + regime + vol + event + history" — **all 7 categories are covered**. No `context_builder.py` finition is required. The next concrete Sprint 3 items are:

- DG-111 — system prompt already directs the model to translate the conformal interval and CI95 bootstrap to natural language (system prompt rule 3). No prompt change required for V1; can be enriched with worked examples if smoke tests reveal poor calibration.
- DG-112 — adversarial test suite is **missing**; this is the bulk of Sprint 3 work.
- DG-114 — suggested questions UX hooks are absent in the route; needs a small TS module.
- DG-042 — `DEFAULT_MODEL = 'claude-haiku-4-5-20251001'` is hard-coded; tier-routing requires reading the subscriber tier and mapping to Haiku/Sonnet/Opus.
- Forbidden tokens post-processing — not implemented; needs a validator on the streamed response.
- Session memory — already present via `history: ReadonlyArray<{role, content}>` param (up to 6 turns); meets the 5-turn requirement.

## Decision

S3.1 audit closed. S3.2 ("DG-110 finition context_builder.py") is **resolved as not-required** — the existing `signal-summary.ts` covers the brief. Sprint 3 effort shifts to DG-042/111/112/114 and forbidden-token filtering.

## Pointer for future work

If we ever need a **Python-side** serializer (e.g. for a backend-driven chatbot reachable via `/api/v1/chat` instead of the Next.js route), the canonical fields list is in this audit; the TS implementation can be mirrored in `src/api/chat/context_builder.py` in a half-day.
