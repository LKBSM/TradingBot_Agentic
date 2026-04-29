# Eval 05 LLM — Refresh delta 2026-04-29

> Re-vérification post-Prompt-04 que les 5 priorités livrées 2026-04-25 (eval_05_llm_implementation) sont toujours en place et n'ont pas régressé.

## Vérifications

| Check | État vérifié | Résultat |
|---|---|---|
| P1 — System prompt ≥ 2 048 tok (cache Haiku threshold) | `tiktoken cl100k_base` sur `SMC_SYSTEM_PROMPT` | **2 840 tok** ✓ |
| P2 — Cascade Haiku→Sonnet supprimée | `grep _narrate_with_cascade src/intelligence/llm_narrative_engine.py` | absent ; seul `_narrate_single` (l.419) ✓ |
| P2 — Modèles à jour (4.5/4.6/4.7) | `DEFAULT_*_MODEL` constants | `claude-haiku-4-5-20251001`, `claude-sonnet-4-6`, `claude-opus-4-7` ✓ |
| P2 — Tier→model routing | `TIER_MODEL_MAP` (l.215) + `model_for_tier()` (l.234) | présents ✓ |
| P3 — Auto-fallback Template | `_fallback_engine = TemplateNarrativeEngine()` (sentinel_scanner.py:119) + `fallback_used` tag (l.466, 618, 630) | présent ✓ |
| P4 — Cache key sans `bar_timestamp` | `SemanticCache.generate_cache_key()` doc + `SCORE_BUCKET_PTS=5` | présent (semantic_cache.py:8, 104, 125) ✓ |
| P5 — Eval CI script | `scripts/eval_05_narratives.py` | présent ✓ |
| Tests LLM stack | `pytest tests/test_llm_narrative_engine.py tests/test_semantic_cache.py tests/test_template_narrative_engine.py tests/test_eval_05_narratives.py` | **99 passed** (11 s) ✓ |

## Impact Prompt 04 sur Prompt 05

* **Aucun**. Prompt 04 a touché `volatility_forecaster.py`, `volatility_lgbm.py`, `security.py`, `main.py` — aucun de ces modules n'est consommé par le pipeline LLM.
* Le défaut `VOL_MODE=har` ne change pas la sémantique du `vol_regime` consommé par le LLM (le forecaster expose toujours `forecast_atr` + `regime_state`, juste plus rapide à calculer).

## Note delta

* eval_05 initial : **4.5/10**
* post-implémentation 2026-04-25 : **~7.5/10** (cache effectif, fallback robuste, observabilité hit rate)
* refresh 2026-04-29 : **7.5/10 confirmé** — pas de régression, toutes les priorités tiennent. Le passage à 8.5+ nécessite :
  * Activation `NARRATIVE_MODE=llm` en prod (encore à `template` par défaut — le hook marketing « AI-powered » reste menteur tant qu'on n'a pas basculé)
  * Premier batch d'eval rubric Opus sur 50+ narratives réelles
  * Multi-langue FR (TAM FR-first du ICP eval_25)

## Action requise

Aucune. Le code est aligné avec la cible. Suite : Prompt 06 (Semantic Cache deep eval — sous-évalué).
