# Eval 09 Sentinel Scanner — Follow-up 2026-04-29

> Suivi post-Prompt-04 de eval_09_sentinel_scanner.md (note 6.5/10).

## 1. Corrections au rapport eval_09

| Affirmation eval_09 | État vérifié | Correction |
|---|---|---|
| « **Mono-symbole en l'état** (pas de MultiSymbolScanner malgré la docstring) » | Faux | `class MultiSymbolScanner` existe à `sentinel_scanner.py:803` avec `start()/stop()/get_stats()`. La phrase doit être révisée — MultiSymbol existe ; la note 3/10 sur scalabilité horizontale reste cependant valide pour d'autres raisons (single-process, GIL, pas de worker dédié par symbole). |

## 2. Impact Prompt 04 sur la latence du scanner

L'étape 6 de `_scan_once` (`VolForecaster.forecast`) bénéficie directement des changements Prompt 04 :

| Mode | Latence forecast pre-Prompt 04 | Latence forecast post-Prompt 04 |
|---|---|---|
| HAR (défaut maintenant) | 32-54 ms (P50/P95) | **inchangé** (32-54 ms) |
| LGBM | ~1 600 ms (legacy build_features non vectorisé) | ~190 ms (-88 % grâce à `_vectorized_event_proximity` + `_vectorized_regime_features`) |
| Hybrid | ~1 700 ms | ~190 ms (-89 %) |

**Conséquence pour le scanner** :
* Avec `VOL_MODE=har` (nouveau défaut) : pipeline P95 ~ **0.5-1 s** (LLM cache hit) ou **4-8 s** (LLM cache miss). Inchangé par Prompt 04 — HAR n'a pas régressé.
* Avec `VOL_MODE=lgbm` ou `hybrid` (fallback ML) : avant Prompt 04 le forecast à lui seul ajoutait 1.6 s au scan. **Désormais 190 ms — gain de ~1.4 s sur chaque scan ML**. La marge entre `_scan_once` et le `time.sleep(60)` redevient confortable.

## 3. Latence du pipeline — décomposition empirique post-Prompt 04

(Estimations conservées de eval_09 §1.1, mises à jour pour la ligne forecast.)

```
1. DataProvider.get_ohlcv (lookback=200)           ~3-15 ms cached
2. validate_ohlcv                                   ~5 ms
3. SmartMoneyEngine.analyze (Numba JIT)             ~200-500 ms
4. RegimeAgent.analyze (HMM inference)              ~50-150 ms
5. NewsAgent.evaluate_news_impact                   ~5-30 ms
6. VolForecaster.forecast (HAR default)             ~32-54 ms     ← inchangé
   (LGBM/Hybrid après Prompt 04)                   ~190 ms        ← -88 %
7. ConfluenceDetector.analyze                       ~1-3 ms
8. SignalStateMachine.on_bar                        ~< 1 ms
9. SemanticCache.get / LLM.generate                 ~10 ms hit / 2-8 s miss
10. SignalStore.publish + Notifier.send_signal      ~200-500 ms (sync HTTP)
                                                    ─────────────────────────
P95 par scan (HAR + LLM cache miss)                ~3-9 s
P95 par scan (HAR + cache hit / template fallback) ~0.5-1.5 s
P95 par scan (LGBM/Hybrid + LLM cache miss)        ~3-9.2 s (vs 4.5-10 s pré-P04)
```

**Verdict** : la latence dominante reste **l'appel LLM** (cache miss = 2-8 s), pas le forecast. Prompt 04 retire le risque ML mais ne change pas le critical path.

## 4. Issues persistantes (non-traitées par Prompt 04)

Re-confirmées dans le code :

* **Polling fixe `time.sleep(60)`** → tampon 0-60 s qui empêche le SLA « 30 s post-close » d'être tenu **garanti**. Solution = event-driven trigger (MT5 `OnBar` ou WebSocket) — encore non implémenté.
* **Backpressure 2/10** : aucune queue Redis/SQLite entre `_scan_once` et `Notifier.send_signal`. Si Telegram tombe, le circuit breaker ouvre, les signaux sont perdus (pas de TTL, pas de dedup en file).
* **Trace ID E2E absent** : le pipeline `DataProvider → … → Telegram` ne propage pas un `trace_id` unique. Debug post-mortem difficile sur erreur sectorielle. À chaîner avec eval_16 observability.

## 5. Note delta

* eval_09 (2026-04-28) : 6.5/10
* post-Prompt-04 : **6.8/10** — gain léger uniquement parce que ML modes sortent de la zone rouge latence. Les bloqueurs structurels (polling, backpressure, trace_id) sont identiques.

## 6. Action immédiate (P0 du sprint suivant)

1. **Brancher `cleanup_expired` SemanticCache** dans `_run_loop` (15 min, P0 d'eval_06 toujours pending).
2. **Ajouter `trace_id` dans `_scan_once` → propager jusqu'à `Notifier.send_signal`** (1 h). Hook `eval_16` observability.
3. **Bencher empiriquement P95** du scanner sur 24 h de replay simulé pour valider les estimations §3 — **aucun bench scanner E2E n'a été exécuté à ce jour** dans les eval_09/eval_16/eval_21 (lacune méthodologique).

Bench E2E manquant = **risque d'écart entre les estimations rapportées et la réalité prod**. À combler avant tout SLA contractuel.
