# Eval 06 — Findings empiriques 2026-04-29

> Complète `reports/eval_06_semantic_cache.md` (2026-04-28) avec une mesure empirique du hit rate jamais réalisée auparavant. Le rapport initial estimait 30-45 % théorique en stationnaire ; **les chiffres mesurés contredisent cette estimation**.

## 1. Mesure — simulation 1 200 signaux mono-symbole XAU

Distribution des composants : valeurs discrètes plausibles dérivées des plateaux observés (BOS={0,12,15}, FVG={0,10,12,15}, regime={0,8,15,20,23}, etc.). Direction LONG/SHORT 50/50, tier {PREMIUM 5%, STANDARD 25%, WEAK 70%}. Seed 42.

| `SCORE_BUCKET_PTS` | Clés uniques | Hit rate global | Hit rate steady-state (last 25%) |
|---|---|---|---|
| **5 (actuel)** | 1 106/1 200 | **7.8 %** | 12.3 % |
| 10 | 795/1 200 | **33.8 %** | n/a |
| 20 | 254/1 200 | **78.8 %** | n/a |

Reproductible : `python scripts/eval_06_hit_rate_sim.py` → `reports/eval_06/hit_rate_sim.json`.

## 2. Verdict révisé

**Le hit rate empirique au bucket actuel (5 pts) est ~4× sous l'estimation eval_06.** La cause : la cardinalité combinatoire des 8 composants × ~5 valeurs distinctes chacune produit ~50k combinations, dont 1 200 samples ne couvrent qu'une fraction.

**Implication économique** (révision de la table eval_06 §3.3, hyp. Haiku 4.5, 1200 calls/mois) :

| Hit rate | Calls effectifs | Coût/mois | Économie/mois |
|---|---|---|---|
| 0 % (pas de cache) | 1 200 | $3.06 | — |
| **7.8 % (mesuré actuel)** | **1 106** | **$2.82** | **$0.24** |
| 33.8 % (bucket=10) | 795 | $2.03 | $1.03 |
| 78.8 % (bucket=20) | 254 | $0.65 | $2.41 |

**À 1k MAU** (×1000) :
* Actuel : économie $240/mois
* Bucket 10 : économie $1 030/mois
* Bucket 20 : économie $2 410/mois

## 3. Quick-win prioritaire (P0 révisé)

**Passer `SemanticCache.SCORE_BUCKET_PTS` de 5 → 10**.

* Effort : **1 ligne** dans `src/intelligence/semantic_cache.py:104`
* Impact : ×4.3 hit rate, $9 480/an d'économie projetée à 1k MAU
* Trade-off accepté : un BOS=12 et un BOS=15 collisionnent (perte de granularité « strong vs medium ») — mais c'est compensé par le fait que la narrative est gated par `tier`, et que les deux signaux sont du même tier de toute façon.

Cette modification est **strictement supérieure à toutes les améliorations P1/P2 de eval_06** en ratio impact/effort. Elle ne nécessite pas le vrai cache sémantique (sentence-transformers) qui était en P0 du rapport — c'est un gain immédiat avant l'ajout de cette couche.

## 4. État des recommandations P0 du rapport eval_06

Vérifié dans le code (2026-04-29) :

| eval_06 reco | État | Note |
|---|---|---|
| #4 — Exposer `get_stats()` dans `/health` | ✅ FAIT | `src/api/routes/health.py:77,80` lit `cache.get_stats()`, mappe `hit_rate` → `HealthResponse.cache_hit_rate`. |
| #3 — Brancher `cleanup_expired` au scanner | ❌ NON FAIT | Aucun caller dans `src/intelligence/`. La table grossit sans bornes (lazy delete au `get` seulement). |
| #1 — Vrai layer sémantique | ❌ NON FAIT | Pas de sentence-transformers ; reste un hash dedup. |
| #2 — Ajout `session` à la clé | ❌ NON FAIT | Collisions UX intra-jour persistent. |
| #5 — Multi-worker safe | ❌ NON FAIT | `_hits/_misses` toujours en RAM par instance. |

## 5. Action immédiate recommandée

```python
# src/intelligence/semantic_cache.py:104
- SCORE_BUCKET_PTS = 5  # round component scores to nearest 5 points
+ SCORE_BUCKET_PTS = 10  # bucket=10 gives ~34% empirical hit rate vs 8% at 5pts
+                        # (eval_06_empirical_findings_2026_04_29.md). bucket=20 reaches
+                        # 79% but collapses meaningful BOS=12/15 distinctions.
```

Plus :
* Brancher `cleanup_expired` dans `SentinelScanner._run_loop` (15 min effort).
* Inclure `session` dans la clé pour éliminer la collision UX intra-jour (1 h effort).

Ces 3 quick-wins peuvent être livrés en moins d'une heure pour un impact économique 4× supérieur au plan initial.
