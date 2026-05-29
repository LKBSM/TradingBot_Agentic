# Eval 06 — Semantic Cache (hit rate, économies LLM)

> **Périmètre audité** : `src/intelligence/semantic_cache.py` (247 l), invocations dans `src/intelligence/llm_narrative_engine.py`, `src/intelligence/main.py`. Tests : `tests/test_semantic_cache.py`.
>
> **Date** : 2026-04-28 · **Branch** : `main` · **Mission** : valider l'efficacité du cache narratives, mesurer/estimer hit rate, identifier collisions, benchmark vs alternatives.

---

## 0. Résumé exécutif — Note **5.0 / 10**

| Axe | Note | Verdict |
|---|---|---|
| Pertinence sémantique | **2/10** | Le nom "SemanticCache" est trompeur — c'est un **cache hash strict** SHA-256, pas un cache sémantique avec embedding. Aucun calcul de similarité. |
| Conception clé | 6/10 | Le bucketing à 5 pts permet une convergence raisonnable sans sortir de la sémantique exacte de hash. |
| Hit rate prévisible | 5/10 | Pas d'instrumentation Prometheus. Estimation théorique 30-50 % sur volume stationnaire ; 0-15 % les premiers jours. |
| Sécurité collision | 7/10 | SHA-256[:16] = 64 bits → faible risque collision (~1 % à 10⁹ entrées). Mais mauvaise sémantique business : 2 setups identiques en components ⇒ **même narrative**, indépendamment du contexte temps/prix. |
| Multi-worker | **3/10** | Compteurs `_hits`/`_misses` en RAM par instance. Avec `uvicorn --workers > 1` les stats sont divergentes. |
| TTL & éviction | 5/10 | TTL 24 h hardcodé en RAM, pas configurable via env. Pas de LRU/LFU — juste `cleanup_expired` qu'aucune crontab n'invoque. |
| Persistance disque | 7/10 | SQLite WAL propre, schema_version migré, `INSERT OR REPLACE` idempotent. |
| Coût opérationnel | 8/10 | Pas d'API call externe (vs OpenAI embedding $0.0001/1k tok). 100 % local. |

**Verdict commercial** : utilisable en MVP perso (hit rate > 0 sur scenarii répétitifs). **Pas commercialisable** dès qu'on annonce "semantic cache" dans une fiche produit ou un BUSINESS_PLAN — c'est un hash cache. Le levier économique le plus fort est de **renommer** en `narrative_dedup_cache`, **rajouter** un vrai cache sémantique en complément (hit rate cible +20-30 pts), et **brancher** la cleanup_expired à un scheduler. Voir §5.

---

## 1. Cartographie code

```python
# src/intelligence/semantic_cache.py
class SemanticCache:
    SCHEMA_VERSION = 1                  # l.37
    SCORE_BUCKET_PTS = 5                # l.104
    TIER_DEFAULT = "UNKNOWN"            # l.105

    def __init__(db_path, ttl_seconds=86400)            # l.39
    def _get_connection() -> sqlite3.Connection         # l.58
    def _init_database()                                # l.67
    def _migrate(conn, from_v)                          # l.83
    @staticmethod _bucket(value, step)                  # l.108
    @classmethod generate_cache_key(signal) -> str      # l.114
    def get(cache_key) -> Optional[dict]                # l.156
    def put(cache_key, narrative_data)                  # l.198
    def cleanup_expired() -> int                        # l.212
    def get_stats() -> dict                             # l.229
    def size() -> int                                   # l.239
```

### 1.1 Clé de cache (`generate_cache_key`, l.114-154)

```python
parts = [
    f"sym={symbol}",
    f"dir={direction}",
    f"tier={tier}",
]
components.sort(key=lambda c: c.name)
for comp in components:
    score = _bucket(comp.weighted_score, 5)  # round to 5-pt bucket
    parts.append(f"{comp.name}={score:.1f}")

raw = "|".join(parts)
return hashlib.sha256(raw.encode()).hexdigest()[:16]
```

**Attributs utilisés** : symbol, signal_type, tier, 8 composants (BOS, FVG, OrderBlock, Regime, News, Volume, Momentum, RSI_Divergence) bucketés.

**`bar_timestamp` exclu volontairement** (l.125 et docstring l.8) — sinon hit rate ~ 0 %. Bonne décision.

### 1.2 Stockage SQLite (l.85-94)

```sql
CREATE TABLE narrative_cache (
    cache_key  TEXT PRIMARY KEY,
    data_json  TEXT NOT NULL,
    created_at REAL NOT NULL,
    hit_count  INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX idx_cache_created ON narrative_cache(created_at);
```

WAL mode + `synchronous=NORMAL` (l.63-64) — corrects pour usage low-write-rate. Index sur `created_at` permet `cleanup_expired` rapide.

### 1.3 TTL (l.42-45, l.177-185)

```python
ttl_seconds: int = 86400  # 24 hours
...
age = time.time() - row["created_at"]
if age > self._ttl:
    conn.execute("DELETE FROM narrative_cache WHERE cache_key = ?", (cache_key,))
    self._misses += 1
    return None
```

Lazy expiry au `get()` + helper `cleanup_expired` (l.212) jamais invoqué automatiquement.

---

## 2. Audit ligne à ligne — bugs & anti-patterns

### Bug n°1 — Nommage trompeur

Le nom `SemanticCache` suggère un cache à embedding (sentence-transformers, OpenAI ada-002, cosine similarity threshold). En réalité c'est un **dedup hash cache** strict. Conséquence directe :

- BUSINESS_PLAN §5.4 affirme "semantic shared cache 40 → 99 % selon scale"
- eval_24 unit economics modélise une marge brute reposant sur ce hit rate
- Marketing client peut annoncer "AI-powered cache" quand c'est un `hashlib.sha256()`

**Risque** : audit externe ou benchmark vs GPTCache → décrédibilisation. **Action** : renommer en `narrative_dedup_cache.py`, OU ajouter un vrai layer sémantique (cf. §5).

### Bug n°2 — Bucketing dépendant de l'ordre des composants

```python
components.sort(key=lambda c: getattr(c, "name", ""))   # l.145
```

Le sort par `name` est correct. **Mais** : si un composant est ajouté/supprimé en aval (e.g. on retire `momentum` du scoring), **toutes les clés cachées avant le change deviennent invalides** silencieusement. Pas de schema bump sur les clés. Cache divergence silencieuse au déploiement.

### Bug n°3 — `_hits`/`_misses` en RAM, non multi-worker

```python
self._hits = 0      # l.47
self._misses = 0    # l.48
```

`get_stats()` retourne ces compteurs. Avec `uvicorn --workers 4`, chaque worker a ses propres compteurs. `/health` ou `/metrics` retourne **les stats du worker qui répond**, pas la vue agrégée. Voir aussi `eval_21_performance.md` & `eval_10_15_findings.md` (mêmes patterns sur `SignalStore._current`).

**Fix** : compter en SQLite via `INSERT INTO cache_stats(metric, n) VALUES ('hit', 1) ON CONFLICT DO UPDATE SET n=n+1` ou exposer via Prometheus Counter.

### Bug n°4 — `cleanup_expired` jamais invoqué automatiquement

`cleanup_expired()` (l.212) supprime les TTL-out, mais grep montre 0 caller dans `src/`. Conséquence : le DB grossit à l'infini ; l'index sur `created_at` finit O(N). Lazy delete (l.180-184) couvre les `get()` mais pas les `put()` sur clés différentes.

À 86 400 s (24 h) × 96 bars/jour × 6 symboles × 50 % avec narrative = **~7 200 entrées max stationnaires** + zombies. Marginal côté disque mais non-zéro côté queries.

**Fix** : appeler `cleanup_expired` toutes les 1 h dans le scanner thread (`sentinel_scanner._run_loop`).

### Bug n°5 — Pas de versionnement de format

```python
"INSERT OR REPLACE INTO narrative_cache (cache_key, data_json, created_at, hit_count) VALUES (?, ?, ?, 0)"
```

`data_json` est un blob JSON sans `schema_version`. Si la structure d'une narrative change (e.g. ajout d'un champ `reasoning_steps`), les caches antérieurs renvoient une narrative incomplète. Pas de fail-fast.

**Fix** : envelopper `data_json` dans `{"v": 1, "payload": {...}}` et invalider à `v` mismatch.

### Bug n°6 — `INSERT OR REPLACE` reset `hit_count` à 0

```python
"INSERT OR REPLACE INTO narrative_cache "
"(cache_key, data_json, created_at, hit_count) "
"VALUES (?, ?, ?, 0)"   # l.207
```

Si `put()` est appelé sur une clé existante, `hit_count` est **remis à 0** ainsi que `created_at` (refresh TTL). Comportement OK pour TTL refresh, **mais** le hit_count perd l'info historique. Pas critique mais fausse les stats `get_stats`.

### Bug n°7 — Pas de protection contre clés malformées

```python
def get(self, cache_key: str) -> Optional[Dict[str, Any]]:
    cur = conn.execute("SELECT data_json, created_at, hit_count FROM narrative_cache WHERE cache_key = ?", (cache_key,))
```

Aucune validation longueur ou pattern. Si un caller passe un cache_key construit ailleurs (via API public ?), pas d'échec rapide. Bénin tant que la classe est consommée uniquement en interne.

### Bug n°8 — Pas de compression `data_json`

Une narrative typique = 250-500 tokens output ≈ 1.5-3 kB JSON. À 7 200 entrées stationnaires : 10-22 MB SQLite. Pas critique. Mais à scale 100 symboles + multi-tier, cumul 200 MB. Compression `zstd` divise par 5 — **option ROI faible mais zero-effort**.

### Bug n°9 — Multi-instance même DB → contention WAL

Le scanner et l'API peuvent ouvrir 2 instances `SemanticCache(db_path=…)` distinctes (vérifié dans `main.py:_calibrate_system` + `routes/narratives.py`). Chacune ouvre/ferme une connexion par appel (`_get_connection` l.58). Sur N=10 calls/sec, c'est **N×2 connect/disconnect/sec**. WAL gère correctement le concurrent read/write mais le coût per-call est ~2-5 ms. À échelle 1k MAU + 1 narrative/sec, le contexte process change pour rien.

**Fix** : connection pool partagé (cf. eval_10_15_team_audit.md quick-win #1, même pattern auth).

### Bug n°10 — Aucune mesure live du hit rate

`get_stats()` existe (l.229) mais n'est **pas exposé via `/metrics`** (Prometheus) ni `/health`. Pour mesurer en prod il faut SSH + Python REPL. Aveugle.

**Fix** : ajouter dans `routes/health.py` : `"semantic_cache": cache.get_stats()`.

---

## 3. Mesure / estimation hit rate

### 3.1 Modèle théorique

Sur XAU M15 96 bars/jour × 6 jours/semaine ≈ 576 bars/semaine. Bars qui produisent un signal valide ≈ 5-15 % (cf. eval_05) → **30-90 narratives/semaine/symbole**.

Cardinalité de la clé :
- `symbol` : 1 (mono-symbole prod)
- `direction` : 2
- `tier` : 4 (PREMIUM/STANDARD/WEAK/INVALID, mais INVALID jamais publié → 3)
- `components` : 8 dimensions × 5-pt bucket. Chaque composant a une plage 0-25 (regime) ou 0-2 (rsi_div) → ~5 buckets en moyenne.

**Cardinalité totale ≈ 2 × 3 × 5⁸ = 2.3M** clés théoriques distinctes. **Mais distribution loin d'uniforme** : la grande majorité des signaux convergent vers un sous-ensemble (BOS=15, FVG=15, etc. quand `require_retest=True`). En pratique la cardinalité observée tournera autour de **50-200** clés sur 2 mois XAU.

### 3.2 Hit rate estimé

| Scénario | Clés émises / mois | Clés cache stationnaires | Hit rate estimé |
|---|---|---|---|
| Cold start mois 1 | 200 | 80 | **0-15 %** |
| Mois 2 stationnaire | 200 | 100 | **30-45 %** |
| Mois 3 stationnaire | 200 | 100 | **40-50 %** |
| Avec 6 symboles | 1200 | 500 | **30-40 %** |

**vs cible BUSINESS_PLAN 60 %** : non atteinte sans embedding sémantique additionnel.

### 3.3 Économies LLM ($)

Hyp. Haiku 4.5 ($1/$5 MTok in/out, sans cache_control), narrative typique = 1300 in + 250 out = $0.0013 + $0.00125 = **$0.00255/call**.

Volume mensuel : 200 narratives × 6 symboles = 1200 calls/mois (si NARRATIVE_MODE=llm activé).

| Hit rate | Calls effectifs LLM | Coût/mois |
|---|---|---|
| 0 % (actuel) | 1200 | $3.06 |
| 30 % | 840 | $2.14 |
| 50 % | 600 | $1.53 |
| 70 % | 360 | $0.92 |

**Marginal en personal-use** ($1-3/mois). **Non-marginal à scale 1k MAU** : 1200 × 1000 = 1.2M calls/mois → $3 060 sans cache vs $1 530 à 50 % hit. Économie annuelle = **$18 000** au break-even.

---

## 4. Collisions — analyse et risque

### 4.1 SHA-256[:16] = 64 bits

Anniversaire approx. : ~2³² entrées avant 50 % collision. Avec 100 entrées stationnaires : risque négligeable (< 1e-15).

### 4.2 Collisions sémantiques (mauvaise abstraction, pas mauvais hash)

**Cas concret** : 2 signaux LONG-PREMIUM sur XAU :
- Signal A : 09h00 UTC — BOS=15, FVG=12, regime=23, news=18, vol=8, mom=2.5, rsi_div=2 — bar = 2480$
- Signal B : 14h00 UTC (5h plus tard) — mêmes scores bucketés à 5 pts — bar = 2515$ (+1.4 %)

Ils produisent **la même clé** (puisque `bar_timestamp` exclu et components bucketés). Ils reçoivent **la même narrative**. Or :
- Le contexte de séance diffère (Asian close vs US open)
- Le prix exécution diffère (+35 $ = +20 pips XAU)
- Les news intra-jour diffèrent

**Conséquence côté UX** : un abonné voit 2 messages Telegram à 5h d'intervalle avec **exactement la même narrative**. Mauvaise impression "AI scripted".

**Fix** :
1. Inclure `session` (asian/london/ny/after) — discriminateur léger, +20 % cardinalité, n'invalide pas la dedup intra-session.
2. Inclure `time_of_day_bucket` (4 buckets de 6h).
3. Ajouter un suffixe LLM "personnalisé" appendix avec timestamp humain.

---

## 5. Top 5 améliorations priorisées

| # | Amélioration | Effort | Impact hit rate | Impact économique | Priorité |
|---|--------------|:------:|:---------------:|:-----------------:|:--------:|
| **1** | **Renommer `narrative_dedup_cache` + ajouter vrai cache sémantique** (sentence-transformers all-MiniLM-L6 local + cosine 0.92 threshold) | M | +20-30 pts | +$10k/an à 1k MAU | P0 |
| **2** | **Ajouter `session` + `time_of_day_bucket` à la clé** | S | -5 pts hit (acceptable) | élimine collisions UX | P0 |
| **3** | **Brancher `cleanup_expired` au scanner** (toutes 1 h) | XS | 0 | -50 ms P95 lookup à long terme | P1 |
| **4** | **Exposer `get_stats()` dans `/health` et `/metrics`** | XS | 0 | observabilité requise | P0 |
| **5** | **Multi-worker safe** : compteurs en SQLite ou Prometheus Counter | S | 0 | requis pour scaling | P2 |

### 5.1 Détail levier #1 — vrai cache sémantique (P0)

```python
# narrative_cache_v2.py
from sentence_transformers import SentenceTransformer
import numpy as np

class HybridCache:
    def __init__(self, hash_cache: SemanticCache, encoder: SentenceTransformer, threshold=0.92):
        self.hash_cache = hash_cache    # exact-match layer
        self.encoder = encoder           # all-MiniLM-L6 (~30 MB, 384 dim)
        self.threshold = threshold
        self.embeddings_db = ...         # FAISS index ou SQLite vector

    def get(self, signal):
        # Tier 1 : exact hash hit
        key = SemanticCache.generate_cache_key(signal)
        narrative = self.hash_cache.get(key)
        if narrative:
            return narrative, "exact"

        # Tier 2 : semantic search
        signal_text = self._signal_to_text(signal)
        emb = self.encoder.encode(signal_text)
        nearest = self.embeddings_db.search(emb, k=1)
        if nearest.distance > self.threshold:
            return self.hash_cache.get(nearest.cache_key), "semantic"

        return None, "miss"
```

Coût : modèle 30 MB en RAM (acceptable), encode ~5-10 ms CPU. Hit rate cumulé estimé **50-70 %** vs 30-45 % actuel.

---

## 6. Plan d'exécution

### Quick wins (≤ 4 h cumulées)
- Brancher `cleanup_expired` (15 min)
- Exposer `get_stats()` dans `/health` (15 min)
- Ajouter `session` à la clé (1 h)
- Renommer `SemanticCache → NarrativeDedupCache` (30 min)
- Ajouter `data_json` versionning (30 min)

### Medium (1-2 jours)
- Vrai layer sémantique (sentence-transformers)
- Compteurs multi-worker safe (Redis ou SQLite agrégés)
- Connection pool partagé (réutilise patch eval_10_15)

### Long term (1 sem)
- LSH index pour O(log N) nearest neighbor à 10k+ entrées
- Eviction LRU + LFU bucketé par symbol
- A/B testing hit rate v1 vs v2 avec gating env var

---

## 7. KPIs cibles (post-implémentation)

| KPI | Avant | Après | Mesure |
|---|---|---|---|
| Hit rate exact | 30-45 % | 35-50 % (split) | `cache.get_stats()` |
| Hit rate sémantique | 0 % | 15-25 % | nouvelle metric |
| Hit rate cumulé | 30-45 % | **50-70 %** | sum of above |
| P95 lookup | 5-10 ms | 8-15 ms | (encode +5 ms acceptable) |
| Coût LLM mensuel @ 1k MAU | $3 060 | **$1 200** | facture Anthropic |
| Collisions UX user-reported | inconnu | < 1 % | feedback Discord |

---

## 8. Trade-offs

| Gain | Coût |
|---|---|
| +20 pts hit rate via sémantique | +30 MB RAM (modèle), +5 ms encode/call |
| Renommage cohérent | Refacto imports (10 fichiers max), risque de PR de churn |
| Compteurs multi-worker | Latence +2-3 ms (Redis hop) |
| `session` dans clé | -5 % hit exact, +20 % cardinalité, élimine collisions inter-session |

---

## 9. Benchmark vs alternatives

| Solution | Type | Hit rate typique | Coût | Multi-tenant | Self-hosted |
|---|---|---|---|---|---|
| **Smart Sentinel actuel** | Hash strict + bucketing | 30-45 % | $0 | non (RAM compteurs) | ✅ |
| **GPTCache** (ZilliZ) | Vector store + threshold | 50-70 % | self-host: free | ✅ | ✅ |
| **Redis Semantic Cache** | RediSearch + cosine | 55-75 % | $20-200/mo Redis Cloud | ✅ | partiel |
| **Portkey** | Managed | 40-65 % | $0.001/cache hit | ✅ | ❌ (managed) |
| **LangChain LLMCache** | Memory + SQLite | 25-50 % | $0 | non | ✅ |

**Notre niche** : avec hash strict + scoring déterministe, on est **dans la moyenne basse**. L'ajout sémantique nous met dans le tier GPTCache à coût similaire.

---

## 10. Note finale & recommandation

**Note : 5.0 / 10.**

Le code lui-même est propre (SQLite WAL, schema_version, INSERT OR REPLACE, lazy TTL). Mais le label "Semantic" est trompeur, le hit rate est plafonné à ~45 %, et les compteurs sont aveugles en multi-worker. C'est un cache MVP solo founder, pas un asset commercial.

**Recommandation** :
1. **P0 (4 h)** : renommer + brancher cleanup + exposer stats + ajouter session → +5 pts immédiat, transparence préservée
2. **P1 (2 j)** : layer sémantique → +20-25 pts, économie LLM × 2 à scale, parité GPTCache
3. **P2 (1 sem)** : multi-worker safe + LSH index → ready pour 10k MAU

Sans ces 3 phases, **éviter de mentionner "semantic cache" dans tout document marketing**.

---

### Annexes
- Code source : `src/intelligence/semantic_cache.py` (247 l)
- Tests : `tests/test_semantic_cache.py`
- Audit LLM amont : `reports/eval_05_llm.md`
- Memory entry : `memory/eval_05_llm_findings.md` (cache_control no-op, NARRATIVE_MODE=template par défaut)
