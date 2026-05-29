# Plan de Commercialisation — Catégorie 17 : Caching & Performance

> **Périmètre** : SemanticCache (LLM narratives), application caching (Redis), DB query optimization, async I/O end-to-end, profiling, latency budgets, throughput, memory footprint, GC tuning.
>
> **Sources** : `reports/eval_06_semantic_cache.md`, `reports/eval_06_empirical_findings_2026_04_29.md`, `reports/eval_21_performance.md`, `reports/eval_10_15_team_audit.md`, code (`src/intelligence/semantic_cache.py`, `src/intelligence/llm_narrative_engine.py`, `src/api/signal_store.py`, `src/intelligence/security.py`, `src/api/routes/{health,narratives,signals}.py`, `infrastructure/docker-compose.yml`).
>
> **Objectif commercial** : >10k MAU, hit rate cache ≥40 %, P95 < 200 ms sur tous endpoints SaaS, multi-worker safe, coût LLM ÷ 2 à 1k MAU.
>
> **Date** : 2026-05-21 · **Branch** : `institutional-overhaul`

---

## 1. État actuel (Audit) — cache hit 7.8 %, sync I/O, single-worker

### 1.1 SemanticCache (hash dedup, mal nommé)

| Aspect | Constat | Référence |
|---|---|---|
| Type réel | Hash dedup SHA-256[:16] sur 8 composants bucketés, **pas** sémantique (aucun embedding) | `src/intelligence/semantic_cache.py:30-158` |
| Hit rate empirique (mesuré 2026-04-29) | **7.8 %** au bucket=5, **33.8 %** au bucket=10 (déjà patché à `SCORE_BUCKET_PTS=10` ligne 104) | `reports/eval_06_empirical_findings_2026_04_29.md:9-15` |
| Hit rate cible eval_06 | 30-45 % en stationnaire (estimation théorique non atteinte au bucket=5) | `reports/eval_06_semantic_cache.md:202-207` |
| Compteurs `_hits/_misses` | RAM par instance → **multi-worker non-safe**, stats divergentes | `src/intelligence/semantic_cache.py:47-48`, `eval_06_semantic_cache.md §2 Bug #3` |
| `cleanup_expired` | Fonction existe mais **jamais appelée** → DB grossit sans bornes | `src/intelligence/semantic_cache.py:216-231` |
| Connection SQLite | `sqlite3.connect()` **par appel** (2-4 ms overhead) | `src/intelligence/semantic_cache.py:58-65` |
| TTL | 24 h hardcodé (lazy expiry au `get()`), non-configurable via env | `src/intelligence/semantic_cache.py:42,182` |
| Compression `data_json` | Aucune ; 7 200 entrées stationnaires ≈ 10-22 MB | `eval_06_semantic_cache.md §2 Bug #8` |
| Collisions UX intra-jour | `bar_timestamp` exclu volontairement, mais aussi `session` → 2 signaux 5 h d'intervalle = même narrative | `eval_06_semantic_cache.md §4.2` |

### 1.2 I/O bloquant dans event loop FastAPI

| Endpoint | Path | Problème | Sévérité |
|---|---|---|---|
| `GET /api/v1/signals/history` | `routes/signals.py:50` | `sqlite3.connect()` + 2 SELECT **dans `async def`** | HIGH — bloque event loop 3-10 ms/call |
| `GET /api/v1/narratives/{id}` | `routes/narratives.py:57` | `store.get_by_id()` SQLite sync dans async | HIGH |
| `POST /api/v1/narratives/chat` | `routes/narratives.py:155` | **`llm_engine._call_api()` sync HTTPS dans async** — 800-2000 ms gel total | CRITICAL |
| `GET /api/v1/dashboard/*` | `routes/dashboard.py:18,55` | SQLite agrégations sync dans async | MED |
| `GET /api/v1/health` | `routes/health.py:67-83` | Lecture `cache.get_stats()` + `cache.size()` (SQLite `COUNT(*)`) sync | LOW-MED |

**RPS soutenu réel** (1 vCPU, mono-worker) : ~150-300 RPS, **knee point produit ≈ 1k MAU FREE / 300 MAU si chat actif** (`eval_21_performance.md §5`).

### 1.3 Multi-worker / scalabilité horizontale bloquée

| Composant | Persistance | Multi-process | Score /10 |
|---|---|---|---|
| `SignalStore._current` | mémoire RLock | ❌ process-local, **flicker entre workers** | 4 |
| `SemanticCache._hits/_misses` | RAM | ❌ divergent par worker | 3 |
| `RateLimiter` (`security.py:103-184`) | deque RAM par IP | ❌ **N workers = N×100 req/min réel** | 2 |
| `CircuitBreaker` | RAM | ❌ OPEN/CLOSED divergent | 3 |
| `SignalStateMachine` | RAM + JSON snapshot | ❌ N machines incohérentes | 3 |

**Verdict** : `uvicorn --workers > 1` impossible sans externaliser ces 5 composants.

### 1.4 Infra Redis présente mais inutilisée

* `infrastructure/docker-compose.yml:75-94` — Redis 7-alpine, AOF, 512 MB, LRU eviction, password-protégé : **service défini mais aucun consommateur côté code**.
* `REDIS_URL=redis://redis:6379/0` passé en env (`docker-compose.yml:42`) mais **aucun `import redis` ou `import aioredis` dans `src/`**.

### 1.5 Profiling / observabilité perf

* Histogram Prometheus `http_request_duration_seconds` partiellement câblé dans `src/api/app.py:144` mais **pas exposé par endpoint** (`eval_21_performance.md §9 QW3`).
* Aucun py-spy, cProfile, scalene, ou flame-graph commité.
* `tests/` ne contient pas de `test_perf_*` ni de bench k6/locust.

### 1.6 Coût $/MAU actuel (`eval_21_performance.md §6.2`)

| MAU | $/MAU actuel | $/MAU cible | Bloqueur |
|---|---|---|---|
| 100 | $1.84 | $1.45 | LLM dominant |
| 1 k | $0.87 | $0.48 | LLM + workers single |
| 10 k | $0.76 | $0.37 | LLM + Postgres mandatory |

LLM = **80-90 %** de la facture à 10k MAU → priorité ABSOLUE au cache hit rate.

---

## 2. Vision cible (cache hit >40 %, async end-to-end, multi-worker safe, 10k MAU)

### 2.1 Cibles dures SLO

| KPI | Baseline | Cible 90j | Cible 180j |
|---|---|---|---|
| Hit rate cache cumulé (hash + sémantique) | 7.8 % (bucket=5) / 33.8 % (bucket=10) | **45 %** | **65 %** |
| P50 `/signals/current` | n/a | **< 20 ms** | < 10 ms |
| P95 `/signals/history` | n/a (estimé 100-150 ms) | **< 100 ms** | < 50 ms |
| P95 `/narratives/chat` | n/a (2-5 s) | **< 2 s** | < 1.5 s |
| P99 scanner tick latency template | non-mesuré | < 250 ms | < 150 ms |
| RPS soutenu (mix) sur 1 vCPU | ~150-300 | **> 800** | > 1 500 |
| Event-loop lag p99 | non-mesuré | < 50 ms | < 25 ms |
| Workers multi-process | 1 (mono) | **4** | 8 |
| Coût $/MAU @ 1k MAU | $0.87 | < $0.50 | < $0.40 |
| Coût LLM @ 1k MAU/mo | $690 | **$350** | $250 |
| Memory RSS sous load | non-mesuré (~250-400 MB) | < 600 MB | < 500 MB |

### 2.2 Architecture cible

```
Client
  │
  ▼
Cloudflare CDN / edge cache (static + /metrics + /signals/current 1-2s TTL)
  │
  ▼
Uvicorn × 4 workers (gunicorn ‑w 4 ‑k uvicorn.workers.UvicornWorker)
  │
  ├─► Redis 7 ──┬─► RateLimiter (sliding window, Lua atomic)
  │             ├─► SignalStore._current (cache 200ms)
  │             ├─► SemanticCache.stats (HINCRBY hits/misses)
  │             ├─► CircuitBreaker state (HSET worker‑shared)
  │             └─► Anthropic prompt-cache observability
  │
  ├─► Postgres (signals, narratives, auth, narrative_cache hot)
  │       └─► asyncpg pool 10-20 conns/worker
  │
  ├─► HybridCache
  │      ├─► Tier 1 : Redis hash exact-key (sub-ms hit)
  │      ├─► Tier 2 : Postgres pg_trgm + pgvector cosine (semantic)
  │      └─► Tier 3 : Anthropic prompt cache (cache_control)
  │
  └─► httpx.AsyncClient (Anthropic, Telegram, Discord, FF JSON)
```

### 2.3 Latency budget par étape (scan template mode, P95)

| Étape | Budget | Source |
|---|---|---|
| DataProvider.get_ohlcv (CSV cached) | 15 ms | `data_providers.py:99` |
| SmartMoneyEngine.analyze (200 bars) | 80 ms | `strategy_features.py` |
| VolForecaster (HAR mode défaut) | 25 ms | `eval_04_volatility_findings.md` |
| NewsAgent.evaluate_news_impact | 5 ms | `news_analysis_agent.py:332` |
| ConfluenceDetector.analyze | 2 ms | `confluence_detector.py` |
| StateMachine.on_bar | 2 ms | — |
| Cache.get (Redis hash) | 3 ms | nouveau |
| Template narrative | 5 ms | `template_narrative_engine.py` |
| SignalStore.publish (asyncpg) | 8 ms | nouveau |
| Telegram.send (async httpx) | 80 ms | — |
| **Total budget P95** | **~225 ms** | |

LLM mode (Sonnet single-call) : +1.2-1.8 s, P95 < 2 s acceptable car non-bloquant scanner.

---

## 3. Gap analysis

| Domaine | Actuel | Cible | Gap |
|---|---|---|---|
| Cache hit rate | 33.8 % (bucket=10 déjà patché) | 65 % cumulé | +30 pts → tier sémantique pgvector |
| Cache multi-worker | RAM par worker | Redis shared HINCRBY | refacto stats + clean cleanup hook |
| I/O async | 5 routes SQLite sync, 1 LLM sync dans async | 100 % async (asyncpg + httpx) | aiosqlite ou asyncpg + to_thread wrappers |
| Workers | 1 | 4-8 | externaliser 5 composants RAM → Redis |
| Connection pooling DB | `connect()` par appel | pool partagé asyncpg 10-20 | swap SQLite→Postgres + pool |
| Profiling baseline | aucun | py-spy + locust CI | nouveau projet `tests/perf/` |
| CDN | aucun | Cloudflare + 1-2 s TTL signals/current | DNS + cache-control headers |
| LLM batching | aucun | Anthropic Batch API si latence tolérée | endpoint async-friendly seulement |
| Redis prod | défini mais non utilisé | client async (redis-py 5.x + asyncio) | install + 5 modules |
| GC tuning | défaut CPython | `PYTHONMALLOC=malloc` + gen2 threshold | déploiement uvicorn flag |
| Memory profiling | aucun | tracemalloc + memray dans CI | nouveau script |

---

## 4. Plan d'exécution

### P0 — Bump SCORE_BUCKET_PTS 5→10 (quick-win cache ×4.3)

**STATUT** : ✅ **DÉJÀ FAIT** au commit en cours (`src/intelligence/semantic_cache.py:104` montre `SCORE_BUCKET_PTS = 10` avec commentaire empirique).

| Item | Détail |
|---|---|
| Fichiers | `src/intelligence/semantic_cache.py:104` |
| Heures | 0.25 h |
| Acceptance | Hit rate ≥ 30 % sur replay 6 ans XAU ; `tests/test_semantic_cache.py::test_bucket_collision` vert |
| Dépendances | — |

**Action restante** : ajouter test de non-régression `tests/test_semantic_cache.py::test_hit_rate_baseline` qui replay les 1 200 signaux du sim et assert hit_rate > 0.30 (régression-guard).

---

### P0 — Migration cache local → Redis (multi-worker safe + persistence)

**Tâche 1 : Wrapper `RedisStatsBackend` pour `SemanticCache`**

| Item | Détail |
|---|---|
| Fichiers | `src/intelligence/semantic_cache.py:30-251` (refacto stats), nouveau `src/intelligence/redis_stats.py` |
| Heures | 4 h |
| Acceptance | `cache.get_stats()` retourne stats consolidées multi-worker (HINCRBY redis ; fallback RAM si Redis down) ; `pytest tests/test_semantic_cache_redis.py` vert |
| Dépendances | `redis>=5.0` async, `REDIS_URL` env var |

**Tâche 2 : Migration cache narrative → Redis (hot tier) + SQLite/PG (cold)**

| Item | Détail |
|---|---|
| Fichiers | nouveau `src/intelligence/cache_backends/redis_cache.py`, nouveau `src/intelligence/hybrid_cache.py`, refacto `src/intelligence/llm_narrative_engine.py:60-65` (injection) |
| Heures | 8 h |
| Acceptance | P95 lookup < 3 ms (vs 5-10 ms SQLite) ; éviction LRU via Redis `maxmemory-policy allkeys-lru` (déjà config `docker-compose.yml:87`) ; TTL configurable via `CACHE_TTL_S` env |
| Dépendances | Tâche 1, `redis>=5.0` |

**Tâche 3 : Externaliser RateLimiter → Redis Lua atomique**

| Item | Détail |
|---|---|
| Fichiers | `src/intelligence/security.py:103-184` (RateLimiter), nouveau `src/intelligence/redis_rate_limiter.py` |
| Heures | 5 h |
| Acceptance | 4 workers × tests parallèles ne dépassent **jamais** 100 req/min/IP global ; latence overhead < 1.5 ms/call ; script Lua atomique pour ZADD/ZREMRANGEBYSCORE/ZCARD |
| Dépendances | Tâche 1 |

**Tâche 4 : Externaliser `SignalStore._current` → Redis + micro-cache 200 ms**

| Item | Détail |
|---|---|
| Fichiers | `src/api/signal_store.py:71` (suppression `_current`), `routes/signals.py:23` (lecture Redis), nouveau `src/api/current_signal_cache.py` |
| Heures | 4 h |
| Acceptance | 4 workers retournent même signal sur `/signals/current` (no flicker) ; latence P95 < 30 ms ; micro-cache process 200 ms absorbe 99 % des hits |
| Dépendances | Tâche 1 |

**Tâche 5 : CircuitBreaker shared state Redis**

| Item | Détail |
|---|---|
| Fichiers | `src/intelligence/circuit_breaker.py:73` (RLock RAM), nouveau `src/intelligence/redis_circuit_breaker.py` |
| Heures | 4 h |
| Acceptance | OPEN détecté par worker A fait fail-fast worker B en < 100 ms ; HSET key=`cb:llm` fields=`state,opened_at,failures` |
| Dépendances | Tâche 1 |

**Sous-total P0 Redis** : **25 h**

---

### P0 — Async I/O end-to-end (aiosqlite/asyncpg, async LLM calls, async telegram)

**Tâche 6 : Wrapper `asyncio.to_thread` sur lectures SQLite des routes (quick-win 1 jour)**

| Item | Détail |
|---|---|
| Fichiers | `src/api/routes/signals.py:50`, `routes/narratives.py:57`, `routes/dashboard.py:18,55`, `routes/state.py` |
| Heures | 4 h |
| Acceptance | `tests/test_routes_async_safe.py` : aucune route ne bloque l'event loop > 50 ms (mesure `time.perf_counter()` avant/après) ; P95 `/signals/history` < 100 ms à 100 RPS |
| Dépendances | — (peut être livré avant Redis) |

**Tâche 7 : `httpx.AsyncClient` pour Anthropic dans route `/chat`**

| Item | Détail |
|---|---|
| Fichiers | `src/intelligence/llm_narrative_engine.py:489-535` (`_call_api`), `routes/narratives.py:155` |
| Heures | 6 h |
| Acceptance | Endpoint `/narratives/chat` n'asphyxie plus le worker ; 10 chats parallèles sur 1 worker (vs 1 actuellement) ; P95 chat < 2 s ; cache_control Anthropic conservé |
| Dépendances | — |

**Tâche 8 : Telegram async (`python-telegram-bot[asyncio]`) + Discord httpx async**

| Item | Détail |
|---|---|
| Fichiers | `src/delivery/telegram_notifier.py:181`, `src/delivery/discord_notifier.py:232` |
| Heures | 5 h |
| Acceptance | Notif send n'attend plus le RTT 300-1000 ms en série ; batch send sur N abonnés en `asyncio.gather()` avec semaphore=10 ; tests : 100 notifs send < 3 s wall |
| Dépendances | — |

**Tâche 9 : Migration SQLite → asyncpg + Postgres (signals + narrative + auth)**

| Item | Détail |
|---|---|
| Fichiers | `src/api/signal_store.py` (300 l), `src/api/auth.py`, `src/intelligence/semantic_cache.py`, nouveau `src/db/pg_pool.py`, migration Alembic `alembic/versions/` |
| Heures | 24 h (dev + tests + migration scripts) |
| Acceptance | Backfill SQLite→PG sans downtime via dual-write ; tests d'intégration `tests/test_pg_signal_store.py` ; pool asyncpg min=5 max=20 par worker ; p99 query < 10 ms |
| Dépendances | Postgres présent docker-compose ; ALTER SQL équivalent ; `DATABASE_URL` env |

**Sous-total P0 Async I/O** : **39 h**

---

### P0 — Profiling baseline (cProfile, py-spy) + budget latency par étape

**Tâche 10 : Setup py-spy + flame-graphs CI**

| Item | Détail |
|---|---|
| Fichiers | nouveau `scripts/profile_scanner.py`, nouveau `scripts/profile_api.py`, `Dockerfile` (add py-spy) |
| Heures | 3 h |
| Acceptance | `py-spy record -o flame.svg -- python -m src.intelligence.main` sur 60 s génère flame-graph ; CI artifact upload sur PR |
| Dépendances | py-spy ≥ 0.4 |

**Tâche 11 : Histogram Prometheus par endpoint + percentiles**

| Item | Détail |
|---|---|
| Fichiers | `src/api/app.py:144` (compléter middleware) ; nouveau `src/api/middleware/perf_metrics.py` |
| Heures | 3 h |
| Acceptance | `/metrics` expose `http_request_duration_seconds{path,method,status,le=...}` avec buckets 5/10/25/50/100/250/500/1000/2500 ms ; Grafana dashboard `dashboards/perf.json` |
| Dépendances | prometheus_client déjà présent |

**Tâche 12 : Latency budget enforcement via tests**

| Item | Détail |
|---|---|
| Fichiers | nouveau `tests/perf/test_latency_budgets.py` |
| Heures | 4 h |
| Acceptance | Pour chaque étape pipeline (data, smc, vol, news, conf, sm, cache, narrative, store, notify) : p95 mesuré ≤ budget §2.3 ; CI échec si dépassé |
| Dépendances | scénario replay déterministe XAU M15 |

**Tâche 13 : Event-loop lag instrumentation**

| Item | Détail |
|---|---|
| Fichiers | nouveau `src/api/middleware/loop_lag.py`, expose Prometheus gauge `event_loop_lag_seconds` |
| Heures | 2 h |
| Acceptance | Lag p99 mesuré sous load ; alarme Prometheus `event_loop_lag_p99 > 0.1` |
| Dépendances | Tâche 11 |

**Sous-total P0 Profiling** : **12 h**

---

### P1 — Connection pooling DB

**Tâche 14 : asyncpg pool global + per-worker config**

| Item | Détail |
|---|---|
| Fichiers | `src/db/pg_pool.py`, `src/api/app.py` (startup/shutdown event), tous les modules SQLite consommateurs |
| Heures | 6 h |
| Acceptance | 0 `connect()` per request, réutilisation 100 % ; metrics `asyncpg_pool_size`, `asyncpg_pool_acquire_time_p95 < 2 ms` |
| Dépendances | Tâche 9 |

**Sous-total P1 Pool DB** : **6 h**

---

### P1 — LLM batching (si supporté par Anthropic Batch API)

**Tâche 15 : Anthropic Batch API integration (narratives non-critiques)**

| Item | Détail |
|---|---|
| Fichiers | `src/intelligence/llm_narrative_engine.py` (nouvelle méthode `generate_batch`), `src/intelligence/batch_scheduler.py` |
| Heures | 12 h |
| Acceptance | Narratives non-urgentes (replay batch, recap quotidien) groupées par 100, **-50 % coût** vs single-call ; fallback single-call si batch fail ou latence > 5 min |
| Dépendances | Anthropic Batch API ; tests : 100 narratives batch < $0.50 |

**Note** : Batch API a une SLA de 24 h max, donc **inapplicable aux signaux temps réel** (max 60 s). Utile pour : (a) `/dashboard/equity_curve` annotations historiques, (b) backtests narrés post-hoc, (c) digest email quotidien.

**Sous-total P1 LLM Batching** : **12 h**

---

### P1 — Sentence-embedding semantic cache vraiment sémantique (vs hash actuel)

**Tâche 16 : Cache sémantique 2-tier (hash + pgvector)**

| Item | Détail |
|---|---|
| Fichiers | nouveau `src/intelligence/semantic_embedding_cache.py`, schema PG `narrative_embeddings(cache_key, embedding vector(384), data_json, created_at)`, install `pgvector` extension |
| Heures | 16 h |
| Acceptance | Tier 1 = hash hit (3.8 % bucket=10 actuel) ; Tier 2 = pgvector cosine ≥ 0.92 sur all-MiniLM-L6-v2 (384-dim, ~30 MB RAM) ; hit cumulé ≥ 55 % sur replay 1200 signaux ; latence encode + search < 15 ms p95 |
| Dépendances | Postgres + pgvector, sentence-transformers ; Tâche 9 |

**Tâche 17 : Renommer `SemanticCache` → `NarrativeDedupCache` + extraire layer sémantique**

| Item | Détail |
|---|---|
| Fichiers | rename `src/intelligence/semantic_cache.py` → `narrative_dedup_cache.py` ; refacto 10+ imports ; conserver alias `SemanticCache = NarrativeDedupCache` 1 release |
| Heures | 3 h |
| Acceptance | Tests verts ; pas de string `"semantic"` dans le mauvais layer ; CHANGELOG.md release notes |
| Dépendances | Tâche 16 |

**Sous-total P1 Cache sémantique** : **19 h**

---

### P1 — Multi-worker gunicorn + load test validation

**Tâche 18 : Gunicorn config + 4 workers**

| Item | Détail |
|---|---|
| Fichiers | `infrastructure/Dockerfile:89` (CMD `gunicorn -w 4 -k uvicorn.workers.UvicornWorker src.api.app:app`), `infrastructure/gunicorn.conf.py` |
| Heures | 3 h |
| Acceptance | 4 workers démarrent en < 5 s ; `/health` consistent entre workers ; aucune erreur "address in use" |
| Dépendances | P0 Redis (tâches 3, 4, 5) terminé |

**Sous-total P1 Multi-worker** : **3 h**

---

### P2 — CDN/edge caching pour static assets + `/signals/current`

**Tâche 19 : Cache-Control headers + Cloudflare integration**

| Item | Détail |
|---|---|
| Fichiers | `src/api/app.py` (CORS + Cache-Control middleware), `src/api/routes/signals.py` (`Cache-Control: public, max-age=2, s-maxage=2`) |
| Heures | 4 h |
| Acceptance | Cloudflare hit rate `/signals/current` > 80 % en heure de pointe ; bandwidth origin -60 % ; pas de stale > 4 s |
| Dépendances | Domaine + DNS Cloudflare |

**Sous-total P2 CDN** : **4 h**

---

### P2 — Read replicas DB (Postgres)

**Tâche 20 : Read replica + routing read/write**

| Item | Détail |
|---|---|
| Fichiers | `src/db/pg_pool.py` (deux pools : write/primary, read/replica), `docker-compose.yml` (postgres-replica service) |
| Heures | 12 h |
| Acceptance | 80 % des SELECT routées vers replica ; lag < 500 ms ; failover automatique si replica down (fallback primary) |
| Dépendances | Tâche 14 ; à activer **uniquement** > 5k MAU |

**Sous-total P2 Read replicas** : **12 h**

---

### P2 — GC tuning + memory profiling

**Tâche 21 : Tracemalloc + memray + GC tuning**

| Item | Détail |
|---|---|
| Fichiers | nouveau `scripts/memray_profile.py`, `Dockerfile` (`PYTHONMALLOC=malloc`, `MALLOC_TRIM_THRESHOLD_=131072`), `src/api/app.py` (gc.set_threshold(700, 10, 10) en prod) |
| Heures | 6 h |
| Acceptance | RSS stable < 600 MB sous load 1k RPS ; pas de leak sur 24 h replay ; flame-graph mémoire par module |
| Dépendances | py-spy + memray installés |

**Sous-total P2 GC** : **6 h**

---

## 5. Tests & validation (load tests Locust, benchmark suite, regression perf CI)

### 5.1 Locust load test suite

| Scénario | Fichier | Cible | Acceptance |
|---|---|---|---|
| `BasicReadUser` | `tests/perf/locust_basic.py` | 500 utilisateurs `/signals/current` + `/health` mix 80/20 | P95 < 100 ms, 0 erreurs |
| `HistoryUser` | `tests/perf/locust_history.py` | 200 utilisateurs `/signals/history` paginé | P95 < 200 ms |
| `ChatUser` | `tests/perf/locust_chat.py` | 50 utilisateurs `/narratives/chat` concurrents | P95 < 3 s, P99 < 5 s |
| `MixedRealistic` | `tests/perf/locust_mixed.py` | 1k MAU pattern réaliste (90 % read, 5 % narrative, 5 % chat) | P95 global < 500 ms |
| `MultiWorkerSafety` | `tests/perf/test_multi_worker.py` | 4 workers, 1k req/IP, vérifier RateLimit global = 100/min | 0 dépassement |

### 5.2 Benchmark perf CI (sur chaque PR)

* GitHub Action `perf-bench.yml` — run scénario replay 24 h XAU M15, mesure :
  * Scanner tick p50/p95/p99
  * Cache hit rate (assert ≥ 30 %)
  * Memory RSS peak (assert < 800 MB)
  * Event-loop lag p99 (assert < 100 ms)
* Comparaison vs baseline `main` ; PR comment auto-généré si régression > 10 %.

### 5.3 Cache hit rate regression guard

* `tests/test_cache_hit_rate.py` : replay 1 200 signaux sim (script `scripts/eval_06_hit_rate_sim.py`), assert :
  * Hit rate ≥ 30 % au bucket=10 (régression bucket)
  * Hit rate ≥ 50 % avec layer sémantique activé (post-tâche 16)

### 5.4 Stress redis fail-over

* `tests/test_redis_outage.py` : kill Redis container pendant 30 s, vérifier que :
  * App reste up (degraded mode)
  * RateLimiter fallback RAM (per-worker, accept overcount)
  * Cache fallback skip (cache miss = LLM call OK)
  * Aucun 500 user-facing

---

## 6. Sécurité (cache poisoning, key collision, Redis auth)

| Risque | Mitigation |
|---|---|
| **Cache poisoning narrative** | Cache key dérivée **uniquement** de payload signal interne ; jamais user-input ; validation pydantic stricte avant `put()` |
| **Collision SHA-256[:16]** | 64 bits → 2³² entrées avant 50 % anniversaire ; négligeable < 10⁴ entrées (cf. `eval_06_semantic_cache.md §4.1`) ; **upgrade [:32] (128 bits) en P2** si scale 100k+ entrées |
| **Redis non-auth** | `requirepass ${REDIS_PASSWORD}` déjà config `docker-compose.yml:87` ; **fail-closed boot** si var absente |
| **Redis ACL** | Créer user dédié `acl setuser sentinel +@read +@write +@hash +@sortedset -@dangerous ~cache:* ~rl:* ~cb:*` ; bannir FLUSHDB/CONFIG/SCRIPT |
| **Redis exfiltration** | Pas d'exposition host (déjà `# ports: # 6379` commenté `docker-compose.yml:82-84`) ; TLS Redis 7.2+ si exposé via VPN |
| **DoS via cache flooding** | RateLimit pré-cache ; LRU `allkeys-lru` (déjà config `docker-compose.yml:87`) + `maxmemory 512mb` |
| **Cache stale data UX** | TTL 24 h ; cleanup cron ; versionning `data_json` (`{"v":1,"payload":...}`) pour invalider à schema bump (`eval_06_semantic_cache.md §2 Bug #5`) |
| **pgvector injection** | Embeddings calculés server-side, jamais user-input direct ; query parameterized |
| **Sentence-transformers model trust** | Pin hash `all-MiniLM-L6-v2` dans `requirements.txt` ; cache HF local container ; pas de download runtime |
| **Connection pool exhaustion** | asyncpg `max_size=20` + `command_timeout=30` ; alarme Prometheus `asyncpg_pool_saturated` |

---

## 7. Métriques (cache hit rate, p50/p99 latency par endpoint, RPS, memory, CPU)

### 7.1 Métriques business critiques

| Métrique | Source | Cible | Alarme |
|---|---|---|---|
| `cache_hit_rate{type="exact"}` | `cache.get_stats().hit_rate` Redis HINCRBY | > 30 % | < 25 % 1h |
| `cache_hit_rate{type="semantic"}` | pgvector tier | > 20 % | < 15 % 1h |
| `cache_hit_rate_cumulative` | sum exact+semantic | > 45 % | < 40 % 1h |
| `llm_calls_per_signal` | counter | < 0.6 (avec cache) | > 0.8 |
| `llm_cost_usd_per_signal` | computed | < $0.015 | > $0.020 |
| `anthropic_cache_read_tokens_ratio` | `usage.cache_read_input_tokens / input_tokens` | > 60 % | < 50 % |

### 7.2 Métriques latence (Prometheus histogram)

| Métrique | Buckets (s) | P95 cible | P99 cible |
|---|---|---|---|
| `http_request_duration_seconds{path="/api/v1/signals/current"}` | [.005, .01, .025, .05, .1, .25] | 0.05 | 0.1 |
| `http_request_duration_seconds{path="/api/v1/signals/history"}` | [.01, .025, .05, .1, .25, .5] | 0.1 | 0.25 |
| `http_request_duration_seconds{path="/api/v1/narratives/chat"}` | [.5, 1, 2, 5, 10] | 2.0 | 5.0 |
| `scanner_tick_duration_seconds` | [.05, .1, .25, .5, 1, 2.5] | 0.25 | 0.5 |
| `cache_get_duration_seconds` | [.001, .003, .005, .01, .025] | 0.005 | 0.01 |
| `event_loop_lag_seconds` | [.01, .025, .05, .1, .25] | 0.05 | 0.1 |
| `db_query_duration_seconds{op}` | [.001, .005, .01, .025, .05] | 0.01 | 0.025 |

### 7.3 Métriques système

| Métrique | Source | Cible | Alarme |
|---|---|---|---|
| `process_resident_memory_bytes` | prometheus_client | < 600 MB | > 800 MB 10min |
| `process_cpu_seconds_total` (rate 5min) | — | < 50 % par worker | > 80 % 10min |
| `python_gc_collections_total{generation="2"}` | — | < 10/min | > 50/min |
| `asyncpg_pool_acquire_seconds` | custom | P95 < 0.002 | P95 > 0.01 |
| `redis_command_duration_seconds` | — | P95 < 0.001 | P95 > 0.005 |
| `up{job="trading-bot"}` | scrape | 1 | 0 |

### 7.4 RPS / throughput

| Métrique | Cible | Méthode |
|---|---|---|
| `http_requests_per_second_sustained` | > 800 RPS mix réaliste (1 vCPU) | locust 30 min ramp |
| `scanner_signals_per_hour` | 4-6 (XAU M15) | counter |
| `notifier_sends_per_minute` | < 100 (rate-limited) | counter |

---

## 8. Risques & mitigations (cache invalidation, stale data, Redis outage)

| Risque | Probabilité | Impact | Mitigation |
|---|---|---|---|
| **Redis outage** (single point) | M | High | Sentinel mode Redis (3 nœuds) en prod ; fallback graceful : RateLimit per-worker, Cache miss = LLM call ; alarme P1 < 1 min |
| **Cache invalidation oubliée** au schema change | H | Med | Versionning `data_json={"v":2,"payload":...}` ; bump auto `SCHEMA_VERSION` rejette les v<current |
| **Stale narrative servie** | M | Med | TTL 24 h ; cleanup cron toutes 1 h dans scanner ; `bar_timestamp` exclus mais `session` ajouté (P1) pour éviter collision inter-session |
| **pgvector index miss** | L | Med | IVFFlat index `(embedding vector_cosine_ops) WITH (lists=100)` ; VACUUM ANALYZE hebdo |
| **Postgres lock contention sous load** | M | High | asyncpg pool `max_size=20` + `command_timeout=30s` ; transactions courtes ; idx sur (created_at, symbol) |
| **Memory leak sentence-transformers** | L | Med | Modèle chargé 1× au startup ; pas de reload ; memray hebdo dans CI |
| **Anthropic API rate limit (5k RPM tier 1)** | M | High | CircuitBreaker (déjà présent `circuit_breaker.py:73`) + Redis state shared ; backoff exponentiel ; fallback Template |
| **Multi-worker race condition state machine** | H | High | Tâche 4-5 : externaliser StateMachine → Redis HSET + Lua atomic ; tests `tests/test_state_machine_redis.py` |
| **Migration SQLite → Postgres downtime** | M | Med | Dual-write 7 jours ; backfill batch ; switch read traffic via flag ; rollback procedure documentée |
| **Locust test instable** | M | Low | Run nightly ; baseline vs main ; tolérance ±10 % |
| **Cache key collision sémantique inter-session** | M | Low UX | Inclure `session={asian,london,ny,after}` dans clé (eval_06 reco #2) |

---

## 9. Dépendances (API, deployment, LLM, observability)

### 9.1 Dépendances inter-catégories

| Catégorie | Dépendance |
|---|---|
| **API (cat 10)** | Tâche 6-9 (async routes) doivent être livrées avant load test 1k MAU |
| **Auth (cat 11)** | Tâche 9 migration PG inclut `auth.py` keystore |
| **Signal Store (cat 12)** | Tâche 4 externalise `_current` ; Tâche 9 migre vers PG |
| **Telegram/Discord (cat 13)** | Tâche 8 async notifiers |
| **Circuit Breaker (cat 14)** | Tâche 5 externalise state Redis |
| **Security (cat 15)** | Tâche 3 RateLimiter Redis |
| **Observability (cat 16)** | Tâches 11-13 dépendent Prometheus déjà déployé |
| **LLM (cat 5)** | Tâche 7 async + Tâche 15 batch + cache_control préservé |
| **Deployment (cat 22)** | Tâche 18 gunicorn + Tâche 19 CDN |
| **Testing (cat 17)** | Section 5 load tests |
| **MLOps (cat 23)** | Tâche 16 sentence-transformers model lifecycle |

### 9.2 Dépendances tech / packages

```
redis>=5.0.0           # async client
asyncpg>=0.29.0        # PG async pool
pgvector>=0.2.5        # PG semantic
sentence-transformers>=2.7.0  # all-MiniLM-L6-v2 ~30 MB
httpx>=0.27.0          # async LLM, telegram, discord
aiosqlite>=0.20.0      # transition layer
gunicorn>=21.2.0       # workers
prometheus-client>=0.20.0  # déjà présent, version check
py-spy>=0.4.0          # profiling
memray>=1.13.0         # memory profiling
locust>=2.27.0         # load tests
```

### 9.3 Dépendances infra

* Redis 7.2+ avec AOF + LRU eviction + password (déjà `docker-compose.yml:77-94`)
* Postgres 15 + pgvector ext (à installer, `init-db.sql:CREATE EXTENSION vector;`)
* Prometheus scrape `/metrics` toutes 15 s (déjà `docker-compose.yml:126-143`)
* Grafana dashboards perf (à créer `infrastructure/grafana/dashboards/perf.json`)
* Cloudflare DNS + page rules (P2, post-traction)

---

## 10. Estimation totale & timeline

### 10.1 Tableau récapitulatif

| Priorité | Tâche | Heures |
|---|---|---|
| **P0** | T1 RedisStatsBackend | 4 |
| **P0** | T2 Cache narrative Redis hot tier | 8 |
| **P0** | T3 RateLimiter Redis Lua | 5 |
| **P0** | T4 SignalStore._current Redis | 4 |
| **P0** | T5 CircuitBreaker Redis | 4 |
| **P0** | T6 `to_thread` wrapper routes SQLite | 4 |
| **P0** | T7 httpx.AsyncClient Anthropic | 6 |
| **P0** | T8 Telegram + Discord async | 5 |
| **P0** | T9 Migration SQLite → asyncpg + PG | 24 |
| **P0** | T10 py-spy profiling setup | 3 |
| **P0** | T11 Prometheus histogram per-endpoint | 3 |
| **P0** | T12 Latency budget tests | 4 |
| **P0** | T13 Event-loop lag instrumentation | 2 |
| **P0** | Bucket bump (déjà fait) | 0.25 |
| **P0** | Test régression hit rate | 1 |
| **Sous-total P0** | | **77.25** |
| **P1** | T14 asyncpg pool global | 6 |
| **P1** | T15 LLM Batch API | 12 |
| **P1** | T16 pgvector semantic cache | 16 |
| **P1** | T17 Rename `SemanticCache` → `NarrativeDedupCache` | 3 |
| **P1** | T18 Gunicorn 4 workers config | 3 |
| **Sous-total P1** | | **40** |
| **P2** | T19 Cloudflare CDN | 4 |
| **P2** | T20 PG read replica | 12 |
| **P2** | T21 GC tuning + memray | 6 |
| **Sous-total P2** | | **22** |
| **Tests perf & load** | Locust suite + CI bench | 12 |
| **Doc & runbooks** | Migration playbook + alarmes | 6 |
| **TOTAL** | | **~157 h** |

### 10.2 Timeline

| Semaine | Sprint | Livrables | Heures |
|---|---|---|---|
| **S1** | Async I/O quick wins | T6, T7, T8 + bench baseline (T10, T11) | 21 |
| **S2** | Redis foundation | T1, T2, T3, T4, T5 | 25 |
| **S3** | Postgres migration kickoff | T9 (dual-write phase) + T13 | 14 |
| **S4** | Postgres go-live + pooling | T9 finalisation + T14 + T18 | 15 |
| **S5** | Profiling & budgets | T12 + Locust suite + Grafana dashboards | 16 |
| **S6** | Cache sémantique | T16, T17 | 19 |
| **S7** | LLM batch + tests | T15 + tests régression + CI perf | 18 |
| **S8** | CDN + GC | T19, T21 + doc | 16 |
| **S9-S10** | Read replicas (différé post-5k MAU) | T20 | 12 |

**Critical path** : S1→S2→S4 (35 h) débloque multi-worker à scale + cache hit rate ≥ 30 %, soit déjà 50 % du ROI.

### 10.3 Coût mensuel infra additionnel

| Composant | Mensuel | Note |
|---|---|---|
| Redis Cloud Essential 1 GB | **$5-15** | self-host docker = $0 mais ops overhead |
| Postgres managed (Railway/Supabase) 1 GB | $10-20 | déjà budgeté `eval_21_performance.md §6` |
| Cloudflare Free tier | $0 | suffisant < 100k req/jour |
| sentence-transformers (self-hosted) | $0 | RAM +30 MB par worker |
| **Total infra additionnel** | **~$15-35/mo** | absorbé dès 50 MAU $19/mo |

### 10.4 ROI économique

| MAU | Économie LLM mensuelle (hit 30 % → 65 %) | Économie annuelle |
|---|---|---|
| 100 | $0.69 → $0.36 = -$0.33/MAU = **$33/mo** | $396 |
| 1 000 | **$690 → $241 = $449/mo** | **$5 388** |
| 10 000 | $6 900 → $2 415 = **$4 485/mo** | **$53 820** |

ROI break-even @ 1k MAU : **~70 jours dev** (157 h × $50/h = $7 850 dev vs $5 388/an LLM économie + scale unlock).

---

## Synthèse

**Chemin** : `C:\MyPythonProjects\TradingBOT_Agentic\reports\commercialization_sprint\17_caching_performance.md`

**Top 3 P0** :
1. **Async I/O end-to-end** (T6-T9, 39 h) : `asyncio.to_thread` routes SQLite + httpx Anthropic + asyncpg migration → débloque event loop, RPS ×3-5
2. **Redis foundation** (T1-T5, 25 h) : externaliser stats cache + RateLimiter + `_current` SignalStore + CircuitBreaker → unlock multi-worker (gunicorn -w 4)
3. **Profiling baseline + budgets** (T10-T13, 12 h) : py-spy + Prometheus histogram + Locust suite → boucle de mesure pour ne pas régresser

**Heures totales** : ~157 h (P0 77 h, P1 40 h, P2 22 h, tests+doc 18 h)

**Coût Redis/mo estimé** : **$5-15** (Redis Cloud Essential 1 GB) ou **$0** self-host docker (déjà dans `docker-compose.yml:77-94`, juste à brancher).
