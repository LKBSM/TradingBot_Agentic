# Plan de Commercialisation — Catégorie 8 : API & Backend Architecture

**Auteur** : Audit backend / API
**Date** : 2026-05-21
**Branche** : `institutional-overhaul`
**Périmètre** : `src/api/*` (app, auth, dependencies, signal_store, tier_manager, models, idempotency_store, latency_tracker, shutdown, openapi_enrichment, 20 routes, 3 middlewares), entrée Uvicorn `src/intelligence/main.py:660`, conteneur `infrastructure/Dockerfile`, contrats Pydantic v2 (`SignalResponse`, `NarrativeResponse`, `InsightSignalV2`, `EnrichRequest`).
**Lecture conjointe** : `reports/eval_21_performance.md` (5.5/10), `reports/eval_10_15_team_audit.md` (57 deltas perf/efficacité), `reports/eval_05_09_refresh_2026_04_29.md`, `reports/eval_16_observability.md` (3.2/10), `reports/eval_29_compliance_findings.md`.

---

## 1. État actuel (Audit)

### 1.1 Topologie process

- **Mono-process Uvicorn**, sans `--workers`, lancé depuis `src/intelligence/main.py:660` (`uvicorn.run(api_app, host="0.0.0.0", port=api_port)`). Aucun gestionnaire de pool (Gunicorn / Hypercorn).
- **3 threads dans le même process** :
  1. Event-loop FastAPI (thread principal)
  2. Scanner daemon (`main.py:646`, `sentinel_scanner.py:170`), poll fixe 60 s `time.sleep(self._poll_interval)` (`sentinel_scanner.py:220`)
  3. Watchdog daemon (`main.py:649`), réveil toutes les 30 s
- **Locks process-local partout** : `SignalStore._lock` (`signal_store.py:70`), `KeyStore._lock` (`auth.py:55`), `IdempotencyStore._lock` (`idempotency_store.py:88`), `CircuitBreaker._lock`, `LatencyTracker._lock`. Toute mise à l'échelle horizontale casse l'invariant single-instance.

### 1.2 Stack FastAPI

- `app.py:182` construit l'app avec lifespan async `_auto_register_default_handlers` (workers stop → stores close).
- **9 middlewares stackés** : CORS, GZip 1 KB (`app.py:207`), `StructuredAccessLogMiddleware` (request_id + tier + latency tracker), `RateLimitHeadersMiddleware`, `GeoBlockMiddleware` (US/QC/UK/OFAC), `request_size_limit` (1 MB header check), `rate_limit_middleware` (in-RAM deque par IP), `security_headers` (HSTS/CSP/X-Frame/Referrer), `request_logging`.
- **20 routers** montés (`app.py:325-343`) — `signals`, `state`, `health`, `health_deep`, `operator`, `prometheus`, `admin`, `admin_audit`, `dashboard`, `narratives`, `legal`, `qa`, `enrich`, `audit`, `insight_history`, `metrics_latency`, `webhook_ack`, `billing`, `webapp`.
- **OpenAPI enrichi** via `install_openapi_enrichment(app)` (`openapi_enrichment.py:80`) — `operationId` snake_case, servers list, descriptions tag. `/api/docs` Swagger UI exposé.
- **Versioning** : tous les routes du domaine sont préfixées `/api/v1/...` (statique en dur dans chaque router, pas de stratégie de bascule v2). `/health` racine existe en doublon pour Docker (`health.py:122`).

### 1.3 Stockages

| Composant | Fichier | Persistance | Thread-safe | Multi-process |
|-----------|---------|-------------|-------------|---------------|
| `SignalStore` | `signal_store.py` | SQLite WAL `signals.db` + `_current` RAM | RLock | ❌ `_current` divergent |
| `KeyStore` | `auth.py` | SQLite WAL `api_keys.db` + cache RAM verify | RLock + LRU 60 s | partiellement (cache divergent par worker) |
| `UserTierManager` | `tier_manager.py` | SQLite WAL `users.db` + `usage_log` | RLock | ⚠️ compteur quotidien divergent |
| `RateLimiter` IP | `intelligence/security.py:100` | deque RAM | Lock | ❌ 4 workers = 4×limite |
| `IdempotencyStore` | `idempotency_store.py:87` | dict + RLock + TTL 24 h | RLock | ❌ |
| `LatencyTracker` | `latency_tracker.py` | rolling-window RAM | Lock | ❌ p95 par worker |
| `AuditLedger` (hash-chain) | `src/audit/*` | SQLite | RLock | ⚠️ single writer requis |
| `WebhookQueue` | `delivery/webhook_queue.py` | RAM + dead-letter | Lock | ❌ |
| `IdempotencyStore` | id | dict 100k entries max | RLock | ❌ |

### 1.4 Bouchons I/O synchrones dans `async def`

Repris d'eval 21 §3.2 + audit code actuel.

| Fichier:ligne | Appel | Sévérité |
|---------------|-------|----------|
| `routes/signals.py:50` | `store.get_history()` → `sqlite3.connect()` + 2 SELECT dans `async def` | **HIGH** — bloque event loop ~3-10 ms par appel |
| `routes/signals.py:26` | `store.get_current()` (RAM) | OK |
| `routes/narratives.py:58` | `store.get_by_id(signal_id)` SQLite sync | **HIGH** |
| `routes/narratives.py:160` | `llm_engine._call_api(model, prompt)` — **HTTPS sync Anthropic** dans `async def` chat | **CRITICAL** — gèle l'event loop 800-3000 ms par appel |
| `routes/dashboard.py:39,65` | `tracker.get_performance_summary()` / `get_equity_curve()` SQLite sync | **HIGH** |
| `routes/insight_history.py:106,296` | `ledger.paginate()` / `find_by_insight_id()` SQLite sync | **MED** |
| `routes/enrich.py:258` | `pipeline.query()` — RAG sync (retrieval + LLM facultatif) | **HIGH** |
| `routes/admin.py:42,54,78` | `key_store.*` SQLite sync | **MED** |
| `auth.require_api_key` | `key_store.verify_key` + `check_rate_limit` + `record_usage` + `tier_manager.*` — 4 SQL calls sync | **HIGH** — exécuté à chaque requête authentifiée |

### 1.5 Sécurité requête (déjà en place vs gaps)

**En place** :
- HSTS / X-Frame / CSP / Referrer-Policy / Permissions-Policy (`app.py:262`).
- Body cap 1 MB via `Content-Length` (`app.py:232`).
- Rate-limit per-IP 100/min (`app.py:243`).
- GeoBlock US/QC/UK/OFAC.
- Idempotency-Key style Stripe sur `/enrich` (`enrich.py:212`).
- Request-id propagé en réponse (`access_log.py:191`).
- Signature HMAC admin (`auth.py:493`, replay window 5 min).
- Sanitization input (`sanitize_string`, `_CONTROL_CHAR_PATTERN`).

**Gaps** :
- `request.client.host` derrière reverse proxy = IP du proxy, pas du client réel → bypass rate-limit (eval 15 finding 15.3).
- Cap 1 MB par header `Content-Length` seulement — chunked transfer encoding contourne (eval 15 finding 15.2).
- CSP autorise `unsafe-inline` pour `script-src`/`style-src` (nécessaire Swagger CDN mais doit être loosé en dev seulement).
- Aucune validation `Origin`/`Referer` sur POST/DELETE (pas de CSRF token, mais cookies non utilisés donc OK pour un API key-based).

### 1.6 Versioning & OpenAPI

- **Pas de routing par version** : `/api/v1/*` est hardcodé dans chaque router. Pas de mécanisme `/api/v2` côte-à-côte. Pas de header `Accept-Version`.
- **OpenAPI** : `install_openapi_enrichment` injecte `servers`, tag descriptions, `operationId` propres. Manque : examples par response (FastAPI permet `responses={200: {"content": {...}}}` ou `examples=`), schémas d'erreur explicites par route, paramètres `description=` partiels.
- **Pas de SDK généré** côté broker (TS / Python). Aucun script CI ne vérifie la non-régression du contrat OpenAPI (snapshot test).

### 1.7 Idempotency, request-id, tracing

- ✅ **Idempotency-Key** : `/enrich` seul (`enrich.py:200`). Aucun autre endpoint mutant.
- ✅ **X-Request-Id** : généré + propagé par `StructuredAccessLogMiddleware`.
- ❌ **Pas de tracing distribué** (W3C `traceparent`, OpenTelemetry SDK absent).
- ❌ **Pas de propagation cross-service** vers le scanner thread (request_id meurt au sortir du middleware).

### 1.8 Verdict

Note actuelle **5.5 / 10** (eval 21). Le code applique correctement les patterns FastAPI modernes (async lifespan, dependencies, Pydantic v2, OpenAPI enrichi, middleware composables). **Trois dettes structurelles** bloquent la commercialisation à >1k MAU :

1. **Sync I/O dans async def** sur 8+ routes (event loop CPU-bound dès 100 RPS).
2. **State process-local** (`_current`, RateLimiter, IdempotencyStore, LatencyTracker) → mono-worker imposé.
3. **Polling 60 s** vs SLA 30 s annoncé (`sentinel_scanner.py:220`) — un retard moyen de 30 s entre la clôture de bougie et la publication signal.

---

## 2. Vision cible

| Dimension | Cible 6 mois | Méthode de mesure |
|-----------|--------------|-------------------|
| **Workers** | 4 workers Uvicorn par instance, état partagé via Redis + Postgres | `ps aux \| grep uvicorn` |
| **Async I/O** | 0 appel `sqlite3.connect` / `requests.post` / `client.messages.create` dans un `async def` | grep CI `git grep -nE '(sqlite3.connect\|requests\.(get\|post)\|client\.messages\.create)' src/api/routes` |
| **Latence p99** | `/api/v1/signals/current` < 50 ms ; `/api/v1/insights/history` < 100 ms ; `/api/v1/enrich` < 1.5 s (cache miss), < 200 ms (replay) ; `/api/v1/narratives/chat` < 2 s | latency_tracker p99 par route |
| **Débit** | **1 000 req/s par instance** sur mix réaliste (80% lectures, 20% writes) | k6 ramp 0→2000 |
| **Concurrent users** | 800+ sans P95 > 500 ms, 5 000+ avec scale-out | k6 |
| **OpenAPI** | Spec complète + 1 example par 2xx response + 1 example par 4xx error + SDK Python + TS générés en CI | `openapi-generator-cli validate`, snapshot test |
| **Versioning** | `/api/v1/*` stable, `/api/v2/*` côte-à-côte pour évolutions breaking, deprecation header `Sunset:` sur v1 le jour où v2 démontre 30 jours de stabilité | header + changelog |
| **Idempotency** | Header `Idempotency-Key` accepté sur **tous les POST/PUT mutants** (`/enrich`, `/admin/keys`, `/webhook/ack`, futurs `/orders/*`) | grep `idempotency_key` dans route handlers |
| **Tracing** | `traceparent` W3C propagé, exporté OTLP vers le collector Observability (cat. 7) | tempo / jaeger UI |
| **Saturation** | event_loop_lag p99 < 50 ms, RSS < 1 GB par worker, FDs < 500 par worker | `/health/deep` |

---

## 3. Gap analysis

| Axe | État | Cible | Effort jours | Priorité |
|-----|------|-------|--------------|----------|
| Async SQLite reads dans routes | sync `sqlite3.connect` | `aiosqlite` pool + `await` | 2 | **P0** |
| Async LLM chat | `_call_api` sync dans `async` | `httpx.AsyncClient` + cache prompt | 1 | **P0** |
| Multi-worker | mono-worker, `_current` mémoire, RateLimiter RAM | 4 workers + Redis pour rate/idemp/cache + DB pour `_current` | 4 | **P0** |
| Polling scanner 60 s | `time.sleep(60)` | event-driven (push MT5 close ou WebSocket) | 4 | **P0** |
| Queue notifications | dict + thread + dead-letter in-process (`webhook_queue.py`) | Redis Streams ou SQLite WAL + worker pool | 3 | **P0** |
| OpenAPI examples + SDK | enrichi mais sans examples | examples sur 100% routes + `openapi-generator-cli` Python + TS en CI | 2 | **P0** |
| Versioning `/v2` | hard-codé `/v1` | sub-app pattern `app.mount('/api/v2', v2_app)` ou router prefix | 1.5 | **P1** |
| Idempotency étendue | uniquement `/enrich` | tous mutants + cleanup expire | 1 | **P1** |
| X-Request-Id + traceparent | request-id présent, traceparent absent | OpenTelemetry SDK + auto-instrumentation FastAPI | 1.5 | **P1** |
| Pagination/filtering standards | curseur ad-hoc sur insights, page/page_size sur signals | RFC 5988 `Link:` + `cursor=` standardisé, `limit/offset` rejeté côté handler | 1 | **P1** |
| Postgres + asyncpg | SQLite 3 stores, WAL plafond 50 writes/s | Postgres 15 + asyncpg pool, garder SQLite pour dev | 5 | **P2** |
| Error responses standardisés | `ErrorResponse(error, detail)` mais pas tous les routes l'utilisent | RFC 7807 `application/problem+json` global | 0.5 | **P2** |
| GraphQL B2B advanced queries | absent | Strawberry GraphQL sub-app `/api/v1/graphql` pour broker analytics | 5 | **P2** |
| Load tests Locust/k6 | aucun | scénarios `signals_read`, `enrich_write`, `chat`, sustained 1h, ramp 0→1k | 2 | **P0** |
| Contract / snapshot tests | aucun | `openapi.json` versionné + diff CI | 0.5 | **P1** |

---

## 4. Plan d'exécution

Notation tâche : **<id>** — titre. **Fichiers** | **Heures** | **Acceptance** | **Dépendances**.

### P0 — Async I/O end-to-end (16 h)

#### P0-A1 — Migrer toutes les lectures SQLite hot-path vers `aiosqlite`
- **Fichiers** : `src/api/signal_store.py:81-300`, `src/api/auth.py:175-225,389-405`, `src/api/tier_manager.py:182-308`, `src/api/routes/signals.py:50`, `src/api/routes/narratives.py:58`, `src/api/routes/dashboard.py:24-76`, `src/api/routes/insight_history.py:104,296`, `src/api/routes/admin.py:*`, `src/audit/hash_chain_ledger.py` (paginate/find_by_insight_id), tests `tests/test_signal_store.py`, `tests/test_routes_async_safe.py` (nouveau).
- **Heures** : 6 h.
- **Stratégie** :
  1. Ajouter `aiosqlite` à `requirements.txt`.
  2. Introduire `SignalStoreAsync` qui expose `async def get_history`, `async def get_by_id` partageant une connexion pool (1 conn par instance, PRAGMA WAL/NORMAL appliqué une fois au boot).
  3. Garder l'API sync legacy pour le scanner thread (qui n'est pas async).
  4. Wrapper de fallback `asyncio.to_thread(...)` pendant la migration progressive.
  5. Tests `pytest-asyncio` qui invoquent 50 lectures concurrentes et asserent event-loop lag p99 < 20 ms via `loop.time()` instrumenté.
- **Acceptance** :
  - `git grep -nE 'sqlite3\.connect' src/api/routes/` retourne **0**.
  - `tests/test_routes_async_safe.py` : 100 reqs `/signals/history` en parallèle, p95 < 80 ms.
  - Bench k6 `signals_read` : 500 RPS soutenu, p99 < 200 ms.
- **Dépendances** : aucune (peut commencer immédiatement).

#### P0-A2 — Détacher l'appel LLM `chat` du event loop
- **Fichiers** : `src/api/routes/narratives.py:99-171`, `src/intelligence/llm_narrative_engine.py:489-535` (nouvelle méthode `async def acall_api`), `src/api/auth.py` (read `request.app.state.app_state.http_client`), `app.py:lifespan` (instancier `httpx.AsyncClient(timeout=30.0)`).
- **Heures** : 4 h.
- **Stratégie** :
  1. Ajouter un `httpx.AsyncClient` global au lifespan (créé au startup, fermé au shutdown via `GracefulShutdownCoordinator`).
  2. Implémenter `LLMNarrativeEngine.acall_api(model, prompt, **kw) -> dict` qui POST `https://api.anthropic.com/v1/messages` avec le même body que `client.messages.create` (Anthropic SDK n'a pas d'async client officiel stable, fallback HTTP brut OK).
  3. Cache prompt Anthropic préservé via `cache_control` (system prompt ≥ 1024 tok, cf eval 05).
  4. Si LLM circuit breaker OPEN, court-circuiter dès l'entrée du handler.
- **Acceptance** :
  - `git grep -n '_call_api' src/api/routes/` retourne 0.
  - 10 chats concurrents : p95 chat < 2 s, p95 autres routes < 100 ms (pas de contamination).
- **Dépendances** : —.

#### P0-A3 — Async notifications Telegram / Discord
- **Fichiers** : `src/delivery/telegram_notifier.py:181`, `src/delivery/discord_notifier.py:232`, `src/delivery/webhook_queue.py`.
- **Heures** : 3 h.
- **Stratégie** : `python-telegram-bot[asyncio]` (déjà dispo), Discord via `httpx.AsyncClient`. La queue webhook (`webhook_queue.py:1`) reste process-local pour l'instant (cf P0-D plus bas).
- **Acceptance** : 100 notifications Telegram en parallèle terminent en < 5 s sans bloquer scanner thread.
- **Dépendances** : —.

#### P0-A4 — Audit final "0 sync I/O dans async def" + CI gate
- **Fichiers** : `.github/workflows/ci.yml` (à créer), `scripts/check_async_safety.py` (nouveau).
- **Heures** : 3 h.
- **Stratégie** : Script qui parse l'AST de `src/api/routes/*.py`, détecte tout `async def` contenant un call à une liste noire (`sqlite3.connect`, `requests.`, `time.sleep`, `client.messages.create`, `pd.read_csv`). Échoue le build si match.
- **Acceptance** : CI fail si un dev ajoute un blocking call.

### P0 — Multi-worker (Gunicorn + Uvicorn workers) (10 h)

#### P0-B1 — Externaliser le state process-local vers Redis
- **Fichiers** : `src/intelligence/security.py:100-180` (RateLimiter Redis), `src/api/idempotency_store.py` (backend Redis pluggable), `src/api/latency_tracker.py` (option Redis pour agrégation cross-worker), `src/api/signal_store.py:201` (retirer `_current` au profit d'un SELECT TOP 1 + cache TTL 200 ms).
- **Heures** : 6 h.
- **Stratégie** :
  1. Wrapper `RedisRateLimiter` (compteurs sliding-window via `INCR + EXPIRE 60` ou `ZADD/ZREMRANGEBYSCORE`).
  2. `IdempotencyStoreRedis` : `SET key value EX 86400 NX` + GETEX. Clash check via comparaison body_hash.
  3. `SignalStore.get_current_async()` lit Postgres/SQLite via curseur LIMIT 1, micro-cache process-local 200 ms TTL (absorbe 99% des hits si `/signals/current` est poll 1/s).
  4. `RedisAdapter` global injecté via `AppState.redis_client` (`dependencies.py:13`).
- **Acceptance** :
  - Lancer 4 workers + 100 IPs concurrentes → rate-limit cohérent à ±5% du seuil global (pas ×4).
  - 1000 lectures `/signals/current` simultanées sur 4 workers → toutes voient le même signal_id.
- **Dépendances** : Redis disponible (déploiement cat. 9).

#### P0-B2 — Gunicorn + UvicornWorker config
- **Fichiers** : `infrastructure/Dockerfile:89`, `infrastructure/gunicorn_conf.py` (nouveau), `infrastructure/docker-compose.yml`.
- **Heures** : 2 h.
- **Stratégie** :
  - `CMD ["gunicorn", "src.intelligence.main:get_api_app", "-k", "uvicorn.workers.UvicornWorker", "-w", "4", "--bind", "0.0.0.0:8000", "--access-logfile", "-", "--log-config", "infrastructure/log_config.json", "--graceful-timeout", "30", "--worker-tmp-dir", "/dev/shm"]`.
  - Refacto `main.py:521-697` pour exposer `get_api_app()` callable + lancer scanner/watchdog **dans 1 SEUL worker** (worker-0 via `--preload` + check `os.getpid() == prefork_master_pid`).
  - Variante : externaliser le scanner dans un container séparé (`smart-sentinel-scanner`) qui ne sert pas l'API → architecture cible cat. 9.
- **Acceptance** :
  - 4 workers actifs, scanner unique, `/api/v1/scanner/status` retourne le même état depuis n'importe quel worker.
  - `wrk -t8 -c200 -d30s` sur `/signals/current` : >1500 RPS soutenu.

#### P0-B3 — Health probes adaptés multi-worker
- **Fichiers** : `src/api/routes/health.py:99`, `src/api/routes/health_deep.py`.
- **Heures** : 2 h.
- **Stratégie** :
  - `/health` = liveness, < 5 ms, ne lit que des stats process-local.
  - `/health/deep` = readiness, vérifie Redis ping, Postgres `SELECT 1`, ledger head, scanner heartbeat (depuis Redis pubsub ou shared file).
  - Kubernetes/Railway pointe `/health` (liveness probe 5 s) et `/health/deep` (readiness 30 s).
- **Acceptance** : 1 worker degraded → `/health/deep` 503, liveness reste 200 (pas de redémarrage en cascade).

### P0 — Queue notifications fiable (12 h)

#### P0-C1 — Migration `WebhookDeliveryQueue` → Redis Streams + worker pool
- **Fichiers** : `src/delivery/webhook_queue.py`, `src/delivery/webhook_drain_worker.py`, `src/api/dependencies.py:46` (webhook_queue), `tests/test_webhook_queue.py` (à étendre).
- **Heures** : 6 h.
- **Stratégie** :
  1. Backend Redis Streams `webhook:deliveries` + consumer group `drain-workers`.
  2. Worker drain externe (process séparé ou thread dédié) qui consomme, fait `httpx.AsyncClient.post`, ACK ou nack.
  3. Backoff exponentiel + jitter (déjà implémenté en mémoire, à reproduire).
  4. Dead-letter dans `webhook:dead` stream + endpoint admin `/api/v1/admin/webhook/dead-letter` pour requeue.
- **Acceptance** : 1000 deliveries enqueued, 1 worker crash mid-drain → 0 perte, 0 doublon (idempotency_key broker).
- **Dépendances** : Redis (cat. 9), HMAC signer en place (déjà OK `webhook_signer.py`).

#### P0-C2 — Notification queue Telegram / Discord
- **Fichiers** : `src/delivery/notification_queue.py` (nouveau), `src/delivery/telegram_notifier.py`, `src/delivery/discord_notifier.py`.
- **Heures** : 4 h.
- **Stratégie** : Même pattern Redis Streams. Permet de débrancher l'envoi notif du scanner (pas de blocage 1-3 s sur Telegram dans le scanner thread).
- **Acceptance** : Scanner publie 10 signaux/s vers la queue, drain worker consomme à 5 msg/s, aucun signal perdu, latence p95 < 10 s du publish à l'envoi.

#### P0-C3 — Backpressure & circuit breakers wired sur les queues
- **Fichiers** : `src/intelligence/circuit_breaker.py`, queue workers.
- **Heures** : 2 h.
- **Stratégie** : Si Telegram circuit OPEN, le worker dépose les msg dans `notifications:retry` au lieu d'échouer. Métriques Prometheus : `notification_queue_depth`, `webhook_queue_depth`, `notification_dead_letter_count`.

### P0 — OpenAPI complet + versioning (8 h)

#### P0-D1 — Examples 2xx + 4xx sur toutes les routes
- **Fichiers** : `src/api/routes/*.py` (tous), `src/api/models.py`, `src/api/openapi_enrichment.py:95-300`.
- **Heures** : 4 h.
- **Stratégie** :
  - Ajouter `responses={200: {"content": {"application/json": {"example": {...}}}}, 422: {...}}` sur les 20 routes.
  - Schema d'erreur RFC 7807 par défaut (`application/problem+json`).
  - `openapi_enrichment.py` injecte les examples missing en fallback.
- **Acceptance** : `openapi-generator-cli validate -i http://localhost:8000/api/openapi.json` passe sans warning.

#### P0-D2 — Snapshot test contrat OpenAPI
- **Fichiers** : `tests/test_openapi_snapshot.py` (nouveau), `tests/snapshots/openapi.v1.json` (généré).
- **Heures** : 1 h.
- **Stratégie** :
  - Test pytest qui appelle `app.openapi()` et compare au snapshot stocké. Toute modif breaking de contrat fait échouer le test → PR review forcé.
  - `pytest --snapshot-update` pour mettre à jour intentionnellement.
- **Acceptance** : `pytest tests/test_openapi_snapshot.py` passe en CI.

#### P0-D3 — Versioning `/api/v2` côte-à-côte
- **Fichiers** : `src/api/app.py:325`, `src/api/v2/` (nouveau package), `src/api/app_v2.py` (factory).
- **Heures** : 2 h.
- **Stratégie** :
  - Mount sub-app : `app.mount("/api/v2", create_v2_app(...))`.
  - v2 reprend les routes critiques (insights, enrich) avec breaking changes documentés (e.g. champ `narrative_short` toujours obligatoire, `setup` renommé `direction_label`).
  - Header `Sunset: Sat, 1 Jan 2027 00:00:00 GMT` sur v1 dès activation v2.
- **Acceptance** : `/api/v1/signals/current` et `/api/v2/signals/current` répondent indépendamment.

#### P0-D4 — Génération SDK Python + TS en CI
- **Fichiers** : `.github/workflows/sdk.yml` (nouveau), `sdk/python/`, `sdk/typescript/` (output).
- **Heures** : 1 h.
- **Stratégie** : `openapi-generator-cli generate -i openapi.json -g python -o sdk/python`. PR auto-générée à chaque release tag.
- **Acceptance** : `pip install sdk/python && python -c "from smart_sentinel import Client; c = Client(...); c.signals.get_current()"` fonctionne.

### P0 — Load tests (4 h)

#### P0-E1 — Suite k6 scénarios SaaS
- **Fichiers** : `tests/load/k6/signals_read.js`, `enrich_write.js`, `chat_burst.js`, `mixed_sustained.js`.
- **Heures** : 4 h.
- **Stratégie** :
  - `signals_read` : ramp 0→1000 vus, 5 min, `GET /signals/current` + `/history` 80/20.
  - `enrich_write` : 50 vus constants, POST `/enrich` avec Idempotency-Key rotatif.
  - `chat_burst` : 20 vus, POST `/narratives/chat`.
  - `mixed_sustained` : 1 h, 200 vus mix réaliste.
  - Seuils Pass/Fail : `http_req_duration{group:read} p(95)<200`, `http_req_failed<0.01`.
- **Acceptance** : CI nightly run k6, échec si seuils non tenus.

### P1 — Idempotency étendue, request tracing (8 h)

#### P1-F1 — Idempotency-Key sur tous les POST/PUT mutants
- **Fichiers** : `src/api/routes/admin.py` (create_key, rotate_key), `src/api/routes/webhook_ack.py`, futurs `/orders/*`.
- **Heures** : 2 h.
- **Stratégie** : Dépendance FastAPI `idempotency_guard(request, body)` réutilisable.
- **Acceptance** : Replay POST `/admin/keys` avec même Idempotency-Key → retour de la même clé, pas 2 entrées dans `api_keys`.

#### P1-F2 — OpenTelemetry SDK + auto-instrumentation
- **Fichiers** : `requirements.txt` (+`opentelemetry-instrumentation-fastapi`), `src/intelligence/main.py` startup, env vars `OTEL_EXPORTER_OTLP_ENDPOINT`.
- **Heures** : 3 h.
- **Stratégie** :
  - `from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor; FastAPIInstrumentor.instrument_app(app)`.
  - Propagation `traceparent` automatique. Spans manuels autour de `pipeline.query()`, `llm.acall_api()`, `store.get_history()`.
  - Exporter OTLP vers Tempo/Jaeger.
- **Acceptance** : Trace complète bout-en-bout `request → auth → store → llm → ledger → response` visible dans Tempo.

#### P1-F3 — Standardiser pagination cursor-based
- **Fichiers** : `src/api/routes/signals.py:42` (page/page_size → cursor), `src/api/routes/insight_history.py` (déjà cursor), `src/api/models.py` (nouveau `CursorPaginationParams`).
- **Heures** : 2 h.
- **Stratégie** :
  - Cursor opaque base64 `{"seq":12345,"ts":...}`, header `Link: <...>; rel="next"`.
  - `page/page_size` deprecated, accepté + warning header.
- **Acceptance** : Pagination stable sous insert concurrent (pas de doublons / skips).

#### P1-F4 — Filtering DSL standard
- **Fichiers** : `src/api/routes/signals.py`, `src/api/routes/insight_history.py`, `src/api/filtering.py` (nouveau).
- **Heures** : 1 h.
- **Stratégie** : Query params `?filter[symbol]=XAUUSD&filter[outcome]=WIN&sort=-created_at`. Parser sécurisé (whitelist colonnes).
- **Acceptance** : `/signals/history?filter[outcome]=WIN&sort=-pnl_pips` retourne signaux fermés gagnants triés.

### P2 — GraphQL B2B + Postgres (10 h)

#### P2-G1 — Strawberry GraphQL sub-app
- **Fichiers** : `src/api/graphql/__init__.py`, `src/api/app.py:343`.
- **Heures** : 5 h.
- **Stratégie** : Types `Insight`, `Signal`, `AuditEntry`, queries `insights(filter: ..., first: Int, after: String)`, dataloader N+1.
- **Acceptance** : Broker peut faire `query { insights(first:50, filter:{symbol:"XAUUSD"}){ id direction conviction_0_100 sources { label } } }`.

#### P2-G2 — Migration Postgres + asyncpg
- **Fichiers** : `requirements.txt` (+`asyncpg`, `alembic`), `alembic/versions/*`, `src/api/signal_store.py` (PostgresAdapter), `src/api/auth.py`, `src/api/tier_manager.py`.
- **Heures** : 5 h.
- **Stratégie** : Alembic migration `001_init` (schema courant), `002_indexes` (partial `WHERE is_active=1`, `WHERE outcome IS NOT NULL`). Connexion pool asyncpg `min_size=2, max_size=10` par worker.
- **Acceptance** : Bench 1k writes/min XAU + 5k reads/s tient sans queue depth croissante.

---

## 5. Tests & validation

### 5.1 Unit / integration
- **Existant** : 1366+ tests, dont `tests/test_signal_store.py`, `tests/test_auth.py`, `tests/test_tier_manager.py`, `tests/test_insight_signal_v2.py`, `tests/test_insight_signal_v2_enrichment.py`, `tests/test_smoke_e2e.py`.
- **À ajouter** :
  - `tests/test_routes_async_safe.py` : event-loop lag p99 < 50 ms sous charge mixte.
  - `tests/test_openapi_snapshot.py` : contrat figé.
  - `tests/test_multi_worker_consistency.py` : RateLimiter Redis cohérent entre workers (test avec 2 instances FastAPI dans le même process).
  - `tests/test_idempotency_replay.py` étendu à toutes les routes mutantes.

### 5.2 Contract tests B2B
- **Outil** : Schemathesis (`pip install schemathesis`).
- **Commande** : `schemathesis run http://localhost:8000/api/openapi.json --checks all --hypothesis-max-examples=200`.
- **Acceptance** : 0 violation contrat (sauf `not_a_server_error` éventuelles 422 attendues).

### 5.3 Load tests (k6)
- **Scénarios** : §P0-E1.
- **Acceptance globale post-P0** :
  - `signals_read` : 1000 RPS sustained, p95 < 200 ms, errors < 1%.
  - `enrich_write` : 50 RPS, p95 < 1500 ms (LLM-dominated), errors < 1%.
  - `chat_burst` : 20 RPS, p95 < 2500 ms.
  - `mixed_sustained` 1h : p95 toutes routes < 300 ms, RSS stable (pas de leak).

### 5.4 Chaos / soak
- **Outils** : `toxiproxy` (latence Redis injectée), `pumba` (kill aléatoire worker).
- **Tests** : kill -9 worker pendant `mixed_sustained` → Gunicorn respawn < 5 s, 0 requête perdue (Gunicorn graceful timeout).
- **Acceptance** : MTBF 99.9% sur 24 h.

---

## 6. Sécurité

### 6.1 Input validation
- **En place** : Pydantic v2 strict, regex sur `signal_id` (`narratives.py:24`), `sanitize_string` sur user input free-form (`narratives.py:149`, `enrich.py:241`).
- **À ajouter** :
  - **Validation `Origin` / `Referer`** sur POST sensibles (admin endpoints) — défense-en-profondeur même sans cookies.
  - **Body cap chunked** (eval 15 finding 15.2) : middleware qui compte bytes streamés et coupe à 1 MB. Effort 2 h.
  - **Validation X-Forwarded-For** : trust uniquement un set de proxies whitelistés via `TRUSTED_PROXIES=10.0.0.0/8,172.16.0.0/12` env var. Sinon prendre `request.client.host` (eval 15 finding 15.3). Effort 30 min.

### 6.2 Output encoding
- **En place** : JSONResponse partout (UTF-8 sortie standard FastAPI), HTML escape inutile (pas de HTML rendu).
- **À ajouter** : `response_model_exclude_none=True` global pour réduire payload + éviter de leaker des champs `None` non documentés (eval 10 finding 10.2). Effort 30 min.

### 6.3 Rate limiting
- **En place** : 100 req/min/IP (in-RAM), 100 calls/min par api_key (KeyStore.check_rate_limit), tier daily quota (UserTierManager.check_rate_limit). 3 couches.
- **À ajouter** :
  - Redis backend (P0-B1).
  - Headers `X-RateLimit-Limit / X-RateLimit-Remaining / X-RateLimit-Reset` standardisés (déjà partiel via `RateLimitHeadersMiddleware`).
  - **Token bucket** plutôt que sliding window — plus tolérant aux bursts brokers.

### 6.4 CORS
- **En place** : Origins from env `CORS_ALLOWED_ORIGINS` (`app.py:194`), `allow_credentials=False` (par défaut).
- **À ajouter** :
  - **Whitelist explicite** documentée (pas `*` jamais).
  - Vérification CI que `CORS_ALLOWED_ORIGINS` ne contient pas `*` en prod.

### 6.5 CSP / Security headers
- **En place** : HSTS 2y, CSP avec `unsafe-inline` (toléré pour Swagger), X-Frame DENY, X-Content-Type nosniff (`app.py:262`).
- **À ajouter** :
  - **CSP nonce** pour Swagger UI au lieu de `unsafe-inline` (effort 2 h, nécessite rebuild Swagger HTML).
  - **Subresource Integrity** sur les `<script src="cdn...">` Swagger.

### 6.6 Secrets / config
- **En place** : env vars (`ANTHROPIC_API_KEY`, `TELEGRAM_BOT_TOKEN`, etc.), fail-fast au boot si manquant (cf eval 15 finding 15.5).
- **À ajouter** :
  - **Rotation key** (déjà implémentée pour API keys broker via `/admin/keys/{id}/rotate`).
  - **Secret scanning** GitHub Actions sur PR.

---

## 7. Métriques

| KPI | Mesure | Cible | Source |
|-----|--------|-------|--------|
| **req/s** par instance | `http_requests_total` rate(5m) | > 500 RPS soutenu post-P0-A | Prometheus |
| **p50 / p95 / p99 latency** par route | `http_request_duration_seconds` histogram | p99 < 200 ms sur reads, < 2 s sur chat | Prometheus + `/api/v1/metrics/latency` |
| **Error rate** | `http_requests_total{status=~"5.."}` / total | < 0.5% sur 1 h | Prometheus |
| **Event-loop lag p99** | custom probe (`asyncio.get_event_loop().time()` delta) | < 50 ms | exporter custom |
| **DB query p95** | spans OpenTelemetry | < 10 ms (Postgres) / < 5 ms (SQLite) | Tempo |
| **Cache hit rate** (SemanticCache + verify_key cache) | `cache_hits / (hits+misses)` | > 60% LLM, > 90% auth | `/health` |
| **Notification queue depth** | `notification_queue_depth` | < 100 sustained | Redis Streams `XLEN` |
| **Webhook dead-letter count** | `webhook_dead_letter_total` | 0 (alerte si > 5/h) | Prometheus |
| **Concurrent users** | distinct api_key_id last 5min | 800+ sans dégradation | access log |
| **RSS / worker** | `process_resident_memory_bytes` | < 1 GB | Prometheus |
| **Open FDs / worker** | `process_open_fds` | < 500 | Prometheus |
| **Saturation Redis** | `redis_cpu_sys_seconds_total` rate | < 0.5 vCPU | redis_exporter |
| **Idempotency-Key replay rate** | replays / total POST | observabilité ; alerte si > 30% (broker misconfiguré) | logs |
| **OpenAPI breaking change** | snapshot diff | 0 en CI sans PR review | `tests/test_openapi_snapshot.py` |

---

## 8. Risques & mitigations

| Risque | Probabilité | Impact | Mitigation |
|--------|-------------|--------|------------|
| Migration aiosqlite casse les tests qui patchaient `sqlite3.connect` | Moyen | Moyen | Tests bench avant/après, garder facade sync `SignalStore` pour scanner thread non-async |
| Redis indisponible → tout rate-limit / idempotency tombe | Faible | Élevé | Fallback graceful : si Redis ping fail, RateLimiter retombe sur in-RAM par worker + log WARN. Pas de fail-closed sur infra (sinon DoS Redis = DoS API). |
| Multi-worker → scanner doublonne (chaque worker démarre 1 scanner thread) | Élevé si pas géré | Critique | Le scanner DOIT vivre dans un container séparé OU le `lifespan` ne démarre le scanner que sur worker-0 (heuristique `gunicorn.worker_id == 0` via env var). |
| `httpx.AsyncClient` Anthropic vs SDK Python officiel diverge | Faible | Faible | Tests intégration contre stub `anthropic.Anthropic.messages.create`, monitor `anthropic_sdk_version` vs custom http client cohérence |
| OpenTelemetry latence overhead | Faible | Faible | Sampling 10% en prod (`OTEL_TRACES_SAMPLER=parentbased_traceidratio,0.1`) |
| Postgres migration sous load → downtime | Élevé si pas planifié | Critique | Dual-write SQLite+Postgres pendant 2 semaines, cutover après validation, rollback < 1 min |
| GraphQL N+1 explosion sur audit ledger | Moyen | Moyen | DataLoader obligatoire + complexity limit (max 100 nodes/query) |
| Breaking v1 → v2 sans communication | Élevé | Critique | Header `Sunset:` 90 jours d'avance + email broker + changelog versionné |
| Cache verify_key 60 s → clé révoquée reste valide | Moyen | Moyen | TTL configurable, baisser à 10 s pour environnements sensibles. Endpoint admin `POST /admin/keys/{id}/revoke?immediate=true` qui force `_cache_invalidate()` cross-worker via Redis PUB/SUB |
| ASGI lifespan ne ferme pas la DB pool en cas de SIGKILL | Faible | Faible | GracefulShutdownCoordinator déjà branché (`app.py:163`), graceful_timeout Gunicorn 30 s |
| Pydantic v2 strict mode break enrich/webhook payloads broker existants | Moyen | Élevé | Mode `model_config = {"extra": "ignore"}` sur les inputs B2B, mode `"forbid"` sur les outputs internes |

---

## 9. Dépendances avec autres catégories

| Cat. | Couplage | Détail |
|------|----------|--------|
| **3. Auth & API keys** | Forte | KeyStore/TierManager ⇒ migration Postgres P2 cohérente, rotation keys déjà OK. Cache Redis cross-worker pour `_cache_invalidate` |
| **6. Observability** | Forte | OpenTelemetry SDK, Prometheus metrics, structured logging déjà partiellement en place (`access_log.py`). Cat. 6 fournit Tempo / Loki / Grafana ; cat. 8 émet les spans/logs |
| **7. Risk & compliance** | Moyenne | GeoBlock middleware déjà actif. Disclaimers (UE 2024/2811) injectés dans `NarrativeResponse` (`narratives.py:79`). Cat. 7 fournit la légalité, cat. 8 expose |
| **9. Deployment** | Critique | Gunicorn 4 workers + Redis + Postgres = changement majeur Dockerfile + docker-compose + Procfile. Bug `docker-compose.yml:35` ports 8080 vs 8000 à corriger (eval 21 §12 annexe) |
| **10. Notifications/Delivery** | Forte | WebhookQueue + Telegram queue partagent l'infra Redis Streams |
| **11. LLM** | Forte | `httpx.AsyncClient` partagé, cache prompt Anthropic ≥ 1024 tok (eval 05), tier-routing Haiku/Sonnet/Opus |
| **12. Data feeds** | Faible | DataProvider côté scanner thread, l'API ne lit jamais directement le feed |

---

## 10. Estimation totale & timeline

### 10.1 Budget heures

| Sprint | Périmètre | Heures | Jours dev (1 ETP) |
|--------|-----------|--------|-------------------|
| **Sprint API-1** (semaine 1-2) | P0-A (async I/O) + P0-D (OpenAPI + SDK) + P0-E (load tests) | **28 h** | 3.5 j |
| **Sprint API-2** (semaine 3-4) | P0-B (multi-worker + Redis state) + P0-C (queue Redis Streams) | **22 h** | 2.75 j |
| **Sprint API-3** (semaine 5) | P1-F (idempotency étendue, tracing, pagination, filter) | **8 h** | 1 j |
| **Sprint API-4** (semaine 6-8) | P2-G (GraphQL + Postgres migration) | **10 h** | 1.25 j |
| **Total P0 + P1** | bloc commercialisation | **58 h** | **~7.5 j** |
| **Total avec P2** | bloc scale-out | **68 h** | **~8.5 j** |

### 10.2 Timeline (calendrier)

```
Semaine 1 : P0-A1 (async SQLite) + P0-A2 (async LLM)
Semaine 2 : P0-A3 + P0-A4 (async notifs + CI gate) + P0-D1 (OpenAPI examples)
Semaine 3 : P0-D2 (snapshot) + P0-D3 (v2 stub) + P0-D4 (SDK gen) + P0-E1 (k6)
Semaine 4 : P0-B1 (Redis externalize state) + P0-B2 (Gunicorn workers)
Semaine 5 : P0-B3 (health probes) + P0-C1 (webhook Redis Streams)
Semaine 6 : P0-C2 (notification queue) + P0-C3 (backpressure)
Semaine 7 : P1-F1 (idempotency global) + P1-F2 (OpenTelemetry)
Semaine 8 : P1-F3 (cursor pagination) + P1-F4 (filtering DSL)
+ : P2 (GraphQL, Postgres) sur trigger commercial (>1k MAU)
```

### 10.3 Points de validation (gates)

- **Gate post-Sprint API-1** : k6 `signals_read` ≥ 500 RPS p99 < 200 ms.
- **Gate post-Sprint API-2** : 4 workers tournent, rate-limit cohérent ±5%, webhook queue 0 perte sous chaos test.
- **Gate post-Sprint API-3** : OpenTelemetry trace bout-en-bout visible, idempotency replay testé sur 100% routes mutantes.
- **Gate post-Sprint API-4** (si déclenché) : Postgres en prod, bench 1k writes/min + 5k reads/s tient.

### 10.4 Bug accessoire à fixer

- `infrastructure/docker-compose.yml:35-40` mappe ports 8080/9090 alors que l'app écoute 8000 (eval 21 §12 annexe). Correction triviale (15 min) à intégrer dans P0-B2.

---

## Synthèse exécutive

- **Livrable** : `C:\MyPythonProjects\TradingBOT_Agentic\reports\commercialization_sprint\08_api_backend.md`
- **Top 3 P0 (28 h)** : (1) **Async I/O end-to-end** (`aiosqlite` + `httpx.AsyncClient` pour LLM), (2) **Multi-worker via Gunicorn 4 workers + state Redis** (RateLimiter, IdempotencyStore, `_current` retiré), (3) **OpenAPI examples + SDK Python/TS + snapshot test + k6 load suite**.
- **Budget total P0+P1** : **~58 heures / 7.5 jours** dev solo, P2 (GraphQL+Postgres) **+10 h** déclenchable sur traction.
- **Bloqueurs** : Redis + scanner-as-separate-container (cat. 9) sinon multi-worker = scanner dupliqué.
- **KPIs go-live** : 1000 RPS soutenu, p99 < 200 ms reads, event-loop lag p99 < 50 ms, 0 sync I/O dans async def via CI gate.
