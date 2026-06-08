# Eval 10 — FastAPI (routes, perf, sécurité)

**Date** : 2026-04-25
**Périmètre** : `src/api/app.py` (174 lignes), `src/api/routes/*.py` (8 routers, ~720 lignes), `src/api/dependencies.py`, `src/api/models.py`.
**Verdict global** : **6.0/10** — fondations propres et versionnées, mais 3 défauts bloquants pour un usage B2B (event-loop bloquant sur SQLite, `/metrics` non auth, opérateur ouvert à toute clé valide).

---

## 1. Cartographie des endpoints

| Route | Méthode | Auth | Tier-gate | Pagination | Modèle réponse |
|---|---|---|---|---|---|
| `/api/v1/signals/current` | GET | API key | non | n/a | `SignalResponse` (204 si vide) |
| `/api/v1/signals/history` | GET | API key | non | ✅ page/page_size | `SignalHistoryResponse` |
| `/api/v1/signals/state` | GET | API key | non | n/a | `PublicStateResponse` |
| `/api/v1/health` | GET | public | n/a | n/a | `HealthResponse` |
| `/health` | GET | public (Docker) | n/a | n/a | `HealthResponse` |
| `/api/v1/admin/keys` | POST/GET/DELETE | HMAC admin | admin only | non | `KeyCreate/Revoke/List` |
| `/api/v1/admin/usage` | GET | HMAC admin | admin only | days clamp | `UsageResponse` |
| `/api/v1/operator/{metrics,risk,kill-switch}` | GET | API key | **❌ pas de tier check** | n/a | `OperatorMetricsResponse` etc. |
| `/api/v1/dashboard/{summary,equity-curve}` | GET | API key | non | days clamp | `Performance/Equity*Response` |
| `/api/v1/narratives/{signal_id}` | GET | API key | ✅ tier gating in-route | n/a | `NarrativeResponse` |
| `/api/v1/narratives/chat` | POST | API key | ✅ INSTITUTIONAL only | n/a | `ChatResponse` |
| `/api/v1/scanner/status` | GET | API key | non | n/a | `ScannerStatusResponse` |
| `/metrics` | GET | **❌ public** | n/a | n/a | text/plain Prometheus |
| `/api/docs` | GET | public | n/a | n/a | Swagger UI |

**Total** : 14 routes effectives. Préfixe `/api/v1/` cohérent (sauf `/metrics` et `/health` Docker, légitime).

---

## 2. Conformité standards REST

| Critère | Statut | Détail |
|---|---|---|
| Versioning explicite | ✅ | `/api/v1/` partout |
| Pagination | ✅ partiel | `/history` ✅, mais admin/keys non paginé (acceptable < 1000 clés) |
| Tri / filtrage | ❌ | `/history` ne supporte ni `sort` ni `filter` (date range, action, symbol) |
| HTTP verbs corrects | ✅ | GET lecture, POST création, DELETE révocation |
| Codes statut | ✅ partiel | 200/204/401/403/404/413/429/500/503 OK ; **400 absent sur narratives/chat (regex fail → 400 OK)** |
| RFC 7807 (problem+json) | ❌ | `ErrorResponse{error,detail}` custom, pas de `type/title/instance` |
| Cohérence erreurs | ❌ | Mix raw dict (`payload_too_large`, `rate_limit_exceeded` middleware) vs `ErrorResponse` (global handler) vs `HTTPException` default `{detail:...}` |
| ETag / If-None-Match | ❌ | Aucun cache HTTP — `/signals/current` polled toutes les secondes |
| HEAD / OPTIONS | partiel | FastAPI gère CORS preflight, mais `allow_methods=["GET","POST","DELETE"]` exclut OPTIONS explicite (CORSMiddleware ajoute) |
| Idempotency-Key | ❌ | POST `/admin/keys` non idempotent — double-clic = 2 clés |
| OpenAPI spec | ✅ | Auto-FastAPI ; `/api/docs` Swagger ; **redoc désactivé** (perte UX dev partenaire) |
| HATEOAS / liens | ❌ | Aucun lien dans payload (acceptable pour API simple) |
| Webhooks sortants | ❌ | Tier INSTITUTIONAL prévoit `webhooks=True` mais aucune route `/webhooks/*` ni dispatcher |
| SDK généré | ❌ | Aucun ; partenaire B2B doit lire OpenAPI à la main |

**Note REST** : 5/10 — fonctionne mais incohérences erreurs + manque ETag/idempotency/webhooks pour entreprise.

---

## 3. Sécurité — analyse ligne par ligne

### 3.1 Failles confirmées

| # | Sévérité | Endpoint | Description | Ref ligne |
|---|---|---|---|---|
| **F1** | 🔴 CRITIQUE | `/metrics` | Aucune auth, expose business metrics (signals_generated, llm_calls, cache_hits, latency histograms). Concurrent peut scraper et déduire fréquence d'usage / cost. | `routes/prometheus.py:12-20` |
| **F2** | 🔴 CRITIQUE | `/api/v1/operator/*` | Auth `require_api_key` mais **pas de tier check** — n'importe quelle clé FREE ($0) accède aux VaR, kill-switch, drawdown et MetricsRegistry complète. | `routes/operator.py:18-77` |
| **F3** | 🟠 HAUTE | `/api/v1/narratives/chat` | Réponse contient `cost_usd` du LLM — leak de la business cost (Sonnet vs Haiku déductible par latence + coût). | `routes/narratives.py:160` |
| **F4** | 🟠 HAUTE | `app.py` | Aucun middleware **security headers** (X-Frame-Options, X-Content-Type-Options, Strict-Transport-Security, Content-Security-Policy). Test `Grep "X-Frame"` = 0 hit. | `app.py:90-95` |
| **F5** | 🟡 MOYENNE | `/admin/keys` POST | Pas d'audit log (qui a créé/révoqué quelle clé, IP source). Non-repudiation impossible pour SaaS B2B sous SOC2. | `routes/admin.py:25-41` |
| **F6** | 🟡 MOYENNE | `narratives.py:155` | Appel à `llm_engine._call_api(...)` (méthode privée). Tight coupling + bypass des contrôles internes potentiels. | `routes/narratives.py:155` |
| **F7** | 🟢 FAIBLE | `narratives.py:147-153` | Le contexte injecté dans prompt LLM contient `record.narrative` (contenu LLM précédent) → boucle d'injection si user upload narrative custom (non implémenté pour l'instant). | `routes/narratives.py:153` |
| **F8** | 🟢 FAIBLE | `app.py:101-108` | Body limit 1MB basé sur `content-length` header — si client omet header (chunked), pas d'enforcement. | `app.py:101-108` |

### 3.2 OWASP API Top 10 (2023) — score

| OWASP | Statut | Commentaire |
|---|---|---|
| API1 — Broken Object Level Authz | ✅ | signal_id formaté (regex), pas d'IDOR détecté |
| API2 — Broken Auth | ⚠️ | TESTING_MODE=1 par défaut (cf. eval_11) ; mais requires_api_key OK quand prod |
| API3 — Broken Object Property Level Authz | ⚠️ | Tier gating sur narrative champs ✅, mais `/operator/*` **pas tier-gated** |
| API4 — Unrestricted Resource Consumption | ⚠️ | Body 1MB ✅, rate-limit IP ✅, mais pas de **tier-based rate** ; chat endpoint peut spammer LLM coûteux |
| API5 — Broken Function Level Authz | 🔴 | `/operator/*` accessible avec n'importe quelle clé valide |
| API6 — Unrestricted Access to Sensitive Business Flow | ⚠️ | `/admin/keys` HMAC ✅ ; pas d'audit log |
| API7 — SSRF | ✅ | Aucune route ne fait de fetch d'URL utilisateur |
| API8 — Security Misconfig | 🔴 | `/metrics` ouvert + pas de security headers |
| API9 — Improper Inventory | ⚠️ | OpenAPI ✅ ; pas de versioning policy (deprecation, sunset headers) |
| API10 — Unsafe Consumption of APIs | ✅ | Anthropic / Telegram appels via circuit breaker |

**Score OWASP** : 5/10 → 5 ⚠️ + 3 🔴 sur 10.

---

## 4. Performance — pièges identifiés

### 4.1 Event-loop blocking 🔴 CRITIQUE

Toutes les routes sont `async def` mais appellent du **SQLite synchrone** (`sqlite3` stdlib) sans `run_in_executor`/`asyncio.to_thread`. Sur un endpoint typique :

```
GET /api/v1/signals/history?page=1&page_size=20
→ store.get_history()  ← BLOQUE l'event-loop sur SQLite
   COUNT(*) sur n signaux + SELECT … OFFSET
```

**Conséquence quantifiée** : avec 1 worker uvicorn, P99 explose dès qu'un client lent ou une requête longue arrive. Sur 100 RPS soutenus, latence médiane peut multiplier ×3-5 vs implémentation non-bloquante. Sources : FastAPI docs section "Sync routes vs async routes", benchmark uvicorn 2024.

**Files concernées** : `signal_store.py` (8 méthodes), `signal_tracker.py` (3 méthodes), `auth.py` KeyStore (6 méthodes), `tier_manager.py` (8 méthodes). **~25 fonctions sync appelées depuis des coroutines async**.

### 4.2 Latence bench (estimation analytique)

| Endpoint | Op SQLite | Bytes payload | P50 estimé | P95 estimé |
|---|---|---|---|---|
| `/signals/current` | 0 (cache mémoire `_current`) | ~300 B | <1 ms | 2 ms |
| `/signals/history?page_size=20` | 1 COUNT + 1 SELECT | ~10 KB | 5-15 ms | 50 ms |
| `/signals/state` | 0 (in-memory state machine) | ~1 KB | <1 ms | 3 ms |
| `/health` | health checker dispatch | ~500 B | 1-5 ms | 30 ms |
| `/narratives/{id}` | 1 SELECT by PK | ~3 KB | 2-8 ms | 25 ms |
| `/narratives/chat` | LLM call (Sonnet) | ~5 KB | 1500 ms | 8000 ms |
| `/operator/metrics` | full metrics dump | ~50-200 KB | 10-30 ms | 80 ms |

**Goulot identifié** : sous charge concurrente, SQLite WAL permet plusieurs lecteurs mais 1 seul écrivain — `record_usage()` appelé **à chaque requête authentifiée** crée une contention écrivain.

### 4.3 Manque d'observabilité

- `request_logging` middleware logs au niveau **DEBUG** (`logger.debug` ligne 133) → invisible avec LOG_LEVEL=INFO en prod. Latence non tracée.
- Pas de **request_id / trace_id** propagé header → log → metrics. Impossible de corréler un signal de bout en bout.
- L'histogramme `http_request_duration_seconds` est labelé par `path` brut → cardinalité explosive si signal_id dans path (n'arrive pas ici, mais piège classique).

---

## 5. Multi-tenant readiness

| Dimension | État | Remédiation |
|---|---|---|
| Isolation utilisateur | ⚠️ partial | KeyStore + UserTierManager mappent clé → user, mais `/signals/current` est **mondial** — tous les abonnés voient le même signal mondial XAUUSD. OK pour produit "alerts globaux" mais limite si on veut watchlists perso. |
| Quotas par tier | ❌ | TIER_CONFIG définit `api_calls_per_day` mais **non enforcé** dans `require_api_key` (cf. eval 11). |
| State partagé entre workers | ❌ | `SignalStore._current` en RAM, pas sync entre workers uvicorn → réponses incohérentes selon worker hit. |
| Webhooks par tenant | ❌ | Pas implémenté |
| 50 utilisateurs × 6 symbols | ✅ tenable | Volume écriture SQLite ~ négligeable ; LLM cost domine (cf. eval 6 / 24) |
| 1k+ utilisateurs | ⚠️ | SQLite WAL ne tiendra pas concurrent writes — migration Postgres requise (cf. eval 12) |

---

## 6. Top 5 améliorations priorisées

| # | Amélioration | Effort | Impact business | Impact technique |
|---|---|---|---|---|
| **R1** | **Auth + tier gate sur `/operator/*` et `/metrics`** (require_admin + Prometheus token) | 1 jour | 🔴 critique : empêche fuite de business metrics & operator privilege escalation | Sécurise OWASP API5 + API8 |
| **R2** | **Async DB layer**: wrapper `asyncio.to_thread` autour des appels SignalStore/KeyStore/TierManager OU migration `aiosqlite` | 3 jours | Permet 5-10× plus de RPS sur même hardware → marge brute supérieure | P99 latence /history : 50 ms → 8-12 ms estimé |
| **R3** | **Tier-based rate limiting** : enforcer `api_calls_per_day` dans `require_api_key` après lookup tier | 1 jour | Empêche abuse FREE / différenciation premium réelle | Lien direct avec business model |
| **R4** | **Audit log + idempotency**: table `admin_audit` (action, actor, ip, ts) + Idempotency-Key sur POST keys | 2 jours | SOC2 / ISO27001 prerequisite ; évite doublons clé | Non-repudiation, traceability |
| **R5** | **Security headers + RFC 7807 errors + request_id propagation** | 1 jour | Pen-test pass + debug 10× plus rapide | Conformité standard, observabilité |

**Matrice effort × impact** (1 = bas, 5 = haut) :

```
Impact ↑
  5 |        R1
  4 |  R2          R3
  3 |        R4
  2 |              R5
  1 |
    +-------------------→ Effort
       1   2   3   4   5
```

R1 et R3 = **quick win, gros impact** → faire en premier.

---

## 7. Plan d'exécution

### Quick wins (< 1 jour)
- **QW1** Auth Prometheus `/metrics` : ajouter check `X-Prometheus-Token` env var (1 h).
- **QW2** Tier check dans `/operator/*` : ajouter `if subscriber["tier"] != "INSTITUTIONAL": raise 403` (15 min).
- **QW3** Cohérence erreurs : passer tous les `JSONResponse` middleware au modèle `ErrorResponse` (30 min).
- **QW4** Security headers : middleware `add_security_headers` (X-Frame-Options=DENY, X-Content-Type-Options=nosniff, Strict-Transport-Security en prod) (30 min).
- **QW5** `request_logging` : passer en INFO + ajouter `request_id` en header de réponse (1 h).
- **QW6** Re-activer `/api/redoc` (1 ligne).

### Moyen terme (< 1 semaine)
- **MT1** `aiosqlite` migration ou `asyncio.to_thread` wrappers sur 25 appels DB ; ajout pool async.
- **MT2** Tier-based rate limit : middleware injecté avec accès au `tier_manager`.
- **MT3** RFC 7807 problem+json (refacto `ErrorResponse` → `ProblemDetails{type,title,status,detail,instance}`).
- **MT4** Idempotency-Key middleware (cache Redis ou SQLite TTL 24h).
- **MT5** OpenAPI tags & descriptions complètes ; génération SDK Python via `openapi-python-client`.
- **MT6** Webhook dispatcher (`/api/v1/webhooks` POST register URL ; queue + signature HMAC).

### Long terme (> 1 semaine)
- **LT1** Migration FastAPI → Starlette routes pures + uvloop ; ASGI lifespan pour pool DB partagé.
- **LT2** Multi-tenant SignalStore (par symbole + watchlist user) avec Redis pub/sub.
- **LT3** API versioning policy : Sunset header, deprecation warnings, /v2 cohabitation.
- **LT4** SDK Python + JS publiés sur PyPI/npm avec exemples Postman.

---

## 8. KPIs mesurables post-amélioration

| KPI | Baseline (estimé) | Cible 30 j | Cible 90 j |
|---|---|---|---|
| P95 latence `/signals/history` | 50 ms | 15 ms | 8 ms |
| P99 latence `/health` | 30 ms | 10 ms | 5 ms |
| Throughput max (1 worker) | ~80 RPS | 250 RPS | 500 RPS |
| Routes auth-gated correctement (% des sensibles) | 70 % | 100 % | 100 % |
| Score OWASP API Top 10 (✅ / 10) | 4 | 8 | 10 |
| Audit log coverage admin actions | 0 % | 100 % | 100 % |
| Conformité RFC 7807 (% endpoints) | 0 % | 100 % | 100 % |
| % requêtes avec request_id traçable | 0 % | 100 % | 100 % |
| TTR (mean time to resolve issue prod) | inconnu | -50 % | -75 % |
| `cost_usd` exposé dans payload public | oui | non | non |

---

## 9. Trade-offs assumés

- **R2 async DB** ajoute une dépendance (`aiosqlite`) et complexifie testing → contre-partie : 5-10× throughput.
- **R3 tier rate limit** peut bloquer un abonné FREE qui faisait du polling agressif → mais c'est précisément le but business.
- **Idempotency** ajoute latence ~1-2 ms par requête + storage 24h → acceptable.
- Webhooks ajoutent surface d'attaque (SSRF si on ne whitelist pas) → mitiger via signature HMAC + rate-limit.

---

## 10. Note finale par dimension

| Dimension | Note /10 | Justification |
|---|---|---|
| Robustesse code | 7 | Patterns propres, dependency injection clean, exception handler global |
| Conformité REST | 5 | Versioning ok, mais erreurs incohérentes, pas d'idempotency/ETag |
| Sécurité | 4 | 2 failles critiques (operator, /metrics) + headers manquants |
| Performance | 5 | Event-loop bloqué sur SQLite — bug architectural majeur |
| Observabilité | 4 | Pas de request_id, logs middleware en DEBUG |
| Documentation | 7 | OpenAPI auto + redoc désactivé (regression) |
| Multi-tenant readiness | 5 | Auth ok, mais `_current` non sync, pas de quotas tier |
| Différenciation B2B | 4 | Pas de webhooks, pas de SDK, pas d'idempotency |
| **Global** | **6.0/10** | **Production-ready pour test perso ; pas pour B2B sans R1-R5** |

---

## Annexe A — fichiers et lignes critiques

- `src/api/app.py:101-108` body size limit (header-only)
- `src/api/app.py:110-125` rate limiter middleware
- `src/api/app.py:128-150` request logging (DEBUG only)
- `src/api/app.py:153-162` global exception handler
- `src/api/routes/prometheus.py:12-20` `/metrics` non auth-gated
- `src/api/routes/operator.py:17-77` opérateur sans tier check
- `src/api/routes/narratives.py:155,160` `_call_api` privé + `cost_usd` exposé
- `src/api/routes/admin.py:25-41` admin actions sans audit log
- `src/api/signal_store.py` toutes méthodes synchrones appelées depuis async routes

## Annexe B — reproductibilité

```bash
# Auth check on /metrics (devrait renvoyer 401 après R1)
curl -i http://localhost:8000/metrics

# Tier check on /operator (devrait renvoyer 403 avec clé FREE après R1)
curl -H "X-API-Key: sk_<free_key>" http://localhost:8000/api/v1/operator/risk

# Bench latence avec wrk
wrk -t4 -c100 -d30s -H "X-API-Key: sk_<test>" http://localhost:8000/api/v1/signals/history?page=1
```
