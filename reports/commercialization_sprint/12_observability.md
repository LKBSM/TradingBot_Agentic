# Plan de Commercialisation — Catégorie 12 : Observability & Monitoring

> **Date** : 2026-05-23
> **Auteur** : Agent Catégorie 12 — Observability & Monitoring (sprint commercial)
> **Branche** : `institutional-overhaul`
> **Objectif structurant** : passer d'une observabilité « structurelle » (les bons os : `/health`, `JSONFormatter`, `MetricsRegistry`, `HealthChecker`, `CircuitBreaker`) à une observabilité **opérationnelle commercialisable** : SLO formels, alerting actionnable, MTTR < 15 min, dashboards business + tech, traçabilité `signal_id` end-to-end, conformité log retention/PII.
>
> **Périmètre** : structured logs (JSON), `/metrics` (Prometheus), `/health` (live/ready/deep), distributed tracing (OpenTelemetry → Tempo/Jaeger), alerting (PagerDuty + Discord + Telegram fallback), dashboards (Grafana business + tech), SLOs/Error-Budget, incident response, status page.
>
> **Périmètre exclu** : compliance RGPD log retention juridique (cf. cat. 18), MLOps experiment tracking (cf. cat. 19), tests CI (cf. cat. 13), perf engineering (cf. cat. 17). Dépendances explicitées §9.
>
> **Statut commercial** : 🟠 **MVP partiellement déblayé**. `MetricsRegistry` est désormais instanciée (Sprint INFRA-1.2 `src/performance/observability.py:181-206`), Sentry stub câblé, `/health/deep` existe (`src/api/routes/health_deep.py:1-328`). MAIS : aucune métrique business émise depuis le scanner, 159 `print()` répartis sur 29 fichiers (vs 109 audités en avril), aucun `trace_id` propagé, alertes Prometheus écrites mais jamais évaluées en prod (Alertmanager pas déployé), pas de dashboard Grafana provisionné, SLO inexistants. Distance go-live commercial **B2C solo** = 16-22h. Distance go-live commercial **B2B-API enterprise SLA** = 80-120h supplémentaires.

---

## 0. Synthèse exécutive (TL;DR)

| Dimension | Note actuelle | Note cible J+30 | Note cible J+90 | Source |
|-----------|--------------:|----------------:|----------------:|--------|
| Logs (structuration JSON + contexte) | 5.0/10 | 8/10 | 9/10 | `reports/eval_16_observability.md:683` |
| Metrics actives (instances + scrape OK) | **3.5/10** (registry câblée, 0 émission business) | 7.5/10 | 9/10 | `src/intelligence/main.py:276-281` |
| Tracing distribué (`trace_id` end-to-end) | 1.0/10 | 5/10 | 8/10 | absence `contextvars` métier |
| Alerting (rules + Alertmanager + runbooks) | 2.5/10 (rules écrites, pas câblées) | 7/10 | 9/10 | `infrastructure/alert-rules.yml` |
| Health checks (live + ready + deep) | **8.5/10** (already strong) | 9/10 | 9/10 | `src/api/routes/health.py`, `health_deep.py` |
| Status page publique | 0/10 | 0 (deferred) | 6/10 | n/a |
| SLO / Error-Budget | 0/10 | 4/10 | 8/10 | n/a |
| Dashboards Grafana (business + tech) | 1/10 (infra `infrastructure/grafana/` présent, 0 dashboard sentinel) | 6/10 | 9/10 | n/a |
| Coût discipline (free-tier first) | 9.0/10 | 9/10 | 8/10 (Pro tier) | — |
| **GLOBAL** | **3.8 / 10** | **6.5 / 10** | **8.2 / 10** | — |

**Top 3 bloqueurs commerciaux à fixer IMMÉDIATEMENT (P0, ~8-10h)** :

1. **Émettre les 6 métriques business minimales** (`signals_emitted_total`, `scan_duration_seconds`, `llm_latency_seconds`, `llm_cost_usd_total`, `circuit_breaker_state`, `cache_lookups_total`) depuis `sentinel_scanner.py:_scan_once()` et `llm_narrative_engine.py:_call_api()`. Sans ces métriques, le `/metrics` retourne du vide business et **aucune alerte business n'est calibrable**.
2. **Fusionner `extra={}` dans `JSONFormatter`** (`src/intelligence/main.py:39-52` patch §4.P0-2) → débloque 100+ call-sites du contexte structuré (`signal_id`, `symbol`, `tier`, `trace_id`). Sans ça, les logs sont non-corrélables → MTTR ×3.
3. **Migrer les 25 `print()` critiques** identifiés dans eval_16 §2.1 (alert_manager, hmac_manager, dead_man_switch, kill_switch_store, mt5_connector, async_order_manager) vers `logger.*` structuré. `hmac_manager.py:220-223` imprime des secrets en clair = fuite GDPR potentielle bloquante.

**Effort total Catégorie 12** : **~92h sur 6 semaines** (split P0=18h, P1=42h, P2=32h). Détail §10.

**Coût additionnel observability stack** : **$0/mois** (free tier Grafana Cloud + Sentry Free + UptimeRobot) → **$30-50/mois** à scale (>100k signaux/jour, > 10 MAU payants).

---

## 1. État actuel (Audit)

### 1.1 Logs — état du logging structuré

| Élément | Fichier:ligne | Constat | Évaluation |
|---------|---------------|---------|------------|
| `setup_logging()` centralisé | `src/intelligence/main.py:55-73` | Switch `LOG_FORMAT=json` ↔ text. Suppression `httpx`, `httpcore`, `hmmlearn` en WARNING. | OK |
| `JSONFormatter` | `src/intelligence/main.py:39-52` | Payload : `ts, level, logger, msg, exception`. **Ignore `extra={}`** — donc tout `logger.info("...", extra={"signal_id": ...})` perd `signal_id` à la sérialisation. | 🔴 BLOQUANT |
| `setup_structured_logging()` alternatif | `src/performance/logging_config.py:1-100+` | **Implémentation orpheline** : contextvars `_trade_id, _agent_id, _session_id, _request_id`, `LogContext` context manager, intègre `extra` proprement. **Non câblée à `main.py`**. | 🟠 Code dupliqué — choix architectural à trancher |
| `print()` dans `src/` | 29 fichiers, **159 occurrences** (grep 2026-05-23 confirme augmentation depuis 109 d'avril) | Bypass total du JSONFormatter. Voir §1.1.a pour le top-10 critique. | 🔴 BLOQUANT (25 critiques) |
| Niveau log par défaut | `LOG_LEVEL=INFO` | OK | OK |
| Filtrage PII | Aucun scrubber actif sur les `logger.exception()` | Risque traces API keys / chat_id Telegram exposés. Le `_scrub_pii` dans `src/performance/observability.py:154-173` ne s'applique qu'à Sentry, pas aux logs locaux. | 🟠 P1 |

#### 1.1.a Top-10 fichiers `print()` (grep 2026-05-23)

```
src/agent_trainer.py                    : 13 occurrences  — CLI demo, OK à laisser
src/agents/monitoring.py                :  9 occurrences  — console.print(rich), OK
src/security/alert_manager.py           :  8 occurrences  — CRITIQUE: chemin alertes
src/security/hmac_manager.py            :  4 occurrences  — CRITIQUE: secrets en clair
src/live_trading/alerting.py            :  3 occurrences  — bypass logging
src/security/dead_man_switch.py         :  2 occurrences  — CRITIQUE: heartbeat
src/agents/data/fred_provider.py        :  8 occurrences  — NOUVEAU (vs eval 04)
src/research/a1_train.py                : 15 occurrences  — NOUVEAU script training
src/intelligence/rag/pipeline.py        : 12 occurrences  — NOUVEAU pipeline RAG
src/agents/sprint2_intelligence.py      :  1 occurrence   — orchestration
```

Cf. `Grep "print\\(" src/` avec `output_mode=count`. **Trois nouveaux fichiers critiques apparus depuis eval_16** : `src/research/a1_train.py` (15), `src/intelligence/rag/pipeline.py` (12), `src/agents/data/fred_provider.py` (8). Ces ajouts sont d'avant que le sprint OBS ait été terminé — il faut les rattraper.

#### 1.1.b Contexte structuré absent

Grep `extra={` dans `src/` : **6 occurrences** au total, dont 4 dans `src/performance/logging_config.py` (le module orphelin). Aucune dans `src/intelligence/sentinel_scanner.py` (~30 `logger.*`) ni `src/intelligence/llm_narrative_engine.py`. Conséquence : `signal_id` est interpolé en chaîne (`"... signal %s ..."`) au lieu d'être un champ JSON. **Logs non-corrélables → impossible de tracer un signal de bout en bout** sans grep manuel des messages.

### 1.2 Metrics — état Prometheus

| Élément | Fichier:ligne | Constat |
|---------|---------------|---------|
| `MetricsRegistry` (registry custom) | `src/performance/metrics.py:1-469` | 469 lignes propres, Counter/Gauge/Histogram, export Prometheus text + JSON. Format conforme. |
| Instanciation au boot | `src/intelligence/main.py:276-281` (via `init_observability`) | ✅ **Câblée depuis Sprint INFRA-1.2** (`src/performance/observability.py:181-206`). 3 métriques pré-enregistrées (`signals_generated_total`, `llm_latency_seconds`, `circuit_breaker_open_total`). |
| Endpoint `/metrics` | `src/api/routes/prometheus.py:11-20` | ✅ Plus de payload vide ; retourne le format Prometheus avec les 3 metrics de base au minimum dès le boot. |
| **Émission depuis le scanner** | `src/intelligence/sentinel_scanner.py` | 🔴 **Aucune ligne** `self._metrics.counter(...).inc(...)`. Le scanner ignore complètement la registry. → Les métriques restent à 0. |
| **Émission depuis le LLM engine** | `src/intelligence/llm_narrative_engine.py` | 🔴 Aucune émission latence/coût/tokens. |
| **Émission depuis circuit breakers** | `src/intelligence/circuit_breaker.py:118-150` | 🔴 Pas de `gauge.set()` sur transition d'état. Le compteur `circuit_breaker_open_total` n'est jamais incrémenté. |
| **Émission depuis notifiers** | `src/delivery/{telegram,discord}_notifier.py` | 🔴 Aucune émission `notifier_send_total{channel, status}`. |
| Cardinalité plan-mandatée | spec §3.1 ci-dessous | 12 métriques cibles, ~1170 séries, < seuil Grafana Cloud Free 10k. |

**Verdict metrics** : registry câblée mais aucun callsite n'émet. `/metrics` est techniquement non-vide (les 3 stub metrics existent) mais business-vide. C'est le bloqueur P0 #1.

### 1.3 Health checks — état

| Endpoint | Source | État | Verdict |
|---|---|---|---|
| `GET /api/v1/health` | `src/api/routes/health.py:116-119` | Riche : status, components, uptime, kill_switch_level, is_trading_active, testing_mode, scanner_running, signals_generated, cache_*, operational_kill_switch | ✅ Excellent |
| `GET /health` (Docker alias) | `src/api/routes/health.py:122-125` | Idem | ✅ OK |
| `GET /api/v1/health/deep` | `src/api/routes/health_deep.py:279-327` | Active probe : audit_ledger, rag_pipeline, cost_quota, webhook_queue, embedder, tier_rate_limiter. 503 si dégradé. Cache 30s par app_state. | ✅ Excellent — au-dessus du standard SaaS |
| `GET /api/v1/health/live` | inexistant | Distinction live vs ready non implémentée. K8s/Railway probes pointent sur `/health` qui fait trop de travail à chaque check. | 🟠 P1 |
| `GET /api/v1/health/ready` | inexistant | Idem. | 🟠 P1 |

**Forces** : `/health/deep` est plus avancé que beaucoup de SaaS commerciaux (RAG fingerprint + audit ledger verify + cost quota snapshot). C'est la dimension la plus mûre de la cat.12.

**Faiblesses** : pas de séparation live/ready, donc un orchestrateur ne peut pas distinguer « process up mais pas prêt à recevoir trafic » de « process bloqué, à redémarrer ». Important pour blue/green deployment.

### 1.4 Tracing — état

```bash
$ Grep -r "trace_id|opentelemetry|otel" src/
src/performance/logging_config.py     # contextvars définis MAIS pas branchés
```

**État** : 0 span métier instrumenté. Aucun `trace_id` propagé. Le pipeline 7 étages (DataProvider → SMC → Regime → News → Vol → Confluence → StateMachine → Narrative → Store → Notifier) est totalement opaque à l'analyse de latence par étage.

**Conséquence commerciale** : impossible de répondre à un client B2B-API qui dit « le webhook a mis 12s à arriver » — on ne sait pas si c'est le LLM (typique 1.2-8s), Telegram (200-1500ms), MT5 fetch (80-300ms), ou un combo. MTTR sur ce type d'incident est **illimité**.

### 1.5 Alerting — état

| Élément | Source | État |
|---|---|---|
| `infrastructure/alert-rules.yml` | `infrastructure/alert-rules.yml` (existe) | Contient des règles **trading legacy** (drawdown, daily PnL) — pas adaptées au Sentinel scanner. Voir §4.P0-4 pour le set sentinel-spécifique à substituer. |
| `infrastructure/alertmanager.yml` | présent | Config présente mais Alertmanager n'est **jamais lancé en prod** : `infrastructure/docker-compose.yml` ne mappe pas le service en production deploy (Railway). |
| Routing Discord / Telegram | n/a | Inexistant. |
| Routing PagerDuty / Opsgenie | n/a | Inexistant. **Pour SLA enterprise B2B, c'est bloquant.** |
| Runbooks par alerte | n/a | Inexistants. |
| Inhibitions / silences | n/a | Inexistants. |

### 1.6 Dashboards — état

```bash
$ ls infrastructure/grafana/
dashboards/ provisioning/
$ ls infrastructure/grafana/dashboards/
# (à vérifier — probablement vide ou legacy RL trading)
```

**Hypothèse** (à confirmer au build) : dashboards legacy de l'époque RL trader, **pas un seul dashboard Sentinel** (signaux émis, score distribution, LLM cost, cache hit rate, circuit breaker state). Le `reports/eval_16_grafana_business.json` mentionné en annexe d'eval_16 §10 n'est pas commité.

### 1.7 Status page — état

Inexistante. Aucune URL `status.smartsentinel.ai`. Aucun composant publique-facing. Pour solo founder en phase pré-MRR : pas bloquant. Pour pitch enterprise B2B : il faudra l'ajouter avant le premier deal.

### 1.8 Sentry / Error tracking — état

| Élément | Fichier:ligne | État |
|---|---|---|
| `init_sentry()` | `src/performance/observability.py:95-151` | ✅ Câblé. Active si `SENTRY_DSN` env var défini. Sample rate 5%. `_scrub_pii` hook drop les events suspects (ANTHROPIC_API_KEY, FRED_API_KEY, TELEGRAM_BOT_TOKEN). |
| Intégration FastAPI | auto-détectée via `sentry_sdk` quand FastAPI importé | ✅ |
| `LoggingIntegration` | `level=INFO`, `event_level=ERROR` | ✅ Tous les `logger.exception()` remonté Sentry sans code supplémentaire. |
| **DSN configuré en prod** | n/a | 🔴 `SENTRY_DSN` non listé dans `.env.example` ni les env vars de prod (cf. `MEMORY.md`). |

Verdict : code prêt, mais le DSN n'est jamais setté → Sentry désactivé silencieusement. P0 trivial : créer un projet Sentry Free et seter le DSN.

### 1.9 Log aggregation — état

Inexistant. Logs stdout uniquement. Sur Railway/Docker, ils sont conservés ~7 jours par l'orchestrateur mais sans recherche structurée. Sur incident, le seul outil est `docker logs sentinel | grep …`.

---

## 2. Vision cible

### 2.1 Stack de référence (free-tier first, scalable)

```
[ Sentinel Scanner (Python) ]
    │
    ├──► logger.* (JSON, extra={signal_id, symbol, trace_id, ...})
    │        │
    │        └──► stdout ──► Promtail (sidecar Loki) ──► Loki (Grafana Cloud Free, 50GB/mo)
    │                   ──► Sentry SDK (errors only) ──► Sentry (Free tier 5k events/mo)
    │
    ├──► MetricsRegistry (12 métriques business)
    │        └──► /metrics ──► Prometheus (Grafana Cloud Free, 10k series)
    │                            │
    │                            └──► Alertmanager ──► Discord webhook (P0/warning)
    │                                             ──► Telegram fallback (P0/critical, redondant)
    │                                             ──► PagerDuty (P0/critical, enterprise SLA J+90)
    │
    ├──► Spans (contextvars J+30, OTel SDK J+90) ──► OTLP HTTP ──► Tempo (Grafana Cloud Free, 50GB/mo)
    │
    └──► /health (live), /health/ready, /health/deep ──► UptimeRobot (Free 50 monitors)
                                                    ──► Status page instatus.com (Free 1 page)
```

### 2.2 SLO formels (cibles commerciales)

| Service | SLI | SLO J+30 (solo testing) | SLO J+90 (paid public) | SLO J+180 (enterprise B2B) |
|---|---|---:|---:|---:|
| **Scanner uptime** | `1 - (down_minutes / total_minutes)` calculé sur `bars_scanned_total` heartbeat | 99.0% | 99.5% | 99.9% |
| **API availability** | `2xx + 3xx / total_requests` sur `/api/v1/signals`, `/api/v1/insights/*`, `/health` | 99.0% | 99.5% | 99.9% |
| **Signal latency end-to-end** | P95 de `scan_duration_seconds` (DataProvider → Notifier dispatched) | < 5s | < 3s | < 2s |
| **LLM narrative latency** | P95 de `llm_latency_seconds` | < 5s | < 4s | < 3s |
| **Webhook delivery latency** | P95 délai entre `signal.created_at` et `webhook.delivered_at` | n/a | < 10s | < 5s |
| **Webhook delivery success** | `delivered / (delivered + failed_after_3_retries)` | n/a | 99.0% | 99.9% |
| **Notification delivery** | `(telegram_sent + discord_sent) / signals_emitted` | 95% | 99% | 99.5% |
| **Data freshness** | bar_age_seconds P99 < bar_period × 2 | 95% | 99% | 99.9% |

**Error budgets dérivés** :
- 99.5% uptime ⇒ 3.6h de downtime autorisé / mois.
- 99.9% uptime ⇒ 43min / mois.

**Politique** :
- Budget brûlé > 50% sur 7j : freeze deploy non-bug-fix, focus stabilisation.
- Budget brûlé > 100% : post-mortem obligatoire + reco P0 immédiat.

### 2.3 MTTR cibles

| Type incident | MTTR J+30 | MTTR J+90 | MTTR J+180 (enterprise) |
|---|---:|---:|---:|
| Scanner mort (heartbeat lost) | < 30 min | < 15 min | < 5 min (auto-restart) |
| Circuit breaker LLM open sustained | < 60 min | < 30 min | < 15 min |
| Telegram/Discord delivery failure | < 60 min | < 30 min | < 15 min |
| API 5xx spike | < 30 min | < 15 min | < 5 min |
| Cost overrun (LLM > budget) | < 4h | < 1h | < 15 min (auto kill switch) |
| Webhook subscriber down | n/a | < 30 min | < 5 min (auto-disable) |

### 2.4 Coût observability stack

| Phase | Volume | Stack | Coût/mo |
|---|---|---|---:|
| **J0 — Solo testing** (< 100 signaux/j, < 5 MAU) | ~60 MB logs/jour | Grafana Cloud Free + Sentry Free + UptimeRobot Free + instatus Free | **$0** |
| **J+90 — Paid public** (< 10k signaux/j, < 100 MAU) | ~600 MB logs/jour | idem (limites Free pas atteintes) | **$0** |
| **J+180 — Enterprise B2B** (~ 50k signaux/j, > 10 customers) | ~3 GB logs/jour | Grafana Cloud Pro $19/mo + Sentry Team $26/mo + UptimeRobot Pro $7/mo + instatus Starter $20/mo + PagerDuty Pro $21/u/mo | **~$93/mo** |
| **J+365 — Scale** (> 100k signaux/j, > 1k MAU) | ~6 GB logs/jour, > 10k séries | Grafana Cloud Advanced $299/mo OR self-hosted Loki+Tempo (Railway $30/mo storage + S3 $5/mo) + Datadog NOT recommended | **~$150-350/mo** |

---

## 3. Gap analysis

| Pilier | État actuel | Cible J+30 | Cible J+90 | Gap effort |
|---|---|---|---|---:|
| Logs JSON + `extra={}` | JSONFormatter ignore extra (`src/intelligence/main.py:39-52`) | Patch JSONFormatter + 30 callsites scanner avec `extra={signal_id, symbol, ...}` | Promtail → Loki + retention 30j | **10h** |
| `print()` → `logger.*` | 159 occurrences | 25 critiques migrées | 100% modules `src/intelligence/*` + `src/api/*` + `src/security/*` | **6h** P0 + 4h P1 |
| Metrics business émises | 0 callsite émet | 6 métriques branchées (scanner, LLM, circuit, cache, notifier, /health) | 12 métriques complètes + recording rules | **12h** P0 + 8h P1 |
| `/metrics` endpoint | Registry instanciée, payload non-vide stub | Idem + 6 émissions actives | + auth basic (token) pour scrape Prometheus externe | **0h** (déjà OK) |
| `/health/live` + `/health/ready` | Inexistant | Endpoints séparés | + K8s/Railway probe config | **2h** |
| `/health/deep` | Excellent | Idem | + alerting sur `503 /health/deep > 2min` | **0h** (déjà OK) |
| Tracing | 0 span | Spans `contextvars` + `trace_id` propagé scanner+LLM | OTel SDK + OTLP exporter → Tempo | **8h** P1 + 12h P2 |
| Alerting rules | Rules legacy RL drawdown | 6 rules sentinel (heartbeat, circuit, latency, /health, cost, cache) | + 4 rules production-grade + runbooks | **6h** P0 + 8h P1 |
| Alertmanager déployé | Config présente, service non-lancé | Container up sur Railway/Docker, Discord+Telegram webhook | + PagerDuty pour enterprise | **6h** P0 + 6h P2 |
| Runbooks | 0 | 3 runbooks (scanner mort, circuit LLM, /health DEGRADED) | 8 runbooks + escalation policy | **4h** P0 + 6h P1 |
| Sentry DSN | Code OK, DSN non setté | DSN configuré, première erreur reçue | + custom tags `signal_id` via `set_context` | **1h** P0 |
| Dashboards Grafana | 0 dashboard sentinel | 2 dashboards (Business + Tech) provisionnés via `infrastructure/grafana/dashboards/` | + 2 dashboards SLO + Cost | **8h** P1 + 6h P2 |
| Log aggregation | stdout uniquement | Promtail sidecar push Loki | Loki search + saved queries | **4h** P1 |
| SLO formels | 0 | 3 SLO documentés + calcul manuel | Recording rules Prometheus + burn rate alerts | **2h** P1 + 8h P2 |
| Status page | 0 | 0 (deferred) | instatus.com Free, 4 composants, webhook depuis circuit breakers | **6h** P2 |
| PII / secrets in logs | Pas de scrubber sur logs locaux | Scrubber appliqué via `logging.Filter` | Pre-commit hook block secrets en hardcode | **3h** P1 |
| Audit log integrity | Audit ledger hash chain présent (`src/audit/`), `/health/deep` le vérifie | OK | Recording rule `audit_ledger_verify_failures_total` + alerte | **2h** P1 |
| Incident response process | Aucun | Doc 1 page (escalation, comms, post-mortem template) | Drill mensuel | **3h** P1 |
| Anomaly detection (metrics) | 0 | 0 | 1 règle 3σ sur `signals_emitted_total` (besoin baseline 7j) | **6h** P2 |

**Total gap** : ~120h brut, **~92h net** après optim/parallélisme (cf. §10).

---

## 4. Plan d'exécution

### P0 — Bloqueurs go-live (~18h, 1.5 semaines solo)

#### P0-1. Brancher les 6 métriques business critiques sur les callsites (8h)

**Fichiers** :
- `src/intelligence/sentinel_scanner.py` — passer `metrics_registry` en constructor (déjà disponible via `AppState`), émettre dans `_scan_once()` et `_emit_signal()`.
- `src/intelligence/llm_narrative_engine.py` — émettre dans `_call_api()` et `_validate_with_haiku()`.
- `src/intelligence/circuit_breaker.py` — émettre dans `_on_failure()` quand transition vers OPEN.
- `src/delivery/{telegram,discord}_notifier.py` — émettre dans `send_signal()`.
- `src/intelligence/semantic_cache.py` — émettre dans `get()` (hit/miss).
- `src/intelligence/main.py:333-352` — passer `metrics_registry` au scanner constructor.

**Métriques** (cf. spec §3.1 d'eval_16) :

```python
# 1. signals_emitted_total {symbol, direction, tier}
# 2. scan_duration_seconds {symbol, stage}     ← histogram, buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10)
# 3. llm_latency_seconds {model, tier}         ← histogram (déjà pré-enregistré INFRA-1.2)
# 4. llm_cost_usd_total {model, tier}          ← counter, increment cost en USD
# 5. circuit_breaker_state {name}              ← gauge, 0=closed/1=half/2=open
# 6. cache_lookups_total {result}              ← counter, result=hit|miss
```

**Patch snippet pour sentinel_scanner.py** (idiomatique, drop-in) :

```python
# Dans __init__:
self._metrics = kwargs.get("metrics_registry")

# Dans _scan_once(), wrapper chaque étage:
with self._span("data_provider.fetch", symbol=self._symbol) as t:
    df = self._data_provider.get_ohlcv(...)
if self._metrics:
    self._metrics.histogram("scan_duration_seconds", "...",
        buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10)
    ).observe(t.elapsed_s, labels={"symbol": self._symbol, "stage": "fetch"})

# Dans _emit_signal(), à la fin:
if self._metrics:
    self._metrics.counter("signals_emitted_total", "...").inc(
        labels={"symbol": s.symbol, "direction": s.signal_type.value, "tier": s.tier.value}
    )
    self._metrics.histogram("confluence_score", "...",
        buckets=(20, 30, 40, 50, 60, 70, 80, 90, 100)
    ).observe(s.confluence_score, labels={"symbol": s.symbol, "tier": s.tier.value})
```

**Acceptance** :
- `curl /metrics` après 5 min de scan retourne **> 50 lignes Prometheus** (vs 3 stub aujourd'hui).
- Histogramme `sentinel_scan_duration_seconds` avec 8 buckets visibles.
- Compteur `sentinel_circuit_breaker_open_total` incrémente quand on force `breaker._state = OPEN` en test.
- Pas de régression sur les 9 smoke tests `tests/test_smoke_e2e.py`.

**Dépendances** : aucune (registry déjà câblée). Bloqueur amont pour P0-4 (alertes calibrées).

#### P0-2. Patch JSONFormatter pour fusionner `extra={}` (15 min + 4h propagation)

**Fichier** : `src/intelligence/main.py:39-52`.

**Patch exact** :

```python
class JSONFormatter(logging.Formatter):
    """Structured JSON log formatter for production."""

    _STANDARD = {
        "args","asctime","created","exc_info","exc_text","filename","funcName",
        "levelname","levelno","lineno","message","module","msecs","name","pathname",
        "process","processName","relativeCreated","stack_info","thread","threadName",
        "msg","getMessage","taskName",
    }

    def format(self, record: logging.LogRecord) -> str:
        import json
        log_entry = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Pull custom fields passed via extra={}
        for k, v in record.__dict__.items():
            if k not in self._STANDARD and not k.startswith("_"):
                log_entry[k] = v
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry, default=str)
```

**Propagation** (4h supplémentaires) : patch des ~30 `logger.*` de `sentinel_scanner.py` (lignes ciblées par eval_16 §2.3 : 383, 476, 493, 526, 533, 550, 578, 582, 617-626) pour ajouter `extra={"signal_id": ..., "symbol": ..., "stage": ...}`.

**Acceptance** :
- `LOG_FORMAT=json python -m src.intelligence.main` puis grep sur stdout : chaque ligne JSON contient bien `signal_id`/`symbol` quand pertinent.
- Test unitaire `tests/test_logging_json.py` (à créer) : vérifie qu'un `logger.info("msg", extra={"signal_id": "abc"})` produit `{"signal_id": "abc", ...}` dans le JSON output.

**Dépendances** : aucune. Préreq pour intégration Loki (P1) qui parse les champs structurés.

#### P0-3. Migrer les 25 `print()` critiques (4h)

**Cibles** (eval_16 §2.1 + nouveaux fichiers identifiés §1.1.a) :

```
src/security/hmac_manager.py:220-223     → logger.warning + extra={"redacted": True}
src/security/alert_manager.py:850-859    → logger.warning + extra={"alert_id", "severity"}
src/security/dead_man_switch.py:620-622  → logger.critical + extra={"healthy": False}
src/persistence/kill_switch_store.py:128 → logger.warning + extra={"halted_before_restart"}
src/live_trading/mt5_connector.py:287    → logger.info + extra={"ticket"}
src/live_trading/async_order_manager.py:172 → logger.info + extra={"fill_price"}
src/live_trading/alerting.py:378-380     → logger.warning(alert.format_text())
src/intelligence/rag/pipeline.py:*       → 12 print → logger.debug (script indexation)
src/agents/data/fred_provider.py:*       → 8 print → logger.info (data fetch progress)
src/intelligence/prompt_registry.py      → 2 print → logger.debug
```

**Acceptance** :
- `grep -rn "print(" src/ --include="*.py" | grep -v "__main__" | grep -vE "(agent_trainer|state_machine_replay|trade_logger|monitoring|evaluate_agent)\.py"` retourne **0 ligne** (modulo scripts CLI tolérés).
- `hmac_manager` ne logue plus de master_key en clair (`grep "master_key" logs/*.log` retourne vide).

**Dépendances** : P0-2 (JSONFormatter étendu) pour profiter pleinement des `extra={}`.

#### P0-4. Substituer alert-rules.yml par 6 règles sentinel + déployer Alertmanager (4h)

**Fichier** : `infrastructure/alert-rules.yml` (réécrire la moitié sentinel, garder le bloc trading legacy en fin si encore d'actualité).

Six règles minimales (cf. eval_16 §5.2 in extenso, repris ici en version condensée) :

```yaml
groups:
- name: sentinel.critical
  rules:
  - alert: SentinelScannerHeartbeatLost
    expr: rate(sentinel_bars_scanned_total[10m]) == 0
    for: 12m
    labels: {severity: critical, service: sentinel}
    annotations:
      summary: "Scanner heartbeat lost on {{ $labels.symbol }}"
      runbook_url: "https://github.com/.../runbooks/scanner_heartbeat.md"

  - alert: SentinelCircuitBreakerOpenSustained
    expr: sentinel_circuit_breaker_state{name=~"llm_api|telegram|discord"} == 2
    for: 5m
    labels: {severity: warning, service: sentinel}

  - alert: SentinelLLMLatencyP99High
    expr: histogram_quantile(0.99, sum(rate(sentinel_llm_latency_seconds_bucket[5m])) by (le)) > 8
    for: 10m
    labels: {severity: warning, service: sentinel}

  - alert: SentinelHealthDegraded
    expr: up{job="sentinel"} == 1 and sentinel_health_status != 0
    for: 2m
    labels: {severity: critical, service: sentinel}

  - alert: SentinelLLMSpendBudget
    expr: increase(sentinel_llm_cost_usd_total[30d]) > 50
    for: 1h
    labels: {severity: warning, service: sentinel, type: cost}

  - alert: SentinelHealthDeepFailure
    expr: probe_http_status_code{instance=~".*/health/deep"} == 503
    for: 5m
    labels: {severity: critical, service: sentinel}
```

**Alertmanager routing** (nouveau fichier `infrastructure/alertmanager.yml` ou patch existant) :

```yaml
route:
  receiver: discord_default
  group_by: [alertname, service]
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 4h
  routes:
    - matchers: [severity="critical"]
      receiver: discord_critical
      continue: true
    - matchers: [severity="critical", service="sentinel"]
      receiver: telegram_fallback   # double notif sur critique

receivers:
  - name: discord_default
    discord_configs:
      - webhook_url: ${DISCORD_WEBHOOK_OPS}
  - name: discord_critical
    discord_configs:
      - webhook_url: ${DISCORD_WEBHOOK_CRITICAL}
        title: "[CRITICAL] {{ .CommonLabels.alertname }}"
  - name: telegram_fallback
    webhook_configs:
      - url: ${TELEGRAM_WEBHOOK_BRIDGE}
        send_resolved: true

inhibit_rules:
  - source_match: {alertname: SentinelScannerHeartbeatLost}
    target_match_re: {alertname: 'SentinelLLMLatencyP99High|SentinelHealthDegraded'}
    equal: [service]
```

**Déploiement** : ajouter le service `alertmanager` dans `infrastructure/docker-compose.yml` ; sur Railway, créer un service worker dédié (image `prom/alertmanager:latest`).

**Acceptance** :
- `docker-compose up alertmanager` démarre, port 9093 répond.
- Forcer `sentinel_scanner._running = False` 13 min → alerte `SentinelScannerHeartbeatLost` arrive sur Discord canal `#ops`.
- `inhibit` empêche le double-firing latency quand scanner mort.

**Dépendances** : P0-1 (métriques émises) ; sinon les `expr` sont toutes à 0.

#### P0-5. Configurer Sentry DSN en prod (1h)

**Étapes** :
1. Créer projet Sentry Free → récupérer DSN.
2. Ajouter `SENTRY_DSN=https://...` dans Railway/.env variables.
3. Ajouter `RELEASE=$(git rev-parse --short HEAD)` au déploiement (Procfile/CI) pour tagging.
4. Mettre à jour `MEMORY.md` § "Env Vars (Production)" + `.env.example`.
5. Forcer une `logger.exception("test sentry boot")` au démarrage en dev pour vérifier la remontée.

**Acceptance** :
- Premier event arrive dans Sentry UI après `RELEASE=test python -m src.intelligence.main`.
- `_scrub_pii` filtre confirmé : tenter d'envoyer un event avec `ANTHROPIC_API_KEY=sk_xxx` → event droppé (log local `"Sentry event dropped"`).

**Dépendances** : aucune.

#### P0-6. Endpoints `/health/live` + `/health/ready` séparés (2h)

**Fichier** : `src/api/routes/health.py` (étendre).

```python
@router.get("/api/v1/health/live", include_in_schema=False)
async def liveness(request: Request):
    """Liveness probe: did the process answer? Return 200 always (unless process dead)."""
    return {"status": "alive", "uptime_seconds": round(time.time() - _BOOT_TIME, 2)}

@router.get("/api/v1/health/ready", include_in_schema=False)
async def readiness(request: Request):
    """Readiness probe: is the scanner running AND not in cold-start calibration?"""
    app_state = request.app.state.app_state
    scanner = getattr(app_state, "scanner", None)
    if scanner is None:
        return JSONResponse({"status": "not_ready", "reason": "scanner_missing"}, status_code=503)
    stats = scanner.get_stats() if hasattr(scanner, "get_stats") else {}
    if not stats.get("running"):
        return JSONResponse({"status": "not_ready", "reason": "scanner_stopped"}, status_code=503)
    if stats.get("bars_processed", 0) < 1:
        return JSONResponse({"status": "not_ready", "reason": "warming_up"}, status_code=503)
    return {"status": "ready"}
```

**Acceptance** :
- `/health/live` répond 200 pendant `_calibrate_system()` (boot ~10s).
- `/health/ready` répond 503 pendant calibration, 200 après première bar scannée.
- `infrastructure/Dockerfile` updated : `HEALTHCHECK CMD curl -f http://localhost:8000/health/live || exit 1`.
- Railway deploy probe pointe sur `/health/ready` (config update).

**Dépendances** : aucune.

### P1 — Hardening pré-paid (~42h, 3 semaines solo)

#### P1-1. Propagation `extra={signal_id, symbol, trace_id}` sur 100% des call-sites scanner (4h)

Étendre P0-2 à `llm_narrative_engine.py`, `confluence_detector.py`, `semantic_cache.py`, `state_machine.py`, `circuit_breaker.py`. Cf. eval_16 §2.3 table de patchs.

**Acceptance** : `grep "extra=" src/intelligence/ src/api/ src/delivery/ -rn | wc -l` > 60 (vs 0 aujourd'hui hors module orphelin).

#### P1-2. Module `src/intelligence/observability.py` — `span()` contextvars (6h)

Voir eval_16 §4.2 Phase 1 (snippet complet 30 lignes). Décorer `_scan_once()`, `_emit_signal()`, `narrate()`, `cache.get/put`, `notifier.send_signal()`. Logs structurés portent un `trace_id` UUID partagé sur tout un cycle.

**Acceptance** :
- `grep -E '"trace_id": "[a-f0-9]+"' logs/*.json` retourne ~12 lignes par signal (1 par span + scan_cycle parent).
- Test E2E : un seul `trace_id` permet de filtrer tout le cycle d'un signal.

#### P1-3. Dashboard Grafana « Business » (4h)

**Fichier** : `infrastructure/grafana/dashboards/sentinel_business.json` (provisioning auto via `infrastructure/grafana/provisioning/`).

Panneaux :
1. Signals emitted (last 24h, by tier, by symbol) — stat panels.
2. Confluence score distribution — heatmap, P50/P75/P95 lines.
3. Cache hit rate — gauge.
4. LLM spend MTD — stat + sparkline.
5. Notifier success rate (Telegram + Discord) — gauge by channel.
6. Active subscribers (FREE/ANALYST/STRATEGIST/INSTITUTIONAL) — depuis signal_store ou tier_manager.

**Acceptance** : import du JSON dans Grafana Cloud → tous les panels affichent des données après 1h de scan.

#### P1-4. Dashboard Grafana « Technical » (4h)

**Fichier** : `infrastructure/grafana/dashboards/sentinel_tech.json`.

Panneaux :
1. Scan duration P50/P95/P99 by stage — multi-line.
2. LLM latency P50/P95/P99 by model — multi-line.
3. Circuit breaker state — state timeline.
4. Bars scanned rate by symbol — heartbeat.
5. Errors rate (by logger.name) — count panel.
6. /health status — state timeline.

**Acceptance** : idem P1-3.

#### P1-5. Promtail sidecar → Loki Grafana Cloud (4h)

**Fichier** : `infrastructure/promtail-config.yml` (nouveau).

```yaml
clients:
  - url: https://logs-prod-XX.grafana.net/loki/api/v1/push
    basic_auth: {username: ${LOKI_USER}, password: ${LOKI_PASS}}
scrape_configs:
  - job_name: sentinel
    static_configs:
      - targets: [localhost]
        labels: {job: sentinel, service: sentinel-api}
    pipeline_stages:
      - json: {expressions: {level: level, signal_id: signal_id, trace_id: trace_id, symbol: symbol}}
      - labels: {level, symbol}
```

**Déploiement** : container sidecar dans `docker-compose.yml`, monte le stdout du sentinel (via `journald` ou fichier rotaté).

**Acceptance** : recherche `{job="sentinel"} |= "circuit"` dans Grafana Loki retourne les events.

#### P1-6. Sentry tags + breadcrumbs custom (3h)

Wrapper `logger.exception()` avec `sentry_sdk.set_context("signal", {"signal_id": ..., "symbol": ...})` pour enrichir les events.

#### P1-7. PII scrubber `logging.Filter` (3h)

**Fichier** : `src/performance/logging_config.py` (étendre l'existant) ou nouveau `src/observability/pii_filter.py`.

```python
class PIIScrubFilter(logging.Filter):
    """Scrub Telegram chat IDs, API keys, email addresses from log records."""
    PATTERNS = [
        (re.compile(r'(api[_-]?key["\s:=]+)([a-zA-Z0-9_-]{16,})'), r'\1***REDACTED***'),
        (re.compile(r'\b\d{10,12}\b'), '***CHAT_ID***'),
        (re.compile(r'[\w._%+-]+@[\w.-]+\.[A-Za-z]{2,}'), '***EMAIL***'),
    ]
    def filter(self, record: logging.LogRecord) -> bool:
        msg = str(record.getMessage())
        for pat, repl in self.PATTERNS:
            msg = pat.sub(repl, msg)
        record.msg = msg
        record.args = ()
        return True
```

**Câblage** : ajouter `handler.addFilter(PIIScrubFilter())` dans `setup_logging()`.

**Acceptance** : test unitaire `tests/test_pii_scrubber.py` vérifie qu'un `logger.info("key=sk_abc123def456...")` produit `key=sk_abc1***REDACTED***`.

#### P1-8. 4 alertes Prometheus supplémentaires (3h)

Cf. eval_16 §5.2 :
- `SentinelSignalRateAnomaly` (3σ baseline)
- `SentinelSemanticCacheCold` (< 5% hit rate)
- `SentinelNotifierFailureRate` (> 20% fail)
- `SentinelDataProviderErrors` (rate > 0.1/min)

#### P1-9. 5 runbooks ops (4h)

**Répertoire** : `docs/runbooks/`.
1. `scanner_heartbeat.md` — diagnostic + fix scanner mort.
2. `circuit_llm_open.md` — Anthropic status check, fallback template, escalation.
3. `health_degraded.md` — composant par composant, recovery actions.
4. `llm_cost_overrun.md` — kill switch, basculer template, post-mortem.
5. `webhook_dead_letters.md` — purge, retry, désactivation subscriber.

Chaque runbook : symptôme, diagnostic (3 commandes max), fix (3 actions max), escalation, post-mortem template.

#### P1-10. SLO documentés + recording rules (3h)

**Fichier** : `infrastructure/recording-rules.yml`.

```yaml
groups:
- name: sentinel.slo
  interval: 30s
  rules:
  - record: sentinel:scanner_uptime_ratio:7d
    expr: avg_over_time((rate(sentinel_bars_scanned_total[10m]) > bool 0)[7d:10m])
  - record: sentinel:llm_latency_p95:5m
    expr: histogram_quantile(0.95, sum(rate(sentinel_llm_latency_seconds_bucket[5m])) by (le))
  - record: sentinel:signals_per_day:1d
    expr: increase(sentinel_signals_emitted_total[1d])
```

+ doc `docs/slo.md` avec les 6 SLO de §2.2 et les error budgets dérivés.

#### P1-11. Incident response playbook (3h)

**Fichier** : `docs/incident_response.md`.

Sections : severity matrix (SEV1/2/3), escalation chain (solo founder J0 → +Telegram backup contact J+30), comms template (status page update, customer email), post-mortem template (5 whys, action items, owners).

#### P1-12. Log retention policy (1h)

Configurer Loki Grafana Cloud → 30 jours.
Documenter dans `docs/log_retention.md` : 30j hot, archivage S3 90j pour audit compliance (cf. cat. 18).

### P2 — Production-grade enterprise (~32h, 2-3 semaines)

#### P2-1. OpenTelemetry SDK + OTLP exporter (12h)

Migrer `src/intelligence/observability.py` Phase 1 (contextvars maison) vers Phase 2 (OTel SDK officiel).

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

trace.set_tracer_provider(TracerProvider(resource=Resource.create({
    "service.name": "sentinel-api",
    "service.version": os.environ.get("RELEASE", "dev"),
})))
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter(endpoint=os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"]))
)
FastAPIInstrumentor.instrument_app(api_app)
```

**Cibles d'instrumentation** : pipeline 7 étages = 7 spans + auto-instru FastAPI = ~10 spans/request.

**Acceptance** : 1 trace par signal arrive dans Grafana Cloud Tempo, exploitable via TraceQL.

#### P2-2. Dashboard Grafana « SLO » (6h)

Panneaux burn rate (fast 1h vs slow 6h), error budget remaining %, MTTR par incident type sur 30j.

#### P2-3. Status page instatus.com (6h)

Composants : Signal Engine, Narrative Engine, REST API, Telegram/Discord, Data Provider. Webhook ingest depuis Alertmanager → bascule auto operational/degraded.

#### P2-4. PagerDuty integration (J+180, 4h, conditionnel B2B enterprise)

Routing : `severity=critical` → PagerDuty service Sentinel → escalation policy (Loukmane → backup contact 15min → all hands 30min). À activer uniquement quand 1er contrat enterprise signé.

#### P2-5. Anomaly detection metrics (6h, conditionnel ≥7 jours de baseline)

3σ rolling sur `signals_emitted_total`, `llm_cost_usd_total`, `cache_hit_rate`. Implé via recording rules + Alertmanager.

#### P2-6. Audit log integrity alerting (2h)

Recording rule `audit_ledger_verify_failures_total` + alerte sur `> 0`. Integrity = compliance-bloquant pour cat. 18.

#### P2-7. Cost dashboard + monthly recap email (4h)

Job hebdo qui agrège `llm_cost_usd_total` par tier × model, push Discord canal `#cost` ; mensuel push email founder.

---

## 5. Tests & validation

### 5.1 Tests unitaires nouveaux

| Fichier | Cible test | Effort |
|---|---|---:|
| `tests/test_logging_json.py` | JSONFormatter sérialise `extra={}`, fields standard exclus, exception bien formaté | 1h |
| `tests/test_pii_scrubber.py` | PIIScrubFilter masque api_key, chat_id, email | 1h |
| `tests/test_metrics_emission.py` | Wrapper qui scan une bar fake → vérifie counter `signals_emitted_total` incrémenté | 2h |
| `tests/test_circuit_breaker_metrics.py` | Forcer OPEN → counter `circuit_breaker_open_total` +1 | 1h |
| `tests/test_health_live_ready.py` | `/health/live` 200 toujours, `/health/ready` 503 pendant warmup | 1h |
| `tests/test_alert_rules_promql.py` | Lint syntaxique des règles via `promtool check rules infrastructure/alert-rules.yml` | 0.5h |
| `tests/test_observability_init.py` | `init_observability()` sans `SENTRY_DSN` n'échoue pas, registry retourne metrics | 1h |
| `tests/test_tracing_span.py` | `with span("x"): raise` → log error avec `trace_id` correct | 1h |

**Total** : ~8.5h de tests, ajout au CI.

### 5.2 Tests d'intégration

- **Smoke E2E** : étendre `tests/test_smoke_e2e.py` avec un test `test_metrics_endpoint_non_empty` qui démarre le scanner, attend 1 bar, `assert "sentinel_bars_scanned_total" in /metrics body`.
- **Chaos test manuel** : forcer scanner kill (`kill -9`) → vérifier alerte `ScannerHeartbeatLost` arrive Discord en < 13 min.
- **Load test** : 100 req/s sur `/health/live` → P95 < 50ms (sans cache invalidation `/health/deep`).

### 5.3 Validation manuelle pré-prod

Checklist déploiement (à intégrer au release process cat. 19) :
- [ ] `curl /metrics | wc -l` > 50 lignes
- [ ] `curl /health/deep` retourne 200 et tous les `ok=true`
- [ ] Sentry reçoit un event de boot (volontaire, supprimer après check)
- [ ] Alertmanager UI accessible, status `Active`
- [ ] Grafana dashboards Business + Tech affichent données
- [ ] 1 cycle scan complet → grep logs JSON contient `signal_id` cohérent sur ~12 lignes

---

## 6. Sécurité (PII / secrets / audit log integrity)

### 6.1 PII / secrets exclus des logs

| Type donnée | Source de risque | Mitigation P0/P1 |
|---|---|---|
| `ANTHROPIC_API_KEY`, `FRED_API_KEY`, `TELEGRAM_BOT_TOKEN` | Stack traces, debug logs, exceptions HTTP avec headers | `_scrub_pii` Sentry (P0, déjà OK `src/performance/observability.py:154-173`) + `PIIScrubFilter` sur stdout (P1-7) |
| `telegram_chat_id` (= identifiant utilisateur partiel) | `logger.info("Sent to chat %s", chat_id)` | PIIScrubFilter masque les nombres 10-12 chiffres |
| Email addresses | Onboarding webhooks, error context | PIIScrubFilter regex email |
| HMAC secrets (master_key) | `src/security/hmac_manager.py:220-223` print direct stdout | P0-3 migration → `logger.warning("...", extra={"redacted": True})` (le secret n'est PAS loggué, juste un flag) |
| OHLCV data | Volume large, pas PII mais bruit | Log uniquement les agrégats (`shape`, `last_bar_ts`), jamais le DataFrame complet |
| Stripe customer IDs | Webhook payments | À traiter dans cat. 10 (auth) + cat. 18 (compliance), pas cat. 12 |

### 6.2 Audit log integrity

- `src/audit/` (présomption — à vérifier au build) implémente une hash-chain pour les events audit.
- `/health/deep` vérifie déjà l'intégrité (`src/api/routes/health_deep.py:48-71`).
- **P1-Audit** : recording rule Prometheus `audit_ledger_verify_failures_total` + alerte sur `> 0` (cf. P2-6). Compliance-bloquant pour cat. 18 (RGPD trail).

### 6.3 Logs immuables (write-only, append-only)

Hors scope cat. 12 (relève cat. 10/18). Le `signal_store.publish()` SQLite WAL est append-only de facto ; si compliance B2B requiert WORM (Write Once Read Many), envisager S3 Object Lock ou Loki bucket lock — décision à J+180.

### 6.4 Accès `/metrics` et `/health/deep`

`/metrics` actuellement publique. **Risque** : exposition cardinalité, latence interne, signaux de fingerprint algo.
- P1 : ajouter middleware `require_metrics_token` (header `X-Metrics-Token` vérifié contre env var `METRICS_SCRAPE_TOKEN`). Prometheus scrape job ajoute le header.
- P1 : idem `/health/deep` (peut leaker structure de subsystems).

---

## 7. Métriques (SLOs)

### 7.1 SLI/SLO récapitulatif (cf. §2.2)

| SLI | Mesure | SLO 90j |
|---|---|---:|
| Scanner uptime | `sentinel:scanner_uptime_ratio:7d` | 99.5% |
| API availability | `sum(rate(http_requests_total{status=~"2..|3.."}[5m])) / sum(rate(http_requests_total[5m]))` | 99.5% |
| Signal latency P95 | `sentinel:scan_duration_seconds:p95` | < 3s |
| LLM latency P95 | `sentinel:llm_latency_p95:5m` | < 4s |
| Webhook delivery success | `webhook_delivered_total / (webhook_delivered_total + webhook_failed_total)` | 99% |

### 7.2 KPI observability (méta-KPI)

| KPI | Cible J+30 | Cible J+90 |
|---|---|---|
| % signaux avec `signal_id` corrélable end-to-end dans logs | 100% | 100% (avec trace OTel) |
| Faux positifs / semaine (alertes acquittées sans action) | < 3 | < 1 |
| Couverture métriques modules critiques | 6/6 (scanner, LLM, cache, notifier, circuit, /health) | 12/12 |
| % `print()` vs `logger.*` dans `src/intelligence/* + src/api/* + src/security/*` | 0% print | 0% print |
| MTTR alert → resolve (auto via Alertmanager) | < 30 min | < 15 min |
| Coût mensuel obs stack | $0 | $0 (Free) ou $93 (Enterprise) |

---

## 8. Risques & mitigations

| Risque | Probabilité | Impact | Mitigation |
|---|---|---|---|
| `MetricsRegistry` custom diverge du standard `prometheus_client` (label ordering, encoding) | Faible | Moyen | Test `promtool check metrics` dans CI ; migrer vers `prometheus_client` officiel **uniquement** si scaling > 1k req/min (eval_16 §8). |
| Cardinality explosion (labels symbol×tier×direction×model) > 10k séries | Moyenne (à 6+ symbols × 4 tiers × 2 dir × 4 models = ~200 séries par metric × 12 = 2400) | Moyen-Haut | Budget cardinalité documenté §3.1 d'eval_16 (~1170 attendu) ; recording rules pour aggregations ; alarme sur `prometheus_tsdb_head_series` > 8000. |
| Logs JSON cassent les outils CLI dev habitués au format text | Faible | Bas | Garder `LOG_FORMAT=text` par défaut en dev (déjà OK `main.py:57`). JSON uniquement en prod via env var. |
| Sentry over-quota free tier 5k events/mois (rate-limit) | Faible (estim < 200/mo solo testing) | Bas | Sample rate 5% des traces déjà ; `before_send` filter peut drop des classes d'errors connues bruyantes. |
| `_scrub_pii` masque trop agressif → masque des champs utiles debug | Moyenne | Bas-Moyen | Regex permissives au début, durcir progressivement avec tests. |
| Promtail/Loki vendor lock-in si on bascule self-host plus tard | Faible | Bas | Format Loki JSON labels = standard ; migration vers ELK/CloudWatch faisable. |
| Alertmanager Discord webhook expires / rate-limit | Faible | Moyen | Fallback Telegram bridge (P0-4 routing). |
| OpenTelemetry SDK ajoute >5% overhead CPU | Faible-Moyenne | Bas-Moyen | Sample rate 5% pour traces ; benchmarks avant rollout (cf. cat. 17). |
| Surface d'attaque agrandie (/metrics public scrape) | Moyenne | Moyen | P1 : authentification scrape token. |
| Dépendance Grafana Cloud / Sentry SaaS (vendor risk) | Moyenne | Bas-Moyen | Free tier = budget = 0 (pas de lock contractuel) ; export config dashboards JSON versionné en repo. |

---

## 9. Dépendances (vers autres catégories)

| Catégorie | Lien | Type dépendance |
|---|---|---|
| **8. API Backend** | `src/api/dependencies.py` AppState, `src/api/app.py` middleware request_logging | Cat. 12 consomme l'AppState pour `metrics_registry`. Cat. 8 doit exposer `request_id` middleware (P1-2 trace). |
| **10. Auth/Security** | `src/security/hmac_manager.py`, `src/security/alert_manager.py`, `src/security/dead_man_switch.py` | Cat. 12 migre les `print()` mais ne change pas la logique sécurité. Coordination requise sur le scrubber. |
| **11. Delivery channels** | `src/delivery/{telegram,discord}_notifier.py` | Cat. 12 ajoute métriques `notifier_send_total`. Cat. 11 doit accepter le wrapping. |
| **13. Testing infrastructure** | CI GitHub Actions | Cat. 12 ajoute tests `tests/test_logging_*` `tests/test_metrics_*` ; cat. 13 doit les inclure au pipeline. |
| **17. Caching/Perf** | `semantic_cache.get()` métriques hit/miss | Cat. 12 instrumente, cat. 17 ne doit pas casser l'API. |
| **18. Compliance** | RGPD log retention 90j max, audit immuable | Cat. 12 livre le mécanisme (retention, PII scrub) ; cat. 18 fournit le cadre légal. |
| **19. MLOps/Deployment** | Procfile, `infrastructure/Dockerfile`, Railway env vars | Cat. 12 ajoute `SENTRY_DSN`, `OTEL_EXPORTER_OTLP_ENDPOINT`, `LOKI_USER/PASS`, `METRICS_SCRAPE_TOKEN`, `DISCORD_WEBHOOK_OPS/CRITICAL`. Cat. 19 doit propager. |

**Path critique** : cat. 12 P0 = pré-req pour cat. 8 production hardening (sans metrics, pas d'alertes API 5xx). Cat. 12 P1 = pré-req pour cat. 11 SLA delivery (sans `notifier_send_total`, pas de SLO webhook). Cat. 12 P2 = pré-req pour cat. 19 enterprise blue/green deployment.

---

## 10. Estimation totale & timeline + coût observability stack/mo

### 10.1 Effort total

| Phase | Tâches | Effort net | Calendrier solo (8-10h/sem) |
|---|---|---:|---|
| **P0 — Bloqueurs go-live** | P0-1 à P0-6 | **18h** | Semaine 1-2 |
| **P1 — Hardening pré-paid** | P1-1 à P1-12 | **42h** | Semaine 3-7 |
| **P2 — Enterprise B2B** | P2-1 à P2-7 | **32h** | Semaine 8-11 (conditionnel B2B) |
| **Total Cat. 12** | — | **92h** | **11 semaines** |

### 10.2 Timeline détaillée (Gantt simplifié)

```
S1  ────[P0-1 metrics emission]────[P0-2 JSON extra]──
S2  ──[P0-3 print migration]──[P0-4 alerts+AM]──[P0-5 sentry]──[P0-6 live/ready]──
S3  ────[P1-1 extra propagation]──[P1-2 contextvars span]────
S4  ──[P1-3 dashboard business]──[P1-4 dashboard tech]──
S5  ────[P1-5 promtail/loki]──[P1-6 sentry tags]──[P1-7 PII filter]──
S6  ────[P1-8 4 alerts]──[P1-9 runbooks]──[P1-10 SLO]──
S7  ────[P1-11 incident playbook]──[P1-12 log retention]──[tests CI]──
            🟢 Go-live MVP commercial B2C (J+45)
S8  ────[P2-1 OpenTelemetry SDK]────
S9  ────[P2-2 dashboard SLO]──[P2-3 status page]──
S10 ────[P2-5 anomaly detection]──[P2-6 audit alerting]──[P2-7 cost dashboard]──
S11 ────[P2-4 PagerDuty — conditionnel 1er deal B2B]────
            🟢 Go-live enterprise SLA (J+90)
```

### 10.3 Coût observability stack/mo (récap)

| Phase | Stack | Coût/mo |
|---|---|---:|
| **J0 — Solo** | Grafana Cloud Free + Sentry Free + UptimeRobot Free + instatus Free | **$0** |
| **J+45 — Paid public B2C** | Idem (limites Free non atteintes < 10k signaux/j) | **$0** |
| **J+90 — Enterprise B2B SLA** | Grafana Cloud Pro ($19) + Sentry Team ($26) + UptimeRobot Pro ($7) + instatus Starter ($20) + PagerDuty ($21 × 1 user) | **~$93/mo** |
| **J+365 — Scale > 100k sig/j** | Grafana Cloud Advanced ($299) OU self-hosted Loki+Tempo S3 ($35) + Sentry Business + DataDog NON | **~$150-350/mo** |

### 10.4 Coût opérationnel (heures/mois en run)

| Phase | Maintenance obs (heures/mo) |
|---|---:|
| J0 | 2h/mo (check dashboards weekly) |
| J+45 | 4h/mo (+ runbook updates) |
| J+90 | 8h/mo (+ alerting tuning + post-mortems) |
| J+365 | 16h/mo (+ self-host ops) |

---

## 11. Annexes

- `reports/eval_16_observability.md` — audit complet ayant servi de base (note 3.2/10).
- `reports/eval_16_metrics_catalog.md` — catalogue 12 métriques détaillé (référence absolue, cardinalité estimée).
- `infrastructure/alert-rules.yml`, `infrastructure/alertmanager.yml`, `infrastructure/prometheus.yml` — configs à patcher P0-4.
- `infrastructure/grafana/dashboards/` — dossier où provisionner les 4 dashboards P1.
- `src/intelligence/main.py:39-52` — JSONFormatter à patcher P0-2.
- `src/intelligence/main.py:276-281` — `init_observability()` déjà câblé (Sprint INFRA-1.2).
- `src/performance/metrics.py:1-469` — `MetricsRegistry` custom.
- `src/performance/observability.py:1-206` — bootstrap Sentry + standard metrics.
- `src/performance/logging_config.py:1-100+` — module orphelin contextvars (décider : merger dans `main.py` ou garder séparé).
- `src/api/routes/health.py`, `src/api/routes/health_deep.py`, `src/api/routes/prometheus.py` — endpoints obs.
- `src/intelligence/circuit_breaker.py` — à instrumenter P0-1.
- `src/intelligence/sentinel_scanner.py` — à instrumenter P0-1 + P1-1 + P1-2.

---

# Synthèse finale (5 lignes)

- **Chemin** : `C:\MyPythonProjects\TradingBOT_Agentic\reports\commercialization_sprint\12_observability.md`
- **Top 3 P0** : (1) Émettre 6 métriques business depuis scanner+LLM+circuit+notifier+cache (`src/intelligence/sentinel_scanner.py`, `src/intelligence/llm_narrative_engine.py`) ; (2) Patcher `JSONFormatter` pour fusionner `extra={}` (`src/intelligence/main.py:39-52`) + propager `signal_id`/`symbol` sur 30 callsites scanner ; (3) Migrer 25 `print()` critiques (`hmac_manager.py`, `alert_manager.py`, `dead_man_switch.py`, `kill_switch_store.py`, `mt5_connector.py`, `async_order_manager.py`, `rag/pipeline.py`) + déployer Alertmanager avec 6 règles sentinel + routing Discord/Telegram.
- **Heures** : P0=18h / P1=42h / P2=32h → **total 92h sur 11 semaines** solo (8-10h/sem).
- **Coût obs stack** : **$0/mo** solo + B2C public (Grafana Cloud Free + Sentry Free + UptimeRobot Free + instatus Free) → **~$93/mo** dès le 1er contrat enterprise B2B (Grafana Pro + Sentry Team + PagerDuty).
- **Dépendances bloquantes** : cat. 8 (AppState exposition `metrics_registry`), cat. 19 (env vars `SENTRY_DSN` + `OTEL_*` propagés Railway), cat. 18 (PII scrubber + log retention 30j conformes RGPD).
