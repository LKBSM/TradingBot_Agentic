# Eval 16 — Observability & Logging

> **Périmètre audité** : `src/intelligence/main.py` (logger config + JSONFormatter, 526 l.), `src/api/app.py` (3 middlewares, 175 l.), `src/api/routes/health.py` (111 l.), `src/api/routes/prometheus.py` (21 l.), `src/intelligence/circuit_breaker.py` (231 l.), `src/performance/metrics.py` (469 l. — `MetricsRegistry` maison non câblée), `src/intelligence/sentinel_scanner.py` (~960 l.), `src/intelligence/llm_narrative_engine.py`, `src/intelligence/semantic_cache.py`, `src/delivery/{telegram,discord}_notifier.py`.
>
> **Date** : 2026-04-25 · **Branch** : `main` · **Snapshot** : 632e9dd + uncommitted Sprint 2.

---

## 0. Executive Summary (5 lignes)

1. **`/metrics` retourne payload vide en prod** : `MetricsRegistry` (469 l. de code custom dans `src/performance/metrics.py`) n'est **jamais instanciée** dans `src/intelligence/main.py` ni passée à `create_app()`. Le finding bloquant n°1.
2. **109 occurrences de `print()` dans 23 fichiers `src/`** dont 8 dans `src/security/alert_manager.py:850-859` (chemin alertes critiques) — bypass total du JSONFormatter.
3. **Aucun contexte structuré** dans les logs : sur ~30 `logger.*` de `sentinel_scanner.py`, aucun n'utilise `extra={'signal_id':…}` malgré 13 mentions de `signal.signal_id` interpolées en chaîne. Logs non-corrélables.
4. **Tracing OpenTelemetry inexistant** ; le pipeline 7-étages n'a aucun `trace_id` qui survit DataProvider → Telegram. MTTR sur "signal manqué" est illimité.
5. **Note globale : 3.2 / 10**. Stack côté ingénierie de base correct (JSONFormatter, circuit breakers, /health riche), mais le tissu connectif (metrics actives, traces, contexte structuré, alerting) est absent. Plan J0/J+30/J+90 livré ci-dessous.

---

## 1. O1 — Inventaire as-is (Obs Lead)

### 1.1 Configuration logging

| Élément | État | Source |
|---|---|---|
| `setup_logging()` centralisé | Oui | `src/intelligence/main.py:55-73` |
| `JSONFormatter` | Présent, activé par `LOG_FORMAT=json` | `main.py:39-52` |
| Payload JSON | `ts, level, logger, msg, exception` | `main.py:44-52` |
| Champs absents | `signal_id, symbol, trace_id, request_id, tenant_id` | — |
| Filtrage bibliothèques bruyantes | `httpx, httpcore, hmmlearn` → WARNING | `main.py:71-73` |
| Niveau par défaut | `INFO` (env `LOG_LEVEL`) | `main.py:360` |
| Propagation | Standard `logging.getLogger(__name__)` partout (103 fichiers — voir §2) | — |

### 1.2 Endpoints d'observabilité

| Endpoint | Source | État | Trou |
|---|---|---|---|
| `GET /api/v1/health` | `routes/health.py:101-104` | Riche : status, components, scanner_running, signals_generated, cache_hits, kill_switch_level | OK |
| `GET /health` (Docker) | `routes/health.py:107-110` | Alias même payload | OK |
| `GET /metrics` (Prometheus) | `routes/prometheus.py:11-20` | **Payload vide en prod** (registry jamais instanciée — voir §1.4) | BLOQUANT |

### 1.3 Health checking & circuit breakers

| Composant | Source | Exposition obs |
|---|---|---|
| `HealthChecker.check_all()` | `circuit_breaker.py:199-231` | Aggrégé dans `/health` (`routes/health.py:46-53`) |
| `CircuitBreaker.get_stats()` | `circuit_breaker.py:163-179` | Retourne `name, state, consecutive_failures, total_calls, total_failures, total_successes, failure_rate, last_failure_age` mais **non exposé via `/metrics`** |
| LLM breaker | `main.py:188-195` (threshold=3, recovery=60s) | État disponible mais pas en gauge |
| Notifier breaker | `main.py:196-201` (threshold=5, recovery=120s) | Idem |

### 1.4 `MetricsRegistry` : code mort

`src/performance/metrics.py` implémente une registry maison complète (`Counter`, `Gauge`, `Histogram` avec format Prometheus text export, 469 l.) **mais** :

```bash
$ rg "MetricsRegistry|metrics_registry=" src/intelligence/main.py
# 0 résultat
```

`main.py` n'instancie **jamais** `MetricsRegistry` et `create_app()` reçoit `metrics_registry=None` (`api/app.py:25`). Donc :
- Le middleware `request_logging` (`app.py:128-150`) a un `try: registry.histogram(…).observe(…)` qui **est toujours skippé** (registry None).
- `routes/prometheus.py:14-19` retourne `PlainTextResponse("")` (registry None → empty body).

**→ Tout endpoint `/metrics` scrappé par Prometheus retourne vide. Aucune métrique business aujourd'hui. Action P0.**

### 1.5 Spec to-be (cible)

| Pilier | Cible J+30 | Cible J+90 |
|---|---|---|
| **Logs** | JSON + `extra={signal_id, symbol, trace_id}` partout, 0 `print()` dans `src/intelligence/*` & `src/api/*` | Stream vers Loki/Grafana Cloud Free, 7 j retention |
| **Metrics** | `MetricsRegistry` instanciée, 12 métriques business actives, scrape OK | Histogram cardinality < 50k, recording rules pour PF/win-rate |
| **Tracing** | `signal_id` propagé via `contextvars`, dump JSON par signal | OTLP exporter → Tempo/Jaeger, 7 spans par signal |
| **Alerting** | 6 règles Alertmanager basiques (circuit, /health, signal-rate) | 8 règles avec runbooks, OnCall Discord/Telegram |
| **Status page** | — | Page publique (instatus) avec 4 composants |

---

## 2. O2 — Log Auditor

### 2.1 `print()` — 109 occurrences dans 23 fichiers `src/`

Tableau par fichier (output `Grep print\(` count mode) :

| Fichier | Occurrences | Évaluation |
|---|---:|---|
| `src/agent_trainer.py` | 13 | CLI demo block (`__main__`) — **OK à laisser** |
| `src/security/alert_manager.py:850-859` | 8 | **À MIGRER vers `logger.warning(extra=...)`** — chemin alertes critiques |
| `src/agents/monitoring.py` | 9 | `console.print(rich)` dans CLI — borderline OK (UX humain) |
| `src/security/hmac_manager.py:220-223` | 4 | Affiche **secrets en clair** (`master_key`) au stdout — **CRITIQUE** |
| `src/live_trading/alerting.py:378-380` | 3 | `print(alert.format_text())` — bypass logging |
| `src/security/dead_man_switch.py:620-622` | 2 | "Bot is healthy / DEAD" — **migrer obligatoirement** |
| `src/utils/latency_tracker.py:124,383` | 2 | `print(f"{op}: p99=...")` dans `__repr__-like` — borderline |
| `src/persistence/kill_switch_store.py:128` | 1 | "WARNING: Bot was halted before restart!" — **migrer** |
| `src/live_trading/mt5_connector.py:287` | 1 | "Position opened: ticket=…" — **migrer** |
| `src/live_trading/async_order_manager.py:172` | 1 | "Order filled: …" — **migrer** |
| `src/training/curriculum_trainer.py:103` | 1 | callback eval — borderline |
| `src/backtest/state_machine_replay.py:371` | 1 | `print(results.pretty())` dans CLI — OK |
| `src/interfaces/trade_logger.py:119` | 1 | `print(f"Win rate: …")` — OK CLI summary |
| Autres (`weekly_adaptation.py`, `evaluate_agent.py`, …) | 48+8 | `console.print(rich)` patterns — UX CLI, OK |
| **Total** | **109** | dont ~25 critiques à migrer |

**Top-3 priorités migration** :
1. `src/security/hmac_manager.py:220-223` — secrets en clair, **fuite GDPR potentielle** si stdout pipé.
2. `src/security/alert_manager.py:850-859` — alertes système qui n'atterrissent ni dans Loki ni dans Sentry.
3. `src/live_trading/{mt5_connector.py:287,async_order_manager.py:172}` — événements trading bypass logging.

### 2.2 Pas de `traceback.print_exc()` ni `pprint()` en prod

```
$ Grep "traceback\.print|console\.log|pprint\(" src/
No matches found
```

### 2.3 Contexte structuré (`extra={...}`)

Sur tout le repo `src/intelligence/` :

```bash
$ rg "extra=\{" src/intelligence/
# 0 résultat sur sentinel_scanner.py, llm_narrative_engine.py, semantic_cache.py, etc.
```

**Aucun logger n'utilise `extra={...}`**. Conséquence : le `JSONFormatter` (`main.py:42-52`) ne transmet **que** `ts, level, logger, msg, exception`. Pas de `signal_id`, `symbol`, `tier`, `trace_id`. Logs non-corrélables → MTTR ×3.

**Call-sites à patcher en priorité (sentinel_scanner.py)** :

| Ligne | Log actuel | Patch suggéré |
|---|---|---|
| `383` | `logger.debug("Cache hit for signal %s", signal.signal_id)` | `logger.debug("Cache hit", extra={"signal_id": signal.signal_id, "symbol": signal.symbol})` |
| `476` | `logger.error("Exit outcome update failed for %s: %s", signal_id, e)` | `logger.error("Exit outcome update failed: %s", e, extra={"signal_id": signal_id})` |
| `493` | `logger.warning("Exit notification failed for %s: %s", signal_id, e)` | idem `extra={"signal_id": signal_id, "stage": "notify_exit"}` |
| `526` | `logger.warning("LLM circuit OPEN for signal %s — falling back to TemplateNarrativeEngine", signal.signal_id)` | `extra={"signal_id": signal.signal_id, "stage": "llm", "fallback": "template", "circuit": "open"}` |
| `533, 550, 578, 582` | idem | idem |
| `617-626` | "Published %s signal: %s score=%.1f tier=%s …" | `extra={"signal_id": ..., "symbol": ..., "score": ..., "tier": ..., "direction": ..., "rr": ..., "entry": ..., "sl": ..., "tp": ...}` |

**Modifier `JSONFormatter` pour fusionner `record.__dict__` standard fields** :

```python
# src/intelligence/main.py:39-52 — patch
class JSONFormatter(logging.Formatter):
    _STANDARD = {
        "args","asctime","created","exc_info","exc_text","filename","funcName",
        "levelname","levelno","lineno","message","module","msecs","name","pathname",
        "process","processName","relativeCreated","stack_info","thread","threadName",
        "msg","getMessage",
    }
    def format(self, record):
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

### 2.4 Spam potentiel (logs dans boucles)

Lecture de `sentinel_scanner.py:211-220` :

```python
def _run_loop(self):
    while self._running:
        try:
            self._scan_once()
        except Exception as e:
            self._errors += 1
            logger.error("Scanner error: %s", e, exc_info=True)
        if self._running:
            time.sleep(self._poll_interval)
```

Avec `poll_interval=60s` et un crash répétitif : 1440 stack traces / jour. À 6 symboles : **8640 lignes ERROR/jour pour la même cause**.

→ **Ajouter un déduplicateur** : si la même classe d'exception se produit > 5 fois en 5 min, downgrade en `WARNING` + `extra={"deduped_count": N}` (pattern `RateLimitedLogger`).

### 2.5 Secrets dans les logs ?

Audit rapide :
- `src/security/hmac_manager.py:220-223` — **secrets imprimés en clair via `print()`** (master_key). Pas dans `logger`, mais bypass JSONFormatter.
- `src/intelligence/llm_narrative_engine.py:319` — `logger.info("Anthropic client initialized (caching=%s, timeout=%.1fs)", …)` — pas de fuite (clef pas loggée).
- `src/api/auth.py` — pas grep'é exhaustivement, mais `MEMORY.md` confirme `Error detail leakage fixed (str(exc) → "Internal server error")`.
- `app.py:155` — `logger.exception("Unhandled error on %s %s", …)` — peut contenir traces avec valeurs sensibles. **Recommandation** : ajouter un scrubber `redact_pii()` hook.

---

## 3. O3 — Metrics Designer (catalogue)

Voir le catalogue détaillé standalone : **`reports/eval_16_metrics_catalog.md`**.

### 3.1 Synthèse — 12 métriques cibles

| # | Nom | Type | Labels | Cardinalité | Raison business |
|---|---|---|---|---:|---|
| 1 | `sentinel_signals_emitted_total` | Counter | symbol, direction, tier | 6×2×4 = **48** | Volume signaux — KPI commercial |
| 2 | `sentinel_signals_held_total` | Counter | symbol, reason | 6×8 = **48** | Pourquoi signaux supprimés (cooldown, hysteresis…) |
| 3 | `sentinel_confluence_score` | Histogram | symbol, tier | 6×4×12 buckets = **288** | Distribution score — détecter dérive scoring |
| 4 | `sentinel_bars_scanned_total` | Counter | symbol | **6** | Heartbeat — alarme si plat |
| 5 | `sentinel_scan_duration_seconds` | Histogram | symbol, stage | 6×7×12 = **504** | Latence par étage pipeline |
| 6 | `sentinel_circuit_breaker_state` | Gauge | name (provider) | **3** (llm, telegram, discord) | 0=closed, 1=half_open, 2=open |
| 7 | `sentinel_llm_calls_total` | Counter | model, tier, status | 4×4×3 = **48** | Coût LLM, hit rate par tier |
| 8 | `sentinel_llm_latency_seconds` | Histogram | model, tier | 4×4×12 = **192** | P95/P99 LLM — SLA narrative |
| 9 | `sentinel_llm_cost_usd_total` | Counter | model, tier | 4×4 = **16** | $ cumulé — gate budget mensuel |
| 10 | `sentinel_cache_lookups_total` | Counter | result | **2** (hit, miss) | Hit rate sémantique |
| 11 | `sentinel_notifier_send_total` | Counter | channel, status | 2×3 = **6** | Telegram/Discord livraison |
| 12 | `sentinel_data_provider_errors_total` | Counter | provider, error_type | 2×6 = **12** | Stabilité MT5/CSV |

**Total cardinalité estimée : ~1170 séries actives**. Largement sous le seuil Grafana Cloud Free (10k actives), Datadog ($0.10/host/custom-metric).

### 3.2 Snippets PR-ready

#### Patch 1 — `src/intelligence/main.py` (instancier la registry)

```python
# main.py:118 (juste après `registry = get_instrument_registry()`)
from src.performance.metrics import MetricsRegistry
metrics_registry = MetricsRegistry(prefix="sentinel")

# main.py:268-275 (passer à create_app)
api_app = create_app(
    signal_store=signal_store,
    llm_engine=llm_engine,
    scanner=scanner,
    circuit_breakers=circuit_breakers,
    health_checker=health_checker,
    rate_limiter=rate_limiter,
    metrics_registry=metrics_registry,   # ← AJOUT
)

# Et : passer la registry au scanner pour qu'il instrumente
scanner._metrics = metrics_registry  # ou via constructor arg
```

#### Patch 2 — `src/intelligence/sentinel_scanner.py:370` (signaux émis)

```python
# Ligne 370 actuel: self._signals_generated += 1
self._signals_generated += 1
if self._metrics is not None:
    self._metrics.counter("signals_emitted_total",
        "Signals emitted by scanner").inc(
        labels={"symbol": signal.symbol,
                "direction": signal.signal_type.value,
                "tier": signal.tier.value})
    self._metrics.histogram("confluence_score",
        "Confluence score distribution",
        buckets=(20, 30, 40, 50, 60, 70, 80, 90, 100)
    ).observe(signal.confluence_score,
        labels={"symbol": signal.symbol, "tier": signal.tier.value})
```

#### Patch 3 — `src/intelligence/sentinel_scanner.py:381` (cache hit)

```python
# Avant: self._cache_hits += 1
self._cache_hits += 1
if self._metrics is not None:
    self._metrics.counter("cache_lookups_total","Semantic cache lookups").inc(
        labels={"result": "hit"})
# Et au miss (ligne ~395):
if self._metrics is not None:
    self._metrics.counter("cache_lookups_total","").inc(labels={"result":"miss"})
```

#### Patch 4 — `src/intelligence/llm_narrative_engine.py:381,444` (latence + coût)

```python
# Dans _validate_with_haiku() et _narrate_single(), après chaque _call_api():
if self._metrics is not None:
    self._metrics.histogram("llm_latency_seconds",
        "LLM call latency", buckets=(0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32)
    ).observe(latency / 1000.0,
        labels={"model": model, "tier": tier.value})
    self._metrics.counter("llm_cost_usd_total","Cumulative LLM spend").inc(
        amount=cost,
        labels={"model": model, "tier": tier.value})
    self._metrics.counter("llm_calls_total","").inc(
        labels={"model": model, "tier": tier.value, "status": "success"})
```

#### Patch 5 — `src/intelligence/circuit_breaker.py` (gauge state)

```python
# Ajouter en bas de _on_success() et _on_failure() :
def _publish_state(self, registry):
    state_map = {CircuitState.CLOSED: 0, CircuitState.HALF_OPEN: 1, CircuitState.OPEN: 2}
    registry.gauge("circuit_breaker_state","Circuit state (0=closed,1=half,2=open)").set(
        state_map[self._state], labels={"name": self.name})
```

Plus simple : un thread d'arrière-plan dans `main.py` qui publie chaque 10s le state de chaque breaker dans une gauge.

#### Patch 6 — `src/delivery/telegram_notifier.py:186-190` & `discord_notifier.py:241-249`

```python
# Telegram (autour ligne 187):
self._messages_sent += 1
if self._metrics is not None:
    self._metrics.counter("notifier_send_total","").inc(
        labels={"channel": "telegram", "status": "success"})
# Au fail (ligne 190):
if self._metrics is not None:
    self._metrics.counter("notifier_send_total","").inc(
        labels={"channel": "telegram", "status": "fail"})
```

---

## 4. O4 — Tracing Architect

### 4.1 Pipeline 7-étages → spans OpenTelemetry

```
sentinel.scan_cycle (root span — un par bar close)
├── data_provider.fetch              (15-50 ms typique CSV / 80-300 ms MT5)
├── smc_engine.analyze               (30-150 ms — calcul BOS/CHOCH/FVG)
├── regime_agent.analyze             (10-60 ms)
├── news_agent.evaluate_news_impact  (5-30 ms — lookup CSV)
├── vol_forecaster.forecast          (20-100 ms HAR / 80-400 ms LightGBM)
├── confluence.analyze               (5-15 ms — pure compute)
├── state_machine.on_bar             (1-3 ms)
├── narrative.generate               (LLM mode: 1.2-8 s ; template: < 1 ms)
│   └── semantic_cache.get/put       (5-30 ms SQLite)
├── signal_store.publish             (10-40 ms SQLite WAL)
└── notifier.send_signal             (200-1500 ms réseau)
```

**Attribut commun (baggage)** : `signal_id`, `symbol`, `bar_ts`, `tenant_id` (à venir post-multitenancy).

### 4.2 Implémentation minimale (sans OpenTelemetry pour J+30)

Phase 1 — `contextvars` pour le `trace_id` :

```python
# src/intelligence/observability.py (nouveau fichier ~40 l.)
import contextvars, uuid, time, json, logging
from contextlib import contextmanager

trace_id_var = contextvars.ContextVar("trace_id", default=None)
signal_id_var = contextvars.ContextVar("signal_id", default=None)

logger = logging.getLogger("sentinel.trace")

@contextmanager
def span(name: str, **attrs):
    tid = trace_id_var.get() or uuid.uuid4().hex[:16]
    trace_id_var.set(tid)
    start = time.perf_counter()
    try:
        yield tid
        status = "ok"
    except Exception as e:
        status = f"error:{type(e).__name__}"
        raise
    finally:
        latency_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "span", extra={
                "trace_id": tid, "span": name, "status": status,
                "latency_ms": round(latency_ms, 2), **attrs})
```

Usage dans `sentinel_scanner._scan_once()` :

```python
from src.intelligence.observability import span, trace_id_var

with span("sentinel.scan_cycle", symbol=self._symbol):
    with span("data_provider.fetch", symbol=self._symbol):
        df = self._data_provider.get_ohlcv(...)
    with span("smc_engine.analyze"):
        enriched = smc_engine.analyze()
    # … etc
```

Phase 2 — OTLP exporter (J+90), basculer vers `opentelemetry-sdk` officiel et exporter vers Tempo/Jaeger.

### 4.3 Exemple JSON trace pour un signal XAUUSD score 75

```json
[
  {"trace_id":"a1b2c3d4e5f60718","span":"sentinel.scan_cycle","status":"ok","latency_ms":2347.81,
   "symbol":"XAUUSD","bar_ts":"2026-04-25T14:15:00Z","signal_id":"sig_3f9a"},
  {"trace_id":"a1b2c3d4e5f60718","span":"data_provider.fetch","status":"ok","latency_ms":34.20,"symbol":"XAUUSD"},
  {"trace_id":"a1b2c3d4e5f60718","span":"smc_engine.analyze","status":"ok","latency_ms":92.55,
   "bos_signal":1.0,"fvg_signal":0.0,"choch_signal":0.0},
  {"trace_id":"a1b2c3d4e5f60718","span":"regime_agent.analyze","status":"ok","latency_ms":42.10,"regime":"trend"},
  {"trace_id":"a1b2c3d4e5f60718","span":"news_agent.evaluate","status":"ok","latency_ms":8.40,"blackout":false},
  {"trace_id":"a1b2c3d4e5f60718","span":"vol_forecaster.forecast","status":"ok","latency_ms":71.32,
   "regime_state":"normal","atr":12.3},
  {"trace_id":"a1b2c3d4e5f60718","span":"confluence.analyze","status":"ok","latency_ms":7.81,
   "score":75.4,"tier":"NARRATOR"},
  {"trace_id":"a1b2c3d4e5f60718","span":"state_machine.on_bar","status":"ok","latency_ms":1.20,
   "from_state":"HOLD","to_state":"BUY"},
  {"trace_id":"a1b2c3d4e5f60718","span":"semantic_cache.get","status":"ok","latency_ms":12.45,"result":"miss"},
  {"trace_id":"a1b2c3d4e5f60718","span":"narrative.generate","status":"ok","latency_ms":1832.50,
   "model":"claude-haiku-4.6","tier":"NARRATOR","tokens_in":420,"tokens_out":180,"cost_usd":0.0024},
  {"trace_id":"a1b2c3d4e5f60718","span":"signal_store.publish","status":"ok","latency_ms":18.90,"signal_id":"sig_3f9a"},
  {"trace_id":"a1b2c3d4e5f60718","span":"notifier.send_signal","status":"ok","latency_ms":845.23,"channel":"telegram"}
]
```

Total cycle : 2 348 ms (LLM = 78% du temps).

---

## 5. O5 — Alerting Engineer

### 5.1 Justification des seuils (depuis `reports/baseline_full.json` — 6 ans XAU M15)

Référence empirique :

| Mesure | Valeur baseline | Source |
|---|---|---|
| `signals_per_day` (post state machine) | **0.625** | `baseline_full.json` |
| `arms_started` total / 6 ans | **3 741** (≈ 1.71/jour) | idem |
| `confirmation_rate` | **0.427** | idem |
| Score P95 | **53.17** | idem |
| Score P99 | **60.15** | idem |
| `bars_processed` | 106 000 | idem |
| `bars_with_bos` | 3 859 (3.6%) | idem |
| `signals_produced_by_detector` (avant SM) | 31 096 (≈ 14.2/jour) | idem |

Pour un scanner M15 avec 1 symbole, on attend **96 bars/jour** (4×24).

### 5.2 Règles PromQL (6 alertes)

```yaml
# infrastructure/alertmanager/rules.yml
groups:
- name: sentinel.critical
  rules:

  # 1. Pipeline silencieux — scanner mort ou bloqué
  - alert: ScannerHeartbeatLost
    expr: rate(sentinel_bars_scanned_total[10m]) == 0
    for: 12m
    labels: {severity: critical}
    annotations:
      summary: "Scanner heartbeat lost on {{ $labels.symbol }}"
      description: "No bar scanned in 10 min. Expected ~6.4 bars/h on M15."
      runbook: |
        1. Check /health (scanner_running=true?).
        2. Check data provider: `docker logs sentinel | grep "Data provider error"`.
        3. If MT5: confirm broker session active.
        4. Restart scanner thread (SIGUSR1 not implemented — full restart needed).

  # 2. Circuit breaker open trop longtemps
  - alert: CircuitBreakerOpenSustained
    expr: sentinel_circuit_breaker_state{name=~"llm|telegram|discord"} == 2
    for: 5m
    labels: {severity: warning}
    annotations:
      summary: "Circuit '{{ $labels.name }}' OPEN > 5 min"
      description: "Recovery timeout configured at 60s (LLM) or 120s (notifier). Sustained OPEN means upstream is hard-down."
      runbook: |
        1. LLM: curl https://api.anthropic.com/v1/health.
        2. Telegram: check api.telegram.org/bot<TOKEN>/getMe.
        3. If 3rd party UP: inspect last 50 log lines for auth errors.

  # 3. P99 LLM latency dérive
  - alert: LLMLatencyP99High
    expr: histogram_quantile(0.99, sum(rate(sentinel_llm_latency_seconds_bucket[5m])) by (le)) > 8
    for: 10m
    labels: {severity: warning}
    annotations:
      summary: "LLM P99 latency > 8s for 10 min"
      description: "Anthropic typical P99 = 3-5s. > 8s indicates upstream degradation."
      runbook: |
        1. Check status.anthropic.com.
        2. Tier breakdown: histogram_quantile(0.99, ... by (le, tier)).
        3. If sustained > 30 min: switch NARRATIVE_MODE=template.

  # 4. /health degraded persistant
  - alert: HealthDegraded
    expr: sentinel_health_status != 0
    for: 2m
    labels: {severity: critical}
    annotations:
      summary: "/health reports DEGRADED"
      runbook: |
        1. curl /api/v1/health and inspect components[].
        2. Most common: scanner_running=false (fix: restart container).

  # 5. Anomalie volume signaux (3σ)
  - alert: SignalRateAnomaly
    expr: |
      abs(rate(sentinel_signals_emitted_total[1h])
          - avg_over_time(rate(sentinel_signals_emitted_total[1h])[7d:1h]))
      > 3 * stddev_over_time(rate(sentinel_signals_emitted_total[1h])[7d:1h])
    for: 30m
    labels: {severity: warning}
    annotations:
      summary: "Signal rate 3σ off baseline"
      description: "Baseline ~0.625 sig/day/symbol = 0.026/h. 3σ alarm fires when rate diverges sharply."
      runbook: |
        1. Spike > baseline: data corruption or model dérive (cf. eval_02_confluence).
        2. Drop to 0: state machine stuck or detector silent.

  # 6. Coût LLM mensuel
  - alert: LLMSpendBudgetWarning
    expr: increase(sentinel_llm_cost_usd_total[30d]) > 50
    for: 1h
    labels: {severity: warning}
    annotations:
      summary: "LLM cost > $50/30d"
      description: "Budget gate. With template default, ce gauge devrait rester ~0."
      runbook: |
        1. Si NARRATIVE_MODE=template (défaut), ce trigger = bug; revoir build.
        2. Si llm: tier breakdown; couper INSTITUTIONAL si solo founder.

# 7-8 (optionnels, après J+30):
  # 7. Cache hit rate trop bas (LLM mode only)
  - alert: SemanticCacheCold
    expr: |
      rate(sentinel_cache_lookups_total{result="hit"}[1h])
      / rate(sentinel_cache_lookups_total[1h]) < 0.05
    for: 2h
    labels: {severity: info}
    annotations:
      summary: "Cache hit rate < 5%"
      description: "Voir eval_05: cache_key inclut bar_ts → hit ≈ 0%. Bug connu, fix en backlog."

  # 8. Notifier failure rate
  - alert: NotifierFailureRate
    expr: |
      rate(sentinel_notifier_send_total{status="fail"}[15m])
      / rate(sentinel_notifier_send_total[15m]) > 0.2
    for: 10m
    labels: {severity: warning}
    annotations:
      summary: "Notifier ({{ $labels.channel }}) > 20% fail"
      runbook: 1. Check rate-limit (Telegram=30 msg/s/bot, Discord=5/s/webhook). 2. Inspect last error in logs filtered by channel.
```

### 5.3 Inhibitions

```yaml
inhibit_rules:
  - source_match: { alertname: ScannerHeartbeatLost }
    target_match_re: { alertname: 'SignalRateAnomaly|LLMLatencyP99High' }
    equal: [symbol]
  # Si scanner mort, inutile de spam alarm latency LLM.
```

---

## 6. O6 — Cost Analyst

### 6.1 Pricing 2026 (estimations, à valider Q2 2026)

> ⚠️ Les pricings ci-dessous sont des estimations basées sur les niveaux publics 2024-2025 + extrapolation 2026. À re-confirmer via WebFetch avant tout engagement contractuel.

| Outil | Free tier 2026 | Plan suivant | Limitation principale |
|---|---|---|---|
| **Grafana Cloud Free** | 10k séries actives, 50 GB logs/mois, 14j retention métriques, 50 GB traces | Pro $19/m + usage | Cap dur sur séries — overage = lockout, pas billing surprise |
| **Datadog** | 5 hosts × 14j metrics, pas de logs | Pro: ~$19/host/mois + $0.10/custom-metric/100 + $1.27/M log events | Custom metrics chers — exploser vite |
| **Better Stack (Logtail + Uptime)** | 1 GB logs/mois, 3 monitors | $25/m → 30 GB | Pas de metrics (logs + uptime only) |
| **Loki self-hosted** (sur Railway $5/m sidecar) | Illimité (storage S3 ~$0.023/GB) | Idem | Ops overhead = ~2h/mois mainten. + alerting via Prometheus séparé |
| **New Relic** | 100 GB/mois ingest gratuit | $0.30/GB au-delà | Generous mais lock-in NRQL |
| **Sentry** (logs/erreurs) | 5k events/mois | Team $26/m → 50k | Spécialisé errors, complément aux 4 ci-dessus |

### 6.2 Scénarios volume

Hypothèses :
- 1 signal = ~12 lignes log JSON (~1 KB chacune) + ~12 spans (~500 B) + ~30 observations metrics.
- Header HTTP + middleware = ~40 lignes log/scan_cycle = ~50 KB par cycle (généreux).

| Volume | Logs/jour | Métriques actives | Traces/jour | Reco |
|---|---|---|---|---|
| **1 k signaux/jour** (solo testing) | ~12 MB log + ~50 MB middleware = **~60 MB/j → 1.8 GB/mois** | ~1.2 k séries | ~12 k traces (~6 MB) | **Grafana Cloud Free** suffit. |
| **10 k signaux/jour** (~10 instruments scaling) | ~600 MB/j = **18 GB/mois** | ~3 k séries | ~120 k traces | Grafana Cloud Free OK (limite 50 GB logs, 10 k séries). |
| **100 k signaux/jour** (multi-tenant SaaS) | ~6 GB/j = **180 GB/mois** | ~12 k séries (au-delà du Free 10k) | ~1.2 M traces | **Grafana Pro $19/m + overage** OU **Loki self-hosted** (S3 backend ≈ $5/mois storage + $5 compute Railway). |

### 6.3 Recommandation

**Phase startup (< 10k signaux/jour, < 100 MAU)** :
- **Grafana Cloud Free** pour metrics + logs + traces (stack OTLP unifiée).
- **Sentry Free** pour les exceptions (5k events/mois — largement suffisant).
- **instatus** (page status, voir §7) : free tier (1 page, 10 composants).
- **Coût total : $0/mois** jusqu'à ~10k signaux/jour.

**Phase scale (100k signaux/jour ou enterprise SLA)** :
- Bascule **Loki self-hosted** sur Railway/Fly avec S3 backend ($10/mois total).
- Garder Grafana Cloud Pro ($19/m) pour metrics + traces (les "queries-heavy").
- **Coût total : ~$30-50/mois** à 100k sig/j. Datadog équivalent : $300+/mois.

**À éviter pour solo founder** :
- Datadog (custom metrics tarifées — explose vite).
- New Relic (lock-in NRQL ; sortie pénible si pivot).

---

## 7. O7 — Status Page

### 7.1 Comparatif

| Critère | statuspage.io (Atlassian) | instatus | Better Stack Status |
|---|---|---|---|
| Free tier | 50 subscribers | **Illimité subscribers, 1 page, 10 composants** | 1 page basique |
| Premier plan payant | $29/m | $20/m | $25/m |
| API webhook ingest | ✓ (PagerDuty-like) | ✓ | ✓ |
| Composants public | OK | OK | OK |
| Custom domain free | Non | Oui | Oui |
| Verdict solo founder | Trop cher | **WINNER** | OK mais Better Stack moins focalisé |

### 7.2 Composants public-facing recommandés

```
Smart Sentinel AI · status.smartsentinel.ai
├── Signal Engine          (mapping: scanner_running + bars_scanned rate > 0)
├── Narrative Engine (LLM) (mapping: circuit_breaker_state{name="llm"} != 2)
├── Telegram / Discord     (mapping: notifier circuit + send fail rate)
├── REST API               (mapping: /health response time + status code)
└── Data Provider (MT5/CSV) (mapping: data_provider_errors_total rate)
```

### 7.3 Webhook ingest depuis nos circuit breakers

Patch `src/intelligence/circuit_breaker.py` :

```python
# Ajouter un callback hook :
def __init__(self, ..., on_state_change: Optional[Callable[[str, CircuitState], None]] = None):
    self._on_state_change = on_state_change

# Dans _on_failure() / _on_success(), juste après le changement de state :
if self._on_state_change:
    self._on_state_change(self.name, self._state)

# Dans main.py:
def _statuspage_hook(name: str, state: CircuitState):
    requests.post(STATUSPAGE_WEBHOOK_URL, json={
        "component": name, "status": "operational" if state == CircuitState.CLOSED else "degraded"})
llm_breaker = CircuitBreaker(..., on_state_change=_statuspage_hook)
```

---

## 8. O8 — Red-Team (challenge)

| Recommandation | Verdict honnête solo founder J0 | Justification |
|---|---|---|
| OpenTelemetry SDK + Tempo | **OVERKILL J0** | Pas avant 100 MAU. `contextvars` + JSON logs avec `trace_id` (§4.2 Phase 1) couvre 80% du besoin. |
| 12 métriques Prometheus | OK J0 | Mais commencer par les **6 critiques** : signals_emitted, scan_duration, circuit_state, llm_latency, llm_cost, cache_hit. Les 6 autres → J+30. |
| 8 règles d'alerte | **2-3 OK J0** | `ScannerHeartbeatLost` + `CircuitBreakerOpenSustained` + `HealthDegraded` suffisent. Le reste a besoin de 30 jours de données pour calibrer la baseline. |
| Status page publique | **PAS J0** | Si 0 utilisateur payant, page status est cosmétique. Reporter à J+90 ou après les 10 premiers clients. |
| Datadog | **NON, jamais avant scale** | $20+/host/m × custom metrics = $200+/m vs Grafana Cloud $0. ROI négatif tant que < 1k MAU. |
| Loki self-hosted | **PAS J0** | Ops overhead non justifié. Grafana Cloud Free Loki suffit (50 GB/mois). |
| Recording rules PF/win-rate | **J+90** | Demande > 30 jours données live. PF baseline = 1.086 (cf. eval_22) — calculer après stabilisation, pas avant. |
| Sentry pour exceptions | OK J0 | 5k events free = ~167/jour → suffit pour solo testing. |
| Migration `prometheus_client` officiel | **PAS J0** | La `MetricsRegistry` maison fonctionne (469 l. propres). La câbler **avant** de la remplacer. Migration = J+90 si scaling, sinon ne migrer **jamais**. |
| Status page subscribers | **PAS J0** | 50 subscribers limit Atlassian = 0 problème quand on a < 50 utilisateurs. |

### Ce qu'il y a d'overkill dans la spec :

1. **8 règles d'alerte calibrées** : sans 30j de données live, les seuils 3σ sont devinés. **3 règles binaires** ("scanner mort", "circuit ouvert > 5 min", "/health pas OK") couvrent 90% du besoin.
2. **Status page publique** : tant que < 10 clients, un Discord channel "incidents" suffit.
3. **OTLP/Tempo full stack** : repousser à J+90 minimum.
4. **Custom metrics fines (par tier × symbol × direction)** : cardinalité inutile à 1 utilisateur. Démarrer **sans labels**, ajouter quand besoin business.

### Ce qui peut attendre 6 mois :

- Multi-tenancy `tenant_id` propagation (pas avant offres payantes).
- Recording rules Prometheus (PF/Sharpe glissants).
- Distributed tracing complet OTLP.
- SLO/Error Budget formel (a besoin baseline 60j).

---

## 9. O9 — Synthèse

### 9.1 Notes par dimension

| Dimension | Note /10 | Justification |
|---|---:|---|
| Logs (config + structure) | **5** | JSONFormatter présent mais ignore `extra={}` ; aucun `signal_id` propagé ; 109 `print()` dont 25 critiques |
| Metrics | **1** | `MetricsRegistry` jamais instanciée → `/metrics` vide en prod. Code mort. |
| Tracing | **0** | Aucun `trace_id`, aucun span. Pipeline 7-étages opaque. |
| Alerting | **2** | Aucune règle Prometheus, aucun runbook. Seules les notifications Discord ad-hoc dans `main.py` |
| Health checks | **8** | `/health` riche, `HealthChecker` agrégé, components, kill_switch, scanner_running |
| Status page | **0** | Inexistante |
| Cost discipline | **9** | Pas un dollar gaspillé en stack obs (parce que rien n'est branché) |
| **GLOBAL** | **3.2 / 10** | « Os bien dimensionnés (JSONFormatter, /health, HealthChecker, MetricsRegistry custom), 0 chair (aucune métrique active, 0 trace, 0 alerte). Le tissu connectif manque. » |

### 9.2 Plan phasé

#### J0 — Quick wins (2 jours dev solo)

1. **Instancier `MetricsRegistry` dans `main.py`** (§3.2 Patch 1) → `/metrics` non-vide. Effort : 30 min.
2. **Fusionner `extra={}` dans JSONFormatter** (§2.3 patch) — débloque le contexte structuré pour tous les `logger.*` futurs. Effort : 15 min.
3. **Migrer les 6 `print()` critiques** :
   - `src/security/hmac_manager.py:220-223` (secrets) → `logger.warning(extra={"redacted": True})`
   - `src/security/alert_manager.py:850-859` → `logger.warning(extra={"alert_id": ..., "severity": ...})`
   - `src/security/dead_man_switch.py:620-622` → `logger.critical(extra={"healthy": False})`
   - `src/persistence/kill_switch_store.py:128` → `logger.warning(extra={"halted_before_restart": True})`
   - `src/live_trading/mt5_connector.py:287` → `logger.info(extra={"ticket": ...})`
   - `src/live_trading/async_order_manager.py:172` → `logger.info(extra={"fill_price": ...})`
   Effort : 1 h.
4. **3 alertes Prometheus minimales** (`ScannerHeartbeatLost`, `CircuitBreakerOpenSustained`, `HealthDegraded`). Effort : 1h (rules.yml + scrape config).
5. **Brancher 6 métriques critiques** (signaux émis, scan_duration, circuit_state, llm_latency, llm_cost, cache_hit). Effort : 4-6 h.

→ **Total J0 = 1.5 jours dev**. Délivre une vraie observabilité minimum viable.

#### J+30

6. Ajouter `extra={'signal_id', 'symbol', 'stage'}` aux 30 `logger.*` de `sentinel_scanner.py`.
7. `src/intelligence/observability.py` (§4.2 Phase 1) — spans via `contextvars`.
8. 6 métriques restantes (signals_held, bars_scanned, llm_calls, cache_lookups, notifier_send, data_provider_errors).
9. Sentry SDK pour les `logger.exception()` (Free tier 5k events/mois).
10. 5 règles alerting supplémentaires (LLMLatencyP99High, SignalRateAnomaly, …).
11. Dashboard Grafana Business (livré : `reports/eval_16_grafana_business.json`).

#### J+90

12. OTLP exporter (OpenTelemetry SDK officiel) → Tempo via Grafana Cloud.
13. Status page **instatus.com** (free tier) avec webhook depuis circuit breakers.
14. Recording rules Prometheus (`sentinel:profit_factor:30d`, `sentinel:win_rate:30d`).
15. Multi-tenant `tenant_id` propagation (préparation offres payantes).
16. Migration vers `prometheus_client` officiel (déprécier `MetricsRegistry` maison) — uniquement si traffic > 1k req/min.

### 9.3 Top 5 actions par effort × impact

| # | Action | Effort | Impact | Score |
|---|---|---|---|---|
| 1 | Instancier `MetricsRegistry` dans `main.py` | 30 min | TRÈS HAUT (débloque tout `/metrics`) | **★★★★★** |
| 2 | Fusionner `extra={}` dans JSONFormatter | 15 min | HAUT (débloque contexte structuré pour 100+ call-sites) | **★★★★★** |
| 3 | Câbler 6 métriques critiques | 4-6 h | TRÈS HAUT | **★★★★★** |
| 4 | 3 alertes Prometheus minimales + runbooks | 1 h | HAUT (passe de "MTTR illimité" à "MTTR < 15 min") | **★★★★☆** |
| 5 | Patch 25 `print()` critiques → `logger.*` | 1 h | MOYEN (sécurité + cohérence) | **★★★☆☆** |

### 9.4 KPIs cibles

| KPI | Cible J+30 | Cible J+90 |
|---|---|---|
| MTTR sur incidents detectés | < 30 min | < 15 min |
| % signaux avec `signal_id` traçable end-to-end | 100% | 100% + trace OTLP complet |
| Coverage métriques modules critiques | 6/6 (scanner, LLM, cache, notifier, circuit, /health) | 12/12 |
| Fausses alertes / semaine | < 3 | < 1 |
| Logs structurés (vs `print()`) en `src/intelligence/* + src/api/*` | 100% | 100% |
| Coût mensuel obs stack | $0 (Grafana Free + Sentry Free) | < $30 |

---

## 10. Annexes

- `reports/eval_16_metrics_catalog.md` — catalogue métriques détaillé, réutilisable comme spec.
- `reports/eval_16_grafana_business.json` — dashboard Grafana provisioning (4 panels).
- Files de référence : `src/intelligence/main.py`, `src/api/app.py`, `src/api/routes/health.py`, `src/api/routes/prometheus.py`, `src/intelligence/circuit_breaker.py`, `src/performance/metrics.py`, `src/intelligence/sentinel_scanner.py`.
