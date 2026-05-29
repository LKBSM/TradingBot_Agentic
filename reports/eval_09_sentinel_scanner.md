# Eval 09 — Sentinel Scanner (boucle temps réel)

> **Périmètre audité** : `src/intelligence/sentinel_scanner.py` (910 l), `tests/test_sentinel_scanner.py`. Dépendances : `src/intelligence/main.py:_calibrate_system`, `src/intelligence/state_persistence.py`.
>
> **Date** : 2026-04-28 · **Branch** : `main` · **Mission** : auditer la cadence, gestion erreurs, scalabilité horizontale, observabilité, backpressure, graceful shutdown, SLA atteignable.

---

## 0. Résumé exécutif — Note **6.5 / 10**

| Axe | Note | Verdict |
|---|---|---|
| Cadence & latence | 6/10 | `time.sleep(60s)` polling fixe + sync sequential pipeline. P95 par scan ~4-8 s. |
| Gestion erreurs | 8/10 | Try/except bien posés sur 7 étapes du pipeline, scanner survit aux exceptions sectorielles. |
| Scalabilité horizontale | **3/10** | Mono-symbole en l'état (1 instance = 1 symbol). Pas de `MultiSymbolScanner` malgré la docstring l.10. |
| Observabilité (trace_id E2E) | **3/10** | 9 compteurs + log info riche, mais aucun trace_id qui survit DataProvider → Telegram. Voir aussi eval_16. |
| Backpressure / queue | **2/10** | Aucune queue. Si Telegram down 10 min : circuit breaker ouvre, signaux perdus, pas de TTL ni dedup. |
| Graceful shutdown | 8/10 | `shutdown()` join thread + `_persist_state_machine` + log stats. Bon. |
| SLA "30s après bar close" | 7/10 | Atteignable côté code (P95 4-8s) **mais** poll-based 60s tampon = **30-90s pratique**. |
| Circuit breaker integration | 9/10 | LLM breaker + Notifier breaker + fallback template. Excellent design défensif. |

**Verdict commercial** : le scanner est **fonctionnellement correct** pour personal-use et soft-launch FREE-only, mais **ne tient pas un SLA "30s après close"** ni "multi-asset" en prod sans refactor. Le passage à 100+ MAU nécessite : event-driven (vs polling), queue Redis pour dedup/backpressure, multi-symbol via process worker dédié.

---

## 1. Cartographie code

### 1.1 Pipeline `_scan_once` (l.227-407) — 10 étapes

```
1. DataProvider.get_ohlcv (lookback=200)              ~3-15 ms cached / 150-400 ms cold
2. validate_ohlcv (data_quality gate)                 ~5 ms
3. SmartMoneyEngine.analyze (Numba JIT)               ~200-500 ms
4. RegimeAgent.analyze (HMM inference)                ~50-150 ms
5. NewsAgent.evaluate_news_impact                     ~5-30 ms
6. VolForecaster.forecast (HAR/LGBM/hybrid)           ~10-50 ms
7. ConfluenceDetector.analyze                         ~1-3 ms
8. SignalStateMachine.on_bar (gating)                 ~< 1 ms
9. SemanticCache.get / LLMNarrativeEngine.generate    ~10 ms hit / 2-8s miss
10. SignalStore.publish + Notifier.send_signal        ~200-500 ms (sync HTTP)
                                                      ─────────────────────────
P95 par scan estimé                                   4-8 s (cache miss LLM)
                                                      0.5-1 s (cache hit / template fallback)
```

### 1.2 Constructor (l.71-144)

12 dépendances injectées (data_provider, smc_factory, regime_agent, news_agent, confluence, llm_engine, cache, signal_store, notifier, vol_forecaster, llm_circuit_breaker, notifier_circuit_breaker). 9 compteurs stats, 1 thread, 1 RLock implicite via state_machine.

### 1.3 Lifecycle

- `start(blocking=True)` (l.155) : restore_state_machine → `_run_loop` (mainthread) ou Thread daemon
- `_run_loop` (l.212-221) : `while self._running: try _scan_once(); time.sleep(60s)`
- `shutdown()` (l.174-180) : `_running=False`, `thread.join(timeout=10)`, `_persist_state_machine`

### 1.4 Erreurs gérées (try/except sectoriels)

| Étape | Comportement sur exception |
|-------|---------------------------|
| DataProvider | log error + `_errors++` + `return None` (skip bar) |
| validate_ohlcv (`DataQualityError`) | log error + `return None` |
| SmartMoneyEngine | log error + `return None` |
| RegimeAgent | log warning + `regime=None` (continue with passthrough) |
| NewsAgent | log warning + `news=None` |
| VolForecaster | log warning + `vol_forecast=None` |
| ConfluenceDetector | pas de try (lève si bug) |
| StateMachine | try/except dans `_step_state_machine` → `return None` |
| LLM/Notifier | CircuitBreaker + algo fallback Template (l.508-560) |

✅ **Excellent** : une exception sectorielle ne tue jamais le scanner.

---

## 2. Audit ligne à ligne — bugs & limitations

### Bug n°1 — Polling fixe `time.sleep(60s)` (l.221)

```python
def _run_loop(self) -> None:
    while self._running:
        try:
            self._scan_once()
        except Exception as e:
            self._errors += 1
            logger.error("Scanner error: %s", e, exc_info=True)
        if self._running:
            time.sleep(self._poll_interval)  # 60s par défaut
```

**Conséquence** : sur M15, un bar clôture à `:00`, `:15`, `:30`, `:45`. Si `_scan_once` finit à `:00:05`, le prochain scan est à `:01:05`. **La détection du bar `:15` arrive entre `:15:05` et `:16:05` worst-case** — ça peut violer un SLA "30s après close".

**Fix P0** : aligner le polling sur les multiples de 15 min, ou descendre à `poll_interval=10s` (mais coût compute × 6).

**Fix P1** : event-driven via WebSocket (cf. eval_08 §6).

### Bug n°2 — Mono-symbole : pas de `MultiSymbolScanner`

La docstring l.10 dit "Supports multi-symbol scanning via MultiSymbolScanner". **Grep confirme** :
- `class MultiSymbolScanner` n'existe nulle part dans `src/`
- L'implémentation actuelle = 1 instance `SentinelScanner` = 1 `symbol`

**Conséquence** : pour scanner 6 symboles, il faut 6 threads → 6 × pipeline sync → contention SMC engine, RegimeAgent, NewsAgent (les agents ne sont **pas thread-safe**, voir eval_01).

**Fix** : créer un véritable `MultiSymbolScanner` qui itère sur `symbols: List[str]` dans le même `_scan_once` en réutilisant les engines partagés. Réduit le coût × N à coût × 1 + N × pipeline-end.

### Bug n°3 — Aucune queue / dedup pour notifications ratées

```python
# l.575-582
except CircuitOpenError:
    self._notification_failures += 1
    logger.warning("Telegram circuit OPEN — skipping notification for %s", signal.signal_id)
```

Si Telegram down 10 min, **toutes les notifications sont silencieusement perdues**. Pas de queue Redis, pas de retry async, pas de TTL.

L'utilisateur ne reçoit jamais le signal. C'est invisible jusqu'à la prochaine notification réussie.

**Fix P1** :
```python
class NotificationQueue:
    def enqueue(self, signal, narrative, ttl_minutes=15)
    def replay_recent(self, since_ts)  # called on circuit close
```

Persistée en SQLite ou Redis. À la fermeture du circuit, replay des signaux non-expirés.

### Bug n°4 — `bar_ts = str(df.index[-1])` insufficient ID

```python
# l.255-257
bar_ts = str(df.index[-1]) if hasattr(df.index, '__len__') else str(len(df))
if bar_ts == self._last_bar_ts:
    return None
```

`str(pd.Timestamp)` peut varier selon la version pandas (microseconds vs nanoseconds). Si Pandas upgrade change le format, `_last_bar_ts` ne match plus → **double-scan du même bar**.

**Fix** : normaliser `df.index[-1].isoformat()` avec timezone explicite.

### Bug n°5 — `validate_ohlcv` lève sur 1 bar corrompu = scanner skip toute la fenêtre

```python
# l.246-252
try:
    validate_ohlcv(df, self._symbol, self._timeframe, strict=True)
except DataQualityError as e:
    logger.error("OHLCV quality gate failed: %s", e)
    self._errors += 1
    return None
```

Un seul bar invalide (NaN dans Close) → scanner skip 1 entier minute. Si le feed est durablement cassé, le scanner reste en boucle d'erreur sans alerter.

**Fix** : compteur `consecutive_data_quality_failures` ; à > N, page operator + fallback DataProvider.

### Bug n°6 — `_send_notification_safe` ne loggue pas le contexte signal

```python
# l.572-585
try:
    if self._notifier_breaker is not None:
        self._notifier_breaker.call(_call_notifier)
    else:
        _call_notifier()
except CircuitOpenError:
    self._notification_failures += 1
    logger.warning("Telegram circuit OPEN — skipping notification for %s", signal.signal_id)
```

Pas de `extra={'signal_id': signal.signal_id, 'symbol': self._symbol}`. Logs JSON non-corrélables. Voir eval_16 même bug.

### Bug n°7 — `_publish_exit_transition` calcule `pnl = delta` sans frais

```python
# l.466-469
delta = exit_px - float(entry)
direction = transition.direction.value if transition.direction else "LONG"
pnl = delta if direction == "LONG" else -delta
```

PnL brut sans spread/slippage. Pour Telegram annoncé = 0 frais incluso. Cohérent avec eval_18 (coûts transaction = 0). À fixer en même temps que le replay harness.

### Bug n°8 — `_restore_state_machine` skip staleness check on cold-start

```python
# l.193-200
restored = load_state_machine(
    self._persistence_path,
    current_bar_ts=None,
    max_staleness_bars=0,  # cold start — no reference bar
)
if restored is not None:
    self._state_machine = restored
```

Sur reload après 7 jours d'arrêt, le scanner restore une state machine **avec un signal ACTIVE de 2026-04-22** alors qu'on est 2026-04-29. Le commentaire l.190-192 dit "scanner loop will overwrite stale state on the next persist cycle" — mais avant ce moment-là, `/api/v1/state` retourne un signal stale.

**Fix** : faire un `validate_ohlcv` rapide AVANT restore et passer le `last_bar_ts` réel à `load_state_machine`.

### Bug n°9 — Aucune métrique `bars_lost_to_quality_gate` ni `bars_lost_to_skip_same_ts`

Compteurs présents : `_bars_scanned`, `_signals_generated`, `_signals_held_by_state_machine`, `_state_transitions_emitted`, `_cache_hits`, `_llm_calls`, `_llm_failures`, `_notification_failures`, `_errors`, `_fallback_uses`.

Manquent :
- bars rejetés par data_quality
- bars skipped same-timestamp (vs nouveaux)
- P50/P95/P99 latence de `_scan_once`
- temps en chaque étape du pipeline

Sans ça, pas moyen de localiser un slowdown en prod.

### Bug n°10 — `time.sleep(60s)` non-interruptible

```python
if self._running:
    time.sleep(self._poll_interval)
```

`shutdown()` set `self._running=False`, mais le thread est en plein `time.sleep(60)`. Le `.join(timeout=10)` (l.178) timeout silencieusement. Le thread daemon meurt avec le process mais shutdown gracieux non-garanti.

**Fix** : `threading.Event.wait(timeout=60)` au lieu de `time.sleep(60)`.

```python
self._stop_event = threading.Event()
...
def shutdown(self):
    self._stop_event.set()
    self._thread.join(timeout=10)
...
def _run_loop(self):
    while not self._stop_event.is_set():
        try:
            self._scan_once()
        except Exception as e:
            ...
        self._stop_event.wait(timeout=self._poll_interval)
```

---

## 3. Profil perf estimé

### 3.1 Latence par étape (XAU M15, 1 symbole)

| Étape | Cold | Warm | P95 |
|-------|-----:|-----:|----:|
| `get_ohlcv` (CSV cached) | 150 ms | 3 ms | 15 ms |
| `validate_ohlcv` | 8 ms | 5 ms | 10 ms |
| `SmartMoneyEngine.analyze` (Numba) | 800 ms | 250 ms | 500 ms |
| `RegimeAgent.analyze` | 200 ms | 50 ms | 150 ms |
| `NewsAgent.evaluate_news_impact` | 30 ms | 5 ms | 30 ms |
| `VolForecaster.forecast` (hybrid) | 80 ms | 20 ms | 50 ms |
| `ConfluenceDetector.analyze` | 3 ms | 1 ms | 3 ms |
| `StateMachine.on_bar` | 1 ms | < 1 ms | 1 ms |
| `SemanticCache.get` | 8 ms | 5 ms | 15 ms |
| `LLM.generate_narrative` (Haiku) | 4 s | — | 8 s (P95) |
| `SemanticCache.put` | 5 ms | 5 ms | 15 ms |
| `SignalStore.publish` | 10 ms | 8 ms | 25 ms |
| `Notifier.send_signal` | 300 ms | 200 ms | 500 ms |
| **Total cache miss + LLM** | **5.6 s** | **4.5 s** | **9.3 s** |
| **Total cache hit (no LLM)** | **1.6 s** | **550 ms** | **1.3 s** |

### 3.2 Multi-symbole estimé (6 symboles, mono-thread)

Pipeline étapes 1-7 partagées partiellement (data + SMC + regime + news + vol par symbole, mais agents pas thread-safe → sérialisation forcée). Coût total = **6 × 1.6 s à 6 × 9.3 s** = **10-56 s par cycle complet**.

→ **Cycle 60s déjà trop court pour 6 symboles cache-miss**. Le scanner devient permanently catching up.

---

## 4. SLO / SLA proposé

### 4.1 Cible client

> "Vous recevez votre alerte Telegram **dans les 90 secondes** après la clôture du bar M15 sur lequel le signal s'est confirmé."

(NB : pas 30 s comme suggéré dans le prompt — pas atteignable avec le polling actuel.)

### 4.2 Décomposition latence cible

| Composant | Cible | Mesure |
|-----------|-------|--------|
| Bar close → polling cycle pickup | ≤ 60 s | `bar_close_ts - scan_start_ts` |
| `_scan_once` end-to-end (cache hit) | ≤ 1.5 s | timer dans `_scan_once` |
| `_scan_once` end-to-end (cache miss) | ≤ 9 s | idem |
| Telegram delivery | ≤ 1 s | acknowledge HTTP |
| **Total P95 cache miss** | **≤ 90 s** | sum |
| **Total P95 cache hit** | **≤ 70 s** | sum |

### 4.3 Implémentation observabilité

```python
# In _scan_once
with timer(self._latency_histogram, "scan_total"):
    ...
    with timer(self._latency_histogram, "scan_smc"):
        smc_engine.analyze()
    ...
```

Exposer dans `/metrics` (Prometheus histogram). Alerter via Grafana si P95 > 90 s ou P99 > 180 s.

### 4.4 SLI à publier

- **Disponibilité scanner** : `% of expected bars scanned successfully` ≥ 99 %/jour
- **Latence signal** : `P95 bar_close → signal_published` ≤ 90 s
- **Cache effectiveness** : `cache_hit_ratio` ≥ 30 %
- **Error rate** : `errors / bars_scanned` ≤ 1 %

---

## 5. Top 5 améliorations priorisées

| # | Amélioration | Effort | Impact latence | Impact reliability | Priorité |
|---|--------------|:------:|:--------------:|:------------------:|:--------:|
| **1** | **Polling event-driven** : `Event.wait` interruptible + alignement clock M15 | S | -30s en moy | shutdown propre | P0 |
| **2** | **Notification queue + replay sur circuit close** | M | 0 | élimine signal-loss | P0 |
| **3** | **Multi-symbol unifié dans `_scan_once`** + agents thread-safe | M | -50% à 6 sym | scalabilité | P1 |
| **4** | **Latency histogram per étape + Prometheus export** | S | 0 | observability ⭐⭐⭐⭐ | P1 |
| **5** | **Trace_id propagé E2E** : DataProvider → Telegram (cf. eval_16) | M | 0 | MTTR ⭐⭐⭐⭐⭐ | P1 |

### 5.1 Détail levier #1 (P0)

```python
import threading
from datetime import datetime, timedelta

class SentinelScanner:
    def __init__(self, ..., bar_aligned: bool = True):
        self._stop_event = threading.Event()
        self._bar_aligned = bar_aligned
        self._tf_minutes = TIMEFRAME_MINUTES[timeframe]  # 15

    def _run_loop(self):
        while not self._stop_event.is_set():
            try:
                self._scan_once()
            except Exception as e:
                self._errors += 1
                logger.error("Scanner error: %s", e, exc_info=True)

            # Sleep until next aligned bar (or poll_interval)
            if self._bar_aligned:
                next_bar = self._next_aligned_close()
                wait_s = max(1.0, (next_bar - datetime.utcnow()).total_seconds() + 5)
            else:
                wait_s = self._poll_interval
            self._stop_event.wait(timeout=wait_s)

    def shutdown(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
```

Aligne sur clock M15 → réveil 5 s après chaque bar close → latence pickup ≤ 5 s vs ~60 s.

### 5.2 Détail levier #2 (P0)

```python
# src/intelligence/notification_queue.py
class NotificationQueue:
    def __init__(self, db_path, ttl_minutes=15):
        self._db = sqlite3.connect(db_path)
        self._ttl = ttl_minutes * 60
        # CREATE TABLE pending_notifications (...)

    def enqueue(self, signal, narrative):
        self._db.execute("INSERT INTO ... (signal_id, payload, created_at) VALUES (...)")

    def replay(self, notifier_fn):
        cutoff = time.time() - self._ttl
        # SELECT WHERE created_at > cutoff
        for row in cursor:
            try:
                notifier_fn(row.payload)
                self._db.execute("DELETE FROM pending_notifications WHERE signal_id=?", row.id)
            except Exception:
                continue  # retry next replay cycle
```

Branché à `_send_notification_safe` : sur `CircuitOpenError`, enqueue. Sur `_notifier_breaker.on_state_change(CLOSED)`, appel `replay`.

---

## 6. Plan d'exécution

### Quick wins (≤ 4 h cumulées)
- `Event.wait` non-interruptible (1 h)
- Bar-aligned polling (1 h)
- Compteurs missing (`bars_quality_rejected`, `bars_skipped_same_ts`) (30 min)
- Latence histogram simple (1 h)
- `extra={'signal_id', 'symbol'}` dans tous les `logger.*` (30 min)

### Medium (1-3 jours)
- Notification queue Redis ou SQLite + replay
- MultiSymbolScanner unifié
- Agents thread-safe (RegimeAgent, NewsAgent)
- `_restore_state_machine` avec staleness check propre
- Prometheus histograms exposés `/metrics`

### Long term (1 sem)
- Event-driven WebSocket ingestion (cf. eval_08 §6)
- OpenTelemetry trace_id E2E
- Dead-letter queue + alerting Discord operator

---

## 7. KPIs cibles post-implémentation

| KPI | Avant | Après | Mesure |
|---|---|---|---|
| Latence pickup bar close | 0-60 s | **< 5 s** | bar_aligned polling |
| P95 `_scan_once` end-to-end | 9 s | 6 s | Prometheus histogram |
| Signaux perdus (notif failures) | inconnu | **< 1 %** | NotificationQueue replay |
| MTTR debug "signal manqué" | ∞ | < 5 min | trace_id E2E |
| Multi-symbol latency à 6 sym | 60 s | 12 s | unified MultiSymbolScanner |
| Disponibilité scanner | inconnu | ≥ 99 % | bars_scanned / expected |

---

## 8. Trade-offs

| Gain | Coût |
|---|---|
| Bar-aligned polling | léger code complexity, mais retire un bug latent du `time.sleep` |
| Notification queue | dépendance Redis ou SQLite, +tests E2E |
| Multi-symbol unifié | refacto agents (thread-safe), risque régression test suite |
| Latence histogram | +dépendance prometheus_client (déjà dans requirements) |
| Trace_id E2E | nécessite OpenTelemetry collector (eval_16 J+90) |

---

## 9. Note finale & recommandation

**Note : 6.5 / 10.**

Le scanner est **bien écrit dans sa structure** : try/except sectoriels, circuit breaker, fallback template, persistence state machine, validation OHLC. C'est un solid 6-7/10 pour un MVP solo. Les 3 points qui empêchent un 8/10 :

1. **Polling 60s fixe** → SLA "30s" inatteignable
2. **Mono-symbole de fait** (`MultiSymbolScanner` est vapor)
3. **Pas de queue notifications** → signaux silently dropped sur Telegram down

**Recommandation immédiate** :
- **P0 (4 h)** : event-driven `Event.wait` + bar-aligned polling → SLA 90 s atteint
- **P0 (1 j)** : notification queue + replay
- **P1 (2-3 j)** : multi-symbol unifié + thread-safe agents
- **P1 (1 sem)** : Prometheus histograms + OpenTelemetry trace_id E2E

Avant Stripe / Paid launch, **les 2 P0 sont obligatoires** sinon ticket support garanti à la première Telegram outage.

---

### Annexes
- Code source : `src/intelligence/sentinel_scanner.py` (910 l)
- Tests : `tests/test_sentinel_scanner.py`
- Persistence : `src/intelligence/state_persistence.py`
- Memory entries : `memory/state_persistence.md`, `memory/signal_state_machine.md`
- Eval amont : `reports/eval_07_signal_state_machine.md`, `reports/eval_08_data_providers.md`, `reports/eval_16_observability.md`
