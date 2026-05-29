# Eval 14 — Circuit Breaker (résilience LLM / Telegram)

**Date** : 2026-04-25
**Périmètre** : `src/intelligence/circuit_breaker.py` (231 l) — `CircuitBreaker` + `CircuitOpenError` + `HealthChecker`/`HealthStatus`.
**Verdict global** : **6.5/10** — implémentation thread-safe et solide du pattern à 3 états ; mais **3 manques structurels** : (1) pas de timeout sur `func()`, (2) pas de window-based failure rate (consecutive only), (3) pas de persistance d'état → flooding service au restart pendant outage.

---

## 1. Architecture

```
CircuitBreaker (dataclass, thread-safe via _lock)
   states: CLOSED ──[N consecutive failures]──▶ OPEN
           OPEN    ──[recovery_timeout elapsed]──▶ HALF_OPEN
           HALF_OPEN ──[K consecutive successes]──▶ CLOSED
                     ──[1 failure]──▶ OPEN

   call(func, fallback=None) → result
       acquires _lock to read state
       releases _lock before calling func
       _on_success / _on_failure manage state transitions

HealthChecker
   register(name, check_fn) → registry
   check() → HealthStatus(healthy: bool, checks: dict)
```

---

## 2. Audit ligne par ligne

| Aspect | Statut | Ligne | Note |
|---|---|---|---|
| 3 états explicites Enum | ✅ | 31-34 | CLOSED/OPEN/HALF_OPEN avec values lowercase |
| Thread-safety | ✅ | 73, 80 | `threading.Lock` + `with self._lock` autour de toute mutation |
| Lock release avant `func()` | ✅ | 109-111 | évite blocage long ; mais voir §3 |
| `_consecutive_failures` reset on success | ✅ | 120 | OK |
| HALF_OPEN→OPEN sur 1 fail | ✅ | 139-143 | bonne pratique : single failure during probe → reopen |
| HALF_OPEN→CLOSED sur K succès | ✅ | 125-129 | success_threshold paramétrable (défaut 2) |
| `recovery_timeout` après dernier failure | ✅ | 152-153 | mesure relative à `_last_failure_time` |
| `fallback` callable | ✅ | 105-107 | retourné si fournie quand circuit OPEN |
| Sliding history `deque(maxlen=...)` | ✅ | 72, 76 | bornée mémoire |
| `get_stats()` | ✅ | 163-179 | total_calls, total_failures, failure_rate, last_failure_age |
| `reset()` manuel | ✅ | 155-161 | OK pour ops |
| `__post_init__` re-init deque | ⚠️ | 75-76 | redondant : field default + post_init reset ; sans bug mais code smell |

---

## 3. Manques structurels

### 3.1 🔴 Aucun timeout sur `func()`

```python
# circuit_breaker.py:110-116
try:
    result = func()  # ← peut hang indéfiniment
    self._on_success()
    return result
except Exception as e:
    self._on_failure(e)
    raise
```

Si `func()` est un appel HTTP qui hang (réseau coupé, serveur LLM stuck) **sans timeout interne**, le circuit ne trippe pas — on attend indéfiniment. Le pattern Hystrix / Polly **inclut** un timeout dans le wrapper.

**Impact** : si Anthropic API hang à 60s, le circuit n'aide pas car aucune exception n'est levée. Le worker reste bloqué sur cet appel, signaux ratés en cascade.

**Fix** :
```python
@dataclass
class CircuitBreaker:
    timeout: Optional[float] = None  # ← nouveau param

    def call(self, func, fallback=None):
        ...
        if self.timeout is not None:
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    future = ex.submit(func)
                    result = future.result(timeout=self.timeout)
            except concurrent.futures.TimeoutError as e:
                self._on_failure(e)
                raise
        else:
            result = func()
        ...
```

Alternative plus propre : exiger que `func` soit déjà borné par l'appelant (Anthropic SDK timeout, requests timeout=...). Documenter explicitement.

### 3.2 🟠 Failure threshold = consecutive (pas window rate)

```python
# circuit_breaker.py:144-150
elif self._state == CircuitState.CLOSED:
    if self._consecutive_failures >= self.failure_threshold:
        self._state = CircuitState.OPEN
```

Un service qui échoue 50 % du temps (n=10 calls : F-S-F-S-F-S-F-S-F-S) **n'ouvrira jamais** le circuit avec threshold=3 consecutive. Pourtant, 50 % failure rate est inacceptable.

**Pattern recommandé** (Netflix Hystrix) : `failure_rate > 50 %` sur les **N derniers appels** (sliding window).

```python
def _failure_rate_window(self, window_size=20) -> float:
    recent = list(self._history)[-window_size:]
    if len(recent) < window_size:
        return 0.0  # pas assez d'historique
    failures = sum(1 for ev in recent if ev[0] == "failure")
    return failures / len(recent)

# in _on_failure / _on_success:
if self._failure_rate_window() > 0.5 and len(self._history) >= 20:
    self._state = CircuitState.OPEN
```

Garder `consecutive_failures` comme override (e.g. 5 consecutives = OPEN immediately).

### 3.3 🟠 Pas de persistance d'état

Au restart du process, `CircuitBreaker._state` revient à CLOSED. Si on était OPEN suite à outage Anthropic, **les premières requêtes après restart vont toutes échouer puis re-OPEN après threshold**. Pendant ce délai, on bombarde un service connu down.

**Fix** : sérialiser `_state`, `_consecutive_failures`, `_last_failure_time` vers `data/circuit_breakers.json` (atomic write) à chaque transition + reload au boot.

### 3.4 🟡 Pas de demi-ouverture concurrente protégée

En HALF_OPEN, **plusieurs threads concurrents** peuvent tous franchir le check :

```python
# circuit_breaker.py:99-102
if self._state == CircuitState.OPEN:
    if self._should_attempt_recovery():
        self._state = CircuitState.HALF_OPEN
```

Le `_lock` est tenu pendant la transition, mais **après** release, N threads exécutent tous `func()` en parallèle. Si le service est encore down, on flood encore.

**Fix Hystrix** : "single probe" — en HALF_OPEN, un seul thread autorisé à tenter `func()`, les autres reçoivent `CircuitOpenError` ou fallback.

```python
_probe_in_flight: bool = False

# in call():
if self._state == CircuitState.HALF_OPEN:
    if self._probe_in_flight:
        # Reject during probe
        if fallback: return fallback()
        raise CircuitOpenError(...)
    self._probe_in_flight = True
try:
    result = func()
finally:
    with self._lock:
        self._probe_in_flight = False
```

### 3.5 🟡 Pas de backoff exponentiel

`recovery_timeout` est fixe (60s par défaut pour LLM). Si l'outage dure 1h, on probe toutes les 60s = 60 probes. Si chacun timeout 30s, on consume 30 minutes de CPU/network gratuitement.

**Best-practice** : exponential backoff avec cap.
```python
def _next_recovery_timeout(self) -> float:
    n_recent_opens = sum(1 for ev in self._history[-20:] if ev[0] == "failure")
    return min(self.recovery_timeout * (2 ** min(n_recent_opens, 6)), 600)
```

### 3.6 🟡 Toutes les exceptions = failure

```python
# circuit_breaker.py:113-115
except Exception as e:
    self._on_failure(e)
    raise
```

Une `ValueError` dans le `func()` (bug de code, pas un service down) trippe le circuit comme un `TimeoutError`. Distinction utile :
```python
expected_exceptions: Tuple[type, ...] = (HTTPError, TimeoutError, ConnectionError)

except Exception as e:
    if isinstance(e, self.expected_exceptions):
        self._on_failure(e)
    raise
```

### 3.7 🟡 Pas de métriques Prometheus

`get_stats()` retourne dict, mais aucun export auto vers `MetricsRegistry`. Idéalement :
- Counter `circuit_breaker_calls_total{name, outcome}`
- Counter `circuit_breaker_state_transitions_total{name, from, to}`
- Gauge `circuit_breaker_state{name}` (0=CLOSED, 1=HALF_OPEN, 2=OPEN)
- Histogram `circuit_breaker_call_duration_seconds{name}`

---

## 4. HealthChecker — duplication

```python
# circuit_breaker.py:199-231
class HealthChecker:
    def check(self) -> HealthStatus: ...
```

```python
# routes/health.py:46-53
health_checker = getattr(app_state, "health_checker", None)
if health_checker is not None:
    result = health_checker.check_all()  # ← attend `check_all`, pas `check`
```

**Mismatch** : `HealthChecker` expose `check()` mais le code de la route appelle `check_all()`. Soit deux classes coexistent (probable : il y a `src/performance/health.py` mentionné en grep §3), soit bug latent.

À vérifier : `src/performance/health.py` qui a une classe `HealthMonitor` (méthode `check_all`). Probablement confusion entre deux instances. Recommandation : un seul `HealthChecker` dans le repo, avec une seule API.

---

## 5. Wiring & usage actuel

D'après `MEMORY.md` et `routes/narratives.py:135-140` :

```python
# narratives.py:133-140
llm_breaker = app_state.circuit_breakers.get("llm")
if llm_breaker is not None:
    if llm_breaker.state == CircuitState.OPEN:
        raise HTTPException(status_code=503, detail="...")
```

⚠️ Ici on **lit l'état** sans appeler `call()`. C'est un check de gating. Mais l'incrément des compteurs n'a lieu que via `call()` — il faut s'assurer que TOUS les appels Anthropic SDK passent par `breaker.call(func)`.

À vérifier dans `llm_narrative_engine.py` : grep `circuit_breaker` ou `breaker.call`.

---

## 6. Chaos engineering — test plan

| Scénario | Comportement attendu | État actuel |
|---|---|---|
| Anthropic API renvoie 500 N fois | Circuit OPEN après threshold, fallback template | ⚠️ doit valider que `call()` est utilisé partout |
| Anthropic API hang (no timeout) | Circuit OPEN après timeout interne | ❌ pas de timeout (cf. §3.1) |
| Anthropic OK puis 50% failure rate | Circuit OPEN | ❌ ne trippe pas (cf. §3.2) |
| Restart pendant OPEN | Reload OPEN état | ❌ reset à CLOSED |
| Recovery probe pendant outage long | Backoff exponentiel | ❌ fixed timeout |
| 100 threads concurrents en HALF_OPEN | 1 probe, 99 fallback | ❌ tous probent |
| Telegram banned → fallback Discord | Fallback channel | n/a (cf. eval_13) |

**Test à ajouter** (chaos engineering) :
```python
# tests/test_circuit_breaker_chaos.py
def test_half_open_concurrent_probes():
    cb = CircuitBreaker(name="test", failure_threshold=1, recovery_timeout=0.1)
    cb._state = CircuitState.OPEN
    cb._last_failure_time = time.time() - 1.0  # eligible for HALF_OPEN

    failures = 0
    def slow_failing(): time.sleep(0.5); raise ConnectionError()

    threads = [Thread(target=lambda: cb.call(slow_failing)) for _ in range(10)]
    for t in threads: t.start()
    for t in threads: t.join()
    # Expected: only 1 probe; current: 10 probes
    assert cb._total_failures < 10  # WILL FAIL with current impl
```

---

## 7. Comparaison libs externes

| Lib | Pros | Cons | Verdict |
|---|---|---|---|
| `circuitbreaker` (PyPI, ~3M dl/mo) | léger, decorator-based | pas de fallback, pas de window | OK pour cas simples |
| `pybreaker` | bien testée, listeners, redis storage | moins maintenu | OK mais idle |
| `aiobreaker` | async natif | jeune | si on passe async |
| `polly` (.NET) | ref. | pas Python | n/a |
| **DIY (actuel)** | adapté, lisible | manques §3 | OK si on bouche les manques |

**Recommandation** : garder le DIY (intégration tight avec MetricsRegistry, simple à maintenir) mais corriger §3.1, §3.2, §3.3.

---

## 8. Top 5 améliorations priorisées

| # | Amélioration | Effort | Impact |
|---|---|---|---|
| **R1** | **Timeout intégré dans `call()`** (paramètre + ThreadPoolExecutor wrapper) | 0.5 jour | 🔴 sans cela, hang LLM = signaux perdus |
| **R2** | **Window-based failure rate** en plus de consecutive (50 % over 20 calls) | 1 jour | 🟠 protège contre flaky services |
| **R3** | **Persistance état** JSON atomic + reload boot | 0.5 jour | 🟠 évite flood au restart pendant outage |
| **R4** | **HALF_OPEN single-probe lock** + backoff exponentiel | 1 jour | 🟡 évite flood lors recovery |
| **R5** | **Métriques Prometheus** (state, transitions, duration histogram) + alerting Grafana si OPEN > 5 min | 1 jour | 🟡 observabilité ops |

**Matrice** :

```
Impact ↑
  5 |  R1
  4 |        R2   R3
  3 |              R4   R5
  2 |
    +-------------------→ Effort
       1   2   3   4   5
```

---

## 9. Plan d'exécution

### Quick wins (< 1 jour)
- **QW1** Timeout param + wrapping ThreadPoolExecutor (3 h)
- **QW2** `expected_exceptions` tuple paramètre (1 h)
- **QW3** Tests chaos : `test_call_hangs_no_timeout`, `test_50pct_failure_rate_no_trip` (2 h)
- **QW4** Persistence JSON atomic write + reload (3 h)
- **QW5** Backoff exponentiel sur `recovery_timeout` capped à 600s (1 h)
- **QW6** Unifier HealthChecker : supprimer dupli `src/performance/health.py` HealthMonitor OR exposer un seul API `check_all` (2 h)

### Moyen terme (< 1 semaine)
- **MT1** Window-based failure rate avec sliding window 20 calls (4 h)
- **MT2** HALF_OPEN single-probe avec `_probe_in_flight` flag (2 h)
- **MT3** MetricsRegistry export auto (counter, gauge, histogram) (4 h)
- **MT4** Alertmanager rule : `circuit_breaker_state{name="llm"} == 2` for > 5 min → page (1 h)
- **MT5** Multi-provider fallback : OpenAI as backup if Anthropic OPEN (`call(anthropic_fn, fallback=lambda: openai_fn())`) (4 h)
- **MT6** Test injection de fautes : pytest fixture qui mock Anthropic 500/timeout/hang (4 h)
- **MT7** SLA dashboard : uptime % par service, MTTR, time-in-OPEN (4 h)

### Long terme (> 1 semaine)
- **LT1** Adaptive thresholds (auto-tune failure_threshold based on historical p99 failures)
- **LT2** Bulkhead pattern : limiter le nombre de calls concurrents par service
- **LT3** Distributed circuit breaker (Redis state share) si on passe multi-worker / multi-instance
- **LT4** Health check synthetic : appeler Anthropic / Telegram périodiquement pour pré-trip avant impact user

---

## 10. KPIs mesurables post-amélioration

| KPI | Baseline | 30 j | 90 j |
|---|---|---|---|
| LLM call timeout coverage | 0 % | 100 % | 100 % |
| Window-based failure rate trip | non | oui (>50%/20) | oui |
| State persistence | non | oui | oui (Redis si distribué) |
| Half-open probe protected | non | oui | oui |
| Mean Time To Detect (MTTD) failure | inconnu | < 30 s | < 10 s |
| Mean Time To Recover (MTTR) | inconnu | < 5 min | < 2 min |
| % outage avec fallback served | inconnu | > 95 % | > 99 % |
| % uptime LLM (% closed) | inconnu | > 99 % | > 99.5 % |
| Prometheus metrics exposed | 0 | 4 (calls/transitions/state/duration) | 6 |
| Multi-provider fallback (OpenAI backup) | non | possible | actif |
| Chaos test coverage | 0 | 5 scenarios | 10 scenarios |

---

## 11. Trade-offs assumés

- **R1 timeout via ThreadPoolExecutor** : ajoute overhead (~0.1 ms par call) et complexité — alternative : exiger timeout SDK natif. Recommandé : combo (timeout SDK + wrapper safety net).
- **R2 window rate** : nécessite plus d'historique avant trip → délai de réaction +15 s typique. Acceptable face au gain de précision.
- **R3 persistance** : write JSON sur disk à chaque transition (~rare) ; négligeable. Multi-worker → besoin Redis (LT3).
- **R4 single-probe** : un thread bloque les autres pendant probe → si timeout est généreux, ralentit recovery. Mitiger avec timeout court de probe.
- **R5 Prometheus** : ajoute dépendance MetricsRegistry partout — déjà câblé dans le projet, marginal.

---

## 12. Note finale par dimension

| Dimension | Note /10 | Justification |
|---|---|---|
| Pattern correctness | 8 | 3-état canonique, transitions correctes |
| Thread safety | 9 | _lock partout sur mutations |
| Timeout protection | 0 | absent |
| Failure detection finesse | 5 | consecutive only, pas window |
| Resilience au restart | 3 | pas de persistance |
| HALF_OPEN safety | 5 | pas de single-probe lock |
| Observabilité | 5 | get_stats() ✅ ; pas d'export Prometheus |
| Configurability | 7 | failure_threshold, recovery_timeout, success_threshold paramétrables ; manque expected_exceptions, timeout, window_size |
| Multi-provider fallback | 4 | fallback callable supporté, mais pas de pattern multi-provider documenté |
| Chaos test coverage | ? (à vérifier eval_17) | tests/test_circuit_breaker.py existe |
| **Global** | **6.5/10** | **Bon pattern, manque les "+1 % qui font la différence" en prod** |

---

## 13. Verdict

- **Garder** : architecture 3-état avec dataclass, _lock pattern, sliding deque history, reset() manuel.
- **Compléter** : R1-R3 sont des deltas faibles à moyen effort, gain énorme en résilience.
- **Décision multi-provider** : avant de coder un OpenAI fallback (LT5 → MT5), valider qu'Anthropic est suffisamment fiable (regarder uptime status.anthropic.com). Si > 99.9 % historique, fallback inutile ; investir le temps ailleurs.

---

## Annexe — fichiers et lignes critiques

- `src/intelligence/circuit_breaker.py:83-116` `call()` sans timeout
- `src/intelligence/circuit_breaker.py:144-150` consecutive-only threshold
- `src/intelligence/circuit_breaker.py:152-153` fixed recovery_timeout
- `src/intelligence/circuit_breaker.py:99-102` HALF_OPEN concurrent probes
- `src/intelligence/circuit_breaker.py:113-115` toutes Exception = failure
- `src/api/routes/narratives.py:133-140` consume `state` direct (pas via call())
- `src/api/routes/health.py:46-53` `check_all` — vérifier alignement avec `check`

## Annexe — chaos test à ajouter

```python
# tests/test_circuit_breaker_chaos.py
import time, threading
from src.intelligence.circuit_breaker import CircuitBreaker, CircuitState

def test_hanging_call_does_not_trip_without_timeout():
    """Without timeout param, a hang call never triggers _on_failure."""
    cb = CircuitBreaker(name="t", failure_threshold=2, recovery_timeout=1.0)
    def hang(): time.sleep(120)  # would hang the test
    # Must wrap in thread + assert it returns within X seconds OR mark expected fail
    t = threading.Thread(target=lambda: cb.call(hang), daemon=True)
    t.start()
    t.join(timeout=3.0)
    assert t.is_alive()  # CURRENT: still hanging — circuit didn't help

def test_50pct_failure_rate_does_not_trip():
    cb = CircuitBreaker(name="t", failure_threshold=3, recovery_timeout=1.0)
    for i in range(20):
        try:
            cb.call(lambda: (_ for _ in ()).throw(ConnectionError()) if i % 2 == 0
                    else "ok")
        except Exception:
            pass
    # 50% failures, but not 3 consecutive → CLOSED still
    assert cb.state == CircuitState.CLOSED  # documents current behavior
```
