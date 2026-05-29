# Eval 12 — Signal Store (SQLite, persistance, backup)

**Date** : 2026-04-25
**Périmètre** : `src/api/signal_store.py` (300 l), `src/api/signal_tracker.py` (202 l).
**Verdict global** : **5.5/10** — schéma versionné propre, WAL bien configuré, mais 4 défauts qui empêchent de s'appuyer sur ce store comme « preuve track record » : (1) bug de désérialisation v3, (2) Sharpe mathématiquement incorrect, (3) pas de backup, (4) `_current` mémoire non sync entre workers.

---

## 1. Architecture

```
SignalStore (writer + reader, in-memory cache + WAL SQLite)
    │
    ├── publish(SignalRecord)  ← scanner / pipeline
    ├── get_current()           ← API lecteur (cache RAM, **non sync entre workers**)
    ├── get_history(page, page_size)   ← API lecteur (table scan COUNT(*))
    ├── get_by_id(id)           ← API lecteur (PK index)
    └── update_outcome(id, ...) ← signal_tracker post-trade

SignalTracker (read-only, ouvre sa propre connexion)
    │
    ├── get_performance_summary(days)   ← Sharpe, PF, win_rate, max DD
    └── get_equity_curve(days)
```

---

## 2. Schema SQLite — audit

| Item | Statut | Note |
|---|---|---|
| `schema_version` table | ✅ | versioned ; migrations idempotentes |
| Migrations v1 → v3 | ✅ pattern | `ALTER TABLE … ADD COLUMN` avec try/except OperationalError |
| Index `idx_signals_created` (DESC implicite via ORDER) | ✅ | suffisant pour `get_history` |
| Index sur `outcome` ou `closed_at` | ❌ | `signal_tracker.get_performance_summary` filtre `WHERE outcome IS NOT NULL AND closed_at >= …` — **table scan** sur N signaux |
| Foreign keys | n/a | pas de table satellite |
| JSON columns (`vol_confidence`) | ⚠️ | colonne `TEXT` qui contient JSON non parsé ; sérialisation côté `publish` non visible (probable string raw passé) |
| Constraints (CHECK, NOT NULL) | ⚠️ | seuls `entry_price`/`stop_loss`/`take_profit`/`rr_ratio` NOT NULL ; pas de `CHECK (rr_ratio > 0)` ni `CHECK (action IN (...))` |
| WAL mode | ✅ | `journal_mode=WAL`, `synchronous=NORMAL` ; bon compromis durabilité/perf |
| Connection-per-call | ✅ | thread-safe, pas de connexion partagée |
| `timeout=30.0` | ✅ | backoff sur SQLITE_BUSY |
| `isolation_level=None` | ✅ | autocommit, simplifie code |

---

## 3. Bugs identifiés

### 3.1 🔴 Bug de désérialisation v3

```python
# signal_store.py:247-275
@staticmethod
def _row_to_record(r: Any) -> SignalRecord:
    return SignalRecord(
        ...
        confluence_score=_get("confluence_score"),
        narrative=_get("narrative"),
        validation_reason=_get("validation_reason"),
        key_confluences=_get("key_confluences"),
        risk_warnings=_get("risk_warnings"),
        market_context=_get("market_context"),
        # ❌ vol_forecast_atr, vol_regime, vol_confidence MANQUENT
    )
```

Les colonnes ajoutées en migration v3 (`vol_forecast_atr`, `vol_regime`, `vol_confidence`) sont **écrites par `publish()`** (ligne 174-196) mais **jamais lues** par `_row_to_record`. Conséquence :

- `/api/v1/signals/history` renvoie `vol_*` = `None` même si en DB.
- `/api/v1/narratives/{id}` même problème (mais `NarrativeResponse` a aussi ces champs en `Optional[float] = None`, donc silencieusement ignoré).
- Bug muet : aucun test ne le détecte (les vol fields sont Optional).

**Fix trivial** : ajouter 3 lignes `vol_forecast_atr=_get("vol_forecast_atr"), …`.

### 3.2 🔴 Sharpe mathématiquement incorrect

```python
# signal_tracker.py:163-179
@staticmethod
def _compute_sharpe(pnls: List[float]) -> float:
    n = len(pnls)
    if n < 2: return 0.0
    mean = sum(pnls) / n
    variance = sum((p - mean) ** 2 for p in pnls) / (n - 1)
    std = math.sqrt(variance) if variance > 0 else 0.0
    if std == 0: return 0.0
    daily_rf = RISK_FREE_RATE / TRADING_DAYS_YEAR
    sharpe = (mean - daily_rf) / std
    return sharpe * math.sqrt(TRADING_DAYS_YEAR)
```

**Problème** : `pnls` = `pnl_pips` par **trade** (variable nombre de trades par jour, parfois 0, parfois 5). L'annualisation `× sqrt(252)` suppose des **returns journaliers indépendants identiquement distribués**. Avec n=20 trades sur 30 jours, le Sharpe est :
- Sur-estimé si le freq des trades est faible (variance sous-estimée car pas de jours zéro inclus).
- Mal calibré : `pnl_pips` n'est pas un return en %, c'est une distance prix. RISK_FREE_RATE est annualisé en % → la soustraction `mean_pips - daily_rf_pct` mélange unités.

**Référence** : Lo (2002) "The Statistics of Sharpe Ratios" — annualisation correcte requiert returns daily continus, pas trade-level. Sinon utiliser **Sortino sur trade returns** ou **Calmar (CAGR / max DD)**.

**Impact business** : un Sharpe affiché sur landing page est **non auditable**. Risque marketing-fraude.

### 3.3 🟠 Profit factor sentinel 999.99

```python
# signal_tracker.py:120
"profit_factor": round(profit_factor, 4) if profit_factor != float("inf") else 999.99,
```

Si `gross_loss = 0` (uniquement des wins), `profit_factor = inf` → forcé à 999.99. UI affichera "999.99" suspicieux. Mieux : retourner `None` ou un flag `infinite_pf: bool`.

### 3.4 🟡 Max DD relatif au peak cumulatif PnL pips

```python
# signal_tracker.py:181-202
peak = 0.0  # ❌ initial peak = 0, donc DD avant 1er win = 0
max_dd = peak - cumulative  # absolute, en pips
return round((max_dd / peak) * 100, 4)
```

Problèmes :
1. Peak initial à 0 : si la séquence commence par 5 SL, le DD est compté comme `peak - cumulative = 0 - (-50) = 50`, mais `peak <= 0` → return 0. **DD prétrade catastrophique = 0 reporté**.
2. Dénominateur = `peak` cumulatif PnL pips, pas l'**equity de départ**. Pour un compte à $10k, perdre 50 pips XAU n'est pas le même % que pour $1k.

**Fix** : exposer DD en pips ET en % d'un equity de référence configurable (`STARTING_EQUITY` env var).

### 3.5 🟡 `_current` cache RAM non sync entre workers

```python
# signal_store.py:71
self._current: Optional[SignalRecord] = None
```

Avec `uvicorn --workers 4` :
- Worker 1 reçoit `publish()` du scanner → met à jour son `_current`.
- Worker 2 reçoit `GET /signals/current` → renvoie son `_current` (ancien ou None).

**Réponses incohérentes selon le worker hit**. Mitigation actuelle : single-process ; mais bloquant pour scaling.

### 3.6 🟡 `update_outcome` ne refresh pas `_current` complètement

```python
# signal_store.py:294-297
if self._current and self._current.signal_id == signal_id:
    self._current.outcome = outcome
    self._current.pnl_pips = pnl_pips
    self._current.closed_at = closed_at
```

Update partiel des 3 champs. OK pour l'usage actuel mais fragile si on ajoute un champ post-trade.

---

## 4. Performance

### 4.1 Profil opérationnel

| Op | Cost SQL | Volume attendu / jour | Goulot ? |
|---|---|---|---|
| `publish()` | INSERT OR REPLACE | 1-50 (selon scanner) | non |
| `get_current()` | 0 (RAM) | 1-100/min API polls | non |
| `get_history(20)` | COUNT(*) + SELECT … LIMIT 20 | 10-100/jour (UI) | ⚠️ COUNT(*) sur table |
| `get_by_id` | SELECT WHERE PK | 100-1000/jour | non (index PK) |
| `update_outcome` | UPDATE | 1-50/jour | non |
| `get_performance_summary` | SELECT … WHERE outcome AND closed_at | 10-100/jour | ⚠️ pas d'index |
| `get_equity_curve(90)` | SELECT … ORDER BY closed_at | 10-100/jour | ⚠️ pas d'index |

**Optimisations index** :
```sql
CREATE INDEX idx_signals_outcome_closed
ON signals(outcome, closed_at) WHERE outcome IS NOT NULL;
```
Index partiel SQLite supporté → range scan O(log n) sur jours.

### 4.2 Concurrent writes

WAL permet 1 writer + N readers. Avec `signal_store.publish()` + `update_outcome()` + `record_usage()` (KeyStore) + `usage_log` (TierManager) tous en SQLite **différents fichiers** (sig.db, api_keys.db, users.db) — **pas de contention inter-store**, ✅ design propre. Mais :

- Au sein de `signals.db`, `publish()` du scanner concurrent avec `update_outcome()` du tracker → serialize via WAL writer lock ; OK pour 1-50 ops/min.
- Si on monte à 10k abonnés × ping `/dashboard/summary` toutes les 30 s → 333 RPS lectures sur `signal_tracker` → table scan sans index = bottleneck (cf. 4.1).

---

## 5. Multi-tenant readiness

| Dimension | Statut | Note |
|---|---|---|
| Isolation par utilisateur | ❌ | Tous les signaux mondiaux ; pas de `user_id` FK |
| Watchlist personnalisée | ❌ | Implique table `user_signal_subscriptions` + filtre sur publish |
| Signaux history filtré par tier | ❌ | Tier FREE voit le même history que INSTITUTIONAL |
| Export CSV / Parquet par utilisateur | ❌ | Aucune route `/dashboard/export` |
| Pagination cursor-based | ❌ | OFFSET-based (perf O(N) sur grandes pages) |
| **Concurrent writes** > 100/s | ❌ | SQLite WAL ne tient pas (LiteFS / Postgres requis) |

---

## 6. Backup & rétention

**État actuel** : aucun backup, aucune rétention.

| Risque | Statut | Conséquence |
|---|---|---|
| Disk failure | 🔴 | Perte totale du track record commercial |
| Bad migration | 🔴 | Pas de snapshot pre-migration |
| `DELETE FROM signals` accidentel | 🔴 | Aucun PITR (point-in-time recovery) |
| Croissance illimitée | 🟠 | Estimation : 50 signaux/j × 6 symbols × 7 ans = **~767k rows** ; ~1KB/row → ~750 MB ; OK pour SQLite mais demande backup |
| RPO / RTO | inconnu | Pas de policy documentée |

**Solution simple (Linux/Docker)** :
- `sqlite3 signals.db ".backup data/backups/signals-$(date +%F).db"` quotidien (atomic via SQLite Online Backup API).
- Push S3 (rclone / AWS CLI) → 30 j retention + Glacier 1 an.
- Restore test mensuel (smoke test `pytest tests/test_backup_restore.py`).

**Solution propre (Postgres)** :
- WAL archiving + base backups (pgBackRest).
- Lecture replica sur Render/Neon.

---

## 7. Migration Postgres — analyse coût/bénéfice

| Critère | SQLite (actuel) | Postgres (Neon/Supabase) | Verdict |
|---|---|---|---|
| Coût hosting | $0 (volume Railway) | $0-19/mo (Neon free 0.5GB / pro $19) | OK pour ≤ 10k signaux/mo |
| Concurrent writes | 1 | 100+ | Postgres si scaling |
| Backup PITR | manuel | natif | Postgres |
| Replication multi-region | non | oui (Neon read replicas) | Postgres pour SaaS prod |
| JSON columns | partiel (TEXT) | `JSONB` natif + index GIN | Postgres |
| Effort migration | n/a | ~1-2 jours (SQLAlchemy + Alembic) | acceptable |
| Tests à mettre à jour | 0 | tous les tests qui hit DB | non-trivial |
| Lock-in | aucun | Neon-specific features (branching) | Postgres standard |

**Recommandation** : rester SQLite tant que < 100 signaux/jour total ET single-process scanner. Migrer à Postgres avant d'accepter des paiements (J0 du go-live commercial).

---

## 8. Top 5 améliorations priorisées

| # | Amélioration | Effort | Impact |
|---|---|---|---|
| **R1** | **Fix `_row_to_record` v3** : ajouter `vol_forecast_atr`, `vol_regime`, `vol_confidence` | 5 min | 🟠 dévoile data déjà persistée ; corrige bug muet |
| **R2** | **Backup quotidien S3** + restore test mensuel | 0.5 jour | 🔴 blocant pour go-live (track record = revenu) |
| **R3** | **Refondre Sharpe et max-DD** : Sharpe sur returns daily (resampler signals → daily P&L), DD en pips ET % d'equity configurable | 1 jour | 🟠 crédibilité marketing |
| **R4** | **Index partiel `idx_signals_outcome_closed`** + benchmark avant/après | 30 min | 🟡 perf 5-10× sur dashboard endpoints |
| **R5** | **Migration Postgres** + Alembic + tests | 2 jours | 🟠 pre-requis go-live > 100 abonnés |

**Matrice** :

```
Impact ↑
  5 |   R2
  4 |       R3   R5
  3 |   R1
  2 |       R4
  1 |
    +-------------------→ Effort
       1   2   3   4
```

---

## 9. Plan d'exécution

### Quick wins (< 1 jour)
- **QW1** Fix `_row_to_record` (5 min) + test régression
- **QW2** Index partiel `idx_signals_outcome_closed` + Migration v4 (30 min)
- **QW3** Profit factor : retourner `None` si gross_loss=0, retirer 999.99 sentinel (15 min)
- **QW4** Backup quotidien : script `scripts/backup_sqlite.py` + cron Railway (1 h)
- **QW5** `__post_init__` validation dataclass : `assert action in (HOLD, OPEN_LONG, …)`, `assert rr_ratio > 0` (30 min)
- **QW6** Constraints SQL : `CHECK (action IN ('HOLD',...))` via migration v4 (15 min)

### Moyen terme (< 1 semaine)
- **MT1** Refondre `_compute_sharpe` : resample par jour calendaire UTC, returns daily, annualisation correcte (4 h)
- **MT2** Max DD en % d'equity de référence + DD pips séparé (2 h)
- **MT3** Sortino + Calmar en plus du Sharpe (1 h)
- **MT4** Export CSV/Parquet : route `/dashboard/export?format=csv&days=90` (4 h)
- **MT5** Cursor-based pagination history (`?after=signal_id`) en plus de page-based (3 h)
- **MT6** Restore test automatisé (`pytest tests/test_backup_restore.py`) en CI (2 h)

### Long terme (> 1 semaine)
- **LT1** Migration Postgres + SQLAlchemy + Alembic (2 jours)
- **LT2** Read replica Postgres pour endpoints lecture (`/dashboard/*`)
- **LT3** Multi-tenant signals (table `user_signal_view`) avec watchlists
- **LT4** Audit log immutable (append-only table avec hash chain) pour preuve track record légale
- **LT5** Streaming export (websocket / SSE) pour dashboards live

---

## 10. KPIs mesurables post-amélioration

| KPI | Baseline | 30 j | 90 j |
|---|---|---|---|
| Bug désérialisation v3 (champs vol) | présent | corrigé | corrigé |
| RPO (data loss potentielle) | total | 24 h | 1 h (PITR Postgres) |
| RTO (temps restoration) | inconnu | < 1 h | < 15 min |
| P95 latence `/dashboard/summary` | ? | -50 % (index) | -80 % (Postgres) |
| Sharpe correctement annualisé | non | oui | oui |
| Max DD % equity | non exposé | exposé | exposé |
| Concurrent writers supportés | 1 | 1 | 100+ (Postgres) |
| Export CSV INSTITUTIONAL | non | oui | oui |
| Coverage `signal_tracker.py` | ? | ≥ 90 % | ≥ 95 % |
| Backup test mensuel green | n/a | oui | oui |

---

## 11. Trade-offs assumés

- **Postgres migration** ajoute coût ($19/mo Neon) et complexité (Alembic, network latency) → mais payback dès 50 abonnés payants.
- **Sharpe daily-resampled** masque le détail trade-by-trade ; conserver les deux exposés.
- **Index partiel** accélère lectures mais ralentit `update_outcome` de quelques % (négligeable).
- **Cursor pagination** breaking change pour clients existants → versionner `/api/v2/signals/history`.
- **Backup quotidien** sur Railway nécessite un cron worker séparé (~$1/mo) ou GitHub Actions cron-based.

---

## 12. Note finale par dimension

| Dimension | Note /10 | Justification |
|---|---|---|
| Schéma & migrations | 8 | versioned, idempotent, additive ; manque CHECK constraints |
| Thread safety | 8 | RLock + connection-per-call ; mais `_current` non sync inter-workers |
| Performance | 6 | WAL ✅ ; mais index manquants, COUNT(*) non optim |
| Sérialisation correcte | 4 | Bug v3 fields manquants |
| Métriques perf (Sharpe, DD) | 3 | Mathématiquement incorrects |
| Backup / DR | 1 | Aucun |
| Multi-tenant | 3 | Pas de FK user_id, single-process `_current` |
| Export commercial | 2 | Aucun export CSV/Parquet |
| Postgres-readiness | 5 | Code propre, migration faisable mais non commencée |
| **Global** | **5.5/10** | **Adapté à test perso ; non commercialisable sans R2-R3-R5** |

---

## 13. Verdict

- **Garder** : SQLite WAL pattern, schema versioning, dataclass `SignalRecord`.
- **Corriger immédiatement** : R1 (5 min), R3 Sharpe (1 j), R4 index (30 min).
- **Avant go-live commercial** : R2 (backup) + R5 (Postgres) sont **non-négociables**.

---

## Annexe — fichiers et lignes critiques

- `src/api/signal_store.py:140-150` migration v3 (colonnes vol_*)
- `src/api/signal_store.py:174-196` `publish()` écrit vol_*
- `src/api/signal_store.py:247-275` `_row_to_record` **ne lit pas vol_*** ← R1
- `src/api/signal_tracker.py:73-77` query sans index (R4)
- `src/api/signal_tracker.py:163-179` Sharpe incorrect (R3)
- `src/api/signal_tracker.py:181-202` max DD initial peak=0 bug (R3)
- `src/api/signal_store.py:71,201-204` `_current` mémoire non sync (LT1 / R5)

## Annexe — script de reproduction

```python
# tests/test_signal_store_v3_deserialization.py
def test_publish_then_get_by_id_includes_vol_fields(tmp_path):
    store = SignalStore(db_path=str(tmp_path/"sig.db"))
    rec = SignalRecord(
        signal_id="abc12345", action="OPEN_LONG", symbol="XAUUSD",
        entry_price=2400.0, stop_loss=2390.0, take_profit=2420.0, rr_ratio=2.0,
        created_at="2026-04-25T00:00:00", vol_forecast_atr=12.5,
        vol_regime="high", vol_confidence='{"lower":10,"upper":15}',
    )
    store.publish(rec)
    fetched = store.get_by_id("abc12345")
    assert fetched.vol_forecast_atr == 12.5  # FAILS with current code
    assert fetched.vol_regime == "high"
```
