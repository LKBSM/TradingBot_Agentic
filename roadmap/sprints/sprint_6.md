# Sprint 6 — Production Hardening

**Période** : Semaines 13-14 (S13-S14, ~2026-08-08 → 2026-08-22)
**Charge estimée totale** : **100 h** productives + buffer 14 h = 114 h
**Objectif** : transformer le système de "code R&D qui marche" vers "production-grade institutionnel". Profiling + vectorisation latence < 250 ms/tick (eval_21 dit non-scalable), replay engine reproductible à la nanoseconde, snapshot store API per-signal (P0-16), versioning modèles + compatibilité ascendante, et l'**extraction physique du package `smart_money/`** promise Sprint 1 (P0-9 résiduel).
**Gate de sortie** : latence p99 ≤ 250 ms/tick, replay bit-à-bit reproductible, snapshot store retrieval < 50 ms, model registry actif, smart_money/ extraction complète sans régression.

---

## 0. Vue d'ensemble — 5 batches

| Batch | Titre                                                        | Heures | Critique chemin |
| ----- | ------------------------------------------------------------ | ------ | --------------- |
| 6.1   | Profiling + vectorisation latence < 250 ms/tick              | 28 h   | ✅              |
| 6.2   | Replay engine reproductible nanoseconde                      | 18 h   | ✅              |
| 6.3   | Snapshot store API per-signal                                | 20 h   | ✅              |
| 6.4   | Versioning modèles + compatibilité ascendante                | 16 h   | ✅              |
| 6.5   | Extraction physique `smart_money/` (P0-9 résiduel)          | 18 h   | ✅              |
| —     | Buffer (debug perf, refactor risk)                           | 14 h   |                 |
| **TOTAL** |                                                          | **114 h** |             |

---

## Batch 6.1 — Profiling + vectorisation latence < 250 ms/tick (28 h)

### Objectif
Le pipeline actuel est non-scalable > 1k MAU (eval_21). Polling 60s, sync I/O, single-worker. Cible eval_21 : < 250 ms par tick par paire. Profiling avec `py-spy` + `cProfile`, vectorisation NumPy/Pandas, async I/O critiques.

### Steps
1. **Profiling baseline** (5 h)
   - `py-spy record -o profile.svg --pid <scanner_pid> -d 60` sur scanner prod local.
   - `cProfile` sur run synthétique 10 000 ticks.
   - Sortie : `reports/sprint_6/profile_baseline.svg` + `profile_baseline_stats.txt`.

2. **Identification hotspots** (3 h)
   - Top 10 functions par cumulative time.
   - Attendu (eval_21) :
     - `strategy_features.compute_atr` (row-by-row, vectorisable).
     - `SmartMoneyEngine.detect_*` (loops Python).
     - `volatility_forecaster.forecast` (HMM predict 1-row).
     - `ConfluenceDetector.score` (dict ops).

3. **Vectorisation `strategy_features`** (6 h)
   - ATR : rolling window NumPy.
   - RSI : `pandas.Series.diff` + `ewm`.
   - MACD : `pandas.Series.ewm`.
   - Speedup attendu : 5-20×.

4. **Vectorisation SmartMoneyEngine** (5 h)
   - BOS detection : NumPy stride tricks.
   - OB detection : pandas `rolling.apply` vectorisé.
   - Speedup attendu : 3-10×.

5. **Async I/O critiques** (4 h)
   - `aiohttp` pour Telegram + LLM.
   - `asyncio.gather` pour parallel multi-symbol.
   - Speedup attendu : I/O bound -50 % latence.

6. **Benchmark final** (3 h)
   - `scripts/benchmark_tick_latency.py` :
     - 10 000 ticks synthétiques.
     - Mesure p50, p95, p99 latence.
   - Cible : p99 ≤ 250 ms.

7. **Rapport** (2 h)
   - `reports/sprint_6/perf_report.md` :
     - Latence avant/après.
     - Hotspots résiduels.
     - Recommandations Sprint 7+ (Cython, Numba).

### Critères d'acceptation
- ✅ p99 latence ≤ 250 ms/tick/symbol.
- ✅ Profile baseline + post-optim générés.
- ✅ Speedup global ≥ 3×.

### Findings audit adressés
- **eval_21** (non-scalable > 1k MAU) — partiel ✅.
- Renforce préparation commercialisation.

### Dépendances
- Aucune externe au sprint.

### Risques
- Vectorisation peut subtilement changer résultats (float precision). Mitigation : tests régression bit-à-bit pré/post.

---

## Batch 6.2 — Replay engine reproductible nanoseconde (18 h)

### Objectif
Garantir que tout backtest est rejouable bit-à-bit à 6 mois d'intervalle. Précurseur snapshot store (batch 6.3) et tear sheets (Sprint 7).

### Steps
1. **Audit non-determinism sources** (3 h)
   - `uuid.uuid4()` (fixé Sprint 3 batch 3.4).
   - `set()`/`dict` iter order (Python 3.7+ ordered, OK).
   - `time.time()` calls hors snapshot.
   - Threading non-determinism (sentinel polling).
   - `np.random` sans seed.

2. **Fix remaining** (4 h)
   - Audit + fix toute source identifiée.
   - Seed global `SEED=42` propagé.

3. **Timestamp standardization** (3 h)
   - Tout timestamp interne : `pd.Timestamp` UTC nanosecond.
   - Conversion explicite à I/O boundaries.

4. **Replay harness amélioré** (4 h)
   - `src/backtest/replay/deterministic_replay.py` :
     - `DeterministicReplay(config, seed)` produit `signals.parquet` + `state_transitions.parquet`.
     - Format Parquet (compression, schema strict).
     - Hash SHA256 par tick state.

5. **Tests reproductibilité** (2 h)
   - `tests/test_deterministic_replay.py` :
     - 2 runs identiques → mêmes SHA256.
     - Re-run après pickle/unpickle state → reprend exactement.

6. **Documentation** (2 h)
   - `docs/algo/deterministic_replay.md`.

### Critères d'acceptation
- ✅ 2 runs identiques produisent mêmes hashes.
- ✅ Tests reproductibilité verts.
- ✅ Format Parquet versionné.

### Findings audit adressés
- Sprint 3 batch 3.4 (reproductibilité) — extension.

### Dépendances
- Sprint 3 batch 3.4 (uuid fix).

### Risques
- Float precision drift sur 7 ans backtest → résiduel acceptable si ε < 1e-12.

---

## Batch 6.3 — Snapshot store API per-signal (20 h)

### Objectif
P0-16 : actuellement aucun snapshot per-signal stocké → impossibilité d'auditer un signal historique. Implémenter store SQLite/Parquet avec API REST.

### Steps
1. **Schéma snapshot** (3 h)
   - Pydantic `SignalSnapshot` :
     - `signal_id`, `timestamp_utc`, `symbol`, `tf`.
     - `ohlcv_window` (60 bars autour).
     - `features` (60 features Sprint 3.1).
     - `weighted_scores` (8 composantes).
     - `logistic_proba` + `lower` + `upper`.
     - `regime` + `narrative_text`.
     - `state_machine_state`.
     - `models_versions` (hash registry).

2. **Storage backend** (5 h)
   - SQLite pour metadata + index.
   - Parquet pour OHLCV windows + features (par symbol, partition par mois).
   - `src/intelligence/snapshot_store/store.py`.

3. **Write path** (4 h)
   - Hook dans `sentinel_scanner` : à chaque signal généré, persist snapshot.
   - Async write (non-blocking).
   - Batch flush toutes les 10 sec.

4. **Read API** (4 h)
   - `src/api/routes/snapshots.py` :
     - `GET /snapshots/{signal_id}` → SignalSnapshot JSON.
     - `GET /snapshots?symbol=XAUUSD&start=...&end=...` → list.
     - Auth : tier ≥ STRATEGIST.
   - Cible latence : < 50 ms retrieve.

5. **Tests** (2 h)
   - `tests/test_snapshot_store.py` :
     - Write + read round-trip.
     - Latency benchmark.
     - Concurrent writes.

6. **Documentation client** (2 h)
   - `docs/client/snapshot_api.md`.

### Critères d'acceptation
- ✅ Snapshot store actif.
- ✅ Latence read p95 < 50 ms.
- ✅ Tests round-trip verts.
- ✅ API doc.

### Findings audit adressés
- **P0-16** — ✅ closed.

### Dépendances
- Sprint 4 (signals avec proba+bandes).

### Risques
- Storage croissance : 1 signal/15min × 6 actifs × 24h × 365j = ~140k snapshots/an × ~50 KB = ~7 GB/an. Mitigation : compression Parquet + S3 archive après 90 jours.

---

## Batch 6.4 — Versioning modèles + compatibilité ascendante (16 h)

### Objectif
Tout modèle persisté (Logistic L1, Mondrian quantiles, HMM, Stacking) doit avoir version + métadonnées + compat ascendante. Précurseur retraining pipeline.

### Steps
1. **Model registry schema** (3 h)
   - `src/intelligence/model_registry/registry.py` :
     - `ModelMetadata` Pydantic : `model_id`, `version`, `created_at`, `training_data_hash`, `train_metrics`, `oos_metrics`, `feature_list`, `hyperparams`.
     - Storage : SQLite metadata + filesystem pickles `models/registry/{model_id}/{version}.pkl`.

2. **Save/load API** (4 h)
   - `registry.save(model, metadata) -> model_id`.
   - `registry.load(model_id, version='latest') -> (model, metadata)`.
   - Version semver : `0.1.0`, `0.2.0`.

3. **Compat ascendante** (3 h)
   - Tout chargement doit gérer schema drift :
     - V0 model → V1 schema → add default values for new fields.
     - V1 model → V0 schema → strict reject avec error.
   - Test régression : load model Sprint 4 dans code Sprint 6.

4. **Migration logistic_l1 + mondrian** (3 h)
   - Sprint 4 models → registry format.
   - Hooks dans `confluence_detector` + `conformal_wrapper`.

5. **Tests** (2 h)
   - Round-trip save/load.
   - Schema drift compat.

6. **Documentation** (1 h)
   - `docs/algo/model_registry.md`.

### Critères d'acceptation
- ✅ Registry actif, 5+ models versionnés.
- ✅ Compat ascendante testée.
- ✅ Loading API < 100 ms.

### Findings audit adressés
- **P1-12** (Versioning JSON state absent) — ✅ closed.
- Préparation MLOps (eval_23).

### Dépendances
- Sprint 4 (models Logistic + Mondrian).

### Risques
- Schema drift complexity → over-engineering. Mitigation : v0 simple, raffinement Sprint 7+.

---

## Batch 6.5 — Extraction physique `smart_money/` (18 h) — P0-9 résiduel

### Objectif
La façade Sprint 1.0 est en place mais le code vit toujours dans `strategy_features.py`. Extraction physique du package + tests régression d'équivalence.

### Steps
1. **Audit `strategy_features.py`** (3 h)
   - Identifier fonctions BOS/CHOCH/OB/FVG/retest exactes.
   - LOC à déplacer : ~600.

2. **Module split** (5 h)
   - `src/intelligence/smart_money/bos.py` — BOS detector + helpers.
   - `src/intelligence/smart_money/choch.py` — CHOCH.
   - `src/intelligence/smart_money/ob.py` — Order Block ICT-conforme.
   - `src/intelligence/smart_money/fvg.py` — Fair Value Gap.
   - `src/intelligence/smart_money/retest.py` — Retest tolerance.
   - `src/intelligence/smart_money/__init__.py` — façade publique.

3. **Update façade** (2 h)
   - `smart_money/__init__.py` : remplacer ré-exports par imports modules.
   - Test backward compat : tous imports `from src.intelligence.smart_money import ...` fonctionnent.

4. **Tests régression** (4 h)
   - `tests/test_smart_money_extraction_parity.py` :
     - Run detector pre-extraction (snapshot) vs post-extraction.
     - 100 % parity sur XAU + EURUSD 2024.
   - Validation snapshots PNG Sprint 2.4 inchangés.

5. **Cleanup `strategy_features.py`** (2 h)
   - Supprimer code SMC déplacé.
   - Remplacer par imports depuis `smart_money/`.

6. **Documentation** (2 h)
   - `docs/algo/smart_money.md` : architecture finale.

### Critères d'acceptation
- ✅ Package `smart_money/` 5 modules + __init__.
- ✅ Parity test 100 % (signals identiques pre/post).
- ✅ Snapshot tests Sprint 2.4 verts.
- ✅ `strategy_features.py` ne contient plus code SMC.

### Findings audit adressés
- **P0-9** (Smart Money pas extrait) — ✅ FULLY closed.

### Dépendances
- Sprint 2 (snapshot tests pour parity).

### Risques
- Régression silencieuse : 1 bit-difference → cascade. Mitigation : parity tests strict + git diff review.

---

## Gate de sortie du Sprint 6 (checklist 12 items)

1. ✅ Profiling baseline + post-optim générés.
2. ✅ p99 latence ≤ 250 ms/tick.
3. ✅ Speedup ≥ 3× sur hotspots.
4. ✅ Replay engine reproductible (2 runs → mêmes hashes).
5. ✅ Snapshot store actif, latence read < 50 ms.
6. ✅ API `/snapshots` documentée.
7. ✅ Model registry actif, 5+ models versionnés.
8. ✅ Compat ascendante testée.
9. ✅ Package `smart_money/` extrait, parity 100 %.
10. ✅ `strategy_features.py` nettoyé.
11. ✅ Suite tests verte.
12. ✅ `sprint_6_retrospective.md` rédigé.

---

## Livrables Sprint 6 (arborescence)

```
src/intelligence/smart_money/
  ├── __init__.py
  ├── bos.py
  ├── choch.py
  ├── ob.py
  ├── fvg.py
  ├── retest.py
  ├── validation.py        # Sprint 2
  ├── tuning.py            # Sprint 2
  └── visualization.py     # Sprint 2

src/intelligence/snapshot_store/
  ├── __init__.py
  ├── store.py
  └── schema.py

src/intelligence/model_registry/
  ├── __init__.py
  ├── registry.py
  └── compat.py

src/backtest/replay/
  └── deterministic_replay.py

src/api/routes/
  └── snapshots.py

models/registry/
  ├── logistic_l1/0.1.0.pkl + metadata.json
  ├── mondrian_xau/0.1.0.pkl
  └── stacking_xau_m15/0.1.0.pkl

scripts/
  └── benchmark_tick_latency.py

tests/
  ├── test_deterministic_replay.py
  ├── test_snapshot_store.py
  ├── test_model_registry.py
  └── test_smart_money_extraction_parity.py

reports/sprint_6/
  ├── profile_baseline.svg
  ├── profile_post_optim.svg
  └── perf_report.md

docs/algo/
  ├── deterministic_replay.md
  ├── model_registry.md
  └── smart_money.md

docs/client/
  └── snapshot_api.md

roadmap/sprints/
  ├── sprint_6.md
  ├── sprint_6_progress.md
  └── sprint_6_retrospective.md
```

---

## Décisions ouvertes pour user

1. **Storage growth snapshot store** : 7 GB/an acceptable local, sinon S3 archive après 90j ?
2. **Cython/Numba** : si p99 > 250 ms même post-vectorisation → autoriser Cython (cible 50 ms) ?
3. **Model registry storage** : SQLite OK pour personal testing, PostgreSQL Sprint 7+ commercial ?

---

**Signé** : Claude, 2026-05-15
