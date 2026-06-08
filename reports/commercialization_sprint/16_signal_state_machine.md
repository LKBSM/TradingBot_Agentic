# Plan de Commercialisation — Catégorie 16 : Signal State Machine & Lifecycle

> **Périmètre** : `src/intelligence/signal_state_machine.py` (922 l), `src/intelligence/state_persistence.py` (137 l), intégration scanner (`src/intelligence/sentinel_scanner.py:39-47, 95-100, 245-265, 462-577, 1030-1147`), tests (54 + 12 + replay determinism), persistence JSON atomique.
>
> **Note actuelle** : **8.0 / 10** (cf. `reports/eval_07_signal_state_machine.md:9-22`) — code excellent mais defaults non empiriquement validés.
>
> **Mission** : verrouiller les paramètres par sweep multi-instrument, durcir la persistence (corruption / multi-process / versioning), exposer l'observabilité (transitions, time-in-state), permettre l'A/B testing en prod. Aboutir à un composant **commercialisable PhD-level**, défendable face à un audit institutionnel.

---

## 1. État actuel (Audit)

### 1.1 Forces existantes

| Axe | Note | Source code |
|---|---|---|
| Conception (pure-logic, déterministe) | 9/10 | `signal_state_machine.py:10-44` — pas d'I/O, pas de wall-clock, replay-safe |
| Thread safety | 9/10 | `signal_state_machine.py:336, 417, 450` — `threading.RLock` sur tous les mutateurs |
| Validation entrée | 9/10 | `BarInput.__post_init__` `signal_state_machine.py:208-223` — NaN/négatifs/OHLC incohérent rejetés |
| Idempotence + ordre | 9/10 | `on_bar` `signal_state_machine.py:421-435` — duplicate/out-of-order comptés, pas mutés |
| Six règles d'exit | 9/10 | `_step_active` `signal_state_machine.py:587-656` — priorité TP/SL → regime → structure → opposing → score → time |
| Persistence atomique | 8/10 | `state_persistence.py:39-73` — `tempfile + os.replace`, staleness guard `state_persistence.py:106-121` |
| Inconsistency check `from_dict` | 8/10 | `signal_state_machine.py:884-907` — ACTIVE sans signal/direction → reset IDLE (ajouté post eval_07) |
| Tests | 8/10 | 54 (state machine) + 12 (persistence) + replay determinism (`tests/test_state_machine_replay.py`) |

### 1.2 Faiblesses bloquantes commercialisation

| # | Faiblesse | Source | Sévérité |
|---|-----------|--------|----------|
| F1 | **Defaults non empiriques.** `enter=75 / exit=55 / confirm=2 / cooldown=2 / max_age=64` (`signal_state_machine.py:128-159`) — la note `max_age` bumped 12→64 (`signal_state_machine.py:139-145`) reconnaît elle-même « state-machine-specific forensic remains TODO ». | `signal_state_machine.py:128-159` | **P0 BLOQUANT** |
| F2 | **Sweep partiel exécuté.** `reports/sweep/sweep_summary.md:1-36` couvre 48 cellules QUICK (30k bars, 2 assets, grid `enter ∈ {55,60,65,70}`, `exit ∈ {35,40,45}`, `confirm ∈ {1,2}`) — **0 / 48 ne passent les gates** (DSR≥1.5, PBO≤0.35, PF_lo>1.0, DM_p<0.05). Le grid n'inclut **PAS** `cooldown` ni `max_age` ni `silent_bars`. Mort-né tant que confluence rework n'a pas livré un score à pouvoir prédictif. | `reports/sweep/sweep_summary.md`, `scripts/sweep_state_machine.py:55-65` | **P0** |
| F3 | **Pas de checksum sur le bundle.** Corruption JSON silencieuse → `json.loads` lève → reset IDLE (`state_persistence.py:101-103`). Pas d'alerte ops. | `state_persistence.py:98-103` | **P0** robustesse |
| F4 | **Mono-process safety.** Si on scale à `n_workers>1` (gunicorn), deux scanners lisent/écrivent le même `state_{symbol}.json` → dernière écriture gagne, états perdus. `os.replace` est atomique pour un même writer, pas pour writers concurrents. | `state_persistence.py:55-61` | **P0 si scale** |
| F5 | **Pas de versioning forward.** `schema_version=1` rejette tout v2 (`signal_state_machine.py:840-843`). Aucun migrator script, aucun fallback. Tout upgrade qui touche `to_dict` casse les snapshots prod. | `signal_state_machine.py:800, 840-843` | **P0 deploy** |
| F6 | **`get_stats()` ne mesure pas la latence.** Manque `latency_arm_to_publish_bars` (p50/p95), `pct_arms_aborted_during_confirmation`, `time_in_each_phase_seconds`. Impossible de monitorer le trade-off latence/qualité. | `signal_state_machine.py:382-401` | **P1 observabilité** |
| F7 | **Pas d'export Prometheus.** `transition_history()` (`signal_state_machine.py:403-406`) tient 200 entries en RAM, **rien** dans `/metrics`. | `signal_state_machine.py:403-406` | **P1** |
| F8 | **Pas d'A/B testing in-prod.** `StateMachineConfig` frozen au démarrage. Aucun moyen de comparer config A vs B sur le même flux de bars. | `signal_state_machine.py:120-184` | **P2 growth** |
| F9 | **`_silent_bars` ne reset pas hors `_step_active`.** Pas un bug actif (utilisé uniquement dans `_step_active`), mais fragile. Cf. eval_07 §2 bug n°3. | `signal_state_machine.py:643-648` | **P2 hygiène** |
| F10 | **Exit `_step_cooldown` paralyse 1 bar additionnel** (`signal_state_machine.py:702-710`). 30-45 min de latence supplémentaire post-exit sur M15. Documenté mais non mesuré empiriquement. | `signal_state_machine.py:698-710` | **P2** |
| F11 | **Calibration ne distingue PAS long vs short.** XAU replay (`memory/xau_replay_findings_2026_04_23.md`) : shorts profitable, longs perdent. Defaults uniformes long/short = on dilue le bon edge. | n/a | **P1** |
| F12 | **Pas de path traversal check** sur `persistence_path`. `Path(persistence_path)` accepté tel quel (`sentinel_scanner.py:157-158`). En SaaS multi-tenant, un user input non sanitisé pourrait écraser un fichier hors `data/`. | `state_persistence.py:51-52`, `sentinel_scanner.py:157-158` | **P1 sec** |
| F13 | **Transition history en RAM uniquement.** 200 entries × ~25 transitions/jour XAU M15 = 8 jours visibles. Insuffisant pour dashboard commercial 30j+. | `signal_state_machine.py:159, 367` | **P1 produit** |

### 1.3 Dépendances upstream non-tenues

- **Confluence rework** : tant que `ConfluenceDetector` produit un score à Pearson −0.023 vs PnL (`memory/MEMORY.md:eval_02`, `confluence_calibration.md`), aucun sweep state machine ne franchira les gates. **La calibration P0 dépend du sprint Confluence (autre conv)**.
- **MTF / htf_alignment** : la machine consomme `signal.confluence_score`. Si la confluence intègre un filtre HTF (en cours autre conv), le tier-sweep doit re-run **après**.
- **Backtest harness** : `src/backtest/state_machine_replay.py` est le ground truth pour le replay. Sa fidélité conditionne la validité du sweep.

---

## 2. Vision cible

Une **state machine production-grade, defaults validés empiriquement sur 7 ans XAU + 5 ans EURUSD**, robuste aux corruptions/race/upgrades, observable en temps réel (Prometheus), avec A/B testing in-prod et override tier-gated.

### 2.1 Critères d'acceptance commerciaux

| KPI | Aujourd'hui | Cible go-live | Mesure |
|---|---|---|---|
| Defaults justifiés par sweep | ❌ choisis à la main | ✅ `reports/eval_07_sweep_full.csv` cellule choisie documentée | `scripts/sweep_state_machine.py --full` |
| Sweep couvre `(enter, exit, confirm, cooldown, max_age, silent)` × multi-asset | partiel (3 paramètres, 2 asset, 30k bars) | 5 paramètres × XAU+EUR+USDJPY × 100% data | grid 432 cellules × 3 asset |
| Gates franchies (DSR≥1.5, PBO≤0.35, PF_lo>1.0, DM_p<0.05) | 0 / 48 | ≥ 1 cellule par asset | `reports/sweep/sweep_summary.md` |
| Persistence corruption recovery | reset silencieux | détecté + logué + alerté | checksum SHA-256 + alert |
| Multi-process safety | écrasement | file-lock OS-level | `fcntl`/`msvcrt` lock OU SQLite WAL |
| Schema versioning forward/backward | rejet v2+ | migrator chain v1→v2 testé | `state_persistence.py:migrate()` |
| Observabilité Prometheus | rien | 12 métriques exposées | `/metrics` |
| Time-in-state distribution p50/p95/p99 | non | mesuré par phase | `get_stats()` |
| A/B testing config | non | 2 configs parallèles + comparator | `ABStateMachine` |
| Tier-gated config override | non | STRATEGIST+ peut override `enter ∈ [70, 90]` | `tier_manager` |
| Replay determinism rate | testé qualitativement | 100% hash equality | golden test |

### 2.2 Argument commercial

> « Notre couche de décision est **un état fini déterministe**, pas une boîte noire ML. Chaque transition HOLD↔BUY↔SELL est traçable à 1 des 6 règles, exprimables en une phrase. Le client sait **pourquoi** il reçoit un signal et **pourquoi** il sort. »

C'est défendable face à MiFID II (cf. `eval_29_compliance_findings.md`) et à un audit B2B broker (`memory/MEMORY.md:dual_b2c_b2b_architecture`).

---

## 3. Gap analysis

| Dimension | État | Cible | Gap (h) |
|---|---|---|---|
| Code qualité (déterminisme, thread-safety) | 9/10 | 9/10 | 0 |
| Defaults empiriques (sweep + analyse) | 2/10 | 9/10 | **40-60h** (dépend confluence) |
| Persistence robustesse (checksum, multi-process, versioning) | 6/10 | 9/10 | 18h |
| Observability (metrics, Prometheus, latency) | 4/10 | 9/10 | 16h |
| Tier-gated override | 0/10 | 7/10 | 10h |
| A/B testing harness | 0/10 | 7/10 | 12h |
| Sécurité (path traversal, JSON injection) | 5/10 | 9/10 | 6h |
| Tests (replay, crash, race) | 7/10 | 9/10 | 14h |
| Documentation / runbook | 4/10 | 8/10 | 6h |
| **TOTAL** | | | **122-142h** |

---

## 4. Plan d'exécution

### P0 — Sweep 432 cellules POST-MTF/confluence rework

**Dépend de** : Confluence rework (autre conv, sprint MTF/htf_alignment). Lancer dès que score recalibré disponible.

**Objectif** : passer du grid 48-cellules QUICK (`reports/sweep/sweep_summary.md`) au grid complet 432 cellules × 3 assets sur 100% des données disponibles.

| # | Tâche | Fichiers | Heures | Acceptance | Dépendances |
|---|-------|----------|-------:|------------|-------------|
| 1.1 | Étendre `scripts/sweep_state_machine.py:55-65` pour inclure `cooldown ∈ {0,2,5}`, `max_age ∈ {12,24,64}`, `silent_bars ∈ {1,2,3}`. Garde `exit < enter`, `enter ∈ {55..85}`, `exit ∈ {35..70}`, `confirm ∈ {1,2,3}`. | `scripts/sweep_state_machine.py` | 4 | Grid produit ≥ 432 cellules valides après contraintes | confluence rework livré |
| 1.2 | Ajouter assets : XAU M15, EURUSD M15, USDJPY M15 (cf. `eval_20_multi_asset.md` — drop BTC+US500). | `scripts/sweep_state_machine.py:60-65` | 2 | 3 ASSETS × ≥432 cellules = ≥1296 runs | data CSVs présents |
| 1.3 | Lancer sweep **full** (pas QUICK) → 100% data 2019-2026. Compute estimé : ~1 min/cellule × 1296 = ~22h GPU/CPU. Background run. | run | 22 (compute) | `reports/sweep/sweep_full.csv` complet | 1.1, 1.2 |
| 1.4 | Analyse : top-3 cellules par asset franchissant gates (DSR / PBO / PF_lo / DM_p) ; heatmap PF × (enter × confirm) ; sensibilité cooldown/max_age. | `scripts/analyze_sweep.py` (nouveau) + `reports/sweep/analysis_2026.md` | 6 | Top-3 documenté + heatmap PNG + recommandation defaults par asset | 1.3 |
| 1.5 | Mise à jour `StateMachineConfig` defaults **par instrument** (cf. F11 long/short distinct + multi-asset). Migrer commentaire docstring `signal_state_machine.py:128-159` avec citations sweep. | `signal_state_machine.py:128-184`, `config.py` (presets `INSTRUMENT_DEFAULTS`) | 4 | Defaults sourced from sweep cell IDs ; commentaires citent `reports/sweep/sweep_full.csv` ligne | 1.4 |
| 1.6 | Distinct long/short config support : ajouter `enter_threshold_long`, `enter_threshold_short` (idem exit). Cf. `memory/xau_replay_findings_2026_04_23.md`. | `signal_state_machine.py:120-184, 484-501` | 6 | Tests dédiés long-only / short-only / asymétrique | 1.5 |
| 1.7 | Documenter dans `reports/commercialization_sprint/16_signal_state_machine.md` la cellule retenue + justification stat (DM p-value, CI bootstrap PF). | doc | 2 | Memo lisible pour audit | 1.4-1.6 |

**Total P0 sweep** : ~46 h dev + ~22 h compute = **~68 h**, bloqué par confluence rework.

### P0 — Calibrer cooldown / lifetime / opposing-lockout par instrument

**Sub-tâche de la sweep, mais à isoler pour traçabilité.**

| # | Tâche | Fichiers | Heures | Acceptance | Dépendances |
|---|-------|----------|-------:|------------|-------------|
| 2.1 | Analyse de sensibilité `cooldown_bars ∈ {0, 1, 2, 5, 10}` à enter/exit fixés sur top-cell. Mesurer : trades/an, PF, time-to-re-entry. | `scripts/analyze_sweep.py` (sous-section) | 4 | Courbe PF(cooldown) par asset | 1.4 |
| 2.2 | Analyse `max_signal_age_bars ∈ {6, 12, 24, 48, 64, 96}`. NB : eval_07 §1.3 note bump 12→64 décidé empiriquement sur trade lifetime, **pas sur signal lifetime** (`signal_state_machine.py:139-145`). À refaire **proprement** sur signal lifetime. | `scripts/analyze_sweep.py` | 4 | Courbe PF(max_age) par asset, valide ou invalide le 64 actuel | 1.4 |
| 2.3 | Analyse `silent_bars_before_score_exit ∈ {1, 2, 3, 5}` — impact sur exits par SCORE_DECAYED vs TIME_EXPIRED. | `scripts/analyze_sweep.py` | 3 | Distribution exits_by_reason par config | 1.4 |
| 2.4 | Recommandation finale par instrument dans `reports/sweep/analysis_2026.md` : tableau `{symbol, enter, exit, confirm, cooldown, max_age, silent, PF, DSR}`. | doc | 2 | Validé par run replay déterministe sur cellule choisie | 2.1-2.3 |

**Total P0 calibration** : **~13 h**.

### P0 — Persistence robustness (corruption, multi-process, versioning)

| # | Tâche | Fichiers | Heures | Acceptance | Dépendances |
|---|-------|----------|-------:|------------|-------------|
| 3.1 | **Checksum SHA-256** sur payload. Ajouter `"sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()` dans `to_dict`. Vérifier au `from_dict`. | `signal_state_machine.py:783-823, 826-908`, `state_persistence.py:53-61, 98-104` | 3 | Test : flip 1 byte du fichier → load renvoie None + log WARN ; metric `state_machine.corrupt_load_total` incrémenté | aucune |
| 3.2 | **Multi-process safety** : file lock OS-level via `portalocker` (cross-platform). `save_state_machine` acquiert lock exclusif ; `load_state_machine` lock partagé. | `state_persistence.py:39-73, 76-137` + `requirements.txt` (portalocker) | 5 | Test concurrent : 4 threads writer × 2 reader sur même fichier → 0 corruption ; benchmark <50ms/save | aucune |
| 3.3 | **Schema versioning** : extraire `_MIGRATIONS = {1: migrate_v1_to_v2, ...}` dict. `from_dict` détecte version, applique migrations chain. | `signal_state_machine.py:825-908` (refactor), nouveau `src/intelligence/state_machine_migrations.py` | 6 | Test : load payload v1 dans code v2 → migré transparent ; v3 dans code v2 → rejet propre avec log | aucune |
| 3.4 | **Path traversal guard** : `persistence_path` doit être relative-to-data-dir OU absolute-allowlisted. `_validate_persistence_path()`. | `state_persistence.py:39-52`, `sentinel_scanner.py:157-158` | 2 | Test : `persistence_path="../../etc/passwd"` rejeté ; symlink escaping rejeté | aucune |
| 3.5 | **Sanitize/validate JSON deserialisation** : `from_dict` doit refuser types inattendus (`Direction` valeurs hors enum, `_Phase` invalide). Aujourd'hui `_Phase(payload["phase"])` lève si invalide — propre, mais `bars_processed` etc. acceptent `int()` cast sans borne sup. | `signal_state_machine.py:846-867` | 2 | Test : payload avec `bars_processed=-1` ou `2**63` rejeté | aucune |

**Total P0 persistence** : **~18 h**.

### P1 — Observability : transitions metrics, time-in-state distribution

| # | Tâche | Fichiers | Heures | Acceptance | Dépendances |
|---|-------|----------|-------:|------------|-------------|
| 4.1 | Ajouter `latency_arm_to_publish_bars_p50/p95/p99` dans `_stats`. Mesure : `_pending_bars` quand `_confirm_arming` est appelé. | `signal_state_machine.py:354-365, 533-583` (instrumenter), 382-401 (exposer) | 3 | `get_stats()` renvoie 3 nouvelles clés ; test unitaire dédié | aucune |
| 4.2 | `time_in_phase_seconds_p50/p95` par phase. Hash map `{_Phase: list[durations]}` (capped). | `signal_state_machine.py:298-368` | 3 | Test : enchaîner 100 transitions, vérifier percentiles | aucune |
| 4.3 | `pct_arms_aborted_during_confirmation = arms_aborted / arms_started` dans `get_stats`. | `signal_state_machine.py:395-401` | 1 | KPI exposé | aucune |
| 4.4 | Export Prometheus : nouveau `src/intelligence/state_machine_metrics.py`. Counters : `signals_emitted_total{symbol,direction}`, `exits_by_reason_total{symbol,reason}`, `arms_aborted_total`, `bars_rejected_total{type}`, `corrupt_loads_total`. Gauges : `current_phase{symbol}`, `cooldown_remaining_bars{symbol}`, `time_in_state_seconds`. Histograms : `signal_lifetime_bars{symbol}`, `arm_to_publish_latency_bars`. | nouveau fichier + `src/api/routes/health.py` integration | 5 | `curl /metrics | grep state_machine_` retourne 12 séries | F7 |
| 4.5 | **Transition history en SQL** : ajouter table `signal_transitions` dans `signal_store.py`. Persister chaque `to_dict()` transition. Query API `/v1/signals/transitions?symbol=XAUUSD&from=2026-01-01&reason=target_reached`. | `src/api/signal_store.py`, `src/api/routes/signals.py`, `signal_state_machine.py:444-446` (hook) | 4 | Test : 30j de transitions queryable, pagination, filtres | aucune |

**Total P1 observability** : **~16 h**.

### P1 — Versioning forward/backward compat lors d'upgrades

| # | Tâche | Fichiers | Heures | Acceptance | Dépendances |
|---|-------|----------|-------:|------------|-------------|
| 5.1 | Migration chain : v1→v2 dummy + test (preuve d'infra avant qu'on en ait besoin). | `src/intelligence/state_machine_migrations.py` | 3 | Test : payload v1 chargé dans code v2 sans data loss | 3.3 |
| 5.2 | Snapshot validator CLI : `python -m scripts.validate_state_snapshot path/to/state.json` — vérifie schema, checksum, consistency, staleness. | nouveau `scripts/validate_state_snapshot.py` | 2 | Exit code 0 si valide, 1 sinon ; CI run sur chaque PR touchant `signal_state_machine.py` | 3.1, 3.3 |
| 5.3 | Runbook ops : `docs/runbooks/state_machine_recovery.md` — étapes en cas de corruption, downgrade rollback, multi-symbol selective reset. | nouveau md | 3 | Lu et signé par ops | aucune |
| 5.4 | Backward compat test in CI : load snapshots historiques (`tests/fixtures/state_snapshots/v1_*.json`) — refuser upgrade qui casse v1. | `tests/test_state_persistence_compat.py` | 2 | CI fail si breaking change sans migration | 5.1 |

**Total P1 versioning** : **~10 h**.

### P1 — Tier-gated config override (commercial monétisation)

| # | Tâche | Fichiers | Heures | Acceptance | Dépendances |
|---|-------|----------|-------:|------------|-------------|
| 6.1 | Définir tier matrix dans `src/api/tier_manager.py` : FREE = read-only ; ANALYST = read-only ; STRATEGIST = override `enter_threshold ∈ [70, 90]` ; INSTITUTIONAL = override tous les paramètres. | `src/api/tier_manager.py`, `src/api/models.py` | 3 | Test : FREE call à `/v1/state-machine/config` retourne 403 | tier_manager existant |
| 6.2 | Endpoint `PATCH /v1/state-machine/config` validé + persisté en SQLite par user. | `src/api/routes/signals.py` (nouveau sous-router), `src/api/signal_store.py` | 4 | Test : STRATEGIST set enter=85 → snapshot après next bar applique 85 | 6.1 |
| 6.3 | Per-user state machine factory : scanner consulte tier au démarrage, instancie config user-specific. | `sentinel_scanner.py:1030-1098` (MultiSymbolScanner factory) | 3 | Test : user A `enter=75`, user B `enter=85`, 1 stream de bars, 2 sorties différentes | 6.2 |

**Total P1 tier-gating** : **~10 h**.

### P2 — A/B testing harness

| # | Tâche | Fichiers | Heures | Acceptance | Dépendances |
|---|-------|----------|-------:|------------|-------------|
| 7.1 | `ABStateMachine` wrapper : instancie 2 `SignalStateMachine` avec configs A/B, feed mêmes bars, log divergences. | nouveau `src/intelligence/ab_state_machine.py` | 4 | Test : config A=enter=75, B=enter=80 sur 1000 bars → divergence comptée | aucune |
| 7.2 | Stats comparateur : `compare(a, b) -> {extra_signals_a, extra_signals_b, divergence_rate, conflict_directions}`. | dans `ab_state_machine.py` | 2 | Test unitaire | 7.1 |
| 7.3 | Persistence A et B séparée (`state_A_{symbol}.json`, `state_B_{symbol}.json`). | `state_persistence.py` (ajouter `variant` arg) | 2 | Test round-trip indépendant | 7.1 |
| 7.4 | Env var `STATE_MACHINE_AB_ENABLED=1` + `AB_CONFIG_B=path/to/config_b.json`. Dashboard endpoint `/v1/admin/ab-compare`. | `src/intelligence/main.py`, `src/api/routes/admin.py` | 4 | Smoke test : start scanner avec AB on → `/v1/admin/ab-compare` retourne JSON live | 7.1-7.3 |

**Total P2 A/B** : **~12 h**.

---

## 5. Tests & validation

### 5.1 Replay golden tests (déterminisme)

| Test | Fichier | Acceptance |
|---|---|---|
| **Golden hash** : hash SHA-256 du JSON `to_dict()` après 10k bars XAU = valeur fixée. | `tests/test_state_machine_golden.py` (nouveau) | Modifier le code change le hash → CI fail demandant update conscient |
| **Multi-config golden** : 5 configs × 1000 bars XAU + EURUSD → 10 hashes, tous reproduits sur 3 runs successifs | idem | Replay 100% déterministe |
| **Order-independence (when relevant)** : feed les mêmes bars dans 2 ordres légaux (timestamps croissants) → mêmes hashes | idem | n/a — ordre déjà strictement contraint, mais test défensif |

### 5.2 Persistence crash tests

| Test | Fichier | Acceptance |
|---|---|---|
| **Crash mid-write** : `save_state_machine` interrompue (`os.kill` du process) au moment du `os.replace` | `tests/test_state_persistence_crash.py` (nouveau) | Soit fichier original intact, soit nouveau complet — JAMAIS de fichier corrompu |
| **Disk full** : mock `OSError(ENOSPC)` sur fdopen → return False, original intact | idem | Test passing |
| **JSON corruption** : flip random bytes du payload → load renvoie None + log WARN + metric | `tests/test_state_persistence.py` (étendre) | Test passing |
| **Staleness** : saved=`2025-01-01`, current=`2026-01-01`, max_staleness_bars=4 → discard | déjà couvert `test_state_persistence.py` | OK |
| **Schema downgrade** : v2 payload, v1 code → rejet propre, pas d'exception | nouveau | Test passing |

### 5.3 Race conditions / concurrency

| Test | Fichier | Acceptance |
|---|---|---|
| **2 threads** appellent `on_bar()` en parallèle sur même machine, bars différents | `tests/test_signal_state_machine.py` (existant) | OK (RLock protège) |
| **Reader + writer** : 1 thread snapshot() en boucle, 1 thread on_bar() — pas de tearing | étendre | Snapshots toujours consistants |
| **Multi-process writers** : 4 process écrivent sur même file via portalocker → 0 corruption | `tests/test_state_persistence_multiprocess.py` (nouveau) | Test passing |

### 5.4 Sweep validation

| Test | Fichier | Acceptance |
|---|---|---|
| **Sweep reproducibility** : run twice avec seed=42 → CSV identique byte-for-byte | `tests/test_sweep_state_machine.py` (nouveau) | OK |
| **Grid coverage** : 432 cellules réellement testées (pas de skip silencieux) | idem | OK |
| **Gate logic** : cellule artificielle PF=2.0 forcée → gates passent | idem | OK |

### 5.5 Stress test long-run

| Test | Fichier | Acceptance |
|---|---|---|
| **Memory leak** : 1M bars consécutifs, `transition_history` cappé à 200 → RSS stable | `tests/test_state_machine_stress.py` (nouveau) | <50 MB de croissance |
| **Latency p99 on_bar** : <500 µs sur Ryzen 5 standard | idem | OK |

---

## 6. Sécurité

### 6.1 Surface d'attaque

| Vecteur | Source | Mitigation |
|---|--------|------------|
| **Path traversal** sur `persistence_path` | `sentinel_scanner.py:157-158`, `state_persistence.py:51-52` | F12 — `_validate_persistence_path()` whitelist dans `data/`, rejet symlinks |
| **JSON injection** dans payload | `state_persistence.py:99-103` | F3 — checksum SHA-256 + validation Pydantic stricte des types `Direction`/`_Phase`/bornes int |
| **DoS via transition_history flooding** | `signal_state_machine.py:367` | Déjà bornée `transition_history_max=200`, mais : ajouter rate limit côté API `/v1/transitions` (déjà géré middleware) |
| **DoS via bars_rejected_invalid spam** | `signal_state_machine.py:357-358` | Counter exposé Prometheus → alert si >100/min |
| **Confidentialité state file** | data/ partagé | Chmod 0600 sur write ; en SaaS multi-tenant, partitioning par user_id obligatoire |
| **Replay attack** (vieux state injecté) | corruption disque | Checksum + staleness guard + version + signature optionnelle HMAC en INSTITUTIONAL |

### 6.2 Sécurisation file write

```python
# state_persistence.py:save_state_machine — additions:
os.chmod(path, 0o600)  # owner read/write only
```

### 6.3 Validation entrée (déjà bonne, à étendre)

- `BarInput.__post_init__` (`signal_state_machine.py:208-223`) rejette NaN/négatifs/OHLC incohérent ✅
- `StateMachineConfig.__post_init__` (`signal_state_machine.py:164-184`) valide thresholds ✅
- À AJOUTER : `_extract_score_and_direction` clamp `[0, 100]` (déjà fait `signal_state_machine.py:730`) ✅
- À AJOUTER : `from_dict` valide bornes int (cf. 3.5)

### 6.4 Compliance (cf. `eval_29_compliance_findings.md`)

- Aucune donnée PII dans le state file ✅
- Mais en SaaS multi-tenant : ajouter `user_id_hash` au state file pour audit log GDPR

---

## 7. Métriques produit

### 7.1 KPIs technique (à exposer Prometheus)

| Métrique | Type | Labels | But |
|---|------|--------|-----|
| `state_machine_bars_processed_total` | Counter | `symbol` | Sanity check pipeline alive |
| `state_machine_signals_emitted_total` | Counter | `symbol, direction` | Volume signal par direction |
| `state_machine_exits_by_reason_total` | Counter | `symbol, reason` | Distribution exits — débogage |
| `state_machine_arms_aborted_total` | Counter | `symbol` | Stabilité confirmation |
| `state_machine_arm_to_publish_latency_bars` | Histogram | `symbol` | Latence interne (≈ `confirm_bars`) |
| `state_machine_signal_lifetime_bars` | Histogram | `symbol, exit_reason` | Distribution lifetime — sanity sur `max_age` |
| `state_machine_current_phase` | Gauge | `symbol, phase` | Vue live dashboard ops |
| `state_machine_cooldown_remaining_bars` | Gauge | `symbol` | Vue live cooldown |
| `state_machine_bars_rejected_total` | Counter | `symbol, reason` | Détection feed cassé |
| `state_machine_transitions_total` | Counter | `symbol, from, to` | Matrice de transitions |
| `state_machine_persistence_corrupt_loads_total` | Counter | - | Alerte ops |
| `state_machine_persistence_save_duration_seconds` | Histogram | - | Latence I/O save |

### 7.2 KPIs business / produit

| KPI | Calcul | Cible go-live |
|---|--------|---------------|
| **Signaux/jour par symbol** | `signals_emitted / days` | XAU M15 : ~0.5-2/jour selon sweep cellule retenue |
| **Confirmation rate** | `arms_confirmed / arms_started` (existe `signal_state_machine.py:395-400`) | ≥ 70% (sinon defaults trop strict) |
| **Time-in-state p50** | `_bars_since_phase_change` distrib par phase | HOLD: 80% du temps ; ACTIVE: 5-10 bars ; COOLDOWN: cooldown_bars |
| **Replay determinism rate** | `% golden tests passing` | 100% |
| **Exit reason distribution** | `exits_by_reason / total_signals` | TARGET_REACHED ≥ 30% en cellule retenue (sinon le edge n'est pas exploité) |
| **% signals held by state machine** | `signals_held_by_state_machine / (signals_held + signals_emitted)` | 20-40% (élimination du bruit confluence) ; <10% = state machine inutile ; >70% = trop strict |

### 7.3 KPIs replay (audit)

| KPI | Source | Cible |
|---|--------|-------|
| Profit Factor cellule retenue XAU M15 7-ans | `reports/sweep/sweep_full.csv` | ≥ 1.20 (CI lo > 1.0) |
| DSR cellule retenue | idem | ≥ 1.5 |
| PBO | idem | ≤ 0.35 |
| DM p-value vs constant baseline | idem | < 0.05 |
| Sharpe annualisé | idem | ≥ 1.0 |

---

## 8. Risques & mitigations

| Risque | Probabilité | Impact | Mitigation |
|---|---|---|---|
| **State corruption mid-write** | Faible (atomic write ok) | Élevé (perte d'état) | Checksum + restore from previous + alert |
| **Sweep compute cost** (22h × 1296 cellules) | Élevée | Moyen | (a) cache enrichment SMC une fois ; (b) parallélisation 8 workers → 3h ; (c) early-stop si cellule ne génère pas de trade au tier 1 |
| **Regime drift** (sweep gagne 2019-2024, perd 2025+) | Moyenne | Élevé | OOS séparé 2025-2026, gates franchies sur OOS uniquement ; re-sweep trimestriel automatisé |
| **Multi-process écrasement** | Élevée si `gunicorn -w 4` | Élevé | Portalocker (3.2) OU migration vers SQLite single-writer pattern |
| **Schema break sur upgrade** | Moyenne | Élevé | Migration chain testée (5.1) + golden snapshots in CI (5.4) |
| **A/B harness shadow signals confonds** | Faible | Faible | Variant tag explicite dans logs/metrics, jamais publié au client |
| **Confluence rework dérape** | Élevée (autre conv) | Bloque P0 | Construire sur **mock confluence améliorée** ; harness sweep prêt pour exécution dès livraison |
| **Tier override mal validé** | Moyenne | Moyen (user shoot foot) | Bornes serveur strictes (cf. 6.1) + tests + UI warning |
| **Path traversal exploité** | Faible | Élevé | F12 mitigation P1, à passer P0 si exposition publique |
| **Mémoire transition_history fuit** | Très faible (deque maxlen) | Faible | Stress test 1M bars (5.5) |

---

## 9. Dépendances

### 9.1 Bloquantes upstream

| Dépendance | État | Owner | Impact si pas livrée |
|---|---|---|---|
| **Confluence rework** (Pearson > 0.1 vs PnL) | EN COURS autre conv | autre conv | Sweep impossible : 0/48 gates passent aujourd'hui |
| **MTF / htf_alignment** | EN COURS autre conv | autre conv | Si HTF filter ajouté → re-run sweep complet après |
| **Tier sweep (Phase 3)** | EN COURS autre conv | autre conv | Coordination : ne pas dupliquer le grid ; mutualiser `scripts/sweep_state_machine.py` |
| **`src/backtest/state_machine_replay.py` ground truth** | LIVRÉ | this scope | Si bug introduit, sweep invalidé |

### 9.2 Aval (consumers)

| Consumer | Impact si state machine change |
|---|--------------------------------|
| `SentinelScanner` (`sentinel_scanner.py:39-47, 462-577`) | Wrapper transparent, intégration déjà testée |
| `SignalStore` (`src/api/signal_store.py`) | Ajouter table `signal_transitions` (P1 #4.5) |
| `TelegramNotifier` / `DiscordNotifier` (`src/delivery/`) | `send_exit` consomme `ExitReason.value` ; pas de change |
| API `/v1/signals/state` (memory `signal_state_machine.md:32`) | À écrire (deferred follow-up) |
| Dashboard `mockups/tradingview_dashboard_mockup.html` | Consomme snapshot + exit chip — JSON contract stable |
| InsightAssembler (`src/intelligence/insight_assembler.py`) | Consomme `signal.signal_type` ; pas impacté |

### 9.3 Outils tiers

| Outil | Usage | Risque |
|---|---|---|
| `portalocker` (3.2) | Multi-process file lock | Cross-platform, mature |
| Prometheus client | Export metrics (4.4) | Déjà dans `requirements.txt` ? Vérifier ; sinon ajouter `prometheus-client` |
| `hashlib` SHA-256 | Checksum (3.1) | stdlib ✅ |

---

## 10. Estimation totale & timeline

### 10.1 Récapitulatif effort

| Bloc | Heures dev | Heures compute |
|---|-----------:|---------------:|
| P0 Sweep 432 cellules × 3 assets | 18 | 22 (background) |
| P0 Calibration cooldown/lifetime/silent | 13 | (inclus dans sweep) |
| P0 Persistence robustness | 18 | - |
| P1 Observability + Prometheus | 16 | - |
| P1 Versioning + migrator | 10 | - |
| P1 Tier-gated override | 10 | - |
| P2 A/B testing | 12 | - |
| Tests transverses (golden, crash, stress, race) | 14 | - |
| Documentation / runbook | 6 | - |
| Sécurité (path traversal, chmod, validation) | 6 | - |
| **TOTAL** | **123 h dev** | **22 h compute background** |

### 10.2 Timeline conditionnelle

**Hypothèse** : 1 dev plein temps, 6h/jour effectives.

| Semaine | Bloc | Livrable | Bloqué par |
|---|------|----------|------------|
| **S1** (30h) | Persistence robustness P0 (18h) + sécurité (6h) + tests crash (6h) | F3-F5, F12 fermés. Checksum + portalocker + path traversal en prod. | aucune |
| **S2** (30h) | Observability P1 (16h) + Versioning (10h) + doc (4h) | F6-F7 fermés. Prometheus live, migrator chain prête. | aucune |
| **S3** (30h) | Tier-gated override (10h) + A/B testing (12h) + tests transverses (8h) | F8 fermé. Tier matrix opérationnelle, A/B harness shippable. | tier_manager existant |
| **S4** (attente / blocage) | (en attente confluence rework) | — | confluence rework |
| **S5-S6** (compute + analyse) | Sweep full 432 × 3 assets (compute 22h background) + analyse (10h) + calibration analyse (13h) | F1-F2-F11 fermés. Defaults justifiés, cellule retenue documentée par asset. | confluence rework livré |
| **S7** (10h) | Mise à jour defaults `StateMachineConfig`, distinct long/short config (1.6), docs finales (2h) | Code defaults reflète sweep. Memo audit prêt. | sweep terminé |

**Timeline réaliste** : **5-7 semaines** dev calendrier, dont **3 semaines indépendantes** (S1-S3) lançables **dès maintenant**, et **2-3 semaines dépendantes** du confluence rework.

### 10.3 Quick wins exécutables immédiatement (≤ 4h cumulées, en parallèle)

| Quick win | Fichier | Heures |
|---|---------|-------:|
| Checksum SHA-256 dans `to_dict`/`from_dict` | `signal_state_machine.py:783-823`, `state_persistence.py` | 1.5 |
| `chmod 0o600` post-save | `state_persistence.py:60` | 0.25 |
| `latency_arm_to_publish_bars` dans `get_stats` | `signal_state_machine.py:382-401` | 1 |
| `pct_arms_aborted` dans `get_stats` | `signal_state_machine.py:395-401` | 0.5 |
| Reset `_silent_bars=0` à l'entrée des phases non-active (F9) | `signal_state_machine.py:484, 505, 698-710` | 0.5 |

**Total quick wins** : **3.75 h** → faisable **avant** le sweep, débloque mesure latence.

---

## Annexes

### A. Liens

- Source code : `src/intelligence/signal_state_machine.py` (922 l), `src/intelligence/state_persistence.py` (137 l)
- Intégration scanner : `src/intelligence/sentinel_scanner.py:39-47, 95-100, 153-265, 462-577, 1030-1147`
- Tests existants : `tests/test_signal_state_machine.py` (760 l, 54 tests), `tests/test_state_persistence.py` (275 l, 12 tests), `tests/test_state_machine_replay.py` (284 l)
- Sweep partiel : `scripts/sweep_state_machine.py`, `scripts/eval_07_state_machine_sweep.py`, `reports/sweep/sweep_summary.md`, `reports/eval_07_sweep.csv`, `reports/eval_07_sweep_top10.md`
- Eval : `reports/eval_07_signal_state_machine.md` (8.0/10, defaults non empiriques = P0)
- Memory : `signal_state_machine.md`, `state_persistence.md`, `xau_replay_findings_2026_04_23.md` (shorts profitable / longs non)
- Forensic L1 (max_age 12→64) : `reports/forensics/L1_timeout_sweep.csv` (cité `signal_state_machine.py:139-145`)

### B. Décisions de design à acter avant exécution

1. **Distinct long/short config ?** Si oui (recommandé cf. F11), refacto P0 1.6 ajoute complexité tier override.
2. **A/B testing scope** : in-prod sur user traffic (risque) ou shadow seulement (safe, recommandé) ?
3. **Sweep parallélisation** : 8 workers locaux (Ryzen) ou cloud spot (AWS) ? Coût négligeable mais setup CI variable.
4. **Versioning policy** : combien de versions backward supportées ? Recommandation : N-2 (v3 code lit v1, v2, v3).
5. **Persistence backend** : reste JSON ou migration SQLite (résout multi-process + query history) ? **Décision plug-in** dans P1 #5.

### C. Hors-scope (non couvert par cette catégorie)

- Choix du score confluence sous-jacent → autre catégorie
- LLM narrative sur les transitions → autre catégorie (déjà couvert eval_05 LLM)
- UI dashboard exit-reason chip → déférée (cf. `signal_state_machine.md` deferred follow-ups)
- Benchmark HMM/BOCD comme remplacement → décision eval_07 §8 : **REPORTER**, gain marginal vs coût dev

---

**Fin du plan.**
