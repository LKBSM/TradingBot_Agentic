# Plan de Commercialisation — Catégorie 13 : Testing Infrastructure

**Auteur** : QA / Test Infrastructure Lead
**Date** : 2026-05-21
**Branche** : `institutional-overhaul`
**Périmètre** : pytest config, coverage, smoke, integration, contract, load, mutation, CI/CD, fixtures, flaky management
**Inputs consolidés** :
- `reports/eval_17_testing.md` (note 5.5/10, 1673 tests inventoriés)
- `tests/` (164 fichiers `test_*.py` au 2026-05-21, +84 nouveaux fichiers vs. baseline eval_17)
- `pytest.ini:1-12` (config minimale actuelle)
- `tests/conftest.py:1-150` (fixtures partagées)
- `.github/workflows/ci.yml:1-150` + `.github/workflows/algo_tests.yml:1-46` (CI partielle existante)
- Présence : `infrastructure/Dockerfile`, `Dockerfile` racine, `infrastructure/docker-compose.yml`
- Absence confirmée : `.coveragerc`, `pyproject.toml`, `tox.ini`, OWASP ZAP, Locust, Pact, mutmut, Hypothesis dans `requirements.txt:89-91`

> **Objectif business** : commercialisation ASAP avec garantie *régression-zero* sur le chemin de revenu (signal → notification → facturation). Le testing devient le **filet** qui autorise les déploiements quotidiens sans peur.

---

## 1. État actuel (Audit)

### 1.1 Volumétrie tests

| Indicateur | Valeur | Source |
|---|---|---|
| Fichiers `test_*.py` dans `tests/` | **164** (vs. 80 en eval_17) | `ls tests/` 2026-05-21 |
| Tests collectés (estimé) | ~2 500–2 800 | extrapolation eval_17 × ratio fichiers |
| Tests cassés à la collection | 2 (`test_long_short_trading.py` import `src.config`, `test_env_debug.py` racine) | `reports/eval_17_testing.md:138-171` |
| Flaky connu | `test_short_roundtrip_pnl` (~3-5 % fail rate) | `tests/test_sprint1_short_rollback.py:223-261` + eval_17 §T3 |
| Coverage global estimé | ~53 % | eval_17 §T10 Q4 |
| Coverage `volatility_forecaster` | **63 %** | eval_17 §T2 |
| Coverage `telegram_notifier` | **10 %** ❌ revenue-critical | eval_17 §T2 |
| Coverage `discord_notifier` | **13 %** | eval_17 §T2 |
| Coverage `main.py` (FastAPI bootstrap) | **36 %** | eval_17 §T2 |
| Coverage `routes/state.py` | **18 %** | eval_17 §T2 |
| Coverage `volatility_lgbm.py` | **0 %** jamais importé | eval_17 §T2 |

### 1.2 Configuration

- `pytest.ini:1-12` minimal : 2 markers (`live`, `benchmark`), `filterwarnings` deux entrées, **pas de** `addopts`, **pas de** `--cov`, **pas de** `--timeout`, **pas de** `strict-markers`.
- `tests/conftest.py:1-150` : fixtures DB tmp / API key / store / metrics / sample_signal / mock_kill_switch. **Pas de** seed RNG global → flaky possible (eval_17 §T3 patch suggéré ligne 126-131).
- `.coveragerc` : absent. Coverage capturée seulement quand modules **importés** — fausse 0/99 % sur `tier_manager`.
- `pyproject.toml` : absent (configuration uniquement via `pytest.ini`).

### 1.3 CI/CD existante

| Workflow | Job | Couverture | Faille |
|---|---|---|---|
| `.github/workflows/ci.yml:64-150` | `test` (data sprints) | 26 fichiers tests Phase 1 uniquement + `--cov-fail-under=55` sur scope étroit (`src/agents/data`, `src/research`, `src/api/insight_signal_v2`, `src/intelligence/rag`) | **Ignore 138 autres test files** (chemin de revenu Telegram/scanner/volatility/state-machine non protégé en CI) |
| `.github/workflows/ci.yml:31-56` | `lint` | ruff+black, `continue-on-error: true` (advisory) | **Non bloquant** → dérive garantie |
| `.github/workflows/algo_tests.yml:1-46` | `algo-tests` | Tous tests sauf `slow/integration/skip_on_ci`, `--maxfail=20` | Pas de coverage publié, pas de gate, dépend de CSV locaux non shippés |

**Verdict** : la CI couvre ~16 % des tests existants avec un gate coverage ; les 138 autres tournent en best-effort sans gate. **Aucun load test, aucun contract test, aucun mutation test, aucun security scan automatisé.**

### 1.4 Fixtures & données

- `data/XAU_15MIN_2019_2024.csv` (97,6 % coverage, 63 MB), `data/XAU_15MIN_2019_2025.csv` (63 % coverage, 53 MB), calendrier économique 47 MB → **~165 MB raw data**, non shippé en CI (cf. commentaire `ci.yml:8-10`).
- `tests/conftest.py:23-50` : SQLite tmp pour signal_store/key_store, bon pattern, mais aucun fixture *golden 7-day OHLCV* committable.
- `tests/eval_llm/` : harness eval LLM avec `rag_baseline.json` (snapshot baseline) + `regression_gate.py` → **pattern exemplaire à généraliser** sur scoring/state-machine.

### 1.5 Sécurité testing

- Aucun `bandit`, `safety`, `semgrep`, `pip-audit`, `trivy` dans CI ni `requirements.txt`.
- Pas de fuzzing / property-based actif (Hypothesis spec livrée `reports/eval_17_property_specs.py` mais **non intégrée** dans `tests/test_property_based.py`).
- Pas de scan OWASP ZAP / OAS-schema fuzzer sur les endpoints API.

---

## 2. Vision cible

| Dimension | Cible commercialisable |
|---|---|
| CI GitHub Actions | **3 workflows** : `ci.yml` (lint+type+unit), `integration.yml` (E2E docker-compose), `nightly.yml` (load+mutation+security) |
| Lint/type | **Bloquant** : `ruff check` (zero error), `black --check`, `mypy --strict` sur `src/api`, `src/intelligence`, `src/delivery` |
| Coverage global | **≥ 75 %** |
| Coverage zones revenue (telegram, signal_store, confluence_detector, sentinel_scanner, volatility_forecaster, auth, tier_manager, insight_signal_v2, billing) | **≥ 85 %** |
| Runtime CI PR (unit) | **< 5 min** sur ubuntu-latest avec `pytest-xdist -n auto` |
| Runtime CI nightly | **< 30 min** (load+mutation subset+security scans) |
| Flaky rate | **0 %** sur 50 runs consécutifs (mesuré par `pytest-rerunfailures --rerun 0` baseline) |
| Mutation score (zones revenue) | **≥ 70 %** (mutmut) |
| Property-based tests | **6 specs Hypothesis** intégrés + 4 nouveaux (resample MTF, billing prorata, idempotency, webhook signature) |
| Load test SLA | API `/insight` p95 < 250 ms à 50 RPS (Locust headless) |
| Contract test | OpenAPI snapshot diff (`schemathesis stateful`) ; Telegram payload Markdown JSON-schema |
| Security scan | Bandit + pip-audit + Trivy image + Semgrep → bloquant si CVSS ≥ 7.0 |
| Test data | DVC remote (Backblaze B2 ~0,80 $/an pour 165 MB) + 2 golden fixtures 7-day committables (~100 KB chacune) |

---

## 3. Gap analysis

| Pilier cible | Présent ? | Gap principal | Effort cible |
|---|---|---|---|
| pytest.ini complet (addopts, markers, strict, timeout) | Partiel (12 lignes) | Manque addopts, strict-markers, strict-config, timeout, filterwarnings ciblés | 1 h |
| `.coveragerc` | Absent | Coverage biaisée par imports conditionnels | 1 h |
| Mypy CI bloquant | Absent | Pas de type-checking → bugs runtime | 8 h (corriger les erreurs) |
| GitHub Actions full suite | Partiel (26 fichiers) | 138 fichiers non gardés | 6 h |
| Coverage telegram_notifier | 10 % → 85 % | Envoi/retry/lang/markdown injection non testés | 10 h |
| Coverage volatility_forecaster | 63 % → 85 % | HMM bootstrap, calibration windows, conformal intervals | 12 h |
| Coverage sentinel_scanner | 59 % → 85 % | Polling, queue, multi-symbol | 10 h |
| Coverage main.py FastAPI | 36 % → 80 % | Lifespan startup/shutdown, error handlers | 6 h |
| Coverage routes/state.py | 18 % → 80 % | Endpoints consultation snapshot | 4 h |
| Coverage signal_store | 66 % → 90 % | Backup/restore, race, schema migrations | 6 h |
| Fix flaky `test_short_roundtrip_pnl` | Existant | RNG non seedé `env.reset()` (env.py:507) | 30 min |
| Fix broken `test_long_short_trading.py` | Existant | Import `src.config` inexistant | 15 min |
| `test_env_debug.py` racine | Existant | Pollue collection | 15 min |
| Property-based Hypothesis | Spec livrée non intégrée | 6 specs `reports/eval_17_property_specs.py` à copier + 4 nouveaux | 6 h |
| Integration tests b2c+b2b | Partiel | Mockups `mockups/b2b_*.json` non backed par tests contract | 8 h |
| Load tests (Locust) | Absent | API non load-tested | 10 h |
| Contract tests OpenAPI | Absent | Schéma drift possible | 6 h |
| Mutation testing | Absent | 5 modules critiques cibles | 8 h setup + nightly 22 h CPU |
| Security scans (Bandit/pip-audit/Trivy/Semgrep) | Absent | CVE jamais détectées | 6 h |
| DVC fixtures | Absent (165 MB raw) | Données indisponibles en CI | 8 h |
| Test pyramid metrics | Absent | Pas de mesure unit/integration/e2e ratio | 2 h |

**Gap total estimé : ~140 h dev + ~22 h CPU/mois nightly mutation**.

---

## 4. Plan d'exécution

> Convention : `[Px]` = priorité ; `Eh` = heures ; `Ax` = acceptance criteria ; `Dep` = dépendances.

### 4.1 P0 — Fondations bloquantes (semaine 1, 18 h)

#### P0-1 — Nettoyage suite : flaky + broken collection
- **Fichiers** :
  - `tests/test_sprint1_short_rollback.py:233` → `env.reset(seed=42)`
  - `tests/test_sprint1_short_rollback.py:_step_n` → propager seed
  - `tests/conftest.py:+150-160` → ajouter `_seed_global_rng` autouse fixture
  - `git rm tests/test_long_short_trading.py` (RL legacy mort, redondant avec `test_sprint1_short_rollback.py`)
  - `git mv test_env_debug.py scripts/debug_env.py` (script diagnostic, pas un test)
  - `pytest.ini` → `testpaths = tests` (déjà présent ligne 2, vérifier exclusion racine)
- **Heures** : 2 h
- **A** : `pytest tests/ --collect-only` retourne 0 error ; `pytest tests/test_sprint1_short_rollback.py::test_short_roundtrip_pnl --count=50` = 50/50 pass (avec `pytest-repeat`)
- **Dep** : aucune
- **Risque** : bas — patch chirurgical déjà documenté `reports/eval_17_testing.md:115-131`

#### P0-2 — Configuration pytest + coverage durcie
- **Fichiers** :
  - `pytest.ini` réécrit avec `addopts = -ra --strict-markers --strict-config --timeout=60 --maxfail=50 --tb=short --durations=20`
  - `pytest.ini` markers complétés : `slow`, `integration`, `property`, `e2e`, `load`, `live`, `benchmark`, `security`, `revenue_critical`
  - `pytest.ini` `filterwarnings` durcis (pas de blanket `ignore::DeprecationWarning`)
  - **Nouveau** `.coveragerc` : `source = src`, `branch = True`, `omit = src/multi_asset/*, src/training/*, src/agents/legacy/*`, `exclude_lines = TYPE_CHECKING, NotImplementedError, __main__`, `fail_under = 70`
  - **Nouveau** `pyproject.toml` partiel : `[tool.ruff]`, `[tool.black]`, `[tool.mypy]` (strict sur `src/api`, `src/intelligence/insight_assembler.py`, `src/intelligence/confluence_detector.py`, `src/delivery/`)
- **Heures** : 3 h
- **A** : `pytest --collect-only -q` ≤ 0 error ; `pytest --cov` produit rapport branch ; `ruff check` + `black --check` exit 0
- **Dep** : P0-1

#### P0-3 — CI GitHub Actions élargie + bloquante
- **Fichiers** :
  - `.github/workflows/ci.yml` réécrit :
    - Job `lint` → **bloquant** (retirer `continue-on-error: true` ligne 47/53)
    - Job `type` → nouveau, `mypy --strict src/api src/intelligence/insight_assembler.py src/delivery`
    - Job `test-fast` → `pytest -n auto -m "not slow and not integration and not e2e and not load"` avec `--cov-fail-under=70` global
    - Job `test-revenue` → matrice ciblée sur zones revenue avec `--cov-fail-under=85` sur `src/api/auth.py src/api/tier_manager.py src/api/signal_store.py src/api/insight_signal_v2.py src/intelligence/confluence_detector.py src/intelligence/signal_state_machine.py src/delivery/telegram_notifier.py`
    - Job `coverage-report` → `actions/upload-artifact` + commentaire PR via `py-cov-action/python-coverage-comment-action`
  - `.github/workflows/algo_tests.yml` → conserver mais ajouter `--cov` + upload artifact
  - `.github/workflows/nightly.yml` (nouveau) → load + mutation + security scans, cron quotidien 03:00 UTC
- **Heures** : 5 h
- **A** : PR sur `main` ouvre 4 jobs verts en < 5 min ; nightly cron tourne en < 30 min
- **Dep** : P0-2

#### P0-4 — Coverage `telegram_notifier` 10 % → 85 %
- **Fichiers** :
  - `tests/test_telegram_notifier_send.py` (nouveau) → couvre `send_signal`, `send_narrative`, retries `tenacity`, circuit breaker, rate limit, escape Markdown injection
  - `tests/test_telegram_notifier_lang.py` (existe partiellement `test_telegram_lang.py`) → multi-langue FR/EN/DE/ES
  - `tests/test_telegram_notifier_markdown_injection.py` → fuzz Markdown special chars (`*_[]()~`)
  - `tests/fixtures/telegram_mock.py` → fixture `httpx_mock` pour `api.telegram.org`
- **Heures** : 10 h
- **A** : `coverage` sur `src/delivery/telegram_notifier.py` ≥ 85 % ; couvre 100 % chemins retry + rate-limit + circuit breaker
- **Dep** : P0-3
- **Note** : eval_10-15 §Telegram a identifié markdown injection comme risque, ce test est **revenue-critical** car Telegram = canal B2C primaire

#### P0-5 — Coverage `volatility_forecaster` 63 % → 85 %
- **Fichiers** :
  - `tests/test_volatility_forecaster_hmm.py` (nouveau) → `fit_hmm`, `predict_regime`, bootstrap edge cases, `n_states=2/3/4`
  - `tests/test_volatility_forecaster_calendar.py` (nouveau) → `event_window_hours`, NFP/CPI/FOMC overlap
  - `tests/test_volatility_forecaster_conformal.py` (nouveau) → conformal interval calibration, miscoverage rate
  - `tests/test_volatility_forecaster_blend.py` (nouveau) → blend weights HAR+LGBM, edge case `blend_w=0/1`
- **Heures** : 12 h
- **A** : coverage ≥ 85 % sur `src/intelligence/volatility_forecaster.py`, branches HMM ≥ 80 %
- **Dep** : P0-3

#### P0-6 — Coverage `sentinel_scanner` 59 % → 85 %
- **Fichiers** :
  - `tests/test_sentinel_scanner_polling.py` (nouveau) → polling loop, interval 60 s, graceful shutdown
  - `tests/test_sentinel_scanner_multi_symbol.py` → XAUUSD + EURUSD parallèle
  - `tests/test_sentinel_scanner.py` existant → compléter notification queue, kill switch interaction
- **Heures** : 10 h
- **A** : coverage ≥ 85 % sur `src/intelligence/sentinel_scanner.py`
- **Dep** : P0-3

### 4.2 P1 — Couverture étendue + intégration (semaines 2-3, 50 h)

#### P1-1 — Property-based testing (Hypothesis)
- **Fichiers** :
  - `cp reports/eval_17_property_specs.py tests/test_property_based.py` (6 specs : score bounds, regime monotonicity, no-flip BUY/SELL, confirm_bars, resample OHLC invariants, reject upsampling)
  - `tests/test_property_based_billing.py` (nouveau) → prorata Stripe edge cases (date frontières mois, refund, upgrade mi-période)
  - `tests/test_property_based_idempotency.py` (nouveau) → `src/audit/idempotency_store.py` clé collision/expiration
  - `tests/test_property_based_webhook_signer.py` → HMAC SHA-256 invariants
  - `requirements.txt` → ajouter `hypothesis==6.99.4`
- **Heures** : 6 h
- **A** : `pytest tests/test_property_based*.py -m property -v --hypothesis-show-statistics` → 100 % pass, ≥ 50 examples par spec
- **Dep** : P0

#### P1-2 — Integration tests dual B2C + B2B
- **Fichiers** :
  - `tests/integration/test_b2c_telegram_flow.py` (nouveau) → mockups `mockups/telegram_b2c.txt` vs. payload réel
  - `tests/integration/test_b2b_webhook_flow.py` (nouveau) → mockups `mockups/b2b_webhook_payload.json` vs. émis
  - `tests/integration/test_b2b_insight_endpoint.py` (nouveau) → contrat `mockups/b2b_insight.json` validé via JSON Schema
  - `tests/integration/conftest.py` → fixture `webhook_receiver` (FastAPI test client tier)
- **Heures** : 8 h
- **A** : 3 flow integration verts, mockups validés byte-for-byte (ou diff explicite avec délta documenté), marker `@pytest.mark.integration` 
- **Dep** : P0, dual architecture livrée

#### P1-3 — Load tests Locust sur API
- **Fichiers** :
  - `tests/load/locustfile.py` (nouveau) → 3 scénarios : `/insight` GET, `/insight/v2` POST, `/health` GET
  - `tests/load/load_targets.yml` → SLA p50/p95/p99 par endpoint
  - `.github/workflows/nightly.yml` → ajouter step Locust headless 50 RPS × 5 min
  - `requirements-dev.txt` (nouveau) → `locust==2.20.0`
- **Heures** : 10 h
- **A** : `/insight` p95 < 250 ms à 50 RPS sur ubuntu-latest ; rapport Locust HTML uploadé artifact
- **Dep** : P0-3

#### P1-4 — Contract tests OpenAPI (Schemathesis)
- **Fichiers** :
  - `tests/contract/test_openapi_schema.py` → snapshot `src/api/app.py:/openapi.json` vs. `tests/contract/openapi_snapshot.json` committé
  - `tests/contract/test_openapi_stateful.py` → `schemathesis run --checks all --hypothesis-deadline=2000` contre serveur Uvicorn test
  - `tests/contract/test_telegram_payload_schema.py` → JSON Schema des payloads sortants Telegram (validation chain)
  - `requirements-dev.txt` → `schemathesis==3.27.0`, `jsonschema==4.21.0`
- **Heures** : 6 h
- **A** : 0 drift sur openapi.json (snapshot diff bloquant) ; Schemathesis 0 failure
- **Dep** : P0-3

#### P1-5 — Coverage routes/main/signal_store
- **Fichiers** :
  - `tests/test_routes_state_endpoints.py` → coverage `src/api/routes/state.py` 18 % → 80 %
  - `tests/test_main_lifespan.py` → coverage `src/intelligence/main.py` 36 % → 80 % (startup/shutdown hooks)
  - `tests/test_signal_store_migrations.py` → schema migrations + backup/restore (eval_10-15 a flagué backup absent)
- **Heures** : 12 h
- **A** : coverage agrégée ≥ 75 % sur ces 3 modules
- **Dep** : P0

#### P1-6 — DVC fixtures + golden datasets
- **Fichiers** :
  - `dvc.yaml` (nouveau) → pipeline `download_xau` + `download_eurusd` ré-générable
  - `data/.dvc` config → remote Backblaze B2 (~0,80 $/an / 165 MB)
  - `tests/fixtures/xau_m15_2024_07_01_to_07_07.csv` (committé, ~50 KB)
  - `tests/fixtures/eurusd_m15_2024_07_01_to_07_07.csv` (committé, ~50 KB)
  - `tests/fixtures/CHECKSUMS.txt` → SHA-256
  - `tests/fixtures/data_loader.py` → `load_xau_7day_golden()`, `load_eurusd_7day_golden()` (cf. eval_17 §T9 lignes 404-415)
- **Heures** : 8 h
- **A** : `dvc pull` en CI < 30 s ; golden datasets utilisables sans DVC pour 80 % des tests unit
- **Dep** : P0-3

### 4.3 P2 — Optimisations & maturité (semaine 4+, 40 h)

#### P2-1 — Mutation testing (mutmut) sur zones revenue
- **Fichiers** :
  - `.mutmut-config` (nouveau) → `paths_to_mutate = src/intelligence/confluence_detector.py,src/intelligence/signal_state_machine.py,src/api/auth.py,src/api/tier_manager.py,src/api/insight_signal_v2.py`
  - `scripts/run_mutation_testing.sh` → `mutmut run --runner "pytest -x -q --timeout=10 tests/test_confluence_detector.py tests/test_signal_state_machine.py tests/test_auth.py tests/test_tier_manager.py tests/test_insight_signal_v2.py"`
  - `.github/workflows/nightly.yml` → step mutmut weekly (Sunday 02:00 UTC) avec gate `mutation_score >= 70`
  - `requirements-dev.txt` → `mutmut==2.5.1`
- **Heures** : 8 h setup + ~22 h CPU/run (eval_17 §T5 estime 18-22 h)
- **A** : Top 5 mutants survivants (identifiés `reports/eval_17_testing.md:202-217`) tués par tests dédiés ; mutation score ≥ 70 %
- **Dep** : P0-3, P1
- **Note** : eval_17 §T10 Q1 considère mutation low-ROI avant 100 MAU. **Décision** : démarrer en *nightly weekly* (1 run/semaine = ~7 h/mois GitHub Actions = ~0,70 $/mois) pour habituer l'équipe, gate seulement à partir de 50 MAU

#### P2-2 — Security scans en CI
- **Fichiers** :
  - `.github/workflows/security.yml` (nouveau) :
    - `bandit -r src/ -ll` (severity ≥ medium, gate)
    - `pip-audit` (CVE deps, gate CVSS ≥ 7)
    - `safety check`
    - `semgrep --config=auto src/` (rules OWASP top-10)
    - `trivy image sentinel-ai:latest` (image scan)
    - `gitleaks detect` (secret scan)
  - `.github/workflows/zap.yml` (nouveau, optional) → OWASP ZAP baseline scan contre staging hebdomadaire
  - `.bandit` config → exclude `tests/`, `scripts/`
  - `requirements-dev.txt` → `bandit==1.7.7`, `pip-audit==2.7.0`, `safety==3.1.0`, `semgrep==1.55.0`
- **Heures** : 6 h
- **A** : 0 CVE CVSS ≥ 7 ; 0 bandit MEDIUM+ ; 0 secret détecté ; rapport SARIF uploadé GitHub Security tab
- **Dep** : P0-3

#### P2-3 — E2E docker-compose
- **Fichiers** :
  - `infrastructure/docker-compose.test.yml` → spec eval_17 §T7 lignes 250-285 (sentinel + smoke sidecar curl)
  - `scripts/smoke_e2e.sh` → wrapper docker compose
  - `.github/workflows/e2e.yml` (nouveau) → build cached + healthcheck + smoke, ubuntu-latest uniquement
- **Heures** : 6 h
- **A** : E2E cold-start < 110 s, warm < 60 s ; smoke endpoint `/health` + `/api/v1/signals` répond 200
- **Dep** : P0-3, P1-6

#### P2-4 — Métriques test pyramid + flaky tracking
- **Fichiers** :
  - `scripts/test_pyramid_metrics.py` → calcule ratio unit/integration/e2e/property/load à partir markers ; sortie JSON `reports/test_pyramid.json`
  - `.github/workflows/nightly.yml` → step `pytest-rerunfailures --rerun 3` en mode tracking, agrège fails par test → flaky dashboard
  - `tools/flaky_tracker.py` → log fail rate par test sur 30 derniers runs ; alerte si > 1 %
- **Heures** : 4 h
- **A** : Dashboard test pyramid mis à jour à chaque PR ; 0 test avec fail rate > 1 % sur 30 runs
- **Dep** : P0-3

#### P2-5 — Coverage RAG + LLM eval harness en CI bloquant
- **Fichiers** :
  - `tests/eval_llm/regression_gate.py` (existe) → faire passer en step CI nightly bloquant
  - `.github/workflows/nightly.yml` → ajouter `pytest tests/eval_llm/ --gate-baseline=tests/eval_llm/rag_baseline.json`
- **Heures** : 4 h
- **A** : Toute régression RAG > 5 % bloque le merge (déjà patterné par `regression_gate.py`)
- **Dep** : P0-3

#### P2-6 — Coverage `discord_notifier` 13 % → 70 %
- **Fichiers** :
  - `tests/test_discord_notifier.py` (compléter) → webhook send, retries, rate limit
- **Heures** : 4 h
- **A** : coverage ≥ 70 % sur `src/delivery/discord_notifier.py`
- **Dep** : P0
- **Note** : Discord est canal B2C secondaire, priorité moindre que Telegram

### 4.4 P3 — Différé post-PMF (>100 MAU)

| Item | Pourquoi différé |
|---|---|
| Mutation testing gate bloquant 70 % | ROI faible avant 100 MAU (eval_17 §T10 Q1) — actuellement en nightly informatif |
| Chaos engineering (Toxiproxy/Pumba) | Pas de multi-instance pour l'instant |
| Performance regression bench continuous | Locust suffit jusqu'à 1 000 RPS |
| Browser E2E (Playwright) sur webapp B2C | Webapp encore en mockup `mockups/webapp_b2c.html` |
| Visual regression testing | Webapp pas livrée |

---

## 5. Validation (test pyramid + métriques)

### 5.1 Ratios cibles

| Type | Count target | % suite | Runtime budget |
|---|---|---|---|
| Unit | 2 000+ | 75 % | < 3 min `-n auto` |
| Integration | 200 | 8 % | < 2 min |
| E2E (docker) | 10-15 | 1 % | < 2 min cold, < 1 min warm |
| Property (Hypothesis) | 10 specs | 1 % | < 1 min `-n 2` |
| Load (Locust nightly) | 3 scénarios | hors PR | < 10 min |
| Contract (Schemathesis) | 1 stateful run | hors PR | < 3 min |
| Mutation (mutmut weekly) | 5 modules | hors PR | ~22 h CPU |

### 5.2 Gates par environnement

| Gate | PR | Merge main | Release tag |
|---|---|---|---|
| Lint `ruff` 0 error | ✅ | ✅ | ✅ |
| Type `mypy --strict` (scope défini) | ✅ | ✅ | ✅ |
| Unit tests | ✅ | ✅ | ✅ |
| Coverage global ≥ 70 % | ✅ | ✅ | ✅ |
| Coverage revenue zones ≥ 85 % | ✅ | ✅ | ✅ |
| Integration | ⚠️ (slow → nightly) | ✅ | ✅ |
| E2E docker-compose | ⚠️ | ✅ | ✅ |
| Property-based | ✅ | ✅ | ✅ |
| Contract test (OpenAPI snapshot) | ✅ | ✅ | ✅ |
| Load test SLA | ❌ | nightly | ✅ |
| Mutation score ≥ 70 % (revenue) | ❌ | nightly informatif | ⚠️ (warn) |
| Security : Bandit/pip-audit/Trivy | ✅ | ✅ | ✅ |
| RAG eval regression gate | ❌ | nightly | ✅ |

### 5.3 Mutation score cible par module

| Module | Cible mutation score |
|---|---|
| `confluence_detector.py` | 75 % |
| `signal_state_machine.py` | 80 % |
| `auth.py` | 85 % |
| `tier_manager.py` | 80 % |
| `insight_signal_v2.py` | 70 % |

---

## 6. Sécurité (security scan en CI)

### 6.1 Outils + cadence

| Outil | Cadence | Gate | Action si fail |
|---|---|---|---|
| `bandit` | Every PR | MEDIUM+ bloquant | Fix obligatoire avant merge |
| `pip-audit` | Every PR + nightly | CVSS ≥ 7 bloquant | Upgrade dep ou pin patched version |
| `safety check` | Every PR | Bloquant | Fix |
| `semgrep --config=auto` | Every PR | ERROR severity bloquant | Fix |
| `trivy image` | Every release tag | CRITICAL bloquant | Re-build image |
| `gitleaks detect` | Every PR | Bloquant | Rotate secret + history rewrite si committé |
| `OWASP ZAP baseline` | Weekly contre staging | Informatif | Ticket si high alert |
| `schemathesis stateful` | Every PR | Bloquant | Fix endpoint |

### 6.2 Couverture OWASP API Top 10

| Risque OWASP | Test associé | Localisation |
|---|---|---|
| API1 Broken Object Level Auth | `tests/test_auth.py` + property-based ownership | À ajouter property tests |
| API2 Broken Authentication | `tests/test_auth.py:KeyStore` | Existant ≥ 92 % |
| API3 Excessive Data Exposure | Contract test OpenAPI snapshot | P1-4 |
| API4 Lack of Rate Limit | `tests/test_rate_limit_headers.py`, `tests/test_tier_rate_limiter.py` | Existant |
| API5 BOLA Function Auth | `tests/test_auth.py` | Existant |
| API6 Mass Assignment | Schemathesis | P1-4 |
| API7 Security Misconfig | Trivy + bandit | P2-2 |
| API8 Injection | Markdown injection tests + Semgrep | P0-4 + P2-2 |
| API9 Improper Asset Management | OpenAPI snapshot diff | P1-4 |
| API10 Insufficient Logging | `tests/test_access_log_middleware.py` | Existant |

---

## 7. Métriques

### 7.1 Coverage par module (cible vs. actuel)

| Module | Actuel | Cible | Δ |
|---|---|---|---|
| `src/delivery/telegram_notifier.py` | 10 % | **85 %** | +75 |
| `src/delivery/discord_notifier.py` | 13 % | 70 % | +57 |
| `src/api/routes/state.py` | 18 % | 80 % | +62 |
| `src/intelligence/main.py` | 36 % | 80 % | +44 |
| `src/api/routes/dashboard.py` | 36 % | 70 % | +34 |
| `src/api/routes/narratives.py` | 35 % | 75 % | +40 |
| `src/api/routes/operator.py` | 18 % | 70 % | +52 |
| `src/api/routes/signals.py` | 47 % | 80 % | +33 |
| `src/api/routes/health.py` | 49 % | 80 % | +31 |
| `src/intelligence/data_providers.py` | 40 % | 75 % | +35 |
| `src/intelligence/volatility_forecaster.py` | 63 % | **85 %** | +22 |
| `src/intelligence/sentinel_scanner.py` | 59 % | **85 %** | +26 |
| `src/api/signal_store.py` | 66 % | 90 % | +24 |
| `src/intelligence/confluence_detector.py` | 82 % | 90 % | +8 |
| `src/intelligence/signal_state_machine.py` | 91 % | 95 % | +4 |
| `src/api/auth.py` | 92 % | 95 % | +3 |
| `src/api/tier_manager.py` | 99 % | 99 % | 0 |
| `src/intelligence/volatility_lgbm.py` | 0 % | 70 % | +70 |

**Coverage global cible : 75 %** (vs. 53 % actuel) — atteignable après P0+P1.

### 7.2 Test runtime tracking

| Métrique | Baseline | Cible | Mesure |
|---|---|---|---|
| Suite full local (séquentiel) | 9-12 min | 5-7 min | `pytest --durations=20` |
| Suite full CI (`-n auto`) | n/a | < 5 min | GitHub Actions duration |
| Top 20 tests les plus lents | eval_17 §A.2 | < 5 s chacun | `--durations=20` |
| Hypothesis specs | n/a | < 60 s total | `--hypothesis-show-statistics` |

### 7.3 Flaky rate

| Métrique | Cible | Mesure |
|---|---|---|
| Flaky test count | 0 (≤ 1 % fail rate sur 30 runs) | `tools/flaky_tracker.py` |
| Test_short_roundtrip_pnl | 0 fail / 50 runs | `pytest --count=50` après P0-1 |
| Re-run rate en CI | < 0,5 % | `pytest-rerunfailures` stats |

### 7.4 Mutation score (nightly weekly)

| Module | Cible | Tolérance |
|---|---|---|
| confluence_detector | 75 % | -2 pts → warn, -5 pts → block |
| signal_state_machine | 80 % | -2 pts → warn, -5 pts → block |
| auth | 85 % | -2 pts → warn, -5 pts → block |
| tier_manager | 80 % | -2 pts → warn, -5 pts → block |

---

## 8. Risques & mitigations

| Risque | Impact | Proba | Mitigation |
|---|---|---|---|
| Coût GitHub Actions explose (>20 $/mois) | M | M | Cache pip+pytest-cache, limit nightly mutation à 1 run/sem, fast-fail PRs avec `--maxfail=10` |
| Tests revenue critical flaky | H | M | Marker `@pytest.mark.revenue_critical` + 0 tolerance, `pytest-rerunfailures` désactivé sur cette zone (un fail = fail) |
| False positives Hypothesis (deadline trop bas Windows) | M | M | `suppress_health_check=[too_slow]` + `deadline=3000`, exec en CI Linux uniquement |
| Mutation testing tue le throughput dev | H | L | Confiner en nightly weekly, ne pas gater PR |
| Coverage gate 85 % casse PRs feature non-revenue | M | M | Gate différencié : 70 % global, 85 % UNIQUEMENT zones revenue listées explicitement |
| DVC remote indisponible (Backblaze down) | M | L | Golden 7-day fixtures committées Git pour 80 % des tests |
| Test rot (tests obsolètes non maintenus) | M | H | Marker `@pytest.mark.deprecated_target` + revue trimestrielle ; `pytest --co` audit chaque sprint |
| Type-checking mypy `--strict` génère 500+ erreurs initiales | M | H | Scope progressif : commencer `src/api/insight_signal_v2.py`, étendre 1 module/sprint |
| Security scan false positives (Semgrep) | M | M | `# nosemgrep:` annotations ciblées + revue à 4 yeux |
| Load test contention sur ressources CI partagées | L | M | Locust headless avec `--users=50 --spawn-rate=10`, scope contained |
| OpenAPI snapshot churn excessif (chaque endpoint touché) | M | H | Snapshot **regénérable** via `make update-openapi-snapshot` avec approbation reviewer dans la PR |

---

## 9. Dépendances

### 9.1 Dépendances inter-catégories commercialisation

| Catégorie | Dépendance pour testing | Bloquante ? |
|---|---|---|
| **01 Architecture** | Refactor pipeline → tests pipeline à mettre à jour | Non, parallèle |
| **02 ConfluenceDetector** | Replacement scoring fn → property tests doivent suivre | Non, parallèle |
| **08 Data Providers** | Fixtures `xau_mini.csv` shippable | Oui pour P1-6 DVC |
| **10 API Auth/Store/Telegram** | Tests dépendent des fix eval_10-15 (markdown injection, backup) | Oui pour P0-4 / P1-5 |
| **12 Deployment** | CI deploy gate dépend de tests verts | Oui — CI test est *préalable* à CD |
| **16 Observability** | Tests métriques `/metrics` endpoint | Non, complémentaire |
| **19 Risk** | Tests kill-switch escalation | Existant, à compléter en P1 |
| **29 Compliance** | Tests geo-block, disclaimers | Existant (`test_geo_block.py`, `test_disclaimers.py`) |

### 9.2 Dépendances outils externes

- **GitHub Actions** (existant) — quota 2 000 min/mois Free tier suffisant si runtime < 5 min × 50 PRs/mois = 250 min
- **Backblaze B2** (à créer) — ~0,80 $/an pour 165 MB DVC remote
- **Codecov** (optional) — gratuit pour open-source, $10/mo private — alternative : `coverage` HTML uploadé GitHub Actions artifact
- **Stripe test mode** — pour `tests/test_billing.py` (existant)

### 9.3 Dépendances Python (à ajouter)

```text
# requirements-dev.txt (nouveau)
pytest>=7.0.0                      # déjà présent
pytest-asyncio>=0.21.0              # déjà présent
pytest-cov>=4.0.0                   # déjà présent
pytest-xdist==3.5.0                 # parallel
pytest-timeout==2.2.0               # timeout protection
pytest-rerunfailures==13.0          # flaky tracking
pytest-repeat==0.9.3                # --count=N
pytest-recording==0.13.1            # VCR.py for LLM snapshots
pytest-mock==3.12.0                 # déjà ?
hypothesis==6.99.4                  # property-based
schemathesis==3.27.0                # OpenAPI fuzzing
locust==2.20.0                      # load tests
mutmut==2.5.1                       # mutation testing
bandit==1.7.7                       # security
pip-audit==2.7.0                    # CVE
safety==3.1.0                       # CVE
semgrep==1.55.0                     # SAST
ruff==0.6.9                         # lint (déjà CI)
black==24.10.0                      # format (déjà CI)
mypy==1.8.0                         # type-check
dvc[s3]==3.45.0                     # data versioning
```

---

## 10. Estimation totale & timeline

### 10.1 Récap heures

| Phase | Heures dev | CPU/mois |
|---|---|---|
| P0 (fondations) | 42 h | ~0 |
| P1 (couverture + intégration) | 50 h | ~10 h CI |
| P2 (maturité) | 32 h | ~30 h CI/mois (mutation+load+security nightly) |
| **Total commercialisation-ready** | **124 h** | ~40 h CPU/mois (~$4/mois) |
| P3 (post-PMF) différé | n/a | n/a |

### 10.2 Timeline proposée (1 dev plein temps)

| Sprint | Semaine | Livrable principal | Status gate |
|---|---|---|---|
| S1 | S+1 | P0-1, P0-2, P0-3, P0-4 → suite verte, CI bloquante, Telegram 85 % | Coverage global 60 %, 0 flaky, 0 broken |
| S2 | S+2 | P0-5, P0-6 → volatility 85 %, scanner 85 % | Coverage global 68 % |
| S3 | S+3 | P1-1, P1-2, P1-5 → property + integration b2c+b2b + main/routes | Coverage global 73 % |
| S4 | S+4 | P1-3, P1-4, P1-6 → load + contract + DVC | Coverage global 75 %, SLA p95 validé |
| S5 | S+5 | P2-1, P2-2 → mutation nightly + security scans | Mutation score baseline mesuré |
| S6 | S+6 | P2-3, P2-4, P2-5, P2-6 → E2E docker + flaky tracker + RAG gate | **GO commercialisation** |

### 10.3 Coût opérationnel mensuel CI

| Item | Coût/mois |
|---|---|
| GitHub Actions (~250 min PR + ~600 min nightly = 850 min) | $0 (sous 2 000 min Free tier) |
| Backblaze B2 storage 165 MB | ~$0,01 |
| Backblaze B2 BW (estimation 50 GB/mois CI pulls) | ~$0,50 |
| Codecov private (optionnel) | $10 ou $0 si artifact-only |
| **Total** | **~$0,5 à $11/mois** |

### 10.4 Critères de sortie GO commercialisation

- [ ] Coverage global ≥ 75 %
- [ ] Coverage revenue zones ≥ 85 % (telegram, scanner, volatility, signal_store, auth, tier_manager, insight_signal_v2, confluence_detector, signal_state_machine)
- [ ] 0 test flaky sur 50 runs consécutifs
- [ ] 0 broken collection
- [ ] CI PR < 5 min ; nightly < 30 min
- [ ] Lint bloquant + Type-check sur scope défini + Security scans verts
- [ ] OpenAPI snapshot test en place
- [ ] Load test SLA validé : `/insight` p95 < 250 ms à 50 RPS
- [ ] Mutation score baseline mesuré (gate informatif d'abord)
- [ ] RAG regression gate actif en nightly
- [ ] DVC + golden fixtures shippés

---

## Annexe — Cartographie des 164 fichiers test

**Catégories** :
- API/Routes : 16 fichiers (`test_api.py`, `test_audit_endpoint.py`, `test_*_endpoint.py`, `test_routes_*`)
- Auth/Billing : 6 fichiers (`test_auth.py`, `test_billing.py`, `test_key_rotation.py`, `test_tier_manager.py`, `test_tier_rate_limiter.py`, `test_pricing_ab.py`)
- Intelligence/Confluence : 8 fichiers
- Volatility : 4 fichiers (`test_vol_*.py`, `test_volatility_forecaster.py`, `test_lgbm_vol.py`, `test_hybrid_vol.py`)
- State machine / Replay : 4 fichiers
- Smart Money (BOS/CHOCH/OB/FVG) : 6 fichiers
- News pipeline : 3 fichiers
- RAG / LLM : 12 fichiers (sous-suite mature avec `eval_llm/` harness)
- Risk / Kill Switch : 5 fichiers
- Delivery (Telegram/Discord/Webhook) : 7 fichiers
- Compliance (geo-block, disclaimers, langue) : 5 fichiers
- Observability : 4 fichiers
- Backtest / CPCV / Walk-forward : 6 fichiers
- Sprints legacy (RL training) : ~25 fichiers (audit pour suppression progressive)
- Smoke / E2E / Production wiring : 3 fichiers
- Phase 2B / A1 / Conformal / Regime : 10 fichiers
- Misc divers : ~40 fichiers

**Action backlog post-P0** : audit des ~25 fichiers `test_sprint*` legacy RL → distinguer ceux qui couvrent encore du code Smart Sentinel actif vs. RL legacy mort (à supprimer comme `test_long_short_trading.py`).

---

## Synthèse exécutive

**Chemin** : `C:\MyPythonProjects\TradingBOT_Agentic\reports\commercialization_sprint\13_testing_infrastructure.md`

**Top 3 priorités P0** :
1. **P0-1 + P0-2 + P0-3 (10 h)** — Suite verte déterministe (fix flaky `test_short_roundtrip_pnl` ligne 233, suppression `test_long_short_trading.py`, exclusion `test_env_debug.py`) + `pytest.ini` durci + `.coveragerc` + CI GitHub Actions bloquante remplaçant les jobs advisory actuels. **Sans ces fondations, tout le reste est instable.**
2. **P0-4 (10 h)** — Coverage `src/delivery/telegram_notifier.py` 10 % → 85 % avec couverture send/retry/rate-limit/markdown-injection/multi-lang. **Telegram = canal B2C primaire ; un bug ici = ticket support + churn immédiat.**
3. **P0-5 (12 h)** — Coverage `src/intelligence/volatility_forecaster.py` 63 % → 85 % sur HMM/calendar/conformal/blend. **Vol = composant ML le plus risqué du pipeline ; à protéger avant tout passage en mode hybride/lgbm en prod.**

**Heures totales pour GO commercialisation** : **124 h dev** (~3-4 semaines plein temps) + **~$4/mois CI** (sous Free tier GitHub Actions).
