# No-Regret Quick Wins — Exécutables Immédiatement Sans Risque

> **Date** : 2026-05-23
> **Auteur** : Architecte Master Plan
> **Périmètre** : items extrêmement sûrs, isolés, additifs ou correctifs évidents. Cible 30 items < 2h chacun, exécutables MAINTENANT en parallèle sans approbation détaillée.
> **Source** : extraction systématique des 20 plans `reports/commercialization_sprint/{01..20}_*.md`.

Pour chaque item : ID, source, description 1 ligne, fichiers, effort, acceptance criteria, impact.

---

## QW-001 — Supprimer `tests/test_long_short_trading.py`
- **Source** : Cat 13 §1.1 (`13_testing_infrastructure.md:138-171`)
- **Description** : fichier de test à `D` dans git status, import `src.config` brisé.
- **Fichiers** : `tests/test_long_short_trading.py`
- **Effort** : 5 min (`git rm tests/test_long_short_trading.py && git commit`).
- **Acceptance** : `pytest --collect-only` zero erreur collection, plus de mention `test_long_short_trading`.
- **Impact** : débloque activation CI bloquante stricte.

## QW-002 — Vérifier `SCORE_BUCKET_PTS=10` + test régression
- **Source** : Cat 17 §1.1 (`17_caching_performance.md:104`)
- **Description** : `SemanticCache.SCORE_BUCKET_PTS` déjà à 10 selon plan (×4.3 hit rate). Vérifier code current + ajouter test régression.
- **Fichiers** : `src/intelligence/semantic_cache.py:104`, `tests/test_semantic_cache.py` (nouveau test).
- **Effort** : 30 min.
- **Acceptance** : test fail si `SCORE_BUCKET_PTS != 10` ; CI vert.
- **Impact** : prévient régression future de la bonne valeur.

## QW-003 — Vérifier `TESTING_MODE` default `"0"` + ajouter gate CI
- **Source** : Cat 10 F-01 (`10_auth_security.md:24`)
- **Description** : F-01 marqué CORRIGÉ 2026-04-29. Vérifier `src/api/auth.py:24` + ajouter test CI qui fail si env `SENTINEL_TESTING_MODE=1` détecté en build prod.
- **Fichiers** : `src/api/auth.py:24`, `.github/workflows/ci.yml` (ajouter step).
- **Effort** : 1 h.
- **Acceptance** : grep `SENTINEL_TESTING_MODE.*"0"` dans auth.py ; CI vert avec `ENVIRONMENT=production` + `TESTING_MODE=1` → fail.
- **Impact** : interdit accidental deploy en mode test bypass auth.

## QW-004 — chmod 0o600 sur state files JSON persistence
- **Source** : Cat 16 (`16_signal_state_machine.md` §1.2 F3)
- **Description** : après chaque `os.replace`, set perms 0o600 (owner only).
- **Fichiers** : `src/intelligence/state_persistence.py:55-73`
- **Effort** : 30 min.
- **Acceptance** : `tests/test_state_persistence_perms.py` vérifie mode 0o600 post-save sur Linux.
- **Impact** : prévient lecture state par autre user système.

## QW-005 — Ajouter checksum SHA-256 dans state persistence JSON
- **Source** : Cat 16 (`16_signal_state_machine.md` F3, `state_persistence.py:98-103`)
- **Description** : ajouter champ `_checksum: "sha256:..."` dans bundle JSON ; détection corruption silente.
- **Fichiers** : `src/intelligence/state_persistence.py:39-121`.
- **Effort** : 1.5 h.
- **Acceptance** : corrompre 1 octet du fichier → load lève `CorruptedStateError` (vs reset silencieux actuel).
- **Impact** : détection corruption ops + alerte vs reset silent.

## QW-006 — Migrer 25 `print()` critiques restants vers logger
- **Source** : Cat 12 §1.1.a (`12_observability.md` top-10 + nouveaux)
- **Description** : top fichiers à migrer : `src/security/alert_manager.py` (8), `src/security/hmac_manager.py` (4 — secrets en clair !), `src/security/dead_man_switch.py` (2), `src/research/a1_train.py` (15), `src/intelligence/rag/pipeline.py` (12), `src/agents/data/fred_provider.py` (8).
- **Fichiers** : 6 fichiers ci-dessus.
- **Effort** : 2 h (find/replace + run tests).
- **Acceptance** : `grep -c "print(" src/security/hmac_manager.py` = 0 ; logs structurés JSON capturent les events.
- **Impact** : 🚨 fuite GDPR potentielle (hmac_manager imprime secrets clear) éliminée. Logs corrélables.

## QW-007 — `restart: unless-stopped` docker-compose
- **Source** : Cat 19 §1.1 (`19_mlops_deployment.md:33`)
- **Description** : confirmer `restart: unless-stopped` sur services prod (déjà partiellement OK).
- **Fichiers** : `infrastructure/docker-compose.yml`
- **Effort** : 15 min (vérif).
- **Acceptance** : tous services prod ont `restart: unless-stopped` ; pas de `restart: never` ou absence.
- **Impact** : auto-recovery après crash container.

## QW-008 — Patcher `JSONFormatter` pour merge `extra={}` proprement
- **Source** : Cat 12 §1.1 P0-2 (`12_observability.md`)
- **Description** : `JSONFormatter` actuel ignore `logger.info("...", extra={"signal_id": ...})` → signal_id perdu.
- **Fichiers** : `src/intelligence/main.py:39-52`.
- **Effort** : 1 h.
- **Acceptance** : `logger.info("test", extra={"k": "v"})` → JSON output contient `"k": "v"`.
- **Impact** : 100+ call-sites débloquent contexte structuré, MTTR ÷ 3.

## QW-009 — Patch B2 swing detector look-ahead `multi_timeframe_features.py:554-566`
- **Source** : Cat 6 (`06_backtest_validation.md` B2 NON-CAUSAL)
- **Description** : `iloc[i+1/+2]` non-causal ; patch émettre swing à `i+2` (shift(+2)).
- **Fichiers** : `src/environment/multi_timeframe_features.py:554-566`.
- **Effort** : 1.5 h + test régression.
- **Acceptance** : `tests/test_smart_money_engine_causality.py` ajouté ; vérification : aucun swing(i) ne dépend de bar > i.
- **Impact** : bloqueur tout backtest MTF actuel ; sans ce patch, gates institutionnels invalides.

## QW-010 — `SENTRY_DSN` env var template + stub câblé
- **Source** : Cat 12 (`12_observability.md` Sentry stub)
- **Description** : ajouter `SENTRY_DSN=` dans `.env.example` + stub init dans `main.py` qui no-op si vide.
- **Fichiers** : `.env.example`, `src/intelligence/main.py`, `src/performance/observability.py`.
- **Effort** : 30 min.
- **Acceptance** : `SENTRY_DSN=` non set → app boot OK ; set → Sentry capture exception test.
- **Impact** : prêt à activer Sentry free tier (Cat 12 cost $0).

## QW-011 — Coverage gate fail-fast au boot scanner
- **Source** : Cat 5 (`05_data_infrastructure.md` §1.3 bug #3)
- **Description** : `SentinelScanner` doit refuser le boot si feed coverage < seuil (95 % par défaut).
- **Fichiers** : `src/intelligence/sentinel_scanner.py:334-340`.
- **Effort** : 1.5 h.
- **Acceptance** : test boot avec `XAU_15MIN_2019_2025.csv` (63 %) → raise `InsufficientCoverageError`.
- **Impact** : élimine consommation silente d'un feed corrompu.

## QW-012 — Brancher `validate_ohlcv` jamais appelé au boot
- **Source** : Cat 5 §1.2 (`05_data_infrastructure.md`)
- **Description** : `src/intelligence/data_quality.py:validate_ohlcv()` existe mais pas appelé au boot scanner ; brancher.
- **Fichiers** : `src/intelligence/sentinel_scanner.py:start()`, `src/intelligence/data_quality.py`.
- **Effort** : 1 h.
- **Acceptance** : démarrage scanner appelle `validate_ohlcv()` ; refuse si `strict=True` et report.has_errors.
- **Impact** : couple QW-011 ; détection feed cassé au boot.

## QW-013 — Fix port docker-compose 8080 → 8000 (cohérence)
- **Source** : Cat 8 (`08_api_backend.md`), Cat 19 (port 8000 entry)
- **Description** : `infrastructure/docker-compose.yml` doit exposer 8000 cohérent avec `Dockerfile:14,89`.
- **Fichiers** : `infrastructure/docker-compose.yml`.
- **Effort** : 15 min.
- **Acceptance** : `docker-compose up` → port `127.0.0.1:8000` répond `/health`.
- **Impact** : pas de confusion deploy.

## QW-014 — Désactiver `continue-on-error: true` lint job → bloquant
- **Source** : Cat 13 §1.3 (`13_testing_infrastructure.md`)
- **Description** : `.github/workflows/ci.yml:31-56` job lint actuellement advisory ; passer bloquant après pre-commit hooks ruff/black fix locally.
- **Fichiers** : `.github/workflows/ci.yml`.
- **Effort** : 30 min + fix lint issues localement avant push.
- **Acceptance** : PR avec lint error → CI fail bloquant.
- **Impact** : prévient dérive code style.

## QW-015 — `pytest.ini` durci (timeout, strict-markers, strict-config)
- **Source** : Cat 13 §1.2 (`13_testing_infrastructure.md`)
- **Description** : ajouter `addopts = --strict-markers --strict-config --timeout=120 --tb=short -ra`.
- **Fichiers** : `pytest.ini`.
- **Effort** : 30 min.
- **Acceptance** : marker non déclaré → erreur ; test > 120s → timeout fail.
- **Impact** : tests deterministes, pas de hang silencieux.

## QW-016 — SemanticCache : ajouter `session_bucket` au hash composite
- **Source** : Cat 17 §1.1, Conflict C5 master plan
- **Description** : intra-jour collision (`bar_ts` exclu + `session` exclu → 2 signaux 5h d'écart même narrative).
- **Fichiers** : `src/intelligence/semantic_cache.py:30-158`, `tests/test_semantic_cache_session_isolation.py` (nouveau).
- **Effort** : 1.5 h.
- **Acceptance** : 2 signaux mêmes composants mais sessions différentes → cache miss (narrative re-générée).
- **Impact** : qualité narrative UX.

## QW-017 — `cleanup_expired` câblé sur SemanticCache (cron quotidien)
- **Source** : Cat 17 §1.1 (`semantic_cache.py:216-231` jamais appelé)
- **Description** : fonction existe, jamais appelée → DB grossit sans bornes.
- **Fichiers** : `src/intelligence/semantic_cache.py`, `src/intelligence/main.py` (scheduler tâche cron).
- **Effort** : 1 h.
- **Acceptance** : test : insérer 1000 entrées avec TTL expiré → après cleanup, 0 entries.
- **Impact** : DB size bornée, latence get() stable.

## QW-018 — Brancher `mode=lgbm` + `mode=hybrid` HMM bug B4 fix
- **Source** : Cat 4 §1.3 B4 (`04_volatility_forecasting.md`)
- **Description** : `_get_regime_multiplier` regarde `idx = latest_idx` (la barre courante) au lieu de la barre cible → tous les forecasts classés `low`. Patch utiliser `target_bar_idx`.
- **Fichiers** : `src/intelligence/volatility_forecaster.py:909-931`.
- **Effort** : 1.5 h.
- **Acceptance** : distribution régime des forecasts sur 1000 bars XAU = bimodale (low + high), pas 100 % low.
- **Impact** : qualité vol forecast régime-aware réelle.

## QW-019 — Drop CSP `unsafe-inline` en dev seulement (strict en prod)
- **Source** : Cat 10 (`10_auth_security.md` F-12 + observation `app.py:262`)
- **Description** : `unsafe-inline` nécessaire Swagger CDN, mais doit être env-gated (`ENVIRONMENT=development`).
- **Fichiers** : `src/api/app.py:261-285`.
- **Effort** : 1 h.
- **Acceptance** : `ENVIRONMENT=production` → header CSP n'inclut PAS `unsafe-inline`.
- **Impact** : durcissement XSS prod.

## QW-020 — `X-Forwarded-For` trusted proxy CIDR list pour rate-limiter
- **Source** : Cat 10 F-15 (`10_auth_security.md`)
- **Description** : RateLimiter actuel utilise `request.client.host` = IP proxy en prod → bypass. Lire `X-Forwarded-For` si proxy CIDR confirmé.
- **Fichiers** : `src/api/app.py:243-256`, `src/intelligence/security.py:100-184`.
- **Effort** : 1.5 h.
- **Acceptance** : test mock proxy → IP réelle lue depuis header ; non-proxy → ignoré.
- **Impact** : rate-limit non spoofable.

## QW-021 — `allow_headers` whitelist explicite vs `["*"]`
- **Source** : Cat 10 F-13 (`10_auth_security.md:203`)
- **Description** : restrict allow_headers : `["Content-Type", "Authorization", "Idempotency-Key", "X-Request-Id"]`.
- **Fichiers** : `src/api/app.py:203`.
- **Effort** : 30 min.
- **Acceptance** : CORS preflight accepte uniquement headers whitelistés.
- **Impact** : surface réduite.

## QW-022 — Ajouter `allowed_mentions={parse: []}` Discord (block @everyone/@here)
- **Source** : Cat 11 C5 (`11_delivery_channels.md`)
- **Description** : `discord_notifier.py` payload `_post` ne déclare pas `allowed_mentions` → user malveillant peut injecter `@everyone` dans narrative.
- **Fichiers** : `src/delivery/discord_notifier.py`.
- **Effort** : 30 min.
- **Acceptance** : narrative contenant `@everyone` → Discord ne pinger personne.
- **Impact** : prévient spam mentions.

## QW-023 — Ajout `WARMUP=500` ou masque NaN explicite sur `bfill()` indicateurs
- **Source** : Cat 6 (`06_backtest_validation.md` B3)
- **Description** : `bfill()` indicateurs lents introduit leak warmup léger ; `WARMUP=500` ou `notna()` mask.
- **Fichiers** : `src/environment/environment.py:802`, `src/environment/multi_timeframe_features.py:184`.
- **Effort** : 1 h.
- **Acceptance** : test : aucun signal émis sur 500 premières bars (warmup).
- **Impact** : élimine leak léger backtest.

## QW-024 — Aligner `armed_window` default 5 vs 30 (mismatch)
- **Source** : Cat 2 G7 P1-2 (`02_smart_money_algo.md:420,510-518`)
- **Description** : `calculate_bos_retest_fast` default `armed_window=5` vs `SMCConfig.RETEST_ARMED_WINDOW=30` → incohérence.
- **Fichiers** : `src/environment/strategy_features.py:420`.
- **Effort** : 30 min.
- **Acceptance** : `tests/test_bos_retest.py` regression test force cohérence (fail si mismatch).
- **Impact** : cohérence comportementale.

## QW-025 — RSI Divergence indexage fix (`down_fractals[i]` vs `lows[i]`)
- **Source** : Cat 2 G6 P1-1 (`02_smart_money_algo.md:849-857`)
- **Description** : bug indexage `lows[i]` au lieu de `down_fractals[i]` ; décalage moyen `-1.18 $` sur XAU.
- **Fichiers** : `src/environment/strategy_features.py:849-857, 865-873`.
- **Effort** : 1.5 h.
- **Acceptance** : `tests/test_sprint7_rsi_divergence.py` étendu (test ICT correctness).
- **Impact** : RSI Divergence détectée sur vraies valeurs swing.

## QW-026 — Assertion `NUMBA_AVAILABLE` au boot prod
- **Source** : Cat 2 P1-4 (`02_smart_money_algo.md:1-50`)
- **Description** : si `NUMBA_AVAILABLE=False` en prod (`ENVIRONMENT=production`), raise au boot (sinon latence ×25 silencieuse).
- **Fichiers** : `src/intelligence/main.py`, `src/intelligence/sentinel_scanner.py:start()`.
- **Effort** : 1 h.
- **Acceptance** : test `tests/test_main_numba_check.py` ; staging warning + métrique Prometheus.
- **Impact** : prévient latence ×25 silente.

## QW-027 — Audit log admin action `payload_digest` SHA-256 (déjà fait, vérifier wire)
- **Source** : Cat 10 F-10 marqué CORRIGÉ
- **Description** : confirmer `src/audit/admin_action_log.py` wiré sur toutes routes admin.
- **Fichiers** : `src/api/routes/admin.py`, `src/api/routes/admin_audit.py`.
- **Effort** : 30 min vérification.
- **Acceptance** : test : POST admin action → row insérée dans `admin_actions` avec hash.
- **Impact** : non-repudiation actes admin.

## QW-028 — Doc `docs/RUNBOOK.md` créé (5 incidents top)
- **Source** : Cat 19 §1.1, Cat 12 P0
- **Description** : créer runbook minimal pour 5 incidents : Anthropic outage, Telegram bot ban, data feed stale, Stripe webhook fail, Fly.io app crash.
- **Fichiers** : `docs/RUNBOOK.md` (nouveau).
- **Effort** : 1.5 h.
- **Acceptance** : 5 sections "Detect / Diagnose / Mitigate / Postmortem".
- **Impact** : MTTR < 30 min pendant solo founder.

## QW-029 — `pyproject.toml` minimal (tool.ruff + tool.mypy + tool.black)
- **Source** : Cat 13 §1.2 (absent)
- **Description** : créer `pyproject.toml` racine avec configs centralisées ; remplace progressivement `pytest.ini`.
- **Fichiers** : `pyproject.toml` (nouveau).
- **Effort** : 1 h.
- **Acceptance** : `ruff check` + `mypy` + `black --check` lisent depuis pyproject.toml.
- **Impact** : config centralisée, lint reproductible.

## QW-030 — `.dockerignore` étendre exclusions (data/, models/, reports/, *.pkl)
- **Source** : Cat 19 §1.1 (déjà présent)
- **Description** : vérifier `.dockerignore` exclut data, models, reports, *.csv > 1 MB, *.parquet, *.pkl. Construire image légère.
- **Fichiers** : `.dockerignore`.
- **Effort** : 30 min.
- **Acceptance** : `docker build` taille image < 500 MB.
- **Impact** : deploy rapide, pas de fuite data dans image.

## QW-031 — Ajouter `sources_manifest.yaml` note "internal only" Dukascopy/FF
- **Source** : Cat 18 §1.4 + Cat 5
- **Description** : documenter dans `data/rag/sources_manifest.yaml` que Dukascopy/FF = backtest interne, NON commercial.
- **Fichiers** : `data/rag/sources_manifest.yaml`.
- **Effort** : 30 min.
- **Acceptance** : entrée présente, daté.
- **Impact** : trace écrite pour audit légal.

## QW-032 — Add `.coveragerc` fichier
- **Source** : Cat 13 §1.2 (absent)
- **Description** : `.coveragerc` avec `source = src/`, `omit = src/agents/sprint*/, tests/, scripts/`.
- **Fichiers** : `.coveragerc` (nouveau).
- **Effort** : 30 min.
- **Acceptance** : `coverage report` retourne pourcentage cohérent (pas biaisé par imports conditionnels).
- **Impact** : coverage mesurable, gate CI 75 % crédible.

## QW-033 — Gunicorn worker class `uvicorn.workers.UvicornWorker` template
- **Source** : Cat 8 §1.1
- **Description** : créer `gunicorn_conf.py` template prêt pour multi-worker activation future (DG-024).
- **Fichiers** : `infrastructure/gunicorn_conf.py` (nouveau).
- **Effort** : 30 min.
- **Acceptance** : `gunicorn -c infrastructure/gunicorn_conf.py src.intelligence.main:api_app` lance avec 1 worker en dev.
- **Impact** : prêt pour scale multi-worker.

## QW-034 — Pre-commit hook `ruff format + ruff check --fix` activé
- **Source** : Cat 13
- **Description** : `.pre-commit-config.yaml` avec ruff hooks.
- **Fichiers** : `.pre-commit-config.yaml`.
- **Effort** : 30 min.
- **Acceptance** : `pre-commit run --all-files` formate sans erreur.
- **Impact** : style code consistent, CI lint plus rapide.

---

## Résumé exécutable

**Total** : 34 quick-wins identifiés.
**Effort cumulé** : ~30 h sur S1-S2 (en parallèle de 🟡 RISKY operational items).
**Risque** : minimal (aucun cassage code prod, tests régression à chaque item).
**Impact agrégé** : 
- Sécurité : 7 items (QW-3, 4, 5, 19, 20, 21, 22).
- Observability : 4 items (QW-6, 8, 10, 27).
- Data integrity : 4 items (QW-11, 12, 23, 31).
- Test/CI : 6 items (QW-1, 14, 15, 29, 32, 34).
- Code quality : 5 items (QW-9, 24, 25, 26, 33).
- Architecture prep : 4 items (QW-7, 13, 28, 30).
- Performance : 4 items (QW-2, 16, 17, 18).

**Recommandation** : exécuter en 2 sprints d'1 semaine, parallèle aux étapes 1-2 du DANGEROUS_CHANGES ordonnancement.

---

**FIN QUICK WINS.** Lire conjointement `00_MASTER_PLAN.md` et `00_DANGEROUS_CHANGES.md`.
