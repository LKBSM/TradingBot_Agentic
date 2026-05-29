# Catalogue des Changements Dangereux & Importants — Gate Utilisateur

> **Objectif** : avant toute exécution du master plan, l'utilisateur doit valider EXPLICITEMENT chaque item de ce catalogue. C'est un gate de sécurité PhD-grade.
> **Date** : 2026-05-23
> **Auteur** : Architecte Master Plan
> **Source** : extraction systématique des 20 plans `reports/commercialization_sprint/{01..20}_*.md`.

---

## Légende

- 🔴 **DESTRUCTIVE** : supprime du code/data/feature, breaking change, action irréversible sans backup
- 🟠 **BIG ARCHITECTURAL** : refactor majeur, dépendance, migration de schéma DB, changement d'hébergeur
- 🟡 **RISKY OPERATIONAL** : change config prod, secrets, paiements, légal, comportement utilisateur visible
- 🟣 **POLITIQUE/MÉTIER** : décision stratégique avec impact long-terme (pricing, pivot, marque, partenariats)
- 🟢 **SAFE** (référence) : ajouts isolés, tests, docs, no-regret (cf. `00_NO_REGRET_QUICK_WINS.md`)

Chaque item :
- ID + titre
- Catégorie source
- Niveau de risque
- Description précise
- Fichiers concernés (file:line)
- Pourquoi dangereux/important
- Réversibilité
- Impact si erreur
- Validation pré-changement
- Plan de rollback
- Pré-requis
- **DÉCISION REQUISE** : ☐ Approuver tel quel / ☐ Approuver avec modifications / ☐ Skip pour l'instant / ☐ Discuter

---

## 🔴 DESTRUCTIVE — Suppressions et breaking changes

### [DG-001] Suppression `parallel_training.py` (RL legacy)
- **Catégorie source** : Cat 19 (`19_mlops_deployment.md` §1.2 bloqueur #1)
- **Niveau** : 🔴 DESTRUCTIVE
- **Description** : supprimer `parallel_training.py` à la racine + références dans Procfile/railway.toml historiques. Sentinel n'utilise plus ce script (entry prod = `src.intelligence.main`).
- **Fichiers concernés** : `parallel_training.py` (racine), `Procfile` (si présent), `railway.toml` (si présent), search `parallel_training` partout.
- **Pourquoi dangereux** : risque oublier qu'un script orchestrator RL legacy le référence et casser un cron training.
- **Réversibilité** : facile (git revert).
- **Impact si erreur** : pipeline training RL legacy cassé (mais non-commercialisé). Tests anciens RL cassent.
- **Validation pré-changement** : `grep -r "parallel_training" .` doit montrer uniquement docs historiques.
- **Plan de rollback** : `git revert <commit>` ; aucune data n'est modifiée.
- **Pré-requis** : aucun.
- **DÉCISION REQUISE** : ☐ Approuver / ☐ Modifications / ☐ Skip / ☐ Discuter

### [DG-002] Suppression `tests/test_long_short_trading.py`
- **Catégorie source** : Cat 13 (`13_testing_infrastructure.md` §1.1 import cassé `src.config`)
- **Niveau** : 🔴 DESTRUCTIVE (mais déjà broken, donc bénin)
- **Description** : supprimer le fichier de test dont l'import `src.config` lève à la collection.
- **Fichiers concernés** : `tests/test_long_short_trading.py` (déjà en `D` dans git status).
- **Pourquoi important** : actuellement le test_collect lève une erreur masquée par `continue-on-error` ; bloque l'activation d'une CI bloquante.
- **Réversibilité** : facile.
- **Impact si erreur** : aucun (déjà cassé).
- **Validation** : `git diff HEAD tests/test_long_short_trading.py` doit montrer le D status.
- **Pré-requis** : aucun. (Item également listé QW-001 dans no-regret.)
- **DÉCISION REQUISE** : ☐ Approuver / ☐ Skip / ☐ Discuter

### [DG-003] Suppression / drop des 3 moteurs risk redondants
- **Catégorie source** : Cat 7 (`07_risk_management.md` §1.1, 9 moteurs identifiés)
- **Niveau** : 🔴 DESTRUCTIVE
- **Description** : drop `src/environment/risk_manager.py`, `src/agents/risk_sentinel.py`, `src/live_trading/live_risk_manager.py`, `src/risk/var_engine.py`, `src/agents/risk_integration.py`, `src/agents/intelligent_risk_sentinel.py` (orphelins RL). Garder UNIQUEMENT `src/risk/kill_switch.py` (canonical) + nouveau RiskManager consolidé.
- **Fichiers concernés** : 6 fichiers ci-dessus + tous les imports / tests s'y référant.
- **Pourquoi dangereux** : risque casser des tests RL legacy qui s'attendent à ces classes (~50+ files à grep).
- **Réversibilité** : difficile (les classes orphelines représentent des heures de dev historique).
- **Impact si erreur** : tests RL cassés, mais RL n'est plus commercialisé → impact opérationnel zéro.
- **Validation pré-changement** : `grep -r "DynamicRiskManager\|RiskSentinel\|LiveRiskManager\|VaREngine" src/ tests/` ; lister tous les call-sites ; chaque suppression isolée par commit.
- **Plan de rollback** : `git revert` séquentiel par classe.
- **Pré-requis** : décision finale Vision B (cf. DG-040), drop RL legacy comme business.
- **DÉCISION REQUISE** : ☐ Approuver / ☐ Modifications / ☐ Skip / ☐ Discuter

### [DG-004] Suppression / déprécation feed XAU_15MIN_2019_2025.csv (63 % coverage)
- **Catégorie source** : Cat 5 (`05_data_infrastructure.md` §1.1 + memory data_quality_audit_2026_04_23.md)
- **Niveau** : 🔴 DESTRUCTIVE (data)
- **Description** : supprimer ou archiver `data/XAU_15MIN_2019_2025.csv` (63 % coverage, source distincte, cause root des "BOS sur 100 % bars"). Garder `XAU_15MIN_2019_2024.csv` (97.6 %) + `XAU_15MIN_2019_2026.csv` (98.72 %) comme références.
- **Fichiers concernés** : `data/XAU_15MIN_2019_2025.csv` ; vérifier qu'aucun script ne le référence en dur.
- **Pourquoi dangereux** : si un benchmark ou un test reproductibility le référence, on perd la capacité de reproduire un audit historique.
- **Réversibilité** : impossible si non archivé (Dukascopy peut ne pas re-fournir le même feed corrompu).
- **Impact si erreur** : un test régression échoue, ou un rapport historique non re-générable.
- **Validation** : `grep -rn "XAU_15MIN_2019_2025" .` ; déplacer dans `data/_archived/` avec README explicatif PLUTÔT que delete.
- **Plan de rollback** : restaurer depuis backup `data/_archived/`.
- **Pré-requis** : avoir backupé sur Cloudflare R2 (Cat 19 P1).
- **DÉCISION REQUISE** : ☐ Approuver / ☐ Modifications / ☐ Skip / ☐ Discuter

### [DG-005] Décom Dukascopy + ForexFactory en pipeline live prod
- **Catégorie source** : Cat 5 (`05_data_infrastructure.md` §1.4), Cat 15 (`15_news_macro_pipeline.md` TL;DR), Cat 18 (`18_compliance_legal.md` §1.3)
- **Niveau** : 🔴 DESTRUCTIVE (provider switch)
- **Description** : couper Dukascopy + ForexFactory en prod live ; restreindre à backtest interne avec note "internal only" dans `data/rag/sources_manifest.yaml`. Live calendar = Trading Economics API ($79/mo). Live OHLCV = Polygon/Tiingo (P1).
- **Fichiers concernés** : `scripts/fetch_forexfactory_live.py`, `scripts/download_dukascopy_xau.py`, `src/agents/news/economic_calendar.py:130-457` (CSV source priority), `infrastructure/cron.yml`.
- **Pourquoi dangereux** : sans Trading Economics live, le scanner perd la source calendar pendant le switchover → blackout impossible.
- **Réversibilité** : facile (env var `CALENDAR_PROVIDER` switch).
- **Impact si erreur** : faux blackouts, signaux mauvais timing news.
- **Validation pré-changement** : TE souscrit, test sandbox key, validate ≥ 100 events/sem ingested over 1 sem.
- **Plan de rollback** : `CALENDAR_PROVIDER=forexfactory_local` env override + restore cron FF.
- **Pré-requis** : DG-027 (souscription TE) + 7 jours validation parallèle TE vs FF cross-check.
- **DÉCISION REQUISE** : ☐ Approuver / ☐ Modifications / ☐ Skip / ☐ Discuter

### [DG-006] Désactivation tier rate-limit dead code → activation réelle
- **Catégorie source** : Cat 10 (`10_auth_security.md` F-02 marqué CORRIGÉ 2026-05-XX)
- **Niveau** : 🔴 DESTRUCTIVE (comportement utilisateur rupture)
- **Description** : déjà câblé selon plan 10. Vérifier la rupture utilisateur : avant le câblage, free users pouvaient tout consommer. Après, hard cap = 30/200/800/2000 signaux/mo.
- **Fichiers concernés** : `src/api/auth.py:463-478`, `src/api/tier_manager.py` rate-limit logic.
- **Pourquoi dangereux** : tout utilisateur existant dépassant son tier voit son service coupé brutalement à la prochaine requête → support tickets explosion.
- **Réversibilité** : facile (env var `TIER_RATE_LIMIT_ENFORCEMENT=warn|enforce`).
- **Impact si erreur** : churn massive si soft-cap warn non implémenté en premier.
- **Validation** : confirmer F-02 effectivement câblé (`grep "check_rate_limit" src/api/auth.py`) ; mode `warn` 7 jours puis `enforce`.
- **Plan de rollback** : `TIER_RATE_LIMIT_ENFORCEMENT=warn` re-set.
- **Pré-requis** : Cat 1 P0-T4 hard caps signaux (`quota_manager.py`) + soft-cap 80 % UX + email warning before enforcement.
- **DÉCISION REQUISE** : ☐ Approuver / ☐ Modifications / ☐ Skip / ☐ Discuter

### [DG-007] Suppression CSV economic_calendar_2019_2025.csv (FF scrapé)
- **Catégorie source** : Cat 15 + Cat 18 (usage commercial déguisé)
- **Niveau** : 🔴 DESTRUCTIVE (data)
- **Description** : retirer `data/economic_calendar_2019_2025.csv` + `data/economic_calendar_HIGH_IMPACT_2019_2025.csv` du build prod (CGU FF interdit usage commercial). Conserver dans `data/_archived/` pour recherche backtest.
- **Fichiers concernés** : `data/economic_calendar_*.csv` (2 fichiers).
- **Pourquoi dangereux** : backtest historique perd 2 sources si non archivées.
- **Réversibilité** : difficile (FF ne re-fournit pas l'archive en clair).
- **Impact si erreur** : impossible de reproduire un backtest historique news-aware.
- **Validation** : copier dans `data/_archived/forexfactory_2019_2025/` AVANT delete prod path.
- **Plan de rollback** : restaurer depuis archive.
- **Pré-requis** : DG-027 (TE) + archive backup.
- **DÉCISION REQUISE** : ☐ Approuver / ☐ Modifications / ☐ Skip / ☐ Discuter

### [DG-008] Drop `models/scoring_v2.lgb` legacy
- **Catégorie source** : Cat 3 (`03_machine_learning.md` §1.3 cartographie modèles, scoring_v2 "legacy plus utilisé ?")
- **Niveau** : 🔴 DESTRUCTIVE
- **Description** : supprimer `models/scoring_v2.lgb` si non référencé prod.
- **Fichiers concernés** : `models/scoring_v2.lgb`.
- **Pourquoi dangereux** : peut être référencé en mode shadow par un agent de calibration.
- **Validation** : `grep -rn "scoring_v2" src/` ; si zero hit, drop OK.
- **Plan de rollback** : copy depuis backup R2.
- **Pré-requis** : Cat 19 P0 model registry + backup R2 systématique.
- **DÉCISION REQUISE** : ☐ Approuver / ☐ Skip / ☐ Discuter

### [DG-009] Drop `tests/test_env_debug.py` à la racine
- **Catégorie source** : Cat 13 §1.1 test cassé à la collection
- **Niveau** : 🔴 DESTRUCTIVE (mais bénin)
- **Description** : retirer `test_env_debug.py` à la racine si présent (test debug ad-hoc cassant la collection).
- **Validation** : `ls test_env_debug.py 2>/dev/null` et vérifier qu'il n'est pas du code prod.
- **DÉCISION REQUISE** : ☐ Approuver / ☐ Skip / ☐ Discuter

### [DG-010] Suppression `scripts/download_economic_calendar.py` (scraper HTML cassé)
- **Catégorie source** : Cat 5 §1.2
- **Niveau** : 🔴 DESTRUCTIVE
- **Description** : retirer scraper HTML FF déprécié (remplacé par `fetch_forexfactory_live.py` puis TE).
- **Réversibilité** : git revert.
- **Pré-requis** : DG-027 (TE) live.
- **DÉCISION REQUISE** : ☐ Approuver / ☐ Skip / ☐ Discuter

### [DG-011] Suppression duplicata `data/macro/` vs `data/research/` (COT + FRED)
- **Catégorie source** : Cat 5 §1.1 (warning duplication)
- **Niveau** : 🔴 DESTRUCTIVE (data)
- **Description** : déduper FRED + COT entre `data/macro/` et `data/research/`. Conserver un seul lieu (recommandé `data/macro/`), supprimer copies dans `data/research/`.
- **Fichiers** : `data/macro/cot_gold.csv`, `data/research/cot_gold.csv` ; idem 5 FRED.
- **Pourquoi dangereux** : si un script charge l'un et un autre l'autre, divergence silente lors d'update.
- **Validation** : `grep -rn "data/research/cot_gold\|data/research/fred_" src/ scripts/` ; rerouter tout vers `data/macro/`.
- **Plan de rollback** : `git revert`.
- **DÉCISION REQUISE** : ☐ Approuver / ☐ Skip / ☐ Discuter

### [DG-012] Drop `Procfile` + `railway.toml` si présents (legacy RL deploy)
- **Catégorie source** : Cat 19 §1.1, `eval_22_deployment.md`
- **Niveau** : 🔴 DESTRUCTIVE (build)
- **Description** : ces fichiers ne sont plus à la racine selon §1.1 Cat 19. Vérifier ; si présents (historique branch), supprimer pour éviter PaaS auto-build pointant vers `parallel_training.py`.
- **Fichiers** : `Procfile`, `railway.toml` (root).
- **Pré-requis** : DG-001 (drop parallel_training.py).
- **DÉCISION REQUISE** : ☐ Approuver / ☐ Skip / ☐ Discuter

### [DG-013] Décom `scripts/download_dukascopy_xau.py` en prod
- **Catégorie source** : Cat 5 + Cat 18
- **Niveau** : 🔴 DESTRUCTIVE (license)
- **Description** : retirer Dukascopy du pipeline live prod. Conserver pour backtest interne avec note "internal only".
- **Pré-requis** : DG-027 (Polygon ou Tiingo live OHLCV souscrit).
- **DÉCISION REQUISE** : ☐ Approuver / ☐ Modifications / ☐ Skip / ☐ Discuter

### [DG-014] Drop `scripts/export_mt5_history.py` (local Windows uniquement)
- **Catégorie source** : Cat 5 §1.2
- **Niveau** : 🔴 DESTRUCTIVE
- **Description** : retirer script local non transférable ; documenter alternatif Polygon/Tiingo.
- **DÉCISION REQUISE** : ☐ Approuver / ☐ Skip / ☐ Discuter

---

## 🟠 BIG ARCHITECTURAL — Refactors majeurs et migrations

### [DG-020] Migration cache local → Redis multi-worker
- **Catégorie source** : Cat 17 §1.3-1.4
- **Niveau** : 🟠 BIG ARCHITECTURAL
- **Description** : externaliser `SemanticCache` compteurs, `RateLimiter` IP, `IdempotencyStore`, `CircuitBreaker` état vers Redis (Upstash free tier ou self-hosted via docker-compose 75-94 déjà défini).
- **Fichiers concernés** : `src/intelligence/semantic_cache.py:47-48`, `src/intelligence/security.py:100-184`, `src/api/idempotency_store.py:87`, `src/intelligence/circuit_breaker.py:118-150`.
- **Pourquoi dangereux** : changement de persistence implique tester chaque path ; comportement réseau peut introduire latence si Redis lointain.
- **Réversibilité** : facile (env var `CACHE_BACKEND=local|redis`).
- **Impact si erreur** : double-counting ou perte de stat (semantic cache divergent) → coût LLM mal mesuré ; rate-limit bypass partiel.
- **Validation pré-changement** : test latence ping Redis < 5 ms ; circuit-breaker fallback si Redis down (fail-open local).
- **Plan de rollback** : `CACHE_BACKEND=local` env override.
- **Pré-requis** : DG-022 (deploy Fly.io cdg) + Upstash souscrit + DG-024 (multi-worker).
- **DÉCISION REQUISE** : ☐ Approuver / ☐ Modifications / ☐ Skip / ☐ Discuter

### [DG-021] Async I/O end-to-end routes critiques
- **Catégorie source** : Cat 8 §1.4 (9 sync I/O dans `async def` identifiés), Cat 17 §1.2
- **Niveau** : 🟠 BIG ARCHITECTURAL
- **Description** : refactor `routes/signals.py:50`, `routes/narratives.py:57`, `routes/narratives.py:155` (Anthropic HTTPS), `routes/dashboard.py:18-55`, `routes/insight_history.py:106-296`, `routes/enrich.py:258`, `routes/admin.py:42-78`, `auth.require_api_key` en async (aiosqlite + httpx async + asyncio sleep replace).
- **Fichiers concernés** : 8 routes + `src/api/auth.py:453` + `src/api/signal_store.py` + `src/intelligence/llm_narrative_engine.py:_call_api`.
- **Pourquoi dangereux** : 800-3000 ms gel event loop sur `narratives/chat` aujourd'hui ; refactor crée régressions si non testé.
- **Réversibilité** : difficile une fois marqué async (les callers doivent awaiter).
- **Impact si erreur** : event loop deadlock, ou pire — bugs silencieux d'await manqué (cf. Cat 11 C1 même problème).
- **Validation** : tests async (`pytest-asyncio`) sur chaque route refactor ; benchmark p99 < 100 ms avant promotion.
- **Plan de rollback** : feature flag `ASYNC_ROUTES=true|false` ; partial rollback par route.
- **Pré-requis** : Cat 13 P0 (tests CI) + Cat 17 P0 (cache).
- **DÉCISION REQUISE** : ☐ Approuver / ☐ Modifications / ☐ Skip / ☐ Discuter

### [DG-022] Hébergement choisi : Fly.io cdg (Paris) prod + Vercel front
- **Catégorie source** : Cat 19 §2.3, Cat 20 §0
- **Niveau** : 🟠 BIG ARCHITECTURAL
- **Description** : créer `fly.toml`, déployer images via GHCR, scale-to-zero staging, Paris region pour latence FR. Frontend Next.js sur Vercel free tier.
- **Pourquoi important** : cementaire le choix infrastructure, coûts récurrents engagés ($30-40/mo prod).
- **Réversibilité** : difficile (re-déployer ailleurs = 12-24 h dev).
- **Impact si erreur** : engagement fournisseur ; switch vers Hetzner / Railway / AWS = effort lock-in.
- **Validation** : test cold-start < 2 s, latence Paris→endpoint < 30 ms p50 depuis VPN FR.
- **Plan de rollback** : conserver Dockerfile portable ⇒ basculer Railway/Hetzner en < 4 h si Fly.io down ou prix monte.
- **Pré-requis** : DG-029 (vault secrets).
- **DÉCISION REQUISE** : ☐ Approuver / ☐ Modifications / ☐ Skip / ☐ Discuter

### [DG-023] Stack frontend lock : Next.js 15 + Tailwind + shadcn/ui + TypeScript strict
- **Catégorie source** : Cat 20 §0
- **Niveau** : 🟠 BIG ARCHITECTURAL
- **Description** : verrou tech front (impacte 470 h dev Cat 20).
- **Réversibilité** : très difficile (réécriture frontend = 200+ h).
- **Validation** : confirmer Next.js App Router supporte 3-vues toggle persistant + SSR + auth via cookie session compatible API key.
- **DÉCISION REQUISE** : ☐ Approuver / ☐ Modifications / ☐ Skip / ☐ Discuter

### [DG-024] Multi-worker Gunicorn + Uvicorn (2-4 workers)
- **Catégorie source** : Cat 8 §1.1, Cat 17 §1.3
- **Niveau** : 🟠 BIG ARCHITECTURAL
- **Description** : passer de mono-worker uvicorn → gunicorn -k uvicorn.workers.UvicornWorker -w 2-4.
- **Pré-requis** : DG-020 (Redis), DG-021 (async I/O), DG-026 (SignalStore consensus multi-process).
- **DÉCISION REQUISE** : ☐ Approuver / ☐ Modifications / ☐ Skip / ☐ Discuter

### [DG-025] Refonte ConfluenceDetector scoring (Logistic L1 + isotonic + conformal)
- **Catégorie source** : Cat 2 P0-5 (`02_smart_money_algo.md`), Cat 3 (calibration layer)
- **Niveau** : 🟠 BIG ARCHITECTURAL
- **Description** : remplacer `_score_*` somme pondérée additive par un modèle calibré (Logistic L1 → LightGBM si fail) + isotonic + conformal. Reframe = calibrer P(win), pas chercher edge directionnel.
- **Fichiers concernés** : `src/intelligence/confluence_detector.py:195-385`, `src/intelligence/scoring/logistic_l1.py:74-100`, nouveau `src/intelligence/scoring/calibration_loop.py`, `src/intelligence/scoring/isotonic.py`.
- **Pourquoi dangereux** : changement comportemental majeur — tier classifications changent → users perçoivent mutation signaux.
- **Réversibilité** : feature flag `SCORING_VERSION=v1|v2`.
- **Impact si erreur** : faux tier PREMIUM ⇒ marketing claim non substanciable ⇒ churn.
- **Validation** : shadow mode 2 sem (logs `score_v1` et `score_v2` côte-à-côte). Brier skill ≥ +5 % gate.
- **Plan de rollback** : env var bascule v1.
- **Pré-requis** : DG-027 (TE data live) + Cat 2 P0-4 (component scores persistés).
- **DÉCISION REQUISE** : ☐ Approuver / ☐ Modifications / ☐ Skip / ☐ Discuter

### [DG-026] Migration SignalStore SQLite → Postgres (P1, post-S16)
- **Catégorie source** : Cat 8 §1.3, Cat 17 §2.2
- **Niveau** : 🟠 BIG ARCHITECTURAL (defer P1)
- **Description** : migration `signal_store.py` SQLite WAL → Postgres uniquement si MAU > 1 k OR business-need B2B haute-fréquence.
- **Pourquoi dangereux** : migration risque DB corruption pendant fenêtre cutover.
- **Réversibilité** : difficile (data migration aller-retour).
- **Validation** : tests dual-write 7j parallèle SQLite + Postgres, comparaison checksums.
- **Plan de rollback** : env var `DB_BACKEND=sqlite|postgres` + replay via WAL.
- **Pré-requis** : décision basée sur MAU réel post-launch.
- **DÉCISION REQUISE** : ☐ Approuver pour P1 / ☐ Skip / ☐ Discuter

### [DG-027] Souscription Trading Economics ($79/mo) — calendar provider switch
- **Catégorie source** : Cat 15 TL;DR, Cat 5 §1.4
- **Niveau** : 🟠 BIG ARCHITECTURAL (provider)
- **Description** : souscrire TE Pro ($79/mo) + brancher `EconomicCalendarFetcher` sur API REST + enrichir CSV historique Actual/Forecast/Previous (ressuscite surprise feature).
- **Pourquoi dangereux** : engagement récurrent ; switchover doit être validé 7j parallèle FF avant kill FF.
- **Réversibilité** : facile cancel (annuel) ; rebranchement FF env var.
- **Validation** : test sandbox key, ≥ 100 events/sem ingested, cross-check FF.
- **DÉCISION REQUISE** : ☐ Approuver / ☐ Modifications / ☐ Skip / ☐ Discuter

### [DG-028] Model registry MLflow ou S3+manifest (Cloudflare R2)
- **Catégorie source** : Cat 3 §1.2, Cat 19 §2.3
- **Niveau** : 🟠 BIG ARCHITECTURAL
- **Description** : créer registry pour 5 `.pkl` actuels + futurs. Choix recommandé : S3-compatible (Cloudflare R2 $1/mo) + `manifest.json` par modèle (git_sha, data_sha, features_fingerprint, gates_passed).
- **Réversibilité** : facile (manifest est plain JSON).
- **DÉCISION REQUISE** : ☐ Approuver / ☐ Skip / ☐ Discuter

### [DG-029] Secrets vault : Doppler free OU HashiCorp Vault dev
- **Catégorie source** : Cat 10 F-06, Cat 19 §2.3
- **Niveau** : 🟠 BIG ARCHITECTURAL
- **Description** : externaliser secrets de `.env` vers vault (Doppler free tier ≤5 users). Rotation 90j, audit log inclus, scoping per-env.
- **Pourquoi dangereux** : si misconfig, secrets en clair restent dans Fly.io env vars → audit non passé.
- **Réversibilité** : facile (re-mettre `.env`).
- **Plan de rollback** : Fly.io secrets natifs comme fallback.
- **DÉCISION REQUISE** : ☐ Approuver / ☐ Modifications / ☐ Skip / ☐ Discuter

### [DG-030] Webhook B2B publisher + DLQ + signature HMAC
- **Catégorie source** : Cat 11 C9, Cat 8 §1.6
- **Niveau** : 🟠 BIG ARCHITECTURAL
- **Description** : nouveau `WebhookPublisher` qui souscrit aux insights émis et enqueue dans `webhook_queue.py` (déjà livré). DLQ + retry exp backoff + signature `t=,v1=` Stripe-style.
- **Réversibilité** : feature flag `WEBHOOK_B2B_ENABLED=true|false`.
- **Pré-requis** : Cat 11 livraison de queue déjà OK.
- **DÉCISION REQUISE** : ☐ Approuver / ☐ Skip / ☐ Discuter

### [DG-031] Queue notifications Redis (rate-limit Telegram/Discord)
- **Catégorie source** : Cat 11 C4, Cat 17
- **Niveau** : 🟠 BIG ARCHITECTURAL
- **Description** : déplacer notifier-queue de RAM → Redis (multi-worker safe + persistance crash). Respect Retry-After header per-provider.
- **Pré-requis** : DG-020 (Redis branché).
- **DÉCISION REQUISE** : ☐ Approuver / ☐ Skip / ☐ Discuter

### [DG-032] Schema InsightSignal version bumps futurs
- **Catégorie source** : Cat 8 §1.6 (versioning v1 hardcodé), historique MTF Phase 1 bump 2.1.0→2.2.0
- **Niveau** : 🟠 BIG ARCHITECTURAL
- **Description** : tout bump v2.3.0+ doit suivre semver + breaking changes documentés + B2B consommateurs notifiés. Schémas obsolètes conservés ≥ 6 mois en mode lecture.
- **DÉCISION REQUISE** : ☐ Approuver process / ☐ Discuter

### [DG-033] Stack obs OpenTelemetry → Tempo/Jaeger + Sentry
- **Catégorie source** : Cat 12 §1
- **Niveau** : 🟠 BIG ARCHITECTURAL
- **Description** : distributed tracing avec OTLP. `trace_id` propagé. Sentry pour exceptions.
- **Validation** : overhead < 5 % CPU.
- **DÉCISION REQUISE** : ☐ Approuver / ☐ Modifications / ☐ Skip / ☐ Discuter

### [DG-034] Sweep state machine 432 cellules POST-scoring rework
- **Catégorie source** : Cat 16 P0 §4
- **Niveau** : 🟠 BIG ARCHITECTURAL (calibration)
- **Description** : sweep `enter × exit × confirm × cooldown × max_age × silent` × 3 assets sur 100 % data 7 ans. ~46 h calcul + analyse.
- **Pré-requis** : DG-025 (scoring v2 validé).
- **DÉCISION REQUISE** : ☐ Approuver post-S8 / ☐ Skip / ☐ Discuter

### [DG-035] CI bloquante 3 workflows GH Actions (ci/integration/nightly)
- **Catégorie source** : Cat 13 §2
- **Niveau** : 🟠 BIG ARCHITECTURAL (process)
- **Description** : remplacer `continue-on-error: true` par bloquant ; ajouter coverage gate 75 %, mypy strict sur src/api+intelligence+delivery.
- **Pourquoi dangereux** : PRs vont fail jusqu'à atteindre 75 % coverage zones revenue.
- **Réversibilité** : revert workflow.
- **Validation** : 1 sem mode "warn" → mode "enforce".
- **DÉCISION REQUISE** : ☐ Approuver / ☐ Modifications / ☐ Skip / ☐ Discuter

### [DG-036] Adoption Pydantic v2 forwards (migration v1 → v2 si reste)
- **Catégorie source** : Cat 2 (`SMCConfig` Pydantic v1 ligne 440-518)
- **Niveau** : 🟠 BIG ARCHITECTURAL
- **Description** : migrer le reste des modèles Pydantic v1 vers v2 (compat strict mode, validators).
- **Pré-requis** : tests existants verts (164 fichiers).
- **DÉCISION REQUISE** : ☐ Approuver / ☐ Skip / ☐ Discuter

### [DG-037] Pipeline incrémental SmartMoneyEngine (sliding window)
- **Catégorie source** : Cat 2 P1-5
- **Niveau** : 🟠 BIG ARCHITECTURAL
- **Description** : refactor `analyze()` batch → `analyze_incremental()` stateful pour latence p99 < 50 ms/bar.
- **Réversibilité** : facile (deux méthodes coexistent).
- **DÉCISION REQUISE** : ☐ Approuver P1 / ☐ Skip / ☐ Discuter

### [DG-038] DSAR endpoints `/me/data` + `/me` (RGPD art. 15 + 17)
- **Catégorie source** : Cat 18 §1.3 bloqueur #3
- **Niveau** : 🟠 BIG ARCHITECTURAL (compliance)
- **Description** : nouveau endpoints `GET /api/v1/me/data` (export JSON 30j SLA) + `DELETE /api/v1/me` (anonymisation signaux + purge SQLite cascade).
- **Pourquoi dangereux** : DELETE cascade peut casser audit-trail si mal conçu (hash-chain ledger preservation requise).
- **Réversibilité** : impossible une fois delete user.
- **Validation** : test e2e avec user dummy + audit ledger validation post-delete.
- **Plan de rollback** : aucun (delete = irréversible) — soft-delete avec retention 30j avant hard-delete recommandé.
- **DÉCISION REQUISE** : ☐ Approuver / ☐ Modifications / ☐ Skip / ☐ Discuter

### [DG-039] Single RiskManager canonique (drop 8 orphelins → 1)
- **Catégorie source** : Cat 7 §1.1
- **Niveau** : 🟠 BIG ARCHITECTURAL
- **Description** : consolider en 1 `RiskManager` qui implémente `IRiskManager` + intègre `KillSwitch` actuel. Expose `position_size`, `risk_score 0-100`, `kill_level`, dans `InsightSignalV2`.
- **Pré-requis** : DG-003 (drop orphelins).
- **DÉCISION REQUISE** : ☐ Approuver / ☐ Modifications / ☐ Skip / ☐ Discuter

---

## 🟡 RISKY OPERATIONAL — Config prod, secrets, paiements, légal, UX rupture

### [DG-040] Vision A vs Vision B — décision défaut Vision B narrative-first
- **Catégorie source** : Cat 3 §2
- **Niveau** : 🟡 RISKY OPERATIONAL (déjà tranchée mais nécessite confirmation écrite)
- **Description** : confirmer par défaut Vision B (narrative-first) selon `memory/a1_verdict_2026_05_01.md`. Pas de retry Vision A pendant ≥ 90 jours sauf forward-test PF > 1.30 (improbable ~5-10 %).
- **Pourquoi critique** : engage 320 h Phase 2B vs 320 h Phase 2A. Reversal coûte 6 mois.
- **DÉCISION REQUISE** : ☐ Approuver Vision B / ☐ Discuter Vision A retry / ☐ Skip pour l'instant

### [DG-041] TESTING_MODE=0 par défaut prod (déjà patché ; gate CI)
- **Catégorie source** : Cat 10 F-01 marqué CORRIGÉ 2026-04-29
- **Niveau** : 🟡 RISKY OPERATIONAL
- **Description** : confirmer `SENTINEL_TESTING_MODE` default = `"0"` (fail-closed) + warning au boot + gate CI qui fail si déployement prod avec `SENTINEL_TESTING_MODE=1`.
- **Validation** : `grep -n "SENTINEL_TESTING_MODE" src/api/auth.py` ; vérifier code current.
- **Pré-requis** : aucun (vérification).
- **DÉCISION REQUISE** : ☐ Confirmer + ajouter gate CI / ☐ Skip / ☐ Discuter

### [DG-042] Activation NARRATIVE_MODE=llm par défaut (per-tier override)
- **Catégorie source** : Cat 9, Cat 24, Conflict C13 du master plan
- **Niveau** : 🟡 RISKY OPERATIONAL
- **Description** : passer du défaut template → tier-routed LLM (FREE=template, STARTER=Haiku, PRO=Sonnet, INSTITUTIONAL=Opus on-demand). Coût LLM monte avec MAU.
- **Pourquoi dangereux** : facture Anthropic peut exploser si abuse (free tier sans hard cap signaux). Cf. DG-046 hard caps.
- **Plan de rollback** : env var `NARRATIVE_MODE=template` global fallback.
- **Pré-requis** : DG-046 (hard caps), DG-020 (Redis cache), DG-052 (cost monitoring + alerting Anthropic spend).
- **DÉCISION REQUISE** : ☐ Approuver / ☐ Modifications / ☐ Skip / ☐ Discuter

### [DG-043] Stripe live mode + Customer Portal activation
- **Catégorie source** : Cat 1 P0-T2/T3
- **Niveau** : 🟡 RISKY OPERATIONAL (vraies CB clients)
- **Description** : créer 4 products + 8 price IDs en Stripe LIVE mode (vs TEST), activer Customer Portal, webhook live signé.
- **Pourquoi dangereux** : vrais paiements → erreurs facturation = remboursements + disputes.
- **Réversibilité** : difficile (subscriptions actives).
- **Validation** : test Checkout en mode TEST end-to-end (carte 4242) ; webhook idempotent test replay même event ; refund automation test ; CGV publiée.
- **Plan de rollback** : désactiver products en Stripe Dashboard ; refund manuel.
- **Pré-requis** : DG-051 (CGU/Privacy signés avocat), DG-053 (geo-block activé production), DG-027 ou alternative data live.
- **DÉCISION REQUISE** : ☐ Approuver / ☐ Modifications / ☐ Skip / ☐ Discuter

### [DG-044] Stripe Tax UE + TVA reverse charge B2B + EU OSS reporting
- **Catégorie source** : Cat 1 P0-T2/T5
- **Niveau** : 🟡 RISKY OPERATIONAL (légal fiscal)
- **Description** : activer Stripe Tax pour FR + UE pays principaux + reverse charge B2B EU. Configurer EU OSS reporting trimestriel.
- **Pourquoi dangereux** : infraction TVA = amende + fermeture compte Stripe.
- **Validation** : test geo-IP FR voit prix TTC TVA 20 % ; B2B EU voit reverse charge.
- **DÉCISION REQUISE** : ☐ Approuver / ☐ Modifications / ☐ Skip / ☐ Discuter

### [DG-045] Geo-block production activation US/QC/UK/OFAC
- **Catégorie source** : Cat 18 §1.2, Cat 1 P0-T5
- **Niveau** : 🟡 RISKY OPERATIONAL
- **Description** : déjà partiellement livré sprint W1 (`src/api/middleware/geo_block.py`). Étendre au front + Stripe Checkout. Test VPN obligatoire.
- **Réversibilité** : env var `GEO_BLOCK_ENABLED=true|false`.
- **Validation** : test VPN US → landing redirige `/restricted-region`, pas CTA paiement visible.
- **DÉCISION REQUISE** : ☐ Approuver / ☐ Skip / ☐ Discuter

### [DG-046] Hard caps signaux/mois par tier (30/200/800/2000)
- **Catégorie source** : Cat 1 P0-T4, eval_24 §11
- **Niveau** : 🟡 RISKY OPERATIONAL (rupture user)
- **Description** : `quota_manager.py` Redis counter + soft 80 % + hard 100 % + upgrade CTA.
- **Pré-requis** : DG-020 (Redis), DG-006 (tier rate-limit câblé).
- **Plan de rollback** : env var `HARD_CAPS_ENABLED=false`.
- **DÉCISION REQUISE** : ☐ Approuver / ☐ Modifications / ☐ Skip / ☐ Discuter

### [DG-047] MiFID 2024/2811 disclosure_mode défaut = qualitative (SL/TP non-chiffrés)
- **Catégorie source** : Cat 18 §1.1, Cat 7
- **Niveau** : 🟡 RISKY OPERATIONAL
- **Description** : nouveau param API `disclosure_mode=numeric|qualitative`, défaut `qualitative` (pas de SL/TP chiffrés sauf user opt-in B2B explicite). Reformuler "BUY/SELL"→"long setup" prompt LLM (déjà fait sprint W3).
- **Pourquoi critique** : entrée mars 2026 directive finfluencer, claim chiffré = risque amende.
- **Réversibilité** : facile (param API).
- **Validation** : tests `test_compliance_copy.py` regex anti-claim probabiliste.
- **DÉCISION REQUISE** : ☐ Approuver / ☐ Modifications / ☐ Skip / ☐ Discuter

### [DG-048] Cookie banner CNIL-compliant (Tarteaucitron auto-hébergé)
- **Catégorie source** : Cat 18 §1.3 bloqueur #4
- **Niveau** : 🟡 RISKY OPERATIONAL (légal)
- **Description** : bandeau cookies 4 catégories (Necessary/Analytics/Marketing/Functional). Pas de tag GA4/Meta avant consent.
- **Réversibilité** : facile.
- **Pré-requis** : Cat 20 P0 landing en ligne.
- **DÉCISION REQUISE** : ☐ Approuver / ☐ Skip / ☐ Discuter

### [DG-049] CircuitBreaker thresholds modification prod
- **Catégorie source** : Cat 8 + Cat 12 (déjà en place threshold=3 LLM, threshold=5 Telegram)
- **Niveau** : 🟡 RISKY OPERATIONAL
- **Description** : modifier les seuils circuit breaker en prod = impacte disponibilité service.
- **Validation** : modification précédée d'analyse historique false-positive rate.
- **DÉCISION REQUISE** : ☐ Approuver (per-changement) / ☐ Discuter

### [DG-050] Activer RC Pro + Cyber assurance (3-5 k€/an)
- **Catégorie source** : Cat 18 §1.1
- **Niveau** : 🟡 RISKY OPERATIONAL (cash + légal)
- **Description** : souscrire Stoïk ou Hiscox bundle.
- **Pourquoi important** : sans RC Pro, capital personnel exposé si litige client.
- **DÉCISION REQUISE** : ☐ Approuver / ☐ Modifications / ☐ Skip / ☐ Discuter

### [DG-051] CGU/CGV/Privacy v2 signées par avocat fintech FR
- **Catégorie source** : Cat 18 §1.3 bloqueur #1 (W4)
- **Niveau** : 🟡 RISKY OPERATIONAL (légal)
- **Description** : engagement cabinet avocat fintech ≥ 5 ans pratique + relecture + signature versionnée + tampon date.
- **Coût** : 3-5 k€ one-shot.
- **Pré-requis aval** : DG-043 (Stripe live mode) bloqué tant que non signé.
- **DÉCISION REQUISE** : ☐ Approuver budget + démarrer RFQ / ☐ Discuter

### [DG-052] Cost monitoring Anthropic + alerte spend
- **Catégorie source** : Cat 12 + Cat 9
- **Niveau** : 🟡 RISKY OPERATIONAL
- **Description** : Prometheus gauge `llm_cost_usd_total` + alerte Discord/email à $X/jour seuil.
- **Pourquoi important** : un free user abusif = $40/mo perte sur 1 user (eval_24 §8 stress S3).
- **Pré-requis** : DG-046 hard caps.
- **DÉCISION REQUISE** : ☐ Approuver / ☐ Skip / ☐ Discuter

### [DG-053] Activation `verify_data_quality` au boot fail-fast
- **Catégorie source** : Cat 5 §1.3 bug #3
- **Niveau** : 🟡 RISKY OPERATIONAL
- **Description** : scanner refuse de booter si feed XAU coverage < seuil (ex 95 %). Évite consommation silente d'un feed 63 %.
- **Validation** : test boot avec `XAU_15MIN_2019_2025.csv` doit fail.
- **Plan de rollback** : env var `STRICT_DATA_QUALITY=false` pour rollback urgence.
- **DÉCISION REQUISE** : ☐ Approuver / ☐ Modifications / ☐ Skip / ☐ Discuter

### [DG-054] Telegram retry + dedup activation (rupture flood-control)
- **Catégorie source** : Cat 11 C1, C2, C3
- **Niveau** : 🟡 RISKY OPERATIONAL
- **Description** : fix python-telegram-bot v20+ async + retry exp backoff sur 429 + dedup `(chat_id, signal_id)` + respect Retry-After.
- **Pourquoi dangereux** : aujourd'hui `send_message` est sync coroutine non awaitée silently → messages perdus invisible. Réparer expose vraies erreurs.
- **Réversibilité** : facile.
- **Validation** : test integration ≥ 30 abonnés simultanés sans flood ban.
- **DÉCISION REQUISE** : ☐ Approuver / ☐ Skip / ☐ Discuter

### [DG-055] Activation HMAC admin replay protection nonce-based
- **Catégorie source** : Cat 10 F-03 OUVERT
- **Niveau** : 🟡 RISKY OPERATIONAL (sécu)
- **Description** : remplacer signature TS-only → include `route+body+ts+nonce` in signed payload. Évite cross-route replay 5 min.
- **Pourquoi critique** : privilege escalation possible.
- **Réversibilité** : difficile (clients admin doivent re-signer).
- **DÉCISION REQUISE** : ☐ Approuver / ☐ Modifications / ☐ Skip / ☐ Discuter

### [DG-056] UNIQUE constraint sur `users.api_key_id`
- **Catégorie source** : Cat 10 F-04 OUVERT
- **Niveau** : 🟡 RISKY OPERATIONAL (sécu + data)
- **Description** : ajouter UNIQUE constraint sur `users.api_key_id` (migration ALTER TABLE).
- **Pourquoi critique** : account hijack possible (même clé liée à plusieurs users).
- **Validation** : dump SQLite, vérifier zero doublon avant ALTER ; fix doublons manuel si présents.
- **Plan de rollback** : DROP CONSTRAINT.
- **DÉCISION REQUISE** : ☐ Approuver / ☐ Skip / ☐ Discuter

### [DG-057] Lire `subscription_expires` lors auth (sinon abonné qui ne paye plus reste premium)
- **Catégorie source** : Cat 10 F-05 OUVERT
- **Niveau** : 🟡 RISKY OPERATIONAL
- **Description** : lecture `subscription_expires` dans `require_api_key` ; downgrade tier auto si expiré.
- **Pourquoi important** : revenue leak (users qui annulent gardent accès premium).
- **Validation** : test fixture user avec `subscription_expires=hier` → tier reverts FREE.
- **DÉCISION REQUISE** : ☐ Approuver / ☐ Skip / ☐ Discuter

### [DG-058] RAG pipeline production activation
- **Catégorie source** : Cat 3 (RAG livré `src/intelligence/rag/pipeline.py`)
- **Niveau** : 🟡 RISKY OPERATIONAL
- **Description** : activer RAG (BM25+dense+RRF) sur 50 sources curées en prod, pas juste recherche.
- **Pré-requis** : sources licenciées (Cat 5/18 compliance).
- **DÉCISION REQUISE** : ☐ Approuver / ☐ Discuter

---

## 🟣 POLITIQUE / MÉTIER — Décisions stratégiques

### [DG-070] Pricing v1 lock définitif : FREE / $29 / $79 / $1990
- **Catégorie source** : Cat 1 P0-T1, eval_27
- **Niveau** : 🟣 POLITIQUE
- **Description** : remplacer BP actuel ($49/$99/$149) par grille recommandée eval_27 (FREE/STARTER $29/PRO $79/INSTITUTIONAL $1990) + annual 16.7 % off + dual trial 14j.
- **Pourquoi critique** : INSTITUTIONAL passe de $149 à $1990 = changement positionnement.
- **Réversibilité** : difficile (clients existants grandfathered).
- **Validation** : peer review 1 trader Persona A + 1 prospect B2B broker.
- **DÉCISION REQUISE** : ☐ Approuver grille v1 / ☐ Modifications / ☐ Discuter

### [DG-071] Pivot B2B-API brokers — budget + dev MVP 80h
- **Catégorie source** : Cat 1, memory `decision_matrix_2026_04_30.md`
- **Niveau** : 🟣 POLITIQUE
- **Description** : développer B2B-API pour IC Markets / Pepperstone / Exness en parallèle B2C. Cible ARR M12 $30-60k → $310k.
- **Réversibilité** : facile (continuer B2C en parallèle).
- **Pré-requis** : Cat 7 risk score consolidé, Cat 11 webhook publisher, Cat 18 DPA template B2B.
- **DÉCISION REQUISE** : ☐ Approuver pivot B2B parallèle / ☐ Focus B2C seul / ☐ Discuter

### [DG-072] Annonce track-record publique 60-90j Telegram (forward only)
- **Catégorie source** : Cat 1 P0, eval_25 §11.2
- **Niveau** : 🟣 POLITIQUE
- **Description** : ouvrir Telegram public channel "Smart Sentinel — Public Tape", forward only (PAS de signaux passés cherrypickés), 60-90j avant monetization.
- **Pourquoi important** : preuve commerciale unique défensable + alignement compliance backtest guardrails.
- **Réversibilité** : impossible (channel public archivé indéfiniment, perceived sunk-cost trap).
- **Validation** : 30j avec ≥ 5 paper trades fermés avant exposition publique.
- **DÉCISION REQUISE** : ☐ Approuver ouverture S2 / ☐ Discuter timing

### [DG-073] Cadrage MiFID : reformulation "signaux"→"analyses" / "lecture algorithmique"
- **Catégorie source** : Cat 18 §1.1, eval_29
- **Niveau** : 🟣 POLITIQUE (marque)
- **Description** : changement vocabulaire commercial sur tous supports (landing, Telegram, Discord, email, API docs, narratives LLM). "Smart Sentinel ne vend pas de signaux mais des analyses algorithmiques contextuelles."
- **Pourquoi critique** : "signaux trading" = mot-clé déclencheur finfluencer-rule.
- **Réversibilité** : difficile (memes, SEO acquis).
- **DÉCISION REQUISE** : ☐ Approuver reformulation totale / ☐ Modifications / ☐ Discuter

### [DG-074] Décision GO instruments GA : XAU+EUR (USOIL P1, drop BTC/US500/JPY/GBP)
- **Catégorie source** : Cat 14 §2.1, eval_20
- **Niveau** : 🟣 POLITIQUE
- **Description** : commercialiser uniquement XAU+EUR en GA S16. USOIL ajouté post-S16 si data Polygon OK. Drop BTC/US500/JPY/GBP backlog.
- **Réversibilité** : facile (presets restent en code).
- **Pré-requis** : sweep EUR validation, suppression message marketing "6 instruments".
- **DÉCISION REQUISE** : ☐ Approuver / ☐ Modifications / ☐ Discuter

### [DG-075] Engagement cabinet avocat fintech FR 3-5 k€
- **Catégorie source** : Cat 18 §1.3 bloqueur #1
- **Niveau** : 🟣 POLITIQUE (cash + légal)
- **Description** : décision dépense one-shot 3-5 k€ pour avocat fintech FR (CGU + Privacy + DPA B2B). RFQ 3 cabinets.
- **Pré-requis** : aucun (à démarrer S1).
- **DÉCISION REQUISE** : ☐ Approuver / ☐ Discuter budget

### [DG-076] Data licensing budget : Trading Economics $79/mo + Polygon (P1)
- **Catégorie source** : Cat 5 §1.4, Cat 15
- **Niveau** : 🟣 POLITIQUE
- **Description** : souscriptions data récurrentes $79-208/mo. Décision essentielle pour live (calendar + OHLCV).
- **Réversibilité** : facile (annuel TE, mensuel Polygon).
- **Pré-requis aval** : DG-005 cut Dukascopy/FF live.
- **DÉCISION REQUISE** : ☐ Approuver TE seul / ☐ Approuver TE+Polygon / ☐ Discuter

### [DG-077] Public déclaration "honest confidence" comme USP
- **Catégorie source** : Cat 1, master plan §1.2
- **Niveau** : 🟣 POLITIQUE (marque)
- **Description** : positioning marque autour de "honest confidence" / "calibration transparente" / "PF rolling exposé live" vs concurrents qui claiment edge.
- **Pourquoi important** : c'est la défense face à reproche "your A1 failed, why should we trust you?"
- **Réversibilité** : difficile (réputation acquise).
- **DÉCISION REQUISE** : ☐ Approuver positionnement / ☐ Modifications / ☐ Discuter

### [DG-078] Open-source rubric LLM narrative (eval_26 Diff #3)
- **Catégorie source** : eval_26 §6.1
- **Niveau** : 🟣 POLITIQUE
- **Description** : publier la rubric d'évaluation narrative LLM (5 critères) sur GitHub open-source pour démontrer méthode.
- **Pourquoi important** : différenciation contre TradingView Copilot (boîte noire).
- **Réversibilité** : difficile (code public).
- **Pré-requis** : rubric stabilisée (Cat 9 P0).
- **DÉCISION REQUISE** : ☐ Approuver / ☐ Modifications / ☐ Discuter

### [DG-079] Refund policy 30j first-month no-questions
- **Catégorie source** : Cat 1 P0-T6
- **Niveau** : 🟣 POLITIQUE (CGV)
- **Description** : remboursement 30j premier mois, no-questions-asked.
- **Validation** : compatible loi Hamon FR 14j minimum.
- **DÉCISION REQUISE** : ☐ Approuver / ☐ Modifications / ☐ Discuter

### [DG-080] Démo INSTITUTIONAL : book demo Calendly, pas auto-checkout
- **Catégorie source** : Cat 1 P0-T10, Conflict C9
- **Niveau** : 🟣 POLITIQUE
- **Description** : INSTITUTIONAL card sur pricing page = CTA "Book a demo" Calendly intégré ; pas Subscribe auto S1-S12 tant que Cat 7 risk + Cat 6 gates non livrés.
- **DÉCISION REQUISE** : ☐ Approuver / ☐ Discuter

### [DG-081] Kill criterion S8 : Vision B Brier skill < +2 % → Pivot B2B only
- **Catégorie source** : Cat 2 §10.4 Kill A, master plan §6
- **Niveau** : 🟣 POLITIQUE
- **Description** : décision écrite à respecter (anti-rationalisation post-hoc). Si scoring v2 fail à S8 → focus B2B-API context layer + Telegram FREE lead-magnet seul.
- **Pré-requis** : commitment écrit dans `reports/governance/kill_criteria_board.md`.
- **DÉCISION REQUISE** : ☐ Approuver kill criterion / ☐ Modifications / ☐ Discuter

### [DG-082] Adhésion médiation conso CM2C/MEDICYS (150 €/an)
- **Catégorie source** : Cat 18 §1.1
- **Niveau** : 🟣 POLITIQUE (légal FR)
- **Description** : obligation L.612-1 code conso : adhésion plateforme médiation pour vente B2C en France.
- **Pré-requis** : DG-051 (avocat valide quelle plateforme).
- **DÉCISION REQUISE** : ☐ Approuver / ☐ Discuter

### [DG-083] Decoy effect : INSTITUTIONAL $1990 visible toujours sur pricing
- **Catégorie source** : Cat 1 P0-T10, eval_27 §4.2
- **Niveau** : 🟣 POLITIQUE (UX)
- **Description** : 4 cards toujours visibles incluant INSTITUTIONAL $1990 (decoy +25-40 % conv PRO).
- **Pré-requis** : DG-070, DG-080.
- **DÉCISION REQUISE** : ☐ Approuver / ☐ Modifications / ☐ Discuter

### [DG-084] Trial dual : 14j sans carte (FREE→STARTER) + 14j avec carte (STARTER→PRO)
- **Catégorie source** : Cat 1 P0-T1, eval_27 §8
- **Niveau** : 🟣 POLITIQUE
- **Description** : double mécanique trial (+$1168 MRR vs freemium-only selon eval_27).
- **Pré-requis** : DG-043 Stripe live, DG-046 hard caps.
- **DÉCISION REQUISE** : ☐ Approuver / ☐ Modifications / ☐ Discuter

### [DG-085] B2B INSTITUTIONAL contrat 12 mois minimum
- **Catégorie source** : Cat 1 P0-T1, eval_27 §7.4
- **Niveau** : 🟣 POLITIQUE
- **Description** : INSTITUTIONAL $1990/mo signe contrat 12 mois min ($23 880 ARR garanti).
- **DÉCISION REQUISE** : ☐ Approuver / ☐ Modifications / ☐ Discuter

---

## Section finale — Recommandation d'ordonnancement

Pour minimiser le risque cumulé, exécuter dans cet ordre :

### Étape 1 (S1-S2) — Bases SAFE no-regret + 🟢 quick-wins
- Cf. `00_NO_REGRET_QUICK_WINS.md` : 30 items < 2h chacun, en parallèle.
- DG-001, DG-002, DG-009 (suppressions code mort, déjà broken).
- DG-041 (vérif TESTING_MODE=0).

### Étape 2 (S2-S4) — 🟡 RISKY OPERATIONAL avec rollback facile
- DG-053 (data quality fail-fast au boot, env var rollback).
- DG-045 (geo-block prod activation, env var rollback).
- DG-047 (MiFID disclosure_mode param API, defaults safe).
- DG-052 (cost monitoring + alerting).
- DG-054 (Telegram retry+dedup).

### Étape 3 (S2-S6) — 🟣 POLITIQUE déclenchées en parallèle
- DG-075 (avocat engagement) — démarrer S1 immédiat.
- DG-040 (Vision B confirmation écrite).
- DG-073 (cadrage MiFID vocabulaire).
- DG-076 (souscription TE).
- DG-072 (track-record Telegram public ouverture S2).
- DG-074 (instruments GA décision).
- DG-070 (pricing v1 lock écrit).

### Étape 4 (S4-S10) — 🟠 BIG ARCHITECTURAL séquentiel avec gates
- DG-022 (Fly.io deploy) → DG-029 (vault) → DG-020 (Redis) → DG-024 (multi-worker).
- DG-021 (async I/O) après DG-020.
- DG-027 (TE souscription + branchement).
- DG-025 (scoring rework) après DG-027 + DG-035 (CI bloquante).
- DG-038 (DSAR endpoints).
- DG-039 (RiskManager consolidé).

### Étape 5 (S10-S16) — 🔴 DESTRUCTIVE et 🟡 RISKY OPERATIONAL à fort impact
- DG-003 (drop risk orphelins) après DG-039.
- DG-005, DG-007, DG-010, DG-013 (data/scripts decom) après DG-027 + DG-076.
- DG-006 (tier rate-limit enforcement) après DG-046 (hard caps).
- DG-043 (Stripe live mode) après DG-051 (CGU signées) + DG-044 (Tax UE) + DG-048 (cookies).
- DG-042 (NARRATIVE_MODE=llm) après DG-052 (cost monitor).

### Étape 6 (post-S16) — Defer P1
- DG-026 (Postgres migration) si MAU > 1 k.
- DG-037 (pipeline incrémental SMC).
- DG-058 (RAG prod).
- DG-078 (rubric open-source).

**Aucune action 🔴 DESTRUCTIVE ne se fait sans backup explicite ET sans validation utilisateur item-par-item.**

---

**FIN CATALOGUE.** Total catalogué : **65 items** dont 14 🔴 destructive, 19 🟠 architectural, 21 🟡 risky operational, 16 🟣 politique. À valider avant toute exécution master plan.
