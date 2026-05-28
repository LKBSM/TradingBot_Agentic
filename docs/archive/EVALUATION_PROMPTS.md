# Smart Sentinel AI — 29 Prompts d'Évaluation Sectorielle

> **Contexte produit** : SaaS d'intelligence de marché propulsé par IA. Pipeline = DataProvider → SmartMoneyEngine → ConfluenceDetector → VolForecaster → LLMNarrativeEngine → SemanticCache → SignalStore → Telegram. Scoring règle-basé 0-100, LLM explique. XAU/USD 15min en focus, 6 instruments présets, timeframes M1→W1. 4 tiers (FREE/ANALYST/STRATEGIST/INSTITUTIONAL) en stand-by (phase de test perso).
>
> **État actuel connu (2026-04-24)** :
> - Sweep 7 ans : production config = **0 trade**, score max observé **55.5/100** (plafond 70 car News+Vol=0 en replay), PF max **0.96**. Verdict provisoire : **non commercialisable en l'état**.
> - Data quality : `XAU_15MIN_2019_2025.csv` couverture 63 % (feed distinct du 2019-2024 à 97.6 %).
> - BOS/CHOCH : corrigé (event/state split), PF 0.39 → 0.94 mais shorts profitables / longs non, score inversement corrélé au PnL.
> - 1366+ tests, 0 régression.
>
> **Règles communes à tous les prompts** :
> 1. Commencer par cartographier les fichiers/symboles concernés (Grep/Glob/Read).
> 2. Évaluer **technique** (code, perf, sécurité, robustesse) **ET commercial** (impact revenu, rétention, différenciation, coût unitaire).
> 3. Se positionner par rapport à **l'état de l'art 2026** (papers récents, produits concurrents).
> 4. Livrer un **diagnostic noté /10**, **top 5 améliorations priorisées** (matrice effort × impact), **plan d'exécution** (quick wins < 1 j, moyen terme < 1 sem, long terme > 1 sem), **KPIs** mesurables post-amélioration.
> 5. Aucune amélioration ne doit dégrader une autre dimension sans l'énoncer explicitement (trade-off assumé).
> 6. Justifier chaque recommandation par une référence (paper, produit, benchmark) OU un chiffre du replay/backtest.

---

## Prompt 01 — Architecture globale & orchestration du pipeline

**Périmètre** : `src/intelligence/main.py`, `src/intelligence/sentinel_scanner.py`, `config.py`, dépendances inter-modules.

**Objectif commercial** : la robustesse de l'orchestration détermine l'uptime perçu par l'abonné payant. Un crash = un signal manqué = churn.

**Mission** :
1. Cartographier le graphe de dépendances (qui appelle qui, couplage, cycles). Produire un diagramme Mermaid.
2. Identifier les points de contention (appels synchrones bloquants, GIL, I/O série).
3. Évaluer la résilience (que se passe-t-il si DataProvider tombe ? LLM timeout ? Telegram rate-limited ?).
4. Mesurer le cold-start (build_system → premier signal) et le steady-state (tick → signal).
5. Vérifier que la séparation producer/consumer est propre (un bug dans LLM ne doit pas bloquer le scanner).
6. Benchmark vs. architectures modernes : event-driven (NATS/Kafka), async FastAPI, actor model.
7. Commercialiser : peut-on multi-tenant (50 utilisateurs, 6 symboles chacun) sans refactor majeur ?

**Livrables** : diagramme, tableau des SPOF, note /10 sur robustesse + scalabilité, top 5 refactors priorisés, budget CPU/RAM par tenant.

---

## Prompt 02 — ConfluenceDetector (cœur du scoring 0-100)

**Périmètre** : `src/intelligence/confluence_detector.py` (fichier modifié non commité), tous les tests associés.

**Objectif commercial** : **le scoring EST le produit**. Si les signaux 75+ ne produisent pas de PnL, le produit n'a aucune valeur perçue. Rappel : sweep 7 ans = 0 trade en prod, score max 55.5 (plafond 70).

**Mission** :
1. Lister chaque composant du score (BOS, CHOCH, OB, FVG, liquidity, regime, vol, news…), leur poids, leur plage, leur corrélation entre eux.
2. Prouver ou réfuter l'**orthogonalité** des features (matrice corrélation sur 7 ans de data).
3. Auditer le plafond 70/100 quand News+Vol=0 : est-ce un bug de design (composants obligatoires) ou un choix assumé ? Impact commercial : seuil 75 inatteignable en replay = produit intestable.
4. Vérifier la **calibration du score** : un score 80 doit correspondre empiriquement à une proba de gain > un score 60 (courbe de calibration, Brier score).
5. Tester la **stabilité temporelle** : le score 75 de 2020 veut-il dire la même chose en 2025 ?
6. Benchmark : Bayesian scoring, logistic regression avec features smart-money, LightGBM classifieur (entraîné sur PnL direct).
7. Proposer un **seuil dynamique** par régime de marché.

**Livrables** : rapport de calibration (reliability diagram), matrice de corrélation, recommandation « garder / refondre / remplacer », note /10 sur valeur commerciale du score actuel.

---

## Prompt 03 — Smart Money Engine (BOS, CHOCH, Order Blocks, FVG)

**Périmètre** : modules smart-money dans `src/intelligence/` et leurs tests (`test_signal_state_machine*`, replay).

**Objectif commercial** : c'est le **marketing hook** (ICT/SMC est très en vogue chez retail 2024-2026). La qualité des détections = crédibilité produit.

**Mission** :
1. Vérifier la conformité aux définitions canoniques (ICT, Lux Algo, etc.) — BOS vs CHOCH split déjà corrigé, mais auditer les Order Blocks (mitigated vs active, lookback, volume filter) et FVG (fill % logic).
2. Quantifier le **taux de faux positifs** sur XAU 15min 7 ans (combien de BOS échouent à produire un retest valide ?).
3. Tester sur d'autres assets (EURUSD, BTCUSD) pour vérifier la généralisabilité — la config actuelle est XAU-centric.
4. Mesurer la **latence de détection** (combien de bars après l'événement réel ?).
5. Shorts profitables / longs non : investiguer le biais directionnel — biais structurel XAU ? Bug d'asymétrie ?
6. Benchmark vs. implémentations open-source (smc-python, freqtrade SMC modules) et produits commerciaux (LuxAlgo, GoldBull).
7. Commercial : quelles visualisations manquent (zones OB sur chart, annotations FVG) pour justifier un tier payant ?

**Livrables** : matrice de confusion BOS/CHOCH/OB/FVG, tableau d'asymétrie long/short par année, liste de features visuelles à ajouter, note /10 sur différenciation.

---

## Prompt 04 — Volatility Forecasting (HAR-RV, LightGBM, Hybrid)

**Périmètre** : `src/intelligence/volatility_forecaster.py`, `src/intelligence/volatility_lgbm.py`, `scripts/colab_*_vol_poc.py`.

**Objectif commercial** : vol forecasting = feature premium (tier STRATEGIST/INSTITUTIONAL). Le claim « 20-35 % RMSE improvement » doit être prouvé hors-échantillon.

**Mission** :
1. Vérifier l'absence de **data leakage** (jointures temporelles, volatilité réalisée calculée avec des bars futures, etc.).
2. Reproduire le benchmark RMSE HAR-RV vs ATR14 sur walk-forward 2019-2025, avec train/test strict.
3. Auditer le Hybrid (HAR base + LGBM residual) : le résidu contient-il du signal ou du bruit blanc ? Test de Diebold-Mariano vs HAR seul.
4. Vérifier la **stabilité du modèle** (drift de performance sur 2022, 2023, 2024 — régimes différents).
5. Le `VOL_MODE=hybrid` par défaut est-il justifié si LGBM seul est comparable ?
6. Coût opérationnel : poids modèle, latence d'inférence (cible < 50 ms par tick).
7. Benchmark 2026 : N-BEATS / TFT vs HAR-RV (papers récents réalisent que les TSFM n'améliorent pas significativement la vol — mais valider avec les derniers arxiv).
8. Commercial : packaging de la sortie (quantiles, VaR 1d, cone de vol à 24h) pour vente tier.

**Livrables** : rapport de walk-forward avec IC bootstrap, arbre de décision VOL_MODE, bench latence, note /10 sur « production-ready vs POC ».

---

## Prompt 05 — LLM Narrative Engine (Claude API, qualité, coût)

**Périmètre** : `src/intelligence/llm_narrative_engine.py`, `src/intelligence/template_narrative_engine.py`, prompts associés.

**Objectif commercial** : c'est **le différenciateur N°1 vs concurrence** (TradingView, LuxAlgo ne font pas de narration). Coût LLM = principale variable d'OpEx.

**Mission** :
1. Auditer les prompts envoyés à Claude : structure (system/user), longueur, variables injectées, anti-hallucination.
2. Vérifier que le **prompt caching** Anthropic est activé (cache_control sur system + examples) et mesurer le cache hit rate réel. TTL 5 min — adapter fréquence d'appel.
3. Choix de modèle : Opus 4.7 ou Sonnet 4.6 ou Haiku 4.5 selon tier ? Tableau coût/qualité.
4. Latence P50/P95/P99 des appels, fallback si timeout.
5. Évaluation qualitative des narratives (rubric : factuelle, actionnable, non-générique, sans hallucination de chiffres).
6. Tester l'injection d'indicateurs dans le prompt (le LLM doit expliquer LE signal, pas en inventer un nouveau).
7. Multi-langue (FR/EN/ES) — impact sur TAM.
8. Commercial : quels tiers reçoivent quel modèle ? Quota par tier ? Pay-per-narrative ?

**Livrables** : tableau coût/1000 signaux par modèle, rubric d'évaluation + 20 narratives notées, plan de caching agressif, note /10 sur différenciation.

---

## Prompt 06 — Semantic Cache (hit rate, économies LLM)

**Périmètre** : `src/intelligence/semantic_cache.py`.

**Objectif commercial** : chaque cache hit = économie directe (~$0.01-0.05 par narrative selon modèle). À 10k signaux/mois, une amélioration de hit rate de 20 % = revenu brut préservé.

**Mission** :
1. Mesurer le hit rate actuel sur 30 jours de replay (logs ou instrumentation à ajouter).
2. Auditer le modèle d'embedding (local vs API), la distance de similarité (cosine threshold), le TTL.
3. Tester des alternatives : cache exact par signature de signal (symbol + score rounded + regime + vol_bucket), cache sémantique par hashing LSH.
4. Vérifier les collisions (cas où un cache hit renvoie une narrative incorrecte parce que trop similaire en surface mais contexte différent).
5. Taille mémoire, éviction strategy (LRU, LFU), persistance disque.
6. Benchmark vs GPTCache, Redis Semantic Cache, Portkey.
7. Commercial : plus le cache est bon, plus la marge brute est élevée — cible hit rate ≥ 60 %.

**Livrables** : courbe hit rate vs threshold, tableau d'économies projetées, note /10 sur efficacité.

---

## Prompt 07 — Signal State Machine (HOLD/BUY/SELL trust layer)

**Périmètre** : `src/intelligence/signal_state_machine.py`, `signal_state_machine.md`, 54 tests associés.

**Objectif commercial** : empêche le spam Telegram (rétention utilisateur) et les faux signaux (crédibilité).

**Mission** :
1. Auditer les paramètres actuels (hystérésis, confirmation bars, cooldown, lifetime, opposing-lockout). Sont-ils dérivés empiriquement ou choisis à la main ?
2. Mesurer le **trade-off latence vs qualité** : un lockout trop long = signaux tardifs ; trop court = whipsaw.
3. Tester sur replay 7 ans : combien de signaux auraient été émis sans / avec state machine ? PF des deux cohortes.
4. Vérifier la persistance (reprise après crash — déjà implémentée mais auditer staleness guard).
5. Commercial : exposer les paramètres en config utilisateur (tier STRATEGIST+) ou garder opaque ?
6. Benchmark : filtres HMM, Bayesian change-point detection.

**Livrables** : heatmap PF vs (hysteresis × cooldown), recommandation de tuning par symbole, note /10.

---

## Prompt 08 — Data Providers & Data Quality

**Périmètre** : `src/intelligence/data_providers.py`, `src/intelligence/data_quality.py`, `scripts/download_dukascopy_xau.py`, `scripts/audit_data_quality.py`.

**Objectif commercial** : **root cause du verdict actuel « non commercialisable »**. Un produit qui tourne sur 63 % de data n'est pas crédible.

**Mission** :
1. Auditer tous les feeds disponibles (Dukascopy, MT5 history export, CSV locaux) — couverture, latence, coût, licence commerciale.
2. Détecter gaps, weekend rollovers, splits, ticker changes (XAUUSD vs GOLD vs XAU/USD).
3. Vérifier la **cohérence OHLC** (H ≥ max(O,C), L ≤ min(O,C), volume > 0).
4. Tester la résistance aux **données corrompues** (NaN, duplicates, out-of-order timestamps).
5. Construire un pipeline d'**ingestion temps réel** (WebSocket vs REST polling) pour la prod — actuellement c'est du replay CSV.
6. Licence : Dukascopy autorise-t-il l'usage commercial ? Sinon, alternatives (Polygon.io, Tiingo, Twelve Data, Databento).
7. Coût : budget data par mois pour 6 instruments × 6 timeframes × tick-level.

**Livrables** : tableau fournisseurs (couverture × latence × coût × licence), plan d'ingestion live, note /10 sur fiabilité, budget OpEx data.

---

## Prompt 09 — Sentinel Scanner (boucle temps réel)

**Périmètre** : `src/intelligence/sentinel_scanner.py`.

**Objectif commercial** : c'est le cœur battant — si ça lag, les signaux sont en retard, les abonnés perdent de l'argent, ils churnent.

**Mission** :
1. Mesurer la **cadence réelle** (bars/seconde traitables), le P99 de la boucle de scan.
2. Vérifier la gestion d'erreurs (une exception symbol-level ne doit pas tuer le scanner multi-symboles).
3. Tester la **scalabilité horizontale** : 1 scanner par symbole vs 1 scanner multi-symbole ?
4. Observabilité : logs suffisants pour debug prod (trace id par signal, corrélation bout en bout).
5. Backpressure : que se passe-t-il si Telegram est down pendant 10 min — file d'attente, TTL, dedup ?
6. Graceful shutdown : état sauvegardé, pas de signaux dupliqués au restart.
7. Commercial : SLA promis à l'abonné (« signal dans les 30 s après la clôture du bar ») atteignable ?

**Livrables** : profil de perf (py-spy / cProfile), liste de edge cases, SLO proposé, note /10.

---

## Prompt 10 — API FastAPI (routes, perf, sécurité)

**Périmètre** : `src/api/app.py`, `src/api/routes/*`.

**Objectif commercial** : l'API est le canal B2B (intégrations partenaires, futur dashboard web). Qualité API = capacité à vendre en entreprise.

**Mission** :
1. Audit RESTful : naming, versioning (`/v1/`), pagination, filtering, erreurs standardisées (RFC 7807).
2. OpenAPI / Swagger auto-généré : complet, exemples, descriptions ?
3. Sécurité : injection, CSRF, SSRF, IDOR sur les routes admin/narratives/signals.
4. Rate limiter actuel (100 req/min IP) : adapté B2B ? Besoin de tier-based (INSTITUTIONAL = 10k req/min) ?
5. Latence P95/P99 par endpoint.
6. Webhook support (push vs pull) pour clients entreprise.
7. Idempotency keys pour POST critiques.
8. Commercial : SDK client (Python, JS) généré depuis OpenAPI — valeur vente B2B.

**Livrables** : rapport de conformité API standards, liste CVE potentiels, bench latence, note /10.

---

## Prompt 11 — Auth & Tier Manager (SaaS gating)

**Périmètre** : `src/api/auth.py`, `src/api/tier_manager.py`, `src/api/dependencies.py`.

**Objectif commercial** : le **gating = le business model**. Bug de gating = revenu perdu ou service offert gratuitement.

**Mission** :
1. Vérifier chaque tier (FREE/ANALYST/STRATEGIST/INSTITUTIONAL) : quelles features gated, comment, où.
2. `SENTINEL_TESTING_MODE=1` bypass — risque si déployé par erreur en prod : audit + alerte CI.
3. Rotation de clés API, révocation, audit log des accès.
4. Stockage mot de passe / clé (hashing Argon2 ?, HMAC, pas de plain text).
5. Intégration Stripe / Paddle pour upgrades de tier (actuellement manuel ?).
6. OAuth / SSO pour tier INSTITUTIONAL.
7. Quota counters : stockage (Redis ?), reset journalier, burst allowance.
8. Commercial : friction d'onboarding FREE → ANALYST, proposer trial 14 j.

**Livrables** : matrice tier × feature × enforcement, plan Stripe intégration, note /10.

---

## Prompt 12 — Signal Store (SQLite, persistance, backup)

**Périmètre** : `src/api/signal_store.py`, `src/api/signal_tracker.py`.

**Objectif commercial** : les signaux historiques = valeur perçue (« preuve de track record »). Perte de données = perte de confiance irréversible.

**Mission** :
1. SQLite suffit-il pour multi-tenant production ? Limite de concurrent writes ?
2. Plan de migration Postgres (Neon / Supabase / RDS) : coût, complexité, temps d'exécution.
3. Schéma : index, foreign keys, normalisation, JSON columns pour flexibilité.
4. Rétention : règle métier (garder N jours par tier ?), archivage S3.
5. Backup : fréquence, RPO/RTO, test de restauration.
6. Query performance : « signals des 30 derniers jours pour l'user X » en < 100 ms ?
7. Audit trail (immutable log) pour preuves commerciales.
8. Commercial : export CSV / Parquet aux abonnés institutionnels.

**Livrables** : plan de migration Postgres, schéma optimisé, politique backup, note /10.

---

## Prompt 13 — Telegram Delivery

**Périmètre** : `src/delivery/telegram_notifier.py`.

**Objectif commercial** : Telegram = canal principal de livraison. Un signal raté ou mal formaté = friction utilisateur directe.

**Mission** :
1. Rate limits Telegram (30 msg/s bot global, 1 msg/s par chat) — gestion actuelle ?
2. Formatage (Markdown V2 escaping, emojis, charts embeddés ?).
3. Deep link vers dashboard web depuis chaque message.
4. Réactions & boutons inline (feedback « utile / pas utile ») pour collecter data → fine-tune scoring.
5. Groupes vs canaux : broadcast institutionnel, PV user FREE.
6. Fallback (Discord, email, SMS, webhook) si Telegram bannit / down.
7. Anti-spam : dedup par signal_id, cooldown utilisateur.
8. Commercial : canal privé payant vs bot PV — UX + conversion.

**Livrables** : spec UX des messages, plan multi-canal, note /10 sur rétention.

---

## Prompt 14 — Circuit Breaker (résilience LLM / Telegram)

**Périmètre** : `src/intelligence/circuit_breaker.py`.

**Objectif commercial** : l'app doit survivre à une panne Anthropic (rare mais arrive). Promesse de service = crédibilité.

**Mission** :
1. Vérifier les seuils (LLM threshold=3 / timeout=60s ; Telegram threshold=5 / timeout=120s) — dérivés d'incidents réels ou heuristiques ?
2. États Open / Half-Open / Closed bien transitionnés, tests d'injection de fautes.
3. Fallback LLM : template_narrative_engine.py activé automatiquement ? Qualité perçue ?
4. Exposé sur /health (déjà fait) + métriques Prometheus / dashboard Grafana.
5. Circuit par provider (Anthropic + OpenAI fallback futur ?).
6. Commercial : SLA 99.5 % réaliste ? Crédits clients en cas de downtime.

**Livrables** : test plan chaos engineering, proposition multi-provider, note /10.

---

## Prompt 15 — Security (validation, rate limiting, secrets)

**Périmètre** : `src/intelligence/security.py`, middleware, gestion env.

**Objectif commercial** : une fuite = game over (presse, churn, RGPD). PhD-level quality demandée par l'user.

**Mission** :
1. Audit OWASP API Top 10 ligne par ligne.
2. Secrets : sont-ils dans `.env` git-ignored ? Jamais loggés ? Rotation prévue ? Vault (Doppler, AWS SM) ?
3. Input validation : regex signal_id, sanitize_string, taille body 1 MB — exhaustif ?
4. Dépendances : scan Snyk / Dependabot, CVE actuels dans requirements.txt.
5. CORS : liste blanche, pas de `*` en prod.
6. Error leakage : déjà corrigé (« Internal server error ») — auditer les stack traces en logs exposés.
7. Pentest : prompt injection via narratives (un user upload du contenu qui se retrouve dans un prompt LLM → exfiltration).
8. Commercial : certification SOC 2 / ISO 27001 requise pour INSTITUTIONAL ?

**Livrables** : rapport OWASP scoré, plan de rotation secrets, note /10.

---

## Prompt 16 — Observability & Logging

**Périmètre** : logger config dans `src/intelligence/main.py`, usage `logging.getLogger` dans `src/intelligence/*`, `src/api/*`, `src/delivery/*`, route `/health` dans `src/api/routes/health.py`, futur `src/observability/` à créer, JSONFormatter via `LOG_FORMAT=json`.

**Objectif commercial** : sans observabilité, MTTR explose ×5-10 → SLA non-tenu, churn institutional, devops cost ×2. Pire qu'un crash visible : un **signal manqué silencieusement** érode la confiance sans qu'on le détecte. PhD-level demandé = stack OTLP complet (logs + metrics + traces corrélés).

**Équipe d'agents** :

| # | Agent | Rôle | Mode | Outputs |
|---|-------|------|------|---------|
| O1 | `Plan` — **Obs Lead** | Cartographie l'existant : où sont les loggers, quels modules logguent quoi, où est le `/health`, quelles métriques existent déjà. Définit la cible (logs + metrics + traces corrélés via `signal_id`/`trace_id`). | **Séquentiel** (en premier) | Inventaire as-is + spec to-be. |
| O2 | `Explore` — **Log Auditor** (very thorough) | Grep `print(`, `pprint(`, `console.log` ; tableau `module × niveau × fréquence` ; détecte spam (INFO > 1/s/symbol), stack-traces en INFO, secrets loggés, contexte manquant (`signal_id`, `tenant_id`). | **Parallèle** | Liste des call-sites à corriger + patches suggérés. |
| O3 | `general-purpose` — **Metrics Designer** | Catalogue Prometheus : counters (`signals_emitted_total{symbol,tf,tier}`), gauges (`circuit_breaker_state{provider}`), histograms (`llm_latency_seconds`, `confluence_score_distribution`, `cache_hit_ratio`). Cardinalité estimée par label. | **Parallèle** | Catalogue + snippets `prometheus_client` PR-ready. |
| O4 | `general-purpose` — **Tracing Architect** | Design des spans OpenTelemetry pour les 7 étages (DataProvider → SmartMoney → Confluence → Vol → LLM → Cache → Store → Telegram). Propagation `trace_id` via context vars. Exporter OTLP. | **Parallèle** | Diagramme des spans + exemple JSON trace pour un signal réel. |
| O5 | `general-purpose` — **Alerting Engineer** | Règles Alertmanager / Grafana OnCall avec seuils dérivés du replay 7 ans (P95/P99 réels, pas inventés). Runbook par alerte. Critères : circuit open > 5 min, signal-rate > 3σ, P99 LLM > 8 s, /health degraded > 2 min. | **Parallèle** (après O3) | YAML règles + runbooks. |
| O6 | `general-purpose` — **Cost Analyst** (WebFetch) | Pricing actualisé 2026 : Grafana Cloud Free, Datadog, Better Stack, Loki self-hosted, New Relic. Courbe coût × volume signaux. Cible < $50/mois jusqu'à 10k signaux/jour. | **Parallèle** | Tableau coût × tooling × scénario volume. |
| O7 | `general-purpose` — **Status Page Designer** | Comparatif statuspage.io / instatus / atlassian, intégration webhook depuis circuit breaker, composants public-facing (API, Scanner, LLM, Telegram). | **Parallèle** | Reco outil + maquette page. |
| O8 | `Plan` — **Red-Team** | Challenge : « les seuils d'alerte sont-ils dérivés des données ou inventés ? Datadog vaut-il 10× le prix de Grafana Cloud pour un solo founder ? OpenTelemetry n'est-il pas overkill avant 100 MAU ? » | **Séquentiel** (avant synthèse) | Liste objections + recommandations pragmatiques. |
| O9 | `general-purpose` — **Synthesis Lead** | Plan phasé J0/J+30/J+90, JSON provisioning Grafana du dashboard Business, patch `/metrics` PR-ready, note /10. | **Séquentiel** (final) | `reports/eval_16_observability.md`. |

**Protocole d'orchestration** :
1. O1 cartographie l'as-is.
2. **O2, O3, O4, O6, O7 en parallèle** (5 tool calls dans un seul message). O5 attend O3.
3. O8 challenge (couper l'overkill phase startup), O9 synthétise.

**Livrables** : inventaire as-is, catalogue logs+metrics+traces, règles d'alerte avec runbooks, comparatif tooling × coût, plan phasé J0/J+30/J+90, patch `/metrics` PR-ready, JSON Grafana, note /10, KPIs (MTTR, % signaux traçables, coverage métriques modules critiques, fausses alertes/sem).

---

## Prompt 17 — Testing Suite (1366 tests, couverture, flaky, mutation)

**Périmètre** : `tests/`, `src/**/test_*.py`, `pytest.ini` / `pyproject.toml`, `conftest.py`, GitHub Actions / CI config, fixtures partagées, snapshots LLM.

**Objectif commercial** : un faux signal poussé en prod = ticket support + perte de confiance. Tests = la **première ligne d'assurance qualité revenu**. PhD-level demandé = mutation testing + property-based, pas de coverage cosmétique.

**Équipe d'agents** :

| # | Agent | Rôle | Mode | Outputs |
|---|-------|------|------|---------|
| T1 | `Plan` — **QA Lead** | Définit la matrice de criticité par module (revenue-impact × user-facing × bug-history) pour pondérer le coverage. Liste les modules à protéger en priorité. | **Séquentiel** (en premier) | Matrice criticité + objectifs coverage par bucket. |
| T2 | `Explore` — **Coverage Auditor** | Exécute `pytest --cov=src --cov-branch --cov-report=html`, parse l'output, classe les modules en 4 buckets (>90 %, 70-90 %, 50-70 %, <50 %). Pondère par criticité de T1. | **Parallèle** (après T1) | Classement 4-buckets + top 10 prioritaire. |
| T3 | `general-purpose` — **Flaky Hunter** | Reproduit `test_short_roundtrip_pnl` : 100× isolé puis 100× dans la suite, capture seed/ordering/state partagé fautif, propose fix (isolation fixture, seed fix, désactivation `pytest-randomly` ciblée). | **Parallèle** | Diagnostic root-cause + patch PR-ready. |
| T4 | `Explore` — **Broken-Import Investigator** | `test_long_short_trading.py` cassé : déterminer si le code testé existe encore ; si oui fixer l'import ; si non supprimer + documenter dans `MEMORY.md`. | **Parallèle** | Verdict fix vs delete + commit. |
| T5 | `general-purpose` — **Mutation Tester** | Configure `mutmut` sur 5 modules critiques (`confluence_detector`, `signal_state_machine`, `volatility_forecaster`, `auth`, `tier_manager`). Lance, classe les mutants survivants par criticité (logique métier vs edge case acceptable). | **Parallèle** (après T2) | Mutation score par module + liste mutants survivants à killer. |
| T6 | `general-purpose` — **Property-Based Designer** | Écrit specs Hypothesis : `ConfluenceDetector` (score ∈ [0,100], monotonie sur composants), `SignalStateMachine` (pas de transition BUY→SELL sans HOLD si lockout), `resample_ohlcv` (H ≥ max(O,C), L ≤ min, sum_volume conservé). Stratégies `@given` réalistes (prix > 0, OHLC bounds, timestamps croissants). | **Parallèle** | 6 fichiers tests Hypothesis PR-ready. |
| T7 | `general-purpose` — **E2E Architect** | Scénario `docker-compose.test.yml` : data-provider mock + sentinel scanner + telegram mock. POST signal synthétique → assert format Telegram. Cible CI < 3 min. | **Parallèle** | `docker-compose.test.yml` + script smoke. |
| T8 | `general-purpose` — **CI Optimizer** | Parallélisation `pytest-xdist`, cache pip + pytest, matrix Python, suppression imports lourds en collection. Mesure avant/après. Snapshot testing LLM (pytest-recording) pour zéro coût API en CI. | **Parallèle** | `.github/workflows/test.yml` optimisé + delta secondes. |
| T9 | `general-purpose` — **Test Data Curator** | 1 dataset XAU 7-jours golden + 1 EURUSD 7-jours, versionnés via Git LFS ou DVC. Hash + checksum pour reproductibilité. | **Parallèle** | Datasets + fixture loader. |
| T10 | `Plan` — **Red-Team** | Challenge : « le mutation score 70 % vaut-il le coût CI ? Hypothesis sur SignalStateMachine ne va-t-il pas exploser le runtime ? Le E2E docker-compose stable en CI Windows ? » | **Séquentiel** (avant synthèse) | Liste fragilités + ajustements obligatoires. |
| T11 | `general-purpose` — **Synthesis Lead** | Plan PR séquencé (5 PRs : flaky fix → broken-import → property-based → mutation gates → E2E + CI parallel). Note /10. Badge coverage publiable. | **Séquentiel** (final) | `reports/eval_17_testing.md`. |

**Protocole d'orchestration** :
1. T1 définit la matrice criticité.
2. **T2, T3, T4, T6, T7, T8, T9 en parallèle** (7 tool calls dans un seul message). T5 attend T2.
3. T10 challenge, T11 synthétise.

**Livrables** : matrice criticité × coverage, top 10 modules prioritaires, fix flaky + broken-import, mutation score 5 modules critiques, 6 specs Hypothesis PR-ready, `docker-compose.test.yml`, CI YAML optimisé, plan PR séquencé, note /10, KPIs (suite < 5 min, coverage critiques ≥ 90 %, mutation score ≥ 70 %, 0 flaky).

---

## Prompt 18 — Backtest & Replay Harness (crédibilité marketing)

**Périmètre** : `src/backtest/state_machine_replay.py`, `src/backtest/news_replay.py`, `scripts/audit_backtest.py`, `scripts/run_backtest.py`, `replay_*.json` à la racine, `tests/test_state_machine_replay.py`, `tests/test_news_replay.py`, `replay_harness.md`, `audit_backtest_2026_04_24.md`.

**Objectif commercial** : **les chiffres affichés sur la landing = la conversion**. Un PF gonflé par look-ahead bias = fraude involontaire = lawsuit potentielle (FTC US, AMF FR). Verdict actuel (PF max 0.96, sweep 7-ans config prod = 0 trade) doit être stress-testé avant tout discours marketing. PhD-level = walk-forward strict + IC bootstrap + SPA test.

**Équipe d'agents** :

| # | Agent | Rôle | Mode | Outputs |
|---|-------|------|------|---------|
| K1 | `Plan` — **Backtest Lead** | Cartographie l'existant : 19 fichiers `replay_*.json/csv` à la racine, scripts d'audit, harnais. Identifie ce qui est in-sample vs déjà splitté, quelles métriques sont publiées vs internes. Définit la « legal-safe stats sheet » (chiffres communicables). | **Séquentiel** (en premier) | Inventaire artefacts + matrice « publiable / interne ». |
| K2 | `Explore` — **Look-Ahead Auditor** (very thorough) | Grep tous les `.shift(-1)`, `.rolling(...).mean()` non-causal, `.fillna(method='bfill')`, `.iloc[i+...]`. Pour chaque suspect, prouver causal/non-causal via test différentiel (`data.iloc[:i+1]` strict vs full-data → diff doit être 0). | **Parallèle** | Liste sites suspects + verdict par site + patches. |
| K3 | `general-purpose` — **Walk-Forward Designer** | Produit `scripts/walkforward.py` propre : split 2019-2022 train / 2023 val / 2024-2025 test. Tuning uniquement train+val. Reporter test OOS séparément. Pas de leakage entre folds (purge + embargo). | **Parallèle** | Script walk-forward + tableau train/val/test avec gap visible. |
| K4 | `general-purpose` — **Cost Model Validator** | Calibre slippage par session (London/NY = 1-2 pips XAU, Asia = 3-5 pips), spread variable, commission broker IC Markets / Pepperstone réel. Fonction `cost(symbol, session, size, side)`. Mesure delta PF avant/après calibration réaliste. | **Parallèle** | Module `costs.py` + impact PF chiffré. |
| K5 | `Explore` — **Sizing Alignment Auditor** | Vérifie que le replay utilise la même formule de sizing que la prod prévue (% equity ATR-target vs fixed lot fictif). Croise avec Prompt 19. | **Parallèle** | Verdict aligné/désaligné + fix. |
| K6 | `general-purpose` — **Monte Carlo Simulator** | Bootstrap 10 000 permutations des trades → IC 95 % pour PF, Sharpe, max DD, Calmar. P-value vs random walk. Si IC PF inclut 1.0 → strat pas significativement différente du hasard. | **Parallèle** | IC bootstrap + p-values + verdict significativité. |
| K7 | `general-purpose` — **SPA / Reality Check** | White's Reality Check ou Hansen's SPA test sur les sweeps de paramètres (audit_backtest.py multi-config). Corrige le multiple-testing bias qui gonfle le « best PF observé ». | **Parallèle** | P-value SPA + meilleure config corrigée. |
| K8 | `general-purpose` — **Regime Decomposer** | Segmente le replay par régime HMM (trending up / down / range). Table PF × régime × année. Identifie le « cheval de bataille » du système (souvent profitable seulement en trending bull). | **Parallèle** | Heatmap régime × année + verdict conditionnel. |
| K9 | `general-purpose` — **OOS Live Tracker Designer** | Design d'un job nightly (cron) qui paper-trade en live et compare aux stats publiées sur la landing. Alerte si déviation > 1σ → marketing à corriger. | **Parallèle** | Spec job + schéma DB pour live-tracking + alerte. |
| K10 | `Plan` — **Marketing Risk Reviewer / Red-Team** | Audite chaque chiffre actuellement publié (BUSINESS_PLAN, README, futur landing) : sourcé par script reproductible avec checksum data ? Sinon → enterrer. « PF 0.96 sweep 7-ans » se vend-il ? Plan B narratif. | **Séquentiel** (avant synthèse) | Liste chiffres publiables vs à enterrer + reformulation legale. |
| K11 | `general-purpose` — **Synthesis Lead** | Verdict global « commercialisable / borderline / pas commercialisable », rapport `reports/eval_18_backtest.md`, `BACKTEST_LEGAL_GUARDRAILS.md` (chiffres autorisés), note /10 crédibilité. | **Séquentiel** (final) | Rapport + guardrails. |

**Protocole d'orchestration** :
1. K1 cartographie + matrice publiable.
2. **K2, K3, K4, K5, K6, K7, K8, K9 en parallèle** (8 tool calls dans un seul message).
3. K10 challenge marketing, K11 synthétise.

**Livrables** : checklist biais (look-ahead, survivorship, multiple-testing) avec ✅/❌, IC bootstrap PF/Sharpe/DD, p-values SPA, table régime × année, `scripts/walkforward.py` PR-ready, module `costs.py` calibré, spec OOS live tracker, `BACKTEST_LEGAL_GUARDRAILS.md`, note /10, KPIs (PF OOS ≥ 1.5 confiance / ≥ 1.2 borderline, Sharpe OOS ≥ 0.8, max DD < 25 %, IC PF strictement > 1.0).

---

## Prompt 19 — Risk Management (sizing, SL/TP, drawdown, kill-switch)

**Périmètre** : `src/risk/` (si existant) + paramètres risk dans `config.py`, `src/environment/strategy_features.py`, `src/agents/news_analysis_agent.py`, embeds Discord (« position sizing »), tout call-site qui calcule `size`/`SL`/`TP`, `src/delivery/telegram_notifier.py` (disclaimer).

**Objectif commercial** : **un seul abonné qui blow up à cause d'un signal = lawsuit + 1000 reviews 1-étoile**. À l'inverse, un **Risk Score** lisible = feature premium tier ANALYST+ (différenciation vs LuxAlgo). PhD-level = Kelly fractionnel + vol-targeting + stress-tests cross-asset.

**Équipe d'agents** :

| # | Agent | Rôle | Mode | Outputs |
|---|-------|------|------|---------|
| R1 | `Plan` — **Risk Lead** | Cartographie tous les chemins de calcul de `size`/`SL`/`TP` dans le repo (replay, prod, Discord embed). Identifie incohérences entre les 3 surfaces. Définit la cible (formule unique source-of-truth). | **Séquentiel** (en premier) | Diagramme de flux risk + matrice incohérences. |
| R2 | `Explore` — **Sizing Logic Auditor** | Trace chaque appel sizing : fixed lot ? % equity ? Kelly ? ATR-vol-target ? Documente la règle actuelle vs la règle live souhaitée. Croise avec Prompt 18 pour aligner replay/live. | **Parallèle** | Documentation sizing par surface + écarts. |
| R3 | `general-purpose` — **SL/TP Strategy Reviewer** | Sur le replay : 3 stratégies SL/TP comparées (ATR-2x, structure-based low du OB, R:R fixe 2:1). Tableau expectancy par stratégie × symbole. Investigue l'asymétrie long/short (shorts profitables / longs non) — biais SL/TP ? | **Parallèle** | Tableau expectancy + recommandation par symbole. |
| R4 | `general-purpose` — **Drawdown Analyst** | Calcule sur replay 7 ans : max DD intra et close-to-close, ulcer index, time-to-recovery, worst-month, worst-week, worst-day. Distribution conditionnelle au régime HMM. Chart equity curve + DD overlay. | **Parallèle** | Rapport DD + chart PNG. |
| R5 | `general-purpose` — **Cross-Signal Correlation Auditor** | Matrice ρ rolling 30j entre paires de presets sur 7 ans (XAU vs DXY, EURUSD vs GBPUSD, BTC vs US500). Identifie clusters (USD-bullish basket, commodity basket). Propose règle portfolio-level cap (somme |β| < seuil). Croise avec Prompt 20. | **Parallèle** | Matrice ρ + spec règle portfolio cap. |
| R6 | `general-purpose` — **Kelly Calculator** | Calcule p (win rate) et b (avg_win/avg_loss) par bucket de score (50-60, 60-70, 70-80, 80+). Kelly full f* = (p·b − q)/b. Kelly fractionnel = f*/4 (Thorp). Sensibilité de f à l'estimation de p (IC bootstrap). | **Parallèle** | Tableau Kelly par bucket + sensibilité. |
| R7 | `general-purpose` — **Vol-Targeting Designer** | Spec `size = target_vol / forecast_vol(t) × equity` ; lien direct avec `volatility_forecaster` (synergy Prompt 04). Quand `forecast_vol` saute → réduction auto exposure. | **Parallèle** | Module `vol_target.py` skeleton + tests. |
| R8 | `general-purpose` — **Kill-Switch Designer** | 4 règles de pause auto : N SL consécutifs, DD journalier > X %, vol IV > 3σ (black-swan), broker disconnect. Pseudo-code + sites d'insertion (`sentinel_scanner`, `telegram_notifier`). | **Parallèle** | `src/risk/kill_switch.py` PR-ready + tests. |
| R9 | `general-purpose` — **Stress Tester** | Simulation 2008 / 2020 mars (COVID gap) / 2023 août — comportement du sizing sous gap up/down 3σ. Le kill-switch déclenche-t-il à temps ? | **Parallèle** (après R8) | Rapport stress 3 scénarios. |
| R10 | `general-purpose` — **Risk Score Productizer** | Design formule Risk Score 0-100 user-facing : composantes confluence × vol_forecast × news_proximity × regime_alignment. Mockup affichage Telegram + dashboard. | **Parallèle** | Spec Risk Score + mockup dans `mockups/`. |
| R11 | `general-purpose` — **Legal Disclaimer Reviewer** | Vérifie présence « Not financial advice » sur Telegram + landing + email + API responses. Template multi-langue (FR/EN/ES). Croise avec Prompt 29 (juridictions). | **Parallèle** | Template disclaimer + audit présence. |
| R12 | `Plan` — **Red-Team** | Challenge : « Kelly est-il dangereux pour un retail FREE ? Le portfolio cap réduit-il trop les signaux émis (impact MRR) ? Le kill-switch user-overridable créé-t-il une lawsuit si l'user override et perd ? » | **Séquentiel** (avant synthèse) | Liste objections + ajustements obligatoires. |
| R13 | `general-purpose` — **Synthesis Lead** | Plan implémentation phasé (P1 kill-switch + disclaimer → P2 Kelly fractionnel + vol-target → P3 Risk Score commercialisable), note /10. | **Séquentiel** (final) | `reports/eval_19_risk.md` + PR plan. |

**Protocole d'orchestration** :
1. R1 cartographie + matrice incohérences.
2. **R2, R3, R4, R5, R6, R7, R8, R10, R11 en parallèle** (9 tool calls dans un seul message). R9 attend R8.
3. R12 challenge, R13 synthétise.

**Livrables** : diagramme flux risk, tableau Kelly par bucket, rapport drawdown + chart PNG, matrice corrélation cross-signaux, `src/risk/kill_switch.py` PR-ready, module `vol_target.py`, spec Risk Score 0-100 commercialisable, mockup Telegram, template disclaimer multi-langue, plan PR phasé P1/P2/P3, note /10, KPIs (max DD < 20 %, time-to-recovery < 30 j, kill-switch testé sur 3 stress scénarios, Risk Score affiché tier ANALYST+).

---

## Prompt 20 — Multi-Asset & Multi-Timeframe (TAM expansion)

**Périmètre** : 6 presets dans `config.py` (XAUUSD, EURUSD, BTCUSD, US500, GBPUSD, USDJPY), `resample_ohlcv()` dans `src/intelligence/data_providers.py`, `InstrumentConfig` (price_decimals, pip_value, session_hours), candidats nouveaux assets (USOIL, NAS100, AUDJPY, ETHUSDT), `tests/test_pipeline_integration.py`.

**Objectif commercial** : **plus d'assets = plus de TAM, mais un asset à PF < 1 nuit plus qu'il n'aide** (dilution + support cost). Multi-asset = bundling possible (« FX Pack », « Crypto Pack ») = pricing flexibility. PhD-level = walk-forward par asset, pas one-size-fits-all.

**Équipe d'agents** :

| # | Agent | Rôle | Mode | Outputs |
|---|-------|------|------|---------|
| M1 | `Plan` — **Multi-Asset Lead** | Cartographie l'existant : 6 presets, leurs configs, données disponibles par symbol × TF, tests passants/échouants par préset. Définit la matrice cible (30 cellules : 6 symbols × 5 TF M5/M15/H1/H4/D1). | **Séquentiel** (en premier) | Inventaire + matrice cible. |
| M2 | `general-purpose` — **Backtest Sweep Engineer** | Exécute `scripts/run_backtest.py` sur 6 symbols × 5 TF (avec walk-forward strict cf. Prompt 18). Agrège : PF, Sharpe, max DD, nb signaux, win rate par cellule. | **Parallèle** (après M1) | `reports/sweep_30cells.csv` + heatmap PF + heatmap Sharpe. |
| M3 | `Explore` — **Instrument Config Auditor** | Audite `InstrumentConfig` pour chaque préset. Vérifie : `price_decimals` (Gold=2, FX=5, JPY=3, Index=1, Crypto=2 — déjà OK), pip_value, session_hours, weekend_behavior, news_relevance_weight, ATR baseline. Identifie params manquants ou copiés-collés à corriger. | **Parallèle** | Tableau `param × préset × verdict` + patches. |
| M4 | `general-purpose` — **Session & Calendar Auditor** | Calendrier UTC des sessions actives par symbol : XAU pic London/NY, EURUSD London+NY, BTC 24/7, US500 RTH 14:30-21:00 UTC. Vérifie que le scanner ne fire pas hors-session. Handle des gaps weekend (Sunday open Forex). Couverture `EconomicCalendarFetcher` par juridiction (NFP USD, ECB EUR, BoJ JPY, BoE GBP). | **Parallèle** | Calendrier sessions + audit news coverage. |
| M5 | `general-purpose` — **Correlation Analyst** | Matrice ρ 30j rolling sur 7 ans entre 6 presets. Identifie clusters (USD-bullish basket : EURUSD/GBPUSD/AUDUSD short ; commodity basket : XAU/XAG/USOIL ; risk-on : US500/NAS100/BTC). Formule règle de portfolio cap. Synergy avec Prompt 19/R5. | **Parallèle** | Matrice ρ + clusters + spec règle portfolio. |
| M6 | `general-purpose` — **Asymmetry Investigator** | Reproduit le test « long PF vs short PF » par symbol × année. XAU est-il unique (shorts profitables / longs non) ou pattern systémique ? Si systémique → bug pipeline ; si XAU-only → biais structurel asset. | **Parallèle** | Tableau asymétrie + verdict bug vs biais structurel. |
| M7 | `general-purpose` — **New Asset Scout** | Pour 4 candidats (USOIL, NAS100, AUDJPY, ETHUSDT) : data feed disponible (Dukascopy ? Polygon ? Tiingo ?), coût mensuel, smart-money relevance, PF estimé sur 1 an pilote, effort onboarding (config + tests + tuning). Croise avec Prompt 08 (data) et Prompt 24 (coûts). | **Parallèle** | 4 fiches new-asset comparables. |
| M8 | `general-purpose` — **Multi-TF Confluence Architect** | Design score-boost confluent multi-TF : signal H4 confirmé par M15 retest = high-conviction. Implémentation : un score TF parent injecté dans le scoring TF enfant. Simulation sur replay : combien de signaux high-conviction émergent ? Impact volume signaux/jour. | **Parallèle** | Spec multi-TF boost + impact chiffré. |
| M9 | `general-purpose` — **TF Tier Strategist** | Quel TF par tier ? M1/M5 = scalping computationally cher → tier STRATEGIST+. D1/W1 = swing low-volume → inclus FREE en teaser. Justifier par usage CPU + valeur perçue. | **Parallèle** | Mapping TF × tier + justification. |
| M10 | `general-purpose` — **Bundling Strategist** | Design 4 packs commerciaux : FX Pack (4 majors), Metal Pack (XAU + XAG + USOIL), Crypto Pack (BTC + ETH + SOL), Index Pack (US500 + NAS100 + DAX). Comparatif vs concurrence (LuxAlgo bundles). Pricing recommandé par pack. Cannibalisation tier all-access. | **Parallèle** | 4 packs spec + pricing + risque cannibalisation. |
| M11 | `Plan` — **Red-Team** | Challenge : « les chiffres du sweep 30-cells sont-ils OOS ou in-sample (gonflé) ? Onboarder USOIL avant que XAU soit profitable est-il prématuré ? Bundles Metal/Index ont-ils assez d'assets pour justifier prix vs all-access ? » | **Séquentiel** (avant synthèse) | Liste fragilités + recommandation pragmatique. |
| M12 | `general-purpose` — **Synthesis Lead** | Recommandation « keep / drop / add » par asset, roadmap 90j (M1 : drop toxiques + tune EURUSD ; M2 : ajouter top candidat ; M3 : bundle launch), `config_proposed.py` PR-ready, mockup pricing page, note /10. | **Séquentiel** (final) | `reports/eval_20_multi_asset.md`. |

**Protocole d'orchestration** :
1. M1 cartographie.
2. **M2, M3, M4, M5, M6, M7, M8, M9, M10 en parallèle** (9 tool calls dans un seul message).
3. M11 challenge marketing/maturité, M12 synthétise.

**Livrables** : `reports/sweep_30cells.csv` + heatmap PF/Sharpe (6×5), audit `InstrumentConfig` par préset, calendrier sessions UTC, matrice corrélation + clusters, verdict asymétrie long/short, 4 fiches new-asset, spec multi-TF boost, mapping TF × tier, 4 packs commerciaux avec pricing, `config_proposed.py` PR-ready, mockup pricing page bundles dans `mockups/`, plan 90j, note /10, KPIs (≥ 4 symbols PF OOS > 1.2, 0 doublon ρ > 0.7, 2 nouveaux assets PF > 1.0 onboardés, FX Pack lancé, +30 % MRR projeté).

---

## Prompt 21 — Performance & Scalabilité

**Périmètre** : profiling global, Docker, concurrency, état partagé, charge.

**Objectif commercial** : capacité à absorber une vague d'abonnés (post-Product Hunt, viral tweet) sans faire tomber le service = capter la croissance.

**Équipe d'agents** (orchestrée par le main agent ; chaque agent reçoit un brief autonome avec chemins de fichiers, périmètre exact, format de sortie attendu) :

| # | Agent (subagent_type) | Rôle | Mode | Inputs | Outputs |
|---|-----------------------|------|------|--------|---------|
| A1 | `Plan` — **Perf Lead** | Cartographie le chemin critique (tick → signal → Telegram), identifie les modules à profiler, distribue les briefs et fixe la définition de « hot path ». | **Séquentiel** (en premier) | `src/intelligence/sentinel_scanner.py`, `main.py`, `config.py` | Diagramme du chemin critique + liste des sous-systèmes à auditer en parallèle. |
| A2 | `Explore` — **Profiler** (very thorough) | Exécute py-spy + scalene sur 1 h de prod simulée (replay 15 min × 4) ; identifie les hot paths CPU, allocation mémoire, GC pressure. | **Parallèle** | Replay harness + scanner | Top 20 frames par CPU time, flamegraph, RSS peak. |
| A3 | `Explore` — **Concurrency Auditor** | Audite chaque `await` du chemin critique ; détecte appels `requests`, `time.sleep`, `pd.read_csv` synchrones bloquant l'event loop ; mesure l'impact GIL sur les workers. | **Parallèle** | `src/intelligence/`, `src/api/` | Liste des sites bloquants avec patch suggéré (asyncio / `to_thread`). |
| A4 | `Explore` — **State & Storage Auditor** | Identifie tout l'état mutable partagé (signal_state_machine, semantic_cache, signal_store SQLite) ; mesure SQLite WAL contention sous 100 RPS ; propose découpe Redis vs Postgres. | **Parallèle** | `src/intelligence/semantic_cache.py`, `src/api/signal_store.py`, `state_persistence.md` | Inventaire état + plan de désétatisation des workers. |
| A5 | `general-purpose` — **Load Tester** | Écrit un scénario locust/k6 (1, 10, 100, 1 000 RPS) ciblant `/v1/signals`, `/v1/narratives`, scanner tick. Mesure P50/P95/P99, error rate, saturation point. | **Parallèle** (après A1) | API routes | Courbe de charge CSV + verdict knee point. |
| A6 | `general-purpose` — **Cost Modeler** | Calcule $/MAU sur 3 scénarios (1k / 10k / 100k users) — combine outputs A2/A4/A5 + tarifs Railway/Fly. Cible < $1/MAU. | **Séquentiel** (après A2/A4/A5) | Outputs A2/A4/A5 | Tableau coût × scénario + sensitivity. |
| A7 | `Plan` — **Red-Team / Cross-Check** | Challenge chaque finding : « le knee point à X RPS est-il réel ou un artefact du test ? Le hot path est-il sur le chemin commercial ? Le coût LLM est-il dans A6 ou ignoré ? » | **Séquentiel** (avant synthèse) | Tous les outputs A2-A6 | Liste des biais détectés + corrections obligatoires. |
| A8 | `general-purpose` — **Synthesis Lead** | Agrège, applique les corrections du Red-Team, produit le rapport final avec note /10 et matrice effort × impact. | **Séquentiel** (final) | Tous outputs + corrections A7 | `reports/eval_21_performance.md`. |

**Protocole d'orchestration** :
1. A1 produit le périmètre.
2. **Lancer A2, A3, A4, A5 dans un seul message multi-tool-call** (parallélisme réel).
3. A6 dépend de A2/A4/A5 → attendre puis lancer.
4. A7 challenge tout. A8 synthétise.

**Livrables** : flamegraph py-spy, liste sites bloquants async, plan de désétatisation Redis, courbe de charge avec knee point, tableau coût × scénario, note /10, top 5 refactors priorisés (effort × impact), KPIs post-amélioration (P99 latence scanner, $/MAU, RPS soutenu).

---

## Prompt 22 — Deployment & Infrastructure

**Périmètre** : `infrastructure/Dockerfile`, `Procfile`, `railway.toml`, CI/CD, secrets, régions, IaC.

**Objectif commercial** : déploiement sans downtime = confiance client. Coût infra = marge brute. Vendor lock = risque stratégique.

**Équipe d'agents** :

| # | Agent | Rôle | Mode | Outputs |
|---|-------|------|------|---------|
| B1 | `Plan` — **Deploy Lead** | Inventorie l'existant (Railway config actuel, vars d'env, secrets, fichiers Docker). Définit la matrice « what must not change » (downtime budget, secrets confidentialité). | **Séquentiel** (en premier) | Carte de l'existant + contraintes dures. |
| B2 | `Explore` — **Docker Auditor** | Scan le Dockerfile : multi-stage présent ? layer caching ? image size mesurée (`docker history`) ? user non-root ? CVE base image (`trivy`/`grype`) ? Healthcheck ? | **Parallèle** | Rapport Dockerfile scoré + image diff proposée. |
| B3 | `general-purpose` — **Provider Comparator** | Tableau Railway / Fly.io / Render / AWS ECS Fargate sur : coût mensuel ($) pour 2 vCPU + 4 Gi RAM + 1 worker, latence cold-start, facilité IaC, support régions, lock-in. Source : pricing pages 2026 (WebFetch). | **Parallèle** | Comparatif 4 providers × 6 critères. |
| B4 | `general-purpose` — **CI/CD Engineer** | Conçoit pipeline (GitHub Actions) : lint → unit tests → smoke tests Docker → deploy → post-deploy smoke → rollback 1-clic. Inclut staging/prod séparation. | **Parallèle** | YAML CI/CD prêt à coller + diagramme flux. |
| B5 | `Explore` — **Secrets & Compliance Auditor** | Vérifie : secrets jamais loggés, rotation possible, vault (Doppler/AWS SM/Railway Vars), `.env` git-ignored, audit log. Croise avec OWASP A07. | **Parallèle** | Plan rotation + risques actuels. |
| B6 | `Plan` — **IaC Specialist** | Convertit la config Railway actuelle en Terraform (ou Pulumi). Évalue effort migration. Documente le « lift » si on quitte Railway demain. | **Parallèle** (après B3) | Module Terraform skeleton + estimation jours. |
| B7 | `general-purpose` — **Region & Latency Analyst** | Mesure latence utilisateur depuis EU/US/APAC vers chaque provider candidat. Calcule l'impact sur le SLA « signal < 30 s ». | **Parallèle** | Heatmap latence × région × provider. |
| B8 | `Plan` — **Red-Team** | Challenge : « le rollback 1-clic est-il vraiment testé ? La rotation secrets coûte-t-elle un downtime ? Le provider le moins cher a-t-il les SLA requis ? » | **Séquentiel** (avant synthèse) | Liste objections + verdicts. |
| B9 | `general-purpose` — **Synthesis Lead** | Produit `reports/eval_22_deployment.md` avec recommandation provider, plan de migration phasé (J0 / J+30 / J+90), note /10. | **Séquentiel** (final) | Rapport final. |

**Protocole d'orchestration** :
1. B1 cartographie.
2. **B2, B3, B4, B5, B7 en parallèle** (5 tool calls dans un seul message).
3. B6 attend B3 (besoin du provider gagnant).
4. B8 challenge, B9 synthétise.

**Livrables** : comparatif providers (4 × 6 critères), Dockerfile diff, YAML CI/CD prêt, module Terraform, plan rotation secrets, heatmap latence, plan de migration phasé J0/J+30/J+90, note /10, KPIs (deploy frequency, MTTR, image size, $/mois).

---

## Prompt 23 — Model Training Pipeline (Colab POCs → Prod)

**Périmètre** : `scripts/colab_*_poc.py`, `src/training/`, `src/intelligence/volatility_lgbm.py`, workflow export → prod, validation, retraining.

**Objectif commercial** : capacité à **itérer vite sur les modèles** = avantage vs concurrence statique. « Modèle mis à jour chaque mois » = argument rétention/upsell.

**Équipe d'agents** :

| # | Agent | Rôle | Mode | Outputs |
|---|-------|------|------|---------|
| C1 | `Plan` — **MLOps Lead** | Cartographie le cycle de vie modèle actuel : où s'entraînent les modèles (Colab), où ils sont stockés, comment ils arrivent en prod, qui les valide. Produit le diagramme « as-is ». | **Séquentiel** (en premier) | Diagramme as-is + liste des fragilités évidentes. |
| C2 | `Explore` — **Reproducibility Auditor** | Vérifie sur chaque `colab_*_poc.py` : seeds explicites (numpy/torch/lgbm/python), `requirements.txt` pinné, hash data snapshot, hyperparams loggés. Note reproductibilité /10 par script. | **Parallèle** | Tableau scripts × critères + commits suggérés. |
| C3 | `Explore` — **Train/Serve Skew Detective** | Compare features calculées en Colab (training) vs `src/intelligence/volatility_forecaster.py` (serving). Cherche : différences de fenêtres, time zone bugs, divergence de calcul ATR/RV, normalisation absente côté serving. C'est *le* tueur silencieux. | **Parallèle** | Liste écarts + impact estimé sur RMSE. |
| C4 | `general-purpose` — **Pipeline Architect** | Conçoit le pipeline cible : Colab → MLflow registry (ou W&B) → validation gate (tests régression RMSE) → shadow deploy → canary → prod. Outils retenus + coût mensuel. | **Parallèle** (après C1) | Diagramme « to-be » + stack outils. |
| C5 | `general-purpose` — **Validation Gate Designer** | Spec les tests de régression modèle : seuils RMSE/MAE/Diebold-Mariano, walk-forward 2024-2025, refus auto si dégradation > X %. Écrit le squelette pytest. | **Parallèle** | `tests/test_model_regression.py` + seuils justifiés. |
| C6 | `Explore` — **Feature Store Specialist** | Évalue Feast vs Tecton vs hand-rolled. Identifie les 5-10 features critiques à mettre dans le store en priorité (HAR-RV components, ATR_14, regime label). | **Parallèle** | Recommandation outil + liste features prioritaires. |
| C7 | `general-purpose` — **Drift Monitor Designer** | Spec le détecteur de drift : KS test sur feature distributions, PSI sur predictions, alerte → trigger retrain. Cadence : mensuel forcé OU drift-triggered. | **Parallèle** | Pseudo-code détecteur + critères trigger. |
| C8 | `Explore` — **Dataset Versioning Specialist** | Évalue DVC vs LakeFS vs Git LFS pour les CSV XAU 6 ans (~50 Mo). Ratio coût/bénéfice solo-founder. | **Parallèle** | Choix justifié + commande `dvc init` ou alternative. |
| C9 | `Plan` — **Red-Team** | Challenge : « shadow mode est-il faisable sans 2× le coût LLM ? Le retrain mensuel est-il rentabilisé ? Feature store n'est-il pas overkill pour un solo founder ? » | **Séquentiel** | Liste critiques + recommandation pragmatique (ce qu'on garde / coupe). |
| C10 | `general-purpose` — **Synthesis Lead** | Plan MLOps 2026 phasé (J0 reproductibilité → J+30 validation gate → J+90 shadow + drift), checklist train/serve skew, note /10. | **Séquentiel** (final) | `reports/eval_23_mlops.md`. |

**Protocole d'orchestration** :
1. C1 cartographie l'as-is.
2. **C2, C3, C5, C6, C7, C8 en parallèle** (6 tool calls dans un seul message). C4 attend C1.
3. C9 challenge (couper ce qui est overkill solo). C10 synthétise.

**Livrables** : diagrammes as-is/to-be, tableau reproductibilité par script, liste écarts train/serve avec impact RMSE, `tests/test_model_regression.py` skeleton, choix outils (registry, feature store, dataset versioning) avec coût mensuel, plan MLOps phasé J0/J+30/J+90, note /10, KPIs (time-to-deploy d'un modèle, % retrain réussis, drift detection latency).

---

## Prompt 24 — Cost Structure & Unit Economics

**Périmètre** : factures LLM (Anthropic), infra (Railway/etc.), data feeds, coûts fixes (Stripe, légal, support), pricing tiers gelés.

**Objectif commercial** : **sans marge brute > 70 %, pas de SaaS viable**. Le coût marginal LLM est la variable n°1 ; un abonné FREE peut coûter plus qu'il ne rapporte si non maîtrisé.

**Équipe d'agents** :

| # | Agent | Rôle | Mode | Outputs |
|---|-------|------|------|---------|
| D1 | `Plan` — **Unit Econ Lead** | Définit le modèle de coût : variables (par signal, par MAU) vs fixes (mensuels). Liste les hypothèses à challenger (nb signaux/jour/tier, hit rate cache, modèle utilisé par tier). | **Séquentiel** (en premier) | Modèle Excel/Markdown vide + liste hypothèses. |
| D2 | `general-purpose` — **LLM Cost Modeler** | Calcule $/signal pour Opus 4.7 / Sonnet 4.6 / Haiku 4.5, avec et sans prompt caching, en injectant les tailles réelles de prompt mesurées (cf. `eval_05_llm_findings.md`, system 420-550 tok). Produit tableau cost × model × cache_hit_rate. | **Parallèle** | Tableau coût LLM + sensitivity hit_rate. |
| D3 | `general-purpose` — **Infra Cost Modeler** | $/MAU sur Railway actuel + provider cible recommandé en Prompt 22. Inclut Postgres futur (cf. Prompt 12), Redis si Prompt 21 le requiert. Scénarios 100 / 1k / 10k MAU. | **Parallèle** | Tableau coût infra × MAU. |
| D4 | `Explore` — **Data Cost Modeler** | Inventaire feeds (Dukascopy gratuit non-commercial, Polygon/Tiingo/Twelve Data tarifs commercial), calcule coût mensuel par symbol × TF nécessaire à la prod multi-tenant. Croise avec `data_quality_audit_2026_04_23.md`. | **Parallèle** | Tableau coût data + verdict licence commerciale. |
| D5 | `general-purpose` — **Fixed Cost Auditor** | Estime fixes mensuels : Stripe 2.9 % + 0.30 €/transaction, légal initial (CGU avocat ~2 k€), support (Crisp/Intercom), email transactionnel, monitoring (Sentry). Solo founder = pas de salaire mais opportunity cost à mentionner. | **Parallèle** | Tableau fixes + breakdown one-shot vs récurrent. |
| D6 | `general-purpose` — **ARPU Target Calculator** | À partir de D2+D3+D4, calcule l'ARPU minimum par tier pour marge brute 80 %. Compare aux pricing actuel BUSINESS_PLAN. Identifie tiers sous-pricé. | **Séquentiel** (après D2/D3/D4) | Tableau ARPU cible vs actuel + delta. |
| D7 | `general-purpose` — **Stress Tester** | Scénarios : Anthropic +100 % prix, Anthropic Opus deprecated → fallback Sonnet, cache hit rate effondré 60 % → 20 %, viral growth 10× volume → cap LLM ? Quantifie impact marge. | **Séquentiel** (après D6) | Tableau scénarios × marge × levier mitigation. |
| D8 | `general-purpose` — **Optimization Strategist** | Top 5 leviers ROI : (a) cache agressif, (b) Haiku par défaut FREE/ANALYST, (c) batching narratives, (d) signature-based cache exact, (e) cap signaux/mois. Quantifie économies par levier. | **Parallèle** (après D2) | Liste leviers triés ROI. |
| D9 | `Plan` — **Red-Team** | Challenge : « le hit rate 60 % cache est-il réaliste sans mesure ? Les tarifs Anthropic 2026 utilisés sont-ils les bons ? L'ARPU cible est-il payable par l'ICP du Prompt 25 ? » | **Séquentiel** (avant synthèse) | Liste hypothèses fragiles + corrections. |
| D10 | `general-purpose` — **Synthesis Lead** | Spreadsheet unit economics finale, sensitivity table coût LLM, recommandation tarifs ajustés, runway/break-even, note /10 viabilité. | **Séquentiel** (final) | `reports/eval_24_unit_economics.md` + spreadsheet. |

**Protocole d'orchestration** :
1. D1 pose le squelette + hypothèses.
2. **D2, D3, D4, D5, D8 en parallèle** (5 tool calls dans un seul message).
3. D6 attend D2/D3/D4 → D7 attend D6.
4. D9 challenge, D10 synthétise.

**Livrables** : modèle unit economics par tier (FREE/ANALYST/STRATEGIST/INSTITUTIONAL) avec coût marginal détaillé, sensitivity coût LLM × cache hit rate × prix Anthropic, ARPU cible vs actuel, top 5 leviers d'optimisation chiffrés, scénarios stress, plan caps signaux/mois, note /10 sur viabilité, KPIs (gross margin %, $/signal, $/MAU, runway mois).

---

## Prompt 25 — Product / Market Fit & ICP

**Périmètre** : `BUSINESS_PLAN_SMART_SENTINEL.md`, `COMMERCIALIZATION_REPORT.md`, landing pages concurrentes, communautés trading (Reddit, Discord, Telegram FR/EN).

**Objectif commercial** : construit-on pour les bonnes personnes ? Le vrai ICP entre retail débutant (grand volume, faible ARPU, haute friction support) et prop trader (faible volume, haut ARPU, exigence technique) n'a pas le même produit. Choisir = renoncer.

**Équipe d'agents** :

| # | Agent | Rôle | Mode | Outputs |
|---|-------|------|------|---------|
| E1 | `Plan` — **PMF Lead** | Lit `BUSINESS_PLAN_SMART_SENTINEL.md` et `COMMERCIALIZATION_REPORT.md`. Liste les 3-5 ICP candidats hérités du business plan (XAU scalper FR, prop firm trader, crypto swing EN, débutant ICT, etc.). Définit la grille d'évaluation (TAM, WTP, accessibilité, fit produit). | **Séquentiel** (en premier) | Liste ICP candidats + grille critères. |
| E2 | `general-purpose` — **ICP Researcher** (persona n°1) | Approfondit ICP n°1 : persona détaillé (âge, expérience, capital, outils actuels, frustrations, JTBD, WTP estimé). Source : forums, threads Reddit, témoignages publics. | **Parallèle** | Fiche persona n°1 (1 page). |
| E3 | `general-purpose` — **ICP Researcher** (persona n°2) | Idem pour ICP n°2. | **Parallèle** | Fiche persona n°2. |
| E4 | `general-purpose` — **ICP Researcher** (persona n°3) | Idem pour ICP n°3. | **Parallèle** | Fiche persona n°3. |
| E5 | `general-purpose` — **Competitive Landing Analyst** | Scrape (WebFetch) les landing pages : TradingView, LuxAlgo, SignalStack, TrendSpider, BullBearish, GoldBull, et 2-3 outsiders. Extrait : value prop, pricing, hero copy, social proof, CTA. | **Parallèle** | Tableau concurrents × 6 dimensions + screenshots clés. |
| E6 | `general-purpose` — **Community Listener** | Mine Reddit (r/algotrading, r/Forex, r/Daytrading), Discord/Telegram publics. Cherche : plaintes récurrentes sur signaux IA, demandes non couvertes, prix considérés acceptables, mots utilisés (vocabulaire). | **Parallèle** | Top 10 plaintes + top 10 demandes + glossaire ICP. |
| E7 | `general-purpose` — **Interview Script Designer** | Conçoit script 15 min cold-call (10 questions max) : problème, alternative actuelle, willingness-to-pay, killer feature. Inclut version FR + EN. Liste 20 prospects à approcher (LinkedIn / Discord / Reddit DMs). | **Parallèle** | Script + liste prospects. |
| E8 | `general-purpose` — **Messaging A/B Designer** | Propose 4-6 variantes de hero copy testant deux axes : (a) « AI-powered signals » vs « Explainable smart money alerts » vs « Institutional-grade gold setups », (b) bénéfice (PnL) vs caractéristique (ICT) vs émotionnel (clarté). | **Parallèle** | Tableau variantes + hypothèse de gain. |
| E9 | `general-purpose` — **Niche Strategist** | À partir de E2-E6 : recommande **UNE seule** niche à dominer en priorité. Justifie par TAM × WTP × accessibilité × fit produit actuel × moat possible. Argue ce qu'on **ne fait pas** (les 2 autres ICP en mode « plus tard »). | **Séquentiel** (après E2-E6) | Recommandation niche + justification chiffrée. |
| E10 | `Plan` — **Red-Team** | Challenge : « la niche choisie a-t-elle vraiment du WTP ? Les concurrents y sont-ils déjà saturés ? Le produit actuel (PF 0.96, BOS uniquement, multi-asset) sert-il vraiment cette niche ou faut-il un pivot fonctionnel ? » | **Séquentiel** (avant synthèse) | Critiques + ajustements de scope. |
| E11 | `general-purpose` — **Synthesis Lead** | Fiche ICP gagnante (1 page), positionnement (Geoffrey Moore template), messaging recommandé, plan d'interviews (10 cibles), note /10 clarté positionnement. | **Séquentiel** (final) | `reports/eval_25_pmf_icp.md`. |

**Protocole d'orchestration** :
1. E1 cadre les 3 ICP candidats.
2. **E2, E3, E4, E5, E6, E7, E8 en parallèle** (7 tool calls dans un seul message).
3. E9 attend tous les outputs ICP/concurrents/communauté.
4. E10 challenge, E11 synthétise.

**Livrables** : 3 fiches persona ICP candidats, tableau concurrents × 6 dimensions, top 10 plaintes/demandes communauté, glossaire ICP, script d'interview FR+EN + 20 prospects, 4-6 variantes hero copy, **recommandation d'UNE niche unique à dominer** avec justification chiffrée et liste explicite de ce qu'on ne fait pas, positionnement Moore-style, note /10 clarté positionnement, KPIs (interviews bookées, cold-reply rate, landing CVR par variante, NPS cible).

---

## Prompt 26 — Competitive Analysis & Différenciation

**Périmètre** : marché 2026 des signaux trading IA.

**Objectif commercial** : **moat ou pas**. Si un gros acteur peut copier en 3 mois, valuation nulle.

**Mission** :
1. Mapper les 10 concurrents directs (TradingView Alerts, LuxAlgo, StockHero, TrendSpider, MarketBull, autres).
2. Features matrix × prix × user base × funding.
3. Identifier 3 différenciateurs uniques et défendables (narrative LLM explicable ? Smart money ICT formalisé ? Multi-TF confluent ?).
4. Analyse des avis clients (Trustpilot, Reddit r/algotrading, Discord communautés).
5. Risque : Bloomberg ou TradingView intègre Claude et écrase tout le monde en 6 mois — plan B.
6. Partnerships potentiels (brokers : ICMarkets, Exness — revshare sur conversions).

**Livrables** : matrice concurrentielle, 3 moats à construire, note /10 sur défendabilité.

---

## Prompt 27 — Pricing Strategy & Tier Design

**Périmètre** : 4 tiers FREE/ANALYST/STRATEGIST/INSTITUTIONAL (gelés en attendant test perso).

**Objectif commercial** : le pricing *est* la stratégie. Mauvais pricing = laisser de l'argent sur la table OU tuer la croissance.

**Mission** :
1. Benchmark pricing des concurrents par feature-bundle.
2. Van Westendorp Price Sensitivity Meter sur ICP.
3. Psychological anchoring (tier « Decoy » à $999 pour rendre $199 raisonnable).
4. Annual vs monthly (annual = +20 % LTV, -30 % churn apparent).
5. Metered vs flat : signaux/mois illimités vs quota ?
6. Enterprise / INSTITUTIONAL : custom pricing, contract minimum 12 mois, SLA.
7. Gratuit vs free-trial 14 j — impact conversion.
8. Commercial : grid de pricing recommandée avec justification chaque ligne.

**Livrables** : pricing page v1, analyse de sensibilité, note /10.

---

## Prompt 28 — GTM / Growth / Acquisition

**Périmètre** : landing, SEO, contenu, social, partenariats.

**Objectif commercial** : CAC < LTV/3, sinon pas de croissance saine.

**Mission** :
1. SEO : quels mots-clés (« xau signals », « smart money concepts AI », « gold trading bot ») — difficulté, volume, intent.
2. Contenu : blog / YouTube / Twitter — cadence soutenable pour fondateur solo.
3. Communauté : Discord / Telegram public gratuit → funnel vers tier payant.
4. Influencers trading (coût CPM, ROI).
5. Product Hunt launch, Reddit AMAs, podcasts.
6. Referral program (1 filleul = 1 mois offert).
7. Paid (Google, Meta, YouTube) — à éviter tant que funnel organic non validé.
8. Metrics : landing conversion %, trial→paid %, MRR growth.

**Livrables** : plan GTM 90 jours, budget marketing, note /10 sur soutenabilité solo.

---

## Prompt 29 — Compliance & Légal

**Périmètre** : disclaimers, CGU, RGPD, licences data, régulations financières.

**Objectif commercial** : **blocker potentiel**. Diffuser des « signaux de trading » peut qualifier en conseil en investissement (CIF en France, IA en Europe, RIA US) → licence obligatoire.

**Mission** :
1. Qualification juridique : signaux éducatifs (safe) vs conseil personnalisé (régulé).
2. Disclaimer « not financial advice » suffisant par juridiction ? (FR : AMF ; EU : MiFID II ; US : SEC).
3. RGPD : DPA, registre traitements, droit à l'oubli, transferts hors-UE (Anthropic US).
4. Licences data : Dukascopy, MT5, broker feeds — usage commercial autorisé ?
5. CGU / CGV rédigées par avocat spécialisé SaaS.
6. PSD2 / KYC si gestion paiements directs (ou déléguer à Stripe/Paddle).
7. Assurance RC Pro + Cyber.
8. Restrictions géographiques (bloquer US / Québec / restricted jurisdictions ?).

**Livrables** : matrice de risque juridique × juridiction, checklist à transmettre à avocat, note /10 sur exposition actuelle.

---

## Protocole d'exécution suggéré

1. **Exécuter les prompts 01 → 09 en priorité** (fondations techniques critiques : pipeline, scoring, data).
2. Puis **10 → 17** (API, sécurité, tests).
3. Puis **18 → 24** (perf, coûts, MLOps).
4. Enfin **25 → 29** (commercial, go-to-market, légal).
5. Chaque exécution produit un rapport dans `reports/eval_NN_<secteur>.md` + mise à jour `MEMORY.md` avec les findings surprenants.
6. Un `reports/eval_synthesis.md` agrège notes /10, top 10 chantiers cross-prompts, priorisation globale.

**Critères d'acceptation d'un rapport** : diagnostic chiffré, top 5 améliorations avec estimation effort × impact, KPIs post-amélioration, pas de recommandation vague (« améliorer la qualité du code » est rejeté ; « extraire ConfluenceDetector.compute_score en 3 fonctions pures testables, coverage 95 %+ » est accepté).
