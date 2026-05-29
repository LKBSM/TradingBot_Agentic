# Master Plan — Commercialisation Smart Sentinel AI

> **Date** : 2026-05-23
> **Auteur** : Architecte Master Plan (unification des 20 plans Sprint commercialisation)
> **Branche** : `institutional-overhaul`
> **Sources** : `reports/commercialization_sprint/{01..20}_*.md`, MEMORY.md, eval_00..29, `memory/mtf_rewiring_2026_05_23.md`, `memory/a1_verdict_2026_05_01.md`, `memory/decision_matrix_2026_04_30.md`, `memory/feedback_multi_view_ux.md`, `reports/roadmap_2026_2027.md`.
> **Statut produit** : pré-launch ; pas d'edge prédictif prouvé (A1 DSR=0, MTF Phase 2 rejet PF lo −0.043, 0/4 strats CI95 PF lo>1.0). Pivot acté **narrative-first + B2B-API** avec UX 3-vues (FOCUS / CO-PILOT / EXPERT).

---

## 0. Executive Summary (½ page)

**Vision produit acté (verrouillée par 3 verdicts empiriques)** :
- **A1 (LightGBM 2 niveaux + 19 features + CPCV 28 paths)** : score 1/6 critères, DSR=0, PBO=0.50, DM stat=+46.7 → **aucun edge prédictif XAU M15** (`memory/a1_verdict_2026_05_01.md`).
- **MTF Phase 2** : filtrage H4-aligné dégrade PF lo CI95 de −0.043 vs gate +0.050 → **`htf_alignment` weight reste 0**, wiring conservé pour observability/narratives uniquement (`memory/mtf_rewiring_2026_05_23.md`).
- **Decision matrix 4 strats (XAU M15/H1, EURUSD, NR4)** : 0/4 franchissent PF lo CI95>1.0 ; EURUSD se dégrade post-2024 = preuve β-capture XAU bull. → **kill paradigme "predict & deliver", pivot "explain & contextualise"** (`memory/decision_matrix_2026_04_30.md`).

**Produit cible** : SaaS narrative-first + B2B-API broker context-layer, calibration honnête (UE 2024/2811 compliant, pas de claim probabiliste), multi-view UX (FREE=FOCUS, ANALYST=+CO-PILOT, STRATEGIST=+EXPERT, INSTITUTIONAL=API JSON). Argument vente = "honest confidence" + "auditable track-record" + "explainability rubric open-sourced" (eval_26 Diff #1+#2+#3).

**Chemin critique 16 semaines** vers GO-LIVE B2C FR-first (MRR cible M6 = $1.5-3k, M12 = $5-7k base case) + B2B-API parallèle (ARR M12 cible $30-60k vers $310k cible). 1 décision GO/NO-GO majeure à S8 (résultats Vision B narrative-first vs continuer P0 algo).

**Budget consolidé** :
- Heures dev P0 only : **~1 100 h** (parmi 3 800 h plans complets)
- Heures dev P0+P1 : **~2 100 h**
- Heures dev plans complets : **~3 800 h** (P0+P1+P2 sur 20 plans)
- Mois calendaires solo (30 h/sem effectif) : P0 = **9 mois**, P0+P1 = **17 mois** → arbitrer ruthlessly
- Coûts récurrents cibles **$580-840/mo** (infra + data + LLM + delivery), variable LLM 20-40 % à 1 k MAU
- Légal one-shot : **6-11 k€** (avocat fintech FR + RC Pro + cyber assurance 3-5 k€/an récurrent)
- Capital initial requis Year 1 : **~$25-35 k** (cash + opportunity cost solo)

**Décision GO/NO-GO majeure** : à S8, si DSR<0.2 sur scoring v2 ET narrative quality eval < 4.0/5 → kill paradigme "signal autonome", focus exclusif B2B-API + Telegram public 60-90j track-record.

---

## 1. Pivot Stratégique Acté

### 1.1 Pourquoi narrative-first + B2B-API plutôt qu'edge prédictif

| Hypothèse testée | Résultat empirique | Source |
|---|---|---|
| LightGBM stacked 2 niveaux trouve un edge XAU M15 | DSR=0, PBO=0.5, CPCV PF=1.008, DM stat=+46.7 (pire que constant) | `memory/a1_verdict_2026_05_01.md` |
| H4 alignment filter améliore PF | PF lo CI95 −0.043 vs gate +0.050 ; n=134 trades ; WR gap 3pp dans noise | `memory/mtf_rewiring_2026_05_23.md` |
| Edge XAU SMC se transfère hors XAU | EURUSD PF 0.854 [0.748, 0.969], dégradation post-2024 inverse exact XAU | `memory/decision_matrix_2026_04_30.md` |
| Confluence additive a pouvoir prédictif | Pearson(score, R) = −0.023, Brier worse than baseline | `reports/eval_02_confluence.md`, `confluence_calibration.md` |

**Verdict** : 4 piliers d'evidence convergent ⇒ pas de claim probabiliste possible ⇒ pas de produit "predict & deliver" commercialisable ⇒ pivot impératif.

### 1.2 Positionnement commercialisable

3 promesses substanciables sans claim de PF/Sharpe :
1. **Honest confidence** — calibration isotonic + conformal prediction + reliability diagrams publics (eval_27, eval_29 alignement MiFID II 2024/2811).
2. **Multi-view UX** (FOCUS/CO-PILOT/EXPERT) — `feedback_multi_view_ux.md` règle persistante. Tier débloque modes, pas signaux.
3. **Auditable track-record** — Telegram public 60-90 j lecture seule, hash-chain ledger (`src/audit/*`), export CSV signé. Différenciation contre LuxAlgo/TradingView qui n'exposent pas leur tape.

### 1.3 Surfaces de vente

| Surface | Mode défaut | Tier débloque | Source |
|---|---|---|---|
| Telegram channel public | FOCUS forcé | gratuit (lead-magnet) | `mockups/telegram_b2c.txt`, eval_25 §11.2 |
| Webapp B2C | CO-PILOT défaut, toggle FOCUS/EXPERT | ANALYST $29 (CO-PILOT), STRATEGIST $79 (EXPERT) | `mockups/v3/best_concept_demo.html` |
| Email digest hebdo | FOCUS | ANALYST | Plan 11 §C6 |
| API B2B JSON | EXPERT (JSON complet) | INSTITUTIONAL $1990 + brokers $999-$4999 | `mockups/b2b_insight.json` |
| Mobile webapp | FOCUS auto <768px | tous tiers | Plan 20 §2.1.3 |

---

## 2. Résolution des Conflits Inter-Catégories

Tableau exhaustif des tensions identifiées en lisant les 20 plans. Pour chaque conflit : argument complet + décision retenue.

### Conflit C1 — Cat 5 data licensing vs Cat 15 news provider

**Tension** : Cat 5 (`05_data_infrastructure.md` §1.4) recommande Trading Economics $79/mo + Tiingo $? + TwelveData $? pour ~$208/mo. Cat 15 (`15_news_macro_pipeline.md` TL;DR) propose Trading Economics $79/mo + NewsAPI $? pour le calendrier+surprise. Risque double-paiement TE.

**Décision** : Trading Economics **$79/mo** souscrit **une fois**, l'instance sert à la fois (a) le calendar avec Actual/Forecast/Previous (Cat 15 P0, résout le surprise=0 historique) et (b) source macro pour Cat 5 le cas échéant. Tiingo et TwelveData déférés en P1. Budget data S1-S8 = **~$80/mo**, montée à ~$130/mo si Polygon.io ajouté pour XAU+EUR live.

**Argument** : le surprise = actual − consensus est le bottleneck pour Pilier 1 event-driven (`reports/3_pillars_implementation_2026_05_13.md:96-99`). Sans TE, le rework `confluence_detector.py` ne peut pas exploiter les news features → Pearson scoring restera bas → P0-5 Cat 2 (refonte Logistic L1) inefficace. Donc TE est upstream des deux plans, à acheter en premier.

### Conflit C2 — Cat 14 P0-G3 tier re-sweep htf_alignment CADUC

**Tension** : `14_multi_asset_mtf.md` §3 G3 affirme « Tiers basés sur replay AVANT `htf_alignment` activé — à re-sweep » et conditionne ce P0 à Phase 3 MTF. Or `memory/mtf_rewiring_2026_05_23.md` confirme **Phase 2 REJET**, Phase 3 non activable.

**Décision** : G3 Cat 14 retiré du P0 et requalifié **CADUC** dans le master plan. Le re-sweep tier `htf_alignment` ne se fera **que** si une amélioration base-strategy ramène PF baseline > 1.0 (re-run `python -m scripts.eval_mtf_alignment` pré-requis). En attendant, `htf_alignment` reste à weight=0 dans `DEFAULT_WEIGHTS` (`src/intelligence/confluence_detector.py:131`). La lecture readout est conservée pour observability + narratives B2C (déjà émise dans Insight v2.2.0).

**Argument** : appliquer un poids sur un filtre validé négatif empiriquement = forçage anti-evidence. Le wiring est conservé (schéma 2.2.0 stable) pour ne pas casser les consommateurs ; uniquement la décision business "lever le poids" est gelée.

### Conflit C3 — Cat 8 SQLite vs Cat 17 Postgres + Redis multi-worker

**Tension** : Cat 8 (`08_api_backend.md` §1.3) liste SQLite WAL + RLock comme persistance actuelle, exposé multi-worker = invariants brisés. Cat 17 (`17_caching_performance.md` §1.3-1.4) recommande Redis pour rate-limiter, cache, sessions ; Postgres pour SignalStore au-delà de 10 k MAU.

**Décision en 3 étapes** :
1. **S1-S6** : conserver SQLite WAL + 1 worker. Externaliser **Redis seul** (`infrastructure/docker-compose.yml:75-94` déjà défini, jamais consommé) pour : rate-limiter IP, semantic cache compteurs, RAM-shared idempotency. Coût +$5-15/mo Cloudflare/Upstash free tier.
2. **S7-S12** : passer à **2-4 workers gunicorn** une fois Redis multi-worker safe.
3. **Post-S16** : migrer SignalStore → Postgres uniquement si MAU > 1 k OR business-need B2B haute-fréquence. SQLite reste viable jusqu'à 5-10 k MAU lecture-dominant.

**Argument** : la prématuration Postgres coûte 24-40 h migration + risque DB corruption pendant fenêtre de cutover. Tant que < 500 MAU et lecture-dominant, SQLite WAL tient à 800-1500 RPS local. Redis répond aux 80 % des besoins multi-worker (cache + rate-limit) sans toucher au schéma data.

### Conflit C4 — Cat 10 rate-limit Redis vs Cat 11 rate-limit per-provider

**Tension** : Cat 10 (`10_auth_security.md` F-15) signale RateLimiter IP-based ne respecte pas X-Forwarded-For = spoofable. Cat 11 (`11_delivery_channels.md` C4) demande rate-limit côté Telegram (>30 abonnés = ban). Risque double implémentation incohérente.

**Décision** : 1 Redis cluster, 2 namespaces distincts :
- `ratelimit:api:ip:{ip}` — pour Cat 10 (entrée API, lecture `X-Forwarded-For` trusted proxy CIDR list).
- `ratelimit:delivery:{provider}:{chat_id}` — pour Cat 11 (sortie Telegram/Discord, respect Retry-After header).

Implémentation centralisée dans `src/intelligence/security.py` (déjà existant) qui devient un module Redis-aware. Cat 11 le réutilise via injection.

### Conflit C5 — Cat 9 cache key bar_ts vs Cat 17 SemanticCache

**Tension** : `09_llm_narratives.md` mention "SemanticCache key sans bar_ts" comme issue ; Cat 17 (`17_caching_performance.md` §1.1) confirme `bar_timestamp` exclu volontairement mais aussi `session` → 2 signaux 5h d'intervalle même narrative. Question : déjà patché ?

**Décision** : vérifier `src/intelligence/semantic_cache.py:30-158` au démarrage S1. Patch :
- `SCORE_BUCKET_PTS = 10` (déjà ligne 104 selon Cat 17 §1.1 — déjà patché).
- Ajouter `session_bucket` (3 buckets : Asia / EU / US) au hash composite pour résoudre collision intra-jour.
- Ajouter test régression `tests/test_semantic_cache_session_isolation.py`.

Effort : ~3 h. **No-regret quick-win** (QW-016 catalogue).

### Conflit C6 — Cat 2 refonte Logistic L1 vs Cat 3 ML Vision A vs Vision B

**Tension** : Cat 2 (`02_smart_money_algo.md` P0-5) propose refonte scoring Logistic L1 + LightGBM ; Cat 3 (`03_machine_learning.md` Vision A/B) demande arbitrer entre retry edge prédictif (A) ou pivot narrative-first (B). Le P0-5 Cat 2 = Vision A déguisée.

**Décision** : Vision **B narrative-first** confirmée comme défaut (par défaut du verdict A1, `memory/a1_verdict_2026_05_01.md:23`). Le P0-5 Cat 2 (Logistic L1 + isotonic + conformal) est **conservé** mais reframé : son objectif n'est plus de "trouver edge directionnel" mais de **calibrer la conviction** (P(win) honnête) pour driver le tier-classification (PREMIUM/STANDARD/WEAK) et alimenter le narrative LLM avec un nombre auditeur-defendable.

Critères d'acceptance modifiés :
- Brier skill score ≥ +5 % vs base rate (calibration, pas edge).
- Reliability diagram monotone.
- PF claim hors-scope (assumé < 1.20).

**Argument** : la mécanique Cat 2 P0-5 est précieuse pour la conviction layer même sans edge. Reframe = on garde 40-56h de dev avec target faisable et conformément au verdict A1.

### Conflit C7 — Cat 19 Fly.io vs Cat 17/12 coûts infra

**Tension** : `19_mlops_deployment.md` §2.3 recommande Fly.io cdg ~$30-40/mo. Cat 12 (`12_observability.md` §0) budget $0-50/mo. Cat 17 (`17_caching_performance.md`) +$5-15/mo Redis. Tot ≈ $35-105/mo infra.

**Décision** : Fly.io cdg (Paris) acté comme hosting prod **+** Cloudflare R2 model registry **+** Upstash Redis free tier (≤10 k commands/jour, $0 to start, $0.20/100k after). Tot baseline = **$35/mo**, montée à $100/mo à 1 k MAU. **Aligne les 3 plans.**

### Conflit C8 — Cat 18 Dukascopy/FF cease vs Cat 5 conserver feed

**Tension** : Cat 18 (`18_compliance_legal.md` §1.3) classe Dukascopy + FF en "usage commercial déguisé" ⇒ bloqueur Stripe + risque cease-and-desist. Cat 5 (`05_data_infrastructure.md`) reconnaît zone grise mais le pipeline existant tourne dessus.

**Décision** : Dukascopy + FF **restreints à backtest interne** (note "internal only" documentée dans `data/rag/sources_manifest.yaml`). En **prod live** :
- Calendar live : migration Trading Economics ($79/mo) — P0 Cat 15.
- OHLCV live : Polygon.io ou Tiingo — P1 Cat 5.
- Backtest historique : Dukascopy/FF conservés (recherche interne, non publié, non commercialisé directement).

Marquage explicite dans `BACKTEST_LEGAL_GUARDRAILS.md` : tout chiffre publié issu de Dukascopy = pas de claim live. P0 absolu (= Item DG-013/014 catalogue).

### Conflit C9 — Cat 1 pricing $1990 INSTITUTIONAL vs Cat 7 risk un-bound

**Tension** : Cat 1 (`01_commercialization_gtm.md` §0) lock pricing FREE/$29/$79/$1990. Cat 7 (`07_risk_management.md`) sans risk-engine consolidé, l'INSTITUTIONAL B2B-API ne peut pas vendre du "risk score" garanti.

**Décision** : INSTITUTIONAL **lancé en S12+** seulement après Cat 7 P0 (1 RiskManager consolidé) + Cat 6 P0 (gates institutionnels documentés). Avant S12, contact INSTITUTIONAL = "Book a demo" Calendly, pas auto-checkout. Cat 1 P0-T10 page pricing : INSTITUTIONAL card affichée mais CTA = book demo, **pas** Subscribe.

### Conflit C10 — Cat 13 testing 75 % coverage vs Cat 17 perf benchmark

**Tension** : Cat 13 (`13_testing_infrastructure.md` §2) demande 75 % coverage global + 85 % zones revenue. Cat 17 ajoute load tests Locust + perf benchmark CI. Cat 12 ajoute traces OpenTelemetry. Concurrence pour le runtime CI < 5 min.

**Décision** : 3 workflows GH Actions distincts (déjà spec'd Cat 13 §2) :
- `ci.yml` (PR-blocking, < 5 min) : lint + type + unit + coverage gate 75 %.
- `integration.yml` (nightly) : E2E docker-compose, contract tests, perf bench seuils.
- `nightly.yml` : mutation + load + security scans.

Les tests perf de Cat 17 et tracing de Cat 12 vont dans `integration.yml` (pas `ci.yml`), respectant le SLA 5 min.

### Conflit C11 — Cat 16 sweep state machine vs Cat 2 confluence rework

**Tension** : `16_signal_state_machine.md` §1.3 « tant que ConfluenceDetector produit un score à Pearson −0.023, aucun sweep state machine ne franchira les gates ». Cat 2 P0-5 attend ~5-7 jours.

**Décision** : Cat 16 P0 (sweep 432 cellules) **séquentiel après** Cat 2 P0-5 livré et validé (Brier skill ≥ +5 %). Pas de sweep parallèle. Cat 16 P0 robustesse (checksum, multi-process safety, versioning) reste parallélisable car indépendant du score. → Cat 16 split en P0a (~30 h sécu/persistence, parallèle dès S1) et P0b (~46 h sweep, séquentiel post Cat 2 P0-5 ≈ S8).

### Conflit C12 — Cat 20 stack Next.js vs Cat 19 hosting Fly.io

**Tension** : `20_product_ux.md` §0 demande Next.js 15 App Router. Cat 19 dimensionne hosting backend Fly.io. Next.js a un déploiement web séparé optimal (Vercel/Cloudflare Pages).

**Décision** : 2 hosting cibles distincts :
- **Frontend Next.js** : Vercel free tier ou Cloudflare Pages (gratuit jusqu'à 500 builds/mo). $0/mo.
- **Backend FastAPI** : Fly.io cdg $30-40/mo (Cat 19).

Pas de fusion. Domain `smartsentinel.ai` → CNAME landing front, `api.smartsentinel.ai` → A record Fly.io.

### Conflit C13 — Cat 9 NARRATIVE_MODE=llm défaut vs Cat 17 coût LLM

**Tension** : Cat 9 (`09_llm_narratives.md`) qualité narrative implique NARRATIVE_MODE=llm prod. Cat 17 §1.6 LLM = 80-90 % facture à 10k MAU.

**Décision** : NARRATIVE_MODE par tier (param env + override per-request) :
- FREE/FOCUS : NARRATIVE_MODE=template (déterministe, no LLM cost).
- ANALYST/CO-PILOT : NARRATIVE_MODE=llm Haiku (low cost ~$0.0008/narrative).
- STRATEGIST/EXPERT : NARRATIVE_MODE=llm Sonnet (medium ~$0.008/narrative).
- INSTITUTIONAL/API : NARRATIVE_MODE=llm Opus on-demand (high ~$0.03/narrative, gated).

Marges brutes par tier alignées (eval_27, eval_24). Coût LLM borné par tier × cache hit rate cible 40 %+ (Cat 17 P0).

### Conflit C14 — Cat 11 webhook B2B publisher vs Cat 8 API versioning

**Tension** : Cat 11 (`11_delivery_channels.md` C9) demande `WebhookPublisher` pipeline pour B2B. Cat 8 (`08_api_backend.md` §1.6) pointe absence de versioning v1/v2 propre. Webhooks ont leur propre versioning (`event.version=2.1.0`).

**Décision** : `mockups/b2b_webhook_payload.json` figé à v1.0.0. Webhook event versioning indépendant du REST API path versioning. Pas de bump REST tant que `InsightSignalV2` (2.2.0 déjà shipped) reste stable. WebhookPublisher consomme directement `InsightSignalV2.model_dump(mode="json")` sans transformation.

### Conflit C15 — Cat 6 backtest IC bootstrap vs Cat 1 GTM track-record public

**Tension** : Cat 6 (`06_backtest_validation.md` §0) interdit publication chiffres sans walk-forward + IC95 (cf `BACKTEST_LEGAL_GUARDRAILS.md`). Cat 1 (`01_commercialization_gtm.md` §2.1.3) demande track-record public 60-90j Telegram pour vendre.

**Décision** : track-record public = **forward only** (paper trading public Telegram, signaux passés non-rétro-cherrypickés). Aucun chiffre historique simulated published. Le Telegram public n'affiche **que** des signaux émis post-S2 forward + screenshots PnL réels. IC95 sur les trades clos est calculé live et exposé dans `/track-record` UI à partir de n≥30 trades.

Ce design est compliant + scientifiquement défendable + minimise risque de cherry-pick claim.

---

## 3. Catégorisation des 20 plans en 5 Piliers

### Pilier A — Foundation (infrastructure prête-prod)

Catégories : **8, 10, 12, 13, 17, 19**.
Objectif : socle technique sur lequel tout repose. Sans ces 6 plans, rien d'autre ne tient en charge ou en sécurité.

**Total effort P0 + P1** : ~750 h. **Critical path** : Cat 19 (deploy) → Cat 10 (auth) → Cat 8 (async) → Cat 17 (cache+multi-worker) → Cat 13 (tests CI) → Cat 12 (obs).

### Pilier B — Algo & Quality (production du signal)

Catégories : **2, 3, 4, 6, 16**.
Objectif : ce qui produit le signal (scoring, vol, ML calibration, state machine, backtest).

**Total effort P0 + P1** : ~900 h. **Critical path** : Cat 5 data (P0 amont) → Cat 2 refonte scoring → Cat 16 sweep state machine → Cat 6 gates institutionnels → Cat 4 vol latence → Cat 3 ML conviction layer.

### Pilier C — Data & Compliance (entrants + légal)

Catégories : **5, 15, 18**.
Objectif : entrants data licenciés (XAU+EUR+macro+news) + légalité claims/CGU/DSAR.

**Total effort P0 + P1** : ~370 h + 6-11 k€ externe. **Critical path** : Cat 18 W4 (avocat) parallèle, Cat 5/15 P0 Trading Economics → Cat 18 DSAR endpoints → Cat 18 RC Pro.

### Pilier D — Delivery & Product (sortie + UX)

Catégories : **9, 11, 14, 20**.
Objectif : ce que voit le client final (LLM narrative, Telegram/webhook/email, UX 3-vues, multi-asset).

**Total effort P0 + P1** : ~750 h. **Critical path** : Cat 9 LLM prod cascade + tier-routing → Cat 20 stack Next.js + webapp MVP → Cat 11 delivery queue Telegram retry + email + webhook publisher → Cat 14 USOIL preset + decimal precision (G3 caduc).

### Pilier E — Business (vendre)

Catégories : **1, 7**.
Objectif : monétiser (pricing, Stripe, GTM) + risk packaging utilisateur (SL/TP affiché conformément, kill-switch consolidé).

**Total effort P0 + P1** : ~400 h. **Critical path** : Cat 1 P0 pricing + Stripe + landing → Cat 7 P0 1 RiskManager + kill-switch unifié + risk score UI → Cat 1 P0 track-record public 60-90j.

---

## 4. Chemin Critique 16 Semaines (Gantt textuel)

Hypothèse : Loukmane solo, **30 h/sem dev effectif** + 8-10 h/sem GTM/marketing/relations clients. Marge réelle = 30 h × 16 sem = **480 h dev sur 16 sem** (P0 plans complets > 1 100 h ⇒ priorisation ruthless ⇒ ce plan ne tente PAS de livrer tout P0 en 16 sem mais le **chemin minimal vers une bêta privée payante FR**).

### S1-S2 — Foundation hardening + légal kickoff

**Catégories actives** : 18 (légal), 10 (auth), 19 (deploy), 13 (CI), 17 (quick-wins).

**Tâches concrètes** :
- Cat 18 P0 : engagement avocat fintech (`18_compliance_legal.md` P0 W4, 6-11 k€ externe, démarrer aujourd'hui les RFQ) — 4 h interne.
- Cat 10 P0 F-04 + F-05 + F-03 (HMAC admin, UNIQUE api_key_id, subscription_expires) — 12 h.
- Cat 19 P0 : créer `fly.toml`, GHCR build, deploy staging Fly.io — 16 h.
- Cat 13 P0 : `.coveragerc`, `pytest.ini` durci, supprimer `test_long_short_trading.py`, 3 workflows GH Actions — 14 h.
- Cat 17 P0 : Redis branché (rate-limit + semantic cache compteurs), `cleanup_expired` câblé, connection pool SQLite — 16 h.
- Cat 12 P0 : 6 métriques business émises + JSONFormatter `extra={}` merge + 25 print() critiques migrés — 16 h.
- QW catalogue items 1-15 (parallèle, < 2 h chacun) — 20 h cumulé.

**Heures totales semaine S1+S2** : ~98 h ÷ 2 = **49 h/sem** (over-allocation alert) ⇒ étendre à S1-S3 si nécessaire.

**Checkpoints** : S2 fin = staging Fly.io répond à `/health`, CI bloquante sur 75 % coverage, avocat engagé avec NDA + draft CGU envoyé.

### S3-S5 — Data + News pivot

**Catégories actives** : 5 (data), 15 (news), 2 (confluence refonte démarrage), 9 (LLM tier-routing).

**Tâches concrètes** :
- Cat 5 P0 : souscription Trading Economics ($79/mo), branchement `EconomicCalendarFetcher` sur TE API (`05_data_infrastructure.md` §1.4 + Cat 15 P0) — 24 h.
- Cat 15 P0 : enrichir CSV historique avec Actual/Forecast/Previous depuis TE archive, ressusciter surprise feature — 16 h.
- Cat 5 P0 : `validate_ohlcv` au boot fail-fast, dépréc Dukascopy live (recherche only), restriction documentée — 12 h.
- Cat 2 P0-4 + P0-5 démarrage (component scores persistés + scaffold Logistic L1 calibration loop) — 28 h.
- Cat 9 P0 : tier-routing Haiku/Sonnet/Opus + cascade off + auto-fallback Template + cache key sans bar_ts (déjà partiellement fait, vérifier) — 18 h.

**Total S3-S5** : ~98 h ÷ 3 = **33 h/sem**. Réaliste.

**Checkpoints** : S5 fin = Calendar live TE, scoring v2 entraînement OOS démarré, narrative tier-routed.

### S6-S8 — Refonte scoring + state machine sweep

**Catégories actives** : 2 (scoring finalisé), 16 (sweep state machine), 6 (gates institutionnels), 4 (vol latence).

**Tâches concrètes** :
- Cat 2 P0-5 finalisation Logistic L1 + isotonic + conformal wrapper + Brier skill validation — 28 h.
- Cat 2 P0-1/2/3 (OB ICT-strict, FVG 0.4 ATR, retest 0.25 ATR) si scoring tient → 30 h, sinon defer P1.
- Cat 16 P0a (checksum, multi-process safety, schema versioning) — 18 h.
- Cat 16 P0b démarrage sweep 432 cellules sur scoring v2 — 24 h.
- Cat 6 P0 : exécuter `reports/eval_18_walkforward_skeleton.py` sur baseline + scoring v2, gates institutionnels documentés — 20 h.
- Cat 4 P0 : latence HAR `_add_features` incrémental → cible 50 ms p99 — 12 h.

**Total S6-S8** : ~132 h ÷ 3 = **44 h/sem** (à boucler en 3-4 sem).

**Checkpoints S8 — DÉCISION GO/NO-GO majeure** :
- Si Brier skill ≥ +5 % ET narrative quality eval ≥ 4.0/5 ET sweep state machine trouve ≥ 1 cellule passe gates : **GO** continuer vers launch B2C.
- Si Brier skill < +2 % AND narrative quality < 3.5/5 : **PIVOT** total vers B2B-API context-layer broker + Telegram public seul (FREE/lead-magnet).

### S9-S11 — Delivery + UX webapp MVP

**Catégories actives** : 11 (delivery), 17 (caching), 20 (UX), 14 (multi-asset).

**Tâches concrètes** :
- Cat 11 P0 : Telegram retry + dedup `(chat_id, signal_id)` + python-telegram-bot v20+ async fix + rate-limiter delivery (`11_delivery_channels.md` C1-C4) — 24 h.
- Cat 11 P0 : module Email digest (mockups/telegram_b2c.txt:58 promesse Analyst $14/mo) — 16 h.
- Cat 17 P0 : async I/O end-to-end routes critiques (`routes/signals.py`, `narratives.py`, `dashboard.py`) — 20 h.
- Cat 20 P0 : stack Next.js 15 + Tailwind + shadcn/ui scaffolded + webapp B2C MVP 3-vues toggle persistant (`mockups/v3/best_concept_demo.html` base) — 60 h.
- Cat 14 P0 G1 (`pip_value`, `weekend_behavior`, `news_relevance_weight`, `atr_baseline_usd` ajoutés à `InstrumentConfig`) + G2 décision GA = XAU+EUR (USOIL P1) — 8 h.

**Total S9-S11** : ~128 h ÷ 3 = **43 h/sem**.

**Checkpoints** : S11 fin = Webapp B2C 3-vues live staging, Telegram retry+dedup actifs, email digest cron-schedulable.

### S12-S14 — GTM + backtest publish + risk management

**Catégories actives** : 1 (GTM Stripe), 6 (gates + walk-forward final), 7 (risk consolidé), 12 (observabilité + alertes).

**Tâches concrètes** :
- Cat 1 P0-T1 à P0-T9 : lock pricing + Stripe products + Checkout/Portal + webhook + landing master EN + FR + hosting Vercel — 60 h.
- Cat 7 P0 : 1 RiskManager canonical (drop 3 moteurs orphelins), `position_size` exposé dans `InsightSignalV2`, risk score 0-100 lisible, kill-switch unifié — 32 h.
- Cat 6 P0 : walk-forward final 3 instruments (XAU+EUR+USOIL si data), `BACKTEST_LEGAL_GUARDRAILS.md` aligné, `/track-record` endpoint export CSV signé — 24 h.
- Cat 12 P0 : alerting Prometheus + Discord webhook + runbooks 5 incidents top — 16 h.

**Total S12-S14** : ~132 h ÷ 3 = **44 h/sem**.

**Checkpoints** : S14 fin = checkout fonctionnel mode TEST, risk score affiché Telegram + webapp, `/track-record` publique live.

### S15-S16 — Hardening + soft launch FR-first

**Catégories actives** : 18 (W4 retour avocat + DSAR + cookie banner), 13 (CI mutation + load), 1 (track-record public Telegram J+30), 11 (multi-langue narrative).

**Tâches concrètes** :
- Cat 18 P0 : retour avocat sur CGU + Privacy, signature, mise en ligne `/terms` `/privacy` HTML, cookie banner Tarteaucitron, DSAR `GET /me/data` + `DELETE /me` — 32 h.
- Cat 13 P0 : test load Locust + mutation subset zones revenue + security scan Bandit/pip-audit/Trivy — 18 h.
- Cat 1 P0 : Telegram public channel "Smart Sentinel — Public Tape" ouvert, premier signaux émis (forward only) — 6 h + monitoring.
- Cat 11 P1 : narrative LLM multi-langue FR/EN (DE/ES post-launch) — 14 h.
- Cat 1 P0 : campagne pre-launch FR (AMA r/Forex, Pine Script TV gratuit, 3 articles SEO W5) — 18 h.

**Total S15-S16** : ~88 h ÷ 2 = **44 h/sem**.

**Checkpoints** : S16 fin = bêta privée FR ouverte sur waiting list (~50-100 emails capturés), premiers payants $29 STARTER possibles, Telegram public 30j de tape.

### Synthèse Gantt (heures par catégorie par sprint)

| Sprint | Cat actives (h cumulées) | Total h | Décision |
|---|---|---|---|
| S1-S2 | 18(4) 10(12) 19(16) 13(14) 17(16) 12(16) + QW(20) | 98 | engagement avocat |
| S3-S5 | 5(36) 15(16) 2(28) 9(18) | 98 | TE live, scoring démarré |
| S6-S8 | 2(58) 16(42) 6(20) 4(12) | 132 | **GO/NO-GO S8** |
| S9-S11 | 11(40) 17(20) 20(60) 14(8) | 128 | webapp live staging |
| S12-S14 | 1(60) 7(32) 6(24) 12(16) | 132 | checkout test mode |
| S15-S16 | 18(32) 13(18) 1(24) 11(14) | 88 | bêta FR ouverte |
| **Total 16 sem** | | **676 h** | **+ GTM 8-10 h/sem cumul 144h** |

**Marge totale** : 480 h × 16 = **480 h prévues, 676 h listées ⇒ déficit de ~200 h.** À résoudre par : (a) overbooking accepté quelques semaines, (b) drop fonctionnalités optionnelles (ex : USOIL preset → P1 post-launch), (c) outsourcing design 30 h externe Cat 20 (~$2 700-3 600).

---

## 5. Budget Consolidé

### 5.1 Heures dev

| Pilier | P0 (h) | P0+P1 (h) | Plan complet (h) |
|---|---:|---:|---:|
| A Foundation (Cat 8/10/12/13/17/19) | 240 | 530 | 760 |
| B Algo & Quality (Cat 2/3/4/6/16) | 420 | 700 | 1 240 |
| C Data & Compliance (Cat 5/15/18) | 280 | 370 | 518 |
| D Delivery & Product (Cat 9/11/14/20) | 270 | 580 | 940 |
| E Business (Cat 1/7) | 300 | 400 | 440 |
| **Total** | **1 510** | **2 580** | **3 898** |

(Chiffres arrondis à partir des Sub-total P0/P1/P2 énumérés dans les 20 plans ; certains plans n'explicitent pas P0 vs P1 ⇒ approximations conservatives.)

**Calendrier solo 30h/sem** :
- P0 seul : ~50 sem ≈ **12 mois**.
- P0+P1 : ~86 sem ≈ **20 mois**.
- Plan complet : ~130 sem ≈ **30 mois**.

⇒ **Le master plan 16 semaines couvre ~50 % des P0** (676 h sur 1 510 h P0 totaux). Le reste est defer à post-launch ou outsource.

### 5.2 Coûts récurrents cibles (S1-S16)

| Item | Provider | Coût/mo | Source |
|---|---|---:|---|
| Backend hosting | Fly.io cdg | $30-40 | Cat 19 §2.3 |
| Frontend hosting | Vercel free | $0 | Cat 20 |
| Model registry + backups | Cloudflare R2 | $1-2 | Cat 19 |
| Cache + rate-limit | Upstash Redis | $0-10 | Cat 17 |
| Secrets vault | Doppler free | $0 | Cat 19 |
| Calendar data | Trading Economics | $79 | Cat 15 |
| OHLCV data live | Polygon.io (P1) | $0 → $99 | Cat 5 |
| LLM API | Anthropic | $50-500 variable | Cat 9 |
| Email | Mailgun pay-as-you-go | $5-35 | Cat 11 |
| Telegram bot | gratuit | $0 | Cat 11 |
| Sentry error tracking | Free tier | $0-26 | Cat 12 |
| Stripe | 2.9 % + $0.30/tx | variable | Cat 1 |
| Stripe Tax | $0.50/tx | variable | Cat 1 |
| Cookie consent | Tarteaucitron self-host | $0 | Cat 18 |
| Calendly Pro | 1 user | $12 | Cat 1 |
| Buffer (social) | starter | $6 | Cat 1 |
| Domain + DNS | Namecheap | $15/an | Cat 1 |
| **Total baseline** | | **~$185-220/mo** | |
| **Total à 1 k MAU** | | **~$400-650/mo** | |
| **Total à 10 k MAU (post-S16)** | | **~$1 200-1 800/mo** | |

### 5.3 Coûts one-shot

| Item | Coût |
|---|---:|
| Avocat fintech FR (CGU + Privacy + DPA template B2B) | 3-5 k€ |
| RC Pro + Cyber assurance an 1 | 3-5 k€ |
| Adhésion médiation conso (CM2C/MEDICYS) | 150 €/an |
| Tailwind UI kit | $149 one-shot |
| Freelance design 15-30 h (Cat 20) | $1 350-3 600 |
| Pine Script Premium TV (optionnel) | $0-30/an |
| **Total one-shot Year 1** | **~7-12 k€ + $1 500-3 700** |

### 5.4 Cash flow Year 1

- **Capital initial** : ~$25 k recommandé (5 k$ minimum si bootstrap dur, eval_28 §0).
- **Burn fixe pré-launch (S1-S16)** : ~$200/mo × 4 mois = $800 + $1 500 one-shot design + €7-12 k légal = **~$9-15 k**.
- **Runway 13.9 mois** estimé par eval_24 §12.4 à partir de $5 k → ce master plan augmente fixe ⇒ runway recalculé ~10-12 mois.
- **MRR cible M6 (S20-S24)** : $1.5-3 k.
- **MRR cible M12** : $5-7 k base case (eval_28 §9.2 robust scenario à 70 % du BP).

---

## 6. KPIs GO/NO-GO Phasés

### Checkpoint S4 (fin Foundation + Data kickoff)

| Métrique | Cible | Source |
|---|---|---|
| Staging Fly.io répond `/health` 200 | ✅ | Cat 19 |
| CI bloque PR si coverage < 75 % | ✅ | Cat 13 |
| Trading Economics calendar live ingested | ✅ ≥ 100 events/sem | Cat 15 |
| Avocat engagé NDA signé | ✅ | Cat 18 |
| Redis branché (rate-limit + cache) | ✅ | Cat 17 |
| **Décision** | **GO continuer / NO-GO = revoir effort allocation** | |

### Checkpoint S8 (DÉCISION MAJEURE GO/PIVOT/KILL)

| Métrique | GO B2C+B2B | PIVOT B2B-only | KILL projet |
|---|---|---|---|
| Brier skill score scoring v2 | ≥ +5 % | +2 à +5 % | < +2 % |
| Narrative quality eval (rubric 5 pts) | ≥ 4.0/5 | 3.5-4.0/5 | < 3.5/5 |
| Sweep state machine ≥ 1 cellule passe gates | ✅ | partial | ❌ |
| Latence HAR p99 | < 60 ms | < 100 ms | > 200 ms |
| Coût LLM/narrative à projection 100 MAU | < $0.020 | < $0.030 | > $0.050 |
| Telegram public channel ouvert ≥ 4 sem | ✅ | ✅ | optional |
| Décision | **continue plan S9-S16** | **focus B2B + Telegram FREE** | **wind-down 2 sem** |

### Checkpoint S12 (Pre-launch readiness)

| Métrique | Cible |
|---|---|
| Checkout Stripe mode TEST fonctionnel end-to-end | ✅ |
| `/track-record` publique avec ≥ 30 trades forward | ✅ |
| Risk score affiché Telegram + Webapp | ✅ |
| Walk-forward XAU+EUR PF lower-bound CI95 | XAU ≥ 1.0 (paper), EUR documenté |
| Geo-block US/QC/UK testé via VPN | ✅ |
| DPA Anthropic signé + DPA Stripe verified | ✅ |
| **Décision** | **GO launch S15** |

### Checkpoint S16 (Soft launch)

| Métrique | Cible |
|---|---|
| Bêta privée FR ≥ 50 signups | ✅ |
| Telegram public ≥ 60j de tape forward | ✅ |
| Premier payant $29 STARTER | optimiste : 1-5 ; réaliste : 0-2 |
| Cookie banner CNIL-compliant | ✅ |
| DSAR endpoints `/me/data` `/me` opérationnels | ✅ |
| CGU/Privacy signés par avocat publiés | ✅ |
| Mean Time To Recovery (MTTR) incident | < 30 min |
| **Décision** | **public soft-launch S17 si tous ✅** |

---

## 7. Roadmap Post-Launch (M4-M12)

Très haut niveau, basé sur P1/P2 des 20 plans + `memory/roadmap_2026_2027.md`.

### M4-M6 (S17-S24)

- **Cat 1** : Product Hunt launch M5, AMA r/Forex, referral M5, premier paid spend $200/mo Google Ads M6.
- **Cat 2** : P1-1 RSI Divergence fix, P1-3 Numba parity test, P1-5 pipeline incrémental, P1-6 cross-instrument validation EUR/USOIL.
- **Cat 14** : USOIL preset onboarding (G4 P1, 12 h).
- **Cat 20** : B2B portal MVP (clés API, usage, webhooks docs).
- **Cat 12** : Grafana dashboards business + tech provisioned.
- **Cat 18** : SOC2 Type 1 prep démarré.
- **Cat 4** : LGBM ONNX quantization (P0 si latence prioritaire) si VOL_MODE=lgbm justifiable post-S16.

### M6-M9

- **Cat 11** : webhook B2B publisher production-grade + premier broker partner contract.
- **Cat 16** : A/B testing state machine config in-prod.
- **Cat 6** : Monte Carlo P(DD>X), P(ruin) pour marketing risque chiffré.
- **Cat 19** : MLOps Level 3 (canary deploy + drift monitoring + auto-retrain).
- **Cat 1** : Discord privé paid-only, newsletter Substack, YouTube channel.

### M9-M12

- **Cat 18** : SOC2 Type 1 audit terminé (10-20 k€).
- **Cat 1** : Influencer Trader Pro FR partenariat M6 / podcast inbound.
- **Cat 5** : data licensing OHLCV complet (Polygon prod + backup Tiingo).
- **Cat 11** : push mobile (FCM/APNs) + in-app inbox.
- **Cat 20** : mobile app native si demand justifiée.
- **Eventuelle reprise Vision A** : seulement si forward-test 90j en mode B montre PF rolling > 1.30 (improbable, ~5-10 % proba).

---

## 8. Risques Stratégiques Top 10 (Cross-Catégorie)

| # | Risque | Probabilité | Impact | Mitigation cross-plan |
|---|---|---|---|---|
| R1 | MiFID II 2024/2811 finfluencer-rule durcie avant launch (mars 2026 entry) | Élevée | Bloquant Stripe | Cat 18 W4 + Cat 9 prompt UE 2024/2811 + `disclosure_mode=qualitative` défaut + footer disclaimers 4 langues + reformulation "signaux"→"analyses". Audit avocat S2-S4. |
| R2 | Anthropic API outage > 4 h | Moyenne | Webapp/Telegram dégradé | Cat 9 cascade Haiku→Sonnet→Opus fallback Template + Cat 8 CircuitBreaker threshold=3 timeout=60s + Cat 11 message dégradé "narrative indisponible, scoring brut affiché". |
| R3 | Provider data outage (TE/Polygon/Dukascopy) cascadé sur scoring | Moyenne | Scanner émet `is_fallback=True` ou stop | Cat 5 multi-provider fallback (P1) + Cat 8 health-check upstream + Cat 12 alerte Discord/Telegram immédiate ops. |
| R4 | Capital cash flow épuisé avant traction | Moyenne | Wind-down forcé | Cat 1 burn-rate $200-400/mo borné + Cat 9 NARRATIVE_MODE=template fallback default si LLM coût > seuil + cash discipline : pas de pub paid avant PF forward > 1.10 et 30 paids. |
| R5 | LuxAlgo / TradingView Copilot release "SMC AI" beta GA M6 = directe concurrence | Élevée | Réduit défensabilité | Cat 1 niche FR-first inattaquable (low SERP density) + Cat 20 différenciation 3-vues + Cat 2 chart renderer SHAP explainability + open-source rubric narrative. |
| R6 | Verdict B narrative-first reproduit verdict A (faiblesse perçue) | Moyenne | Conv $0, churn élevé | Reframe : produit = "honest confidence" pas "PF claim" ; Cat 6 + Cat 18 alignés vocabulaire conformeAMF/MiFID ; pricing $29 (low barrier, low expectation). |
| R7 | Dukascopy/FF cease-and-desist | Faible | Backtest blocked + repu | Cat 5 + Cat 18 : note "internal only" dès S1 + migration Trading Economics live S3-S4 + Polygon OHLCV live P1 ⇒ découplage prod from sources risquées. |
| R8 | Cyber incident (leak `.env`, vol token Telegram) | Moyenne | Spam Telegram + perte credibility | Cat 10 P0 vault Doppler + rotation 90j + Cat 18 RC Pro + cyber assurance + Cat 12 alerting + DSAR purge users si breach. |
| R9 | Burn-out solo founder (effort réel 50+ h/sem) | Élevée | Project stall | Master plan calibré 30 h/sem dev + 8-10 h marketing. Discipline batch dimanche 14-18h marketing inégociable (eval_28 §3.1). Si déficit 200 h S16 ⇒ outsource design + drop optionnel. |
| R10 | Régulateur AMF déclenche enquête "signaux" wording | Moyenne | Bloquant FR-first | Cat 18 W4 cabinet avocat audit copy AVANT publication + Cat 9 prompt UE 2024/2811 hardcoded + reformulation préventive "lecture algorithmique" / "analyse de marché". |

---

## 9. Annexes — Références croisées

### 9.1 Documents source

- 20 plans : `reports/commercialization_sprint/{01..20}_*.md`
- Evals : `reports/eval_{00..29}_*.md` (synthèse maîtresse : eval_00_synthesis.md)
- Verdicts : `memory/a1_verdict_2026_05_01.md`, `memory/mtf_rewiring_2026_05_23.md`, `memory/decision_matrix_2026_04_30.md`
- UX : `memory/feedback_multi_view_ux.md`, `docs/value/best_product_concept.md`
- Compliance : `memory/sprint_w1_compliance_2026_04_29.md`, `BACKTEST_LEGAL_GUARDRAILS.md`
- Roadmap : `memory/roadmap_2026_2027.md`, `reports/roadmap_2026_2027/PLAN_12_MOIS.md`

### 9.2 Gates institutionnels (Cat 6 + Cat 3 cohérents)

- DSR (Deflated Sharpe Ratio) ≥ 1.5 (Bailey-LdP 2014)
- PBO (Probability of Backtest Overfitting) ≤ 0.35
- CPCV PF lower-bound CI 95 % > 1.0
- DM-test p < 0.05 vs baseline
- n_trades ≥ 30
- Brier skill score ≥ +5 % (calibration uniquement, pas edge)
- Reliability diagram monotone

### 9.3 Pricing v1 lock (Cat 1 + Cat 24 + Cat 27 cohérents)

- FREE — FOCUS only, 30 signaux/mo, Telegram public read-only.
- STARTER $29/mo — FOCUS + CO-PILOT, 200 signaux/mo, narrative Haiku.
- PRO $79/mo — + EXPERT mode + chat Sonnet, 800 signaux/mo, dashboard cohort.
- INSTITUTIONAL $1990/mo (book demo) — API JSON Opus on-demand + webhook B2B + 2000 signaux/mo + SLA + custom rubric.
- Annual : -16.7 % (2 mois offerts).
- Trial dual : 14 j sans carte (STARTER unlock) + 14 j avec carte (PRO unlock).

---

**FIN MASTER PLAN.** Lire conjointement `00_DANGEROUS_CHANGES.md` et `00_NO_REGRET_QUICK_WINS.md` avant toute exécution.
