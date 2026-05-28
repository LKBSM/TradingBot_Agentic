# Decision Gate Review V2 — Revue critique du plan d'exécution M.I.A. Markets

**Date** : 2026-05-26
**Auteur** : second instance Claude Code (revue indépendante)
**Plan audité** : `reports/commercialization_sprint/00_DANGEROUS_CHANGES.md` (69 items DG-001 à DG-085)
**Vision de référence** : `docs/value/{client_information_explained.txt, client_relevance_review.md, best_product_concept.md, information_enrichment_recommendations.md}` + démo `mockups/v3/best_concept_demo.html`
**Validation utilisateur** : Partie 1 validée le 2026-05-26 avec ajustements DG-030 DROP, DG-036 DEFER reformulé, DG-071 DEFER condition strict "MRR B2C > $5k 3 mois", DG-078 DROP

---

## Vision produit non-négociable (rappel)

M.I.A. Markets = **indicateur de marché conversationnel** Or + FX, B2C retail large. Architecture en 3 couches :

- **C1 — FOCUS** : un verdict en une phrase + PF historique + alerte event imminent, ≤10s, mobile-first
- **C2 — CO-PILOT** : 6 cartes hiérarchisées + chatbot sidebar permanent, 30-60s
- **C3 — EXPERT** : waterfall 8 composantes + conformal viz + sources RAG, à la demande
- **Chatbot = moat** : définit jargon, décompose conviction, refuse pédagogiquement les ordres
- **Compliance UE 2024/2811** : pas de promesse de gain, `edge_claim=False`, posture éducative

**Règle d'arbitrage** : tout item qui ne sert pas C1/C2/C3, le chatbot, l'acquisition B2C 12 mois ou la compliance non-négociable est suspect.

---

## Partie 1 — Cartographie d'alignement (impitoyable)

**Règles de notation**

- ✅ aligné · 🟡 partiellement · ❌ non aligné
- **KEEP** = critique tel quel
- **MODIFY** = utile mais à reformuler / re-cadrer
- **DEFER** = pas inutile mais ne sert pas la conquête B2C 12 mois — toujours assorti d'une condition de réactivation
- **DROP** = à éliminer sans regret

### 🔴 DESTRUCTIVE (14 items)

| ID | Titre | Aligné | Sert | Verdict | Justification |
|---|---|---|---|---|---|
| DG-001 | Suppression `parallel_training.py` (RL legacy) | ✅ | infra | **KEEP** | Code mort RL qui pourrit le repo et risque PaaS auto-build vers mauvais entry. |
| DG-002 | Suppression `tests/test_long_short_trading.py` | ✅ | infra | **KEEP** | Bloque l'activation de la CI bloquante (DG-035). |
| DG-003 | Drop 6 moteurs risk redondants | 🟡 | infra | **DEFER** (post-DG-039) | Drop avant remplacement = casser des tests sans bénéfice. |
| DG-004 | Suppression feed XAU_15MIN_2019_2025.csv (63%) | ✅ | compliance + C1 (PF) | **KEEP** | Cause root "BOS sur 100% bars". Corrompt le hero PF. Archive, pas delete. |
| DG-005 | Décom Dukascopy + ForexFactory live | 🟡 | compliance | **MODIFY** | Découpler (cf. Partie 3 doublon caché : fusionner avec DG-007, 010, 013). |
| DG-006 | Activation tier rate-limit réelle | ✅ | monetization | **KEEP** | Sans, pas de tiers payants exigibles. |
| DG-007 | Suppression CSV economic_calendar FF | ✅ | compliance | **KEEP** (post-DG-027) | Licence FF interdit usage commercial. |
| DG-008 | Drop `models/scoring_v2.lgb` legacy | ✅ | infra | **KEEP** | Trivial. |
| DG-009 | Drop `test_env_debug.py` racine | ✅ | infra | **KEEP** | Quick win. |
| DG-010 | Suppression `download_economic_calendar.py` | ✅ | compliance | **KEEP** (post-DG-027) | Scraper déprécié. |
| DG-011 | Dédup `data/macro/` vs `data/research/` | ✅ | infra | **KEEP** | Risque divergence silente. |
| DG-012 | Drop `Procfile` + `railway.toml` legacy | ✅ | infra | **KEEP** | Empêche PaaS auto-build legacy. |
| DG-013 | Décom `download_dukascopy_xau.py` prod | ✅ | compliance | **KEEP** (post-DG-076) | Licence Dukascopy zone grise commerciale. |
| DG-014 | Drop `export_mt5_history.py` | ✅ | infra | **KEEP** | Local Windows uniquement. |

**Sous-total 🔴** : KEEP 12 · MODIFY 1 · DEFER 1 · DROP 0

### 🟠 BIG ARCHITECTURAL (20 items)

| ID | Titre | Aligné | Sert | Verdict | Justification + condition réactivation |
|---|---|---|---|---|---|
| DG-020 | Migration cache → Redis multi-worker | 🟡 | infra | **DEFER** | **MAU > 200**. Mono-worker tient. |
| DG-021 | Async I/O end-to-end routes | 🟡 | infra | **DEFER** | **p99 latency observée > 2s sur narratives/chat OU MAU > 500**. |
| DG-022 | Fly.io cdg + Vercel front | ✅ | infra | **KEEP** | Bloquant go-live. |
| DG-023 | Stack frontend Next.js 15 | ✅ | C1/C2/C3 | **KEEP** | Support architecture progressive uniforme + sections collapsibles tier-gated. |
| DG-024 | Multi-worker Gunicorn | 🟡 | infra | **DEFER** | **MAU > 200**. Dépend DG-020. |
| DG-025 | Refonte scoring (Logistic L1 + iso + conformal) | ✅ | C1 + chatbot | **KEEP** | Sans, score Pearson −0.023 cosmétique. |
| DG-026 | Migration SQLite → Postgres | ❌ pré-mature | infra | **DEFER** | **MRR > $5k OU paid subs > 100**. SQLite WAL tient. |
| DG-027 | Souscription Trading Economics $79/mo | ✅ | compliance + C1 event | **KEEP** | Bloquant Stripe live + bannière event ≤4h. |
| DG-028 | Model registry MLflow / S3+manifest | 🟡 | MLops | **DEFER** | **≥ 2 modèles re-trainés en prod**. Manifest JSON suffit V1. |
| DG-029 | Secrets vault Doppler/Vault | 🟡 | compliance sécu | **MODIFY** | Fly.io secrets natifs V1 (gratuit). Doppler DEFER **MRR > $5k OU team > 1**. |
| DG-030 | WebhookPublisher B2B + DLQ + HMAC | ❌ | B2B | **DROP** | Pré-mature B2B. Si pivot B2B confirmé, recréer item à ce moment-là. |
| DG-031 | Queue notifications Redis | 🟡 | delivery | **DEFER** | **> 30 abonnés Telegram simultanés observés**. Dépend DG-020. |
| DG-032 | Versioning schema InsightSignal | ✅ | compliance contrat | **KEEP** | Process écrit, peu d'effort. |
| DG-033 | Stack obs OTel + Tempo/Jaeger + Sentry | 🟡 | infra | **MODIFY** | Sentry KEEP (free essentiel). OTLP/Tempo DEFER **team > 1**. |
| DG-034 | Sweep state machine 432 cellules | ✅ | C1 (PF) | **KEEP** (post-DG-025) | Defaults non empiriques aujourd'hui. |
| DG-035 | CI bloquante 3 workflows | ✅ | infra/qualité | **KEEP** | Mode warn 1 sem → enforce. |
| DG-036 | Migration Pydantic v1 → v2 | 🟡 dette | infra | **DEFER** | **Avant tout refactor touchant un modèle Pydantic v1 OU upgrade Pydantic majeur breaking**. Règle boy scout. |
| DG-037 | Pipeline incrémental SMC (P1) | ❌ pré-mature | perf | **DEFER** | **Latence p99 ressentie > 200ms OU plainte utilisateur**. |
| DG-038 | DSAR endpoints (RGPD) | ✅ | compliance | **KEEP** | Bloquant Stripe live B2C UE. |
| DG-039 | Single RiskManager canonique | 🟡 | C3 + B2B | **MODIFY** | Focus minimal : expose `risk_score 0-100` + `kill_level` dans InsightSignal mode EXPERT. Pas tout unifier. |

**Sous-total 🟠** : KEEP 8 · MODIFY 3 · DEFER 8 · DROP 1

### 🟡 RISKY OPERATIONAL (19 items)

| ID | Titre | Aligné | Sert | Verdict | Justification |
|---|---|---|---|---|---|
| DG-040 | Vision B (narrative-first) défaut | ✅ | direction produit | **KEEP** | Décision écrite anti-réversion. |
| DG-041 | TESTING_MODE=0 défaut prod + gate CI | ✅ | compliance | **KEEP** | Évite data leak. |
| DG-042 | NARRATIVE_MODE=llm défaut tier-routed | ✅ | C2 + chatbot | **KEEP** | Narratif = véhicule des pépites. |
| DG-043 | Stripe live + Customer Portal | ✅ | monetization | **KEEP** | Pré-requis CGU avocat + Tax UE + geo-block. |
| DG-044 | Stripe Tax UE + reverse charge | ✅ | compliance (contrainte) | **KEEP** | Non-négociable fiscal. |
| DG-045 | Geo-block prod US/QC/UK/OFAC | ✅ | compliance (contrainte) | **KEEP** | Non-négociable légal. |
| DG-046 | Hard caps signaux/mois par tier | ✅ | monetization | **KEEP** | Sans, OPEX LLM exploding. |
| DG-047 | MiFID disclosure_mode=qualitative | ✅ | compliance (contrainte) | **KEEP** | Directive finfluencer mars 2026. |
| DG-048 | Cookie banner CNIL Tarteaucitron | ✅ | compliance (contrainte) | **KEEP** | CNIL non-négociable. |
| DG-049 | CircuitBreaker thresholds modif prod | ❌ | infra | **DROP** | Pas un livrable, c'est un process. Reformuler en SOP `docs/runbooks/circuit_breaker_tuning.md`. |
| DG-050 | RC Pro + Cyber assurance | ✅ | compliance (contrainte) | **KEEP** | Price of doing business. |
| DG-051 | CGU/CGV/Privacy v2 avocat fintech FR | ✅ | compliance (contrainte) | **KEEP** | Bloquant Stripe live. |
| DG-052 | Cost monitoring Anthropic + alerte | ✅ | monetization (marge) | **KEEP** | 1 free abusif = $40/mo perte. |
| DG-053 | verify_data_quality boot fail-fast | ✅ | compliance qualité | **KEEP** | Évite feed 63% silencieux. |
| DG-054 | Telegram retry + dedup | ✅ | delivery + C1 FREE | **KEEP** | Canal FREE essentiel. |
| DG-055 | HMAC admin replay nonce-based | ✅ | compliance sécu | **KEEP** | Privilege escalation possible. |
| DG-056 | UNIQUE constraint api_key_id | ✅ | compliance sécu | **KEEP** | Account hijack possible. Trivial. |
| DG-057 | Lire `subscription_expires` au auth | ✅ | monetization | **KEEP** | Revenue leak trivial à fix. |
| DG-058 | RAG pipeline production | 🟡 | C3 + chatbot | **MODIFY** | Découper : (a) 12 papers curés + mini-fiches KEEP S4-6 = DG-058a, (b) BM25+dense+RRF DEFER **churn M1 > 20% OU MAU > 500** = DG-058b. |

**Sous-total 🟡** : KEEP 17 · MODIFY 1 · DEFER 0 · DROP 1

### 🟣 POLITIQUE / MÉTIER (16 items)

| ID | Titre | Aligné | Sert | Verdict | Justification |
|---|---|---|---|---|---|
| DG-070 | Pricing v1 FREE/$29/$79/$1990 | ✅ | monetization | **KEEP** | Grille eval_27 défensable. |
| DG-071 | Pivot B2B-API brokers parallèle | ❌ disperse | monetization B2B | **DEFER strict** | **MRR B2C > $5k 3 mois consécutifs**. Vision B2C = ligne tenue. |
| DG-072 | Track-record public Telegram 60-90j | ✅ | C1 + acquisition | **KEEP** | Sans, hero "PF 1.30" non substanciable. |
| DG-073 | Reformulation "signaux" → "analyses" | ✅ | compliance (contrainte) | **KEEP** | MiFID finfluencer. |
| DG-074 | Instruments GA : XAU+EUR seuls | ✅ | direction produit | **KEEP** | 5/6 presets sans CSV. |
| DG-075 | Avocat fintech FR 3-5k€ | ✅ | compliance (contrainte) | **KEEP** | Lead time RFQ 2-3 sem. Démarrer S1. |
| DG-076 | Data licensing TE + Polygon | 🟡 | compliance + data | **MODIFY** | TE only V1. Polygon DEFER **proof commercial XAU OU MRR > $5k**. |
| DG-077 | Positioning "honest confidence" USP | ✅ | acquisition + chatbot | **KEEP** | Seul angle défendable face à "your A1 failed". |
| DG-078 | Open-source rubric LLM narrative | ❌ | moat long terme | **DROP** | Sans audience pour le voir. Marketing thought-leadership ≠ levier acquisition B2C. |
| DG-079 | Refund 30j first-month | ✅ | monetization | **KEEP** | Standard SaaS + loi Hamon. |
| DG-080 | INSTITUTIONAL = book demo Calendly | ✅ | monetization B2B | **KEEP** | Pas auto-checkout SLA. |
| DG-081 | Kill criterion S8 : Brier < +2% → B2B | ✅ | direction produit | **KEEP** | Discipline anti-rationalisation. |
| DG-082 | Médiation conso CM2C/MEDICYS | ✅ | compliance (contrainte) | **KEEP** | L.612-1 obligatoire B2C FR. |
| DG-083 | Decoy INSTITUTIONAL $1990 permanent | ✅ | monetization (+25-40% conv) | **KEEP** | Pricing psychology. |
| DG-084 | Dual trial 14j sans CB + 14j avec CB | ✅ | monetization | **KEEP** | +$1168 MRR vs freemium-only. |
| DG-085 | B2B INSTITUTIONAL contrat 12 mois | ✅ | monetization B2B | **KEEP** (post-DG-071) | $23,880 ARR/contrat. |

**Sous-total 🟣** : KEEP 13 · MODIFY 1 · DEFER 1 · DROP 1

### Distribution globale finale — sur plan original

| Verdict | 🔴 | 🟠 | 🟡 | 🟣 | **Total** | **%** |
|---|---|---|---|---|---|---|
| **KEEP** | 12 | 8 | 17 | 13 | **50** | **72.5%** |
| **MODIFY** | 1 | 3 | 1 | 1 | **6** | **8.7%** |
| **DEFER** | 1 | 8 | 0 | 1 | **10** | **14.5%** |
| **DROP** | 0 | 1 | 1 | 1 | **3** | **4.3%** |
| Total | 14 | 20 | 19 | 16 | **69** | **100%** |

**Items ajoutés (DG-100+)** : 28 dont **DG-100 DROP** (toggle 3 modes abandonné post-validation 2026-05-26). **27 items ajoutés effectifs**, dont **10 en P0-strict-MVP V1**.

---

## Partie 2 — Les angles morts du plan existant

Le plan d'origine raisonne en "ingénieur qui veut tout terminer", pas en "produit qui doit conquérir un marché B2C". Il manque **28 items critiques** organisés en 7 angles morts. Je les numérote DG-100+.

### 🚨 Angle mort #1 — L'architecture progressive uniforme + hiérarchie de contenu (5 items manquants) — RÉVISÉ 2026-05-26

**Constat** : la vision produit la plus structurante (3 couches d'information, démontrée dans `mockups/v3/best_concept_demo.html`) n'a **aucun item dédié** dans le plan d'exécution. C'est l'absence la plus grave.

**Décision UX** (révision 2026-05-26 post-validation utilisateur) : abandon du toggle FOCUS/CO-PILOT/EXPERT au profit d'une **architecture progressive uniforme**. Le toggle forçait le client à choisir 3 concepts avant la valeur. La nouvelle architecture : un seul layout responsive avec hero card permanent + sections collapsibles dépliables au clic + gating tier par disponibilité de contenu (pas par layout).

| ID | Titre | Cat | Priorité | Justification |
|---|---|---|---|---|
| ~~DG-100~~ | ~~Implémentation toggle 3 modes~~ | — | **DROP** | Pivot architecture progressive uniforme (cf. décision 2026-05-26). 4e DROP du plan. |
| **DG-101-MODIFIED** | Renderer unique CO-PILOT + sections collapsibles tier-gated | 🟠 | **P0-strict** | Un seul layout, sections additionnelles (waterfall, conformal viz, sources RAG) gated par tier. Économie ~30-40h dev vs 3 layouts. |
| DG-102 | Table `user_preferences` (langue, tier, watchlist) | 🟠 | **DEFER** | Sans toggle, persistance pref moins critique. DEFER MAU > 500. |
| **DG-103** | Mobile-first responsive (<768px) | 🟠 | **P0-strict** | 60-70% retail = mobile. Sans, bounce massif. Effort ~16h. |
| **DG-104** | Email digest format compact (lien webapp pour détail) | 🟡 | **P1** | Cohérent avec récap multi-surfaces. Effort ~6h. |

### 🚨 Angle mort #2 — Le chatbot comme pilier wiring (5 items manquants)

**Constat** : le plan parle de NARRATIVE_MODE=llm (DG-042) mais **aucun item ne câble le chatbot sur la richesse algo** (8 composantes, conformal, BOCPD, jump ratio, stats J.*). Le moat #1 est invisible dans le plan d'exécution.

| ID | Titre | Cat | Priorité | Justification |
|---|---|---|---|---|
| **DG-110** | Wire chatbot context-injection 8 composantes ("Pourquoi 72 ?") | 🟠 | **P0** | Réponse pédagogique structurée des contributions BOS/FVG/OB/etc. Effort ~20-30h (P0.4 Livrable 4). |
| **DG-111** | Wire chatbot Q&A conformal + stats J.* ("Quelle marge d'erreur ?") | 🟠 | **P0** | Traduire ACI Gibbs-Candès en langage humain. Effort ~12-18h. |
| **DG-112** | Tests adversariaux refus pédagogique ("Dois-je acheter ?") | 🟠 | **P0** | Compliance UE 2024/2811 incarnée + différenciation anti-finfluenceur. Effort ~6-10h (P0.8 Livrable 4). |
| **DG-113** | Métriques engagement chat (questions/session, satisfaction, abandon) | 🟡 | **P1** | Sans, on ne sait pas si le moat fonctionne. Effort ~8-12h. |
| **DG-114** | 6 questions suggérées contextuelles par scénario | 🟡 | **P0** | Démontre la valeur chatbot en <5s landing. Effort ~10-14h (cf. démo HTML section 3). |

### 🚨 Angle mort #3 — Onboarding <10s et hook commercial (3 items manquants)

**Constat** : pas un seul item sur le hook commercial visible immédiatement (Problème #5 du Livrable 1). Le plan suppose que le prospect "découvrira la valeur" — c'est faux, il bounce en 10s.

| ID | Titre | Cat | Priorité | Justification |
|---|---|---|---|---|
| **DG-120** | Landing page hero card "track-record honnête" permanent | 🟠 | **P0** | *« 329 setups · 1.30 [1.12-1.49] · walk-forward 7 ans »* en hero. Effort ~8-12h (P0.1 Livrable 4). |
| **DG-121** | Onboarding 4-step contextuel première connexion | 🟡 | **P1** | Activation rate trial→paid ×2-3 (benchmark SaaS). Effort ~16-22h (P1.8 Livrable 4). |
| **DG-122** | Bannière event imminent ≤4h avec chronomètre live | 🟠 | **P0** | Pépite calendar visible (eval pépite hero conditionnel). Effort ~12-16h (P0.6 Livrable 4). |

### 🚨 Angle mort #4 — Funnel d'acquisition et conversion (4 items manquants)

**Constat** : le plan a Stripe live (DG-043) et trials (DG-084) mais **aucun item sur le funnel** entre landing visit et paiement. C'est l'angle mort growth.

| ID | Titre | Cat | Priorité | Justification |
|---|---|---|---|---|
| **DG-130** | Funnel analytique Plausible (visits → signup → trial → conv) | 🟡 | **P1** | Sans, on ne peut pas optimiser. Effort ~8-12h. |
| **DG-131** | Email automation cycle de vie (welcome / D3 / D7 / D13 trial end) | 🟡 | **P1** | Rétention M3 +20-30% (benchmark SaaS). Effort ~18-26h (P1.9 Livrable 4). |
| **DG-132** | Page pricing avec decoy $1990 + dual trial visible | 🟠 | **P0** | Concrétise DG-070 + DG-083 + DG-084. Effort ~10-16h. |
| **DG-133** | Tooltips définitions au survol sur termes résiduels | 🟡 | **P0** | Friction cognitive (P0.7 Livrable 4). Effort ~10-14h. |

### 🚨 Angle mort #5 — Preuve sociale et rétention (3 items manquants)

**Constat** : DG-072 (track-record Telegram) existe mais c'est insuffisant. Manque témoignages, cas d'usage, performance publique mensuelle. Sans, conversion = exclusivement froide.

| ID | Titre | Cat | Priorité | Justification |
|---|---|---|---|---|
| **DG-140** | 10 témoignages clients pilote (FREE users → quote + photo) | 🟣 | **P1** | Social proof B2C. Acquisition cold→warm. Effort ~12h hors-code. |
| **DG-141** | 3-5 cas d'usage publiés (formats "comment j'utilise Sentinel") | 🟣 | **P1** | SEO + crédibilité. Effort ~16h hors-code. |
| **DG-142** | Tableau performance public mensuel (PnL paper agrégé, mis à jour J+5) | 🟣 | **P0** | Concrétise DG-072 + DG-077 honest confidence. Effort ~14-20h. |

### 🚨 Angle mort #6 — Pédagogie EXPERT (4 items manquants — P1.1 à P1.4 du Livrable 4)

**Constat** : le mode EXPERT décrit dans le Livrable 2 a **aucun item plan**. Pourtant c'est ce qui justifie le tier STRATEGIST $79.

| ID | Titre | Cat | Priorité | Justification |
|---|---|---|---|---|
| **DG-170** | Waterfall pédagogique 8 composantes (hover explicatif) | 🟠 | **P1** | Anti-blackbox visible. Effort ~18-24h (P1.1 Livrable 4). |
| **DG-171** | Conformal interval visualisé (bande + point + ticks) | 🟠 | **P1** | Différenciation technique majeure. Effort ~14-18h (P1.2 Livrable 4). |
| **DG-172** | Stats J.* traduites (win rate, drawdown, exposure time, skew) | 🟠 | **P1** | Sophistication perçue. Effort ~16-22h (P1.3 Livrable 4). |
| **DG-173** | Tracker validité live "Lecture valable encore 2h47" | 🟠 | **P1** | Urgence honnête (A.3 valid_until_utc exploité). Effort ~12-16h (P1.4 Livrable 4). |

### 🚨 Angle mort #7 — Analytique produit (4 items manquants) — RÉVISÉ 2026-05-26

**Constat** : sans Plausible + event tracking dès la Vague 1, **toutes mes conditions de réactivation DEFER sont inobservables = DEFER devient DROP par défaut**. Cette faille d'architecture du plan a été corrigée le 2026-05-26 : Plausible + event tracking core deviennent **P0-strict-MVP V1**, pas V2.

**Métrique → outil de mesure** :
- MRR > $5k → Stripe natif ✅ (pas de besoin custom)
- p99 latency > 2s → Sentry ✅ (KEEP dans DG-033-MODIFIED)
- MAU > 200, churn > 20%, engagement chat → **Plausible + event tracking REQUIS V1**

| ID | Titre | Cat | Priorité | Justification |
|---|---|---|---|---|
| **DG-160** | Plausible self-hosted privacy-first (CNIL compatible, no cookie banner) | 🟡 | **P0-strict V1** | Sans, DEFER inopérants. Self-hosted EU = pas de PII, pas de cookies tiers, CNIL compliant. Effort ~6h. |
| **DG-161** | Event tracking core (6 events : signal_view, chatbot_question, section_expanded, upgrade_clicked, signup, paid_conversion) | 🟡 | **P0-strict V1** | Sans events, ni funnel ni rétention mesurables. Effort ~10-14h. |
| DG-162 | Cohort analysis rétention M1/M3/M6 dashboard | 🟡 | **V2** | Visualisation, pas collecte. Reste V2. Effort ~8-12h. |
| DG-163 | Historique 50 dernières lectures user + PnL paper individuel | 🟡 | **V2** | Visualisation user-facing, pas critique 1er paiement. Reste V2. Effort ~22-30h. |

### Synthèse Partie 2 — RÉVISÉ 2026-05-26

**28 items manquants** identifiés. Après filtre strict "qu'est-ce qui doit exister pour qu'un premier client paie en confiance" + correction analytique en Vague 1 :

**P0-strict-MVP : 10 items** (vs 13 P0 brut initialement annoncés)

| # | ID | Titre | Effort |
|---|---|---|---|
| 1 | DG-101-MODIFIED | Renderer unique + sections collapsibles tier-gated | ~16-24h |
| 2 | DG-103 | Mobile-first responsive | ~16h |
| 3 | DG-110 | Wire chatbot 8 composantes | ~20-30h |
| 4 | DG-112 | Tests adversariaux refus pédagogique | ~6-10h |
| 5 | DG-114-REDUCED | 3 questions suggérées contextuelles | ~6-8h |
| 6 | DG-120 | Landing hero card track-record | ~8-12h |
| 7 | DG-132 | Page pricing avec decoy + dual trial | ~10-16h |
| 8 | DG-142 | Tableau performance public (forme simple) | ~14-20h |
| 9 | DG-160 | Plausible self-hosted | ~6h |
| 10 | DG-161 | Event tracking core 6 events | ~10-14h |
| **TOTAL P0-strict-MVP** | | | **~112-160h** |

**5 items reportés en Vague 2 (accélérateurs, pas bloqueurs 1er paiement)** :
- DG-102 user_preferences → DEFER MAU > 500
- DG-111 Wire chatbot conformal/J.* → V2 (sous-ensemble DG-110)
- DG-122 Bannière event ≤4h → V2 (pépite ≠ bloquante 1er paiement)
- DG-133 Tooltips définitions → V2 (réduit friction ≠ bloquant)
- DG-114 reste, mais réduit de 6 à 3 questions

**1 nouveau DROP** : DG-100 Toggle 3 modes → architecture progressive uniforme préférée. Économie ~30-40h dev.

**Le plan d'origine est gravement incomplet sur la vision produit elle-même.**

---

## Partie 3 — Sur-dimensionnements et erreurs de phasage

5 types de défauts identifiés.

### 3.1 — Pré-maturés (déjà DEFER en Partie 1, rappel)

| ID | Défaut | Correction |
|---|---|---|
| DG-020 | Redis avant <200 MAU | DEFER MAU > 200 |
| DG-021 | Async I/O avant plainte perf | DEFER p99 > 2s OU MAU > 500 |
| DG-024 | Multi-worker avant Redis | DEFER MAU > 200 |
| DG-026 | Postgres avant MAU > 1k | DEFER MRR > $5k OU subs > 100 |
| DG-037 | Pipeline incrémental SMC | DEFER latence > 200ms ressentie |

### 3.2 — Sur-dimensionnés (MODIFY en Partie 1, détail correction)

| ID | Défaut | Correction recommandée |
|---|---|---|
| **DG-028** MLflow ou S3+manifest | MLflow = stack pro complète pour 1 modèle. | Manifest JSON dans Cloudflare R2 ($1/mo). DEFER MLflow ≥ 2 modèles prod re-trainés. |
| **DG-029** Doppler/Vault | Outil équipe pour solo dev. | Fly.io secrets natifs (gratuit, audit basique). DEFER Doppler MRR > $5k. |
| **DG-033** OTel + Tempo/Jaeger + Sentry | Stack distributed tracing pour mono-process. | Sentry seul (free tier). DEFER OTLP/Tempo team > 1. |
| **DG-039** Single RiskManager total (8→1) | Consolider 8 moteurs orphelins = scope creep 40-60h. | Expose `risk_score 0-100` + `kill_level` dans `InsightSignalV2` mode EXPERT. Les orphelins restent en archive. |
| **DG-058** RAG complet BM25+dense+RRF | Pipeline complet 60-80h pour pépite "Sources cliquables" qui peut tenir avec 12 papers curés. | DG-058a = 12 papers + mini-fiches inline (KEEP S4-6, ~20h). DG-058b = pipeline complet (DEFER churn > 20% OU MAU > 500). |
| **DG-076** TE + Polygon | $79+$129 = $208/mo récurrent. | TE only V1. Polygon DEFER proof commercial XAU OU MRR > $5k. Économise $129/mo immédiat. |

### 3.3 — Mal séquencés

| Défaut | Items concernés | Correction |
|---|---|---|
| **DG-003 avant DG-039** | Drop orphelins risk avant consolidation = casser tests sans bénéfice | Inverser : DG-039 d'abord (expose risk_score), DG-003 après (purge orphelins) |
| **DG-035 avant DG-002** | CI bloquante activée avec test cassé = build rouge perpétuel | Inverser : DG-002 (drop test) puis DG-035 (CI bloquante warn 1 sem → enforce) |
| **DG-006 avant DG-046** | Tier rate-limit activé sans hard caps Redis = OPEX LLM exploding | Inverser : DG-046 (hard caps via DB counter SQLite si Redis defer) puis DG-006 |
| **DG-005/007/010/013 avant DG-027** | Cut FF/Dukascopy avant TE = blackout calendar | Plan d'origine ok (étape 5 = post-DG-027), mais fusionner les 4 items (cf. §3.4) |

### 3.4 — Doublons cachés

| Items | Doublon | Correction |
|---|---|---|
| **DG-005 + DG-007 + DG-010 + DG-013** | Tous = "cut FF + cut Dukascopy". 4 items pour 1 action atomique cohérente. | **Fusionner en DG-005-FUSED** : "Migration providers calendar+OHLCV : cut FF live + cut Dukascopy live + delete scrapers + archive backtest CSV". 1 commit, 1 PR. |
| **DG-027 + DG-076** | Se chevauchent : DG-027 souscrit TE, DG-076 décide TE+Polygon. | **Fusionner en DG-027-CONSOLIDATED** : décision politique = TE only V1, action exécutive = souscrire + brancher. |

### 3.5 — Dette légale déguisée en "RISKY OPERATIONAL" ou "POLITIQUE"

**Constat** : 9 items sont présentés comme des décisions, alors qu'ils sont **non négociables juridiquement**. Les classer comme décision = laisser croire qu'on peut les skipper. C'est faux et dangereux.

**Reclassement recommandé : nouvelle catégorie 🔵 CONTRAINTE LÉGALE** (à designer proprement, jamais à droper) :

| ID | Catégorie origine | Reclassement | Pourquoi non-négociable |
|---|---|---|---|
| DG-044 | 🟡 | 🔵 contrainte fiscale | Stripe Tax UE infraction = fermeture compte |
| DG-045 | 🟡 | 🔵 contrainte légale | Geo-block OFAC = obligation US/CA/UK |
| DG-047 | 🟡 | 🔵 contrainte légale | MiFID directive finfluencer mars 2026 |
| DG-048 | 🟡 | 🔵 contrainte légale | CNIL cookie banner obligatoire |
| DG-050 | 🟡 | 🔵 contrainte assurance | RC Pro solo dev = capital perso exposé sinon |
| DG-051 | 🟡 | 🔵 contrainte légale | CGU avocat = bloquant Stripe live |
| DG-073 | 🟣 | 🔵 contrainte légale | MiFID vocabulaire "signaux"→"analyses" |
| DG-075 | 🟣 | 🔵 contrainte légale | Corollaire DG-051 (budget avocat) |
| DG-082 | 🟣 | 🔵 contrainte légale | L.612-1 code conso obligatoire B2C FR |

**Implication** : la "décision" sur ces items n'est pas "le fait-on ?", c'est "**quand et comment le faire proprement**". Aucun de ces items n'est éligible à DROP/DEFER.

### Synthèse Partie 3

- **5 items sur-dimensionnés** → MODIFY avec correction précise économisant ~80-120h dev + $129/mo récurrent
- **4 erreurs de séquençage** identifiées
- **2 fusions de doublons** : DG-005/007/010/013 → DG-005-FUSED · DG-027/076 → DG-027-CONSOLIDATED
- **9 items reclassés en 🔵 contrainte légale** : on ne décide plus de les faire, on décide comment

---

## Partie 4 — Re-séquençage en chemin critique B2C

Trois vagues, alignées sur la conquête B2C (pas la complétude technique). Items hors-vague (DROP) listés en fin.

### 🚀 Vague 1 — MVP commercialisable (semaines 1-6)

**Objectif** : un premier client B2C paie en confiance. Tout ce qui n'est pas indispensable est repoussé.

**Ordre exécution** :

**S1 — Décisions politiques + cleanup**
- DG-040 Vision B confirmation écrite (décision blocker tout l'aval)
- DG-070 Pricing v1 lock écrit
- DG-074 Instruments GA XAU+EUR (drop BTC/US500/JPY/GBP)
- DG-073 Reformulation "signaux"→"analyses" décidée
- DG-075 Avocat fintech RFQ démarré (lead time 2-3 sem)
- DG-001/002/008/009/011/012/014 (cleanup code mort, 7 items quick win parallèle)
- DG-041 TESTING_MODE=0 vérification + gate CI

**S2 — Data quality + infra deploy**
- DG-004 archive feed XAU 63%
- DG-027-CONSOLIDATED Trading Economics souscrit (V1, TE only)
- DG-053 verify_data_quality fail-fast au boot
- DG-022 Fly.io cdg deploy initial
- DG-023 Stack Next.js init
- DG-029-MODIFIED Fly.io secrets natifs (V1 simplifié)
- DG-054 Telegram retry+dedup
- DG-055 HMAC nonce
- DG-056 UNIQUE api_key_id
- DG-057 subscription_expires read

**S3 — Scoring + frontend MVP core**
- DG-025 Refonte scoring (Logistic L1 + iso + conformal)
- **DG-101-MODIFIED Renderer unique + sections collapsibles (P0-strict)**
- **DG-120 Landing hero card track-record (P0-strict)**
- **DG-160 Plausible self-hosted (P0-strict)**
- **DG-161 Event tracking core 6 events (P0-strict)**
- DG-006 Tier rate-limit warn mode (préparation enforce S5)

**S4 — Chatbot pilier + compliance UX**
- DG-042 NARRATIVE_MODE=llm tier-routed
- **DG-110 Wire chatbot 8 composantes (P0-strict)**
- **DG-112 Tests adversariaux refus pédagogique (P0-strict)**
- **DG-114-REDUCED 3 questions suggérées contextuelles (P0-strict)**
- DG-038 DSAR endpoints (RGPD)
- DG-047 MiFID disclosure_mode défaut qualitative
- DG-048 Cookie banner CNIL (à confirmer : si Plausible self-hosted = pas besoin si pas d'autres trackers)
- DG-005-FUSED Migration providers (cut FF/Dukascopy, archive backtest)
- DG-052 Cost monitoring Anthropic + alerte spend
- **DG-103 Mobile-first responsive (P0-strict)**

**S5 — Monetization + lancement**
- DG-046 Hard caps signaux/mois (DB counter SQLite, pas Redis)
- DG-006 Tier rate-limit enforce
- DG-045 Geo-block prod activation front + Stripe
- DG-044 Stripe Tax UE
- DG-050 RC Pro souscrite
- DG-082 Médiation conso CM2C
- **DG-132 Page pricing avec decoy + dual trial (P0-strict)**
- DG-058a 12 papers curés + mini-fiches RAG (KEEP S4-6)
- DG-072 Track-record Telegram public ouverture
- DG-077 Positioning honest confidence implementé landing
- **DG-142 Tableau performance public mensuel forme simple (P0-strict)**
- DG-079 Refund 30j codifié CGU
- DG-080 INSTITUTIONAL Calendly demo
- DG-083 Decoy $1990 visible pricing
- DG-084 Dual trial 14j+14j
- DG-035 CI bloquante mode warn (sera enforce S6)
- DG-032 Schema versioning process écrit

**S6 — Lancement Stripe live**
- DG-051 CGU/CGV/Privacy signées avocat
- DG-043 Stripe live + Customer Portal activation
- DG-035 CI bloquante enforce
- DG-034 Sweep state machine 432 cellules (post-DG-025)

**Gate de sortie Vague 1** :
- ✅ 1er paiement Stripe live encaissé
- ✅ CGU signée par avocat fintech FR
- ✅ Hero card visible mobile + desktop
- ✅ Chatbot répond aux 6 questions types (incl. refus pédagogique)
- ✅ Audit compliance UE 2024/2811 + RGPD passé
- ✅ Architecture progressive uniforme opérationnelle (hero card + sections collapsibles tier-gated)
- ✅ Track-record Telegram public ≥ 30 trades clôturés visibles

### 📈 Vague 2 — Activation acquisition (semaines 7-14)

**Objectif** : passer de "ça fonctionne" à "ça attire et convertit".

**S7-S8 — Onboarding + accélérateurs conversion** (analytique core déjà livrée en V1)
- **DG-162 Cohort analysis dashboard** (visualisation sur events V1)
- **DG-121 Onboarding 4-step** (P1.8)
- **DG-130 Funnel analytique tracking** (basé sur events V1)
- **DG-131 Email automation cycle de vie**
- **DG-122 Bannière event ≤4h** (pépite calendar, repoussé de V1 strict)
- **DG-133 Tooltips définitions** (réduit friction cognitive)
- **DG-111 Wire chatbot conformal + stats J.*** (suite DG-110 livré V1)
- **DG-114-FULL passage de 3 à 6 questions suggérées**
- DG-033-MODIFIED Sentry activé
- DG-113 Métriques engagement chat
- DG-102 user_preferences (langue, watchlist) — si MAU > 100 observé

**S9-S10 — Pédagogie EXPERT**
- **DG-170 Waterfall pédagogique 8 composantes** (P1.1)
- **DG-171 Conformal interval viz** (P1.2)
- **DG-172 Stats J.* traduites** (P1.3)
- **DG-173 Tracker validité live** (P1.4)

**S11-S12 — Transparence + preuve sociale**
- **DG-163 Historique 50 lectures user + PnL paper** (P1.5)
- **DG-140 10 témoignages clients pilote**
- **DG-141 3-5 cas d'usage publiés**

**S13-S14 — Personnalisation + rétention**
- **DG-104 Email digest format FOCUS** (cohérence multi-surfaces)
- DG-081 Kill criterion S8 documenté (anti-rationalisation, déclencheur futur)

**Gate de sortie Vague 2** :
- ✅ Conv landing→trial > 2%
- ✅ Activation trial→paid > 15%
- ✅ Churn M1 < 25%
- ✅ MAU > 100
- ✅ MRR > $500

### 🛡 Vague 3 — Solidification (semaines 15+)

**Objectif** : scaling, features avancées, et **possiblement** pivot B2B selon métriques.

**Conditionnel MAU > 200** :
- DG-020 Redis branché
- DG-024 Multi-worker Gunicorn
- DG-031 Queue notifications Redis

**Conditionnel p99 > 2s OU MAU > 500** :
- DG-021 Async I/O end-to-end

**Conditionnel MRR > $5k OU subs > 100** :
- DG-026 Postgres migration
- DG-029 Doppler/Vault upgrade

**Conditionnel team > 1** :
- DG-033 OTLP + Tempo/Jaeger upgrade

**Conditionnel MRR B2C > $5k 3 mois consécutifs (déclencheur B2B)** :
- DG-071 Pivot B2B-API parallèle
- DG-085 B2B INSTITUTIONAL contrat 12 mois

**Conditionnel churn > 20% OU MAU > 500** :
- DG-058b RAG complet BM25+dense+RRF

**Conditionnel ≥ 2 modèles re-trainés prod** :
- DG-028 MLflow upgrade (si manifest JSON saturé)

**Conditionnel avant refactor Pydantic v1** :
- DG-036 Migration boy scout

**Conditionnel latence ressentie > 200ms** :
- DG-037 Pipeline incrémental SMC

**Conditionnel post-DG-039** :
- DG-003 Drop 6 moteurs risk orphelins

**Conditionnel B2C PMF + audience établie 3+ mois** :
- DG-039 RiskManager full consolidation
- Items P2 Livrable 4 (replay chart, compare actifs, tear sheet PDF, voice, notifications proactives, mode apprentissage, app mobile native React Native)

### Items hors-vague (DROP confirmés) — 4 DROP au total

| ID | Statut | Justification |
|---|---|---|
| DG-030 | DROP | WebhookPublisher B2B = pré-mature, recréer si DG-071 activé |
| DG-049 | DROP | Process opérationnel, reformuler en `docs/runbooks/circuit_breaker_tuning.md` |
| DG-078 | DROP | Open-source rubric sans audience = marketing thought-leadership, pas levier B2C |
| **DG-100** | **DROP (2026-05-26)** | **Toggle 3 modes remplacé par architecture progressive uniforme. Économie ~30-40h dev.** |

### Synthèse Partie 4 — RÉVISÉ 2026-05-26

- **Vague 1 (S1-S6, ~240-280h)** = **40 items** dont **10 nouveaux P0-strict-MVP** (incluant analytique core DG-160+161 pour rendre les DEFER observables) = MVP commercialisable, 1er paiement Stripe live
- **Vague 2 (S7-S14, ~180h)** = **20 items** = funnel mesurable, rétention >75% M1, accélérateurs conversion
- **Vague 3 (S15+, conditionnel)** = **17 items** = scaling + features avancées + pivot B2B optionnel
- **4 DROP confirmés** hors-vague (DG-030, DG-049, DG-078, DG-100)
- **Tous les items DEFER ont une condition de réactivation OBSERVABLE en V1** grâce à Plausible + event tracking core (DG-160 + DG-161) qui sont maintenant P0-strict-MVP

---

## Partie 5 — Les 5 décisions politiques à trancher MAINTENANT

Sur les 16 items 🟣, les 5 qui débloquent le plus le reste et exigent une décision écrite immédiate.

### Décision 1 — **DG-075 + DG-051 : Démarrer RFQ avocat fintech FR (3-5 k€)**

**Enjeu** : sans CGU/CGV/Privacy signées par un avocat fintech FR ≥5 ans pratique, Stripe live (DG-043) est bloqué juridiquement. Lead time minimum 2-3 semaines RFQ + 2-3 semaines relecture = **5-6 semaines sans démarrer = pas de revenue avant S8 au mieux**.

**Option par défaut** : **GO immédiat S1**. Budget 3-5 k€ engagé cette semaine. RFQ 3 cabinets en parallèle, sélection sous 5 jours.

**Si on ne tranche pas** : tout le calendrier Stripe live glisse de 4-6 semaines. Trésorerie pendant le glissement = brûlée.

**Alternative défendable** : utiliser un cabinet alternatif type **DPO-as-a-service** (Privacy seul, ~1500€) si on accepte de prendre CGU/CGV générique sans relecture spécifique. **Non recommandé** — risque fermeture compte Stripe sur litige.

### Décision 2 — **DG-040 : Confirmer Vision B (narrative-first) par écrit**

**Enjeu** : sans décision écrite formelle dans `reports/governance/kill_criteria_board.md`, la tentation de retour Vision A (RL trading bot) reste vivace. Vision A = 320h Phase 2A, Vision B = 320h Phase 2B. **Aller-retour = 6 mois perdus.**

**Option par défaut** : **CONFIRMER Vision B**. Mémo `a1_verdict_2026_05_01.md` est sans appel (DSR=0.000, PBO=0.5, CPCV PF=1.008). Engagement écrit : pas de retry Vision A pendant ≥ 90 jours sauf forward-test PF > 1.30 (proba 5-10%).

**Si on ne tranche pas** : sous pression "pourquoi pas réessayer A1 avec X feature ?", le pivot Vision B perd sa cohérence. Roadmap incohérente = exécution diffuse.

**Alternative défendable** : aucune. A1 a été testé sur 7 ans XAU walk-forward CPCV. Le verdict est statistiquement définitif. La seule question = honneur du verdict.

### Décision 3 — **DG-070 : Lock pricing v1 FREE / $29 / $79 / $1990**

**Enjeu** : sans grille tarifaire fixée, on ne peut pas développer la landing page (DG-132), pas configurer Stripe (DG-043), pas construire le decoy (DG-083). **Tous les items monetization aval sont bloqués.**

**Option par défaut** : **GO grille eval_27** (FREE / STARTER $29 / PRO $79 / INSTITUTIONAL $1990) + annual 16.7% off + dual trial 14j+14j.

**Si on ne tranche pas** : prospects voient un pricing flou = pas de conversion. Concurrence (LuxAlgo $39.95, TradingView $14.95) prend la décision pour nous.

**Alternative défendable** : grille BP actuel ($49/$99/$149). **Non recommandée** — INSTITUTIONAL sous-pricé ×13 selon eval_27 (marge réelle 48%), pas de decoy effect, pas de B2B viable.

### Décision 4 — **DG-074 : Instruments GA = XAU + EUR seuls (drop BTC/US500/JPY/GBP)**

**Enjeu** : 5/6 presets actuellement sans CSV propre (eval_08). Marketing actuel "6 instruments" = mensonge. **Tant qu'on ne tranche pas, on continue à dev pour BTC/US500 et on ment au prospect.**

**Option par défaut** : **GO XAU + EUR seuls** en GA S6. USOIL ajouté post-S16 si data Polygon validée. Drop marketing "6 instruments" partout.

**Si on ne tranche pas** : effort dev dispersé sur 6 actifs, qualité dégradée sur tous. Compliance fragile (claim non substanciable sur 5 actifs).

**Alternative défendable** : garder XAU + EUR + USOIL. **Non recommandée immédiatement** — Polygon $129/mo bloqué par DG-076 MODIFY, USOIL backtest non validé.

### Décision 5 — **DG-073 + DG-077 : Reformulation totale "signaux"→"analyses" + positioning "honest confidence"**

**Enjeu** : MiFID directive finfluencer entre en vigueur mars 2026. Le wording "signal trading" est un mot-clé déclencheur. **Sans reformulation totale (landing, Telegram, narratives, API docs, emails), risque d'amende et de fermeture Stripe.** Et sans positioning "honest confidence", aucun différenciateur défendable face à "your A1 failed".

**Option par défaut** : **GO reformulation totale** + USP "honest confidence" écrit en landing hero secondaire. *« Le seul indicateur qui assume ce qu'il ne sait pas. »*

**Si on ne tranche pas** : compliance fragile (risque amende mars 2026) ET positioning indifférencié (commodité prix vs LuxAlgo).

**Alternative défendable** : reformulation partielle (Telegram + narratives seuls). **Non recommandée** — landing page = première impression, doit être 100% compliance et 100% différenciée.

### Synthèse Partie 5

**Les 5 décisions sont mutuellement bloquantes** : sans D1 (avocat), pas de Stripe live. Sans D2 (Vision B), pas de roadmap. Sans D3 (pricing), pas de landing. Sans D4 (instruments), pas de marketing. Sans D5 (vocabulaire), pas de compliance.

**Toutes les 5 peuvent et doivent être tranchées en moins d'une heure aujourd'hui.** Aucune ne demande de dev. Toutes sont du papier.

---

## Partie 6 — Synthèse pour décision

### Les chiffres — RÉVISÉ 2026-05-26

| Métrique | Valeur |
|---|---|
| Items plan original analysés | **69** |
| KEEP | **50** (72.5%) |
| MODIFY | **6** (8.7%) |
| DEFER | **10** (14.5%) — toutes avec condition de réactivation observable |
| DROP | **3** (4.3%) sur plan original |
| Items manquants identifiés (angles morts) | **28** (DG-100+) dont **DG-100 DROP** post-révision |
| Items ajoutés effectifs | **27** (28 − 1 DROP) |
| Items P0-strict-MVP (bloquants 1er paiement) | **10 / 27** (filtre strict appliqué) |
| Total items dans le plan révisé | **66 + 27 = 93** items (69 original − 3 DROP + 27 ajoutés) |
| **Effort estimé Vague 1 (MVP, S1-S6)** | **~240-280h** dont ~140h dev + ~120h compliance/marketing |
| Effort estimé P0-strict-MVP seul (10 items ajoutés) | **~112-160h** |
| Effort estimé Vague 2 (Activation, S7-S14) | **~180h** |
| **Économie via DROP toggle 3 modes** | **~30-40h dev récupérées** |

### Les 3 items les plus critiques du plan original (sans lesquels rien ne marche)

1. **DG-025 — Refonte scoring (Logistic L1 + isotonic + conformal)**
   Sans, score reste Pearson −0.023 = cosmétique. Hero PF non substanciable. Pépite C.1 du Livrable 1 invisible. Tier classifications fausses. = produit indéfendable au prix premium.

2. **DG-027-CONSOLIDATED — Souscription Trading Economics ($79/mo)**
   Bloquant Stripe live (licence FF/Dukascopy zone grise commerciale). Bloquant bannière event ≤4h (pépite H.1+H.2). Bloquant calendar live en prod. 1 souscription = 3 déblocages.

3. **DG-051 — CGU/CGV/Privacy v2 signées avocat fintech FR**
   Sans, Stripe live impossible. Sans Stripe live, pas de revenue. Lead time RFQ + relecture = 5-6 semaines = chemin critique le plus long du plan. **À démarrer aujourd'hui.**

### Les 3 items les plus inutiles à dropper sans regret

1. **DG-030 — WebhookPublisher B2B + DLQ + HMAC**
   B2B est Plan B (DG-071 lui-même DEFER MRR B2C > $5k). Construire le webhook avant le client = feature pour un produit inexistant. Recréer si pivot B2B activé.

2. **DG-078 — Open-source rubric LLM narrative**
   Marketing thought-leadership avant audience. Sans 1000+ followers techniques, le repo open-source = bouteille à la mer. Réactiver post-PMF B2C + rubric stable 3+ mois.

3. **DG-049 — CircuitBreaker thresholds modification prod**
   N'est pas un livrable, c'est un Standard Operating Procedure continu. Reformuler en `docs/runbooks/circuit_breaker_tuning.md`. Pas un decision-gate.

### Les 3 items manquants les plus graves (à ajouter sans discussion) — RÉVISÉ 2026-05-26

1. **DG-101-MODIFIED — Renderer unique + sections collapsibles tier-gated**
   La hiérarchie de contenu (hero + sections dépliables, gating par tier) est **absente du plan d'exécution**. Sans, vision = mockup, plan = autre produit. Architecture progressive uniforme préférée au toggle 3 modes (DG-100 DROP).

2. **DG-110 + DG-112 — Wire chatbot 8 composantes + refus pédagogique**
   Le moat #1 (chatbot = pilier) n'a aucun item wiring dans le plan. Sans, chatbot = wrapper LLM générique, pas l'assistant conversationnel M.I.A. décrit dans Livrable 2.

3. **DG-103 + DG-160 + DG-161 — Mobile-first + analytique produit V1**
   60-70% du retail = mobile, plan l'ignore. **Plausible + event tracking core sont P0-strict V1** car sans, tous les DEFER deviennent invisibles donc inopérants.

### Si tu ne devais faire qu'UNE chose dans les 30 prochains jours

> **Tranche les 5 décisions politiques de la Partie 5 cette semaine** (Vision B + pricing + instruments + vocabulaire + démarrage RFQ avocat). Aucune ne demande de dev. Toutes débloquent la roadmap entière. Le code peut attendre, ces décisions ne le peuvent pas.

Concrètement, lundi prochain :
- **Lundi matin** : écris `docs/governance/decisions/2026-05-26_5_political_locks.md` qui acte D1-D5
- **Lundi après-midi** : envoie 3 RFQ avocat fintech FR (Lexing, Hashtag Avocats, Couvrelles & Marchand-Berdat — ou cabinets équivalents)
- **Mardi-mercredi** : commence DG-101-MODIFIED + DG-120 (renderer unique + sections collapsibles + hero card landing) en parallèle de la sélection avocat
- **Jeudi-vendredi** : DG-025 démarre (refonte scoring) — c'est le plus long item algo, 60h+ d'effort, à kicker tôt

**À S+4, tu auras** : décisions politiques actées + landing visible + scoring refait + avocat sélectionné = chemin critique courant. À S+6, premier Stripe live encaissé est atteignable.

---

## Annexe — Récap exécutif — RÉVISÉ 2026-05-26

**Verdict global** : le plan original (69 items) est **techniquement compétent mais commercialement myope**. Il optimise pour la complétude technique d'un système, pas pour la conquête B2C d'un marché. **72.5% des items sont à garder**, **27 items manquants** (dont **10 P0-strict-MVP**) doivent être ajoutés. La pondération du plan révisé se déplace de **infrastructure → produit + chatbot + mobile + analytique core V1**.

**Trois corrections structurantes apportées suite à validation utilisateur 2026-05-26** :

1. **Toggle 3 modes (DG-100) → DROP, architecture progressive uniforme préférée**
   Le toggle forçait un choix de mode avant compréhension de la valeur. Remplacé par un layout unique avec sections collapsibles tier-gated. Économie ~30-40h dev. 4 DROP au total.

2. **Filtre P0-strict-MVP : 10 items vs 13 P0 brut**
   Différence entre "doit exister pour qu'un client paie" et "serait bien d'avoir". 5 items reportés en V2 (DG-102 user_pref, DG-111 chatbot conformal, DG-122 bannière event, DG-133 tooltips, DG-114 réduit à 3 questions).

3. **Analytique produit (DG-160 + DG-161) basculée P0-strict V1**
   Sans Plausible + event tracking en V1, tous les DEFER (MAU > 200, churn > 20%, engagement chat) deviennent invisibles donc inopérants. C'était une faille d'architecture du plan révisé v1. Effort additionnel ~16-20h pour rendre observables 10 DEFER.

**Conditions de succès du plan révisé v2** :
1. Les 5 décisions politiques sont tranchées **cette semaine**
2. **Plausible + event tracking core sont livrés en V1** (S2-S3) — sans, conditions DEFER mortes
3. Le chatbot wiring (DG-110 + DG-112 + DG-114-REDUCED) est livré **avant** Stripe live — le moat doit fonctionner au jour 1
4. Le mobile-first responsive (DG-103) est livré **avant** la première campagne d'acquisition — 60-70% du trafic sera mobile
5. L'architecture progressive uniforme remplace officiellement le toggle 3 modes dans toute la documentation (mockup HTML à mettre à jour)

**Ce que ce rapport ne dit pas** : le verdict empirique de A1 (Vision A = morte) reste valide. M.I.A. Markets vend de la **compréhension augmentée**, pas du profit. Toute dérive vers une promesse de gain = ré-ouverture du procès A1 + risque MiFID. La discipline narrative-first n'est pas un slogan, c'est la condition de survie commerciale du produit.

**FIN DU RAPPORT.**
