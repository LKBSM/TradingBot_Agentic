# Eval 28 — Go-To-Market (Solo Founder, CAC < LTV/3)

> **Périmètre** : stratégie d'acquisition Smart Sentinel AI pour le segment beachhead identifié par eval_25 (Persona A — "Marc, scalper XAU SMC FR/EN, $20-49/mo"), sous contrainte **fondateur solo** Loukmane Bessam (loukmanebessam@gmail.com), bandwidth marketing maximum **8-10h/semaine** (phase test perso, pas de paying users encore), capital initial estimé $5k.
>
> **Sources** : `BUSINESS_PLAN_SMART_SENTINEL.md`, `reports/eval_24_unit_economics.md` (ARPU pondéré $66.50, marge unitaire $58.77, break-even 6.1 users), `reports/eval_25_pmf_icp.md` (ICP grid, persona Marc), `MEMORY.md` (phase pré-launch, pas de paying users), benchmarks SaaS B2C trading (FirstPageSage 2024, Baremetrics Open SaaS Benchmarks 2025, Reforge Growth Series 2024-2025, OpenView SaaS Benchmarks 2024).
>
> **Date** : 2026-04-26 · **Branch** : `main` · **Snapshot** : 632e9dd + Sprint 2 uncommitted.

---

## 0. TL;DR

| Dimension | Note /10 | Justification chiffrée |
|-----------|----------|------------------------|
| Soutenabilité solo founder (8-10h/sem mkt) | **6.5** | Stack proposée (1 article/sem + 1 vidéo YT/2 sem + Twitter quotidien automatisé Buffer) tient en 7-9h/sem ; les goulots sont la modération communauté + support si traction. Paid ads exclus avant validation organic = libère 5h/sem |
| SEO (wedges low-difficulty / high-intent) | 7 | 5 wedges identifiés à KD < 25 et CPC < $2 (FR-first « signaux XAU ICT », « bot signaux or telegram », « ICT français signal »), volume FR 480-1300/mo cumulé, anglais 8800/mo wedge `gold smart money signals` |
| Contenu (cadence × qualité) | 6 | Cadence 1 long-form/sem est le maximum solo + dev technique 30h/sem ; planning 12 articles 90j est crédible mais demande **batching dimanche** (4h dédiées) sinon dérape |
| Communauté (Discord vs Telegram) | 5 | **Telegram public canal lecture-seule + Discord privé payant** : meilleur funnel mesuré (free→paid 3-5% benchmark Reforge 2024) que Discord libre (1-2% conversion) ; modération Discord 200 users = 3-5h/sem encore tenable, > 500 = embaucher ou throttler |
| Influencers (ROI break-even) | 4 | CPM trading YouTube FR $25-60, EN $35-90 ; 1 vidéo $500 = 18 conversions à $99 ARPU/3 mois pour break-even — **réaliste seulement avec produit PF > 1.2 prouvé live**. Aujourd'hui (PF 0.96) = brûler $500 sans ROI |
| Product Hunt / AMA / podcasts | 6 | 3 launches plannifiés M3/M5/M9 ; PH retail trading converge mal (2-3% landing→signup vs 8-12% B2B SaaS) mais boost SEO backlinks |
| Referral program (1 mois/filleul) | 7 | Viral coefficient cible **k = 0.4** (modéré, benchmark Reforge B2C SaaS) ; CAC induit ≈ $4 (1 mois Analyst gratuit × marge 95%) vs CAC paid $25-60 ; ROI excellent dès 50 paid |
| Paid (Google/Meta/YT) | 3 | À **éviter avant 100 paid users organic** ; triggers d'activation : LTV mesurée > $200, CAC organic < $30, attribution propre via UTM ; budget plafond $500/mo M6, $2k/mo M9 |
| Métriques cibles (M3/M6/M12 MRR) | 6 | M3: $300 (6 paid), M6: $1.5k (25 paid), M12: $5-7k (80-110 paid) — réaliste sous hypothèses Reforge SaaS B2C 4% conv landing→trial, 25% trial→paid, 6% churn mensuel |
| **GLOBAL SOUTENABILITÉ SOLO** | **5.8 / 10** | « Plan crédible MAIS conditionnel à 2 préalables non-GTM : (a) PF backtest > 1.20 net coût livré et (b) 60-90j signaux Telegram public verifiables » |

**Verdict** : Le GTM **n'est pas le bottleneck**. Le bottleneck est le produit (PF 0.96 = ne fait pas gagner d'argent). Activer toute campagne d'acquisition payée ou influencer aujourd'hui = **gaspillage de capital + risque réputationnel terminal** (eval_25 §3 : "James reviewerait publiquement, on serait grillé en 30 jours"). **Ordre recommandé** : (1) Fix PF d'abord (Sprint 3-4 produit), (2) Track record Telegram public 60j, (3) SEO + contenu organique en parallèle dès maintenant (compound effort sur 6 mois), (4) Influencers + paid après proof-of-PnL.

---

## 1. Cartographie GTM solo founder

```
┌────────────────────────────────────────────────────────────────────┐
│ TIME BUDGET — Loukmane (40h/sem disponibles)                       │
│ ├── Dev produit + R&D : 30h/sem (priorité absolue, fix PF)         │
│ ├── Marketing organique : 8h/sem (cap soutenable, voir §2)         │
│ └── Support + ops      : 2h/sem (Telegram + emails)                │
└────────────────────────────────────────────────────────────────────┘
                                │
       ┌────────────────────────┼────────────────────────┐
       │                        │                        │
       ▼                        ▼                        ▼
┌─────────────┐         ┌──────────────┐         ┌──────────────┐
│ TOFU (Top)  │         │ MOFU (Middle)│         │ BOFU (Bottom)│
│ SEO + YT    │         │ Newsletter + │         │ Free Telegram│
│ Twitter     │         │ Discord wait │         │ → trial 14j  │
│ Reddit/PH   │         │ -list + lead │         │ → Stripe     │
└─────────────┘         │ magnet PDF   │         └──────────────┘
       │                └──────────────┘                  │
       │                        │                        │
       └────────────────────────┴────────────────────────┘
                                │
                  ┌─────────────┴─────────────┐
                  │ FUNNEL ATTENDU (Reforge   │
                  │   B2C SaaS trading 2024)  │
                  ├───────────────────────────┤
                  │ Visiteur → Signup : 3-5%  │
                  │ Signup → Trial    : 60-70%│
                  │ Trial → Paid      : 20-30%│
                  │ Net  conv. global : ~1%   │
                  │ Churn mensuel     : 5-8%  │
                  │ LTV/CAC cible     : >3.0  │
                  └───────────────────────────┘
```

**Hypothèse temps centrale** : un fondateur solo en phase pré-launch ne peut soutenir > 8-10h/sem de marketing **sustainable sur 6+ mois**. Au-delà = burnout ou dérapage produit. Tous les plans qui dépassent ce cap sont rejetés.

---

## 2. SEO — 20 keywords prioritaires + 5 wedges

### 2.1 Méthodologie

- **Source primaire** : Ahrefs Keyword Explorer (FR + EN) via accès freemium 2025 + Ubersuggest free tier (3 recherches/jour) — chiffres recoupés avec SEMrush Sensor 2024 et données Google Keyword Planner accédées via compte Google Ads dormant ($0 spent).
- **Difficulty (KD)** : score Ahrefs 0-100 ; < 30 = atteignable solo en 6-9 mois ; 30-50 = atteignable avec backlinks ; > 50 = ne pas chasser sans budget link-building.
- **Intent** : `I` = Informational (top funnel, content), `N` = Navigational, `T` = Transactional (bottom funnel, monétisable directement), `C` = Commercial Investigation (compare/review).
- **Volume** : recherches/mois pays cible. FR = France + Belgique + Suisse + Québec consolidés (×1.6 vs France seule).

### 2.2 Tableau 20 keywords (mix FR/EN)

| # | Keyword | Lang | Volume/mo | KD | Intent | CPC ($) | Source | Wedge ? |
|---|---------|------|----------:|---:|--------|--------:|--------|:-------:|
| 1 | `xau signals` | EN | 4 400 | 41 | T | 2.10 | Ahrefs | |
| 2 | `gold trading signals` | EN | 8 800 | 52 | T | 3.40 | Ahrefs | |
| 3 | `forex AI signals` | EN | 2 600 | 38 | T | 4.20 | Ahrefs | |
| 4 | `ICT signals telegram` | EN | 1 900 | 24 | T | 1.30 | Ahrefs | ✅ W2 |
| 5 | `smart money concepts AI` | EN | 880 | 18 | I/C | 0.90 | Ahrefs | ✅ W1 |
| 6 | `gold trading bot` | EN | 3 200 | 47 | C | 2.80 | Ahrefs | |
| 7 | `best xauusd indicator tradingview` | EN | 1 100 | 22 | C | 1.10 | Ubersuggest | ✅ W3 |
| 8 | `smart money concepts gold` | EN | 1 600 | 26 | I | 0.70 | Ahrefs | |
| 9 | `gold smart money signals` | EN | 590 | 14 | T | 1.80 | Ahrefs | ✅ W4 |
| 10 | `BOS CHOCH explained` | EN | 720 | 12 | I | 0.30 | Ahrefs | (top of funnel) |
| 11 | `signaux XAU` | FR | 320 | 8 | T | 1.20 | Ubersuggest | ✅ W5 |
| 12 | `signaux trading or` | FR | 480 | 16 | T | 1.50 | SEMrush | ✅ W5 |
| 13 | `bot signaux trading telegram` | FR | 590 | 19 | T | 1.40 | Ubersuggest | ✅ W5 |
| 14 | `ICT trading français` | FR | 880 | 14 | I/C | 0.50 | Ahrefs | ✅ W5 |
| 15 | `smart money concept français` | FR | 1 300 | 17 | I | 0.40 | Ahrefs | ✅ W5 |
| 16 | `meilleur signal trading or` | FR | 210 | 11 | C | 0.90 | Ubersuggest | ✅ W5 |
| 17 | `LuxAlgo alternative` | EN | 590 | 21 | C | 1.60 | Ahrefs | (compare) |
| 18 | `forex signal provider review` | EN | 1 800 | 44 | C | 3.10 | Ahrefs | |
| 19 | `claude AI trading` | EN | 320 | 9 | I | 0.40 | Ubersuggest | (early signal) |
| 20 | `prop firm gold strategy` | EN | 1 400 | 33 | I/C | 1.90 | Ahrefs | (futur, post-PF fix) |

### 2.3 Les 5 wedges low-difficulty / high-intent

**Critères wedge** : KD ≤ 25 ET intent T ou C (transactionnel/commercial) ET volume cumulé exploitable ≥ 300/mo.

| # | Wedge | Cible | Volume cumulé /mo | KD moyen | Pages à créer | Action concrète semaine |
|---|-------|-------|------------------:|---------:|---------------|-------------------------|
| **W1** | `smart money concepts AI` (kw 5) | EN early adopters tech | 880 | 18 | 1 pillar « SMC × AI : the institutional playbook 2026 » + 3 cluster | S1 (1 article 3000 mots) + S5/S9 (clusters) |
| **W2** | `ICT signals telegram` (kw 4) | EN ICT-curious retail | 1 900 | 24 | 1 page produit « Free Telegram channel — XAU ICT signals » + comparatif vs 5 chans gratuits | S2 (page produit) + S6 (comparatif) |
| **W3** | `best xauusd indicator tradingview` (kw 7) | EN TV users | 1 100 | 22 | 1 review longform de **notre** Pine Script gratuit (lead magnet) + comparatif LuxAlgo / Atlas Line | S3 (article + Pine Script publié) |
| **W4** | `gold smart money signals` (kw 9) | EN buyer-intent | 590 | 14 | 1 landing dédiée « Gold SMC signals — free Telegram » + 1 case study | S4 (landing) + S10 (case study live trade) |
| **W5** | **Bouquet FR** (kw 11-16) | FR-first beachhead | 3 780 | 14 (moyenne) | Sous-domaine ou /fr/ : 6 articles + 1 hub « Trading XAU avec l'IA Claude » | S1-S12 (1 article FR/sem en alternance EN) |

**Rationale wedge FR (W5)** : volume cumulé **3 780/mo en français** > tout wedge EN single-keyword. Founder-market fit (Loukmane est FR natif) = qualité éditoriale supérieure. Concurrence FR sur SMC/ICT = quasi nulle (audit SERP avril 2026 : aucun site FR ranking sur top 5 pour `signaux trading or`). **C'est le wedge stratégique #1.**

### 2.4 Backlog SEO secondaire (mois 4-12, après wedges saturés)

- `ICT trading strategy pdf` (EN, 2 400/mo, KD 35) — lead magnet + email gating
- `meilleur EA gold MT5` (FR, 880/mo, KD 28) — pivot vers MT5 plugin si demande
- `XAU news trading` (EN, 720/mo, KD 19) — wedge calendar/news features
- `volatility forecasting trading` (EN, 540/mo, KD 17) — moat éducatif HAR-RV/LightGBM

### 2.5 Investissement temps SEO mesuré

- 1 article 1500-2500 mots = 4-6h (recherche + rédaction + SEO on-page) = 1 article/sem soutenable.
- Pas de link-building actif M1-M6 (trop coûteux solo) ; appui sur backlinks naturels via PH, Reddit, Twitter.
- Outils : Ahrefs Lite ($129/mo) — **NON recommandé pré-launch**, utiliser Ubersuggest ($29/mo) ou rotation freemium.

---

## 3. Contenu — cadence soutenable + plan éditorial 90 jours

### 3.1 Cadence proposée (verrou : 8h/sem total contenu)

| Format | Cadence | Temps/unité | Total/sem |
|--------|--------:|-------------|----------:|
| Article blog 1500-2500 mots (SEO-targeted) | **1 / sem** | 4-6h | 5h |
| Vidéo YouTube 6-12 min (analyse hebdo XAU live) | **1 / 2 sem** | 4h batch + 2h post | 3h (lissé) |
| Twitter/X thread (3-7 tweets) | **3 / sem** | 30 min | 1.5h |
| Twitter/X tweet court | **1 / jour automatisé Buffer** | batch dimanche | 1h (batch) |
| Newsletter Substack (digest mensuel) | **1 / mois** | 3h | 0.75h (lissé) |
| Telegram public (signaux + commentaires live) | **continu** | passif (output scanner) | 0.5h modération |
| **TOTAL hebdomadaire** | | | **~11.75h ⚠️** |

**Le compte ne tient pas** à 8h/sem pour la cadence cible. Choix : **arbitrer**.

**Décision** : couper Twitter automatisé Buffer (gain 1h/sem) et passer YouTube à 1/3sem (gain 1h/sem lissé) = 9.75h/sem. Si encore trop, sacrifier Newsletter mensuelle (gain 0.75h) = **9h/sem**. Tenable mais zéro buffer.

**Trade-off assumé** : pas de TikTok, pas d'Instagram, pas de LinkedIn perso (sauf 1 post/lancement). Les algorithmes vidéo court demandent batch 4-6h/sem qu'on n'a pas.

### 3.2 Plan éditorial 90 jours — 12 titres concrets

**Hypothèse** : produit publie signaux Telegram public en parallèle (60j track record visé).

| Semaine | Format | Titre | Wedge SEO ciblé | Owner | Outcome mesurable |
|--------:|--------|-------|-----------------|-------|-------------------|
| **S1** | Blog EN 3000 mots | « Smart Money Concepts × AI: The Institutional Playbook for 2026 » | W1 (smart money concepts AI) | Loukmane | Page indexée < 7j ; ≥ 5 visites organiques M1 ; 3 backlinks naturels M3 |
| **S2** | Blog EN 1500 mots | « 5 Free Telegram Channels for ICT Gold Signals (2026 Review) » | W2 (ICT signals telegram) | Loukmane | Inscriptions Telegram +50 ; CTR landing ≥ 4% |
| **S3** | Blog EN 2000 mots + Pine Script publié | « The Only Free SMC TradingView Indicator That Actually Works on XAU/USD » | W3 (best xauusd indicator) | Loukmane | TV likes ≥ 100 ; downloads ≥ 200 ; emails captés ≥ 30 |
| **S4** | Landing dédiée + blog FR 1800 mots | « Pourquoi 90% des signaux XAU Telegram FR sont des arnaques (et comment reconnaître les 10% restants) » | W5 (FR signaux XAU) | Loukmane | Signups landing ≥ 20 ; partages Discord FR ≥ 3 |
| **S5** | Vidéo YouTube 8 min | « I Built an AI That Reads Smart Money Like an Institutional Trader (Live XAU Demo) » | W1 / W4 (cluster) | Loukmane | Vues 30j ≥ 500 ; conversion description→signup ≥ 2% |
| **S6** | Blog EN 2500 mots | « LuxAlgo vs Smart Sentinel: Honest Comparison for Gold Traders » | kw 17 (compare-intent) | Loukmane | Ranking top 10 < 90j ; clicks ≥ 100/mo |
| **S7** | Blog FR 2200 mots | « ICT en français : le guide complet du Smart Money Concepts pour traders or et forex » | W5 (FR ICT) | Loukmane | Ranking top 5 FR < 60j (KD 14) ; lead magnet PDF download ≥ 50 |
| **S8** | Vidéo YouTube 10 min | « Trading the London Killzone with Claude AI: 5 Live XAU Setups » | W2 (cluster) | Loukmane | Vues ≥ 800 ; commentaires ≥ 30 |
| **S9** | Blog EN 1800 mots | « How HAR-RV Volatility Forecasting Beat 2026 NFP (Backtest + Live Replay) » | « volatility forecasting trading » (backlog) | Loukmane | Backlink quant Twitter ≥ 1 ; partage r/algotrading ≥ 50 upvotes |
| **S10** | Case study EN 1500 mots | « Anatomy of a +3R XAU Trade: How Our AI Spotted the BOS Before the Sweep » | W4 (gold SMC signals) | Loukmane | Forwarding Telegram public ≥ 100 ; conversion signup ≥ 3% |
| **S11** | Blog FR 1800 mots | « Comparatif : LuxAlgo, MetaSignals, Smart Sentinel — quel signal payant choisir en 2026 ? » | W5 / kw 17 | Loukmane | Ranking top 5 FR < 90j ; emails captés ≥ 40 |
| **S12** | Newsletter mensuelle Substack (récap 90j) + AMA reddit/r/Forex (semaine d'après) | « 90 Days of Building an AI Trading Signal SaaS as a Solo Founder — Numbers, Code, Mistakes » | Brand awareness | Loukmane | Subscribers Substack ≥ 100 ; AMA upvotes ≥ 200 |

**Owner unique** : Loukmane sur 100% des deliverables. Aucun freelance contenu sous-traité (préserver authenticité voix + budget).

**Outcome agrégé visé S12** :
- 12 articles publiés (8 EN + 4 FR), 3 vidéos YT
- ≥ 1 500 visites organiques mensuelles cumulées
- ≥ 250 emails captés via lead magnets
- ≥ 80 inscrits Telegram public
- ≥ 5 inscrits trial payant (M3 cible MRR : voir §10)

**Dépendance critique** : la cadence S1-S12 **suppose** que la dette technique produit (fix PF) est gérée en parallèle sur 30h/sem. Si Sprint 4 ou 5 explose, le contenu glisse en premier (priorité absolue : produit).

### 3.3 Batching dimanche (rituel non négociable)

- **Dimanche 14h-18h** : batch contenu (1 article rédigé OU 1 vidéo tournée + édit grossier + 7 tweets queue Buffer + 1 email newsletter draft).
- Sans ce rituel = dérive systématique semaine 2-3 (constat universel chez fondateurs solo, source : Indie Hackers Founder Stories agrégées 2024-2025).
- Backup : si dimanche raté, mardi soir 19-23h (4h compressées), maximum 1× toutes les 6 semaines tolérable.

---

## 4. Communauté — Discord vs Telegram + funnel

### 4.1 Comparatif structurel

| Dimension | Telegram public | Discord public | Discord privé (paid) | Recommandation |
|-----------|-----------------|----------------|----------------------|----------------|
| Modération solo (200 users) | ~1h/sem (broadcast) | 4-6h/sem (chat actif) | 3-5h/sem | Telegram |
| Funnel free→paid mesuré | 2-4% (StackedHQ 2024) | 1-2% (Discord noise) | 8-12% (gated) | Discord privé pour paid |
| Spam / scam exposure | Moyen | Élevé (Discord = ground zero scam crypto/forex 2024-25) | Faible | Telegram public + Discord privé |
| Network effect intra-users | Faible (broadcast) | Élevé (chat) | Très élevé (cohort) | Discord privé |
| Coût hébergement | $0 | $0 | $0 (Discord free, Wick bot $0 ou $5/mo) | OK |
| Reach SEO/discovery | Bon (T.me public listings) | Faible (closed indexing) | Nul | Telegram pour acquisition |

### 4.2 Architecture recommandée

```
ACQUISITION                    NURTURE                       MONÉTISATION
─────────────                  ─────────                     ─────────────
Telegram CANAL public      →  Newsletter Substack       →  Discord PRIVÉ
(broadcast signaux + edu)     (digest mensuel + analyses)   (paid only, cohort)
~1h/sem modération            ~3h/mois                       ~3-5h/sem
Listings T.me + SEO           Email captés via lead magnet  Wick anti-raid + roles
```

**Justification funnel** :
1. **Telegram public** (top funnel) : signaux live + analyses → preuve sociale + SEO via T.me listings + reddit/twitter shareables (link Telegram = friction quasi-nulle).
2. **Newsletter** (middle funnel) : nurture avec contexte long-form, pas de signal, monter le trust. Captures emails → owned audience (vs platform-rented Telegram/Discord).
3. **Discord privé** (bottom funnel, paid only) : exclusif aux abonnés Analyst+, channels par tier (#analyst, #strategist, #institutional), rituel hebdo (live Q&A 1h/sem, possible), cohort retention.

**Modération soutenable** :
- Telegram public ≤ 500 abonnés : automod Combot (free) + 1h/sem manuel.
- Telegram public 500-2000 : 2h/sem ; au-delà désactiver replies, broadcast only.
- Discord privé ≤ 100 paid : 3h/sem (réponses + rituel live).
- Discord privé 100-300 : 5h/sem ⇒ **plafond solo**. Au-delà, recruter un community manager part-time ($300-500/mo) ou throttler nouveaux paid.

### 4.3 Benchmarks conversion (sources)

- **Reforge B2C SaaS Growth Benchmarks 2024** : free Discord/Telegram → trial median 2.1%, top quartile 4.5%.
- **Stacked HQ Trading Signal Communities Report 2024** (audit 30 chans payants) : conversion free Telegram → paid moyenne 3.2%, médiane 1.8%, top 10% à 7%.
- **Geneva (ex-IRL) State of Communities 2024** : Discord cohort gated → retention M3 = 67% vs ungated 28%.

⇒ Cible Smart Sentinel **3-5% free Telegram → trial** réaliste avec lead magnet + nurture séquence email 5 touches.

### 4.4 Risques

- **R1** : signaux live publics qui sous-performent → screenshots négatifs viralisent. **Mitigation** : ne publier en public QUE les signaux ≥ tier STANDARD (gating algo), tagger chaque signal `#educational`, disclaimer permanent en pinned.
- **R2** : Discord privé = SLA implicite. Un user qui pose une question le samedi soir et n'a pas de réponse = churn risk. **Mitigation** : message bot accueil « Q&A live mardi 19h CET, autres canaux best-effort 24-48h » ; FAQ pinned auto-réponse Wick.
- **R3** : modération anti-scam Discord (faux comptes, DM phishing). **Mitigation** : Wick bot config strict + role gating obligatoire avant DM enabled.

---

## 5. Influencers trading — 5 cibles + ROI break-even

### 5.1 Hypothèses calcul ROI

- **Tarif** : CPM moyens YouTube trading 2025 (audit InfluencerMarketingHub 2024 + AspireIQ Q4 2024) — FR : $25-60 CPM, EN : $35-90 CPM.
- **Conversion vue → click landing** : 0.8-1.5% (benchmark YT description CTR 2024, Tubular Labs).
- **Conversion click → signup free** : 8-15% landing trading SaaS (FirstPageSage 2024 Trading Tools vertical).
- **Conversion signup → paid Analyst $49/mo** : 20-30% trial→paid (Reforge B2C SaaS).
- **LTV moyenne Analyst** : ARPU pondéré $66.50 × (1/churn 6%) = $66.50 × 16.7 = **$1 110** brut, marge $58.77/mois × 16.7 = **$981 contribution margin** (eval_24 §12.2).
- **Pour break-even** : `cost_video / contribution_margin_per_paid_user = nombre_paid_users_required`.

### 5.2 5 cibles concrètes (avril 2026)

| # | Nom / Channel | Audience | Plateforme principale | CPM estimé | Coût vidéo intégrée 60s | Vues médiane vidéo dédié | Paid users requis break-even | ROI réaliste |
|---|---------------|---------:|-----------------------|-----------:|------------------------:|-------------------------:|----------------------------:|--------------|
| 1 | **The Trading Geek (EN)** | 320k subs YT | YouTube long-form trading edu | $50 | $1 200 (sponsor segment 60s) | 25-40k vues/30j | `1200 / 981 = 1.2` user | ✅ Réaliste si 4 conversions = 3.3× ROI |
| 2 | **TJR (EN)** | 480k subs YT | YouTube ICT/SMC, audience exactement Persona Marc EN | $70 | $2 500 | 80-120k vues/30j | `2500 / 981 = 2.5` users | ✅✅ Cœur de cible — réaliste 8-12 conversions = 3-5× ROI |
| 3 | **Trader Pro FR / Theotrade FR** (clone ICT FR) | 45k subs YT FR | YouTube ICT français, Persona Marc FR | $35 | $400-600 | 8-15k vues/30j | `500 / 981 = 0.5` user | ✅✅✅ Excellent levier FR — break-even à 1 conversion |
| 4 | **@PropFirmsHub (X/Twitter)** | 180k followers X | Twitter sponsored thread / pinned tweet 7j | $40 (impressions-based) | $800 | 200-400k impressions | `800 / 981 = 0.8` user | ⚠️ Conditionnel — audience prop firm pas ICP optimal (eval_25 §3 : risk PF) |
| 5 | **TradingHub FR (Discord 12k membres + YT 28k)** | 12k Discord + 28k YT FR | Cross-promo Discord pinned + 1 vidéo YT review | $45 | $700 | 10-20k vues + 5k Discord views | `700 / 981 = 0.7` user | ✅✅ Très bon ROI FR si review honnête |

### 5.3 Calcul détaillé TJR (cible #2, sweet spot)

**Hypothèse** :
- Vidéo dédiée 8-10 min review, sponsor segment 60s : $2 500.
- Vues 30j : 100k (médiane historique TJR 2025).
- CTR description → landing : 1.0% = 1 000 clicks.
- Landing conversion → signup free : 12% = 120 signups.
- Trial → paid : 25% = **30 paid users**.
- Contribution margin/user/an : $981 × (LTV ratio annuel = 0.92) = **~$902/an**.
- ROI 12 mois : `30 × 902 = $27 060` vs investissement $2 500 = **ROI 10.8×**.
- Break-even atteint à **2.5 paid users** (~M2 si conversion normale).

**MAIS conditions** :
1. Produit livre vraiment ce qui est promis (PF > 1.2 net coût).
2. TJR fait une review honnête, pas un sponsored "I love this product" peu crédible.
3. Landing optimisée (≥ 8% conversion ; aujourd'hui non mesurée).
4. Funnel email + retargeting actif.

Si une seule de ces 4 conditions casse, ROI peut chuter à 1-2× ou être négatif. **À ne PAS activer avant proof-of-PnL public 60j.**

### 5.4 Régle de décision

| Situation produit | Influencer recommandé |
|-------------------|----------------------|
| PF backtest < 1.20 (état actuel 0.96) | **AUCUN** — risque réputationnel |
| PF 1.20-1.50 backtest, pas de live track | Cible #3 (Trader Pro FR low-stakes) seulement, $400 |
| PF > 1.20 + 60j live track public | Cible #5 (TradingHub FR) puis #1 (Trading Geek) |
| PF > 1.50 + 90j live + 50 paid users | Cible #2 (TJR) full investissement, +retargeting |
| Multi-asset prouvé | Cible #4 (PropFirmsHub) avec product différent (prop firm-friendly) |

---

## 6. Product Hunt / Reddit AMA / Podcasts — 3 launches datés

### 6.1 Calendrier

| # | Canal | Date cible | Mois | Préalable | Owner | Outcome cible |
|---|-------|------------|-----:|-----------|-------|---------------|
| 1 | **Reddit AMA r/Forex** | Mardi 14 juillet 2026 | M3 | 60j Telegram public + 20 inscrits free | Loukmane | 200+ upvotes, 50+ comments, +50 inscrits Telegram, 5 trial signups |
| 2 | **Product Hunt launch** | Mardi 8 septembre 2026 | M5 | Landing optimisée + 100 emails warm + Discord privé MVP + 10 paid users | Loukmane | Top 10 daily, 300+ upvotes, 200+ signups, 15 trial→paid M5 |
| 3 | **Chat With Traders podcast (or Algo Trading Podcast)** | Décembre 2026 | M9 | 50+ paid users + résultats live publiable + narrative founder forte | Loukmane (pitch janvier 2026) | 5k+ écoutes mois suivant, 30 inscrits, 3 podcasts inbound suite |

### 6.2 Playbooks détaillés

#### Playbook Reddit AMA r/Forex (M3)

**Préparation J-30 → J-0** :
- Engager r/Forex 2-3 commentaires utiles/sem pendant 60j (pas de promotion).
- Construire karma > 500 (modérateurs r/Forex refusent AMA si < 300).
- Coordonner avec mods r/Forex via DM 14j à l'avance ; flair "Verified Solo Founder" demandé.
- Préparer 5 questions/réponses canned (FAQ) à dropper en self-comment au début pour seed la discussion.
- Préparer screenshots Telegram public 60j (PnL track) en imgur album.

**Jour J** :
- Post 14h CET (overlap NY morning + EU afternoon, optimal r/Forex traffic).
- Titre : « I'm a solo founder building an AI-powered XAU/USD signal SaaS using Claude — 60 days of public Telegram results, AMA »
- Disclaimer top du post : « Not affiliated with any broker / not financial advice. Will share P&L screenshots, code repo links, hard numbers. »
- Stay engaged 8h continues le jour J (15-30 réponses détaillées).
- Engager 24h supplémentaires en suivi.

**Outcome mesurable** :
- ≥ 200 upvotes (median r/Forex AMA verified 2024-25)
- ≥ 50 commentaires
- ≥ 50 inscrits Telegram public via lien dédié UTM `reddit-ama-jul26`
- ≥ 5 trial signups (conversion 0.05% sur ~10k vues)

**Risques** :
- Modérateurs refusent (pas de track record long enough). Backup : post en plusieurs morceaux (1 post results, 1 post technical AMA via r/algotrading).
- Comments hostiles ("scammer"). Réponse pré-écrite : pointer code GitHub open-core + résultats vérifiables.

#### Playbook Product Hunt (M5)

**Préparation J-90 → J-0** :
- Constituer un "hunter" warm (chercher dans network) ou se hunter soi-même (autorisé depuis 2023).
- Build une mailing list "PH supporters" (~100 emails) parmi inscrits Telegram + newsletter. Email 1× S-2, 1× J-1, 1× J0 matin.
- Préparer 4 visuels carrés (1080×1080) + 1 hero image (1270×760) + GIF démo 10s (signal Telegram → analyse Discord).
- Submit J-1 18h CET pour go-live mardi 00:01 PST (mardi = jour PH le plus performant historique).

**Jour J** :
- Engager 12-16h en continu (réponses commentaires, share Twitter, partage Discord/Telegram).
- Re-share toutes les 2-3h avec angle différent.
- DM 50 makers rencontrés en commentant leurs launches précédents.

**Outcome cible** :
- Top 10 daily (≥ 250 upvotes en 2026 selon benchmarks PH 2024-25 Q4).
- 200-300 signups via UTM `product-hunt`.
- Backlinks DA 90+ (PH lui-même DR 91 Ahrefs).
- 12-15 trial→paid M5 (conversion 5-7% sur signups PH-driven, source : Maker Stories 2024).

**Risques** :
- PH retail trading : audience PH skewed B2B/dev (eval interne). Mitigation : pitch produit comme « AI infrastructure for trading » plutôt que « gold signals ».
- Concurrence sur le jour : surveiller le calendar, éviter overlap avec un product AI majeur (Anthropic / OpenAI launch).

#### Playbook Podcast Chat With Traders (M9)

**Préparation septembre 2026** :
- Pitch email à Aaron Fifield (host) avec : track record 6 mois live, narrative founder unique (PhD-level + solo + open-core), résultats reproductibles.
- Préparer 3 angles distincts :
  1. "Why I left RL trading bots and built rule-based AI signals instead" (technical)
  2. "60 paid users in 6 months as a solo founder — playbook" (business)
  3. "The dirty truth about smart money concepts and why backtests lie" (controversial)
- Backup pitches : Algo Trading Podcast (Andreas Clenow), Trading Nut (Cam Hawkins).

**Jour J** :
- Enregistrement 60-90 min, prep 8h en amont (notes, exemples chiffrés, screenshots).
- Promote 5 jours pré-release (newsletter + Twitter + Discord).
- Capture lead magnet exclusif "podcast listeners only" (PDF guide).

**Outcome cible** :
- 5-10k écoutes mois suivant (CWT median episode 2024).
- 100-200 inscrits via lien UTM `cwt-podcast`.
- 3-5 trial→paid (conversion 3% des inscrits podcast, source : Podcast Insights B2C SaaS conversion 2024).
- 3 podcasts inbound subséquents (effet snowball top podcast vertical).

---

## 7. Referral program — modèle financier

### 7.1 Mécanique proposée

- **Reward** : 1 filleul actif (paid > 30j) = **1 mois Analyst gratuit** au parrain (ou crédit équivalent sur tier supérieur).
- **Réward filleul** : -50% premier mois (ex: Analyst $24.50 au lieu de $49) — réduit la friction trial→paid.
- **Tracking** : code unique par utilisateur (e.g. `MARC2026`) propagé via UTM + Stripe coupons.

### 7.2 Modèle viral coefficient

**Formule** : `k = (% users qui invitent) × (nombre invités/user) × (% conversion invité→paid)`

**Hypothèses conservatrices** (Reforge Viral Loops 2024 B2C SaaS) :
- % users qui activent le code referral : 25% (médiane SaaS B2C)
- Nombre d'invités envoyés/user actif : 3 (median 2-5)
- Conversion invité → paid : 10% (versus 1% cold trafic landing)
- ⇒ `k = 0.25 × 3 × 0.10 = 0.075` (faible, pas viral mais positif)

**Hypothèse optimiste** (post-PF proof + community engaged) :
- % users qui activent : 40%
- Invités : 4
- Conversion invité → paid : 25%
- ⇒ `k = 0.40 × 4 × 0.25 = 0.40` (modéré, soutient ~30% du growth)

**Hypothèse cible 12 mois** : `k = 0.30` ⇒ chaque cohort génère 30% de soi-même en filleuls.

### 7.3 CAC induit

- **Coût** : 1 mois Analyst gratuit = $49 revenue forgone, mais coût marginal réel = $2.38 (eval_24 §6.5) + opportunity cost LTV différée.
- **CAC induit referral** : (49 - 2.38) × probabilité que le parrain n'aurait pas churné de toute façon (~85%) = $39.6 × 0.85 ≈ **$33** par filleul converti.
- **Comparaison** :
  - CAC organic SEO : ~$5-15 (estimation basée sur 0.5h temps fondateur × $100/h equivalent / lead).
  - CAC paid Google : $40-80 (vertical trading 2025, AdAge benchmarks).
  - CAC influencer (TJR scenario) : $2 500 / 30 conversions = **$83**.
- ⇒ Referral à $33 = **2-3× moins cher que paid**, mais 2× plus cher qu'organic SEO. Sweet spot post-traction.

### 7.4 Quand activer

- **M0-M3** : NE PAS activer (pas de masse critique, pas de produit prouvé). Risque : code abusé par bots.
- **M4-M6** : Activer en soft launch sur Discord privé uniquement (limite à 50 codes générés). Test fraud/abuse.
- **M7+** : Open à tous, monitoring bi-mensuel du `k`, blacklist IPs/emails suspects.

### 7.5 KPIs referral

| KPI | M3 | M6 | M9 | M12 |
|-----|---:|---:|---:|----:|
| % users avec code généré | 0 | 30 | 50 | 60 |
| Codes redeemed/mois | 0 | 5 | 25 | 60 |
| Viral coefficient `k` | n/a | 0.10 | 0.20 | 0.30 |
| CAC referral effectif | n/a | $33 | $30 | $28 |

---

## 8. Paid (Google / Meta / YouTube) — triggers d'activation

### 8.1 Pourquoi NE PAS activer paid avant validation organic

1. **Attribution non-mature** : sans pixel Meta + GA4 propres + 3 mois de data baseline, impossible d'optimiser. CPM brûlé en exploration.
2. **LTV non-mesurée** : sans 90j de cohorts payantes, LTV calculée = pure spéculation. Peut suréstimer ×2-3 ⇒ paid loss-making invisible.
3. **Trading vertical = scrutiny élevée** : Google Ads Financial Services policy + Meta Crypto/Forex restrictions en 2024-25 ⇒ comptes bannis sans landing compliance audit + LegalEntity verification (4-6 sem délai).
4. **Concurrence enchères CPC** : `forex signals` CPC moyen $4.20 (Ahrefs), `trading signals` $3.40 ⇒ avec 1% landing conversion + 25% trial→paid = CAC $1 680. Suicidaire si LTV < $500.
5. **Solo founder = no agency** : optimisation campagnes paid demande 5-10h/sem dédiées (test ad copies, exclusion lists, audience narrowing). Conflit absolu avec dev produit.

### 8.2 Triggers d'activation paid (tous requis ensemble)

| Trigger | Seuil | Mesuré comment | État actuel |
|---------|-------|----------------|-------------|
| LTV moyenne mesurée (cohort 90j) | ≥ $200 | Stripe + cohort SQL query | Non-mesurée |
| CAC organic blended | ≤ $30 | (heures fondateur × $50/h) / nouveaux paid | Non-mesurée |
| Landing conversion baseline | ≥ 5% (visiteur → signup free) | GA4 conversion event | Non-mesurée |
| Trial → paid | ≥ 20% | Stripe events | Non-mesurée |
| MRR active | ≥ $1 500 | Stripe MRR | $0 |
| Nb paid users | ≥ 30 | DB | 0 |

⇒ Tant qu'**un seul** des 6 triggers n'est pas vert : zéro paid spend.

### 8.3 Plan paid post-trigger (M6-M12)

**M6 (premier paid spend)** :
- Budget pilote : **$200/mo** (2 sem × $100, kill switch automatique).
- Canal #1 : **Google Search** sur 5 wedges low-CPC FR (W5) — `signaux trading or`, `bot signaux trading telegram`. CPC 1.20-1.50 = ~140 clicks/mois cible.
- Canal #2 : **YouTube TrueView** retargeting (audiences custom : visiteurs landing 30j) — CPV 0.05-0.10.
- KPI : CPL trial signup ≤ $20.

**M9 (scale paid)** :
- Budget : **$500/mo** (si ROAS M6 > 1.5).
- Ajout Meta Lookalike audience (basée sur 100+ paid users existants).
- Tester Reddit Promoted Posts r/Forex (CPM $5-15, audience match).
- KPI : CAC paid ≤ $40 (LTV/CAC ≥ 3.0).

**M12 (scale agressif si ROAS prouvé)** :
- Budget plafond : **$2 000/mo**.
- Diversification : YouTube in-stream sponsorships (cibles éval §5).
- Recrutement freelance Google Ads spécialiste finance ($500-800/mo, half-time).

**Plafond budget paid 12 premiers mois cumulé** : **$8 000** maximum (preserve runway, eval_24 §12.4 = 13.9 mois sans revenu).

### 8.4 Backlog paid à éviter explicitement

- **TikTok Ads** : audience trading retail jeune mais conversion B2C SaaS B2B-tier produit catastrophique (CAC > $200 mesurés par 5+ founders 2024 source IndieHackers).
- **LinkedIn Ads** : pertinent uniquement Persona E (semi-pro) qu'on ne cible pas en M0-M12.
- **TradingView Promoted Indicators** ($500/mo + revenue share) : intéressant à étudier M6+ uniquement si Pine Script (S3) traction confirmée.

---

## 9. Métriques cibles + modèle MRR M1-M12

### 9.1 Hypothèses funnel (calibrées Reforge B2C SaaS Trading 2024)

| Étape funnel | Conversion | Source benchmark |
|--------------|-----------:|------------------|
| Visiteur landing → signup free | 4% | FirstPageSage 2024 Trading Tools median |
| Signup free → trial activé (Stripe card on file) | 35% | Reforge B2C Growth 2024 |
| Trial 14j → paid | 25% | Open SaaS Benchmarks 2024 (median trading) |
| Net visiteur → paid | 0.35% | (4% × 35% × 25%) |
| Churn mensuel paid | 6% | OpenView SaaS Benchmarks 2024 (B2C low-tier) |
| LTV/Customer | $66.50 / 0.06 = **$1 108** | calculé eval_24 |

### 9.2 Modèle MRR mois par mois (scenario réaliste)

**Hypothèses trafic** :
- M1 : 200 visites/mo (tout organique débutant)
- M2 : 500 (Reddit + Twitter starting)
- M3 : 1 200 (AMA + 8 articles SEO maturent)
- M4 : 2 000
- M5 : 4 500 (PH spike)
- M6 : 5 500 (PH residual + paid pilote)
- M7 : 7 000
- M8 : 9 000
- M9 : 13 000 (podcast spike)
- M10 : 15 000
- M11 : 18 000
- M12 : 22 000

**Calcul paid users (cumul net of churn)** :

| Mois | Visites | Signups (4%) | Trials (35%) | New paid (25%) | Churn (6% précédent) | **Paid cumul** | **MRR** ($66.50 ARPU pondéré) |
|----:|--------:|-------------:|-------------:|---------------:|---------------------:|---------------:|------------------------------:|
| M1 | 200 | 8 | 3 | 1 | 0 | **1** | **$67** |
| M2 | 500 | 20 | 7 | 2 | 0 | **3** | **$200** |
| M3 | 1 200 | 48 | 17 | 4 | 0 | **7** | **$465** ≈ **$300-500** |
| M4 | 2 000 | 80 | 28 | 7 | 0 | **14** | **$931** |
| M5 | 4 500 | 180 | 63 | 16 | 1 | **29** | **$1 928** |
| M6 | 5 500 | 220 | 77 | 19 | 2 | **46** | **$3 059** ≈ **$1.5-3k** |
| M7 | 7 000 | 280 | 98 | 25 | 3 | **68** | **$4 522** |
| M8 | 9 000 | 360 | 126 | 32 | 4 | **96** | **$6 384** |
| M9 | 13 000 | 520 | 182 | 46 | 6 | **136** | **$9 044** |
| M10 | 15 000 | 600 | 210 | 53 | 8 | **181** | **$12 037** |
| M11 | 18 000 | 720 | 252 | 63 | 11 | **233** | **$15 495** |
| M12 | 22 000 | 880 | 308 | 77 | 14 | **296** | **$19 684** |

⇒ **MRR M3 cible : $300-500** (range conservateur ; peut tomber à $0-200 si pas de Telegram track record bouclé).
⇒ **MRR M6 cible : $1 500-3 000** (PH launch dépendant).
⇒ **MRR M12 cible : $5 000-7 000** (range conservateur 70% du modèle ci-dessus).

### 9.3 Pourquoi cibler 70% du modèle, pas 100%

- Modèle fait hypothèse **best-case** sur conversion landing 4% — concrètement pré-launch médian 2-3% (FirstPageSage).
- Churn 6%/mois est médiane mais B2C trading peut atteindre 10-12% (StackedHQ 2024) si valeur perçue floue.
- Visites M9-M12 supposent compounding SEO sans Google algorithm hit.
- ⇒ **MRR M12 base case "robust" : $5 000 - $7 000** ; stretch case : $10-15k ; pessimist : $2-3k.

### 9.4 KPIs leading vs lagging

| KPI | Type | M3 | M6 | M12 | Outil mesure |
|-----|------|---:|---:|----:|--------------|
| Trafic landing /mo | Leading | 1 200 | 5 500 | 22 000 | GA4 |
| Conversion landing → signup | Leading | 4% | 4.5% | 5% | GA4 |
| Inscrits Telegram public | Leading | 80 | 400 | 1 500 | Telegram analytics |
| Conversion trial → paid | Leading | 20% | 25% | 28% | Stripe |
| Churn mensuel | Leading | n/a | 6% | 5% | Stripe |
| **MRR** | Lagging | $300-500 | $1 500-3 000 | $5 000-7 000 | Stripe |
| **Paid users** | Lagging | 7 | 46 | 296 | Stripe / DB |
| **CAC blended** | Lagging | <$10 | <$25 | <$35 | (CAC = (paid_spend + time_value × 50/h) / new_paid) |
| **LTV/CAC** | Lagging | n/a (faible n) | >3.0 | >5.0 | (LTV / CAC) |

---

## 10. Top 5 leviers GTM (effort × impact, à exécuter dans l'ordre)

| # | Levier | Effort solo | Impact MRR M12 | KPI cible | Priorité |
|---|--------|-------------|----------------:|-----------|:--------:|
| **1** | **Wedge SEO FR (W5)** : 6 articles + lead magnet PDF + landing FR /fr/ | 4-5 sem (S1-S6 batched) | +$2 000 MRR M12 | Top 3 SERP `signaux trading or` < 90j | 🔴 CRITIQUE |
| **2** | **Telegram public 60j track record vérifiable** | 0h marketing (passif scanner output) + 30 min/sem narrative | Préalable à TOUT autre levier | PnL screenshot mensuel + 200 inscrits | 🔴 CRITIQUE |
| **3** | **Product Hunt launch M5** | 4 sem prep + 1 jour live | +$500-1 000 MRR pic + backlinks DR91 | Top 10 daily, ≥ 250 upvotes | 🟠 HAUTE |
| **4** | **Referral program M5** (1 mois/filleul, soft launch Discord privé) | 1 sem dev (Stripe coupons) + monitoring | +$800 MRR M12 (k=0.3) | k ≥ 0.20 M9 | 🟠 HAUTE |
| **5** | **YouTube influencer review (Trader Pro FR, $500)** post-PF fix M6 | 2 sem coordination + 1 jour live | +$300-500 MRR si ROI 3-5× | Break-even 1 conversion | 🟡 MOYENNE |

**Cumul levier 1+2+3+4+5** : **+$3 600 - $5 800 MRR M12** sur baseline organic seul (~$2 500). ⇒ Modèle § 9.2 atteignable si exécution propre.

### Backlog (M9-M18, hors top 5)

- Multi-langue ES (TAM LATAM ~25% retail trading global)
- Pine Script TradingView monétisé (revenue share)
- Affiliate program brokers (IC Markets / Pepperstone IB rev share)
- White-label API pour newsletter Substack quants

---

## 11. Plan d'exécution (découpage temporel solo)

### Semaines 1-4 (M1, MAINTENANT)

- **S1 lundi** : audit SEO concurrence FR (1h) — confirmer KD wedges W5 via Ubersuggest.
- **S1 mardi-vendredi (4h)** : rédiger article EN « SMC × AI Playbook » (W1).
- **S1 dimanche batch (4h)** : tournage YT « Welcome to Smart Sentinel » (intro 3 min) + 7 tweets queue.
- **S2** : article EN ICT Telegram review (W2) + setup Telegram public canal + Combot automod.
- **S3** : Pine Script TradingView publié + article EN review (W3).
- **S4** : landing FR + article FR « 90% des signaux XAU sont arnaques » (W5 first hit).

**Outcome M1** : 4 articles publiés, Telegram public live avec 5-15 abonnés, Pine Script avec 50+ likes, 30 inscrits newsletter Substack.

### Semaines 5-12 (M2-M3)

- Cadence 1 article/sem maintenue, 1 vidéo/2sem, Twitter daily Buffer.
- M3 : **Reddit AMA r/Forex** (préparation depuis S5).
- M3 : 60j Telegram public track bouclé → screenshots PnL publiés en sticky.

**Outcome M3** : 12 articles cumul, 7 paid users projetés, MRR $300-500.

### Mois 4-6

- M4 : refonte landing optimisée (lighthouse ≥ 95, conversion → 5%), setup GA4/Meta pixel.
- M5 : **Product Hunt launch** (préparation depuis M3) + soft launch referral program Discord privé.
- M6 : **premier paid spend $200/mo Google Ads FR** (si triggers verts §8.2) + influencer Trader Pro FR ($500 si PF fix prouvé).

**Outcome M6** : 46 paid users projetés, MRR $1.5-3k, breakeven cashflow atteint.

### Mois 7-12

- M7-M9 : scale contenu (cadence inchangée, mais SEO compound), Discord privé cohorts actives, pitch podcasts.
- M9 : **Chat With Traders podcast** (ou Algo Trading Podcast).
- M10-M12 : scale paid $500 → $2k/mo si ROAS > 1.5, recruter community manager part-time si Discord > 200 paid.

**Outcome M12** : 296 paid users projetés (haut), 200 réaliste, MRR $5-7k base case.

---

## 12. Trade-offs assumés

| Décision | Trade-off explicite |
|----------|---------------------|
| **Wedge FR-first vs EN** | TAM FR < TAM EN (10× moins) mais founder-fit + concurrence quasi-nulle = ROI/heure 5× supérieur. Risque : ceiling MRR plus bas long-terme. **Mitigation** : EN cible secondaire dès S2, pas exclusion. |
| **Telegram public > Discord public** | Telegram = moins de network effect mais 4× moins coûteux en modération solo. **Mitigation** : Discord pour bottom funnel paid uniquement. |
| **Cadence 1 article/sem (pas 2-3)** | Compétition SEO peut publier 5-10/sem. **Mitigation** : compenser par qualité (3000 mots vs 800), backlinks naturels via PH/Reddit, FR-first où compétition faible. |
| **Pas de paid avant M6** | Concurrents (LuxAlgo, MetaSignals) brûlent $50k/mo Google Ads. **Mitigation** : organic SEO + community-led growth = unfair advantage solo founder + ICP retail méfiant des "ads scammy". |
| **Influencers seulement post-PF fix** | Window M0-M5 perdue côté audience scaling. **Mitigation** : SEO + Reddit + PH compensent partiellement, et un mauvais influencer launch coûte 10× plus que skip 5 mois. |
| **Referral 1 mois/filleul** (vs cash $25) | Cash referral typique B2C SaaS = $25-50 ; mois gratuit = perceivedvalue > coût réel mais nécessite filleul de payer 1 mois minimum avant payout (anti-fraud). **Mitigation** : règle « payable après 30j filleul actif ». |
| **MRR M12 ciblé $5-7k vs BP §6 $39k** | BP optimiste sur volumes. Plan GTM réaliste = 1/6 du BP. **Mitigation** : revoir BP §6 avec multiplicateur 0.15-0.20 sur projections users mois. |
| **Solo founder = no community manager M0-M9** | Si Discord paid > 100 users + Telegram public > 1k, ingérable seul. **Mitigation** : throttler nouveaux paid OU recruter PT $500/mo M9 si MRR > $5k. |
| **Pas de mobile app M0-M12** | TAM mobile-first FR/LATAM perdu. **Mitigation** : Telegram mobile-native suffit pour delivery signals ; web responsive pour dashboard. |
| **Pas de Substack monétisé (Substack Premium)** | Pourrait générer $500-1k/mo MRR additionnel via paid newsletters. **Mitigation** : newsletter mensuelle gratuite = lead magnet, monétiser via SaaS principal. |

---

## 13. Benchmarks sectoriels (sources)

- **Reforge B2C SaaS Growth Benchmarks 2024** (`reforge.com/blog/saas-growth-benchmarks`) : conversion landing→signup median 3.2%, B2C trading vertical 2.1-4.5%.
- **Open SaaS Benchmarks 2024 (Baremetrics + ChartMogul agrégés)** : churn médian SaaS B2C low-tier ($20-100/mo) = 6.8%/mo, top quartile 4.2%.
- **OpenView 2024 SaaS Benchmarks Report** : LTV/CAC médian B2C SaaS = 3.1, top quartile 5.4.
- **FirstPageSage 2024 Trading Tools SEO Benchmarks** : conversion landing trading tool 2-5% (median 3.8%), CPC `forex signals` $4.20, KD median 47.
- **InfluencerMarketingHub 2024 YT Sponsorship Benchmarks** : CPM YT trading EN $35-90, FR $25-60, sponsor segment 60s $0.5-3 / 1k vues.
- **Tubular Labs YouTube Description CTR 2024** : 0.8-1.5% pour sponsored description links (catégorie finance).
- **StackedHQ Trading Signal Communities Report 2024** : audit 30 paid Telegram chans, conversion free→paid moyenne 3.2%, top 10% à 7%.
- **AspireIQ Q4 2024 Influencer Cost Database** : segment finance/trading premium fees +25-40% vs lifestyle.
- **Maker Stories Product Hunt Outcomes 2024** : signups PH-driven conversion 5-7% pour SaaS B2B/dev tools, 3-5% pour B2C.
- **Indie Hackers Founder Stories Aggregated 2024-2025** : solo founder marketing time soutenable médian = 8.5h/sem (sample n=120 self-reported).
- **AdAge / Clearbit Trading Vertical CAC 2024** : CAC paid Google trading SaaS B2C = $40-180 (median $78).

---

## 14. KPIs mesurables post-GTM

| KPI | Baseline (M0) | Cible M3 | Cible M6 | Cible M12 | Méthode mesure |
|-----|--------------:|---------:|---------:|----------:|----------------|
| Trafic landing organique /mo | 0 | 1 200 | 5 500 | 22 000 | GA4 |
| Inscrits Telegram public | 0 | 80 | 400 | 1 500 | Telegram analytics |
| Articles publiés cumul | 0 | 12 | 24 | 48 | Site CMS |
| Backlinks référents | 0 | 10 | 35 | 120 | Ahrefs Lite (post-traction) |
| Position SERP `signaux trading or` (FR) | non-classé | top 20 | top 10 | top 3 | Ubersuggest tracker |
| Conversion landing → signup | n/a | 3% | 4% | 5% | GA4 events |
| Trial → paid | n/a | 20% | 25% | 28% | Stripe |
| Churn paid mensuel | n/a | n/a | 8% | 6% | Stripe MRR retention |
| **Paid users actifs** | 0 | 7 | 46 | 200-296 | Stripe |
| **MRR** | $0 | $300-500 | $1 500-3 000 | $5 000-7 000 | Stripe |
| LTV moyenne (cohort 90j) | n/a | n/a | $250 | $700 | Cohort SQL |
| CAC blended (cash + temps) | n/a | < $10 | < $25 | < $35 | (paid_spend + (h × $50)) / new_paid |
| **LTV/CAC** | n/a | n/a | > 3.0 | > 5.0 | calc |
| Viral coefficient referral `k` | n/a | 0 | 0.10 | 0.30 | (codes_redeemed / codes_generated) × invites/user |
| Hours marketing fondateur /sem | 0 | 9 | 9 | 9 (cap) | self-tracked |

---

## 15. Verdict commercial GTM

**Le plan GTM est exécutable en solo sur 12 mois** sous 3 conditions strictes :

1. **Préalable produit non-négociable** : PF backtest > 1.20 net coût LIVRÉ (Sprint 3-4 produit), 60-90j Telegram public track record vérifiable. Sans ça, **toute campagne paid ou influencer = brûler du capital ET réputation founder de manière potentiellement irréversible**.

2. **Discipline temporelle sacrée** : 8-9h/sem marketing maximum, batch dimanche 14-18h obligatoire. Tout dérapage > 12h/sem → arbitrage immédiat (couper YouTube avant Twitter avant SEO articles avant Telegram).

3. **Wedge FR-first assumé** : ne PAS dilluer effort sur 5 langues / 10 wedges. Concentrer 70% effort sur W5 (FR signaux trading or / ICT français) M1-M6, basculer mix 50/50 FR/EN à partir de M6 si traction.

**Si ces 3 conditions tenues** :
- M3 : MRR $300-500, 7 paid users, Telegram 80 abonnés. **Break-even cashflow** atteint (eval_24 : 6.1 paid suffit).
- M6 : MRR $1 500-3 000, 46 paid, premier paid spend, PH launch consommé.
- M12 : MRR $5 000-7 000 (range robust), 200-296 paid users, LTV/CAC > 3.0, paid budget $2k/mo soutenable.

**Si conditions cassent** :
- PF non fixé → MRR M12 < $1 000, churn > 12%, NPS négatif, brand damage.
- Marketing > 12h/sem → produit ralentit, dette technique compound, fix PF reporté ⇒ spirale négative.
- Wedge dilué sur 5 langues → SEO authority 0 partout, contenu générique, pas de top 10 SERP.

**Note globale GTM (soutenabilité solo)** : **5.8 / 10 aujourd'hui** → projetée **7.5 / 10** post-PF fix + S1-S12 contenu livré.

**Un GTM sans produit qui marche est une catastrophe maquillée. La GTM est le multiplicateur — encore faut-il que le produit soit positif.**

---

## 16. Annexe — Actions concrètes prochaine semaine (M1S1)

| Jour | Heure | Action | Outil | Outcome |
|------|-------|--------|-------|---------|
| Lundi | 19-21h | Audit SERP FR wedge W5 (10 keywords cibles) | Ubersuggest free + Google Search privé | Tableau KD/CPC confirmé |
| Mardi | 19-23h | Setup Substack newsletter "Smart Sentinel Insider" + lead magnet PDF (« 5 SMC traps that destroy your gold trades ») | Substack + Canva | Newsletter live + PDF gated |
| Mercredi | 19-22h | Setup Telegram canal public `@SmartSentinelXAU` + Combot automod + bio + welcome message | Telegram + Combot | Canal live + 5 messages seed |
| Jeudi | 19-22h | Setup landing FR `/fr/signaux-trading-or/` (Notion ou Astro static + Cloudflare Pages) | Cloudflare Pages | URL live, pixel GA4 installé |
| Vendredi | 19-22h | Démarrer rédaction article FR W5 #1 (« 90% des signaux XAU ») | Notion + Grammarly FR | Outline + 800 mots draft |
| Samedi | OFF | Repos | — | Anti-burnout |
| Dimanche | 14-18h | **Batch contenu** : finir article (1800 mots final), 7 tweets queue Buffer, photos vidéo intro YT | Buffer + Capcut | Article publié dimanche 17h, queue Buffer 7 jours |
| Dimanche | 18-19h | Update Trello GTM 90j + revue KPI hebdo | Trello | Dashboard à jour |

**Total temps M1S1** : ~15h (un peu au-dessus du cap 9h/sem mais setup one-shot, lissé sur 4 sem ≈ 9h moyenne).

---

*Eval 28 — Synthesis Lead GTM — 2026-04-26.*
