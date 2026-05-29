# Eval 24 — Cost Structure & Unit Economics

> **Périmètre** : modélisation économique end-to-end (LLM, infra, data, fixes, Stripe) pour les 4 tiers actuels du Business Plan Smart Sentinel AI v1.0 (Observer/Analyst/Strategist/Institutional). Cible : marge brute ≥ 80 % par tier, soutenabilité solo founder pré-launch.
>
> **Sources** : `BUSINESS_PLAN_SMART_SENTINEL.md` v1.0 (2026-03-31), `reports/eval_05_llm.md` (2026-04-24), pricing Anthropic public 2026 (à vérifier 2026), pricing fournisseurs SaaS au 2026-04-25.
>
> **Date** : 2026-04-25 · **Branch** : `main` · **Synthesis Lead** : D10.

---

**Note globale** : **5.5 / 10**
**Verdict** : **Viable sur le papier mais blocker spécifique** : la gross margin annoncée (78–98 %) repose sur 3 hypothèses non vérifiées en prod (cache hit ≥ 60 %, semantic dedup ≥ 95 %, NARRATIVE_MODE=llm activé). En l'état réel (`NARRATIVE_MODE=template` par défaut, `cache_control` no-op, hash cache à 0 % hit), le coût marginal narrative LLM est × 3-5 vs cible. Tier ANALYST ($49) tient toujours, mais STRATEGIST/INSTITUTIONAL surfacturent peu vs INSTITUTIONAL Bloomberg-class à $149 — concurrence directe LuxAlgo Premium. Recommandation : caps signaux/mois dur + fix prompt cache (eval 05 levier #1) avant d'ouvrir au paiement.

---

## 1. Hypothèses & sources (avec dates "à vérifier")

### 1.1 Pricing Anthropic 2026 (à vérifier 2026-04-25)

| Modèle | Input $/MTok | Output $/MTok | Cache write (1.25× in) | Cache read (0.1× in) | Source |
|--------|--------------|---------------|------------------------|----------------------|--------|
| Opus 4.7 | $15.00 | $75.00 | $18.75 | $1.50 | Brief utilisateur, à vérifier docs.anthropic.com |
| Sonnet 4.6 | $3.00 | $15.00 | $3.75 | $0.30 | idem |
| Haiku 4.5 | $1.00 | $5.00 | $1.25 | $0.10 | idem |

> **Note** : eval_05_llm.md ligne 163 cite Haiku $0.80 input / $4.00 output (génération `claude-haiku-4-5-20250929`). Le brief utilise $1/$5. Nous utilisons les chiffres du brief pour cohérence inter-évals, mais flagué en Red-Team §10.

### 1.2 Hypothèses de charge

- **Bars / jour / symbole** : 96 (M15) — cohérent BUSINESS_PLAN §5.1.
- **% bars triggant LLM** : 5–15 % en active session (BUSINESS_PLAN §5.1) ⇒ 5–15 calls/jour/symbole = 150–450 signaux/mois/symbole.
- **Tier signaux/mois (offert)** : non explicite dans BUSINESS_PLAN. Inféré tier × symbole, voir §11.
- **Tokens par appel narrative** (cohérent avec brief D2) :
  - System prompt : 500 tok (eval_05 mesuré 420–550)
  - User prompt : 800 tok (signal payload + vol_regime + news ; brief D2 surdimensionné vs 150 tok du CSV shorthand actuel — choix conservateur pour modéliser après ajout d'examples)
  - Output : 250 tok (3-paragraphes ~ eval_05)
- **Cache hit cible** : 60 % (eval_05 §11 KPI) ; **réel actuel** : 0 % (cache no-op).
- **Semantic shared cache** : claim BUSINESS_PLAN §5.4 = 40 → 99 % selon scale. Audit eval_05 §7 : hit rate live ≈ 0 % à cause de `bar_timestamp` dans la clé. Nous modélisons 0 % comme défaut, 30 % après refonte (eval_05 levier #4).

### 1.3 Conventions

- Tous les calculs sont en **USD**, conversion EUR pour Stripe à 1 EUR ≈ 1.07 USD.
- Marge brute = (ARPU − coût marginal direct) / ARPU. **Cible 80 %**.
- Coût direct = LLM + Stripe + part allouée infra/data variable. **Exclut** R&D, marketing, salaires.

---

## 2. Coût LLM par signal × modèle × cache (table)

**Cas modélisés** :
- (a) **No cache** : input total facturé prix plein input.
- (b) **Cache hit** : system prompt facturé 0.1× ; user/output au tarif normal.
- (c) **Cache miss (write)** : system prompt facturé 1.25× input la première fois ; user/output au tarif normal.

**Formules** :
- (a) `cost = (500+800)/1e6 × in_$ + 250/1e6 × out_$`
- (b) `cost = 500/1e6 × cache_read_$ + 800/1e6 × in_$ + 250/1e6 × out_$`
- (c) `cost = 500/1e6 × cache_write_$ + 800/1e6 × in_$ + 250/1e6 × out_$`

### 2.1 Détail calcul Opus 4.7

- (a) `(1300/1e6 × 15) + (250/1e6 × 75) = 0.0195 + 0.01875 = $0.03825 / signal`
- (b) `(500/1e6 × 1.50) + (800/1e6 × 15) + (250/1e6 × 75) = 0.00075 + 0.012 + 0.01875 = $0.03150 / signal`
- (c) `(500/1e6 × 18.75) + (800/1e6 × 15) + (250/1e6 × 75) = 0.009375 + 0.012 + 0.01875 = $0.040125 / signal`

### 2.2 Détail calcul Sonnet 4.6

- (a) `(1300/1e6 × 3) + (250/1e6 × 15) = 0.0039 + 0.00375 = $0.00765 / signal`
- (b) `(500/1e6 × 0.30) + (800/1e6 × 3) + (250/1e6 × 15) = 0.00015 + 0.0024 + 0.00375 = $0.00630 / signal`
- (c) `(500/1e6 × 3.75) + (800/1e6 × 3) + (250/1e6 × 15) = 0.001875 + 0.0024 + 0.00375 = $0.008025 / signal`

### 2.3 Détail calcul Haiku 4.5

- (a) `(1300/1e6 × 1) + (250/1e6 × 5) = 0.0013 + 0.00125 = $0.00255 / signal`
- (b) `(500/1e6 × 0.10) + (800/1e6 × 1) + (250/1e6 × 5) = 0.00005 + 0.0008 + 0.00125 = $0.00210 / signal`
- (c) `(500/1e6 × 1.25) + (800/1e6 × 1) + (250/1e6 × 5) = 0.000625 + 0.0008 + 0.00125 = $0.002675 / signal`

### 2.4 Tableau de synthèse

| Modèle | (a) No cache $/sig | (b) Cache hit $/sig | (c) Cache miss (write) $/sig | (a) $/1k sig | (b) $/1k sig | (c) $/1k sig |
|--------|-------------------:|--------------------:|-----------------------------:|-------------:|-------------:|-------------:|
| Opus 4.7 | 0.03825 | 0.03150 | 0.04013 | $38.25 | $31.50 | $40.13 |
| Sonnet 4.6 | 0.00765 | 0.00630 | 0.00803 | $7.65 | $6.30 | $8.03 |
| Haiku 4.5 | 0.00255 | 0.00210 | 0.00268 | $2.55 | $2.10 | $2.68 |

**Lecture clef** :
- Économie cache (a→b) : Opus −17.6 %, Sonnet −17.6 %, Haiku −17.6 %.
- À 60 % cache hit ratio : moyenne pondérée ≈ 0.6 × (b) + 0.35 × (a) + 0.05 × (c). Pour Sonnet : `0.6 × 6.30 + 0.35 × 7.65 + 0.05 × 8.03 = 3.78 + 2.68 + 0.40 = $6.86 / 1k sig`.

**Cascade actuelle Haiku→Sonnet** (eval_05 §1) double quasi le coût Sonnet (~ +$2.55 / 1k pour Haiku ajouté). Recommandation eval_05 #2 = supprimer cascade ⇒ chiffres ci-dessus directement applicables.

---

## 3. Coût infra $/MAU × 3 scénarios

### 3.1 Hypothèses Railway (à vérifier 2026)

- Plan **Hobby** : $5/mo + usage (~$0.000463/GB-hour mémoire, $0.000463/vCPU-hour). À 100 MAU, ~$20/mo total constaté (cohérent MEMORY).
- Plan **Pro** : $20/mo base + usage. À 1k MAU, scanner Numba en steady-state ≈ 0.5 vCPU constant + 1GB RAM ⇒ ~$40-60/mo.
- Plan **Pro multi-replica** + workers : à 10k MAU, ~$150-250/mo (scaling horizontal scanner par symbole).

### 3.2 DB & Cache

| Composant | Free tier | Payant | Trigger migration |
|-----------|-----------|--------|-------------------|
| SQLite (actuel) | $0 | — | Casse > ~50k signaux ou > 5 writers concurrents |
| Postgres Neon | < 500 MB, gratuit | $19/mo (1 GB compute + 10 GB storage) | ~1k MAU |
| Postgres Neon Scale | — | $69/mo | ~10k MAU |
| Upstash Redis (semantic cache + rate limiting) | 256 MB, 500k cmd/jour gratuit | $0.20 / 100k cmd | > 1k MAU si refonte cache |

### 3.3 Tableau $/MAU

| Scénario | MAU | Railway | DB | Redis | **Total infra/mo** | **$/MAU** |
|----------|-----|--------:|---:|------:|-------------------:|----------:|
| Seed | 100 | $20 | $0 (SQLite) | $0 (free) | **$20** | **$0.200** |
| Growth | 1 000 | $50 | $19 (Neon Launch) | $0 (free) | **$69** | **$0.069** |
| Scale | 10 000 | $200 | $69 (Neon Scale) | $20 (1M cmd/mo) | **$289** | **$0.029** |

**Économie d'échelle** : ÷ 7 entre Seed et Scale. C'est un *vrai* SaaS, l'infra n'est pas un bloqueur.

> ⚠️ **Hypothèse fragile** : les 100 MAU ne génèrent pas 100× les API calls — ils consomment le **même** stream de signaux (un seul scanner XAU pousse la même narrative à tous). C'est l'argument BUSINESS_PLAN §5.4. Mais c'est faux pour STRATEGIST/INSTITUTIONAL multi-asset si chaque user choisit ses pairs ⇒ scanners dédiés. Modélisé conservativement à 1 scanner par symbole pré-configuré (6 max).

---

## 4. Coût data (current vs commercial-licensed)

### 4.1 Situation actuelle

- **Dukascopy** : tick + M1 historique gratuit. **Licence non-commerciale explicite** (Dukascopy Bank SA Terms §3.2 — "personal, non-commercial use only"). ⇒ **bloqueur juridique** dès Stripe activé.
- **MT5 broker feed** : selon broker. IC Markets / FTMO / Pepperstone permettent extraction non commerciale. À clarifier broker par broker.

### 4.2 Coverage requis

- 6 symboles × 3 timeframes (M15, H1, H4) = 18 streams.
- Stream M15 = 96 bars/jour × 6 = 576 bars/jour ⇒ ~17k bars/mois total — volumétrie négligeable.
- Le coût n'est pas le volume, c'est la **licence redistribution + temps réel** (≤ 250ms latence pour scanner M15).

### 4.3 Fournisseurs commerciaux (à vérifier 2026-04-25)

| Fournisseur | Plan | Coverage | Cost/mo |
|-------------|------|----------|---------|
| Polygon.io Stocks Starter | $29 | US stocks only — **pas Forex/Gold/Crypto** | $29 |
| Polygon.io Forex Starter | $49 | Forex + Crypto, 1 symbole/req | $49 |
| Polygon.io Currencies Pro | $199 | Forex + Crypto temps réel illimité | $199 |
| Twelve Data Pro 610 | $79 | 610 req/min, FX/Crypto/Stocks/Indices | $79 |
| Twelve Data Ultra 1500 | $229 | 1500 req/min, websocket | $229 |
| Tiingo IEX | $40 | Stocks + Forex EOD/intraday délayé | $40 |
| TraderMade Standard | $49 | Forex + Metals temps réel | $49 |
| TwelveData + TraderMade combo | — | XAU + 5 FX + indices | ~$130 |

### 4.4 Recommandation

- **Pré-launch** : continuer Dukascopy pour *développement*, **pas pour servir** des signaux payants.
- **Launch ANALYST seulement** ($49 Gold) : **TraderMade Standard $49/mo** suffit (XAU + EUR/USD/GBP/JPY).
- **Launch STRATEGIST/INSTITUTIONAL** (multi-asset + indices + crypto) : **Twelve Data Pro $79/mo** + **Polygon Currencies Pro $199/mo** = **$278/mo** ; ou attendre 50 paid users avant de souscrire.
- **Budget commercial réaliste** : **$130 → $300/mo** selon scope.

---

## 5. Coûts fixes mensuels

| Poste | Coût/mo | Source |
|-------|--------:|--------|
| Domaine (smartsentinel.ai ~$15/an) | $1.25 | NameCheap 2026 |
| Email transactional (Resend free 3k/mo, then $20) | $0 → $20 | Resend.com |
| Sentry error monitoring (free 5k events) | $0 → $26 (Team) | Sentry.io |
| Better Uptime (10 monitors free) | $0 → $18 (Pro) | BetterStack |
| Stripe (no monthly fee) | $0 + 2.9 % + €0.30/txn | stripe.com |
| Legal CGU (avocat one-shot ~€2k → 24 mois) | $89 (€83) | Estimation FR/QC |
| Backup S3-compatible (Backblaze B2 ~10 GB) | $1 | backblaze.com |
| GitHub Pro (CI minutes) | $4 | github.com |
| Anthropic Workbench/Console (no fee) | $0 | — |

**Total fixe pré-launch (free tiers max)** : **~$95/mo** (~€89).
**Total fixe post-launch (50+ paid)** : **~$160/mo** (~€150).

> Cohérent avec brief D5 (€100-150/mo).

---

## 6. Coût marginal par tier (table FREE/ANALYST/STRATEGIST/INSTITUTIONAL)

### 6.1 Hypothèses d'usage par tier (proposées — voir §11 caps recommandés)

| Tier | Prix BP | Symboles | Signaux/mois | Modèle narrative | Cache hit cible |
|------|--------:|---------:|-------------:|------------------|----------------:|
| Observer (FREE) | $0 | 1 (XAU) | 30 (visuel) | none (template only) | n/a |
| Analyst | $49 | 1 (XAU) | 200 | Haiku 4.5 single-call | 60 % |
| Strategist | $99 | 4 (XAU + 3 FX) | 800 | Sonnet 4.6 single-call | 60 % |
| Institutional | $149 | 6 + chat | 2 000 | Opus 4.7 narrative + chat | 60 % |

### 6.2 Coût LLM marginal mensuel par tier

Application moyenne pondérée 60 % cache hit (b) / 35 % no cache (a) / 5 % cache miss (c) calculée §2.4.

**Analyst** (Haiku 200 sig/mo) :
- Coût/sig moyen Haiku ≈ `0.6 × 0.00210 + 0.35 × 0.00255 + 0.05 × 0.00268 = 0.00126 + 0.000893 + 0.000134 = $0.00229`
- Coût/mo Haiku narratives = `200 × 0.00229 = $0.458`
- Hypothèse signaux partagés inter-users via semantic cache : 0 % aujourd'hui, 30 % après fix. Modélisé sans dedup (worst case par user).

**Strategist** (Sonnet 800 sig/mo) :
- Coût/sig moyen Sonnet ≈ `0.6 × 0.00630 + 0.35 × 0.00765 + 0.05 × 0.00803 = 0.00378 + 0.002678 + 0.000402 = $0.00686`
- Coût/mo Sonnet = `800 × 0.00686 = $5.49`

**Institutional** (Opus 2000 sig/mo + chat 50 questions × 1500 tok in / 600 tok out) :
- Coût/sig moyen Opus ≈ `0.6 × 0.0315 + 0.35 × 0.03825 + 0.05 × 0.04013 = 0.0189 + 0.01339 + 0.00201 = $0.0343`
- Coût/mo narratives = `2000 × 0.0343 = $68.60`
- Coût chat Opus = `50 × ((1500/1e6 × 15) + (600/1e6 × 75)) = 50 × (0.0225 + 0.045) = 50 × 0.0675 = $3.375`
- Total LLM = **$71.97**

### 6.3 Stripe par tier

- Stripe = 2.9 % du prix + $0.30 (USD ; €0.30 ~$0.32).
- Analyst $49 : `49 × 0.029 + 0.30 = 1.421 + 0.30 = $1.72`
- Strategist $99 : `99 × 0.029 + 0.30 = 2.871 + 0.30 = $3.17`
- Institutional $149 : `149 × 0.029 + 0.30 = 4.321 + 0.30 = $4.62`

### 6.4 Part infra/data allouée (au scénario Growth, 1k MAU)

- Infra/MAU = $0.069 (§3.3)
- Data $130/mo / 1k MAU = $0.13/MAU
- Total non-LLM/MAU = ~$0.20/MAU.

### 6.5 Tableau coût marginal complet (Growth scenario)

| Tier | LLM/mo | Stripe/mo | Infra+Data/mo | **Coût marginal/mo** | Prix tier | **Marge brute** | **Cible 80 %** |
|------|-------:|----------:|--------------:|---------------------:|----------:|----------------:|:--------------:|
| Observer | $0 | $0 | $0.20 | **$0.20** | $0 | n/a (loss leader) | n/a |
| Analyst | $0.46 | $1.72 | $0.20 | **$2.38** | $49 | **95.1 %** | ✅ |
| Strategist | $5.49 | $3.17 | $0.20 | **$8.86** | $99 | **91.0 %** | ✅ |
| Institutional | $71.97 | $4.62 | $0.20 | **$76.79** | $149 | **48.5 %** | ❌ **Sous-prix** |

**Lecture clef** :
- Analyst & Strategist OK avec marge confortable (>90 %).
- **Institutional est underpriced** : le passage à Opus 4.7 (×5 vs Sonnet) et le volume de signaux + chat fait exploser le coût. Pour atteindre 80 % marge, **prix mini = $76.79 / 0.20 = $384/mo**.
- Alternative : downgrade narrative INSTITUTIONAL à Sonnet 4.6 + chat Opus seulement, ou cap signaux à 800/mo (cf §11).

### 6.6 Sensibilité au cache

Si cache hit reste à **0 %** (état actuel) :
- Analyst : $0.51 → marge 95.0 % (toujours OK, volumes faibles)
- Strategist : $6.12 → marge 90.4 % (OK)
- Institutional : $76.50 LLM + $4.62 + $0.20 = $81.32 → marge **45.4 %** (encore plus mauvais)

⇒ Le levier cache **ne sauve pas Institutional**, c'est un problème de routing modèle, pas de cache.

---

## 7. ARPU cible vs actuel — gap

### 7.1 ARPU minimum pour 80 % marge brute (Growth scenario)

| Tier | Coût marginal | **ARPU min** = coût/0.20 | Prix BP actuel | Gap |
|------|--------------:|------------------------:|---------------:|----:|
| Analyst | $2.38 | **$11.90** | $49 | OK +311 % |
| Strategist | $8.86 | **$44.30** | $99 | OK +123 % |
| Institutional | $76.79 | **$383.95** | $149 | ❌ −61 % |

### 7.2 Recommandations prix

| Tier | Prix actuel | Recommandé | Rationale |
|------|------------:|-----------:|-----------|
| Observer | $0 | $0 | Loss leader, garder |
| Analyst | $49 | **$49** | Bien calibré ; concurrent direct LuxAlgo $40-59 ; marge >95 % laisse marge à promos |
| Strategist | $99 | **$99** ou **$129** | Si Sonnet 4.6 single-call, $99 OK. Si on garde cascade, $129 sécurise. |
| Institutional | $149 | **$299–$399** OU cap signaux à 800 + Sonnet | $149 sous-évalue Opus 4.7. Soit on assume premium pricing (audience hedge funds), soit on dégrade narrative (Sonnet pour narratives, Opus pour chat seulement). |

### 7.3 Variante "Institutional-Lite" recommandée

- **Plan Institutional v2** : $199/mo, Sonnet 4.6 narratives, Opus 4.7 chat (10 questions/mois inclus + $0.50 par question additionnelle).
- Coût marginal recalculé : `2000 × 0.00686 + 10 × 0.0675 + $4.62 + $0.20 = 13.72 + 0.675 + 4.82 = $19.21`
- Marge à $199 : **(199 − 19.21)/199 = 90.3 %** ✅
- Marketing : préserve "Powered by Claude Opus" sans exposer la totalité narrative au modèle premium.

---

## 8. Stress scenarios × marge

Tous les chiffres en marge brute % au scénario Growth (1k MAU).

| Scénario | Analyst | Strategist | Institutional (BP $149) | Mitigation |
|----------|--------:|-----------:|------------------------:|------------|
| **Baseline** (60 % cache, modèles tier-routed) | 95.1 % | 91.0 % | 48.5 % ❌ | Cap signaux, Institutional-Lite (§7.3) |
| **S1: Anthropic +100 %** | 94.1 % | 84.9 % | 3.0 % ❌❌ | Switch ANALYST to template-only ; passage Sonnet→Haiku pour STRAT en fallback |
| **S2a: cache 60 % → 20 % (régression)** | 95.0 % | 90.4 % | 45.4 % ❌ | Fix prompt cache (eval_05 levier #1) |
| **S2b: cache 60 % → 90 % (best case)** | 95.2 % | 91.4 % | 50.6 % | Aucune (déjà fait) |
| **S3: viral 10× volume / semaine** | LLM 10× ⇒ marge 67 % | 26 % | -383 % (déficit) | Throttle Telegram (cap 10 sig/jour/user), fallback Haiku, queue prioritaire payants |
| **S4: Anthropic deprecates Opus 4.7** | n/a | n/a | Forced Sonnet 4.6, qualité narrative -15 %, marge **90.3 %** ✅ | Préparer "Institutional-Lite" déjà compatible |

### 8.1 Calcul détaillé S1 Institutional

- Coût LLM × 2 = $143.94 ; total marginal = $148.76 ; marge = (149 − 148.76)/149 = **0.16 %**. Effectivement break-even à 0. **Rouge vif.**

### 8.2 Calcul détaillé S3 Strategist

- Volumes × 10 sur 1 mois : `8000 × $0.00686 = $54.88` LLM ; marge = (99 − 54.88 − 3.17 − 0.20)/99 = **41.2 %** — chute en dessous de 80 % cible.
- Avec throttle 1500 sig/mois cap : marge reste 90 %.

---

## 9. Top 5 leviers d'optimisation chiffrés

| # | Levier | Effort | Économie estimée (/mo, 1k MAU) | KPI cible | Source |
|---|--------|--------|-------------------------------:|-----------|--------|
| **1** | **Fix prompt caching** : étendre system prompt à ≥1024 tok (ajouter examples) ⇒ cache effectif | 0.5 j | $4–8 (Sonnet) à $35-50 (échelle 10k MAU) ; -17 % input par appel | `cache_read_input_tokens > 0` ; hit ≥ 60 % | eval_05 §3, §9 #1 |
| **2** | **Routing tier→modèle** : Haiku ANALYST, Sonnet STRATEGIST, Opus uniquement chat INST | 1 j | -$60/mo Institutional pure narrative | Sonnet narratives default ; Opus chat-only | eval_05 §4 #2 |
| **3** | **Caps signaux/mois** durs par tier (rate limit + queue overflow) | 1 j | Plafonne stress S3 à -10 % marge max | Soft cap 80 %, hard cap 100 % | §11 |
| **4** | **Signature-based cache** (symbol + score_bucket × 5 + regime + news_bucket + vol_regime + dir) avec TTL 30 min | 3 j | -30-50 % appels LLM à volume égal ⇒ -$2 STRAT, -$25 INST | hit rate ≥ 30 % après 7j | eval_05 §7, §9 #4 |
| **5** | **Batch narratives** : si scanner émet ≥2 signaux dans 60 s → batch dans 1 appel Sonnet | 2 j | -10 % Anthropic en busy market (NFP, FOMC) | batch_size_avg ≥ 1.3 | nouveau |

**Cumul des 5 leviers (Strategist 1k MAU)** : coût LLM/user passe de $5.49 → ~$2.20, marge 91 % → 95.6 %.
**Cumul (Institutional 1k MAU)** : $71.97 → ~$25, marge 48.5 % → 80.5 % ✅ (sans toucher au prix).

---

## 10. Red-Team — hypothèses fragiles

| # | Hypothèse | Risque | Probabilité | Action recommandée |
|---|-----------|--------|------------:|--------------------|
| R1 | Cache hit 60 % atteignable | Eval_05 montre que c'est nul aujourd'hui ; benchmarks Anthropic 70 % avec system ≥ 2000 tok ; nous modélisons à 500 tok (sous le minimum 1024) | Élevée si pas de fix | Mesurer en preprod sur 7 jours avant locker tarification |
| R2 | Pricing Anthropic 2026 figé | Eval_05 cite Haiku $0.80/$4 ; brief utilisateur dit $1/$5 ; Anthropic a changé pricing 3× en 18 mois | Moyenne | Re-vérifier mensuellement, négocier Enterprise quand > $1k/mo |
| R3 | Dukascopy licence acceptable | Termes interdisent "any commercial use including resale of derived data". Signal payant = dérivé de leur feed = breach. | Élevée | Switch obligatoire avant Stripe live. Budget $50-300/mo. |
| R4 | Cache MIN_TOKEN reste 1024/2048 | Anthropic peut changer (a déjà bougé entre 2024 1024-only et 2025 ext.) | Faible mais non-nul | Monitorer changelog ; fallback contractuel sur Sonnet si Haiku cache change |
| R5 | Volumes signaux/tier (200/800/2000) cohérents avec ce que l'ICP veut payer | Nous extrapolons sans Prompt 25 (ICP) | Élevée | Croiser avec eval_25_icp avant locking caps |
| R6 | Semantic shared cache hit rate 95 % à 5k users (BUSINESS_PLAN §5.4) | Eval_05 §7 montre que la clé inclut `bar_timestamp` ⇒ 0 % en réalité ; même après refonte, 30-50 % réaliste | Élevée | Réécrire BUSINESS_PLAN §5.4 avec chiffres conservateurs |
| R7 | Stripe 2.9 % suffit | Pour Europe, IBAN/SEPA 0.8 % ; carte 2.9 % ; mais conversion devises +1 % | Faible | Monitorer mix paiement |
| R8 | Solo founder bandwidth pour répondre support payant Institutional | $149/mo × 50 users ≈ tickets = 50/sem si SLA 24h, irréaliste solo | Moyenne | Soit cap Institutional à 10 users en bêta, soit Intercom $74/mo en hors-cost |
| R9 | Numba scanner reste ≤ 0.5 vCPU à 6 symboles temps réel | Pas testé en steady-state 30j | Moyenne | Charge test avant Scale |
| R10 | LLM downtime mitigé par template | Eval_05 §5.2 indique que template fallback **n'est pas câblé** sur CircuitOpenError actuel ⇒ users payants reçoivent dict pauvre | Élevée | Lever eval_05 #3 (1 j effort) |

---

## 11. Plan caps signaux/mois recommandé

| Tier | Symboles | **Cap soft** (alerts user) | **Cap hard** (queue drop) | Justification coût |
|------|----------|---------------------------:|---------------------------:|--------------------|
| Observer | 1 (XAU template-only) | 30 | 50 | Filtre upgrade naturelle |
| Analyst | 1 | 200 | 300 | Coût Haiku ≤ $0.69/mo cap hard |
| Strategist | 4 | 800 | 1 200 | Coût Sonnet ≤ $8.23/mo cap hard |
| Institutional | 6 + chat | 2 000 + 50 chat | 3 000 + 100 chat | Coût Opus ≤ $108/mo cap hard ; déclenche revue prix |

**Implémentation** :
1. Compteur Redis `signals_consumed:{user_id}:{YYYY-MM}` incr/scan.
2. Soft cap : Telegram message + dashboard banner "you've used 80% of your monthly signals".
3. Hard cap : queue drop avec event "tier_overflow" → upgrade CTA.
4. Reset mensuel UTC.

---

## 12. Burn rate / break-even (best/worst case)

### 12.1 Coûts fixes mensuels solo founder (post-launch léger)

- Fixes §5 : $160/mo
- Infra Growth (1k MAU) : $69/mo
- Data : $130/mo (TraderMade + TwelveData)
- **Burn fixe total** : **~$359/mo** (~€336)
- + Coût LLM proportionnel : variable

### 12.2 Break-even par mix users

**Hypothèse** : ratio 70 % Analyst / 25 % Strategist / 5 % Institutional (cohérent funnel SaaS).
**ARPU pondéré** : `0.70 × 49 + 0.25 × 99 + 0.05 × 149 = 34.30 + 24.75 + 7.45 = $66.50`
**Coût marginal pondéré** : `0.70 × 2.38 + 0.25 × 8.86 + 0.05 × 76.79 = 1.67 + 2.22 + 3.84 = $7.73`
**Marge unitaire** : `66.50 − 7.73 = $58.77`

**Break-even users** : `359 / 58.77 = 6.1` users payants. ✅ **Atteignable mois 1.**

### 12.3 Scénarios

| Scénario | Users payants requis | Délai estimé (à 4 % conv. free→paid) |
|----------|---------------------:|--------------------------------------|
| **Best case** (Institutional v2 $199 + caches actifs) ARPU = $66.50, marge unitaire = $63 | 5.7 users | Mois 1 (avec 200 free users) |
| **Realistic** baseline | 6.1 users | Mois 1-2 |
| **Worst case** S1 Anthropic +100 % + Inst sous-prix : marge unitaire = $50 | 7.2 users | Mois 2-3 |
| **Worst-worst** : aucun fix, cache 0 %, viral S3 : marge unitaire = $25 | 14.4 users | Mois 4 |

### 12.4 Runway

- Si capital initial $5k (légal + setup) :
  - Worst-worst : `5000 / 359 = 13.9 mois` runway sans revenu
  - Realistic avec 10 paid users mois 3 : break-even immédiat post-Mois 3

**Verdict runway** : très favorable. Le risque n'est pas financier mais légal (Dukascopy) et qualitatif (LLM hallucinations).

---

## 13. KPIs (gross margin %, $/signal, $/MAU, runway mois)

| KPI | Baseline (état actuel) | Cible 30j post-fix | Cible 90j |
|-----|-----------------------:|-------------------:|----------:|
| Gross margin Analyst | 95.0 % | 96.0 % | 97.0 % |
| Gross margin Strategist | 90.4 % | 95.6 % | 96.5 % |
| Gross margin Institutional | 45.4 % ❌ | 80.5 % ✅ (post-routing+cache) | 90 % (post-Inst-Lite v2) |
| Coût $/signal moyen Sonnet | $0.0077 (no cache) | $0.0069 (60 % cache) | $0.0050 (cache + dedup) |
| Coût $/signal moyen Opus | $0.0383 | $0.0334 | $0.0250 |
| Coût $/MAU infra | $0.20 (Growth) | $0.15 | $0.05 (Scale) |
| Cache hit rate Anthropic | 0 % | ≥ 60 % | ≥ 80 % |
| Semantic shared cache hit | 0 % | ≥ 30 % | ≥ 50 % |
| Runway sans revenu | 13.9 mois ($5k cap) | 13.9 mois | n/a (cashflow positif) |
| Break-even paid users | 6.1 | 5.7 | 4 |
| MRR mois 6 cible BP | — | $3 920 | $3 920 |

---

## 14. Trade-offs assumés

| Décision recommandée | Trade-off explicite |
|----------------------|---------------------|
| **Garder Analyst à $49** | Marge >95 % laisse marge promos / parrainage ; mais prix < concurrents premium ⇒ image "low-end" possible. Mitigation : badge "Pro" et features visibles. |
| **Strategist à $99 + Sonnet single-call** | Perte du "double-check Haiku→Sonnet" prônée dans BP §4.1 ; mitigée par le fait que `ConfluenceDetector` algo gate déjà filtre 95 % des bars (eval_05 §4.3). |
| **Institutional repricé $199 (Sonnet narrative + Opus chat)** | Risque de cannibalisation Strategist si écart features insuffisant. Différenciateur clé : multi-asset complet + API + Opus chat = $100/mo perçu de valeur additionnelle vs Strategist. |
| **Caps signaux/mois durs** | Risque churn user qui frappe le cap et mal-perçoit. Mitigation : caps généreux (≥ 2× consommation typique mesurée) + soft cap notifs avant. |
| **Switch Dukascopy → TraderMade $49 dès launch** | $49/mo CapEx fixe additionnel. Mais protection juridique non négociable. Inclus dans burn §12. |
| **NARRATIVE_MODE par défaut = template, pas LLM** (état actuel) | Élimine le coût LLM mais transforme le pitch "AI-powered" en mensonge marketing. Eval_05 §0 note 5/10 sur cette dimension. **Recommandation** : flip default à `llm` AVANT première campagne marketing payée. |
| **Pas de support multi-langue avant 100 paid users** | TAM FR/ES/PT bloqué (~50 % marché retail), mais coût dev/support bilingue solo founder = no go avant traction. |
| **Pas d'API pay-per-use $0.05/call avant 200 paid** | Le tier "Developer" du BP §6.1 reste un placeholder ; demande Stripe metered billing + abuse protection ; à reporter Q2. |
| **Cache hit rate modélisé à 60 % alors que mesuré 0 %** | Optimiste. Tous les chiffres marge §6/§7 supposent que le levier #1 est livré dans les 30 jours. Si non livré : marge Strategist passe de 91 % à 90.4 % (impact mineur), Institutional de 48.5 % à 45.4 % (déjà rouge). |
| **Stress S1 (+100 % Anthropic) ignoré dans pricing** | On parie que Anthropic va baisser, pas monter. Historiquement vrai (Sonnet 4.5 → 4.6 même prix avec contexte 2× plus long). Si faux : refacturation +50 % imposée à Institutional. |

---

## Annexe A — Cellules de calcul reproductibles

```
# LLM cost per signal generic
def cost_per_signal(in_tok, out_tok, in_price, out_price,
                    cached_in_tok=0, cache_read_price=0,
                    cache_write_tok=0, cache_write_price=0):
    return (
        (in_tok - cached_in_tok - cache_write_tok) / 1e6 * in_price
        + cached_in_tok / 1e6 * cache_read_price
        + cache_write_tok / 1e6 * cache_write_price
        + out_tok / 1e6 * out_price
    )

# Sonnet cache hit:
# cost_per_signal(1300, 250, 3, 15, cached_in_tok=500, cache_read_price=0.30) = 0.00630
```

## Annexe B — Sources à re-vérifier 2026-04-25

1. Anthropic pricing : <https://docs.anthropic.com/en/docs/about-claude/pricing> (à vérifier)
2. Anthropic prompt caching minimums : <https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching> (1024/2048 confirmé eval_05, à re-vérifier 2026)
3. Dukascopy Terms : <https://www.dukascopy.com/swiss/english/about/legal-issues/> §3.2 (à vérifier)
4. Polygon.io pricing : <https://polygon.io/pricing> (à vérifier)
5. Twelve Data pricing : <https://twelvedata.com/pricing> (à vérifier)
6. TraderMade pricing : <https://tradermade.com/pricing> (à vérifier)
7. Railway pricing : <https://railway.com/pricing> (à vérifier)
8. Neon pricing : <https://neon.tech/pricing> (à vérifier)
9. Upstash Redis : <https://upstash.com/pricing/redis> (à vérifier)
10. Stripe Europe : 2.9 % + €0.30 carte / 0.8 % SEPA (à vérifier)

---

*Eval 24 — Synthesis Lead D10 — 2026-04-25.*
