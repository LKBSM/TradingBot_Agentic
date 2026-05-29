# Plan de Commercialisation — Catégorie 1: GTM & Commercialisation

> **Date** : 2026-05-21
> **Auteur** : Agent Catégorie 1 — Commercialisation & GTM (sprint exhaustif)
> **Branche** : `institutional-overhaul`
> **Objectif structurant** : passer Smart Sentinel AI du statut « techniquement remarquable, commercialement non vendable » à « beachhead payant XAU SMC FR-first, MRR M6 = $1.5-3k » en 12-16 semaines de travail solo (8-10h/sem marketing + 30h/sem produit).
> **Périmètre exclu** : produit (PF), data quality, RGPD/compliance, perf engineering, intelligence engine, observabilité. Ces piliers sont des **dépendances**, pas des livrables de cette catégorie. Cf. §9.
>
> **Statut MVP commercial** : 🔴 NO-GO immédiat. Le plus court chemin vers GO est dans la section 4. Décision réelle de vendre = subordonnée à validation Phase 0 produit + 60-90j Telegram public PF live > 1.10 (cf. eval_25 §10, eval_28 §15, decision_matrix_2026_04_30.md).

---

## 0. Synthèse exécutive (TL;DR)

| Dimension | Note actuelle | Cible 12 mois | Source |
|-----------|--------------:|--------------:|--------|
| PMF / ICP (clarté segment beachhead) | 4.5/10 | 8/10 | `reports/eval_25_pmf_icp.md:7` |
| Pricing v1 (rigueur grille + marge) | 5.5/10 (BP gelé $49/$99/$149) | 8.5/10 (FREE / $29 / $79 / $1990 + annual) | `reports/eval_27_pricing.md:11-25` |
| GTM solo (soutenabilité 8-10h/sem) | 5.8/10 | 7.5/10 | `reports/eval_28_gtm.md:24` |
| Unit economics (marge brute ≥80%) | 5.5/10 (Institutional sous-prix ×2.6) | 8.5/10 (grille v2) | `reports/eval_24_unit_economics.md:11` |
| Défensabilité commerciale | 3.5/10 | 7.5/10 (post track-record public + niche FR) | `reports/eval_26_competitive.md:23` |
| **Verdict commercial** | **NO-GO** | **GO conditionnel M+6** | — |

**Top 3 bloqueurs commerciaux à fixer IMMÉDIATEMENT (P0)** :
1. **Refonte grille pricing** : remplacer `tier_manager.py` FREE/$49/$99/$149 par FREE/$29/$79/$1990 + annual 16.7% off + decoy implicite (eval_27 §0, §9).
2. **Page Track-Record publique** + Telegram channel public 60-90j en lecture seule (eval_25 §11, eval_26 Diff #1, eval_28 §10 levier #2).
3. **Landing pages FR-first + EN secondaire** sur les 5 wedges SEO (W1-W5), avec hero V3 ("Stop revenge trading") + V4 ("Trade like institutions") A/B (eval_25 §8, eval_28 §2.3).

**Effort total Catégorie 1** : **~252h de travail (Loukmane seul) sur 16 semaines**. Détail §10.
**Chemin critique** : (a) Pricing v1 lock → (b) Track-record public 60j → (c) Landing W5 + AMA r/Forex → (d) PH launch M5 → (e) Referral M5 → (f) Influencer Trader Pro FR M6.

---

## 1. État actuel (Audit)

### 1.1 Pricing — où en est le code aujourd'hui

| Élément | Fichier:ligne | Constat | Problème commercial |
|---------|---------------|---------|---------------------|
| Tier definitions | `src/api/tier_manager.py:36-67` | 4 tiers hardcodés FREE / ANALYST $49 / STRATEGIST $99 / INSTITUTIONAL $149 | INSTITUTIONAL **sous-prixé ×13** : marge réelle 48.5% vs cible 80% (eval_24 §6.5, §7.2). Cannibalisation tiers (gap 49→99 trop large pour Marc, $99→$149 trop faible vs valeur). |
| Stripe config | (à vérifier — pas de fichier Stripe live trouvé) | Non commitée, pas de webhook live | Bloqueur paiement immédiat. |
| Pricing display | Aucune landing produit publique | Mockups en `mockups/pricing_bundles.md` + `mockups/webapp_b2c.html` | Pricing existe en draft, jamais affiché. |
| Tier rate-limit | `src/api/tier_manager.py` (rate-limit logic) | Dead code identifié dans eval 10-15 audit | Quota non appliqué → users free peuvent tout consommer = burn LLM non maîtrisé. |
| Bundles thématiques | `mockups/pricing_bundles.md:17-23` | 4 packs (FX/Metal/Crypto/Index) proposés à $29-49/mo | Risque cannibalisation interne (ANALYST $19 < FX Pack $29) flagué dans le mockup lui-même `mockups/pricing_bundles.md:113`. Non implémenté. |

### 1.2 ICP — où en est la documentation

| Élément | Source | Constat |
|---------|--------|---------|
| Persona A (Marc XAU SMC retail FR) | `reports/eval_25_pmf_icp.md:41-66` | Persona détaillé (WTP $20-49, canaux Discord/YouTube FR, JTBD = éviter FOMO/sleep), founder-fit max (5/5). |
| Persona B (James prop firm EN) | `reports/eval_25_pmf_icp.md:70-96` | **À DEFER 6-12 mois** (PF 0.96 = perte challenge garantie). |
| Persona C (Sophie semi-pro) | `reports/eval_25_pmf_icp.md:100-125` | **À DEFER 18+ mois** (track record audité requis). |
| Interviews discovery exécutées | 0 mentionné | **Aucune** à 2026-05-21 ⇒ ICP encore théorique. |

### 1.3 GTM — actifs marketing existants

| Actif | Statut | Source |
|-------|--------|--------|
| Landing pages publiques | 🔴 0 publiée | `mockups/webapp_b2c.html` (draft non publié) |
| Domaine `smartsentinel.ai` | 🟡 réservé (mentionné `eval_24 §5`) | Pas de DNS pointé, pas de hosting |
| Site / blog (CMS) | 🔴 inexistant | — |
| Telegram channel public (track record) | 🔴 non créé | Référencé `eval_25 §11.2` comme à créer S2 |
| Discord privé paid | 🔴 non créé | — |
| Newsletter (Substack ou autre) | 🔴 0 abonné | — |
| Pine Script TradingView | 🔴 0 publié | Référencé `eval_28 §2.3 W3` |
| YouTube channel FR/EN | 🔴 0 vidéo | — |
| Twitter/X handle | 🟡 (à vérifier — pas confirmé) | Possible mais inactif |
| Track-record vérifié MyFXBook | 🔴 0 | Identifié `eval_26 Diff #1` comme moat #1 |
| Documents de vente (one-pager B2B, deck) | 🔴 0 | — |
| Compliance pages `/terms`, `/privacy` | 🟢 livrées sprint W1 2026-04-29 (`memory/sprint_w1_compliance_2026_04_29.md`) | Existent, à brancher landing |

### 1.4 Funnel & métriques actuelles

- **Visiteurs landing/mois** : 0 (pas de landing).
- **Free signups** : 0.
- **Paid users** : 0 (mode testing perso, `SENTINEL_TESTING_MODE=1` par défaut, cf. `MEMORY.md`).
- **MRR** : $0.
- **CAC** : non mesurable (pas d'attribution, pas de UTM tracking, pas de GA4).
- **LTV** : non mesurable (pas de cohort).
- **Churn** : non mesurable.

### 1.5 Capital & runway

- **Capital initial estimé** : $5k (eval_28 §0).
- **Burn fixe pré-launch** : $359/mo (`reports/eval_24_unit_economics.md:357`).
- **Runway sans revenu** : 13.9 mois (eval_24 §12.4) — marge confortable, **le risque n'est pas le cash, c'est le temps founder et la réputation**.

### 1.6 Vérdict synthétique de l'audit

Le produit a déjà tous les composants techniques d'un SaaS commercial sérieux (4-tier, Stripe-ready, multi-langue compliance, Telegram, API, SignalStore audit-trail). Ce qui manque est **100% commercial** : aucune landing publique, aucun track record public, aucun client interviewé, pricing v1 gelé sur un BP de mars 2026 jamais relu à l'aune des unit-economics post-eval_05 LLM. **Pas un seul euro n'est facturable aujourd'hui.**

---

## 2. Vision cible commercialisation (qu'est-ce qui doit être vrai pour vendre demain)

### 2.1 Définition « vendable demain » (M+0 d'un lancement payant)

1. **Une grille pricing publique cohérente** affichée sur `smartsentinel.ai/pricing` : FREE / STARTER $29 / PRO $79 / INSTITUTIONAL $1990, plus toggle annual (-16.7%), avec marge brute ≥80% sur chaque tier payant **modélisée** (eval_27 §9).
2. **Un parcours d'achat fonctionnel** : landing → CTA → Stripe Checkout → email confirmation → onboarding 5 étapes → Telegram link + dashboard URL. Zéro friction technique.
3. **Une preuve commerciale** : 60-90 jours de signaux Telegram publics, lecture seule, screenshots PnL exportables, dashboard `/track-record` accessible sans auth, et idéalement MyFXBook verified (eval_26 Diff #1).
4. **Un ICP testé** : 15 interviews Persona A bouclées (FR + EN), avec verbatims qui valident WTP $20-49, JTBD "éviter mauvais trade par semaine", objections connues et adressées (eval_25 §11).
5. **Un wedge SEO en cours d'indexation** : au moins 6 articles publiés sur wedge W5 (FR — `signaux trading or`, `ICT français`, etc.), 1 article cornerstone EN sur W1 (`smart money concepts AI`), Pine Script TV publié (eval_28 §3.2).
6. **Une compliance commerciale propre** : geo-block US/QC/UK actif, disclaimers FR/EN multi-langue, CGU/CGV/Politique de confidentialité visibles, opt-in cookies, Stripe configuré pour TVA UE (cf. `memory/sprint_w1_compliance_2026_04_29.md`).
7. **Un funnel mesurable** : GA4 + Meta pixel + UTM standardisés + cohort SQL prêt + dashboard d'attribution interne.

### 2.2 Définition « scalable d'ici M+6 »

8. **Product Hunt launch consommé** (1 jour M5) + AMA r/Forex M3 + 2 podcasts inbound contactés.
9. **Referral program live** (Stripe coupons + tracking codes) avec k cible 0.20 M9.
10. **Premier paid spend** ($200/mo Google Search FR sur wedge W5, M6 si triggers verts).
11. **Discord privé paid-only** actif (≤200 membres = 3-5h/sem modération solo).
12. **Newsletter Substack mensuelle** lead-magnet, ≥250 abonnés.

### 2.3 Définition « durable d'ici M+12 »

13. **MRR $5-7k base case, 200-296 paid users** (eval_28 §9.2 robust scenario à 70% du modèle).
14. **LTV/CAC > 3.0** mesuré sur cohort M6-M9.
15. **Churn < 8%/mo** sur cohort active 90j+.
16. **NPS > 30** sur cohort beta payant.
17. **Concept B "Co-Pilot"** déployé en façade web (cf. `docs/value/best_product_concept.md:61-122`), pas juste signaux Telegram.

### 2.4 Hypothèses fondatrices (à valider, pas à postuler)

- **H1** : Marc paie $29-49/mo si on lui montre 60j live track record et qu'on parle français. Source : eval_25 §2 + `reports/eval_27_pricing.md:165-167`. **À valider en interviews J+21.**
- **H2** : Le wedge FR a une concurrence quasi-nulle (audit SERP avril 2026, 0 site FR ranking top 5 sur `signaux trading or` → eval_28 §2.3 W5). **À re-vérifier S1.**
- **H3** : Le LLM-narrative reste différenciant 12 mois (eval_26 §6.1 : TradingView Copilot beta mars 2026 = menace 35% à 12 mois). **À monitorer trimestriellement.**
- **H4** : Solo founder peut soutenir 8-9h/sem marketing pendant 12 mois sans burnout (eval_28 §3.1). **Mitigation** : batch dimanche 14-18h non négociable.
- **H5** : Pricing $29 STARTER ne brûle pas le tier FREE (Marc churn de $0 vers $29 acceptable si proof track record). **À A/B-tester M5+.**

---

## 3. Gap analysis détaillée

> Légende sévérité : **P0** = bloqueur go-live commercial (ne peut pas vendre 1€ sans). **P1** = performance / fiabilité commerciale (peut vendre mais marge ou conversion sous-optimale). **P2** = nice-to-have post-launch.
> Légende statut : 🔴 absent · 🟡 partiel · 🟢 livré.

### 3.1 Tableau gap analysis (38 items)

| # | Item | Current | Target | Sévérité | Source |
|---|------|---------|--------|----------|--------|
| **PRICING & MONÉTISATION** | | | | | |
| 1 | Grille pricing publique | 🔴 hardcoded BP $49/$99/$149 | 🟢 FREE/$29/$79/$1990 + annual | **P0** | eval_27 §0 |
| 2 | Page `/pricing` web | 🔴 inexistante | 🟢 dark theme 4 cards + comparator + FAQ | **P0** | mockups/pricing_bundles.md |
| 3 | Stripe products + price IDs | 🔴 non créés | 🟢 8 prix (4 mensuel + 4 annuel) | **P0** | eval_24 §6.3 |
| 4 | Stripe Checkout + Customer Portal | 🔴 non câblé | 🟢 hosted checkout + portail self-service | **P0** | — |
| 5 | TVA UE + EU OSS reporting | 🔴 non configuré | 🟢 Stripe Tax activé, taux par pays | **P0** | eval_29 compliance |
| 6 | Webhook Stripe → SignalStore tier | 🔴 absent | 🟢 `customer.subscription.created/updated/deleted` → upgrade tier | **P0** | — |
| 7 | Trial 14j sans carte (FREE) ET 14j avec carte (PRO) | 🔴 absent | 🟢 dual-mode trial | **P0** | eval_27 §8 |
| 8 | Decoy display "INSTITUTIONAL $1990" visible | 🔴 absent | 🟢 4 cards toujours visibles | **P1** | eval_27 §4.3 |
| 9 | Hard caps signaux/mois par tier | 🔴 dead code | 🟢 Redis counter + soft 80% + hard 100% + upgrade CTA | **P0** | eval_24 §11 |
| 10 | Refund / money-back 30j policy | 🔴 absente | 🟢 CGV explicite + Stripe refund automatisé | **P0** | eval_29 |
| **ICP & DISCOVERY** | | | | | |
| 11 | 15 interviews Persona A FR/EN | 🔴 0 | 🟢 ≥15 enregistrées + dashboard JTBD | **P0** | eval_25 §11 |
| 12 | Onepager ICP commercial (interne) | 🔴 absent | 🟢 1 page format Notion + Confluence ready | **P1** | — |
| 13 | Script interview FR + EN | 🟢 livré | 🟢 — | — | eval_25 §7.1/7.2 |
| 14 | Calendrier Calendly + Zoom branded | 🔴 absent | 🟢 Calendly Pro $12/mo + Zoom | **P0** | — |
| 15 | Dashboard interviews (verbatims, WTP, objections) | 🔴 absent | 🟢 Google Sheet ou Notion DB | **P0** | — |
| **LANDING & SITE** | | | | | |
| 16 | Domaine + DNS + hosting | 🟡 domaine seul réservé | 🟢 smartsentinel.ai + /fr/, Vercel | **P0** | — |
| 17 | Landing master EN (hero V3 + features + pricing + FAQ + footer compliance) | 🔴 inexistante | 🟢 Lighthouse ≥95, ≥4% CVR | **P0** | eval_25 §8 |
| 18 | Landing master FR (hero V3 traduit + wedge W5) | 🔴 inexistante | 🟢 idem EN | **P0** | eval_28 §2.3 W5 |
| 19 | 5 landings wedge dédiées (W1-W5) | 🔴 0 | 🟢 5 pages SEO-optimized | **P1** | eval_28 §2.3 |
| 20 | A/B test framework hero V3 vs V4 | 🔴 absent | 🟢 Vercel split / PostHog Experiments | **P1** | eval_25 §8 |
| 21 | Lead magnet PDF (FR + EN, ex: "Le guide du trader XAU SMC français") | 🔴 absent | 🟢 2 PDF 12-20p | **P1** | eval_28 §10 levier 1 |
| 22 | Cookies consent + RGPD opt-in | 🔴 absent | 🟢 Klaro ou CookieYes, 4 catégories | **P0** | eval_29 |
| 23 | `/terms`, `/privacy`, `/refund`, `/legal` | 🟢 W1 livré endpoints API | 🟢 versions HTML web | **P0** | sprint_w1_compliance_2026_04_29.md |
| **TRACK-RECORD & PROOF** | | | | | |
| 24 | Telegram public channel "Smart Sentinel — Public Tape" | 🔴 absent | 🟢 60-90j signaux + commentaire narrative + screenshots P&L | **P0** | eval_25 §11.2 |
| 25 | Page `/track-record` publique | 🔴 absente | 🟢 dashboard read-only export CSV signé | **P0** | eval_26 Diff #1 |
| 26 | MyFXBook account (verified) | 🔴 absent | 🟢 lié EA MT5 ou import manuel | **P1** | eval_26 §5 plainte #2 |
| 27 | Audit trail signal CSV export | 🟡 SignalStore code OK, pas exposé public | 🟢 endpoint `/api/v2/track-record/export.csv` | **P0** | — |
| 28 | Disclaimer "paper-trading / lecture algorithmique" permanent | 🟡 partiel (sprint W1) | 🟢 footer + cards | **P0** | eval_29 + UE 2024/2811 |
| **CONTENU SEO & SOCIAL** | | | | | |
| 29 | 12 articles 90j (8 EN + 4 FR) | 🔴 0 | 🟢 12 publiés | **P1** | eval_28 §3.2 |
| 30 | YouTube channel + 3 vidéos | 🔴 0 | 🟢 channel + 3 vid M3 | **P1** | eval_28 §3.2 |
| 31 | Pine Script TradingView publié | 🔴 0 | 🟢 1 indicator gratuit, ≥100 likes | **P1** | eval_28 §3.2 S3 |
| 32 | Twitter/X automation Buffer (15 tweets/sem) | 🔴 0 | 🟢 Buffer $6/mo + queue batched | **P2** | eval_28 §3.1 |
| 33 | Newsletter Substack (digest mensuel) | 🔴 0 | 🟢 ≥250 abonnés M3 | **P2** | eval_28 §3.1 |
| **COMMUNAUTÉ** | | | | | |
| 34 | Discord privé paid-only (Wick bot, roles tier) | 🔴 absent | 🟢 channels par tier, Q&A live mardi 19h | **P1** | eval_28 §4.2 |
| 35 | Combot Telegram automod | 🔴 absent | 🟢 free tier suffit | **P1** | eval_28 §4.2 |
| **REFERRAL & VIRAL** | | | | | |
| 36 | Programme referral (1 mois Analyst/filleul) | 🔴 absent | 🟢 Stripe coupons + UTM codes + leaderboard | **P1** | eval_28 §7 |
| **ATTRIBUTION & ANALYTICS** | | | | | |
| 37 | GA4 + Meta pixel + UTM standard | 🔴 0 | 🟢 events landing/signup/trial/paid/cancel + cohorts | **P0** | eval_28 §8.2 |
| 38 | PostHog ou Plausible (alternative privacy) | 🔴 0 | 🟢 PostHog cloud free tier 1M events | **P1** | — |

### 3.2 Comptage par sévérité

- **P0** : 22 items (bloquent toute facturation).
- **P1** : 13 items.
- **P2** : 3 items.

### 3.3 Top 5 risques business si P0 non livrés

1. **Stripe Tax UE non configuré** → infraction TVA 2025 → amende + fermeture compte Stripe → 0€ encaissés post-fermeture.
2. **Pas de track-record public** → impossible de vendre à $29 sans prouver PF>1.0 → CAC infini.
3. **INSTITUTIONAL $149 maintenu** → marge négative dès le 1er client si Opus narrative full → loss-making par construction.
4. **Pas de cookies consent + opt-in RGPD** → infraction CNIL/EU GDPR + interdiction Google Ads → blocage growth canal #1.
5. **Pas de hard cap signaux** → 1 user abusif sur tier FREE consomme 5 000 narratives Sonnet/mois → $40/mo perte sur 1 free user (cf. eval_24 §8 stress S3).

---

## 4. Plan d'exécution

> **Format imposé** par le cahier des charges : chaque tâche P0 = titre + fichiers/livrables + effort (h) + critères d'acceptation + dépendances.
> **Owner** : Loukmane sauf mention contraire. Pas de sous-traitance avant M6 sauf legal one-shot.

### P0 — Bloquants go-live

Total P0 : **22 tâches, ~158h**, 8-10 semaines calendaire sur 16-20h/sem.

#### P0-T1 — Lock grille pricing v1 (modélisation + décision)

- **Titre** : Locker la grille pricing v1 FREE / STARTER $29 / PRO $79 / INSTITUTIONAL $1990 + annual 16.7% off
- **Livrables / fichiers** :
  - `reports/commercialization_sprint/02_pricing_v1_lock.md` — décision écrite avec marge brute par tier, sensibilité, justifications PSM, anchoring.
  - Update `BUSINESS_PLAN_SMART_SENTINEL.md` §6 (remplacer ancienne grille).
  - Update `src/api/tier_manager.py:36-67` (remplacer enums + prix + caps).
- **Effort** : **6h** (4h modélisation + 2h doc).
- **Critères d'acceptation** :
  - [ ] Doc reprend les 4 tiers avec ARPU mensuel + annuel + marge brute % calculée (eval_24 §6.5 formulas).
  - [ ] INSTITUTIONAL contrat 12 mois mini documenté (eval_27 §7.4).
  - [ ] Decoy effect justifié chiffré (eval_27 §4.2).
  - [ ] Hard caps signaux/mois par tier explicites (200 / 800 / 2000) (eval_24 §11).
  - [ ] Validé par soi-même + un peer review (idéalement 1 trader Persona A consulté).
- **Dépendances** : aucune (peut démarrer immédiatement).

#### P0-T2 — Créer Stripe products + 8 price IDs (mensuel + annuel)

- **Titre** : Configurer Stripe live avec products + prices + tax + coupons
- **Livrables / fichiers** :
  - Stripe Dashboard : 4 products (FREE, STARTER, PRO, INSTITUTIONAL) avec 8 prices (USD mensuel + annuel).
  - Stripe Tax activé pour FR/UE.
  - `.env.production` : `STRIPE_PUBLIC_KEY_LIVE`, `STRIPE_SECRET_KEY_LIVE`, `STRIPE_WEBHOOK_SECRET`, price IDs.
  - `infrastructure/stripe_config.yaml` (référence des price IDs versionnée).
- **Effort** : **5h**.
- **Critères d'acceptation** :
  - [ ] 8 prices visibles dans Stripe Dashboard avec metadata `tier=...` et `interval=...`.
  - [ ] Stripe Tax configuré FR (TVA 20%) + UE pays principaux + reverse charge B2B EU.
  - [ ] Test Checkout en mode TEST : carte 4242 → subscription créée → webhook reçu.
  - [ ] Coupon code `BETA50` créé (50% off 1er mois) prêt pour soft launch.
  - [ ] Page de pricing affiche prix incl. TVA pour visiteurs FR (geo-IP).
- **Dépendances** : P0-T1 (grille verrouillée).

#### P0-T3 — Câbler Stripe Checkout + Customer Portal dans l'API

- **Titre** : Brancher `/api/v2/billing/checkout` et `/api/v2/billing/portal` + webhook handler
- **Livrables / fichiers** :
  - `src/api/routes/billing.py` (nouveau) — endpoints `POST /checkout/session`, `POST /portal/session`, `POST /webhooks/stripe`.
  - `src/api/billing_store.py` (nouveau) — table `subscriptions` (user_id, stripe_customer_id, tier, status, current_period_end).
  - `tests/test_billing_routes.py` (≥10 tests).
  - Webhook handler gère `checkout.session.completed`, `customer.subscription.updated`, `customer.subscription.deleted`, `invoice.payment_failed`.
- **Effort** : **12h**.
- **Critères d'acceptation** :
  - [ ] User crée Checkout session côté front, paie en mode TEST, le tier passe dans `signal_store.users` dans les 5 secondes (event-driven).
  - [ ] Customer Portal permet cancel, change plan, update card.
  - [ ] Webhook idempotent (replay du même event = no-op).
  - [ ] Coverage ≥85% sur `billing_store.py`.
  - [ ] Logs structurés JSON avec `subscription_event_type`, `user_id`, `amount_cents`.
- **Dépendances** : P0-T2.

#### P0-T4 — Hard caps signaux/mois + soft-cap UX

- **Titre** : Implémenter compteurs Redis + soft-cap 80% + hard-cap 100% + upgrade CTA
- **Livrables / fichiers** :
  - `src/intelligence/quota_manager.py` (nouveau) — `consume_signal(user_id, tier) -> {allowed, remaining, soft_warning}`.
  - Hook dans `src/intelligence/sentinel_scanner.py` avant publication signal.
  - Telegram message templates "80% used" + "100% reached — upgrade".
  - `tests/test_quota_manager.py`.
- **Effort** : **8h**.
- **Critères d'acceptation** :
  - [ ] Compteur reset mensuel UTC.
  - [ ] FREE bloqué après 30 signaux, STARTER après 200, PRO après 800, INSTITUTIONAL après 2000 (eval_24 §11).
  - [ ] Soft-cap 80% envoie Telegram + email + dashboard banner.
  - [ ] Hard-cap génère event `tier_overflow` capté en analytics.
  - [ ] Pas de bypass via API (gating au niveau scanner + endpoint).
- **Dépendances** : P0-T3.

#### P0-T5 — Configurer geo-block US/QC/UK + disclaimer dynamique

- **Titre** : Étendre middleware compliance sprint W1 au front + Stripe Checkout
- **Livrables / fichiers** :
  - `src/api/middlewares/geo_block.py` (déjà partiellement livré W1, à vérifier) — confirmer block sur `/api/v2/billing/*`.
  - Front : intercept landing avec banner "Service indisponible dans votre pays" si IP US/QC/UK/OFAC.
  - Stripe : restriction par pays activée sur products INSTITUTIONAL (B2B only).
- **Effort** : **4h** (90% déjà fait sprint W1).
- **Critères d'acceptation** :
  - [ ] Test depuis VPN US → landing redirige `/restricted-region`, pas de CTA paiement visible.
  - [ ] Test depuis IP FR → CTA visible, TVA appliquée, disclaimer FR.
  - [ ] Audit log des refus géo dans `compliance_log.jsonl`.
- **Dépendances** : sprint W1 livré (✅).

#### P0-T6 — Money-back 30j policy + Stripe refund automation

- **Titre** : Documenter et automatiser la politique de remboursement 30j
- **Livrables / fichiers** :
  - `docs/legal/refund_policy_v1.md` (FR + EN).
  - Page `/refund` HTML reprenant policy.
  - `src/api/routes/billing.py` : endpoint `POST /refund/request` qui crée un Stripe Refund si <30j depuis premier paiement.
  - Email template "Refund processed".
- **Effort** : **4h**.
- **Critères d'acceptation** :
  - [ ] Policy claire : 30j first month only, no questions asked.
  - [ ] Bouton "Request refund" dans Customer Portal.
  - [ ] Refund traité en <24h ouvré, email confirmation auto.
  - [ ] Audit trail dans `billing_store.refunds`.
- **Dépendances** : P0-T3.

#### P0-T7 — Domaine, DNS, hosting Vercel + sous-domaines

- **Titre** : Mettre en ligne smartsentinel.ai + /fr/ + /track-record + /pricing
- **Livrables / fichiers** :
  - DNS : A + CNAME pointés vers Vercel.
  - Vercel project linked au repo `landing-smartsentinel` (à créer).
  - Routes : `/`, `/fr/`, `/pricing`, `/track-record`, `/terms`, `/privacy`, `/refund`, `/about`, `/contact`.
  - HTTPS forcé, redirect www→apex.
- **Effort** : **3h**.
- **Critères d'acceptation** :
  - [ ] Lighthouse ≥95 perf + SEO sur `/` et `/fr/`.
  - [ ] HTTPS A+ rating sur SSL Labs.
  - [ ] Sitemap.xml + robots.txt prod.
  - [ ] OG tags + Twitter cards configurés.
- **Dépendances** : aucune (peut paralléliser).

#### P0-T8 — Landing master EN (hero V3 + features + pricing + FAQ)

- **Titre** : Construire la landing principale EN avec messaging "Stop revenge trading"
- **Livrables / fichiers** :
  - `landing-smartsentinel/pages/index.tsx` (Next.js ou Astro).
  - Sections : Hero V3, "How it works" (3 steps), Features (6 cards), Pricing (4 cards), Track Record teaser, FAQ (12 Q), Footer compliance.
  - Copy : versions hero V3 et V4 prêtes pour A/B.
  - Images : screenshots Telegram + dashboard mockup (`mockups/webapp_b2c.html` references).
- **Effort** : **14h** (8h dev + 6h copy).
- **Critères d'acceptation** :
  - [ ] CVR baseline mesurée ≥3% visiteur → signup free dans la 1ère semaine (eval_28 §9.1).
  - [ ] Mobile responsive 100% (Tailwind / Chakra).
  - [ ] CTA primaire "Start free — no card" + secondaire "See live signals".
  - [ ] Hero text test V3 ("Stop revenge trading. Start trading with clarity.") vs V4 ("Trade like the institutions") configuré dans Vercel Edge Config ou PostHog Experiments.
  - [ ] FAQ inclut: "Is this financial advice?" / "What's the refund policy?" / "Do you guarantee profit?" (eval_29 anti-finfluencer).
- **Dépendances** : P0-T1 (grille pricing), P0-T7 (hosting), P0-T22 (analytics).

#### P0-T9 — Landing FR master (traduction + adaptations cultural FR-first)

- **Titre** : Adapter la landing pour le marché FR (wedge W5)
- **Livrables / fichiers** :
  - `landing-smartsentinel/pages/fr/index.tsx`.
  - Copy FR (Loukmane natif).
  - Hero V3 FR : "Arrêtez de tilter. Tradez avec lucidité."
  - Section dédiée "Pourquoi français ?" (founder-fit).
  - Pricing affichée TTC TVA 20%.
- **Effort** : **8h**.
- **Critères d'acceptation** :
  - [ ] CVR FR ≥4% (founder-fit + low competition).
  - [ ] Hreflang tags FR/EN propres.
  - [ ] FAQ FR adaptée (refund 14j minimum loi Hamon, etc.).
  - [ ] Disclaimer FR conforme AMF (eval_29).
- **Dépendances** : P0-T8 (template).

#### P0-T10 — Page `/pricing` interactive (toggle annuel/mensuel + comparator)

- **Titre** : Construire la page pricing autonome avec toggle, comparator, FAQ pricing
- **Livrables / fichiers** :
  - `landing-smartsentinel/pages/pricing.tsx` (+ `/fr/pricing`).
  - 4 cards toujours visibles (FREE / STARTER / PRO / INSTITUTIONAL) + decoy effect (eval_27 §4).
  - Toggle "Save 2 months" (annual).
  - Feature comparator table 25 lignes.
  - FAQ : "Can I change plan?", "Is there a free trial?", "Refund policy", "What is INSTITUTIONAL?".
- **Effort** : **6h**.
- **Critères d'acceptation** :
  - [ ] Toggle persiste in-session.
  - [ ] Click "Subscribe PRO" → Stripe Checkout en <2s.
  - [ ] INSTITUTIONAL bouton = "Book a demo" (Calendly intégré).
  - [ ] Mobile : cards stackées, comparator scrollable.
- **Dépendances** : P0-T3, P0-T8.

#### P0-T11 — Cookies consent + RGPD opt-in (Klaro/CookieYes)

- **Titre** : Bandeau cookies conforme RGPD + opt-in granulaire
- **Livrables / fichiers** :
  - Intégration Klaro (open-source) ou CookieYes ($10/mo).
  - 4 catégories : Necessary / Analytics / Marketing / Functional.
  - Page `/cookies` détaillée.
  - Tag manager : GA4 + Meta pixel + PostHog **conditionnés** au consent.
- **Effort** : **3h**.
- **Critères d'acceptation** :
  - [ ] Pas de tag tiers exécuté avant consent (validé via réseau dev tools).
  - [ ] Consent log persistant (Klaro localStorage + serveur).
  - [ ] CTA "Tout accepter" + "Refuser" + "Personnaliser" — pas de dark pattern (CNIL guideline 2022).
- **Dépendances** : P0-T7.

#### P0-T12 — Endpoints HTML `/terms`, `/privacy`, `/refund`, `/legal`

- **Titre** : Versions HTML des CGU/CGV/Privacy publiées
- **Livrables / fichiers** :
  - 4 pages HTML générées depuis Markdown sources (`docs/legal/`).
  - Liens footer permanent sur toutes les pages.
  - Versions FR + EN.
- **Effort** : **3h** (contenu déjà 80% rédigé sprint W1).
- **Critères d'acceptation** :
  - [ ] Pages indexables (pas de noindex).
  - [ ] Version + date publication visible (versioning légal).
  - [ ] Lien CGV obligatoire dans Stripe Checkout (Stripe Tax l'exige).
- **Dépendances** : sprint W1 ✅, P0-T7.

#### P0-T13 — Telegram channel public "Smart Sentinel — Public Tape"

- **Titre** : Créer le canal Telegram public et publier les 60-90 premiers jours
- **Livrables / fichiers** :
  - Channel Telegram (public, lecture seule, Combot anti-spam).
  - Pinned message : disclaimer "paper-trading / educational" + link `/track-record`.
  - Bot Sentinel push automatiquement les signaux algo (déjà câblé `src/delivery/telegram_notifier.py`).
  - Image branding 1280×640.
- **Effort** : **4h** (créa + setup) + 30 min/sem maintenance.
- **Critères d'acceptation** :
  - [ ] Channel live, ≥1 signal/jour publié automatiquement.
  - [ ] Combot config strict (anti-spam, anti-bot DM).
  - [ ] T.me listing optimisé (description SEO `XAU smart money signals — public tape`).
  - [ ] 50+ abonnés organiques M+1 (cible).
- **Dépendances** : produit livre des signaux en continu (✅).

#### P0-T14 — Page `/track-record` publique read-only

- **Titre** : Dashboard public exportable avec historique signaux + PnL + métriques
- **Livrables / fichiers** :
  - `src/api/routes/track_record.py` (nouveau) — endpoint `GET /track-record/summary` + `/track-record/export.csv`.
  - `landing-smartsentinel/pages/track-record.tsx` — affichage : nb signals, PF, hit rate, max DD, avg R, 30 derniers signaux table.
  - SignalStore query optimisée (index sur `created_at`).
  - Signature CSV (HMAC) pour anti-falsification.
- **Effort** : **10h**.
- **Critères d'acceptation** :
  - [ ] Données refreshed toutes les 5 min.
  - [ ] Filtres : période (7j/30j/90j/all), instrument (XAU/EUR/...).
  - [ ] Export CSV signé HMAC (header `X-Signature`).
  - [ ] Charts : equity curve, distribution R-multiples.
  - [ ] Tagger explicite "Paper-trading / Educational. Past performance ≠ future results." (eval_29).
- **Dépendances** : P0-T7, SignalStore prêt (✅).

#### P0-T15 — Onboarding 5 étapes post-checkout

- **Titre** : Mailer + dashboard onboarding pour nouveaux paid users
- **Livrables / fichiers** :
  - Email séquence (Resend) : Welcome / Telegram setup / Dashboard tour / First signal expectations / Day 7 check-in.
  - Page `/welcome` (post-checkout) : 5 étapes interactives (link Telegram, set timezone, choisir instruments, configurer alerts, link broker optional).
  - Calendly link pour appel onboarding 30 min (INSTITUTIONAL only).
- **Effort** : **8h** (5h dev + 3h email copy).
- **Critères d'acceptation** :
  - [ ] Time-to-first-signal < 5 min après paiement (Telegram link + bot activated).
  - [ ] Email open rate ≥45% (Resend analytics).
  - [ ] Onboarding completion rate ≥60%.
  - [ ] Mesuré en analytics event-by-event.
- **Dépendances** : P0-T3.

#### P0-T16 — Interviews discovery Persona A (15 bouclées)

- **Titre** : Conduire 15 interviews user research Persona A (Marc XAU SMC retail FR/EN)
- **Livrables / fichiers** :
  - Calendly Pro $12/mo set up avec 30min slots.
  - 15 enregistrements Zoom (cloud) + transcripts.
  - Dashboard Google Sheet : 15 lignes × 20 colonnes (JTBD #1/2/3, WTP $20/$49/$79, objections, outils actuels, etc.).
  - 1 page synthèse patterns + verbatims marquants.
- **Effort** : **25h** (15 × 30min interview + 15 × 60min écoute/synthèse + 5h sourcing/outreach).
- **Critères d'acceptation** :
  - [ ] ≥10 Persona A FR + ≥3 Persona A EN + ≥2 contrôle (Persona B prop firm pour confirmer red-team).
  - [ ] ≥7/15 confirment WTP $20-49 ⇒ déclenche GO niche A (eval_25 §11.3).
  - [ ] WTP median documenté + 3 objections principales adressées dans FAQ landing.
  - [ ] Décision GO/NO-GO niche A documentée dans `reports/commercialization_sprint/03_decision_niche_a.md`.
- **Dépendances** : P0-T14 (Calendly setup), produit a quelque chose à montrer (mockup ou demo Loom OK pré-Phase 1).

#### P0-T17 — GA4 + Meta pixel + UTM standardisés

- **Titre** : Setup tracking complet visiteur→signup→trial→paid
- **Livrables / fichiers** :
  - GA4 property smartsentinel.ai (events conf : `view_landing`, `signup_free`, `start_trial`, `purchase`, `cancel`).
  - Meta pixel (events `Lead`, `CompleteRegistration`, `StartTrial`, `Subscribe`).
  - Convention UTM documentée : `?utm_source=reddit&utm_medium=ama&utm_campaign=r-forex-jul26&utm_content=ama-thread`.
  - Server-side Stripe → GA4 (purchase event server-side via Measurement Protocol pour iOS 14+ tracking).
- **Effort** : **6h**.
- **Critères d'acceptation** :
  - [ ] Test end-to-end : visit landing via UTM `?utm_source=test` → signup → trial → purchase → GA4 event funnel visible dans Explorer.
  - [ ] Conditionnel au cookies consent (P0-T11).
  - [ ] Conversion API Meta CAPI configurée (server-side).
- **Dépendances** : P0-T7, P0-T11, P0-T3.

#### P0-T18 — Trial logic 14j (FREE sans carte + PRO avec carte)

- **Titre** : Implémenter dual-mode trial
- **Livrables / fichiers** :
  - Logique Stripe trial 14j sur PRO (subscription avec `trial_period_days=14`).
  - FREE = pas de trial, accès permanent capped à 30 signaux/mois.
  - Email J+7 (trial PRO) : reminder trial ends in 7d + features used.
  - Email J+13 : trial ends tomorrow.
- **Effort** : **5h**.
- **Critères d'acceptation** :
  - [ ] Card on file required PRO trial → conversion trial→paid mesurée ≥20% (eval_28 §9.1).
  - [ ] FREE peut upgrader à PRO sans re-créer compte.
  - [ ] Trial cancellable depuis Customer Portal.
- **Dépendances** : P0-T3.

#### P0-T19 — Webhook Stripe → SignalStore tier upgrade

- **Titre** : Synchroniser tier dans `signal_store.users` depuis events Stripe
- **Livrables / fichiers** :
  - Handler `customer.subscription.updated` → update `users.tier`.
  - Handler `customer.subscription.deleted` (cancel) → downgrade à FREE.
  - Handler `invoice.payment_failed` → retry logic + Dunning email J+3, J+7, J+14, downgrade J+15.
- **Effort** : **6h**.
- **Critères d'acceptation** :
  - [ ] Latence event Stripe → tier update < 5s P95.
  - [ ] Idempotent (replay events safe).
  - [ ] Tests intégration end-to-end : create sub → update tier → cancel → tier=FREE.
- **Dépendances** : P0-T3, P0-T4.

#### P0-T20 — Disclaimer "paper-trading / educational" sur tous les supports

- **Titre** : Conformité messaging finfluencer (UE 2024/2811 + AMF)
- **Livrables / fichiers** :
  - Update `mockups/telegram_b2c.txt` (déjà partiellement adapté W1) — confirmer wording.
  - Footer landing : "Smart Sentinel AI n'est pas un conseil en investissement. Lecture algorithmique éducative. Performance passée ≠ performance future."
  - Telegram pinned message identique.
  - Track-record page tagger.
- **Effort** : **3h**.
- **Critères d'acceptation** :
  - [ ] Audit checklist eval_29 cochée intégralement (`reports/eval_29_compliance_findings.md`).
  - [ ] Vocabulaire : "lecture / analyse / scan / signal éducatif" et **jamais** "conseil / recommandation d'achat / signal de trading garanti".
  - [ ] Versioning des disclaimers dans `docs/legal/disclaimers_v1.md`.
- **Dépendances** : sprint W1 ✅.

#### P0-T21 — Lead magnet PDF FR + EN

- **Titre** : 2 PDFs cornerstone "Guide du trader XAU SMC français" + "ICT × AI Playbook 2026"
- **Livrables / fichiers** :
  - 2 PDFs 12-20 pages chacun.
  - Landing dédiée `/guide-xau-fr` + `/smc-ai-playbook` avec form opt-in email.
  - Connecté Resend (Substack alternative) pour double opt-in + séquence nurture 5 emails.
- **Effort** : **14h** (10h rédaction + 4h design Canva).
- **Critères d'acceptation** :
  - [ ] Download rate ≥30% landing visit.
  - [ ] Email capture ≥80% downloads.
  - [ ] Bottom CTA dans PDF : "Try Smart Sentinel free".
- **Dépendances** : P0-T7.

#### P0-T22 — Decoy display + comparator (page pricing)

- **Titre** : Maximiser conversion PRO via decoy INSTITUTIONAL toujours visible
- **Livrables / fichiers** :
  - Composant `PricingGrid` avec 4 cards en ligne (FREE / STARTER / **PRO badge "Most popular"** / INSTITUTIONAL).
  - Badge "Most popular" sur PRO.
  - Comparator détaillé 25 features × 4 tiers.
- **Effort** : **4h**.
- **Critères d'acceptation** :
  - [ ] Mesurer impact decoy : A/B test 1 mois "decoy visible" vs "decoy hidden". Cible : +25% conv vers PRO (eval_27 §4.2).
- **Dépendances** : P0-T10.

### P1 — Performance & reliability (post go-live, M2-M6)

Total P1 : **13 tâches, ~74h**.

#### P1-T23 — 12 articles SEO publiés (8 EN + 4 FR) sur 90j

- **Livrables** : 12 articles dans `/blog` (CMS Astro ou Ghost).
- **Effort** : **48h** (4-6h × 12 articles batched dimanches).
- **Critères** : top 5 SERP FR sur `signaux trading or` à M+3, top 10 EN sur `ICT signals telegram`.
- **Dépendances** : P0-T7, P0-T11, P0-T17.

#### P1-T24 — Pine Script TradingView publié (gratuit, lead magnet)

- **Livrables** : 1 indicator Pine Script publié sur TradingView marketplace gratuit.
- **Effort** : **6h**.
- **Critères** : ≥100 likes M+2, ≥50 emails captés via lien dans description.

#### P1-T25 — YouTube channel + 3 vidéos M3

- **Effort** : **12h** (4h × 3 vidéos batched).
- **Critères** : 500 vues cumul M3, 30+ commentaires, 1.5% CTR vers landing.

#### P1-T26 — Newsletter Substack mensuelle

- **Effort** : **2h** setup + 3h/mois.
- **Critères** : ≥250 abonnés M+3, open rate ≥35%.

#### P1-T27 — Discord privé paid-only (Wick bot + tier roles)

- **Livrables** : Discord server, channels par tier, Wick anti-raid, bot Stripe-sync pour assigner roles.
- **Effort** : **8h** setup + 3-5h/sem modération.
- **Critères** : ≥50 paid users actifs M+6, NPS ≥40, retention M3 ≥67% (eval_28 §4.3).

#### P1-T28 — Programme referral (Stripe coupons + tracking)

- **Livrables** : code unique par user, UTM, Stripe coupon 1 month free, leaderboard public.
- **Effort** : **8h**.
- **Critères** : k ≥ 0.20 M+9, CAC referral ≤$33 (eval_28 §7).

#### P1-T29 — A/B test framework hero V3 vs V4 (PostHog Experiments)

- **Effort** : **3h** setup + ongoing.
- **Critères** : winner identifié avec p<0.05 sur 4 semaines, CVR +20% sur winner.

#### P1-T30 — Email nurture séquence (5 touches free → trial)

- **Effort** : **6h** copy + setup Resend automation.
- **Critères** : conversion free → trial 5% M+6 (eval_28 §4.3 lead magnet bench).

#### P1-T31 — Reddit AMA r/Forex M+3

- **Effort** : **8h** prep + 8h jour J.
- **Critères** : ≥200 upvotes, ≥50 inscrits Telegram via UTM `reddit-ama-jul26`, ≥5 trial signups (eval_28 §6.2).

#### P1-T32 — Product Hunt launch M+5

- **Effort** : **20h** prep (mailing list, visuels, hunter coordination) + 16h jour J.
- **Critères** : top 10 daily, ≥250 upvotes, ≥200 signups via UTM `product-hunt`, ≥15 trial→paid (eval_28 §6.2).
- **Dépendances** : P0 done + P1-T23/24/25 (assets backed).

#### P1-T33 — Influencer Trader Pro FR (1 vidéo $400-600 sponsored)

- **Effort** : **8h** coordination + ROI tracking.
- **Critères** : Break-even 1 conversion, ROI 3-5× attendu (eval_28 §5.3).
- **Dépendances** : **PF backtest > 1.20 livré + 90j live track** (eval_28 §5.4 décision rule).

#### P1-T34 — MyFXBook verified account

- **Effort** : **6h** integration.
- **Critères** : compte MyFXBook public, signaux uploadés auto, lien sur landing.
- **Dépendances** : EA MT5 ou import script.

#### P1-T35 — Cohort analytics + churn dashboard

- **Effort** : **6h**.
- **Critères** : query SQL cohort par mois d'inscription, dashboard Metabase ou Retool, métriques LTV / churn / NRR exposées dans `/admin/analytics`.

### P2 — Nice-to-have post-launch

Total P2 : **6 tâches, ~22h**, à exécuter M6-M12.

- **P2-T36** : Bundles thématiques FX/Metal/Crypto/Index (`mockups/pricing_bundles.md`) — **4h**. Tester en M+6 sur cohort sélectionnée.
- **P2-T37** : Affiliate program brokers (IC Markets, Pepperstone IB rev share) — **6h**. Activer M+9 si DataFeed migration cohérente.
- **P2-T38** : Substack premium paid newsletter ($5-10/mo) — **3h**. Optionnel M+9.
- **P2-T39** : TikTok ads experiments — **2h** test. Probablement à éviter (eval_28 §8.4 CAC trading TikTok >$200).
- **P2-T40** : Multi-langue ES + DE (Klaro + Crowdin) — **5h** setup, contenu sous-traité.
- **P2-T41** : Webhook publique B2B-API (pré-pivot eval_26 §6.2 plan B) — **2h** scoping.

---

## 5. Tests & validation

### 5.1 A/B tests prioritaires (par ordre de valeur)

| # | Test | Hypothèse | Audience | Métriques | Durée | Effort |
|---|------|-----------|----------|-----------|-------|--------|
| 1 | Hero V3 ("Stop revenge trading") vs V4 ("Trade like institutions") | V3 winner (eval_25 §8) | EN + FR landing | CVR visit→signup, p<0.05 | 4 sem | 2h setup |
| 2 | Decoy INSTITUTIONAL visible vs hidden | +25% conv PRO (eval_27 §4.2) | Pricing page | CVR vers PRO | 4 sem | 1h setup |
| 3 | Pricing $29 vs $39 STARTER | $39 marge mais $29 funnel | EN landing | CVR signup→trial, ARPU | 4 sem | 1h |
| 4 | Trial 14j sans carte vs avec carte | Avec carte +400% trial→paid (Reforge) | EN landing | conversion trial→paid | 6 sem | 2h |
| 5 | "Save 2 months" vs "Save 20%" toggle | "2 months" plus parlant (eval_27 §5.5) | Pricing | CTR toggle annual, annual mix % | 4 sem | 1h |
| 6 | CTA "Start free" vs "See live signals" | Hypothèse : "See" 2x curiosity click | EN landing | CTR hero CTA | 2 sem | 1h |
| 7 | FR landing avec founder photo vs sans | Trust signal FR-first | FR landing | CVR | 4 sem | 1h |
| 8 | Onboarding 5 étapes complet vs simplifié 3 étapes | Trade-off completion vs activation | Paid users | activation rate, time-to-first-signal | 6 sem | 3h |

### 5.2 Attribution model (tracking sources de revenue)

- **Last touch UTM** par défaut (Stripe metadata `utm_source`, `utm_medium`, `utm_campaign` stocké sur subscription).
- **Multi-touch** : événements GA4 stockés 30j, modèle linear pour CAC blended.
- **Server-side server-to-server pour iOS 14+** : Stripe → Measurement Protocol GA4 + Meta CAPI.
- **Convention UTM** documentée dans `docs/marketing/utm_conventions_v1.md`.

### 5.3 Conversion tracking funnel

| Étape | Outil mesure | KPI | Cible M3 | Cible M12 |
|-------|--------------|-----|---------:|----------:|
| Visit landing | GA4 + Plausible | sessions /mo | 1 200 | 22 000 |
| Signup free | GA4 event `signup_free` + Stripe | conversion % | 3% | 5% |
| Free → Trial activated | Stripe `customer.created` + `subscription_schedule.created` | conversion % | 35% | 50% |
| Trial → Paid | Stripe `invoice.paid` first | conversion % | 20% | 28% |
| Onboarding completion | Custom event `onboarding_complete` | completion % | 50% | 70% |
| 7-day active | Custom event `user_active_7d` | retention % | 60% | 80% |
| 30-day active | idem | retention % | 40% | 65% |
| 90-day active (LTV proxy) | idem | retention % | n/a | 50% |
| Churn mensuel | Stripe MRR retention | % | n/a (early) | 6% |

### 5.4 Procédures de validation pré-launch (checklist commerciale)

- [ ] Test parcours achat complet avec 3 cartes (Visa, Mastercard, Amex) en TEST puis 1 paiement réel $1.
- [ ] Test refund en 1 clic depuis Customer Portal.
- [ ] Test geo-block depuis VPN US, QC, UK.
- [ ] Test cookie consent (refuser → aucun tag tiers).
- [ ] Test funnel attribution : visite UTM → signup → purchase → vérifier UTM stored Stripe metadata.
- [ ] Test downgrade FREE depuis Customer Portal → tier=FREE dans 5s.
- [ ] Test trial → expiration sans paiement → tier=FREE, pas de double-charge.
- [ ] Test failed payment → dunning email J+3.
- [ ] Test cancel subscription → access reste jusqu'à `current_period_end`.
- [ ] Test Telegram bot link generation post-purchase (lien unique non-réutilisable).
- [ ] Test mobile responsive sur iPhone Safari + Android Chrome.
- [ ] Test Lighthouse perf ≥95 sur landing + pricing.
- [ ] Test affichage compliant pour visiteurs FR (TVA TTC affichée).

---

## 6. Sécurité

### 6.1 RGPD opt-in

- **Cookies consent granulaire** : 4 catégories (Necessary obligatoire / Analytics opt-in / Marketing opt-in / Functional opt-in). Klaro ou CookieYes.
- **Tracking tiers conditionnel** : GA4, Meta pixel, PostHog **uniquement après consent** Analytics ou Marketing.
- **Droits utilisateur** : page `/account/data-rights` avec boutons "Download my data" (JSON export 30j SLA) + "Delete my data" (hard delete 30j SLA après confirmation email).
- **Registre des traitements** : `docs/legal/registre_rgpd_v1.md` listant 7 finalités (compte, billing, analytics, marketing email, support, monitoring sécurité, archivage légal).
- **DPO** : pas obligatoire solo founder < 250 employés, mais email contact RGPD documenté.
- **Politique de conservation** : 7 ans Stripe (obligation fiscale FR), 13 mois cookies analytics max, 3 ans inactif (puis purge auto).
- **Sous-traitants** : liste publique avec leur RGPD compliance (Stripe, Resend, GA4, PostHog, Anthropic, Railway, Vercel, Cloudflare).

### 6.2 Données clients

- **Hachage email** côté serveur pour analytics (SHA256 salé) — Meta CAPI accepte hash.
- **PII separation** : `signal_store.users` contient `user_id` seulement, table séparée `pii_store` avec email, country, name → encrypted at rest (AES-256), accès restreint.
- **Logs structurés** : zero PII dans logs JSON, redact via filter middleware.
- **Backup chiffré** : Backblaze B2 ($1/mo) + AES-256 client-side avant upload.
- **Pas d'API key user-side** : authentification via OAuth Telegram (déjà partiellement câblé) ou magic link email.
- **2FA optionnel** : sur INSTITUTIONAL tier obligatoire (TOTP via authentic.io ou Authy).

### 6.3 Paiements Stripe (sécurité)

- **Stripe Radar** activé (anti-fraude ML natif Stripe, $0.05/transaction).
- **3D Secure** forcé sur paiements >$50 (Stripe automatique).
- **Webhook signature** : `stripe_signature` header vérifié dans `src/api/routes/billing.py` (HMAC SHA256).
- **PCI compliance** : Stripe SAQ-A par Checkout (pas de carte stockée serveur Sentinel) — bandeau "Powered by Stripe" en footer.
- **Failed payment retry policy** : 3 attempts (J+3, J+7, J+14), puis downgrade auto, email final.
- **Dispute handling** : alerte email Loukmane sur `charge.dispute.created`, réponse < 24h, evidence kit prêt (audit trail signaux, screenshots usage, dates).
- **Stripe Connect** : non utilisé (pas de marketplace).

### 6.4 Fraude (referral + abus)

- **Referral fraud detection** : email + IP fingerprint, blacklist auto si 3 codes utilisés depuis même IP/24h, manuel review au-delà.
- **Trial abuse** : limite 1 trial par email + 1 par carte, Stripe Radar custom rule.
- **Coupon abuse** : `BETA50` limité à 100 utilisations totales + 1 par customer.
- **Bot signups** : Cloudflare Turnstile sur signup form (free).
- **Rate limiting** : middleware déjà câblé (100 req/min, cf. MEMORY production wiring) → étendre à 5 signup/h/IP.
- **Telegram link unique** : token signé HMAC, expire 24h, invalide après usage.

### 6.5 Réputation et compliance commerciale

- **Wording finfluencer** : audit régulier (mensuel) du contenu landing/blog vs grille AMF/CNMV/ESMA. Eval_29 livré sprint W1 ✅.
- **Anti-greenwashing performance** : tout claim chiffré (PF, hit rate) doit être sourcé `/track-record`, jamais inventé.
- **DMCA / image rights** : tous les screenshots TradingView/MT5 dans content marketing doivent être de notre compte propre, pas screenshots tiers.
- **Trademarks** : audit "Smart Sentinel AI" — vérifier non-conflit avec "Sentinel One" (cyber), "Sentinel BMS" (banking) — déposer marque EU si scaling envisagé M+6.

---

## 7. Métriques de succès

### 7.1 KPIs commerciaux (par horizon)

| KPI | Définition | M1 | M3 | M6 | M9 | M12 |
|-----|-----------|---:|---:|---:|---:|----:|
| **MRR** | Recurring revenue Stripe / mois | $67 | $300-500 | $1 500-3 000 | $5 000-7 000 | $7 000-10 000 |
| **ARR** | MRR × 12 | $800 | $4-6k | $18-36k | $60-84k | $84-120k |
| **Paid users actifs** | Stripe `active` subscriptions | 1 | 7 | 46 | 136 | 200-296 |
| **Trial conversion** | trial → paid 14j | n/a | 20% | 25% | 28% | 28% |
| **Free → Trial** | free signup → trial activated | n/a | 35% | 40% | 45% | 50% |
| **Landing CVR** | visit → signup | 3% | 3% | 4% | 4.5% | 5% |
| **CAC blended** | (paid spend $ + temps_founder × $50) / new_paid users | n/a | < $10 | < $25 | < $30 | < $35 |
| **CAC paid** | paid spend $ / new paid (only paid-acquired) | n/a | n/a | $40 | $50 | $60 |
| **CAC referral** | (1 month gratis × marge ARPU) / paid referrals | n/a | n/a | $33 | $30 | $28 |
| **LTV (cohort 90j)** | ARPU pondéré × 1/churn | n/a | n/a | $700 (proxy) | $900 | $1 108 |
| **LTV / CAC** | ratio | n/a | n/a | 3.0 | 4.0 | 5.0 |
| **Payback period** | mois pour récupérer CAC | n/a | 1 mois (organic) | 1.5 mois | 2 mois | 2.5 mois |
| **Churn mensuel** | (cancel + downgrade) / paid debut mois | n/a | n/a (n trop faible) | 8% | 7% | 6% |
| **NRR** | Net Revenue Retention (expansion - churn) | n/a | n/a | 100% | 105% | 110% |
| **NPS** | sondage post-30j | n/a | n/a | >30 | >40 | >50 |
| **Visites landing /mo** | GA4 sessions | 200 | 1 200 | 5 500 | 13 000 | 22 000 |
| **Telegram public abonnés** | channel members | 5 | 80 | 400 | 900 | 1 500 |
| **Newsletter abonnés** | Substack/Resend list | 10 | 100 | 400 | 800 | 1 500 |
| **Articles SEO publiés cumul** | blog count | 1 | 12 | 24 | 36 | 48 |
| **Backlinks référents** | Ahrefs DR count | 2 | 10 | 35 | 70 | 120 |
| **Position SERP `signaux trading or`** | Google FR | non | top 20 | top 10 | top 5 | top 3 |
| **% MRR annual mix** | annual subs / total subs | 0% | 0% | 25% | 35% | 40% |
| **MRR INSTITUTIONAL** | sub × $1990 | $0 | $0 | $0 | $1990 (1 client) | $3 980 (2 clients) |

### 7.2 KPIs commerciaux secondaires (santé du funnel)

- **Average ARPU** : cible $66-74 pondéré 70% STARTER / 25% PRO / 5% INSTITUTIONAL (eval_24 §12.2).
- **Time-to-first-signal** : <5 min après paiement.
- **Time-to-value** : <24h (premier signal valide reçu).
- **Trial conversion split par canal** : organique (cible 30%) vs referral (cible 35%) vs paid (cible 18%).
- **Cohort retention M3 / M6 / M12** : 75% / 60% / 45% (B2C SaaS trading median).
- **Coupon redemption rate** : <10% sur paid (=indicateur dépendance promo).

### 7.3 Métriques compliance & qualité

- **Disclaimer présent sur 100% des assets** (audit mensuel).
- **Refund rate** < 8% (au-delà = qualité produit ou pricing problème).
- **Stripe dispute rate** < 0.5% (au-delà = bandwidth Stripe Radar dégrade).
- **Onboarding completion** ≥ 60%.
- **Time-to-cancel-from-signup** distribution : <7j = "tilt churn" (mauvais fit), >30j = "value churn" (à investiguer).

### 7.4 Métriques de soutenabilité solo

- **Heures marketing/sem (cap 8-10h)** : auto-tracking via Toggl ou notebook. Alerte si >12h/sem 3 semaines consécutives → arbitrage immédiat (couper Twitter > YT > articles > Telegram dans cet ordre, eval_28 §15.2).
- **Heures support/sem** (cap 2-3h) : Intercom ou Crisp metric.
- **NPS founder satisfaction** : self-assessment mensuel pour détecter burnout.

---

## 8. Risques & mitigations

### 8.1 Tableau risques commerciaux (top 12)

| # | Risque | Probabilité | Impact $ | Sévérité | Mitigation | Owner |
|---|--------|------------|---------:|----------|-----------|-------|
| R1 | **Produit PF < 1.10 live → churn massif post-trial** | Haute (PF actuel 0.96) | -$15k ARR M12 | 🔴 critique | Subordonner go-live à **Phase 0 produit validée** (eval_25 §10) ; NO-GO Stripe live tant que PF<1.10 forward 60j | Catégorie 3 (perf) |
| R2 | **TradingView lance Alert Copilot natif + free** | Moyenne (35% à 12 mois) | -60% PRO conv | 🔴 critique | Plan B B2B-API (eval_26 §6.2 Scénario A) + pivoter wedge ultra-vertical XAU institutionnel ($499-999) | — |
| R3 | **LuxAlgo embarque Claude/GPT narrative** | Élevée (50% à 12 mois) | -30% différenciation | 🟠 haute | Accélérer Diff #1 track-record + Diff #2 hyper-spé XAU FR (eval_26 §6.2 Scénario B) | — |
| R4 | **Pricing INSTITUTIONAL $1990 absurde aux yeux Marc** | Moyenne | Decoy effect cassé | 🟠 haute | Page INSTITUTIONAL crédible (features réelles, pas placeholder) ; sinon retirer ou re-pricer $499 (eval_27 §4.3) | Loukmane |
| R5 | **Stripe Tax UE mal configuré → amende TVA** | Faible | -$3-5k amende | 🟠 haute | Stripe Tax activé dès le 1er paiement + audit accountant FR ($300 one-shot) | Loukmane |
| R6 | **Google Ads refuse compte (financial services policy)** | Moyenne | Bloque growth canal M+6 | 🟡 moyenne | Préparer LegalEntity verification J0, alternatives Reddit Promoted Posts + Twitter Ads | Loukmane |
| R7 | **Cold reply rate interviews < 5% → ICP non validé** | Moyenne | Décision GO/NO-GO mal informée | 🟡 moyenne | Diversifier 4 canaux d'outreach (Discord, Twitter reply-guy, LinkedIn warm, r/Forex) ; objectif 40 cold messages → 5 interviews | Loukmane |
| R8 | **Solo founder burnout (>12h/sem mkt × 4 sem)** | Moyenne | Arrêt complet 4-8 sem | 🟠 haute | Cap dur 9h/sem + batch dimanche obligatoire + arbitrage immédiat si dépassement (eval_28 §15) | Loukmane |
| R9 | **Discord/Telegram inondé de scams/bots** | Élevée | Réputation négative | 🟡 moyenne | Wick + Combot config strict + role gating obligatoire avant DM | Loukmane |
| R10 | **Trial abuse (1 user × 10 cartes virtuelles)** | Moyenne | -$200-500/mo loss | 🟢 faible | Stripe Radar custom rules + 1 trial / email + IP fingerprint | Loukmane |
| R11 | **Refund rate > 15%** | Moyenne | Marge brute affectée | 🟡 moyenne | Onboarding 5 étapes solides + Day 7 check-in email + customer call gratuit si refund demand | Loukmane |
| R12 | **Anthropic prix +100%** | Faible | INSTITUTIONAL marge -50% | 🟢 faible | Pricing INSTITUTIONAL avec clause re-négociable annuelle ; switch Sonnet narratives + Opus chat-only (eval_24 §7.3) | Loukmane |

### 8.2 Plans de contingence (3 scénarios)

#### Contingence A — PF reste < 1.10 forward à M+3

- **Action immédiate** : NE PAS activer Stripe live. Continuer Telegram public + content marketing pour build d'audience uniquement.
- **Pivot 1** : pivoter messaging du SaaS B2C vers "Smart Sentinel AI Lab — Open Research Newsletter" (Substack premium $10/mo, target 200 abonnés = $24k ARR proxy).
- **Pivot 2** : B2B-API early (pre-pivot) — pitch IC Markets / Pepperstone leur licence engine = $30-60k ARR par broker (eval_26 §6.2 plan B).
- **Runway** : 13.9 mois → couvre pivot.

#### Contingence B — Cold reply rate <5% sur interviews M+1

- **Diagnose** : messaging cold cassé. Refondre script J+14.
- **Switch canal** : abandonner outreach Discord/Twitter, switcher Reddit DM + LinkedIn warm via réseau.
- **Plan B** : ouvrir 1 canal payant micro-budget (LinkedIn Sales Navigator $80/mo + cold email Apollo $50/mo) pour 1 mois.

#### Contingence C — Concurrent (LuxAlgo + Claude) sort en M+3

- **Action immédiate** : pivoter messaging vers Diff #1 (track-record) + Diff #2 (XAU FR-first hyper-spé).
- **Comparatif honnête** publié : "LuxAlgo Insights vs Smart Sentinel : same Claude API, but who shows you the trades that lost?".
- **Accélérer MyFXBook integration** (eval_26 §6.2 Scénario B).

---

## 9. Dépendances autres catégories

| Catégorie | Livrable attendu | Bloquant pour | Deadline pour |
|-----------|------------------|---------------|---------------|
| **Catégorie 2 — Compliance & Legal** | Sprint W4 relecture CGU/CGV avocat ($2k one-shot, déjà commencé W1-W3) | P0-T6, P0-T12, P0-T20 | M+1 |
| **Catégorie 3 — Performance & Perf Engineering** | PF backtest > 1.10 net coût XAU M15 sur 60j forward | **TOUT le go-live commercial** (P0-T13 à T22 inutiles sans cela) | M+2 |
| **Catégorie 3 — Performance** | Hard caps signaux/mois implémentation Redis | P0-T4 | M+1 |
| **Catégorie 4 — Data Quality & Licensing** | Switch Dukascopy → TraderMade $49/mo (licence commerciale) | Stripe live (eval_24 §4) | M+1 |
| **Catégorie 5 — Observability** | GA4-compatible health endpoint / `/metrics` propre | P0-T17 (tracking serveur) | M+1 |
| **Catégorie 6 — Intelligence engine** | Concept B "Co-Pilot" UI (`docs/value/best_product_concept.md`) | Pricing PRO $79 défendable | M+3 |
| **Catégorie 7 — DevOps & Deployment** | Domaine prod stable, monitoring uptime ≥99.5% | P0-T7, INSTITUTIONAL SLA (eval_27 §7.5) | M+1 |
| **Catégorie 8 — Testing** | Tests E2E billing + tier upgrade (≥30 tests) | P0-T3, T19 | M+1 |
| **Catégorie 9 — UX / Frontend** | Implémentation `landing-smartsentinel` repo Next.js/Astro | P0-T7, T8, T9, T10 | M+1 |
| **Catégorie 10 — Content / Copy** | Articles cornerstones SEO (1 EN + 1 FR pillar) | P1-T23 | M+1 (start) |

**Path critique commercial** = Catégorie 3 (Perf) + Catégorie 4 (Data License) + Catégorie 9 (Frontend) **en parallèle** de cette catégorie 1.

---

## 10. Estimation totale & timeline

### 10.1 Effort total (en heures de Loukmane)

| Section | Tâches | Heures | % temps |
|---------|-------:|-------:|--------:|
| **P0 — Bloquants go-live** | 22 | **158h** | 63% |
| **P1 — Performance & reliability** | 13 | **74h** | 29% |
| **P2 — Nice-to-have** | 6 | **22h** | 9% |
| **TOTAL Catégorie 1** | 41 | **254h** | 100% |

### 10.2 Sur 16 semaines à 16h/sem = 256h disponibles ✅

Cap soutenable Loukmane (eval_28 §3.1) :
- 8-10h/sem marketing
- 6-8h/sem produit en collaboration avec catégories 3-9 (path critique)
- 16h/sem max sur Catégorie 1 réservé sprint commercial intensif

### 10.3 Sprint 1 (Semaines 1-6) — Setup commercial + Telegram public

**Objectif** : Sortie Telegram public, lock pricing, premières landings, interviews lancées.

| Semaine | Tâches livrées | Heures |
|--------:|---------------|-------:|
| S1 | P0-T1 (pricing lock 6h) + P0-T2 (Stripe products 5h) + P0-T7 (hosting 3h) + P0-T20 (disclaimers 3h) | 17h |
| S2 | P0-T3 (Stripe Checkout 12h) + P0-T13 (Telegram public 4h) | 16h |
| S3 | P0-T8 (Landing EN 14h) + P0-T11 (cookies 3h) | 17h |
| S4 | P0-T9 (Landing FR 8h) + P0-T10 (pricing page 6h) + P0-T22 (decoy 4h) | 18h |
| S5 | P0-T14 (track-record 10h) + P0-T17 (GA4 6h) | 16h |
| S6 | P0-T4 (quota 8h) + P0-T19 (webhook sync 6h) + tests fin sprint | 16h |

**Outcome S6** : Landing live FR+EN, pricing publique, Stripe TEST fonctionnel, Telegram public 30j cumulés, GA4 mesure trafic.

### 10.4 Sprint 2 (Semaines 7-10) — Interviews + Stripe LIVE + premiers signups

**Objectif** : 15 interviews terminées, Stripe live, GO/NO-GO niche A décidé S+10.

| Semaine | Tâches livrées | Heures |
|--------:|---------------|-------:|
| S7 | P0-T16 (interviews 25h / 4 sem) + P0-T18 (trial 14j 5h) | 11h (interviews lissé) |
| S8 | P0-T15 (onboarding 8h) + P0-T21 (lead magnet 14h /2) | 15h |
| S9 | P0-T21 finalisation + P0-T6 (refund policy 4h) + P0-T12 (legal HTML 3h) | 14h |
| S10 | P0-T5 (geo-block confirm 4h) + tests pre-launch (8h) + **DÉCISION GO/NO-GO** | 12h |

**Outcome S10** :
- ✅ Si GO : Stripe live, 1-3 premiers paid users, Telegram public 60j (track-record visible).
- ❌ Si NO-GO : continuer Phase 0 produit, geler Stripe live, persister contenu + Telegram.

### 10.5 Sprint 3 (Semaines 11-16) — Growth organique (P1)

**Objectif** : Reddit AMA, contenu SEO compounding, Discord privé, referral.

| Semaine | Tâches livrées | Heures |
|--------:|---------------|-------:|
| S11 | P1-T23 (articles SEO start 4h × 2) + P1-T29 (A/B PostHog 3h) + P1-T30 (email nurture 6h) | 17h |
| S12 | P1-T23 articles (8h) + P1-T31 (Reddit AMA prep 8h) | 16h |
| S13 | P1-T31 AMA jour J (8h) + P1-T28 (referral 8h) | 16h |
| S14 | P1-T27 (Discord privé 8h) + P1-T23 articles (8h) | 16h |
| S15 | P1-T26 (Newsletter 5h) + P1-T34 (MyFXBook 6h) + P1-T35 (cohort dash 6h) | 17h |
| S16 | P1-T24 (Pine Script 6h) + P1-T25 (YT vidéos start 6h) + P1-T32 (PH prep 4h) | 16h |

**Outcome S16** : 10-15 paid users, MRR $700-1 000, AMA fait, Discord 30 paid, Pine Script live, articles 12 cumul.

### 10.6 Sprint 4 (Mois 5-12) — Scale + Paid + Influencers

**Objectif** : PH M5, paid spend M6, influencer M6+, MRR $5-7k M12.

- M5 : P1-T32 PH launch jour J + P1-T25 YT vidéos finalisation.
- M6 : Premier paid spend $200/mo Google FR + P1-T33 influencer Trader Pro FR (si PF >1.20).
- M9 : Podcast Chat With Traders + paid scale $500/mo.
- M12 : Paid $2k/mo cap + recruter community manager PT si Discord >200 paid.

### 10.7 Chemin critique (Gantt simplifié)

```
S1 ──┬─ Pricing lock (T1)
     │
S2 ──┴─ Stripe Checkout (T2-T3)
              │
S3-S4 ────────┴─ Landings EN+FR (T7-T10)
                       │
S5 ────────────────────┴─ Track-record + GA4 (T14, T17)
                              │
S6 ─────────────────────────── Quota + Webhook (T4, T19)
                                    │
S7-S10 ─────────────────────────────┴─ Interviews + Stripe LIVE + Pre-launch tests
                                              │
                                              ├──> DÉCISION GO/NO-GO (S10)
                                              │
S11-S16 ──────────────────────────────────────┴─ Growth organique (P1)
                                                    │
M5 ─────────────────────────────────────────────────┴─ PH launch
M6 ─────────────────────────────────────────────────── Paid + influencer
M9 ─────────────────────────────────────────────────── Podcast + scale
M12 ────────────────────────────────────────────────── MRR $5-7k, 200-296 paid
```

**Path critique non-compressible** :
1. P0-T1 (pricing lock) → P0-T2 (Stripe products) → P0-T3 (Checkout) → P0-T19 (webhook) → tests Stripe live = **~30h, 4 semaines**.
2. P0-T13 (Telegram public) → 60 jours track record = **60 jours calendaires (passif)**.
3. P0-T16 interviews = **4 semaines à 6h/sem**.

Le path critique est **donc dominé par les 60 jours de Telegram public**, pas par les heures de dev. C'est pourquoi la priorité #1 est de **démarrer le Telegram public en S1**, indépendamment du reste.

### 10.8 Budget cash (hors temps founder)

| Poste | Coût/mo | Coût one-shot | Total 6 mois |
|-------|--------:|--------------:|-------------:|
| Domaine + hosting (Vercel free → Pro $20/mo M+3) | $0-20 | — | $40 |
| Stripe Tax | 0.5% des transactions | — | ~$20 (sur $4k revenue) |
| Resend email (transactional + nurture) | $0 free → $20 M+3 | — | $40 |
| Calendly Pro | $12 | — | $72 |
| PostHog Cloud | $0 free tier 1M events | — | $0 |
| Klaro/CookieYes | $0-10 | — | $30 |
| TraderMade (data licence) | $49 | — | $294 |
| Ahrefs Lite or Ubersuggest | $29 | — | $174 |
| Legal one-shot CGU/CGV avocat FR | — | $2 000 | $2 000 |
| Stripe Radar (fraud) | 0.05/txn | — | ~$10 |
| Backblaze B2 backup | $1 | — | $6 |
| GitHub Pro CI minutes | $4 | — | $24 |
| Total | ~$155/mo | $2 000 | **~$2 710** |

Plus :
- P1-T33 influencer Trader Pro FR (si PF > 1.20) : $400-600 one-shot
- M+6 paid spend pilote : $200/mo × 6 = $1 200
- PH visuals freelance (optional) : $200

**Budget cash total 12 mois** : **~$5 000** (couvert par capital initial estimé eval_28 §0).

---

## 11. Annexes

### 11.1 Sources principales

- `reports/eval_25_pmf_icp.md` — ICP, personas, niche beachhead, hero copy variants.
- `reports/eval_27_pricing.md` — Pricing v1, PSM, anchoring, annual vs monthly, INSTITUTIONAL plancher, grille v1.
- `reports/eval_28_gtm.md` — SEO wedges, contenu 90j, communauté, influencers, PH, referral, paid triggers, MRR model.
- `reports/eval_24_unit_economics.md` — Marge brute par tier, coûts marginaux, break-even, runway.
- `reports/eval_26_competitive.md` — 10 concurrents, différenciateurs défendables, copycat scenarios, plans B.
- `reports/eval_29_compliance_findings.md` — Geo-block, disclaimers, finfluencer compliance.
- `memory/sprint_w1_compliance_2026_04_29.md` — Sprint W1 compliance livré.
- `docs/value/best_product_concept.md` — Concept B "Co-Pilot" recommandé.
- `docs/value/client_relevance_review.md` — Audit pertinence champs InsightSignalV2.
- `mockups/pricing_bundles.md` — Mockup pricing bundles thématiques.
- `mockups/webapp_b2c.html`, `mockups/telegram_b2c.txt`, `mockups/b2b_insight.json`, `mockups/b2b_webhook_payload.json` — Drafts UI.
- `BUSINESS_PLAN_SMART_SENTINEL.md` — Plan d'origine v1.0 (à mettre à jour).
- `reports/decision_matrix_2026_04_30.md` — Décision matrix 0/4 strats franchissent PF lo > 1.0.

### 11.2 Glossaire métriques

- **MRR** : Monthly Recurring Revenue (somme des subscriptions actives × ARPU pondéré).
- **ARR** : Annual Recurring Revenue = MRR × 12.
- **CAC** : Customer Acquisition Cost (paid spend + temps founder valorisé) / nouveaux paid users.
- **LTV** : Lifetime Value = ARPU × 1 / churn mensuel.
- **NRR** : Net Revenue Retention = (MRR fin - MRR new) / MRR début × 100.
- **Payback period** : CAC / contribution margin mensuelle.
- **k (viral coefficient)** : (users qui invitent %) × (invités/user) × (conversion invité→paid %).
- **CVR** : Conversion Rate (étape funnel donnée).
- **CPM** : Cost per Mille impressions.
- **DR** : Domain Rating (Ahrefs).
- **KD** : Keyword Difficulty (Ahrefs).
- **PSM** : Price Sensitivity Meter (Van Westendorp).
- **PMC / IPP / OPP / PME** : Points de prix Van Westendorp (cf. eval_27 §3).

### 11.3 Conventions UTM standard

```
?utm_source=<canal>&utm_medium=<format>&utm_campaign=<id>&utm_content=<variant>

Exemples :
- ?utm_source=reddit&utm_medium=ama&utm_campaign=r-forex-jul26&utm_content=ama-thread
- ?utm_source=producthunt&utm_medium=launch&utm_campaign=ph-sep26&utm_content=hero-V3
- ?utm_source=substack&utm_medium=email&utm_campaign=newsletter-m3&utm_content=cta-track-record
- ?utm_source=google&utm_medium=cpc&utm_campaign=signaux-trading-or-fr&utm_content=ad-v1
- ?utm_source=youtube&utm_medium=sponsor&utm_campaign=trader-pro-fr-m6&utm_content=desc-link
- ?utm_source=organic&utm_medium=seo&utm_campaign=smc-ai-playbook&utm_content=cta-footer
```

### 11.4 Hero copy v1 (à locker sprint 1)

**FR (V3 recommandée)** :
> # Arrêtez de tilter. Tradez avec lucidité.
> Notre IA vous dit quand attendre, pas seulement quand entrer. Calme. Structuré. Explicable.
> [Commencer gratuitement — sans carte]  [Voir les signaux publics →]

**EN (V3 recommandée)** :
> # Stop revenge trading. Start trading with clarity.
> Our AI tells you when to wait, not just when to enter. Calm, structured, explainable.
> [Start free — no card]  [See live signals →]

**Backup variant V4 (smart money angle)** :
> # Trade like the institutions, paid like one.
> Smart money concepts (BOS, CHOCH, FVG, OB) detected automatically on XAU/USD M15 — with the rationale spelled out.

### 11.5 Top 3 messages compliance (à NE JAMAIS oublier)

1. **Footer permanent** : "Smart Sentinel AI n'est pas un conseil en investissement. Lecture algorithmique éducative. Performance passée ≠ performance future. Capital à risque."
2. **Telegram pinned** : "📋 Canal éducatif — analyses algorithmiques. Aucun signal présenté ne constitue un ordre ni un conseil financier."
3. **Pricing page micro-copy** : "Smart Sentinel AI vous aide à prendre vos propres décisions de trading avec plus de structure. Nous ne tradons pas pour vous, ne gérons pas votre argent, et ne vous donnons pas d'ordre d'achat ou de vente."

---

**Fin du plan de commercialisation. 11 sections, ~1100 lignes. Prochaine étape : exécuter P0-T1 (lock pricing) cette semaine.**
