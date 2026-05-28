# Plan de développement focus — Produit performant AVANT commercial

**Date** : 2026-05-27
**Décision utilisateur** : focus exclusif développement technique. Le commercial (Stripe live, marketing, ventes) attendra que le produit soit solide.
**Référence** : remplace temporairement le séquencement Vague 1 commercial-mixte.

---

## 🎯 Logique du pivot focus

**Vendre un produit fragile = suicide commercial**. Si on lance avec :
- Scoring cosmétique (Pearson −0.023 sur v1)
- Bugs sécurité ouverts (F-03 HMAC replay, F-04 api_key_id duplicate, F-05 expiry non lue)
- Feed XAU à 63 % de coverage corrompu
- Chatbot qui peut halluciner sans refus pédagogique adversarialement testé
- Cost monitoring absent (un user abusif = $40/mo de perte invisible)
- Telegram async coroutines non awaitées (messages perdus silencieusement)

→ premiers clients = churn massif + plaintes + suspension Stripe + réputation détruite.

**Approche nouvelle** :
1. **Sprint Tech 1-3** : solidifier le **moteur** (scoring, data, sécurité, observability)
2. **Sprint Tech 4-6** : enrichir le **moat** (chatbot, sources, track record)
3. **Sprint Tech 7+** : passer au commercial seulement quand "performance-ready"

---

## 📊 État actuel d'après l'historique git

### ✅ Acquis (commits récents)

| Commit | Item | État |
|---|---|---|
| `c8407f5` | Rebrand M.I.A. Markets | ✅ Fait |
| `e61c3e4` | Next.js 15 + shadcn/ui bootstrap | ✅ Fait |
| `505eb72` | MarketReadingCard hero (F2) | ✅ Fait |
| `6aec3b9` | Sections collapsibles structure/régime/vol/events/history (F3) | ✅ Fait |
| `902f260` | ChatPanel + Q&A scriptés + refus pédagogique (F4) | ✅ Fait |
| `bf425a5` | Landing hero + how-it-works + pricing teaser (F5) | ✅ Fait |
| `066eff0` | Chatbot Claude SSE streaming via /api/chat (V2.1) | ✅ Fait |
| `f40d641` | InsightSignalV2 2.1.0 + LGBM-Isotonic-Conformal pipeline | ✅ Pipeline existe |
| `5895f2e` | LLM narrative generator (Claude + template fallback) | ✅ Fait |
| `3ba7fd9` | LightGBM-backed conviction + full mockup contract | ✅ Fait |

### ⚠️ À valider (peut-être fait, à vérifier)

| Item | À vérifier |
|---|---|
| DG-025 modèle entraîné prod | Le `models/calibrated_conviction_v1.pkl` existe-t-il ? Est-il chargé en prod ? Brier skill validé ? |
| DG-110 wire chatbot 8 composantes | Le contexte InsightSignalV2 complet est-il vraiment injecté dans le prompt LLM ? |
| Sécurité (DG-055/056/057/041) | Les 4 bugs sécurité ouverts (F-03/04/05 + TESTING_MODE default) sont-ils fermés ? |
| DG-022 Fly.io deploy | Commit `4dd74f5` Railway→Docker fait, mais Fly.io déployé ? Domain DNS configuré ? |

### ❌ Pas encore fait (à confirmer)

- DG-004 archive feed XAU 63 % coverage
- DG-053 verify_data_quality boot fail-fast
- DG-034 sweep state machine 432 cellules
- DG-054 Telegram retry + dedup
- DG-052 cost monitoring Anthropic + alerte spend
- DG-058a 12 papers académiques RAG curés
- DG-142 tableau performance public mensuel
- DG-160 Plausible self-hosted
- DG-161 event tracking core 6 events
- DG-033 Sentry activé
- DG-112 tests adversariaux refus pédagogique (≥30 patterns FR + 18 EN)

---

## 🏗️ Plan en 6 sprints techniques

**Total estimé** : ~10-12 semaines de dev focusé. Pas d'effort commercial pendant cette phase.

---

### Sprint Tech 1 — Cœur algorithmique (3-4 sem)

**Objectif** : que le scoring soit VRAIMENT calibré et empiriquement validé. Sans, "PF 1.30" est marketing creux.

#### Items

| Item | Effort | Acceptance |
|---|---|---|
| **Audit DG-025** : vérifier que le pipeline LGBM-Isotonic-Conformal est entraîné sur 7 ans XAU walk-forward, modèle déployé en prod | 2-4h | `models/calibrated_conviction_v1.pkl` chargé, Brier skill ≥ +5 % vs baseline empirique |
| **DG-025 finition** si nécessaire : entraînement Colab walk-forward CPCV (López de Prado 2018), export modèle, déploiement, feature flag `SCORING_VERSION=v2` | 16-24h | Pipeline LGB→Isotonic→ACI fonctionnel en prod, fallback v1 si modèle absent |
| **DG-034 sweep state machine 432 cellules** sur 7 ans XAU + EUR (Colab) | 16-24h | Defaults empiriques pour `enter × exit × confirm × cooldown × max_age × silent` |
| **DG-004 archive feed XAU 63 %** → `data/_archived/` + README explicatif | 1h | Feed corrompu plus jamais chargé en prod |
| **DG-053 verify_data_quality** au boot fail-fast | 4-6h | Scanner refuse de booter si feed < 95 % coverage |
| **Tests DG-025 + DG-034** : couverture ≥ 85 % sur src/intelligence/scoring/ | 8-12h | Tests unitaires + intégration + Brier skill validation |

**Gate de sortie** : scoring v2 chargé, Brier skill validé, sweep state machine livré, data quality boot fail-fast actif.

---

### Sprint Tech 2 — Sécurité critique (1-2 sem)

**Objectif** : fermer les 4 bugs sécurité ouverts. **Bloquant absolu** avant tout lancement.

#### Items

| Item | Effort | Acceptance |
|---|---|---|
| **DG-041 TESTING_MODE=0 défaut prod + gate CI** | 1-2h | Test CI fail si `SENTINEL_TESTING_MODE=1` en prod env |
| **DG-055 HMAC admin replay nonce-based** (F-03 OUVERT) | 4-6h | Tests adversariaux replay/cross-route passent |
| **DG-056 UNIQUE constraint api_key_id** + migration SQLite (F-04 OUVERT) | 2-4h | ALTER TABLE appliqué, tests doublon échouent |
| **DG-057 lecture subscription_expires** dans `require_api_key` (F-05 OUVERT) | 2-4h | Test : user expiré reverts FREE, accès premium refusé |
| **Audit sécurité complet** : grep des secrets en clair, vérification CORS, rate limiter, sanitize input | 4h | Report `docs/audits/security_M0_2026.md` produit |

**Gate de sortie** : 4 vulnérabilités fermées, audit sécurité passé, gate CI bloque mode test en prod.

---

### Sprint Tech 3 — Chatbot pilier solide (2-3 sem)

**Objectif** : le chatbot est le moat. Il doit être robuste, contextualisé, anti-fail.

#### Items

| Item | Effort | Acceptance |
|---|---|---|
| **Audit DG-110** : le chatbot injecte-t-il bien le contexte InsightSignalV2 complet (8 composantes + uncertainty + structure + regime + vol + event + history) ? | 2-4h | Vérification via prompt logs |
| **DG-110 finition** si gaps : intégrer `context_builder.py` qui sérialise tous les champs critiques | 8-16h | Question "Pourquoi 72 ?" → décomposition 8 composantes avec contributions chiffrées |
| **DG-111 chatbot conformal + stats J.*** : explication intervalle [54-82] + IC bootstrap 1.30 [1.12-1.49] | 8-12h | Question "marge d'erreur ?" → réponse traduit ACI Gibbs-Candès en langage humain |
| **DG-112 tests adversariaux refus pédagogique** : ≥ 30 patterns FR + 18 patterns EN, recall ≥ 98 %, faux positifs < 5 % | 6-10h | `tests/test_chatbot_adversarial.py` passe avec ces seuils |
| **DG-114-REDUCED 3 questions suggérées contextuelles** dynamiques | 6-8h | Q1 dépend conviction, Q2 dépend top component, Q3 dépend event ≤4h OU conviction ≥70 OU fallback |
| **DG-042 NARRATIVE_MODE=llm tier-routed** : FREE=template, STARTER=Haiku, PRO=Sonnet, INSTITUTIONAL=Opus | 4h | Logs montrent le bon modèle par tier |
| **Validation forbidden tokens post-processing** : `contains_forbidden_token` bloque tout slip "achetez/buy/garanti" | 4h | Test injection prompt malicieux → fallback safe |
| **Session memory 5-turn** par session | 4-6h | Chatbot répond cohéremment à 3 questions consécutives |

**Gate de sortie** : chatbot répond aux 6 questions types incl. refus pédagogique scripté, 30+ tests adversariaux passent, aucun vocabulaire interdit ne fuite.

---

### Sprint Tech 4 — Delivery + observability (1-2 sem)

**Objectif** : la production ne plante pas silencieusement, les coûts sont sous contrôle, les bugs sont visibles.

#### Items

| Item | Effort | Acceptance |
|---|---|---|
| **DG-054 Telegram retry + dedup** : fix python-telegram-bot v20+ async + retry exp backoff 429 + dedup `(chat_id, signal_id)` + respect Retry-After | 8h | Test ≥ 30 abonnés simultanés sans flood ban, zéro message perdu |
| **DG-052 cost monitoring Anthropic** : Prometheus gauge `llm_cost_usd_total` + alerte Discord/email à $X/jour | 4-6h | Alerte se déclenche à seuil en test |
| **DG-033 Sentry activé** : DSN configuré, exceptions remontent, source maps front | 2-4h | Test exception → apparaît dans dashboard Sentry |
| **Circuit breaker thresholds validation** : analyse 30j logs false-positive, ajustement si nécessaire | 2-4h | Report `docs/audits/circuit_breaker_thresholds.md` |

**Gate de sortie** : Telegram robuste, cost monitoring actif, Sentry capture exceptions, circuit breakers calibrés.

---

### Sprint Tech 5 — Sources + Track Record (2 sem)

**Objectif** : substancier le PF affiché, ajouter sources académiques cliquables, transparence vérifiable.

#### Items

| Item | Effort | Acceptance |
|---|---|---|
| **DG-058a sources RAG curées 12 papers académiques + mini-fiches inline** | 16-20h | López de Prado, Corsi, Gibbs & Candès, Barndorff-Nielsen, Adams & MacKay, Angelopoulos & Bates, Avellaneda, Cont, Engle, Hasbrouck, Lo, Pedersen → 12 papers cliquables dans mode EXPERT |
| **DG-142 tableau performance public** `/track-record` : stats agrégées + tableau trades + equity chart + filtre période + cron J+1 23:59 | 14-20h | Page accessible sans auth, stats PF avec IC, equity curve SVG, disclaimer paper-trading |
| **Nightly cron stats agrégées** : agrégation `signals.db` sur 1m/3m/6m/all + snapshot R2 | 4-6h | Snapshot quotidien fonctionnel |
| **Bootstrap CI calculation validation** : IC 95 % sur PF, 1000 itérations resample | 2-4h | IC reproductible, test unitaire |

**Gate de sortie** : sources RAG 12 papers visibles, track-record public mis à jour quotidiennement, IC bootstrap validé.

---

### Sprint Tech 6 — Infra deploy + analytique (1-2 sem)

**Objectif** : prod déployée, métriques observables, finition produit.

#### Items

| Item | Effort | Acceptance |
|---|---|---|
| **DG-022 Fly.io cdg deploy backend** finition + DNS `api.mia.markets` | 4-6h | Backend accessible, health check OK, monitoring Fly metrics |
| **DG-029-MODIFIED Fly.io secrets natifs** : tous les secrets (Anthropic, Telegram, Stripe test) via `fly secrets set` | 1-2h | Aucun secret en clair dans `fly.toml`, `.env.production` ou git |
| **DG-103 mobile-first responsive audit** : test viewports 375/393/768/1024/1440, Lighthouse ≥ 90 mobile | 8-12h | Page lecture mobile fonctionne, FAB chatbot ≥1024px sidebar |
| **DG-160 Plausible self-hosted** sur Fly.io | 6h | `analytics.mia.markets` accessible, no third-party cookies, registration invite-only |
| **DG-161 event tracking core 6 events** : signal_view, chatbot_question, section_expanded, upgrade_clicked, signup, paid_conversion | 10-14h | Events arrivent dans Plausible dashboard sous 30s |

**Gate de sortie** : prod déployée Fly.io + Vercel, Plausible reçoit events, mobile-first opérationnel.

---

## 📋 Récap effort total

| Sprint | Items | Effort | Cumul |
|---|---|---|---|
| **1 — Cœur algorithmique** | DG-025 audit/finition + DG-034 + DG-004 + DG-053 | 47-69h | ~3-4 sem |
| **2 — Sécurité critique** | DG-041 + DG-055 + DG-056 + DG-057 + audit | 13-20h | ~1-2 sem |
| **3 — Chatbot pilier** | DG-110 + DG-111 + DG-112 + DG-114 + DG-042 + session memory | 42-64h | ~2-3 sem |
| **4 — Delivery + obs** | DG-054 + DG-052 + DG-033 + CB calibration | 16-22h | ~1-2 sem |
| **5 — Sources + Track Record** | DG-058a + DG-142 + nightly + bootstrap CI | 36-50h | ~2 sem |
| **6 — Infra + analytique** | DG-022 + DG-029 + DG-103 + DG-160 + DG-161 | 29-40h | ~1-2 sem |
| **TOTAL** | | **~180-265h** | **~10-12 semaines** |

---

## 🎯 Critère "performance-ready" (gate avant pivot commercial)

Quand ces 10 critères sont validés simultanément, le produit est prêt pour le commercial :

- [ ] Scoring v2 chargé en prod avec Brier skill ≥ +5 % vs baseline
- [ ] Sweep state machine 432 cellules livré, defaults empiriques
- [ ] Feed XAU 100 % coverage, verify_data_quality fail-fast actif
- [ ] 4 vulnérabilités sécurité fermées (F-03/04/05 + TESTING_MODE gate CI)
- [ ] Chatbot répond aux 6 questions types avec contexte InsightSignalV2 injecté
- [ ] 30+ tests adversariaux refus pédagogique passent
- [ ] Telegram retry + dedup activé, 0 message perdu sur 30 abonnés
- [ ] Cost monitoring Anthropic actif avec alerte spend
- [ ] Track-record public `/track-record` opérationnel avec PF + IC bootstrap quotidien
- [ ] 12 sources RAG cliquables dans mode EXPERT
- [ ] Plausible self-hosted + 6 events core opérationnels
- [ ] Mobile-first Lighthouse ≥ 90

**À ce stade**, on peut activer le commercial (Stripe live, pricing page, marketing, RC Pro souscrite, Iubenda déployé) sans risque de "ventre mou".

---

## 🚀 Brief pour l'autre instance Claude Code

**Mission** : exécuter les 6 sprints techniques ci-dessus dans l'ordre. Priorité ABSOLUE Sprint 1-3 (cœur algo + sécurité + chatbot). Ignorer pour l'instant tout item commercial (Stripe live, pricing UI, RC Pro, Iubenda, marketing copies).

**Prérequis lecture** :
1. `docs/governance/dev_focus_plan_2026_05_27.md` (ce document)
2. `docs/governance/vague1_execution/briefs/` (briefs détaillés des items DG-110, DG-112, DG-114, DG-101, DG-103, DG-160, DG-161 déjà rédigés)
3. `docs/governance/decision_gate_review_v2.md` Partie 4 (séquencement détaillé items DG-001 à DG-085)

**Ordre d'exécution recommandé** :

```
Sprint 1 (3-4 sem)
  → Audit DG-025 actuel : `models/calibrated_conviction_v1.pkl` chargé ?
  → Si manque : entraînement Colab walk-forward XAU 7 ans, export modèle
  → DG-004 archive feed corrompu
  → DG-053 verify_data_quality boot fail-fast
  → DG-034 sweep state machine 432 cellules (Colab)
  → Tests couverture ≥85 % src/intelligence/scoring/

Sprint 2 (1-2 sem) — peut être parallélisé avec Sprint 1
  → DG-041 + DG-055 + DG-056 + DG-057
  → Audit sécurité global

Sprint 3 (2-3 sem) — après Sprint 1 (besoin scoring v2 stable)
  → Audit DG-110 actuel : contexte InsightSignalV2 bien injecté ?
  → DG-111 + DG-112 + DG-114-REDUCED + DG-042
  → Session memory 5-turn

Sprint 4 (1-2 sem) — peut être parallélisé avec Sprint 3
  → DG-054 + DG-052 + DG-033

Sprint 5 (2 sem) — après Sprint 1 stable
  → DG-058a (12 papers RAG)
  → DG-142 (track-record public)
  → Nightly cron + bootstrap CI

Sprint 6 (1-2 sem) — fin de phase tech
  → DG-022 + DG-029 + DG-103 + DG-160 + DG-161
```

**Mise à jour PROGRESS** : créer `docs/governance/vague1_execution/PROGRESS.md` à mettre à jour à chaque fin de sprint avec :
- Items livrés
- Items reportés (avec raison)
- Métrique gate de sortie validée oui/non
- Décisions ad-hoc prises

**Quand "performance-ready"** : l'autre instance ouvre une PR "FEAT: performance-ready gate validated" qui coche les 10 critères. **Là** je reprends pour pivoter sur le commercial (Stripe live, marketing, Iubenda, RC Pro).

---

## 🚫 Ce qu'on NE FAIT PAS dans cette phase

Items reportés **post performance-ready** :

- ❌ Stripe live activation (DG-043)
- ❌ Stripe Tax UE + reverse charge (DG-044)
- ❌ Page pricing complète (DG-132)
- ❌ Page marketing / méthodologie / about
- ❌ Iubenda Pro souscription
- ❌ RC Pro Hiscox/Intact souscription
- ❌ Email automation cycle de vie (DG-131)
- ❌ Onboarding 4-step (DG-121)
- ❌ Souscription Trading Economics (DG-027)
- ❌ Avocat fintech canadien (Phase 2 post-revenue)
- ❌ Geo-block prod activation (DG-045) — peut rester en mode test pendant dev
- ❌ Track-record Telegram public ouverture (DG-072) — attendre 30+ trades clôturés

Tous ces items sont **prêts à exécuter** (briefs/copies/configs déjà rédigés) mais on les déclenche **après** que les 10 critères performance-ready soient validés.

---

## ⏱️ Timeline réaliste

- **M0-M3** (sprints tech 1-3) : cœur algorithmique + sécurité + chatbot pilier
- **M3-M4** (sprints tech 4-6) : delivery + sources + infra
- **M4 fin** : performance-ready gate validée
- **M4-M5** : pivot commercial (Stripe live, pricing, Iubenda, RC Pro, marketing)
- **M5+** : premier paiement Stripe live encaissé

**Soit ~5 mois total avant premier revenu**, mais avec un produit qui ne plantera pas.

C'est ~1 mois de plus que le séquencement Vague 1 mixte initial, mais avec une qualité produit **incomparablement supérieure**. Mieux vaut vendre à M5 un produit solide qu'à M2 un produit fragile qui détruit la réputation.

---

## 📊 Pourquoi ce pivot est juste

| Argument | Validation |
|---|---|
| **Premier business** | Tu n'as pas le luxe de pivoter en cas d'échec produit. Performance-first = derisking. |
| **Réputation fintech** | Un seul incident silencieux (Telegram messages perdus, hallucination chatbot, scoring incohérent) détruit la marque dans une niche aussi observée. |
| **Bootstrap = pas d'avocat M0-M3** | Sans avocat, ton seul bouclier = qualité produit + posture éducative cohérente. La qualité produit est non négociable. |
| **Le moat est le chatbot + scoring calibré** | Sans scoring v2 validé empiriquement + chatbot anti-hallucination, le moat est faux et la valeur perçue creuse. |
| **Vague 1 commercial était trop optimiste** | 6 semaines pour MVP commercialisable = compressé sur un produit qui exige ~10-12 sem de tech. Mieux vaut être honnête. |

---

## ✅ Validation utilisateur attendue

Acquittement de ce plan = **GO autre instance Claude Code sur Sprint 1**.

Si tu valides, je :
1. Crée `docs/governance/vague1_execution/PROGRESS.md` (vide, à remplir au fil des sprints)
2. Mets à jour le BATTLE_PLAN.md pour refléter ce nouveau séquencement
3. Mets à jour MEMORY.md avec la décision pivot focus dev
4. Brief final pour l'autre instance Claude Code

**Réponds "GO" ou "ajuste X"** et j'enchaîne.
