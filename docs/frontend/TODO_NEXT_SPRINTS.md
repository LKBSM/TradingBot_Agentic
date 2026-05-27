# Frontend — Sprints suivants

**Date** : 2026-05-26 · **Mis à jour** : 2026-05-27 (V2 livré)
**Contexte** : Sprint F0→F5 (architecture progressive uniforme V1) + V2.0→V2.4 livrés. Ce document liste ce qui reste hors V1+V2, avec la justification et le déclencheur de réactivation.

## ✅ V2 livré (2026-05-27)

| Sprint | Commit  | Description courte                                          |
|--------|---------|-------------------------------------------------------------|
| V2.0   | c8407f5 | Rebrand Smart Sentinel → M.I.A. Markets (dans webapp/)      |
| V2.1   | 066eff0 | Real Claude chatbot via /api/chat SSE (Anthropic SDK)        |
| V2.2   | 4e616af | Vitest 40 tests + Playwright E2E + Lighthouse CI scaffold   |
| V2.3   | 3a598ce | PWA manifest + dynamic icons + iOS meta tags                |
| V2.4   | 2a45967 | Pédagogie EXPERT (waterfall + conformal viz)                |

Ordre ci-dessous = à exécuter par priorité approximative. Chaque entrée mentionne l'item Decision Gate correspondant (cf. `docs/governance/decision_gate_review_v2.md`).

---

## 🔴 Sprint d'intégration — Passe légale (avant toute démo externe)

Dépend du **terminal légal low-cost** qui livre en parallèle (cf. décision politique 2026-05-26).

| # | Tâche                                                        | Fichiers cibles                                                       | Source                |
|---|--------------------------------------------------------------|-----------------------------------------------------------------------|-----------------------|
| 1 | Remplacer le disclaimer du hero card                         | `components/insight/DisclaimerStub.tsx`                               | UE 2024/2811 + MiFID  |
| 2 | Réécrire le refus pédagogique chatbot                        | `mocks/chatbot_responses.json` (champs `buy-or-not.reply`)             | MiFID finfluencer     |
| 3 | Remplacer les 5 liens du footer par les pages réelles        | `components/Footer.tsx` (`LEGAL_LINKS`)                                | L.612-1 + CNIL        |
| 4 | Réécrire le bandeau compliance footer                        | `components/Footer.tsx` (bloc `data-legal-pending="footer-disclaimer"`) | UE 2024/2811          |
| 5 | Réécrire les features et CTAs des 4 tiers de pricing         | `components/landing/PricingSection.tsx` (`TIERS`)                      | Loi Hamon + DG-073    |
| 6 | Réécrire la disclosure `edge_claim=False` du hero            | `components/landing/HeroSection.tsx` (paragraphe italique)             | DG-077                |

**Méthode** : grep `LEGAL-PENDING` puis `data-legal-pending=` pour trouver tous les marqueurs. Aucune ligne ne doit subsister avec ce marqueur après la passe.

---

## 🟠 Sprint d'intégration backend — API réelle

Dépend du moteur Python qui expose `InsightSignalV2 v2.1.0` via HTTP.

| # | Tâche                                                                                   | Fichiers cibles                                              | Estimation |
|---|-----------------------------------------------------------------------------------------|--------------------------------------------------------------|------------|
| 1 | Créer `lib/api/client.ts` avec fetcher SWR/RSC                                           | nouveau                                                       | 4h         |
| 2 | Valider la réponse au runtime avec **zod** ou **valibot**                                 | nouveau `types/insight.zod.ts`                                | 4h         |
| 3 | Remplacer `SAMPLE_SIGNALS` par un appel `GET /api/v1/insights/latest?instrument=…&tf=…`    | `lib/mocks.ts` deprecated, `components/insight/InsightGallery.tsx` | 3h         |
| 4 | Brancher le chatbot sur `POST /api/v1/chat/{signal_id}` au lieu des réponses scriptées    | `components/chat/ChatPanel.tsx`, `lib/chatbot.ts`             | 6h         |
| 5 | Reactiver les rewrites `next.config.js` `/api/*` → backend Fly.io                         | `next.config.js`                                              | 1h         |
| 6 | Gérer les états error / loading / empty proprement (SWR fallback, skeleton, retry)        | tous les composants insight                                    | 8h         |

**Marqueurs à grep** : `unsafe-cast` dans `lib/mocks.ts` et `lib/chatbot.ts`, commentaires `(V2)` et `zod` dans le code.

---

## 🟡 Sprint Auth + Stripe + monetization (DG-043 + DG-044 + DG-070-084)

Bloquant pour le 1er paiement Stripe live. Dépend de la passe légale (CGU/CGV signées avocat).

| # | Tâche                                                                | Composants                                              |
|---|----------------------------------------------------------------------|---------------------------------------------------------|
| 1 | NextAuth.js (ou Auth.js) avec providers email + Google                | nouveau `app/(auth)/`                                   |
| 2 | Stripe Customer Portal embedded                                      | nouveau `app/(account)/`                                |
| 3 | Activer les CTAs pricing (actuellement `disabled`)                     | `components/landing/PricingSection.tsx`                  |
| 4 | Tier gating réel : sections premium verrouillées pour FREE/STARTER     | `components/insight/sections/*` (badge "🔒 PRO")        |
| 5 | Stripe Tax UE + checkout                                               | nouveau                                                  |
| 6 | Dual trial 14j sans CB + 14j avec CB                                   | new flow                                                 |
| 7 | Decoy INSTITUTIONAL 1990€ visible + Calendly                            | déjà visible, reste à brancher Calendly                  |

---

## 🟡 Analytics + monitoring (DG-160 + DG-161 — P0-strict-MVP)

| # | Tâche                                                                            | Notes                                                                  |
|---|----------------------------------------------------------------------------------|------------------------------------------------------------------------|
| 1 | Plausible self-hosted CNIL-compatible                                            | Pas de cookie banner si pas d'autres trackers, hosting EU              |
| 2 | Event tracking core 6 events                                                     | `signal_view`, `chatbot_question`, `section_expanded`, `upgrade_clicked`, `signup`, `paid_conversion` |
| 3 | Sentry (`@sentry/nextjs`) free tier                                              | DG-033 MODIFIED                                                        |
| 4 | Dashboard cohort retention M1/M3/M6                                              | V2 visualisation                                                      |

---

## 🟡 Mobile + PWA (DG-103)

Le mobile-first est déjà appliqué (375px breakpoint). Reste à packager comme PWA pour install écran d'accueil.

| # | Tâche                                                  |
|---|--------------------------------------------------------|
| 1 | `public/manifest.json` (déjà stub dans README v1)      |
| 2 | Service worker minimal (Workbox)                        |
| 3 | Icons (favicon, apple-touch-icon, 192/512)              |
| 4 | Notifications push (opt-in pour les events ≤ 4h)         |
| 5 | Tester sur device iOS Safari + Android Chrome           |

---

## 🟢 Sprint Améliorations chatbot (Phase 2B RAG — DG-058a)

| # | Tâche                                                                          |
|---|--------------------------------------------------------------------------------|
| 1 | Activer l'input libre du chatbot                                               |
| 2 | Brancher le RAG sur 12 papers curés (mini-fiches inline)                       |
| 3 | Streaming SSE pour les réponses longues                                        |
| 4 | Détection multi-tour : "explique-moi le jump ratio" → réponse contextualisée    |
| 5 | Disclaimer compliance dynamique selon la question                              |

---

## 🟢 Sprint Pédagogie EXPERT (DG-170 → DG-173 — V2 strategist tier)

| # | Tâche                                                                       |
|---|-----------------------------------------------------------------------------|
| 1 | Waterfall pédagogique 8 composantes avec hover explicatif                    |
| 2 | Visualisation graphique de l'intervalle conformel                            |
| 3 | Stats J.* traduites (win rate, drawdown, exposure time, skew)                |
| 4 | Tracker validité live "Lecture valable encore 2h47"                         |

---

## 🟢 Améliorations UX accessoires

- **Tooltips contextuels** sur termes techniques résiduels (DG-133) — reporté V2 (tier-gated).
- **Bannière event ≤ 4 h** avec chronomètre live au-dessus du card (DG-122) — actuellement matérialisée par badge "Imminent" dans la EventSection, peut devenir un bandeau hero global.
- **Persistance ouvert/fermé des sections** en localStorage (DG-102 user_preferences) — reporté MAU > 500.
- **Onboarding 4-step à la première connexion** (DG-121) — reporté V2 conversion.
- **Email digest format compact** (DG-104) — reporté multi-surfaces V2.

---

## 🟢 Internationalisation

| Locale  | Plan                                                                                                                |
| ------- | ------------------------------------------------------------------------------------------------------------------- |
| `en`    | Réactivation quand audience anglophone observée (signup `language=en` > 30 % du flux)                                |
| `de`    | Réactivation conditionnelle après PMF FR validée                                                                    |
| `es`    | Idem                                                                                                                |

Les fichiers `messages/{en,de,es}.json` restent dans le code pour réactivation rapide.

---

## 🟢 Tests automatisés

| Niveau         | Outil suggéré                       | Statut V1                          |
| -------------- | ----------------------------------- | ---------------------------------- |
| Unit logic     | Vitest + @testing-library/react    | ❌ Aucun test écrit V1              |
| E2E            | Playwright                          | ❌                                  |
| Visual         | Chromatic (Storybook)               | ❌                                  |
| Lighthouse CI  | `lhci autorun`                      | ❌ — cible mobile ≥ 90 à valider manuellement |
| A11y           | `@axe-core/playwright`              | ❌                                  |

À budgéter au sprint d'intégration backend (tests E2E sur les golden paths).

---

## 🟢 SEO

| # | Tâche                                                          |
|---|----------------------------------------------------------------|
| 1 | `app/sitemap.ts` + `app/robots.ts`                              |
| 2 | OpenGraph + Twitter Cards par page                              |
| 3 | Structured data (`JSON-LD` Product / FAQ pour les questions chatbot) |
| 4 | Réactivation progressive des 10 articles FR de `content/articles/fr/` sous un préfixe `/articles/[slug]` |

---

## Dernière chose

Le `webapp/content/articles/fr/*.md` (10 articles pédagogiques substantiels) est **conservé dormant** dans le repo (cf. `OUT_OF_SCOPE.md`). Quand on ouvrira un blog pédagogique, ces articles serviront de base. Ne pas les supprimer.
