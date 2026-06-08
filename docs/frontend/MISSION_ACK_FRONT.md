# MISSION_ACK_FRONT — Compréhension de la mission frontend

**Date** : 2026-05-26
**Branche** : `institutional-overhaul`
**Terminal** : front-end (parallèle au terminal légal low-cost)

## Compréhension en 5 lignes

1. **Produit à incarner** : Smart Sentinel Co-Pilot = *indicateur de marché conversationnel*, posture éducative (jamais d'ordres), hero permanent + sections collapsibles tier-gated + chatbot pilier (cf. `decision_gate_review_v2.md` Partie 2 Angle mort #1 post-révision 2026-05-26).
2. **Architecture imposée** : Next.js 15 App Router + Tailwind + shadcn/ui (DG-023), mobile-first 375px-first, dark+light mode, Lighthouse ≥ 90 mobile. Pas de toggle 3 modes (DG-100 DROP), pas de backend/auth/Stripe/Telegram dans ce sprint.
3. **Composants centraux** : `<MarketReadingCard />` à 3 couches (verdict 5s → sections dépliables → bouton chatbot), `<ChatPanel />` slide-over desktop / fullscreen mobile avec 4-6 questions scriptées + refus pédagogique + input libre désactivé V1.
4. **Données** : type TS strict dérivé du schéma Pydantic `InsightSignalV2 v2.1.0` (cf. `client_information_explained.txt` Parties 2 + 4), 3 signaux mockés (XAU M15 bull conv 72, EURUSD H1 bear conv 58, XAU H4 neutre conv 42), réponses chatbot scriptées en JSON.
5. **Dépendance terminal légal** : tout wording compliance (disclaimer card, refus chatbot, footer CGV/Privacy, pricing « analyses » vs « signaux ») balisé `LEGAL-PENDING` pour passe d'intégration ultérieure.

## Hypothèses documentées (avant validation utilisateur)

- **Working dir** : utiliser `webapp/` existant (Next.js 14 + next-intl + Tailwind déjà initialisés), rewrite du contenu obsolète Phase 2B (dashboard, transparency, glossary, chat, pricing actuels conçus pour l'ancienne architecture). Économie ~2-3h vs `apps/web/` neuf. **À CONFIRMER**.
- **Next.js 14 → 15** : upgrade obligé (DG-023 + brief utilisateur). Codemods officiels + retests des routes existantes.
- **i18n** : `webapp/` supporte fr/en/de/es via next-intl. Conserve l'infra mais V1 expose FR uniquement (par cohérence avec wedge eval_25 + acquisition FR-first). Stubs en/de/es restent en place pour ne pas casser middleware. **À CONFIRMER**.
- **shadcn/ui** : init `new-york` style (plus sobre, cohérent inspirations Linear/Stripe/Bloomberg/Pitchbook).
- **Mockup `mockups/v3/best_concept_demo.html`** : laissé intact (déjà refait en architecture progressive uniforme 2026-05-26 selon `OUT_OF_SCOPE.md`), sert de référence visuelle pour les sections collapsibles et le chatbot — pas de réécriture, je pars de zéro côté Next.js.

## Hors-scope (rappel pour discipline)

❌ Auth · ❌ Stripe · ❌ Backend / API réelles · ❌ Telegram UI · ❌ Wording légal définitif · ❌ Plausible / analytics · ❌ Email automation
