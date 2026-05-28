# MISSION_ACK_LANDING — Compréhension du brief landing commerciale

**Date** : 2026-05-27
**Branche** : `institutional-overhaul`

## Compréhension en 5 lignes

1. **Produit à incarner** : M.I.A. Markets *Multi-asset Intelligence Assistant for Markets*, le chatbot s'appelle **Sentinel** (persona interne). Posture éducative non-négociable, vocabulaire « analyses » et jamais « signaux » (lock politique D5).
2. **Différence vs F0-V2** : la landing actuelle (`HeroSection` / `DemoSection` / `HowItWorks` / `Pricing` / `Footer`) **décrit** le produit. La nouvelle landing doit **le faire vivre** — hero = vraie lecture XAU live + chatbot déjà ouvert, scroll = démonstration progressive (multi-marché → conversations scriptées → comparaison Avant/Après → honest confidence pleine page → pricing 3 tiers seulement (1990€ enterré) → FAQ → footer).
3. **Hero différenciateur unique** (cf. `client_relevance_review.md`) : Profit Factor 1.30 [1.12, 1.49] sur 329 setups walk-forward 7 ans + calendrier événementiel + intervalle conformel. Aucune promesse de gain, `edge_claim=False` assumé comme argument anti-finfluenceur.
4. **Exigences niveau prod dès J1** : Lighthouse mobile ≥ 95 / LCP < 1.5s / poids < 500 KB hors fonts+images / headers sécu (CSP strict, HSTS, X-Frame, Referrer-Policy, Permissions-Policy) / cookie banner CNIL 4 catégories (placeholder LEGAL-PENDING) / WCAG 2.1 AA / SEO complet (OG image 1200×630, JSON-LD SoftwareApplication+FAQPage, sitemap, robots, hreflang FR/EN même si EN désactivé).
5. **Hors-scope V1 (déjà documenté)** : pas d'auth réelle, pas de Stripe checkout fonctionnel, pas de backend, pas de trackers, pas de fausses reviews. CTA pricing ouvrent modal « bientôt disponible ». Tout wording légal placeholder marqué `LEGAL-PENDING` pour la passe légale stratégie bootstrap (`legal_bootstrap_strategy_2026_05_26.md`).

## Locks utilisateur 2026-05-27 (validés)

1. **Brand** : « **MIA Markets** » sans points (orthographe officielle révisée 2026-05-27 — propagation depuis « M.I.A. Markets » V2.0).
2. **Hero sans chiffres de performance** : aucun PF / IC / hit-rate ne sort de la Section 5. Le hero démontre le produit en action, point. Les vrais chiffres (PF 0,786, sous-perf −318pp vs Buy&Hold, verdict A1 statistiquement définitif, `edge_claim=False`) vivent dans Section 5 « Honest Confidence » avec citation imposée :
   > *« Aucun indicateur de marché ne devrait promettre des gains. Nous n'en faisons pas. Ce que nous offrons, c'est une compréhension augmentée du marché — pas une performance financière. »*
   L'honnêteté est le moat, pas la faiblesse.
3. **i18n V1 = FR seul**. Infra next-intl gardée, 302 EN/DE/ES → FR maintenu. Toggle EN visible dans le header mais **désactivé/grisé** avec tooltip « English version — coming soon » pour signaler l'ambition internationale future.
4. **Geo-restrict 9 pays** : CA hors Québec · FR · BE · CH · LU · UK · AU · NZ · IE. Fusionner « Canada hors Québec » (pas deux items « Canada » + « Québec exclu » distincts — incohérence compliance). US toujours exclus jusqu'à M9-M12.

## Hypothèses documentées (avant validation)

- **Branding** : le brief écrit « MIA Markets » sans points, la mémoire et le codebase utilisent « M.I.A. Markets ». Je garde **M.I.A. Markets** par cohérence avec le rebrand acté 2026-05-26 + manifest PWA + system prompt LLM déjà déployés. **À CONFIRMER** si tu préfères la version sans points.
- **Working dir** : continuation dans `webapp/` (Next.js 15 + shadcn/ui + ChatProvider + Anthropic SDK déjà en place après V2). La landing actuelle (`app/[locale]/page.tsx` composé de `HeroSection/DemoSection/HowItWorksSection/PricingSection`) est **reconstruite**, pas étendue — la `HowItWorksSection` notamment disparaît (remplacée par les démonstrations de Sections 3-5).
- **Conservation V2** : le `MarketReadingCard` (V1 F2-F3) + `ChatPanel` (V2.1 Claude live + scripted fallback) + `ExpertSection` (V2.4) restent les briques de base. La landing les compose différemment.
- **Posture compliance Early Access** : appliquée partout (badge discret « Early Access · Educational Use » + bandeau bootstrap). Geo-restrict FR/BE/CH/LU mentionné dans le footer, plus de référence US/QC/UK puisque la stratégie bootstrap geo-block ces juridictions.
- **Lighthouse target** : ≥ 95 mobile mesuré localement (`npx lighthouse … --form-factor=mobile --preset=mobile`). LCP cible < 1.5s sur Slow 4G simulé via Chrome DevTools throttling.

## Lecture des 6 documents — confirmée

- ✅ `docs/value/client_information_explained.txt` — schéma `InsightSignalV2 v2.1.0` + 12 blocs + cascade LLM (déjà lu V1)
- ✅ `docs/value/client_relevance_review.md` — hiérarchie pépites : **H0 (PF+IC) hero absolu** · 6 soutiens (direction, conviction calibrée, blackout news, régime traduit, vol forecast vs naïve, badge 8 facteurs) · 5 secondaires (invalidation, retest armé, BOS/FVG/OB pour SMC, session, sources RAG)
- ✅ `docs/governance/decision_gate_review_v2.md` — architecture progressive uniforme, hero permanent + sections collapsibles tier-gated (déjà lu V1)
- ✅ `docs/governance/decisions/2026-05-26_5_political_locks.md` — D1 Vision B confirmée · D2 DEFER M3 (bootstrap legal) · D3 pricing $0/$29/$79/$1990 · D4 instruments **XAU + EUR seuls** en GA · D5 vocabulaire « analyses » + USP « honest confidence »
- ✅ `docs/governance/OUT_OF_SCOPE.md` — toggle 3 modes DROP, architecture progressive uniforme adoptée (déjà lu V1)
- ✅ `mockups/v3/best_concept_demo.html` — référence créative, ne PAS recopier le toggle obsolète (déjà survolé V1)
