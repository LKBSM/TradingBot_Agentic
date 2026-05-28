# Landing V1 — MIA Markets (livré 2026-05-27)

> Document de référence pour la landing commerciale de MIA Markets.
> Couvre la composition par sections, les contraintes (locks), la
> compliance, les tests et les prochaines étapes connues.

---

## 1. Cadre d'exécution

**Mission.** Livrer une landing qui **démontre** le produit (pas qui le
décrit), avec posture éducative UE 2024/2811 et architecture
progressive uniforme (hero permanent + sections collapsibles +
chatbot pilier).

**Périmètre.** Sprints L1 → L6 (~7 commits). FR uniquement V1 ; EN/DE/ES
en place dans i18n mais 302 → FR (toggle désactivé avec tooltip).

**Bundle cible.** `/[locale]` ≤ 150 kB First Load JS. Atteint **147 kB**.

---

## 2. Locks utilisateur (engagements 2026-05-27)

| # | Lock | Implémentation |
|---|------|----------------|
| 1 | « MIA Markets » sans points | `webapp/components/Footer.tsx`, `Nav.tsx`, `chatbot_responses.json`, métadonnées SEO. |
| 2 | Aucun chiffre de performance en hero. Citation imposée en Section 5. | `HeroLive.tsx` (zéro chiffre), `HonestConfidenceSection.tsx` (citation + PF 0,786 + −318pp + DSR 0). |
| 3 | FR uniquement V1, toggle EN désactivé | `LocaleToggle.tsx` (chip FR active + bouton EN disabled + tooltip "coming soon"). |
| 4 | 9 pays Phase 1 — Canada hors Québec, FR, BE, CH, LU, UK, AU, NZ, IE | `Footer.tsx` (bloc géo explicite), `FaqSection.tsx` (Q4 éligibilité). |

---

## 3. Composition de sections

`app/[locale]/page.tsx` orchestre 7 sections + nav + footer dans cet ordre :

```
HeroLive                  ← L2 — MarketReadingCard live + ChatPreview, no perf
MultiMarketSection        ← L3 — S2 — XAU + EUR + Coming Soon
ConversationReplaySection ← L3 — S3 — 3 tuiles rejouables avec typing effect
BeforeAfterSection        ← L4 — S4 — SVG comparison chaos vs lecture MIA
HonestConfidenceSection   ← L4 — S5 — PF 0,786 · −318pp · DSR 0 · citation imposée
PricingSection            ← L5 — FREE / 9€ / 19€ post-pivot + Calendly B2B discret
FaqSection                ← L5 — 6 questions Accordion
(Footer dans layout)      ← L5 — 9 pays + Early Access + DSAR/médiateur
```

**Section IDs (anchors).** `#multi-marche` · `#conversations` ·
`#avant-apres` · `#honnetete` · `#tarifs` · `#faq`.

**Rythme visuel.** `bg-muted/20` sur S3 et `bg-foreground/[0.02]` sur
S5 cassent la monotonie sans tonner. Le reste est `bg-background`
naturel.

---

## 4. Composants nouveaux (L1-L5)

| Fichier | Rôle |
|---------|------|
| `app/sitemap.ts`, `app/robots.ts` | SEO essentiel — pages indexables + crawl rules. |
| `app/opengraph-image.tsx` | OG card 1200×630 générée via next/og. |
| `components/seo/JsonLd.tsx` | SoftwareApplication + 3 Offers structured data. |
| `components/a11y/SkipLink.tsx` | Skip-link sr-only focus-visible → #main. |
| `components/compliance/CookieBanner.tsx` | CNIL 4-catégories + hook `useCookieConsent`. |
| `components/LocaleToggle.tsx` | FR chip + EN disabled tooltip. |
| `components/landing/HeroLive.tsx` | Hero vivant — MarketReadingCard + ChatPreview, no marketing H1 visible. |
| `components/landing/HeroMarketReading.tsx` | Client wrapper qui anime + câble `useChat().openFor`. |
| `components/landing/HeroChatPreview.tsx` | Thinking + intro + 3 questions + CTA chatbot. |
| `components/landing/MultiMarketSection.tsx` | S2 multi-actifs. |
| `components/landing/ComingSoonCard.tsx` | Placeholder dashed BTC/USD · Bientôt. |
| `components/landing/InsightGalleryClient.tsx` | Gallery + slot renderAfter, câble chatbot. |
| `components/landing/ConversationReplaySection.tsx` | S3 — 3 conversations rejouables. |
| `components/landing/ConversationReplayCard.tsx` | State machine + typing + IntersectionObserver auto-play. |
| `components/landing/BeforeAfterSection.tsx` | S4 — SVG comparison chaos / MIA. |
| `components/landing/HonestConfidenceSection.tsx` | S5 — vrais chiffres + citation + 3 colonnes posture. |
| `components/landing/PricingSection.tsx` (réécrit) | Pricing post-pivot 3 tiers + Calendly B2B aside. |
| `components/landing/FaqSection.tsx` | 6 questions Accordion Radix. |
| `components/Footer.tsx` (réécrit) | Brand + Produit + Légal + 9 pays + Early Access + disclaimer. |

---

## 5. Compliance & légal

- **UE 2024/2811** (finfluencers) : tous les CTA "S'abonner" en
  `aria-disabled` pour V1, posture éducative explicite, chatbot
  refuse formellement les questions de type "dois-je acheter ?".
- **MiFID II disclosure** : section Honnêteté Conformelle publie les
  chiffres défavorables (PF backtest 0,786 ; DSR 0 ; PBO 0,50).
- **RGPD** : CookieBanner CNIL 4-cat avec refus aussi visible
  qu'accepter. Hook `useCookieConsent` pour gater les trackers
  V1.x.
- **Hamon FR + protection conso UE étendue** : refund 30 j explicite
  dans Pricing + FAQ Q5.
- **DSAR + Médiateur CM2C** : liens placeholder dans le footer
  (`LEGAL-PENDING` markers à activer post-Iubenda Pro M+3).
- **Geo Phase 1** : 9 pays inclus, US + Québec exclus (revue M+9).
  Middleware geo-block (`src/api/middleware/geo_block.py`) déjà côté
  API ; webapp affiche la disponibilité explicite.

---

## 6. Performance & accessibilité

**Optimisations livrées.**

- Inter via `next/font/google` avec `preload: true`, subsets `latin` +
  `latin-ext` uniquement (no CJK).
- CSS-only animations (`hero-stagger`, `hero-thinking-dot`), pas de
  Framer Motion. `@media (prefers-reduced-motion: reduce)` désactive
  proprement.
- ImageResponse OG via `next/og` (PNG dynamique, pas d'asset binaire).
- SVG pur pour BeforeAfter (pas de chart lib — −60 kB vs Recharts).
- Lazy boundaries naturelles : `ChatPanel` côté layout, sections en
  Server Components sauf `HeroMarketReading` + `HeroChatPreview` +
  `ConversationReplayCard` + `InsightGalleryClient`.

**A11y livré.**

- `SkipLink` → `#main`.
- `aria-labelledby` sur chaque `<section>`.
- `role="img"` + `<title>` + `<desc>` sur les SVG illustratifs.
- Contrastes WCAG AA respectés (palette `--sentinel-*`).
- Toggle thème, locale et accordion focus-visible avec ring.

**Lighthouse cible.** Performance / a11y / best-practices / SEO ≥ 95.
Configuration prête dans `tests/lhci/.lighthouserc.json`.
Commande : `npm run test:lhci`.

---

## 7. Tests

| Suite | Statut |
|-------|--------|
| `npm run typecheck` | ✅ vert |
| `npm run build` | ✅ vert · `/[locale]` 147 kB |
| `npm test` (Vitest) | ✅ 40/40 |
| `npm run test:e2e` (Playwright) | ⏳ scénarios mis à jour, à exécuter avant déploiement |
| `npm run test:lhci` | ⏳ à câbler en CI une fois le domaine en prod |

**E2E mis à jour.** `tests/e2e/landing.spec.ts` couvre désormais brand
+ multi-marché + conversations + honnêteté + pricing + FAQ + footer 9
pays. Les anchors `/#demo` → `/#multi-marche` dans `sections.spec.ts`
et `chatbot.spec.ts`.

---

## 8. Prochaines étapes connues (hors V1)

1. **Brancher Stripe** + sortir les CTA d'`aria-disabled` (post-pivot
   Sprint 2). Bloquant tarification réelle.
2. **Activer EN** une fois la traduction validée par un FR/EN
   bilingue + relecture compliance.
3. **Remplir les pages légales** (CGU, Confidentialité, Mentions,
   Cookies, Médiateur) — `LEGAL-PENDING` markers présents dans
   `PricingSection`, `Footer`, `chatbot_responses.json`.
4. **Calendly URL réelle** — actuellement `calendly.com/mia-markets/demo`
   placeholder dans `PricingSection.tsx`.
5. **Vérifier la cible Lighthouse ≥ 95** en environnement réseau
   réaliste (4G simulé). La build locale satisfait les seuils mais
   le LHCI n'est pas câblé en CI.
6. **Réseaux sociaux** — placeholders à brancher quand les comptes
   sont publics.
7. **Sentinel scanner UX live** (post-pivot Sprint 4) : pour
   l'instant la lecture XAU/M15 du hero est un échantillon statique
   (`getHeroSampleSignal()`), elle deviendra dynamique quand la
   livraison `/api/insight_signal_v2` sera branchée à la webapp.

---

## 9. Décisions architecturales notables

- **Pas de Framer Motion.** Animations CSS-only suffisent et coûtent
  ~0 kB. Le `prefers-reduced-motion` est respecté.
- **Pas de chart lib.** SVG pur pour BeforeAfter, ConvictionGauge
  déjà custom. Recharts/Chart.js réservés à un éventuel dashboard
  pro V2.
- **EN désactivé plutôt que masqué.** Signal honnête au visiteur
  que c'est en route, sans servir une traduction bâclée.
- **INSTITUTIONAL retiré de la grille publique.** Un decoy à
  1 990 € abimait la perception du tier 19 € sur des prospects
  retail (cf. `pivot_positioning_2026_05_27.md`). Remplacé par un
  bloc Calendly discret.
- **Honnêteté Conformelle pleine page.** Ce n'est pas un footnote :
  c'est le moat. Publier publiquement PF 0,786 + DSR 0 est un
  engagement contractuel qui structure tout le reste du discours.

---

## 10. Références

- `docs/governance/decisions/2026-05-27_pivot_positioning_audit.md`
- `docs/governance/PROPAGATION_BRIEF_2026_05_27.md`
- `docs/architecture/MIA_MARKETS_ARCHITECTURE.md`
- `reports/audit/AUDIT_ALGO_2026_05_27.md`
- `reports/a1_verdict_2026.md` (DSR, PBO, edge_claim=false)
- `docs/frontend/MISSION_ACK_LANDING.md` (mission de départ)
- `docs/frontend/TODO_NEXT_SPRINTS.md` (TODO long-terme V2.x → V4)

---

**Auteur livré.** Claude Opus 4.7, supervisé par Loukmane Bessam.
**Date.** 2026-05-27.
