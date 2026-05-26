# M.I.A. Markets — Webapp

**Multi-asset Intelligence Assistant for Markets** · indicateur de marché
conversationnel pour XAU/USD et FX. Le chatbot porte le nom **Sentinel**
(personnage assistant interne, c'est lui qui s'adresse à l'utilisateur).

Stack : **Next.js 15 (App Router) · TypeScript strict · Tailwind CSS ·
shadcn/ui (`new-york`) · next-intl · next-themes**.

Architecture **progressive uniforme** (cf.
`docs/governance/decision_gate_review_v2.md` Partie 2 Angle mort #1, révision
2026-05-26) : un layout unique, hero permanent + sections collapsibles
tier-gated, chatbot pilier accessible partout. Pas de toggle 3 modes.

## Quick start

```bash
cd webapp
npm install
npm run dev                  # http://localhost:3000
```

Le backend n'est pas requis en V1 — toute la donnée vient de
`mocks/sample_signals.json` typée via `types/insight.ts`.

## Scripts

| Script              | Action                                                  |
| ------------------- | ------------------------------------------------------- |
| `npm run dev`       | Serveur de développement sur le port 3000               |
| `npm run build`     | Build production (Next.js + type-check via tsc)         |
| `npm run start`     | Serve un build production                               |
| `npm run typecheck` | `tsc --noEmit` sur l'ensemble du projet                 |
| `npm run lint`      | Lint via `eslint-config-next`                           |
| `npm run format`    | Formattage Prettier (avec tri Tailwind)                 |

## Structure

```
webapp/
├─ app/
│  ├─ globals.css                # Variables CSS shadcn light + dark + sentinel-*
│  └─ [locale]/
│     ├─ layout.tsx              # Theme + NextIntl + Tooltip + ChatProvider + Nav/Footer
│     └─ page.tsx                # Landing : Hero + Demo + HowItWorks + Pricing
├─ components/
│  ├─ ui/                        # shadcn primitives (Button, Card, Badge,
│  │                             # Accordion, Dialog, Sheet, Tooltip, Tabs)
│  ├─ theme-provider.tsx         # Wrapper next-themes
│  ├─ theme-toggle.tsx           # Bouton dark/light (Sun/Moon)
│  ├─ Nav.tsx                    # Sticky brand + anchors + theme toggle
│  ├─ Footer.tsx                 # Compliance bandeau + 5 liens LEGAL-PENDING
│  ├─ insight/                   # Cœur produit
│  │  ├─ MarketReadingCard.tsx   # Orchestrateur couches 1 + 2
│  │  ├─ InsightGallery.tsx      # Bridge useChat() (client)
│  │  ├─ InsightSections.tsx     # Accordion des 5 sections
│  │  ├─ VerdictHeader.tsx       # Couche 1
│  │  ├─ ConvictionGauge.tsx     # Couche 1
│  │  ├─ TemporalBadge.tsx       # Couche 1
│  │  ├─ DisclaimerStub.tsx      # Couche 1 (LEGAL-PENDING)
│  │  └─ sections/
│  │     ├─ StructureSection.tsx
│  │     ├─ RegimeSection.tsx
│  │     ├─ VolatilitySection.tsx
│  │     ├─ EventSection.tsx
│  │     └─ HistorySection.tsx
│  ├─ chat/                      # Pilier conversationnel (moat #1)
│  │  ├─ ChatProvider.tsx        # Context isOpen / activeSignal / turns
│  │  ├─ ChatPanel.tsx           # Sheet slide-over responsive
│  │  ├─ ChatMessage.tsx         # Bulles user / assistant
│  │  ├─ SuggestedQuestions.tsx  # Chips
│  │  └─ ChatInputStub.tsx       # Input désactivé V1 (tooltip "bientôt")
│  └─ landing/
│     ├─ HeroSection.tsx         # Positioning + track-record honnête
│     ├─ DemoSection.tsx         # 3 cards via InsightGallery
│     ├─ HowItWorksSection.tsx   # 3 étapes verdict / sections / chatbot
│     └─ PricingSection.tsx      # 4 tiers placeholder (LEGAL-PENDING)
├─ lib/
│  ├─ utils.ts                   # cn() = clsx + tailwind-merge
│  ├─ mocks.ts                   # Loader typé sample_signals.json
│  ├─ chatbot.ts                 # Loader scripted responses + fallback
│  └─ insight-formatters.ts      # Toutes les chaînes user-visible (FR)
├─ types/
│  ├─ insight.ts                 # Mirror TS de InsightSignalV2 v2.1.0
│  └─ chatbot.ts                 # ChatbotQuestion / ChatbotScript
├─ mocks/
│  ├─ sample_signals.json        # 3 signaux mockés
│  └─ chatbot_responses.json     # 15 paires Q/A scriptées
├─ content/
│  └─ articles/fr/*.md           # Articles pédagogiques dormants (V2)
├─ messages/
│  └─ {fr,en,de,es}.json         # FR actif · EN/DE/ES dormants (302 → FR)
├─ middleware.ts                 # next-intl + 302 EN/DE/ES → FR équivalent
├─ i18n.ts                       # Config locales (pattern next-intl 3.22+)
├─ components.json               # shadcn config (style new-york, lucide)
├─ tailwind.config.ts            # Theme étendu + sentinel-bull/bear/neutral/warn
└─ tsconfig.json                 # strict + noUncheckedIndexedAccess
```

## Internationalisation

- FR seul exposé en V1 (locale par défaut, sans préfixe URL).
- EN / DE / ES inactifs : les fichiers de traduction restent dans le code
  pour préserver l'infrastructure next-intl. Toute requête `/en/*`, `/de/*`
  ou `/es/*` est 302-redirigée vers l'équivalent FR via `middleware.ts`.
- Aucun stub de locale vide n'est indexable (pas de risque SEO duplication).

## Conformité (placeholder — `LEGAL-PENDING`)

Tout le wording compliance affiché dans la card, le chatbot, le pricing et
le footer est en placeholder marqué `LEGAL-PENDING`. Le passage d'intégration
remplacera ces blocs par les textes livrés par le terminal légal (CGU/CGV,
disclaimer MiFID, refus pédagogique).

## Données démo

`mocks/sample_signals.json` contient trois signaux mockés calibrés sur le
schéma `InsightSignalV2 v2.1.0` :

| Index | Instrument | TF  | Direction      | Conv | Particularité                          |
| ----- | ---------- | --- | -------------- | ---- | -------------------------------------- |
| 0     | XAUUSD     | M15 | BULLISH_SETUP  | 72   | Structure complète, FOMC dans 18h      |
| 1     | EURUSD     | H1  | BEARISH_SETUP  | 58   | Régime ranging, jump ratio 0.42, BCE 3h |
| 2     | XAUUSD     | H4  | NEUTRAL        | 42   | Consolidation, en attente retest FVG    |

Pour itérer sur la donnée pendant le développement : éditer ce JSON, le type
est appliqué par cast assertif dans `lib/mocks.ts` (TODO : validation zod au
sprint d'intégration backend).

## Hors-scope V1

Auth · Stripe · backend / API réelles · Telegram UI · wording légal
définitif · Plausible / analytics · email automation. Voir
`docs/frontend/TODO_NEXT_SPRINTS.md` pour la roadmap complète des
sprints suivants (passe légale, intégration backend, auth + Stripe,
analytics, PWA mobile, RAG chatbot, pédagogie EXPERT, SEO, tests).

## Composant inventory

`docs/frontend/component_inventory.md` liste tous les composants livrés
en F0→F5 avec leur statut (READY / PARTIAL / LEGAL-PENDING) et les
marqueurs `LEGAL-PENDING` à grep avant chaque release tag.

## Validation manuelle (avant tout merge prod)

```bash
cd webapp
npm run typecheck           # tsc --noEmit
npm run lint                # ESLint + next/typescript rules
npm run build               # Next.js build production
npm run dev                 # Smoke test runtime
# Puis dans un second terminal :
npx lighthouse http://localhost:3000 --view --preset=desktop
npx lighthouse http://localhost:3000 --view --form-factor=mobile
# Cible mobile : performance ≥ 90, accessibilité ≥ 90, best-practices ≥ 90
```
