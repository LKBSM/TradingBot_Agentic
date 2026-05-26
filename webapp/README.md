# Smart Sentinel AI — Webapp

Indicateur de marché conversationnel pour XAU/USD et FX. Stack : **Next.js 15
(App Router) · TypeScript strict · Tailwind CSS · shadcn/ui (`new-york`) ·
next-intl · next-themes**.

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
│  ├─ globals.css           # Variables CSS shadcn light + dark + sentinel-*
│  └─ [locale]/
│     ├─ layout.tsx         # ThemeProvider + NextIntl + Tooltip + Nav/Footer
│     └─ page.tsx           # Landing (réécrite en sprint F5)
├─ components/
│  ├─ ui/                   # shadcn primitives (Button, Card, Badge,
│  │                        # Accordion, Dialog, Sheet, Tooltip, Tabs)
│  ├─ theme-provider.tsx    # Wrapper next-themes
│  ├─ theme-toggle.tsx      # Bouton dark/light (Sun/Moon)
│  ├─ Nav.tsx               # (réécriture F5)
│  └─ Footer.tsx            # (réécriture F5)
├─ lib/
│  ├─ utils.ts              # cn() = clsx + tailwind-merge
│  └─ mocks.ts              # Loader typé des 3 signaux mockés
├─ types/
│  └─ insight.ts            # Mirror TS de InsightSignalV2 v2.1.0
├─ mocks/
│  └─ sample_signals.json   # 3 signaux : XAU M15 bull, EURUSD H1 bear,
│                           # XAU H4 neutral
├─ content/
│  └─ articles/fr/*.md      # Articles pédagogiques dormants (réutilisation V2)
├─ messages/
│  └─ {fr,en,de,es}.json    # FR actif · EN/DE/ES dormants (302 → FR)
├─ middleware.ts            # next-intl + 302 EN/DE/ES → FR équivalent
├─ i18n.ts                  # Config locales
├─ components.json          # shadcn config (style new-york, lucide)
├─ tailwind.config.ts       # Theme étendu + sentinel-bull/bear/neutral/warn
└─ tsconfig.json            # strict + noUncheckedIndexedAccess
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
`docs/frontend/TODO_NEXT_SPRINTS.md` (livré en F5).
