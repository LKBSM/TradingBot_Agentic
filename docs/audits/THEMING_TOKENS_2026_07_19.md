# Système de thèmes — tokens UI + sélecteur (4 thèmes)

Branche : `feat/design-tokens-theming` (worktree dédié, depuis main `0951d34`).
Ligne : **présentationnel only**. Aucun changement moteur / logique IA / textes de la
ligne. La couleur reste réservée au SENS (états zones/liquidité via `sentinel-*`) — aucun
vert/rouge décoratif « achat/vente » introduit par un thème.

## Phase A — Socle (LIVRÉ, en attente de validation live)

### Mécanisme retenu (final)
- `next-themes` en **4 thèmes nommés** : `terminal` (défaut) · `atelier` · `schema` · `ardoise`.
- `attribute="data-theme"` → **`data-theme="<id>"` sur `<html>`** (un attribut propre, source
  unique ; pas de pollution de classe, pas de flash pour les thèmes non-défaut). Les valeurs de
  tokens vivent sous `[data-theme='…']` dans `globals.css`.
  *(Étape intermédiaire écartée : `attribute="class"` + `value` map — le multi-token
  `'dark theme-x'` casse `DOMTokenList.remove` de next-themes, et une seule classe laissait des
  scories de classe brute résiduelles.)*
- Les **3 thèmes sombres** pilotent le variant Tailwind `dark:` via
  `darkMode: ['variant', ['[data-theme=terminal] &', '[data-theme=schema] &', '[data-theme=ardoise] &']]`.
  Atelier (clair) en est volontairement absent → tous les `dark:` existants fonctionnent sans être touchés.
- **Sans flash SSR** : script bloquant next-themes (pose `data-theme` avant paint) +
  `suppressHydrationWarning` sur `<html>` + `:root` = valeurs Terminal (fallback cohérent).
- **Persistance** : `localStorage['theme']` (défaut next-themes). Défaut = `terminal`. **Vérifié**
  (bascule + reload conservent le thème, 0 pageerror).
- `color-scheme` déclaré par thème (dark/light) → widgets natifs + scrollbars cohérents.

### Tokens sémantiques (par RÔLE, format HSL — `tailwind.config.ts` inchangé côté mapping)
Surfaces : `--background` (bg) · `--card`/`--popover` (panel) · `--secondary`/`--muted`/`--accent`
(surfaces élevées / hover). Texte : `--foreground` (texte) · `--muted-foreground` (atténué).
Accent / focus : `--primary` + `--ring` (accent de marque du thème) · `--primary-foreground`.
Bordure : `--border` / `--input`. Rayon : `--radius`. Danger UI : `--destructive`.
**États (SENS, réservés)** : `--sentinel-bull` · `--sentinel-bear` · `--sentinel-neutral` ·
`--sentinel-warn` · **`--sentinel-liq` (nouveau)**. Narration : **`--font-narrative`** (Atelier → serif).

Ajouts `tailwind.config.ts` : `sentinel.liq` + famille `font-narrative`.

### Palettes (hex de référence → mappés sur les tokens)
| Rôle | Terminal | Atelier | Schéma | Ardoise |
|---|---|---|---|---|
| bg | #0a0f1c | #f6f4ee | #14161a | #1a1613 |
| panel | #111a2c | #efece3 | #1b1e24 | #241f1a |
| texte | #e7ecf6 | #26282c | #dfe3e8 | #efe9e0 |
| atténué | #8b94ab | #6a6b66 | #8b9099 | #a99f92 |
| accent | #4d9de0 | #1d6f6a | #5fb3c4 | #d8b878 |
| bull | #37b98c | #2f8f6b | #c3c8cf | #6bbf9a |
| bear | #dd6b7a | #b4564f | #7c828b | #d98b7a |
| liquidité | #d6a24a | #b07d1c | #5fb3c4 | #d8b878 |
| radius | 0.75rem | 0.75rem | 0.5rem | **1rem** |
| narration | sans | **serif** | sans | sans |

### Fichiers touchés (Phase A)
- `webapp/app/globals.css` — 4 blocs de tokens (remplace `:root`/`.dark`).
- `webapp/app/[locale]/layout.tsx` — config `ThemeProvider` 4 thèmes.
- `webapp/tailwind.config.ts` — `darkMode` variant + `sentinel.liq` + `font-narrative`.
- `webapp/components/theme-toggle.tsx` — raccourci clair/sombre (Atelier ↔ Terminal),
  en attendant le sélecteur 4-thèmes (Phase C).

### Vérifications
- `tsc --noEmit` : **vert**. `next build` : **vert** (exit 0).
- Captures écran témoin `/app` dans les 4 thèmes : `theme_{terminal,atelier,schema,ardoise}.png`
  (worktree racine webapp). Contrastes OK, aucun texte illisible. Les 500 visibles = backend
  FastAPI absent en local (indépendant du theming).

## Phase B — Migration écrans (LIVRÉ, en attente de validation live)

Décisions fondateur : accent Terminal bleu confirmé ; **logos = or de marque FIXE** (badge « M »
Nav/AppHeader + MiaAgentLogo laissés intacts, theme-independent volontairement).

### B1 — UI fonctionnelle + états (SENS)
- `components/app/InstrumentSidebar.tsx` — or `#c9a961` (×6) → `--primary` / `--ring`
  (l'onglet TF actif, épingles, dot fraîcheur suivent l'accent du thème). **Vérifié en capture**.
- `components/zones/ZoneLifecycleCard.tsx` — direction bull/bear, badge « non invalidée »,
  relation prix « inside », barre de comblement → `sentinel-bull` / `sentinel-bear` / `sentinel-warn`.
- `components/zones/ZoneTimeline.tsx` — dots de phase → `sentinel-neutral/warn/bear/bull`
  (formed/interaction/terminal/ongoing).
- `components/market-reading/LivePrice.tsx` — `TONE_CLASS` bull/bear/warn → `sentinel-*`.
- `components/market-reading/sections/StructureSection.tsx` — ton « warn » → `sentinel-warn`.
- `components/auth/AccountPanel.tsx`, `components/billing/SubscriptionPanel.tsx` — badge
  « Propriétaire » ambre → `sentinel-warn`.

### B2 — Chart (canvas, lightweight-charts)
- `components/app/ReadingChart.tsx` — `palette()` lit désormais les tokens LIVE via
  `getComputedStyle(document.documentElement)` (chrome : `--muted-foreground` / `--border` /
  `--secondary` ; bougies bull/bear : `--sentinel-bull` / `--sentinel-bear`). Effet re-keyé sur
  `resolvedTheme` (les bascules dark→dark repeignent). Triplet HSL émis en `hsl(H, S%, L%)` /
  `hsla(…, a)` (format accepté par la lib). **Non visible en local (backend absent → « Données
  indisponibles ») : à valider en live avec données.**
- Laissé fixe (palette de détection sobre, theme-neutre, à re-mapper seulement si souhaité après
  revue live) : `LEVEL` (bos/choch/retest), `ZONE` (ob/fvg), `LIQUIDITY` (bsl/ssl), `LIVE`,
  `HIGHLIGHT`, `structureMarkers.ts`. Ces hues restent réservées au SENS.

### Vérifications Phase B
- `tsc` vert · `next build` vert (exit 0) · **vitest 482/482** (0 régression).
- Captures `/app` 4 thèmes : onglet TF actif suit l'accent (bleu/teal/cyan/or), logo or fixe.

## Phase C — Réglages « Apparence » + sélecteur (LIVRÉ)
- **Source unique** : `lib/theme/themes.ts` (id, base clair/sombre, nom, description, swatch de
  prévisualisation static). Consommée par le menu Nav ET le picker réglages.
- **Menu de thème (Nav + AppHeader)** : `components/theme/ThemeMenu.tsx` — bouton palette →
  menu accessible (role=menu, `menuitemradio`, coche sur l'actif), popover click-outside + Échap,
  mount-guard anti-mismatch. Remplace l'ancien `theme-toggle.tsx` (supprimé). **Vérifié en capture**.
- **Section « Apparence » (`/compte`)** : `components/theme/AppearancePicker.tsx` — 4 vignettes
  cliquables (mini-maquette bg/panel/accent/bull/bear + nom + 1 ligne), `role=radiogroup`,
  application immédiate + persistance. Ajoutée dans `AccountPanel`.
- **e2e** : `theme-and-pwa.spec.ts` réécrit → pilote le menu et vérifie `data-theme` (atelier↔terminal).

### Vérifications Phase C
- `tsc` vert · `next build` vert · **vitest 482/482** (0 régression).
- Menu 4-thèmes fonctionnel + persistance reload confirmés (Playwright). Captures :
  `phaseC_menu_open.png`, `theme_{terminal,atelier,schema,ardoise}.png`.

## Reste à valider EN LIVE (backend + session requis — non observables en local)
1. **Chart** (B2) : bougies/chrome themés avec vraies données ; bascule dark→dark repeint.
2. **Section Apparence** (`/compte`) : rendu derrière l'auth (redirige sans session en local).
   Le rendu des vignettes est prouvé identique via le menu Nav (mêmes primitives).
