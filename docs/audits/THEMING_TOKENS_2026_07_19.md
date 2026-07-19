# Système de thèmes — tokens UI + sélecteur (4 thèmes)

Branche : `feat/design-tokens-theming` (worktree dédié, depuis main `0951d34`).
Ligne : **présentationnel only**. Aucun changement moteur / logique IA / textes de la
ligne. La couleur reste réservée au SENS (états zones/liquidité via `sentinel-*`) — aucun
vert/rouge décoratif « achat/vente » introduit par un thème.

## Phase A — Socle (LIVRÉ, en attente de validation live)

### Mécanisme retenu
- `next-themes` en **4 thèmes nommés** : `terminal` (défaut) · `atelier` · `schema` · `ardoise`.
- `attribute="class"` + `value` map = **une seule classe `.theme-*` par thème** sur `<html>`
  (le multi-token `'dark theme-x'` casse `DOMTokenList.remove` de next-themes → interdit).
- Les **3 thèmes sombres** pilotent le variant Tailwind `dark:` via
  `darkMode: ['variant', ['.theme-terminal &', '.theme-schema &', '.theme-ardoise &']]`.
  Atelier (clair) en est volontairement absent. → tous les `dark:` existants continuent de
  fonctionner sans les toucher.
- **Sans flash SSR** : script bloquant next-themes + `suppressHydrationWarning` sur `<html>`
  + `:root` = valeurs Terminal (fallback pré-hydratation cohérent).
- **Persistance** : `localStorage['theme']` (défaut next-themes). Défaut = `terminal`.
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

## Phase B — Migration écrans (À FAIRE, après GO)
Cibles de dette hardcodée identifiées :
- `components/app/InstrumentSidebar.tsx` — or `#c9a961` (×6) → `--primary`/`--ring`.
- `components/Nav.tsx`, `components/app/AppHeader.tsx` — dégradé badge logo ambre + `text-white`.
- `components/chat/MiaAgentLogo.tsx` — hex bull/bear (asset de marque, à trancher).
- `components/zones/ZoneLifecycleCard.tsx`, `ZoneTimeline.tsx`, `market-reading/LivePrice.tsx`,
  `sections/StructureSection.tsx`, `auth/AccountPanel.tsx`, `billing/SubscriptionPanel.tsx`
  — classes palette `emerald/rose/amber/sky` (SENS) → `sentinel-*`.
- `components/app/ReadingChart.tsx` (+ `lib/chart/structureMarkers.ts`) — palette canvas
  (lightweight-charts) : lire les tokens via `getComputedStyle` au montage + au changement de
  thème (source unique), en gardant les couleurs de détection réservées au sens.

## Phase C — Réglages « Apparence » (À FAIRE, après GO)
Section dans `AccountPanel` (`/compte`) : 4 vignettes cliquables (nom + 1 ligne), aperçu,
application immédiate + persistance. Le `theme-toggle` deviendra un sélecteur 4-thèmes.
