# AUDIT — Mission UI-1 · Socle visuel (tokens, thèmes, shell 3 colonnes)

Branche : `feat/ui-1-tokens-theming-shell` (worktree dédié, base `origin/main` 86b852e).
Périmètre : frontend Next.js 15 uniquement. **Moteur / API / données : 0 diff.**
Référence visuelle : `docs/design/reference-desktop.html` (validée 100 %).

---

## 0. Décisions fondatrices (verrouillées avec le fondateur)

| # | Fork | Décision |
|---|------|----------|
| 1 | Attribut thème | **Renommer `data-theme` → `data-design`** (le brief l'exige verbatim). |
| 2 | Intégration shell | **Route group `app/[locale]/(product)/`** ; chrome marketing → `(site)`. |
| 3 | Persistance | **Réutiliser next-themes** (déjà flash-free via script inline ; pas de cookie). |
| 4 | Réglages | Entrée rail **→ `/compte`** (route existante, pas de doublon). |
| 5 | Recompo /app | **Shell complet, fidèle réf** : le shell possède rail (MARCHÉS/UT) + chat ; le centre /app = `ReadingColumn` seul ; `MobileWorkspace` (<1280) conserve ses colonnes. |

---

## 1. Phase A — Tokens & thèmes (commit `8856b4c`)

- **Tokens littéraux verbatim** (`--bg … --r-s`) injectés sous chaque bloc de thème dans
  `app/globals.css`, **à côté** des tokens HSL shadcn existants (préservés : tout le site
  marketing + `ui/*` en dépendent). Une valeur = un thème ; le switch échange les deux
  vocabulaires d'un coup.
- 4 sélecteurs `[data-theme=…]` → **`[data-design=…]`** ; `darkMode` tailwind + attribut
  next-themes + assertion e2e alignés. Ids de thème et persistance localStorage inchangés.
- **JetBrains Mono** via `next/font/google` (`--font-mono`, poids 400/500/600, préchargé) ;
  famille tailwind `mono` → `var(--font-mono)` ; helper `.mono` (`tabular-nums` → aucun
  layout shift quand un prix change).
- `prefers-reduced-motion` : brake global (animations + transitions neutralisées).
- **`useDesign()`** (`lib/theme/useDesign.ts`) : utilitaire `setDesign` (wrapper next-themes)
  pour câblage UI-2 dans Réglages. Zéro état dupliqué.

Vérifs Phase A : tsc 0 · build vert · vitest 514/514.

## 2. Phase B — Shell 3 colonnes + composants de base

### 2.1 Route groups
`app/[locale]/layout.tsx` ne porte plus que les **providers** (Theme, intl, Auth, Chat,
ChartView, Tooltip). Le chrome est choisi par groupe (URLs **inchangées**, route groups
transparents) :
- **`(site)/layout.tsx`** — Nav + `main` + Footer + ChatPanel flottant + CookieBanner.
  Routes : landing, abonnement, conditions, confidentialite, connexion, inscription,
  methodology, mot-de-passe-oublie.
- **`(product)/layout.tsx`** — `ProductShell`. Routes : app, scanner, zones, compte.

### 2.2 Shell (`components/shell/`)
- **`ProductShell`** — grille `.app-shell` (rail 232 / center 1fr / chat 338), `.no-chat`
  (2 colonnes) hors /app. `SkipLink` + `id="main"` sur le centre.
- **`ShellRail`** — search + MARCHÉS + UNITÉ DE TEMPS + ESPACE (nav routes réelles) +
  Freshbox + microcopie. Le combo actif est **lu/écrit dans l'URL** (`?instrument=&
  timeframe=`), déjà source de vérité de /app → aucune remontée d'état. Choisir un marché/UT
  navigue vers /app avec ce combo.
- **`ShellChat`** — colonne chat dockée (/app seul) hébergeant `AppChatSidebar` inchangé
  (ChatProvider partagé depuis le layout locale → contexte aligné par les effets de /app).
- **`shell.css`** — port **verbatim** des règles de composants + chrome de la référence,
  **scopé sous `.app-shell`** (aucune fuite/collision globale). Couleurs 100 % via tokens
  littéraux → 4 thèmes gratuits.

### 2.3 Composants de base (`components/shell/primitives.tsx`)
`Card` + `CardHeader` · `Btn` (default/primary/acc/danger) · `Dot` (pulse) · `LiveBadge` ·
`EarlyAccessBadge` (ea) · `LegalInline` · `Tagx` (bear/bull/neu) · `TfPill` · `SearchField` ·
`Input` + `FieldLabel` · `Freshbox` · `LayerChip`. Wrappers fins émettant les classes de la
référence ; icônes lucide re-stylées par le CSS scopé (l'attribut de présentation cède au CSS).

### 2.4 Recompo /app
`DesktopWorkspace` (AppWorkspace) rend désormais **uniquement `ReadingColumn`** ; le rail et
le chat appartiennent au shell. `MobileWorkspace` (<1280px) **inchangé** (InstrumentSidebar +
onglets). Logique combo/URL, focus deep-link, effets chat : **intacts**.

### 2.5 i18n
- **Réutilisé** (déjà 9 locales) : `app.sidebar.markets/searchPlaceholder/searchAria/navAria`,
  `nav.scanner/zones/account`, `landing.hero.badgeLive` (« Lecture en direct »),
  `legal.disclaimer.chart` (microcopie éducative).
- **2 clés neuves** `app.rail.timeframe` / `app.rail.space`, injectées **CRLF-safe** dans les
  **9 fichiers** (parité stricte 966×9, arabe intact) — insertion pure, aucun reformat.
- Libellés « App » (produit) hardcodés comme le fait déjà `AppHeader` (précédent codebase).

---

## 3. Mapping réutilisé vs créé

| Élément | Statut |
|---|---|
| Système de thèmes (next-themes, 4 thèmes, pickers) | **Réutilisé** (attribut renommé) |
| Logique marchés/épingles/fraîcheur `InstrumentSidebar` | **Réutilisé** (mobile) ; rail = nouvelle vue URL-driven |
| `AppChatSidebar`, `ReadingColumn`, `MobileWorkspace` | **Réutilisés** tels quels |
| `Nav`, `Footer`, `ChatPanel`, `CookieBanner` | **Réutilisés** (déplacés dans `(site)`) |
| Tokens littéraux, `shell.css`, `primitives.tsx`, `ProductShell`/`ShellRail`/`ShellChat`, `useDesign`, route groups | **Créés** |

---

## 4. Écarts restants vs référence (honnêtes, pour UI-2)

1. **Prix/variation par marché dans le rail** — la réf affiche « Or 2 392,35 −0,42 % ». Non
   affiché : seul le combo actif dispose d'une lecture réelle ; inventer des prix multi-marchés
   violerait la ligne d'honnêteté. → différé (feed multi-marché / contexte de lecture partagé).
2. **Âge « il y a X s » de la Freshbox** — la réf montre un compteur live ; le rail (dans le
   layout) n'a pas la lecture. Affiché honnêtement : « Lecture en direct » + libellé du combo
   actif. → différé (même feed partagé).
3. **`apphead` + `legalbar` de /app** — la réf les place **dans le contenu /app** (pas le
   shell). Le contenu /app existant (`ReadingColumn` + header marché + `EarlyAccessBadge`/
   disclaimer déjà livrés PR antérieures) est **conservé tel quel** (règle « contenu
   inchangé »). Les primitives correspondantes (`LiveBadge`, `EarlyAccessBadge`, `LegalInline`)
   sont livrées et prêtes à assembler l'apphead exact en UI-2.
4. **Libellé « Réglages »** — la réf écrit « Réglages » ; on affiche **« Compte »** (clé i18n
   existante `nav.account`, route `/compte`) pour rester dans le vocabulaire de l'app.
5. **`error.tsx` / `not-found.tsx`** restent au niveau `[locale]` (providers only) → ils
   perdent la Nav marketing. Régression mineure sur états d'erreur uniquement ; à trancher.
6. **Traductions `app.rail.*`** — fournies pour les 9 locales (termes UI standards) ; relecture
   native recommandée (arabe notamment) avant prod, comme le reste de l'i18n.

---

## 5. Vérifications

| Gate | Résultat |
|---|---|
| `tsc --noEmit` | **0 erreur** |
| `next build` | **vert** — 14 routes aux URLs inchangées |
| `next lint` | **vert** (1 warning pré-existant `SubscriptionPanel.tsx`, hors périmètre) |
| `vitest` | **58 fichiers / 516 tests** |
| Backend / API / détection | **0 diff** |

### Tests ajoutés / mis à jour
- `components/shell/__tests__/ShellRail.test.tsx` (neuf) — 4 sections, nav → routes réelles,
  sélection écrite dans l'URL.
- `components/app/__tests__/AppWorkspace.test.tsx` — recomposé : combo via `initialCombo`
  (URL = source de vérité), flow fetch/skeleton/erreur/retry conservé, assertions rail/chat
  retirées (déplacées au shell).
- `components/app/__tests__/responsive.test.tsx` — desktop : rend la lecture, rail+chat = shell.
- `tests/claims-cleanup.test.ts` — résolveur de routes **route-group-aware** (transparent).
- `tests/e2e/theme-and-pwa.spec.ts` — assertion `data-design` + **persistance après reload**.
- `tests/e2e/shell.spec.ts` (neuf) — rail visible, nav ESPACE, **0 scroll horizontal @1280px**.

> e2e non exécutés localement (serveur + navigateur) → couverts par la CI `webapp-ci`.

---

## 6. Risques / points d'attention pour UI-2

- **`setDesign`** est prêt (`useDesign()`) — brancher dans le picker Réglages ; `ThemeMenu` /
  `AppearancePicker` existants peuvent y migrer.
- **Apphead/legalbar exacts** : assembler avec les primitives livrées quand une source de prix
  live du marché actif sera disponible côté /app.
- **Feed multi-marché** : condition pour les prix/variation du rail et l'âge Freshbox live.
- **Scanner/Zones/Compte** : le rail MARCHÉS/UT y navigue vers /app (comportement voulu) ; si
  UI-2 veut un état combo propre à ces pages, prévoir un provider partagé.
- **Réconciliation main** : chantier responsive parallèle → surveiller `useStackedLayout`,
  `AppWorkspace`, `globals.css` au merge.
