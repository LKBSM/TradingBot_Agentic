# AUDIT — VZ-1 · Geste unifié « clic → graphique → cadrage »

Branche : `feat/vz-1-focus-unifie` (depuis `origin/main` @ 8795155, PR #89). Frontend
uniquement, zéro diff moteur. Date : 2026-07-27.

## 0. Objectif livré

Un seul geste, appris une fois : cliquer un élément dans un panneau amène le graphique
dans la vue et le cadre sur cet élément, sans que l'utilisateur touche au graphique. Les
**trois familles** se comportent désormais de façon identique :

- **A) ZONES** — Order Blocks / FVG.
- **B) ÉVÉNEMENTS** — BOS / CHOCH (auparavant **non cliquables**).
- **C) NIVEAUX** — poches de liquidité **et** repères temporels (repères RG-1c/d absorbés
  dans le geste unifié après confirmation live du périmètre).

Avant : trois chemins distincts et incohérents (zones = `FocusCommand` horizontal + saut
sec ; repères = canal `referenceLevel` + `scrollIntoView` non animé ; événements = rien).
Après : **un seul module de focus** et **un seul état de sélection** partagés.

## 1. Capacités réelles de la bibliothèque de charting

`lightweight-charts` **5.2.0** (Apache-2.0).

| Capacité | Réel | Détail |
|---|---|---|
| Cadrage horizontal (temps) | ✅ | `timeScale().setVisibleRange({from,to})` (temps absolu) — utilisé. |
| Cadrage **vertical** (prix) | ✅ | Via le callback `autoscaleInfoProvider` de la série (déjà en place pour le plancher 0,3 %). Généralisé : d'un prix unique (RG-1c/d) à une **plage cible [min,max]** lue depuis `framingTargetRef`. `priceScale().setAutoScale(true)` force le recalcul immédiat. |
| **Animation / easing natif** | ❌ | `setVisibleRange` **saute**. Aucun easing. **Construit à la main** : boucle `requestAnimationFrame` interpolant X (temps) et Y (plage prix) image par image, easing `easeInOutCubic`, ~400 ms (`lib/chart/focusController.ts::animateCamera`). |
| Markers | ✅ | `createSeriesMarkers` — le marqueur de l'événement sélectionné est repeint à l'accent. |
| Lignes de prix | ✅ | `createPriceLine` — ligne de niveau (repère / poche / événement). |

**Le point critique (cadrage vertical) existe** ; le vrai coût de la mission était
l'animation, entièrement bâtie côté application.

## 2. Architecture

- **`lib/chart/focusController.ts`** — LE module unique. Math de cadrage pure et testable
  (`frameZone` / `frameEvent` / `frameLevel`) + le tween caméra rAF (`animateCamera`) +
  easing. Aucun DOM.
- **`lib/chart/viewState.tsx`** — `selection: ChartSelection | null` devient la **source
  unique**. `highlightZoneId` (surbrillance bleue) et `referenceLevel` (ligne repère) en
  sont **dérivés** → un seul élément sélectionné dans tout le produit. Le chatbot
  (`focus_zone`/`highlight_zone`/`clear_highlight`) et les panneaux passent par le même
  `select`/`clearSelection`. **Escape** déselectionne globalement (au niveau du provider,
  donc actif sur toutes les surfaces, landing incluse).
- **`components/app/ReadingChart.tsx`** — un effet unique piloté par `selection` : calcule
  le cadre via le focusController, lance le tween (X+Y), `scrollIntoView` fluide, gère
  l'indicateur de bord et la restauration de la vue précédente au deselect.
- **`lib/chart/zoneOverlayPrimitive.ts`** — respiration de la zone sélectionnée
  (`setHighlightPulse`, opacité seule, composé, 60 fps).

Le **verrou d'identifiant n'est ni élargi ni contourné** : une ZONE ne porte que son id
moteur (géométrie résolue par id). ÉVÉNEMENTS et NIVEAUX portent leur géométrie sur le
**canal de sélection distinct** — exactement le chemin que le repère RG-1c/d empruntait
déjà, jamais la liste blanche `coerceViewAction` (qui interdit tout champ prix/niveau).

## 3. Seuils de cadrage retenus (validés 2026-07-27)

- **ZONE** — de la bougie de formation à la bougie courante, occupation cible **42 %**,
  bornée **[25 %, 60 %]**, marges verticales **≥ 15 %** haut/bas.
- **ÉVÉNEMENT** — centré sur la bougie de confirmation, **≥ 20 bougies avant / ≥ 10 après**
  (jamais au-delà de la bougie courante). Niveau franchi **et** bougie de confirmation
  visibles ensemble (Y englobe le niveau + le haut/bas de la bougie), marge 15 %.
- **NIVEAU** — niveau **et** prix courant visibles ensemble, marge 15 %. **Seuil de
  lisibilité** : si l'occupation de la fourchette de bougies récentes tomberait sous
  **12 %** de la hauteur (`LEVEL_READABILITY_MIN`) — c.-à-d. bougies écrasées — on bascule
  en repli : fenêtre lisible sur les bougies vivantes + **indicateur de bord discret**
  (chevron ▲/▼) montrant la direction du niveau hors écran. La lisibilité prime.
- **Animation** — ~400 ms, `easeInOutCubic`. `scrollIntoView` fluide ~300 ms.

## 4. Honnêteté de l'animation (§D)

Le mouvement est celui d'une **caméra**, jamais du prix. Aucune flèche/traînée
prix→cible, aucune particule directionnelle, aucune hiérarchie d'importance. L'indicateur
de bord (niveau) indique une **position**, pas un mouvement. La respiration de la zone
ne touche que l'opacité de la surbrillance ; la géométrie ne bouge jamais.

## 5. Accessibilité & responsive

- Éléments de liste focalisables, **Entrée** sélectionne, **Échap** déselectionne +
  restaure la vue, `aria-pressed` correct. L'état sélectionné n'est jamais signalé par la
  seule couleur (bordure/anneau + marqueur en plus).
- `prefers-reduced-motion` : transitions supprimées, cadrage **instantané**, respiration
  désactivée, `scrollIntoView` en `auto`. Le geste reste pleinement fonctionnel.
- < 768 px : `scrollIntoView({block:'nearest'})` amène le graphique dans la vue avant le
  cadrage (le graphique est souvent hors écran dans l'onglet « Lecture »).
- Cible disparue : message honnête (« Cet élément n'est plus détecté dans la lecture
  courante. »), aucun cadrage, rien d'inventé.

## 6. Tests

- **Unitaires** (nouveaux) : `focusController.test.ts` (14 — occupation 25-60 %, marges,
  couple événement visible, repli lisibilité + indicateur, tween/easing/cancel,
  reduced-motion), `selection.test.tsx` (8 — sélection unique inter-familles, dérivations,
  Escape), `structureMarkers.test.ts` (+2 — emphase événement sélectionné).
- **Tests existants mis à jour** : `zone-click-to-chart`, `zone-focus-deeplink` (sondes
  migrées de `view.focus` vers `selection`). **Suite complète verte.**
- **Playwright** 1280×800 **et** 390×844 (`vz-1-focus.spec.ts`) : clic→sélection + toggle,
  Escape→deselect, sélection unique inter-familles (un événement déselectionne la zone),
  geste sous 768 px. **4/4 verts** (sur la galerie landing, sans backend).
- `tsc --noEmit` **0 erreur**, `next build` **vert**.

## 7. Ce qui n'a pas pu / dû être fait

- **Cadrage vertical = fenêtre exacte pendant la sélection.** Tant qu'un élément est
  sélectionné, l'axe Y est tenu sur la fenêtre cadrée (pan horizontal libre). C'est
  volontaire (la caméra reste cadrée) ; se relâche au deselect. Le drag vertical manuel de
  l'axe reste possible (désactive l'autoscale jusqu'au prochain cadrage).
- **Assertions e2e sur le canvas.** Playwright ne lit pas les pixels du canvas ; le geste
  est vérifié via `aria-pressed` (état de sélection unique = source unique), et la math de
  cadrage par les tests unitaires du focusController. Pas d'assertion pixel du zoom.
- **Repère temporel = famille C incluse** après re-cadrage du périmètre en cours de
  mission (la donnée `reference_levels` RG-1c existe ; mon diagnostic initial, sur un
  checkout antérieur, l'avait manquée). Aucune donnée nouvelle inventée.
- **e2e backend** : le proxy `:8000` (auth/api) refuse la connexion en local — pré-existant
  et sans effet sur ces tests (galerie landing avec fixtures).

## 8. Fichiers touchés (frontend uniquement)

`lib/chart/focusController.ts` (nouveau), `lib/chart/viewState.tsx`,
`lib/chart/viewActions.ts`, `lib/chart/structureMarkers.ts`,
`lib/chart/zoneOverlayPrimitive.ts`, `components/app/ReadingChart.tsx`,
`components/app/DesktopReading.tsx`, `components/app/ReadingColumn.tsx`,
`components/app/StructureCard.tsx`, `components/app/LiquidityCard.tsx`,
`components/market-reading/sections/StructureSection.tsx`, `components/shell/pages.css`,
`messages/*.json` (9 locales, 5 clés), tests unitaires + `tests/e2e/vz-1-focus.spec.ts`.
