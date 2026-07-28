# AUDIT — VZ-1b · Correctif de cadrage (événements temporels + zoom des zones)

Branche : `feat/vz-1b-cadrage-correctif` (depuis `origin/main` @ 8af18c9, PR #90 VZ-1).
Frontend, zéro diff moteur. Correctif de VZ-1. Date : 2026-07-28.

## Défaut 1 — les événements ne menaient qu'au prix, pas au moment

**Cause racine : un problème de ROUTAGE, pas de capacité de la bibliothèque.** Le cadrage
temporel (déplacement horizontal vers une bougie précise) **est disponible** et déjà
utilisé — `animateCamera` anime l'axe X via `timeScale().setVisibleRange({from,to})` en
**temps** (secondes), et `frameEvent` calcule une fenêtre temporelle centrée sur la bougie
de confirmation.

Le vrai coupable : dans **`RegimeCard.tsx`**, les panneaux **Maturité** et **Dernier
événement** rendaient les lignes d'événement avec un bouton `<PxBtn value={e.level}>` qui
appelle `traceLevel → setReferenceLevel` — soit une **ligne de prix horizontale, sans
dimension temporelle** (cadrage vertical seul). D'où « va au niveau 4 100, trace une ligne,
mais pas à la bougie ».

Correctif : nouveau composant **`EvBtn`** (même affordance que `PxBtn`, `pxbtn`/`aria-pressed`)
qui sélectionne l'événement sur le **canal EVENT** de la sélection unifiée (avec
`atSec = broken_at`, `direction`, `level`, `kind`) → passe par `frameEvent` → **cadrage
temporel sur la bougie de confirmation** (marqueur emphasé + ≥20 bougies avant / ≥10 après,
niveau franchi visible en contexte). Les 4 emplacements d'événement convertis :
`RegimeCard.tsx` niveau franchi de l'ancre CHOCH, « événements depuis », extrême franchi du
dernier, journal. Les `PxBtn` de **vrais niveaux** (bornes de range, repères calendaires
jour/semaine) restent en `traceLevel` — un repère n'est pas un événement.

**Structure de marché** (`StructureCard`/`StructureSection`) route déjà ses BOS/CHOCH par le
canal EVENT depuis VZ-1 — inchangé, correct. **Mobile `RegimeSection`** n'a aucun événement
cliquable (lecture seule) — rien à corriger. Étiquette du marqueur enrichie : **date + heure
complètes** (`formatLocalDayHm`, « BOS ↓ · 28/07 14:00 ») ; le niveau franchi s'affiche sur
l'étiquette d'axe adjacente.

**Les événements exposent bien l'horodatage** (`broken_at` ISO) — aucune extension moteur.

## Défaut 2 — zoom trop serré sur les zones

Les seuils sont des **constantes nommées centralisées** dans **`lib/chart/focusController.ts`**.
Correctif :

- `ZONE_OCCUPANCY_TARGET` 0,42 → **0,20** ; `ZONE_OCCUPANCY_MIN` 0,25 → **0,12** ;
  `ZONE_OCCUPANCY_MAX` 0,60 → **0,30**. À 20 % d'occupation, la marge est **~40 %** de
  chaque côté (règle respectée par construction).
- `frameZone` accepte désormais le **prix courant** (`price`) et l'inclut dans la vue tant
  que l'écart le permet **sans faire tomber la bande sous 12 %** (`ZONE_OCCUPANCY_MIN`).
  Au-delà de ce plancher, on s'arrête (le prix peut rester hors champ) plutôt que de
  dézoomer les bougies en cheveux — la lisibilité de la zone prime. Constante de respiration
  au-delà du prix : `ZONE_PRICE_PAD_FRAC = 0,25` de la hauteur de bande.
- Horizontal (formation → bougie courante + marge) : **déjà correct**, inchangé.

**Tension assumée** : quand le prix est replié dans une vue asymétrique, la marge du côté
opposé peut descendre sous 40 % (la fenêtre grandit). C'est le compromis voulu par la
mission (« mieux vaut un cadrage un peu large », « voir la zone sans voir le prix n'a aucune
utilité ») ; la garantie ≥40 % vaut pour le cadre de base (sans repli du prix).

## Capacités réelles de cadrage temporel

`lightweight-charts` 5.2.0 : le déplacement horizontal vers une bougie précise **est
disponible** via `timeScale().setVisibleRange({from,to})` en temps absolu (epoch secondes),
animé image par image par le tween rAF `animateCamera`. Le cadrage vertical passe par
`autoscaleInfoProvider` (plage cible `framingTargetRef`). Aucun easing natif — tout est
construit côté application (inchangé depuis VZ-1).

## Emplacement des constantes de cadrage

**Toutes** dans `webapp/lib/chart/focusController.ts`, en tête de module : `ZONE_OCCUPANCY_TARGET`
/ `_MIN` / `_MAX`, `ZONE_PRICE_PAD_FRAC`, `EVENT_BARS_BEFORE` / `_AFTER`, `FRAME_MARGIN_FRAC`,
`LEVEL_CONTEXT_BARS`, `LEVEL_READABILITY_MIN`, `CAMERA_TWEEN_MS`. Prêtes à ajuster après
essai live.

## Comportement VZ-1 inchangé

Sélection unique produit-wide, re-clic/Échap qui restaurent la vue, transitions animées,
`prefers-reduced-motion`, aucune animation directionnelle prix→cible, verrou d'identifiant
intact (événement/niveau sur canal distinct).

## Tests

- **Unitaires** (`focusController.test.ts`, mis à jour) : occupation **12–30 %** (cible ~20 %),
  marges **≥40 %** (cadre de base), **prix courant replié** quand l'écart le permet, **plancher
  12 % respecté** pour un prix lointain (pas de dézoom), formation+courante dans la fenêtre.
  Suite complète **635 tests verts**.
- **Playwright** 1280×800 **et** 390×844 (`vz-1-focus.spec.ts`) : geste unifié clic→sélection
  + toggle, Escape, sélection unique inter-familles, sous 768 px — **4/4 verts**.
- `tsc --noEmit` **0 erreur**, `next build` **vert**.

## Ce qui n'a pas pu être fait

- **Assertions e2e sur le cadrage lui-même** : le cadrage (zoom/position) est peint sur le
  canvas et le clic d'événement RegimeCard vit sur `/app` desktop (backend requis) — non
  testable headless. La math de cadrage (occupation, prix inclus, plancher, couple
  événement) est couverte par les tests unitaires du `focusController` ; l'e2e vérifie le
  geste au niveau DOM (aria-pressed). Validation visuelle du zoom/animation à faire **en
  live**.

## Fichiers touchés (frontend uniquement)

`lib/chart/focusController.ts`, `components/app/ReadingChart.tsx`,
`components/app/RegimeCard.tsx`, `lib/chart/__tests__/focusController.test.ts`,
`docs/audits/AUDIT-vz-1b-cadrage-correctif.md`.
