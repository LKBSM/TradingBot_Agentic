# Audit — Événements de structure cliquables + cadrage « vue d'ensemble »

**Date :** 2026-08-20
**Branche :** `feat/structure-events-clickable-and-chart-framing` (worktree dédié, depuis `origin/main` @ `acc72d8`)
**Nature :** diagnostic lecture seule + **vérification live** (aucune modification de code produit)

## Verdict

**Les deux objectifs de la mission sont déjà implémentés et livrés sur `origin/main`.**
La vérification Playwright (1280×800 + 390×844) le prouve visuellement. Aucune
implémentation n'était requise.

## Objectif A — événements de structure cliquables (verrou d'id réel)

Les trois surfaces sont cliquables, toutes via l'**id réel émis par le moteur**
(`event.id` = `<kind>_<iso>_<dir>`), jamais par matching prix/heure reconstruit :

| Surface | Fichier | Mécanisme |
|---|---|---|
| Panneau Structure (lignes CHOCH/BOS en tête) | `components/app/StructureCard.tsx:284-329` | `selectEvent()` → `eventId(kind,ev)`, `role=button`, clavier, classe `sel` |
| Journal des événements (tuile « Dernier événement » → détail Donnée) | `components/app/RegimeCard.tsx` (`EvBtn`, `case 'last'`) | chaque entrée = bouton → `selectEvent()` canal temporel |
| Structure mobile (accordéon) | `components/market-reading/sections/StructureSection.tsx:142-152,232-248` | `<Row aria-pressed>` → même `selectEvent`/`eventId` |
| Résolution côté graphique | `components/app/ReadingChart.tsx:1370-1385` | `findEventById(structure, id)` **rejette** un id inconnu/périmé — « never fall back to the nearest timestamp » |

Livré par **STR-2** (`cf7fc1b`, 2026-08-17).

## Objectif B — cadrage « vue d'ensemble » (marge proportionnelle, transition sobre)

`lib/chart/focusController.ts` implémente déjà exactement la règle demandée :

- `frameEvent` : **20 bougies avant / 10 après** + marge verticale **15 %** du span (relatif, pas de constante en dur).
- `frameZone` : **~100 bougies** (bornes 60–120), élargi par **VZ-1c** (`936ab53`, 2026-07-28) pour corriger « cadrage trop serré ».
- `animateCamera` : tween **~400 ms**, `easeInOutCubic` doux, les deux axes, respect `prefers-reduced-motion`. Panning caméra, jamais le prix.
- Zones **et** événements partagent le même chemin (`ReadingChart.tsx:1303` effet unifié `selectionKeyStr`).

## Preuve live (`structure-events-shots/`)

Spec : `tests/e2e/structure-events-verify.spec.ts` (bougies alignées sur les
timestamps réels de la fixture). **3 tests verts.**

| Capture | Constat |
|---|---|
| `01-desktop-initial` | vue large initiale, overlays structure |
| `02-desktop-choch-framed` | clic CHOCH → cadré sur niveau **2384.20**, marqueur « CHOCH ↑ · 26/05 à 05:30 » (= 09:30 UTC, vrai `broken_at`), contexte avant/après |
| `03-desktop-bos-framed` | clic BOS → cadré sur **2391.50**, « BOS ↑ · 26/05 à 07:15 » — **bougie distincte du CHOCH** (verrou d'id : chaque événement pointe SA bougie) |
| `04-desktop-restored` | re-clic → restauration de la vue antérieure |
| `05-mobile-initial` / `06-mobile-choch-framed` | même geste et même cadrage à 390×844 (marge proportionnelle tient à l'échelle) |

## Hypothèse sur l'écart perçu

Scénario récurrent du projet (cf. incident DATA-1) : la capture de la mission
correspond très probablement à un **build déployé / worktree en retard** sur
`origin/main`. Le code à jour produit déjà le comportement demandé.

## Nuances réelles restantes

1. **Empan événement (~30 bougies) < empan zone (~100)** — réglage possible si les événements semblent encore serrés ; mais 30 est un choix assumé (garder lisible le couple « niveau cassé + bougie de confirmation »).
2. **Tuile « Dernier événement » du Régime** ouvre un panneau détail ; le saut-au-graphique se fait via le bouton *dans* le détail, pas au clic sur la tuile.

## Correctif implémenté sur cette branche — événement ancien hors-fenêtre

**Manque traité (le seul vrai bug fonctionnel).** Avant : cliquer un événement dont
la bougie de confirmation est plus ancienne que la fenêtre de bougies chargée
faisait cadrer la caméra sur du **vide** à gauche des données (l'effet ne
déclenchait pas `loadOlder`).

**Fix (`components/app/ReadingChart.tsx`, présentationnel — 0 détection touchée) :**
- Dans l'effet de sélection unifié, la branche `event` détecte `evSec < première bougie chargée` : elle **diffère** le cadrage, capture le point de restauration, amène le graphique à l'écran, et lance `loadOlder()`.
- Un nouvel effet différé pagine l'historique **page par page** (via le `loadOlder`/`reachedStart` existants de CHART-2) jusqu'à ce que la bougie de l'événement soit chargée — ou que le début de l'historique / un backstop de `MAX_EVENT_HISTORY_PAGES` (6) soit atteint — puis **anime une seule fois** vers l'événement (même `frameEvent` + `animateCamera` que le cas en-fenêtre).
- Jamais de cadrage sur du vide ; le chip « chargement de l'historique » (`isLoadingOlder`) existant donne le retour visuel pendant la pagination ; le marqueur accentué + la ligne de niveau réapparaissent une fois la bougie chargée.
- Réutilise strictement le chemin de ciblage existant (id-lock `findEventById`, `frameEvent`, `animateCamera`) — aucune reconstruction de cible par prix/heure.

**Preuve (`structure-events-shots/`, spec `structure-events-verify.spec.ts`) :**

| Capture | Constat |
|---|---|
| `07-oldevent-initial` | vue récente ; l'événement ancien est hors écran ; **aucun** auto-paging au chargement (asserté `before` non requêté) |
| `08-oldevent-framed` | clic → `?before=` déclenché (asserté) → caméra cadrée sur la bougie « CHOCH ↓ · 23/05 à 17:45 », ligne de niveau **2375.00**, contexte avant/après |

**Tests :** tsc 0 · vitest chart/app/useCandles **232/232** · Playwright vérif **3/3** · non-régression vz-1-focus + chart2-amplitude **20 passed / 12 skipped**.
