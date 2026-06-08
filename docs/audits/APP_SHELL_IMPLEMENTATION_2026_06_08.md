# Implémentation — Shell /app produit fini (graphique, pin, états d'erreur)

**Date** : 2026-06-08
**Branche** : `feat/app-shell-chart-pin-error-states` (depuis `institutional-overhaul` @ `c0699e0`)
**Périmètre** : 100 % frontend (`webapp/`). Aucune modification backend / moteur / mappers.
**Layout retenu (bloc D)** : **« Graphique d'abord »** (chart-led) — choix founder.

---

## 1. Objectif

Rendre la vue `/app` visible en « produit fini » avec des données **mock réalistes**,
pour visualiser le rendu final avant que le backend ne serve des données réelles.
Partout où une donnée ne peut pas être obtenue → état **« Données indisponibles »**
propre (jamais d'écran vide).

Contraintes tenues :
- ❌ Aucune API externe de données, **aucune « TradingView API »** (lightweight-charts
  n'est qu'une **bibliothèque d'affichage**).
- ❌ Aucun marché hors catalogue V1 : **XAUUSD + EURUSD × M15/H1/H4 = 6 combos**.
- ❌ Aucun mot interdit / score 0-100 de conviction — posture niveau 1.5 strict.
- Staging git explicite (jamais `git add -A`), pas de force push.

---

## 2. Mock vs réel — et où brancher le backend plus tard

| Donnée | Aujourd'hui | Branchement futur |
|---|---|---|
| **Lecture** (`MarketReading`) | `lib/mockReadings.ts` → `getMockReading()` (6 combos), calqué sur le contrat Pydantic v2.0.0 | Passer `READING_DATA_SOURCE = 'live'` dans `lib/mockReadings.ts` → la vue repasse sur `fetchMarketReading()` (`GET /api/market-reading`). **Point de bascule unique.** |
| **Bougies** (graphique) | `getMockCandles()` — marche aléatoire **déterministe** (PRNG seedé), se terminant sur le `close_price` de la lecture, enveloppe englobant les niveaux | Remplacer `getMockCandles()` par un vrai flux `Candle[]` (même forme : `time` = UNIX s croissant). |
| **Overlays** (BOS/CHOCH/OB/FVG/retest) | Lus **directement** depuis `reading.structure` | Aucun changement — déjà branché sur le contrat réel. |

Le module `lib/mockReadings.ts` porte un bandeau **TEMPORAIRE** explicite avec la
procédure de suppression. La couche mock est isolée derrière l'option `source` de
`useMarketReading` : le chemin `live` (réel) reste intact et testé.

### Démonstration intentionnelle de l'état d'erreur
`XAUUSD H4` a son **flux de bougies volontairement marqué indisponible**
(`CANDLE_FEED_UNAVAILABLE` dans `mockReadings.ts`) : la lecture textuelle reste
affichée, mais le panneau graphique montre le placeholder **« Graphique
indisponible »** (dégradation gracieuse en conditions réelles). Les 5 autres combos
affichent le graphique complet. Pour rétablir : retirer la clé du set.

---

## 3. Composants créés / modifiés

### Créés
- `webapp/lib/mockReadings.ts` — mocks 6 combos + générateur de bougies + flags.
- `webapp/lib/market-reading/pins.ts` — hook `usePinnedCombos` (localStorage, sync multi-onglets, sanitise hors-catalogue).
- `webapp/components/app/ReadingChart.tsx` — graphique Lightweight Charts (v5) + overlays SMC, props typées sur le contrat réel.
- `webapp/NOTICE` — attribution Apache 2.0 (Lightweight Charts).
- Tests : `lib/__tests__/mockReadings.test.ts`, `lib/market-reading/__tests__/pins.test.tsx`, `components/app/__tests__/InstrumentSidebar.test.tsx`, `components/app/__tests__/ReadingColumn.test.tsx`.

### Modifiés
- `components/app/InstrumentSidebar.tsx` — recherche-filtre (catalogue figé) + section « Épinglés » + toggle pin + état actif renforcé.
- `components/app/ReadingColumn.tsx` — layout chart-led (graphique en hero via `dynamic`/`ssr:false`, sinon placeholder).
- `components/app/AppWorkspace.tsx` — props `initialCombo` + `dataSource`.
- `components/app/ReadingPlaceholders.tsx` — carte « Données indisponibles » (500) + `ChartUnavailable`.
- `components/market-reading/MarketReadingCard.tsx` — slot `chartSlot` (landing inchangée).
- `app/[locale]/app/page.tsx` — démarrage sur XAU M15.
- `app/[locale]/methodology/page.tsx` + `components/Footer.tsx` — section/lien « Attributions ».
- `lib/market-reading/hooks.ts` — option `source` (live|mock).
- `package.json` / `package-lock.json` — dépendance `lightweight-charts@5.2.0`.
- Tests existants alignés (`AppWorkspace`, `responsive`, `hooks`) + `vitest.setup.ts` (timeout async).

---

## 4. Tests & build

- **Vitest** : **113/113 verts** (17 fichiers), en parallèle et en séquentiel.
  - Nouveaux : filtre catalogue, pin + persistance localStorage, états d'erreur,
    rendu/indisponibilité du graphique, déterminisme & enveloppe des bougies.
- **`tsc --noEmit`** : OK.
- **`next build`** : ✅ vert. Route `/[locale]/app` = **14.5 kB / 157 kB** first-load ;
  le chunk `lightweight-charts` est **code-splitté** (chargé à la demande, hors bundle initial).
- **Smoke** : `GET /fr/app` → 200, colonne marchés + recherche + catalogue rendus,
  aucun log d'erreur runtime côté serveur dev.

---

## 5. Commits (branche `feat/app-shell-chart-pin-error-states`)

| Bloc | Hash | Sujet |
|---|---|---|
| A | `7209d7d` | feat(app/A): mocks centralisés 6 combos + états « Données indisponibles » |
| B | `f002f9d` | feat(app/B): recherche-filtre catalogue + pin avec persistance locale |
| C | `e935c55` | feat(app/C): graphique Lightweight Charts + overlays SMC + attribution |
| D | `52539cc` | feat(app/D): layout « Graphique d'abord » + lecture XAU M15 par défaut |

---

## 6. Hors-scope (V1.1, documenté)

- Overlays fins : zone d'incertitude dédiée, repère biais poli (scope bêta = chandeliers + OB + FVG + niveaux).
- Le flux de bougies reste mock (aucun endpoint de bougies backend dans le périmètre).
- Persistance pin = localStorage (pas de sync compte).

---

## 7. Suite

- `git push` de la branche effectué. **PR à créer par le founder** sur GitHub.
- Quand le backend sert les données : basculer `READING_DATA_SOURCE = 'live'`,
  brancher un vrai flux de bougies, puis supprimer `lib/mockReadings.ts`.
