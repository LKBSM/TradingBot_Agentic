# MKT-1 — Registre de marchés + sélecteur réutilisable + fondation « prêt pour 80 marchés »

Branche : `feat/market-registry-and-selector` (worktree dédié `wt-market-registry`, depuis `origin/main` `0d4a906`).
Date : 2026-08-21.

## Objectif

Rendre le produit STRUCTURELLEMENT prêt à passer de 2 marchés à 80+ **sans dupliquer de
code par marché**. Ce PR livre la FONDATION ; il ne peuple pas 80 marchés.

Décisions du fondateur (diagnostic) appliquées :
1. **Séquence** : fondation + adoption sur /app et /zones dans CE PR ; /actualites (branchement
   news) et /scanner en missions de suivi réutilisant le même composant.
2. **driver_currencies** : le registre **référence** `config/event_market_map.json` (inchangé) ;
   le pipeline news n'est pas touché.
3. **Épinglés** : au niveau **marché** (`mia.pinnedMarkets.v1`), avec mention « non synchronisé ».

## Livrable A — Le registre = source unique (calqué sur TF-1, déjà validé)

- `config/markets.json` — source de vérité unique. Champs par marché : `id, label, symbol,
  type (metal|fx|crypto|index), priceDecimals, glyph, timeframes[]`.
- `scripts/gen_markets.mjs` → `webapp/lib/markets.generated.ts` (mode `--check` pour la CI/tests).
- `src/intelligence/market_registry.py` — façade backend (dependency-light : json + stdlib).
- `webapp/lib/markets.ts` — façade frontend (helpers : `marketLabel/PriceDecimals/Glyph/Timeframes`,
  `ALL_MARKET_IDS`, `MARKET_LABEL`, `MARKET_PRICE_DECIMALS`).

**Les déclarations en dur dispersées ont été REMPLACÉES (pas doublées) :**

| Site avant | Après |
|---|---|
| `src/intelligence/lookback_config.py` `supported_instruments()` (lisait les clés de `lookback_depths.json`) | lit `market_registry.all_ids()` |
| `src/api/routes/live_price.py` `frozenset({"XAUUSD","EURUSD"})` | `frozenset(supported_instruments())` |
| `webapp/lib/market-reading/perimeter.ts` `SUPPORTED_INSTRUMENTS = ['XAUUSD','EURUSD']` | `= ALL_MARKET_IDS` |
| `formatters.ts` `INSTRUMENT_LABEL` + `PRICE_DECIMALS` | dérivés de `MARKET_LABEL` / `MARKET_PRICE_DECIMALS` |
| `use-reading-formatters.ts` `PRICE_DECIMALS` (doublon) | `MARKET_PRICE_DECIMALS` |
| `CalendarMonthView/Workspace.tsx` `MARKETS = ['XAUUSD','EURUSD']` | `= ALL_MARKET_IDS` |
| `CalendarPreview.tsx` `ALL_MARKETS = new Set([...])` | `new Set(ALL_MARKET_IDS)` |

`candles.py`, `market_reading.py`, `state.py` (structure), `chatbot`, `signal_summary_provider`
consommaient déjà `supported_instruments()` → propagation transparente via ce seul seam.

**Invariants garantis (tests) :** le périmètre backend = le registre ; tout marché actif possède
un preset de prévision (`market_registry.all_ids() ⊆ volatility_forecaster.get_instrument_registry()`) ;
toute timeframe d'un marché existe dans TF-1.

## Livrable B — `<MarketSelector>` réutilisable (recherche + épinglés + timeframe)

`webapp/components/market/MarketSelector.tsx` — un seul composant, trois variantes de rendu
pour les trois shells qui en avaient besoin, **même logique** (registre + recherche + épingles) :
- `rail` — colonne toujours ouverte, CSS du rail de la coquille (`.mkt/.tf/.rail-lbl`).
- `panel` — colonne toujours ouverte, Tailwind (sidebar mobile /app).
- `bar` — forme compacte d'en-tête : bouton-liste marché + pastilles timeframe (en-tête /zones).

- `webapp/lib/market-reading/market-pins.ts` — `usePinnedMarkets()`, localStorage
  `mia.pinnedMarkets.v1`, assaini contre le registre, sync inter-onglets, mention
  « non synchronisé » (9 locales).
- La timeframe affichée dérive des `timeframes` du marché sélectionné (structure par-marché,
  prête pour 80), le garde M1 (`NEXT_PUBLIC_LB1_ENABLE_M1`) s'appliquant par-dessus.
- **Honnêteté (LIGNE §3)** : recherche/épingles n'affichent QUE des marchés du registre (aucun
  fantôme) ; recherche sans résultat → message explicite « Aucun marché ne correspond à … »,
  jamais de repli silencieux.

### Système d'épingles COMBO retiré (superseded)
`InstrumentSidebar.tsx`, `pins.ts` (épingles par combo `mia.pinnedCombos.v1`) et leurs tests
supprimés — remplacés par le sélecteur partagé + épingles par marché. Pas de double système.

## Livrable C — Rattachement news (différé, décision fondateur)

Aucune logique news nouvelle dans ce PR. Le rattachement automatique
(marché sélectionné → news de ses `driver_currencies`) est déjà câblé côté backend
(`calendar_service.attach_markets()`) et sera branché sur le nouveau sélecteur dans la mission
de suivi /actualites.

## Adoption

| Surface | Avant | Après |
|---|---|---|
| /app rail (desktop) | `ShellRail` blocs MARCHÉS + UNITÉ codés en dur | `<MarketSelector variant="rail">` |
| /app sidebar (mobile) | `InstrumentSidebar` (combos) | `<MarketSelector variant="panel">` |
| /zones en-tête | 2× `Segmented` marché/TF codés en dur | `<MarketSelector variant="bar">` |

## Discipline & tests

- **tsc** : 0 nouvelle erreur (3 erreurs pré-existantes sur `lib/scanner-chat/__tests__/
  dictation-copy-honesty.test.ts` — vérifiées présentes sur `origin/main` propre ; `next build`
  les exclut, d'où le vert en prod).
- **vitest** : `markets-guard` (registre = source unique + garde anti-liste-en-dur) et
  `MarketSelector` (7/7 : registre, recherche, no-result, épingle, timeframe, bar) verts en
  isolé ; suite globale 112 passés (échecs résiduels = timeouts de démarrage de worker
  Defender, pas des assertions — flake connu).
- **pytest** : `test_market_registry.py` (nouvelle, invariants) + `test_lookback_config.py`
  verts (41 passés).
- **Test d'honnêteté « add a market = one line »** : entrée fictive `TESTMKT` ajoutée à
  `markets.json` → régénération → le sélecteur la rend (MarketSelector 7/7 toujours vert) et
  **0** fichier de page/composant/backend ne la référence (grep) → preuve qu'ajouter un marché
  ne touche AUCUN code de page. Entrée retirée avant livraison.
- **Build** : `npm run build` (next build) vert — `.next/BUILD_ID` généré (Next ne l'écrit
  qu'en cas de succès : tsc + ESLint prod OK, y compris le nouveau module généré).
- **Playwright** (`mkt1-market-selector.spec.ts`) : **4/4 verts** — 1280×800 (rail /app :
  marchés du registre + recherche + no-result + épingle ; bar /zones : bascule de marché →
  URL `instrument=EURUSD`) et 390×844 (panel /app : marchés + recherche ; bar /zones rendu).
  Servi par `next start` sur le build de prod. Captures dans `docs/audits/mkt1-shots/`
  (`app-rail-desktop`, `zones-bar-desktop`, `app-panel-mobile`, `zones-bar-mobile`).

## Notes / dette

- `state.py:_guess_symbol` garde un repli littéral `"XAUUSD"` : c'est un défaut d'extraction du
  symbole d'un scanner, pas une déclaration de catalogue de marchés — laissé intentionnellement.
- `lookback_depths.json` conserve sa clé `instruments` mais **uniquement pour les overrides de
  profondeur** (l'ensemble actif est désormais le registre). Documenté par le seam
  `supported_instruments()`.

## Merge

Sur `main` UNIQUEMENT après confirmation visuelle live du fondateur (captures ci-dessus).
