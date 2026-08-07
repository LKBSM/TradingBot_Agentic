# AUDIT CHART-1 — Zoom horizontal & étiquettes : diagnostic (Section 1, lecture seule)

> Branche `feat/chart-1-zoom-horizontal` (worktree `C:/MyPythonProjects/wt-chart-1`), base `main` @ `a464f80` (inclut PERF-2 fluidité — à ne pas régresser).
> **Aucun code écrit.** Livrable : état du zoom, données réellement disponibles par unité de temps, cause de l'empilement des étiquettes.

---

## A) ÉTAT ACTUEL DU ZOOM

**Bibliothèque** : `lightweight-charts` **5.2.0** (`webapp/package.json`). Une série `CandlestickSeries` + un **primitive canvas** `ZoneOverlayPrimitive` (zones/liquidité) + markers BOS/CHOCH. `components/app/ReadingChart.tsx` (~1560 lignes) + `lib/chart/focusController.ts`.

- **Zoom horizontal** : via `timeScale` (`setVisibleLogicalRange`, `fitContent`). **Aucune borne dure** sur le nombre de bougies visibles. Seul garde-fou : `minBarSpacing: 4` (px/bougie, plancher de densité) → en dézoome fort, la charte affiche **moins de bougies** plutôt que des traits illisibles (`ReadingChart.tsx:518`). **Il n'y a PAS de limite « 6 semaines » dans le code du zoom.**
- **Zoom vertical** : automatique (`autoscaleInfoProvider`), plancher 0,3 % du prix médian ; surchargé par le cadrage VZ-1 quand une zone/événement est sélectionné.
- **Boutons** (`ReadingChart.tsx:1498-1523`) : **+** (`zoom(0.7)`), **−** (`zoom(1.4)`) → **axe temporel uniquement** ; **⤢ Ajuster** (`fitContent`) → les deux axes (= « vue par défaut »). **Pas de bouton « aller à la bougie la plus récente »** (seul le canal chat `focus_price`→`scrollToRealTime` le fait, sans bouton).
- **Gestes** (`ReadingChart.tsx:530-542`) : molette (`handleScale.mouseWheel`), **pincement** (`handleScale.pinch`), **glissement latéral** souris (`pressedMouseMove`) et doigt (`horzTouchDrag`), inertie tactile. Glissement vertical désactivé (prix auto). **Tout est déjà branché.**
- **Vue par défaut** : les **90 dernières bougies** (+3 de marge droite) au premier chargement ; l'état de zoom est préservé entre rafraîchissements (`ReadingChart.tsx:195, 923-937`).

**⇒ Le zoom lui-même n'est PAS le mur.** Le vrai plafond est la **quantité de bougies chargées** (§B) : on ne peut pas dézoomer au-delà de ce que le front possède.

---

## B) LES DONNÉES DISPONIBLES — la vraie contrainte

### Ce que le FRONT reçoit aujourd'hui
- **400 bougies. Point.** `CHART_CANDLE_LIMIT = 400` (`webapp/lib/market-reading/hooks.ts:336`), via `GET /api/candles?instrument&timeframe&limit=400` (dernières N). Fixé délibérément par le commit **`88eb894` « profondeur affichée à 400 bougies (cache SQLite) »** (perf).
- **La lecture (`/api/market-reading`) N'EMBARQUE PAS de bougies** (`types/market-reading.ts` : candles seulement via l'enveloppe `/api/candles`). `buildChartSlot(reading, candles…)` reçoit les 400 de `useCandles`.
- **Vérifié en direct** (backend local, candles.db) : `limit=400`→**400** ; `limit=1000`→**1000** ; `limit=5000`→**HTTP 422** (plafond API `MAX_LIMIT=1000`, `src/api/routes/candles.py:54`).

**400 bougies selon l'unité** (indépendant du zoom, c'est le plafond de données) :

| Unité | 400 bougies ≈ | Profondeur candles.db (cible LB-1, défaut ; override XAUUSD **vide**) |
|---|---|---|
| **M15** | ~4 jours de cotation | **1 mois ≈ 2 880 bougies** |
| **H1** | ~2,4 semaines | 6 mois ≈ 4 387 |
| **H4** | ~9 semaines | 2 ans ≈ 4 387 |
| **D1** | ~1,6 an | 5 ans ≈ 1 831 |

> Note sur « ~6 semaines visibles » : sur **M15**, 400 bougies ≈ 4 jours ; ~6 semaines correspondrait à ~2 880 bougies (la **cible candles.db M15 = 1 mois**), ou à **~9 semaines sur H4** (400 bougies). À confirmer en live selon ton unité — mais le correctif est le même quelle que soit l'unité : **lever le plafond de 400 et charger à la demande jusqu'à la profondeur réelle**.

### Ce que la BASE contient (chemin de service = candles.db, PAS de réseau)
- **candles.db** (dev, cache) : XAUUSD **M15 2 233 · H1 933 · H4 806** ; cibles LB-1 ci-dessus.
- **`config/lookback_depths.json`** : profondeurs par durée (M15 1mo, H1 6mo, H4 2y, D1 5y). **L'override `XAUUSD` est vide `{}`** → défaut (donc **pas** 70k/24mo comme parfois annoncé).
- **CSV profond** `data/XAU_15MIN_2019_2026.csv` = **172 874 bougies M15 (2019→2026, 7 ans)** — mais c'est un **fichier de backfill/test**, **pas** le chemin de service live ; en prod candles.db est borné aux cibles LB-1.

### Peut-on demander une fenêtre plus ancienne ?
- **NON.** `/api/candles` ne sert **que les N dernières** (`get_last_n_candles`) — **aucun paramètre `from/to/before`**. `/api/coverage` (métadonnées) et `/api/structure?time_from` (zones déjà détectées) n'apportent **pas** de bougies.
- **Coût de charger davantage** : lecture **candles.db = quasi gratuite** (SQLite, ~10 ms, aucun quota). Un fetch TwelveData = 1 crédit (8/min, 800/j) mais **inutile** pour l'historique déjà en base. **Donc : dézoomer plus loin ne coûte que des lectures DB bornées** — PAS « tout charger d'un coup », PAS le CSV 7 ans.

**⇒ Réponse à la question critique** : l'historique réellement disponible sans réseau = **la profondeur candles.db par unité** (M15 ~1 mois, H1 ~6 mois, H4 ~2 ans, D1 ~5 ans). C'est la borne honnête du dézoome. Le front n'en voit aujourd'hui que **400 bougies** faute (1) d'un plafond front trop bas et (2) d'un endpoint par intervalle.

---

## C) LES ÉTIQUETTES — cause de l'empilement

- **Rendu** : sur **canvas** via le primitive `ZoneOverlayPrimitive` (`lib/chart/zoneOverlayPrimitive.ts`), peint **en phase** avec les bougies (acquis PERF-2 `86a0d2d` — **ne pas revenir à des `<div>` HTML**). `_drawZoneLabel` (l.664-700).
- **Cause de l'empilement à gauche** : `x = Math.max(2, box.left + offset)` (`zoneOverlayPrimitive.ts:676`). Quand la **bougie de formation** d'une zone est **hors fenêtre** (à gauche des 400 chargées), `timeToCoordinate` donne un x **négatif** → **clampé à x=2**. **Toutes** les zones anciennes hors-champ atterrissent donc à **x≈2, en haut à gauche**, superposées. `y` dépend du prix de la zone → elles s'empilent verticalement au même coin.
- **Collisions** : **AUCUNE gestion** (pas de décalage, pas de regroupement, pas de seuil de densité). Curation des zones = 24 actives / 12 testées (`zoneLayout.ts:50-51`) — cap défensif, ne règle pas l'empilement.
- **Badge « EN DIRECT · provisoire »** : **HTML** absolu `left-2 top-2`, **sans z-index ni espace réservé** (`ReadingChart.tsx:1458-1492`). Peint par-dessus le canvas → **recouvre** les étiquettes empilées, dans le même coin.
- **Masquage de couche** : correct — masquer OB/FVG/liquidité/breaks **retire** les étiquettes (aucun résidu ; filtrage avant construction des modèles, `ReadingChart.tsx:369-441`).
- **i18n** : « en test » `chart.inTest`, « touché » `chart.touched`, badge `chart.liveBadge` (« EN DIRECT · provisoire »/« LIVE · provisional »), « OB »/« FVG » peints en dur sur le canvas. `messages/fr.json` + `en.json`.

**⇒ Le défaut s'aggravera avec le dézoome** : plus de bougies visibles = plus de zones de formation hors des 400 → plus d'étiquettes clampées à x=2. **Corriger le zoom sans corriger les étiquettes empirerait l'affichage** — d'où le traitement conjoint.

---

## DÉCISIONS (utilisateur, 2026-08-07)
1. **Borne de dézoome = profondeur candles.db** par unité (M15 ~1 mois, H1 ~6 mois, H4 ~2 ans, D1 ~5 ans) — lectures SQLite gratuites, **pas** de réseau, **pas** le CSV 7 ans. Message discret « début des données disponibles » à la limite.
2. **Chargement = endpoint par intervalle** : `GET /api/candles?…&before=<ts>&limit=N` (lecture candles.db) + **pagination à la demande** côté front quand on dézoome — bougies déjà affichées jamais effacées, état de chargement visible sur la partie non chargée, réessai si échec. **Interdit** : tout charger d'un coup.

## Pistes retenues pour le GO (à valider — AUCUN code encore)
1. **Bornes** : resserrement max ≈ **20 bougies** (via `minBarSpacing`/clamp du zoom) ; dézoome max = **profondeur candles.db de l'unité** (borne honnête), pas au-delà.
2. **Charger plus, à la demande** : (a) relever le plafond front au-delà de 400 **et** (b) ajouter un **paramètre d'intervalle** à `/api/candles` (`before=<ts>&limit=N`, lecture candles.db) pour paginer l'historique quand on dézoome — bougies affichées jamais effacées, état de chargement visible, réessai si échec. **Interdit** : tout charger d'un coup / lire le CSV 7 ans.
3. **Limite atteinte** : message discret « début des données disponibles pour cette unité ».
4. **Au-delà de la fenêtre d'analyse** (M15 2 880 ; H1/H4/D1 500) : bougies affichées mais **aucune structure détectée** → l'indiquer (ne pas laisser croire « pas de structure »).
5. **Étiquettes** : ancrer chaque étiquette **à sa zone** (pas de clamp aveugle à x=2) ; **anti-collision** (décalage/regroupement « N zones » au survol) ; **réserver l'espace du badge EN DIRECT** (plan supérieur) ; réduire la **densité d'annotations** à fort dézoome.
6. **Perf** : mesurer le rendu avant/après à plusieurs zooms ; rester sur le primitive canvas (PERF-2).

---

## RÉALISATION (2026-08-07)

### Données — profondeur à la demande (pas de rechargement, jamais tout d'un coup)
- **Backend** `src/storage/candles_cache_store.py::get_candles_before(instrument, tf, before_ts, n)` : lit jusqu'à `n` bougies **strictement avant** `before_ts` (ascendant, index `ts DESC`, borne comparée en ISO). Miroir de `get_last_n_candles`.
- **Backend** `src/api/routes/candles.py` : param `before=<epoch UTC>` + champ `has_more`. `before` → `get_candles_before(before, limit+1)` ; `has_more = len(block) > limit` ; on retire la bougie sentinelle la plus ancienne (contiguïté). **Plancher atteint = 200 vide** (jamais 404, jamais de mur silencieux). Le chemin initial (sans `before`) calcule aussi `has_more`. `MAX_LIMIT=1000`.
- **Front** `lib/market-reading/api-client.ts` : `fetchCandlesPage()` → `{candles, hasMore}`, ajoute `&before=`. `lib/market-reading/hooks.ts::useCandles` : `mergeCandlesByTime` (union par temps), **fusionne** la fenêtre fraîche dans l'historique existant (une clôture de bougie ne **remplace** jamais l'historique), expose `hasMoreHistory / loadingOlder / olderError / loadOlder`. `loadOlder` demande `before = première bougie affichée`, **préfixe** la page ; sur échec **garde les bougies** + `olderError` (réessai possible).

### Zoom + navigation
- `ReadingChart.tsx` : `MIN_VISIBLE_BARS = 20` (resserrement max borné pour molette/pincement **et** boutons `zoom()`), `HISTORY_PREFETCH_BARS = 15`. Abonnement `subscribeVisibleLogicalRangeChange` **placé dans l'effet de création** (lié au `chart` vivant, comme le crosshair — jamais orphelin) : (1) clamp du zoom-in, (2) `loadOlder()` quand le bord gauche approche l'origine, (3) bascule des drapeaux *plancher* / *hors-fenêtre* (uniquement au changement, pas de churn React). Boutons molette (centré curseur), +/−, **Ajuster** (`fitContent`), **bougie la plus récente** (`toLatest`).
- **Limite d'historique** : pilule discrète haut-centre « début des données disponibles pour cette unité » (+ état *chargement* et *erreur avec réessai*).
- **Hors fenêtre d'analyse** : pilule bas-centre « bougies affichées mais aucune structure détectée au-delà de N » (`analysisWindowBars` passé par `ReadingColumn`/`DesktopReading`).

### Étiquettes — ancrage + anti-collision + badge
- `lib/chart/zoneOverlayPrimitive.ts` : `placeLabels(items, reserved, plotH, cap=14)` **pur** (dé-collision : décalage vers le bas jusqu'au dégagement, sinon regroupe en débordement ; jamais au-dessus de l'ancre ; jamais hors plot). Seed avec le **rect réservé du badge** + front + liquidité. Débordement → **une** pilule « N zones ». Badge `badgeReserve {w:150,h:22}` réservé quand marché fermé ou live → **ne recouvre plus** les étiquettes.

### Détection — INCHANGÉE
Aucune règle touchée. Le zoom est purement affichage ; les zones hors fenêtre ne sont simplement pas dessinées. `analysisWindowBars` n'est **que lu** pour l'indicateur.

### Performance
- Dé-collision `placeLabels` : O(n²) mais **n plafonné à 14** → coût négligeable par frame. Dessin de l'overlay inchangé sinon (primitive canvas PERF-2 conservé, **aucun retour aux div HTML**).
- Drapeaux plancher/hors-fenêtre : `setState` **uniquement au changement** → pas de re-render par frame de molette.
- Pagination : ajoute des bougies à la série (rendu natif lightweight-charts), bornée par ce qui est chargé ; **jamais** de lecture du CSV 7 ans, **jamais** tout l'historique d'un coup.

### Tests
- Backend : `test_candles_cache_store.py` (+5 `get_candles_before`), `test_candles_endpoint.py` (+3 `before`/`has_more`) — **40 pass**.
- Front unit : `useCandles.test.ts` (+3 pagination : préfixe/plancher, no-op au plancher, échec garde les bougies), `placeLabels.test.ts` (5 : pas de chevauchement, dégagé du badge, regroupement au cap, pas hors plot, jamais au-dessus de l'ancre) — **17 pass**.
- `tsc --noEmit` : **0 erreur**. Build : vert.
- Playwright `chart1-zoom.spec.ts` (1280×800 + 390×844) : **8 pass** — rendu du graphique + jeu complet de contrôles présents (dont « bougie la plus récente », qui n'apparaît que lorsque la pagination d'historique est câblée) + contrôles opérables sans jamais vider le graphique.

### Limite E2E connue (à valider en LIVE)
La **cascade interactive** zoom-out → prefetch → `loadOlder` → pilules *plancher*/*hors-fenêtre* n'est **pas** pilotable de façon fiable par les événements synthétiques Playwright (molette/glisser n'atteignent pas le canvas lightweight-charts ; `getVisibleLogicalRange()` renvoie `null` dans le harnais). L'abonnement se déclenche correctement sur `setVisibleLogicalRange` (prouvé), et **toute la logique** (fusion/plancher/préservation + dé-collision + bornes) est couverte par les tests unitaires. → **La molette/glisser réels et le déclenchement des pilules doivent être confirmés en live** avant merge.
