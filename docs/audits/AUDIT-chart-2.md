# AUDIT — CHART-2 · Amplitude du graphique et contrôles de zoom

Périmètre : le graphique de `/app`. **Zéro modification des règles de détection** — le
zoom, le chargement d'historique et les étiquettes sont de l'affichage ; zones et
événements restent exactement ceux détectés sur la fenêtre d'analyse.

Branche : `feat/chart-2-amplitude` (worktree dédié, depuis `origin/main`).

---

## 1. La cause exacte de la limite de dézoome

Deux bornes **côté front**, dans cet ordre — ni l'une ni l'autre n'était le serveur.

**Borne primaire — `minBarSpacing: 4` (le mur du geste).**
`webapp/components/app/ReadingChart.tsx`, options `timeScale`. Lightweight-charts
interdit de comprimer les bougies sous ce plancher. Nombre de bougies affichables à
dézoome maximal = `largeur_plot / minBarSpacing`. Avec `4` sur un plot ~1000 px →
**~250 bougies max** ≈ **~2 jours de séance en M15** : exactement le symptôme. Le geste
« refuse d'aller au-delà » parce que l'espacement est déjà au plancher.

**Borne secondaire — bloc figé de 400 bougies.**
`webapp/lib/market-reading/hooks.ts` → `CHART_CANDLE_LIMIT = 400`. Le front demandait
un unique bloc des 400 dernières bougies par combo, et lightweight-charts ne défile
jamais hors de l'étendue chargée.

**Ce n'était PAS le serveur.** `GET /api/candles` plafonne à `MAX_LIMIT = 1000`, et la
base contient bien plus que 400 bougies (voir §2) : à 400, ni le serveur ni la base
n'étaient la contrainte.

---

## 2. L'historique réellement disponible (base `data/candles.db`)

| Combo | Bougies en base | Couverture | Cible config (`lookback_depths.json`) |
|---|---|---|---|
| XAU/EUR **M15** | 2 233 | ~23 jours | 1mo |
| XAU/EUR **H1** | 933 | ~39 jours | 6mo (borné par l'historique fournisseur) |
| XAU/EUR **H4** | 806 | ~3,3 mois | 2y (borné fournisseur) |
| **D1 / W1** | à la demande | — | 5y |

Le stockage profond LB-1 est rempli (borné par la profondeur d'historique Twelve Data,
pas par nous). **Coût de charger plus** : OHLC seul, gzippé, ~20 KB pour ~2 000 bougies,
**aucun appel fournisseur** (lecture SQLite indexée, ~qq ms). Le chargement se fait
malgré tout **par intervalle à la demande** (voir §4), par principe de performance.

---

## 3. Bornes retenues

- **`minBarSpacing` responsive** (`minBarSpacingFor`) au lieu du `4` fixe :
  `≥1280 px → 0.5` (dézoome plein, centaines/milliers de bougies) · `≥768 px → 1` ·
  `<768 px → 2` (téléphone : pas de bougies-cheveux). Ré-appliqué au resize via un
  `ResizeObserver`. **Aucune borne de zoom arbitraire ajoutée** : le dézoome va
  jusqu'à l'étendue réellement chargée, elle-même bornée par les données disponibles.
- **Chargement initial** : `CHART_CANDLE_LIMIT = 400` (inchangé) — ouverture rapide sur
  l'action récente ; l'historique suit à la demande.
- **Page d'historique** : `OLDER_PAGE_LIMIT = 500` par appel (bien < 1000).

---

## 4. Stratégie de chargement par intervalle

**Backend** (additif, lecture seule) :
- `CandlesCacheStore.get_candles_before(instrument, timeframe, before_ts, n)` — les `n`
  bougies **strictement plus anciennes** qu'un horodatage, ascendantes. Comparaison en
  ISO-8601 UTC (le format de stockage), donc chronologique.
- `GET /api/candles?…&before=<epoch_s>` — renvoie la page antérieure. Nouveau champ
  **`has_more_history`** : vrai s'il reste des bougies plus anciennes que la plus
  ancienne renvoyée. Une page arrière vide est un **200** honnête (« début de
  l'historique atteint »), jamais un 404 « le flux a cassé ».

**Front** (`useCandles`) :
- `loadOlder()` récupère la page précédente (`before` = temps de la plus ancienne
  bougie chargée) et la **préfixe** à la série ; `isLoadingOlder`, `olderError`,
  `reachedStart` exposés. Les bougies déjà affichées **ne disparaissent jamais** ; sur
  échec, `olderError` porte le message + un bouton « Réessayer ».
- Le rafraîchissement à la clôture d'une bougie **fusionne** la fenêtre récente sans
  perdre l'historique paginé (`mergeRecent`).

**ReadingChart** :
- Un abonnement `subscribeVisibleLogicalRangeChange` déclenche `loadOlder()` quand le
  bord gauche s'approche de la plus ancienne bougie chargée (`from < 12`).
- À la prépose d'une page, la vue est restaurée **par temps** (`setVisibleRange`) et non
  par indices logiques (qui glissent), donc l'écran ne saute pas.
- **État de chargement** visible côté gauche (chip « Chargement de l'historique… »).

**Messages honnêtes** :
- `reachedStart` + bord gauche visible → « **Début des données disponibles pour cette
  unité de temps** » (jamais un mur invisible ni un vide).
- Fenêtre visible plus ancienne que la fenêtre d'analyse (`analysis_window_bars` du
  header, déjà émis) → « **Hors de la fenêtre d'analyse — aucune structure détectée
  ici** » : une portion sans annotation n'est jamais présentée comme « sans structure ».

---

## 5. Contrôles — Variante 1 retenue

**Deux variantes proposées au diagnostic** : (1) barre flottante révélée au survol /
focus, visible en permanence sur pointeur grossier ; (2) poignée persistante qui se
déploie. **Retenue : Variante 1** — elle répond exactement à la demande (« apparaissent
au survol, restent sur mobile »), reste la plus simple à rendre accessible (révélation
native via `focus-within`), et n'ajoute aucun élément permanent sur grand écran.

Mise en œuvre :
- Barre `opacity-0` au repos → `group-hover:opacity-100` `group-focus-within:opacity-100`
  ; **`@media(hover:none) → opacity-100`** (pointeur grossier / mobile : toujours
  visible, le pincement ne suffit pas à tout le monde). Les boutons restent dans le DOM
  (opacité, pas `display`) donc **tabbables** ; `focus-within` les révèle au clavier.
- Boutons : **Zoom avant / Zoom arrière / Bougie récente / Vue par défaut** (+ le toggle
  « poches intactes » conditionnel). L'ancien `Ajuster` (fitContent global) est remplacé
  par **Vue par défaut** (fenêtre récente ancrée à droite) ; **Bougie récente** ramène au
  dernier chandelier.
- **Accessibilité (non négociable)** : le plot entier est une région focalisable
  (`role="application"`, `tabIndex=0`) avec raccourcis **`+/−` zoom, `←/→` déplacement,
  `Début` bougie récente, `0` vue par défaut**. Les raccourcis **s'ajoutent** aux
  boutons, ne les remplacent pas.

---

## 6. Étiquettes — méthode de gestion des collisions

Module pur et testable `webapp/lib/chart/zoneLabelLayout.ts` (`layoutZoneLabels`),
appelé par le primitive de canevas à chaque frame (coordonnées pixel déjà calculées) :

1. **Regroupement spatial** — les étiquettes dont l'ancre tombe dans la même cellule
   pixel (`clusterX×clusterY`) se fondent en **une** pastille « **N zones** » ; le détail
   (par famille/état) reste au **survol** (tooltip + hit-test enregistrés). C'est la
   réduction de densité au dézoome : plus de zones au même endroit → plus de
   regroupement automatique.
2. **Dé-collision verticale** — les étiquettes uniques restantes sont décalées
   verticalement jusqu'à ne plus se recouvrir (garantie : **aucune étiquette n'en
   recouvre une autre**).
3. **Plafond de densité** — au plus `maxLabels` (22) étiquettes/pastilles par frame ; le
   surplus de plus faible priorité est laissé sans étiquette (la boîte reste dessinée),
   plutôt qu'un mur de texte.
4. **Badge d'état** — le rectangle du badge « Marché fermé / EN DIRECT » est passé en
   zone **réservée** (`reserved`), donc aucune étiquette n'est placée dessous.

Priorité de placement : zones actives > touchées ; à rang égal, la plus récente (plus à
droite) garde sa place. Les étiquettes de familles masquées ne laissent aucun résidu
(recalculées à partir des seules boîtes visibles).

---

## 7. Mesures de rendu (avant / après)

Coût **ajouté** par frame = la passe de placement d'étiquettes (le reste du rendu — les
chandeliers — est lightweight-charts, inchangé ; le `minBarSpacing` responsive ne change
pas le coût par frame). Micro-benchmark de la fonction pure `layoutZoneLabels`
(2 000 itérations après échauffement, environnement de test) :

| Niveau de zoom | Zones visibles | Coût / frame |
|---|---|---|
| Resserré (zoom in) | 20 | ~0,6 ms |
| Intermédiaire | 80 | ~0,7 ms |
| Dézoome maximal (pire cas) | 400 | ~0,72 ms |

- **Avant** : les étiquettes étaient dessinées directement, sans passe de placement
  (aussi O(n) en dessin, mais sans dé-collision) → coût de placement nul, illisibilité
  garantie en densité.
- **Après** : passe bornée `O(min(zones, 22))` avec `MAX_NUDGES = 6` — **< 1 ms/frame**
  même à 400 zones, très en deçà du budget de 16 ms, et **uniquement sur redraw**
  (pan/zoom), jamais en boucle idle. Mesure sur la fonction pure en environnement de
  test (jsdom) — pessimiste vs le navigateur réel.

Le chargement d'historique est **hors chemin critique** (fetch réseau asynchrone, page
SQLite ~qq ms, ~20 KB gzip, aucun appel fournisseur) et ne bloque pas le rendu : les
bougies affichées restent à l'écran pendant le chargement.

---

## 8. Tests

- **Backend** : `tests/test_candles_cache_store.py` (`get_candles_before` : ordre,
  vide, limite ≤0, cutoff naïf=UTC) + `tests/test_candles_endpoint.py`
  (`before`, `has_more_history` vrai/faux, page vide = 200 non 404). **40 passés.**
- **Front unitaire** : `zoneLabelLayout.test.ts` (non-recouvrement à 3 densités,
  regroupement « N zones », rect réservé du badge, plafond de densité, hors-plot) ;
  `useCandles.test.ts` (paging : préfixe sans perdre les bougies, `reachedStart`,
  échec de page → `olderError`). **Suite vitest complète : 902 passés.**
- **Playwright** (1280×800, 1920×1080, 390×844 — **15 passés**) : vue par défaut +
  toolbar ; contrôles au survol (desktop) / visibles (mobile) + atteignables au
  clavier ; dézoome+déplacement clavier → chargement d'historique à la demande (une
  requête `before=`), bougies conservées, « début des données » + « hors fenêtre
  d'analyse » ; page d'historique en échec → « Historique indisponible » + « Réessayer ».
- **tsc** : 0 erreur. **`next build`** : vert.

---

## 9. Discipline

- Zéro modification de la détection (zoom/historique/étiquettes = affichage).
- Zéro régression de performance (§7) ; suite complète verte.
- i18n **fr natif + 8 locales (repli EN)** pour tous les nouveaux messages
  (`app.chart.*` : recent, defaultView, historyStart, outsideAnalysis, loadingHistory,
  historyError, retry, zonesCluster, controlsAria, keyboardRegionAria).
- Staging explicite (jamais `git add -A`), pas de force push.
- **Merge sur `main` seulement après confirmation live.**
