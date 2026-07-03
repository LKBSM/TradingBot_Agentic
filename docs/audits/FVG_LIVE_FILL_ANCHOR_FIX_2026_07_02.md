# Fix ancrage de l'encadré « comblement live » du FVG — 2026-07-02

**Branche** : `fix/fvg-fill-anchor` (worktree dédié, depuis `origin/main` réconcilié `b803e04` = PR #21 incluse)
**Périmètre** : affichage uniquement (`webapp/components/app/ReadingChart.tsx`). Détection et `fill_level` : **intouchés** (aucun changement moteur, aucun changement de `zoneLayout.ts`).

## Symptôme

L'encadré ambre « comblement live » d'un FVG est bien placé au repos, mais **se déforme
(s'étire / s'effondre) pendant un pan horizontal** du graphe : un côté suit les bougies,
l'autre reste collé au bord de l'écran.

## Cause

Dans la boucle rAF `computeRects` de `ReadingChart.tsx`, les bornes **horizontales** des
boîtes passaient par un clamp en coordonnées **écran** :

- `xAt(createdSec)` (bord gauche = bougie de formation) : coordonnée temps **puis
  `clampX` → bornée à `[0, plotRight]`** ;
- `activeRightX()` (bord droit = barre courante + 1,5 barre) : idem, plus un fallback
  `plotRight`.

Tant que les deux ancres sont visibles, le clamp est un no-op → « bien placé au repos ».
Dès qu'une ancre sort du viewport (le moindre pan vers l'arrière fait sortir la barre
courante à droite), **cette borne se fige sur le bord écran pendant que l'autre continue de
suivre le graphe** → déformation continue pendant le geste, jusqu'à l'effondrement en
liseré de 2 px (`Math.max(2, …)`) collé contre l'axe des prix.

Les bornes verticales (prix purs via `priceToCoordinate`) étaient saines.

**Point clé vs l'hypothèse initiale** : les rectangles OB/FVG passaient par le **même code
clampé** (l'encadré live hérite verbatim du `left`/`width` de sa boîte FVG parente). Ils ne
« s'étiraient pas » qu'en apparence : sur une grande boîte pâle, le bord épinglé se lit
comme un clipping naturel ; sur le petit encadré ambre vif, la déformation saute aux yeux.
Le fix corrige donc le calcul **partagé** — les boîtes de zone en profitent aussi (le
liseré 2 px fantôme au bord du plot disparaît également pour elles).

## Fix (3 morceaux, tous côté rendu)

1. **Géométrie pure temps+prix** : `xAt` et `activeRightX` renvoient la coordonnée temps
   brute de lightweight-charts (v5 : `timeToCoordinate` résout aussi hors viewport pour un
   point de données), **sans clamp**. Le fallback `plotRight` ne subsiste que si la barre
   courante n'est pas sur l'échelle de temps (cas défensif).
2. **Clipping au conteneur** : l'overlay passe de `inset-0` à `inset-y-0 left-0` +
   `right: priceGutterWidth` (largeur de la gouttière de l'axe des prix, lue chaque frame
   dans la même boucle rAF, commit React seulement au changement). La garantie « ne jamais
   déborder sur la gouttière » — la raison d'être du clamp — est préservée, mais par
   **clipping** (présentation) au lieu de **déformation** (géométrie).
3. **Épinglage du label de type (chrome uniquement)** : le code OB/FVG en haut-gauche d'une
   boîte glisse jusqu'au premier pixel **visible** de sa boîte (`labelLeft = max(0, -left)`)
   — sans cela, le label d'une zone formée hors fenêtre (bord gauche off-plot, cas courant
   au repos) serait devenu invisible. La géométrie de la boîte, elle, n'est jamais ajustée.
   Le label « comblement live » (ancré à droite, barre courante visible au repos) n'a pas
   besoin d'épinglage : il suit le graphe, c'est le comportement voulu.

Position et dimensions logiques de l'encadré : **inchangées** (mêmes ancres temps+prix
qu'avant ; seul le comportement hors-viewport change).

## Vérification

- **tsc** : 0 erreur. **Tests front** : 385/385 verts. **`next build`** : OK.
- **Harnais Playwright dédié** (`webapp/verify_fvg_anchor.mjs`, non commité — artefact de
  session) : Next dev + interception de toutes les routes `/api/*` (fixture
  market-reading réelle XAUUSD M15, bougies synthétiques cohérentes, tick SSE injecté au
  milieu de la bande du FVG actif `FVG_bullish_20260630063000`) ; mesures DOM
  (`boundingBox`) de l'encadré **pendant un vrai drag souris, bouton enfoncé**.

| Mesure (largeur de l'encadré, px) | AVANT (origin/main) | APRÈS (fix) |
|---|---|---|
| Repos | 60.1 | 60.8 |
| Pan gauche 140 px (ancres visibles) | 60.8 constant | 60.8 constant |
| **Pan vers l'historique 320 px** (barre courante sort à droite — le cas du bug) | **60.8 → 52.1 → 2 → 2 → 2** (effondrement, liseré épinglé au bord) | **60.8 constant (6/6 mesures)** |
| Suivi du graphe (Δleft vs Δdrag) | n/a | −112 px vs −112 px (1:1 exact) |
| Zoom avant | bloqué à 2 px (cascade du bug) | 60.8 → 86.6 (∝ barSpacing, attendu) |
| Pan après zoom | — | 86.6 constant (4/4) |

Captures : `webapp/verify_{baseline,fixed}_{rest,midpan_gauche,midpan_historique,zoom}.png`
(baseline mi-pan : liseré ambre + label « comblement live » épinglés contre l'axe des prix ;
fixé : plus rien d'épinglé, l'encadré suit les bougies et se clippe naturellement au bord).

Note : la **hauteur** des boîtes peut varier pendant un pan — c'est l'autoscale du prix de
lightweight-charts (les boîtes restent collées à leurs prix), pas une déformation.

## Fichiers

- `webapp/components/app/ReadingChart.tsx` — seul fichier modifié.
- Ce rapport.

## Reste à faire

- Confirmation du rendu en live par le fondateur → **push + merge sur main** (pas avant).
