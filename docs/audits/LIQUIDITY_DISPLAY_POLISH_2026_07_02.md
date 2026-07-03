# Affichage liquidité externe — lignes bornées + figées au contact (2026-07-02)

Branche : `feat/liquidity-display-polish` (worktree dédié, base main `793c9a0`).
Périmètre : **affichage uniquement**. Détection et états liquidité inchangés — on
LIT `status` / `created_at` / `swept_at` / `broken_at` émis par le moteur, on ne
recalcule rien, on n'invente aucune poche.

## Problème

Les poches de liquidité (BSL/SSL) étaient dessinées via `createPriceLine` de
lightweight-charts : une price line traverse **tout** le canvas par construction
(aucune option de bornage). Résultat : des traits pleine largeur visuellement
lourds, sans notion de « depuis quand » ni de « jusqu'où ». Les poches `broken`
étaient par ailleurs exclues du graphe.

## Rendu livré (cible validée sur maquette)

| État | Ligne | Extrémité droite | Marqueur | Libellé |
|---|---|---|---|---|
| intacte | pleine, opacité 0.9 | bougie courante + pad (comme un OB actif) | pastille de prix sur l'échelle (trait masqué, `lineVisible:false`) | « Liquidité achat/vente · intacte » au bord gauche |
| prise (swept) | pointillée (dashed), 0.55 | **figée au premier contact** (`swept_at`) | point au contact | « prise » |
| cassée (broken) | pointillé serré (dotted), 0.40 | **figée au premier contact** (min `swept_at`/`broken_at`) | petite croix × | « cassée » |

- Une poche **prise reste affichée** (touché ≠ cassé) — jamais supprimée.
- Une poche cassée qui avait été prise avant fige au **premier** contact
  (cas réel observé : H4 `LIQ_bsl_equal_highs_20260626210000`, swept
  2026-07-01T13:00 puis broken 2026-07-02T09:00 → figée au 07-01T13:00).
- Libellés factuels seulement — aucun « cible », « objectif » ni direction.
- Libellé produit : « balayée » → « **prise** » (formatters + glossaire),
  décision utilisateur au GO. Palette conservée (BSL bleu-teal, SSL rose-violet).

## Mécanique

- `webapp/lib/chart/liquidityLines.ts` — builder pur : chaque poche devient un
  segment `{createdSec, contactSec|null, status, labels}`. `poolContactSec()` =
  min des timestamps de contact émis par le moteur (null si intacte ; défensif :
  null aussi si swept/broken sans timestamp parseable → le segment s'étend à la
  bougie courante dans le style de son état, rien d'inventé).
- `webapp/components/app/ReadingChart.tsx` — les segments sont rendus dans
  l'**overlay HTML existant des boxes OB/FVG** (même boucle rAF, mêmes helpers
  `xAt`/`activeRightX`), plus jamais en price line pleine largeur. Seule la
  pastille d'axe des intactes reste une price line, **trait masqué**.
- Toggle « poches intactes seulement » : bouton gouttelette à côté des contrôles
  zoom, `aria-pressed`, persisté `localStorage` (`mia.chart.liquidityIntactOnly`),
  défaut = tout visible. Filtre d'affichage pur, réversible, aucune donnée
  supprimée (le panneau Structure liste toujours tous les états).
- Le calque `liquidity` (chat) continue de masquer/afficher l'ensemble.

## Fix connexe découvert pendant la vérification

Les canvases de lightweight-charts portent leur propre `z-index` (1-2) : le
canvas de l'axe temporel **interceptait les clics** sur les contrôles bas-gauche
(zoom compris, préexistant). Ajout de `z-10` sur le conteneur de contrôles.

## Réconciliation main

main a avancé pendant la session (PR #21 `b803e04` filtre minTime des markers,
PR #22 `903e313` géométrie pure temps+prix + clipping par gouttière) →
**mergé, pas écrasé** : les segments liquidité adoptent la même géométrie non
clampée et se clippent au bord du plot via le conteneur.

## Vérification

- `tsc --noEmit` : 0 erreur. `next build` : OK.
- Vitest : **43 fichiers / 391 tests verts** post-merge (9 nouveaux sur le
  builder : ancres, premier contact, broken par défaut, intactOnly réversible).
- Visuel (backend live XAUUSD M15, données réelles avec les 3 états) :
  - avant : `liquidity_polish_avant.png` (traits pleine largeur) ;
  - après : `liquidity_polish_apres.png` (segments bornés, prise/cassée figées) ;
  - toggle : `liquidity_polish_apres_intactes_seules.png` (prises/cassées
    masquées, bouton pressé, réactivation sans perte).

## Tests de la mission

(a) ligne bornée à droite ✅ (b) figée au contact avec l'état réel ✅ (c) poche
prise toujours affichée, style dédié ✅ (d) toggle réversible sans perte ✅
(e) aucune poche inventée, détection non mutée (builder read-only) ✅
