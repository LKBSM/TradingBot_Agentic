# AUDIT — Suppression de la barre d'outils de zoom/graphique

**Branche** : `fix/remove-chart-toolbar` (worktree dédié `C:\MyPythonProjects\wt-remove-chart-toolbar`, depuis `origin/main` = `769d57e`, PR #171)
**Date** : 2026-08-18
**Cible** : overlay de contrôle superposé au graphique (`+`, `−`, `»`, agrandissement, goutte) — **suppression intégrale**, pas de stylisation.

---

## 0bis — Vérification de dépendance : RÉSOLU (pas de blocage)

La barre **n'est PAS un widget tiers fermé** (aucun widget TradingView). C'est un overlay **100 % maison** rendu en JSX dans notre composant `ReadingChart` (librairie de fond = `lightweight-charts`, dont les propres contrôles ne sont pas utilisés ici). → Suppression propre **par retrait de code**, **aucune occultation CSS** nécessaire.

---

## 1 — Diagnostic

**Fichier unique concerné** : `webapp/components/app/ReadingChart.tsx`

| Élément retiré | Détail |
|---|---|
| Conteneur de la barre | `<div role="group" aria-label="Contrôles du graphique">`, positionné `absolute bottom-2 left-2 z-10` |
| Bouton `+` | zoom avant (`zoom(0.7)`) |
| Bouton `−` | zoom arrière (`zoom(1.4)`) |
| Bouton `»` | bougie la plus récente (`scrollToRecent`) |
| Bouton `⤢` | vue par défaut (`resetDefaultView`) |
| Bouton `💧` (goutte) | **filtre d'affichage de la liquidité** — voir note ci-dessous |
| Sous-composant `ChartControl` | fabrique des boutons, devenu inutilisé |

**Positionnement `absolute`** → le retrait **ne laisse aucun espace vide ni décalage de layout** (le canvas ne réservait pas de place pour la barre).

**Réutilisation du composant** (risque de régression vérifié) :
- `/app` (via `ReadingColumn` + `DesktopReading`) ✅ cible
- **Page d'accueil / vitrine** (via `LandingReadingChart`, données fictives) → barre retirée là aussi (décision : *partout*, composant unique)
- **PAS** sur `/zones` ni `/scanner` ✅ (aucune régression possible sur ces pages)

### Point important tranché avec le fondateur
La « goutte » n'était **pas un sélecteur de thème** mais un **filtre d'affichage de la liquidité** (« poches intactes seulement » ↔ « tout afficher »), donc une commande de *lecture de marché*. Décisions retenues :
1. **Goutte** → *retirée aussi* : le filtre repasse au défaut **« tout afficher »** (`intactOnly: false` figé). **La détection de liquidité n'est pas touchée** — seul le filtre d'affichage revient à son défaut documenté.
2. **Raccourcis clavier** (`+ − ← → Début 0` sur `role="application"`) → **conservés** : le zoom/pan reste pilotable au clavier ; seule la barre visible disparaît.
3. **Vitrine** → barre retirée partout (composant unique, zéro prop conditionnelle).

---

## 2 — Implémentation

Modifications dans `webapp/components/app/ReadingChart.tsx` :
- Retrait du conteneur de la barre (5 boutons) et du sous-composant `ChartControl`.
- Retrait du code devenu inutilisé : `toggleLiquidityIntactOnly`, `setLiquidityIntactOnly`, `hasLiquidityPools`, la constante `LIQUIDITY_INTACT_ONLY_KEY` et son état persistant.
- `buildLiquidityLines(structure, { intactOnly: false })` : filtre liquidité au défaut « tout afficher ».
- Import `lucide-react` réduit à `{ Loader2, RotateCw }` (icônes encore utilisées ailleurs) ; retrait de `Plus, Minus, ChevronsRight, Maximize2, Droplets`.
- **Intacts** : canvas, chandeliers, couches SMC (OB/FVG/liquidité/BOS-CHOCH peintes par la primitive de série), croisillon, avis de bord honnêtes (« début des données » / « hors fenêtre d'analyse »), reprise d'historique, **raccourcis clavier**, panneau Régime.

Les clés i18n désormais inutilisées (`chart.zoomIn/zoomOut/recent/defaultView/controlsAria/liqShowAll/liqShowIntactOnly`) sont **laissées en place dans les 10 locales** : inoffensives et retirer les 10 fichiers risquerait la parité pour aucun gain fonctionnel.

Diff : **2 fichiers, +17 / −157**.

---

## 3 — Discipline & vérifications

- **tsc** (`tsc --noEmit`) : **vert** (0 erreur).
- **Build** (`next build`, `CI=1`) : **vert** (exit 0, table de routes générée ; `/app` 38.1 kB). *Nota : warning non fatal `output: standalone` sur le symlink `.next/standalone` — n'affecte pas le build.*
- **Playwright** (`chart2-amplitude.spec.ts`, `next start` prod, port dédié) : **12 passed / 12 skipped** (les 12 skips = combos pointeur/projet non concordants, par design), aux viewports **1280×800, 1920×1080, 390×844** (desktop + iPhone-12). Le spec a été **mis à jour pour vérifier l'ABSENCE** de la barre (group + 5 boutons `toHaveCount(0)`) tout en conservant les tests clavier/historique (qui prouvent que le clavier fonctionne toujours).

### Test d'honnêteté — captures avant/après
`docs/audits/chart-toolbar-shots/{before,after}/chart-{1280x800,390x844}.png`
- **AVANT** : les 5 boutons (`+ − » ⤢ 💧`) en bas-à-gauche.
- **APRÈS** : aucun bouton. Le reste est **identique** : badge « Marché fermé » (haut-gauche), libellé « Heure locale · UTC−4 » (bas-gauche), zone du canvas inchangée.
- *(Captures prises en état marché fermé → canvas sans bougies dessinées ; sans incidence sur ce qui change : la barre.)*

---

## Reste à faire
- **Confirmation visuelle live du fondateur** sur `/app` (et vitrine) — **seule condition de merge sur `main`**.
- Optionnel ultérieur : purge des clés i18n mortes `chart.*` sur les 10 locales.
