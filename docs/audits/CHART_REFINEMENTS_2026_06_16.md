# Raffinements graphique — FVG borné, stabilité au scaling, lookback

**Date :** 2026-06-16
**Branche :** `feat/chart-refinements` (basée sur `feat/front-polish-zones-chatbot`)
**Périmètre :** front + une exposition backend read-only/descriptive. **Détection
NON touchée**, aucune zone recalculée, aucune projection. Clair + sombre.

---

## Pourquoi pas depuis `main`

`main` ne contient **aucun graphique** (pas de composant chart, `lightweight-charts`
absent des dépendances). Tout le rendu des zones bornées (`ReadingChart.tsx`,
`collect_zones`, `mitigated_at`, lookback 500) vit sur `feat/front-polish-zones-chatbot`,
non mergé. Sur décision (rebase), la branche `feat/chart-refinements` part donc de ce tip
— conforme au SETUP de la mission (« idéalement après que la feat zones soit landée »).

## Réponses de confirmation (état avant travaux)

| Question | Réponse |
|----------|---------|
| Profondeur de comblement FVG exposée au front ? | **Non.** Schéma `FairValueGap` = `level_high/low`, `status`, `mitigated_at` (timestamp). Aucune profondeur numérique. |
| Effet du scaling de l'axe prix sur les zones ? | Boîtes = overlay HTML positionné via `priceToCoordinate`/`timeToCoordinate`, recalculé seulement sur `subscribeVisibleLogicalRangeChange` (axe **temps**) + `ResizeObserver`. **Aucun abonnement à l'axe prix** → jitter au drag vertical. |
| Lookback affiché ? | `useCandles` ne passait **aucune limite** → défaut client **200** (backend cap 1000, assembler cache 500). |

---

## A — FVG borné en hauteur (portion ouverte uniquement) · commit `6014b36`

Un FVG partiellement comblé **rétrécit** : la boîte ne montre plus que la portion encore
ouverte, s'arrêtant « juste sous les mèches » à la profondeur de pénétration max.

**Exposition read-only ajoutée** (pas de détection touchée) :
- `FairValueGap.fill_level: Optional[float] = None` — la mèche la plus profonde dans la
  bande (clampée à `[level_low, level_high]`), **distincte de `mitigated_at`** (qui est un
  timestamp). Descriptif, jamais prédictif.
- `_fvg_lifecycle` calcule la pénétration max sur les `high/low` déjà émis par l'engine :
  bullish → plus bas atteint (le gap se comble par le haut), bearish → plus haut atteint.
  `collect_zones` + `_zones_to_models` la propagent. **Aucun seuil de détection lu/modifié.**

**Front** : `openFvgBand()` (dans `zoneLayout.ts`) borne la boîte à la portion ouverte —
bullish `high = fill_level`, bearish `low = fill_level` — **uniquement** si la direction est
connue et `fill_level` est strictement dans la bande ; sinon bande pleine (jamais de devinette).

**Tests** : 2 Python (`fill_level=None` actif / pénétration la plus profonde, sans
régression sur un dip ultérieur plus faible) + 5 vitest (`openFvgBand` bullish/bearish/
hors-bande/direction inconnue + passage via `buildZoneModels`).

## B — Stabilité au scaling (suppression du jitter) · commit `44b91c4`

**Diagnostic** : redraw désynchronisé de l'overlay (pas de rescaling accidentel). Le drag
vertical de l'axe prix (`handleScale.axisPressedMouseMove`) n'émet aucun événement →
l'overlay restait figé en Y puis « snappait » au prochain événement temps.

**Correctif** : recompute piloté par une boucle `requestAnimationFrame` → les boîtes se
redessinent **en phase** avec le canvas pour tous les changements d'échelle (prix, temps,
autoscale, resize). Garde géométrique `rectsEqual` (arrondi ½ px) : React ne re-render que
si une boîte bouge réellement → chart au repos = une lecture de coordonnées par frame,
zéro re-render. Remplace l'ancien `subscribeVisibleLogicalRangeChange` + `ResizeObserver`.

## C — Lookback à 400 bougies · commit `88eb894`

`useCandles` passe désormais `CHART_CANDLE_LIMIT = 400` (« quelques centaines », pas
« immense »). Servi depuis le **cache SQLite** (`store.get_last_n_candles`, lecture seule) :
backend cap 1000, assembler cache 500, et les 6 combos ont ≥ 505 bougies en cache →
**aucun appel Twelve Data supplémentaire**. Test `useCandles` renforcé (`limit: 400`).

---

## Discipline & vérification

- **Détection intacte** : seules les couches lifecycle/mapper (lecture des `high/low`
  produits) et le front sont modifiées. `strategy_features.py` / `smart_money` non touchés.
- **Exposition** : un seul champ ajouté, optionnel, read-only, descriptif (`fill_level`).
- **Commits séparés** A / B / C. Pas de `git add -A` (fichiers untracked préexistants
  laissés intacts).
- **Tests** : Python 78 verts (`market_reading` mappers/schema/assembler) ; vitest **160
  verts** (23 fichiers) ; `tsc --noEmit` clean.
- **Build** : `next build` vert (`/[locale]/app` 159 kB First Load JS).
- **Thème** : clair + sombre (le composant restyle uniquement, palette résolue par thème).

### Fichiers touchés
```
src/intelligence/market_reading_schema.py        (+ fill_level)
src/intelligence/market_reading_mappers.py       (_fvg_lifecycle, collect_zones, _zones_to_models)
tests/test_market_reading_mappers.py             (+2 tests)
webapp/types/market-reading.ts                   (+ fill_level)
webapp/lib/chart/zoneLayout.ts                   (openFvgBand)
webapp/lib/chart/__tests__/zoneLayout.test.ts    (+5 tests)
webapp/components/app/ReadingChart.tsx           (rAF + rectsEqual)
webapp/lib/market-reading/hooks.ts               (CHART_CANDLE_LIMIT=400)
webapp/lib/market-reading/__tests__/useCandles.test.ts (assert limit)
```
