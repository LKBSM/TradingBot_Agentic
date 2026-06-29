# Audit — Détection de la liquidité externe (BSL / SSL)

**Branche :** `feat/liquidity-detection` (worktree dédié, depuis `main` consolidé)
**Date :** 2026-06-29
**Périmètre :** extension *descriptive* du moteur SMC — détecter **où** se trouvent les poches de
liquidité et **si** elles sont intactes ou prises. Aucune notion de cible, de draw, de biais ou de
prédiction de retournement.

---

## 1. Ligne éditoriale (inviolable)

Le moteur émet des **faits présents/passés observables**, jamais une intention de prix :

- ✅ « Equal highs à 2412.3, **intacts** » — fait présent.
- ✅ « Equal lows à 1.0832, **pris** (mèche sous + clôture revenue au-dessus) » — fait passé.
- ❌ « le prix va chercher la liquidité » / « cible » / « objectif » / « draw on liquidity » /
  « biais » / « setup » — **INTERDIT**, et aucun champ du schéma ne le porte (cf. §4 et le test
  `test_no_predictive_fields`).

Un *sweep* est un événement **déjà arrivé**, pas une promesse de réaction.

---

## 2. Définitions

### 2.1 Poches d'égalité — EQH / EQL
Une poche **equal_highs** (BSL) / **equal_lows** (SSL) est un *cluster* de points de swing dont les
prix tiennent dans une tolérance d'égalité :

```
eps = max(EQ_TOLERANCE_ATR × ATR, EQ_TOLERANCE_PIPS_FLOOR)
```

Le clustering est glouton sur les fractales causales (`UP_FRACTAL` / `DOWN_FRACTAL`, déjà décalées de
N barres — **aucune réécriture** de la détection de swing existante). Une poche n'est retenue que si
elle agrège **≥ `EQ_MIN_TOUCHES`** swings. Le **niveau** de la poche est l'extrême du cluster (plus
haut des highs pour BSL, plus bas des lows pour SSL) — décision produit validée.

### 2.2 Poches de range — range_high / range_low
L'extrême de la fenêtre `LIQ_LOOKBACK` constitue une poche externe isolée **uniquement** s'il n'est
pas déjà porté par un cluster d'égalité (dé-duplication : pas de double-comptage de l'extrême).

### 2.3 Externe vs interne
```
BSL externe  ⇔  level ≥ range_high − eps
SSL externe  ⇔  level ≤ range_low  + eps
```
Tout le reste est interne. `is_external` est un fait géométrique, pas un score.

### 2.4 Cycle de vie — intact / swept / broken
Évalué barre par barre à partir de la confirmation de la poche (`scan_from` = **dernier** swing du
cluster — la poche n'est entièrement connue qu'à ce moment, donc **aucun look-ahead**) :

| Côté | broken (close traverse) | swept (mèche dépasse, close revient) |
|------|--------------------------|--------------------------------------|
| BSL  | `close > level`          | `high > level` **et** `close ≤ level` |
| SSL  | `close < level`          | `low < level` **et** `close ≥ level`  |

La boucle s'arrête au premier `broken`. Un simple *approche* sans dépassement de mèche reste
`intact` (test `test_no_sweep_on_approach`).

---

## 3. Paramètres (`SMCConfig`)

Ajoutés à `src/environment/strategy_features.py`, threadés jusqu'au collecteur via l'assembler — donc
**réglables** sans toucher au moteur :

| Param | Défaut | Rôle |
|-------|--------|------|
| `EQ_TOLERANCE_ATR` | `0.10` | tolérance d'égalité en multiple d'ATR |
| `EQ_TOLERANCE_PIPS_FLOOR` | `0.0` | plancher absolu de tolérance (`max(ATR×mult, floor)`) |
| `EQ_MIN_TOUCHES` | `2` | nombre minimal de swings dans la tolérance |
| `LIQ_LOOKBACK` | `200` | fenêtre (barres) de recherche de poches |

---

## 4. Schéma (`MarketReadingStructure.liquidity_pools : list[LiquidityPool]`)

```python
LiquiditySide  = Literal["bsl", "ssl"]
LiquidityKind  = Literal["equal_highs", "equal_lows", "range_high", "range_low"]
LiquidityStatus = Literal["intact", "swept", "broken"]

class LiquidityPool(BaseModel):
    id: str               # LIQ_<side>_<kind>_<YYYYMMDDHHMMSS>  (stable, ancrage display + agent)
    side: LiquiditySide
    kind: LiquidityKind
    level: float          # extrême du cluster
    touches: int
    is_external: bool
    status: LiquidityStatus
    created_at: datetime  # premier swing connu (plus tôt observable)
    swept_at: datetime | None
    broken_at: datetime | None
    user_flagged: bool = False
```

**Aucun champ prédictif** : pas de `target`, `bias`, `probability`, `draw`, `objective`, `expected`,
`direction`. Le test `test_no_predictive_fields` vérifie que l'ensemble des clés émises est
*exactement* le jeu descriptif ci-dessus et **disjoint** du jeu interdit.

---

## 5. Impact mesuré — 6 combos

Mesure reproductible (`scripts/audit/liquidity_impact.py`), 60 points de lecture régulièrement
espacés par combo, `LIQ_LOOKBACK=200`, cap `MAX_LIQUIDITY_POOLS=8`. Données :
`docs/audits/liquidity_impact_data.json`.

| Combo | Barres | Poches/lecture (moy.) | % externe | % intacte | % prise (swept) | % cassée (broken) |
|-------|-------:|----------------------:|----------:|----------:|----------------:|------------------:|
| XAUUSD_M15 | 172 849 | 7.95 | 25.2 | 53.9 | 7.3 | 38.8 |
| XAUUSD_H1  | 43 218  | 7.68 | 26.0 | 54.4 | 8.9 | 36.7 |
| XAUUSD_H4  | 11 363  | 7.88 | 25.4 | 55.6 | 5.3 | 39.1 |
| EURUSD_M15 | 174 481 | 7.97 | 25.1 | 53.1 | 5.2 | 41.6 |
| EURUSD_H1  | 43 606  | 7.90 | 25.3 | 51.9 | 9.1 | 39.0 |
| EURUSD_H4  | 11 253  | 7.92 | 25.3 | 54.7 | 9.1 | 36.2 |

Répartition par nature (somme sur les 60 lectures, ordre de grandeur stable inter-combos) :
les clusters d'égalité (`equal_highs` + `equal_lows`) dominent (~160–206 chacun), les extrêmes de
range (`range_high` + `range_low`) sont minoritaires (~49–59 chacun) — cohérent avec une fenêtre de
200 barres qui contient typiquement plusieurs paires d'égalité pour un seul couple d'extrêmes.

### Lecture
- **Volume gérable et stable** : ~7.7–8.0 poches par lecture, le cap à 8 mord (priorité
  externe → statut → récence), ce qui borne la charge d'affichage sans amputer l'information utile.
- **~1/4 des poches sont externes**, ~3/4 internes — la frontière externe/interne est franche et
  cohérente entre instruments et timeframes (25.1–26.0 % externe partout).
- **Les sweeps purs sont rares (5–9 %)** ; la majorité des poches franchies le sont par *close-through*
  (`broken`, 36–42 %). C'est la signature attendue : la plupart des niveaux finissent traversés en
  clôture, et l'état `swept` strict (mèche + retour) reste l'exception — ce qui valide d'avoir séparé
  les deux états plutôt que de tout étiqueter « pris ».
- **Plus de la moitié des poches restent intactes** à l'instant de lecture (51.9–55.6 %).

Aucune métrique de performance/PnL n'est produite ni implicite : la couche est strictement
descriptive.

---

## 6. Discipline & non-régression

- BOS / CHOCH / OB / FVG : **inchangés** (collecteur ajouté côté *mappers*, pas dans le moteur).
- Détection de swing : **réutilisée**, non dupliquée.
- Tests : `tests/test_liquidity_pools.py` — **19/19 verts** (clustering, tolérance frontière,
  min_touches, externe/interne, extrême de range isolé, cycle intact/swept/broken, pas de sweep sur
  approche, pas de look-ahead avant connaissance de la poche, id stable, absence de champ prédictif,
  garde colonnes manquantes / vide, cap, run moteur réel bout-en-bout).
- Régression large : **271 passed, 0 failed**.

---

## 7. Fichiers touchés

- `src/environment/strategy_features.py` — 4 champs `SMCConfig`.
- `src/intelligence/market_reading_schema.py` — `LiquiditySide/Kind/Status`, `LiquidityPool`,
  `MarketReadingStructure.liquidity_pools`.
- `src/intelligence/market_reading_mappers.py` — `collect_liquidity_pools`, `_pool_lifecycle`,
  `_cluster_swings`, `_liquidity_to_models`, câblage dans les deux retours du mapper structure.
- `src/intelligence/market_reading_assembler.py` — injection `_liquidity` (params via `engine.config`).
- `tests/test_liquidity_pools.py` — 19 tests.
- `scripts/audit/liquidity_impact.py` — mesure reproductible 6 combos.
- `docs/audits/liquidity_impact_data.json` — données de §5.
- `docs/audits/LIQUIDITY_DETECTION_2026_06.md` — ce rapport.
