# AUDIT RG-1c — Niveaux de référence : valeurs fausses + tracé manquant

Branche : `feat/rg-1c-niveaux-correctif` (worktree dédié, depuis `main` = `788a03c`).
Cible : panneau « Régime de marché » → tuile « Niveaux de référence » + tracé chart.

---

## 1. Diagnostic (avant correctif)

### DÉFAUT 1 — valeurs fausses : cause racine

Chaîne du calcul (avant) : `webapp/lib/market-reading/reference-levels.ts::referenceLevels(daily, weekly)`
```
prevDayHigh/Low = candles[length-2].high / .low     // UNE bougie
dayOpen         = candles[length-1].open            // UNE bougie
```

- **Aucune agrégation.** Le code **indexait positionnellement** la série D1/W1 servie par
  `/api/candles` : `[-1]` = « jour/semaine courant », `[-2]` = « veille/semaine précédente ».
  Il lisait **le haut et le bas d'UNE seule bougie**, jamais le max/min sur toutes les bougies
  de la période. La suspicion de la brief était exacte.
- **Définition du « jour » déléguée à Twelve Data.** La frontière provenait de la bougie
  `1day`/`1week` de TD (`_TIMEFRAME_MAP`), dont l'ancrage est **opaque** (ni 17:00 NY, ni une
  frontière MC-1). Pas de définition unique par instrument dans le produit.
- **Décalage d'un cran.** `warm_candles` applique `drop_unclosed_candles`, donc la dernière
  bougie servie est la dernière **close** ; `[-1]` était déjà « hier » et `[-2]` « avant-hier ».
- **Symptôme prouvé (2,77 $).** Dès que la série servie sous « D1 » n'est pas une vraie
  journalière pleine (populate à la demande partiel, bougie fine), `high−low` s'effondre au
  range d'**une seule bougie**. Vérification indirecte sur le cache local (`data/candles.db`) :
  l'or cote ~4000-4200 et **une seule bougie H4 fait déjà 17 à 100 $** de range — un « haut de
  la veille » et un « bas de la veille » séparés de **2,77 $** ne peuvent provenir que d'une
  bougie unique, jamais d'une journée d'or.

> Transparence : les octets live exacts n'ont pas pu être rejoués (pas de clé
> `TWELVE_DATA_API_KEY` en local, base locale antérieure aux D1/W1). Le mécanisme est néanmoins
> prouvé par la lecture du code + les ranges réels en cache.

### Fuseau retenu (décision fondateur)

**New York (MC-1), par instrument.** La frontière « jour » = `InstrumentHours.close_time` dans
`InstrumentHours.tz` (America/New_York) :
- **XAUUSD** : rollover **17:00 NY** (bougie de trading 18:00 NY → 17:00 NY, la pause 17-18 n'a
  aucune bougie).
- **EURUSD** : **17:00 NY** → 17:00 NY.
- **Semaine** : du dimanche (ouverture) au vendredi 17:00 NY ; le lundi du bloc Mon-Ven est
  l'étiquette de semaine.

C'est **la même config par instrument que MC-1** (statut de marché) — **une seule définition**,
en ZoneInfo, DST-safe. En juillet (EDT = UTC−4) le rollover 17:00 NY = **21:00 UTC**.

### DÉFAUT 2 — tracé manquant : cause racine

Le clic **mettait déjà à jour l'état** (`setReferenceLevel`) **et** `ReadingChart` **créait une
`createPriceLine`**. Ce qui manquait :
- **La vue n'était jamais amenée au niveau.** `autoscaleInfoProvider` calcule la bande de prix
  depuis **les bougies seules** ; lightweight-charts n'inclut pas les price-lines dans
  l'autoscale → un repère hors bande (typiquement les extrêmes de semaine) était **peint
  hors-écran** = « ça ne dessine rien ». Aucun rescale ne l'y ramenait.

---

## 2. Correctif

### A. Calcul correct (backend, source unique — zéro diff moteur de détection)

- **`src/intelligence/reference_levels.py`** (nouveau) : `compute_reference_levels(instrument,
  candles, now)` agrège **max(high)/min(low) sur TOUTES les bougies** de chaque **journée/semaine
  de trading MC-1** (bucketing par `close_time` NY en ZoneInfo). `dayOpen`/`weekOpen` = open de
  la **première** bougie de la période courante.
- **Garde « données insuffisantes »** : un repère n'est émis **que si sa période est entièrement
  couverte** (la plus vieille bougie disponible commence **avant** le début de la période).
  Sinon `None` + le panneau affiche « Données insuffisantes pour {période} » — jamais une valeur
  partielle présentée comme complète.
- **Source** : bougies **H1 déjà en cache** (lookback 500 ≈ 4 semaines de trading — couvre veille
  ET semaine ; repli H4/M15), **indépendant de la TF affichée**.
- **`market_reading_assembler.reference_levels()`** + attache dans `_with_status` (frais par
  requête, **jamais persisté**, exactement comme `market_status`). Champ
  `MarketReading.reference_levels` (Optional).

### B. Tracé au clic (bring-to-view) + style distinct

- `ReadingChart.autoscaleInfoProvider` **inclut désormais le prix du repère tracé**
  (`referenceLevelPriceRef`) → la vue vient au niveau ; toggle `autoScale` off→on pour forcer le
  recalcul. Ligne **fine solide accent** + **label d'axe explicite** « Haut de la veille · 4 202,03 »
  (nom du repère **et** valeur), distincte des cassures (pointillé gris), de la liquidité et des
  boîtes OB/FVG. Un seul repère à la fois ; rien tracé au chargement ; re-clic = retrait.
- **Canal séparé** `referenceLevel` (jamais `coerceViewAction`) → **verrou d'id moteur ni élargi
  ni contourné**.

### C. Geste généralisé

Tout **niveau de prix** affiché dans les panneaux Donnée est cliquable/traçable, même étiquette
explicite (`PxBtn`) : bornes de structure (tuile **Position**), **niveaux franchis BOS/CHOCH**
(tuiles Maturité & Dernier événement, + journaux). Les **deltas/moyennes** (distances, ATR moyen)
restent en texte : ce ne sont pas des niveaux de prix, les tracer serait trompeur.
La tuile **Tendance** n'expose aucun extrême de prix (mesure = déplacement close vs amplitude,
pas des swings) → rien à tracer, cohérent avec le concept honnête.

### Front (mapping seul)

`referenceLevelsFromPayload()` mappe le payload serveur ; suppression de `useReferenceCandles`
(fetch D1/W1) et de l'indexation `[-2]`. `RegimeCard` reçoit `referenceLevelsPayload`.

---

## 3. Écarts mesurés — six repères (avant → après)

| Repère | Avant (bug) | Après (correctif) |
|---|---|---|
| Ouverture du jour | open d'**une bougie** (hier, décalé) | open de la **1ʳᵉ bougie** de la journée MC-1 courante |
| Ouverture de la semaine | open d'**une bougie** | open de la **1ʳᵉ bougie** de la semaine MC-1 courante |
| Haut de la veille | `candle[-2].high` (ex. **4 056,11**) | **max(high)** sur toutes les bougies de la veille MC-1 |
| Bas de la veille | `candle[-2].low` (ex. **4 053,34** → spread **2,77 $**) | **min(low)** sur toutes les bougies de la veille MC-1 |
| Haut de la semaine | `candle[-2].high` | **max(high)** sur toutes les bougies de la semaine MC-1 |
| Bas de la semaine | `candle[-2].low` | **min(low)** sur toutes les bougies de la semaine MC-1 |

**Écart clé** : le spread veille passe d'un range **d'une bougie** (~2,77 $, prouvé impossible
pour l'or) à l'**amplitude réelle de la journée** (dizaines de dollars). Cohérence garantie et
**testée** : `prev_week_high ≥ prev_day_high` quand la veille est dans la semaine courante (le
test qui aurait attrapé le bug). Vérification visuelle live sur deux journées à confirmer côté
fondateur (les valeurs exactes dépendent du feed TD).

---

## 4. Tests

**Backend** (`tests/test_reference_levels.py`, 7 + wiring `tests/test_market_reading_assembler.py`, 2) :
- haut de la veille = **max des hauts de TOUTES les bougies** (valeur en dur) ;
- haut & bas de veille de **bougies différentes** (sauf journée à 1 bougie) ;
- **haut semaine ≥ haut veille** quand la veille ∈ semaine courante ;
- **période incomplète → `None`** + flag, jamais partiel ;
- frontière **17:00 NY** (EURUSD) et non minuit UTC ;
- série vide → tout `None` ; wiring assembler agrège depuis H1 caché.

**Frontend** (`reference-levels.test.ts`, `rg1-regime.test.tsx`) :
- mapping payload → panneau ; **clic → ligne tracée + étiquette explicite**, re-clic → retrait,
  **un seul** ; **rien tracé au chargement** ; période incomplète → repère absent + « Données
  insuffisantes » ; **id de zone inventé toujours rejeté** (`viewActions.test.ts`, inchangé).

**Résultats** : tsc **0**, build **exit 0**, front **≥600** verts, back (modules touchés + suite)
verts. i18n fr + en complets, 7 autres locales = EN (convention UI-2c), 9×JSON valides.

---

## 5. Discipline

Zéro diff moteur de **détection** (SMC intact) — seul l'agrégat calendaire arithmétique est
ajouté. Staging explicite (pas de `git add -A`), pas de force push. Merge sur main **après
validation live du fondateur** uniquement.
