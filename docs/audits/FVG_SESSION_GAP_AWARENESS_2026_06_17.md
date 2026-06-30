# FVG — Session-gap awareness (correction « gap de séance = FVG »)

**Date** : 2026-06-17
**Branche** : `fix/fvg-session-gap-awareness` (depuis `institutional-overhaul`)
**Portée** : correction de *justesse* de la détection FVG (suppression de faux positifs).
BOS / CHOCH / Order Block **inchangés** — seul le FVG est concerné.

---

## 1. Problème

La détection de FVG repose uniquement sur la géométrie 3 bougies
(`low[i] > high[i-2]` en haussier / `high[i] < low[i-2]` en baissier), sans
aucune notion de fermeture de marché. Quand la fenêtre de 3 bougies enjambe une
**clôture de séance** (week-end, coupure quotidienne de l'or, jour férié), le
« trou » de prix n'est **pas** un déséquilibre de momentum — c'est juste le
marché qui était fermé — mais il satisfait la même condition géométrique et se
fait étiqueter FVG à tort.

## 2. Approche — data-driven, aucun calendrier codé en dur

La fermeture est détectée via l'**écart de temps** entre les timestamps des
bougies. Sur une TF propre, deux bougies consécutives sont espacées du **pas
nominal** (~15 min en M15) ; si l'écart est nettement supérieur à ce pas
(> `mult × nominal`), c'est qu'une fermeture s'est produite là. Cette méthode
attrape week-ends, coupure quotidienne de l'or et fériés, et **généralise à tout
marché futur** (indices/actions) sans table d'horaires à maintenir.

Avant d'émettre un FVG, on vérifie les deux écarts **dans** la fenêtre 3 bougies
(`i-2 → i-1` ET `i-1 → i`). Si l'un dépasse le seuil → fermeture détectée → le
FVG n'est **pas** émis. Une fenêtre à intervalles normaux reste inchangée : les
vrais FVG continus ne bougent pas.

## 3. Emplacement retenu — source de vérité unique

`src/environment/strategy_features.py` →
`SmartMoneyEngine._add_smc_base_features()`, juste après le calcul de
`FVG_SIGNAL`.

C'est le **détecteur FVG unique** du produit :
- `src/intelligence/smart_money/` n'est qu'une **façade** qui ré-exporte
  `SmartMoneyEngine` (extraction physique reportée au Sprint 6).
- Tous les chemins convergent ici : assembleur produit
  (`market_reading_assembler.py`), backtest, scanner, training. Corriger ici =
  **un seul point**, le moins invasif.

**Accès aux timestamps** : `self.df` est indexé par un `DatetimeIndex`
(l'assembleur construit `df` avec `index=[c.ts]` ; les providers font
`set_index('datetime')`). La détection a donc accès aux timestamps **sans aucune
plomberie** — pas besoin de toucher la couche collecte.

**Robustesse** : le filtre est gardé sur `isinstance(index, pd.DatetimeIndex)`
(les frames de test à index entier conservent le comportement géométrique
legacy) et sur `mult > 0`. Sur une bougie supprimée, **les quatre** colonnes
`FVG_SIZE / FVG_DIR / FVG_SIZE_NORM / FVG_SIGNAL` sont remises à 0, pour que tous
les consommateurs en aval (`confluence_detector`, `insight_v2/builder`,
`market_reading_mappers`, `readout_mappers`, `sentinel_scanner`) voient un état
« pas de FVG » cohérent.

## 4. Seuil retenu : `FVG_SESSION_GAP_MULT = 1.5` (× pas nominal)

Le pas nominal est calculé **depuis les données** comme la **médiane** des écarts
inter-bougies de l'index (robuste : > 98 % des bougies sont au pas nominal, donc
la médiane = le pas exact). Nouveau champ `SMCConfig.FVG_SESSION_GAP_MULT`
(défaut `1.5`, `0.0` désactive → comportement legacy).

Justification par la distribution réelle des écarts (XAU + EURUSD, 7 ans) — elle
est **bimodale très nette**, le seuil 1,5× tombe dans la zone vide entre le pas
nominal et le 1er vrai gap :

| TF | pas nominal | part des bougies au pas | 1er bucket suivant peuplé |
|----|----|----|----|
| M15 | 15 min | 98.9 % | 75 min (coupure quotidienne or) |
| H1 | 60 min | 95.6 % | 120 min |
| H4 | 240 min | 96.4 % | 480 min |

Aucun faux déclenchement possible sur intervalles normaux ; le 1er écart réel est
toujours ≥ 2× le pas.

## 5. Comptage FVG avant / après — moteur réel, `SmartMoneyEngine.analyze()`

Filtre OFF (`FVG_SESSION_GAP_MULT=0`) vs ON (`=1.5`), `FVG_THRESHOLD=0.1` défaut.
Sources historiques Dukascopy (M15 ; H1/H4 par resample).

| Combo | avant | après | retirés | % retiré |
|----|----|----|----|----|
| **XAUUSD M15** | 27 608 | 26 551 | 1 057 | 3.83 % |
| **XAUUSD H1** | 6 479 | 6 025 | 454 | 7.01 % |
| **XAUUSD H4** | 1 885 | 1 749 | 136 | 7.21 % |
| **EURUSD M15** | 29 310 | 29 005 | 305 | 1.04 % |
| **EURUSD H1** | 6 632 | 6 513 | 119 | 1.79 % |
| **EURUSD H4** | 1 889 | 1 802 | 87 | 4.61 % |

Conforme à l'impact attendu : **EURUSD quasi propre** (week-end dominical seul,
1–5 %) ; **or = résidu plus marqué** (coupure quotidienne 75 min, ~1 400
occurrences → 4–7 %) ; le % croît avec la TF (mêmes fermetures, moins de bougies).

## 6. ⚠️ Finding — le feed **live** (TwelveData) est continu 24/7

Les chiffres ci-dessus viennent des **sources à vrais trous** (CSV Dukascopy =
chemin backtest). Le cache **live** `data/candles.db` (TwelveData) est en
revanche **sans aucun trou** :

- XAUUSD / EURUSD M15 sur 8 jours traversant le week-end : **96 bougies/jour
  samedi ET dimanche**, couverture 24h, `max_gap = pas nominal`.
- Ces bougies week-end ne sont pas plates (OHLC variables, le prix glisse
  réellement) mais **`volume == 0`**… tout comme **100 %** des bougies en séance
  lundi-vendredi. `volume==0` ne discrimine donc **rien** dans ce feed.

**Conséquences (mesurées) :**
1. Le filtre écart-temps est un **no-op sur le live tel qu'alimenté aujourd'hui**
   (aucun écart de timestamp à détecter). Il reste pleinement actif sur le
   backtest et toute source à vrais trous, et généralisera aux indices/actions.
2. Le feed live ne produit **pas** de faux FVG aux frontières de séance : à
   l'ouverture du lundi le saut `|open − close_préc| / ATR` vaut médiane
   **0.02–0.07**, max **0.29** (un FVG exige > 0.10 ATR ; un vrai gap Dukascopy
   saute de plusieurs ATR). Le week-end est **rempli** de bougies continues → pas
   de discontinuité → pas de FVG-de-clôture.
3. `volume==0` étant universel ici, on **n'a pas** câblé de complément volume : il
   supprimerait 100 % des FVG live (signal toxique).

**Pourquoi c'est néanmoins la bonne implémentation** : la méthode écart-temps est
correcte, robuste et sans calendrier codé en dur, exactement comme spécifié. Le
vrai sujet du chemin live est **distinct** — TwelveData fabrique des bougies
week-end synthétiques à volume nul — et relève de l'**ingestion de données**
(faut-il les écarter/flagger en amont ?), pas de la détection FVG. À traiter
séparément si/quand le live devient prioritaire.

## 7. Tests

`tests/test_fvg_session_gap.py` (24 cas, tous verts) :
- (a) fenêtre à intervalles normaux + géométrie réelle → FVG émis (M15/H1/H4,
  haussier & baissier) — **non-régression**.
- (b) même géométrie mais saut de temps dans la fenêtre (`i-1→i` **et** `i-2→i-1`)
  → aucun FVG.
- (c) cas baissier symétrique sous fermeture.
- **Narrowness** : un gap **hors** de la fenêtre laisse le FVG intact.
- Toggle `FVG_SESSION_GAP_MULT=0` → comportement legacy restauré.
- Suppression cohérente des 4 colonnes `FVG_*`.
- Index entier (non-Datetime) → filtre no-op (legacy préservé).

Non-régression : `test_sprint5_fvg_threshold`, `test_market_reading_mappers`,
`test_market_reading_assembler`, `test_confluence_detector`,
`test_data_quality_bos_regression`, `test_multi_instrument`,
`test_insight_assembler`, `test_pipeline_integration`, `test_mtf_wiring`,
`test_production_wiring`, `test_chantier3_smoke_e2e` (+ autres) — **0 régression**.

## 8. Fichiers touchés

- `src/environment/strategy_features.py` — champ `FVG_SESSION_GAP_MULT` +
  bloc de suppression session-gap dans `_add_smc_base_features`.
- `tests/test_fvg_session_gap.py` — nouveau (24 tests).
- `docs/audits/FVG_SESSION_GAP_AWARENESS_2026_06_17.md` — ce rapport.
