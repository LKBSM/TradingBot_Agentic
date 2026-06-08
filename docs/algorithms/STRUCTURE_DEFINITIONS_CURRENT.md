# STRUCTURE_DEFINITIONS_CURRENT — Définitions de détection ACTUELLES (extraites du code)

> **Audit de validation algorithmique — Phase 1.1**
> Date : 2026-06-07 · Branche : `audit/validation-algorithmique-detection-structures`
> Périmètre : moteur de détection SMC (`SmartMoneyEngine`) + mappers MarketReading + moteur de régime.
> Objet : documenter, en français clair, **ce que le code fait réellement** (pas ce qu'il devrait faire). La comparaison aux standards canoniques est dans `STRUCTURE_DEFINITIONS_AUDIT.md`.

---

## 0. Cartographie du moteur — où vit quoi

| Couche | Fichier | Rôle |
|---|---|---|
| **Détection brute** (BOS/CHOCH/OB/FVG/fractals/retest) | `src/environment/strategy_features.py` (classe `SmartMoneyEngine`, ~1213 LOC) | Enrichit un DataFrame OHLCV avec colonnes `BOS_*`, `CHOCH_*`, `FVG_*`, `OB_*`, `*_FRACTAL` |
| **Façade institutionnelle** | `src/intelligence/smart_money/__init__.py` | Re-exporte `SmartMoneyEngine` (extraction physique différée Sprint 6) |
| **Mapping → produit** | `src/intelligence/market_reading_mappers.py` | Transforme la dernière ligne de features en `MarketReadingStructure` + `MarketReadingRegime` |
| **Assemblage lazy** | `src/intelligence/market_reading_assembler.py` | Orchestre fetch candles → pipeline SMC → structure/régime/events → description |
| **Régime descriptif** | `src/intelligence/market_reading_mappers.py` (`_derive_trend`/`_derive_volatility`/`_derive_market_phase`) | Calcule trend/volatilité/phase **à partir des closes**, indépendamment du moteur SMC |
| **Description LLM** | `src/intelligence/haiku_description_engine.py` | Génère la phrase descriptive (reçoit **seulement `tags` + `regime`**) |
| **Schéma produit** | `src/intelligence/market_reading_schema.py` | Contrat Pydantic `MarketReading` v2.0.0 |

> ⚠️ **Point d'architecture cardinal** : le `MarketReading` n'est PAS produit par le `ConfluenceDetector` (scoring 0-100). L'assembler par défaut passe `confluence_signal=None` (`market_reading_assembler.py:84-126`). Le `MarketReading` décrit **un état structurel**, pas un setup scoré. Le scoring/confluence reste sur le flux legacy `InsightSignalV2`.

---

## 1. SWING POINTS (Fractals)

- **Fonction** : `SmartMoneyEngine._add_smc_base_features()` — `strategy_features.py:597-710`
- **Méthode** : fractale de Williams vectorisée, **causale** (anti look-ahead).

**Conditions déclenchantes**
- Fenêtre `window_size = 2·N + 1` (N = `FRACTAL_WINDOW`, défaut **2** → fenêtre de 5 bougies).
- `UP_FRACTAL` à la bougie `i` si `high[i] == max(high[i-2 : i+2])` (rolling `center=True`).
- `DOWN_FRACTAL` à la bougie `i` si `low[i] == min(low[i-2 : i+2])`.
- **Décalage causal** : `.shift(N)` — un fractal n'est connu que N bougies après sa formation. Les N premières et N dernières lignes sont forcées à `NaN`.

**Sortie** : colonnes `UP_FRACTAL` (prix du high), `DOWN_FRACTAL` (prix du low), `NaN` sinon.

**Paramètres**
| Param | Défaut | Source |
|---|---|---|
| `FRACTAL_WINDOW` | 2 | `SMCConfig` |

---

## 2. BOS — Break of Structure

- **Fonction** : `_calculate_bos_choch_numba()` (`strategy_features.py:32-134`), fallback Python `_calculate_bos_choch_python()` (159-234). Appelée via `calculate_bos_choch_fast()`.
- **Comparaison** : sur le **close** (corps), pas sur le wick.
- **Structure de référence** : `current_high_structure` / `current_low_structure` = extrêmes des **fractals** accumulés depuis le dernier reset (pas les high/low de bougies brutes).

**Conditions déclenchantes**
- **BOS de continuation haussier** (`strategy_features.py:117-123`) : état précédent `bos_signal[i-1] >= 0` ET `close[i] > current_high_structure` ET `allow_bos_up`.
- **BOS de continuation baissier** (124-130) : `bos_signal[i-1] <= 0` ET `close[i] < current_low_structure` ET `allow_bos_down`.
- **BOS de renversement** (100-115) : si l'état précédent était de signe opposé, le break déclenche simultanément un CHOCH (voir §3).
- Après un break, la structure est ré-ancrée sur le dernier fractal (`current_high_structure = last_fractal_high`, etc.).

**Distinction bull/bear** : signe de `bos_signal` / `bos_event` (+1 haussier, −1 baissier).

**Sorties** (`strategy_features.py:734-737`)
| Colonne | Sémantique |
|---|---|
| `BOS_SIGNAL` | −1/0/+1 — **état de tendance propagé** (legacy). Reste à ±1 tant qu'aucun break opposé. |
| `BOS_EVENT` | −1/0/+1 — flag d'événement, **uniquement la bougie du break** (continuation ET renversement). |
| `BOS_BREAK_LEVEL` | niveau structurel cassé sur les bougies d'événement (`NaN` sinon). |

**Paramètres** : `FRACTAL_WINDOW=2` (structure) ; pas de buffer ATR ni de confirmation multi-bougies sur le break lui-même.

---

## 3. CHOCH — Change of Character

- **Fonction** : intégrée dans `_calculate_bos_choch_numba()` (`strategy_features.py:100-115`).
- **Définition codée** : CHOCH = **BOS de renversement**. Il se déclenche au même instant qu'un BOS qui casse la structure **dans le sens opposé à l'état de tendance précédent**.

**Conditions déclenchantes**
- CHOCH haussier (+1) : `bos_signal[i-1] == -1` ET `close[i] > current_high_structure`.
- CHOCH baissier (−1) : `bos_signal[i-1] == 1` ET `close[i] < current_low_structure`.

**Sortie** : `CHOCH_SIGNAL` (−1/0/+1), non nul uniquement sur les bougies de renversement.

> **Note** : il n'existe pas de notion séparée « CHOCH d'abord, BOS ensuite » dans le code. CHOCH et BOS-de-renversement sont **le même événement** (même bougie). Voir divergence M-CHOCH dans l'audit.

**Sous-produit** : `CHOCH_DIVERGENCE` (`strategy_features.py:817-879`) — divergence RSI fractale (non utilisée par le mapper MarketReading). ⚠️ Bug d'indexation connu P0-15.

---

## 4. ORDER BLOCKS (OB)

- **Fonction** : `_add_smc_order_blocks()` — `strategy_features.py:756-815`.
- **Définition codée** : motif **englobant directionnel** sur 2 bougies (PAS la définition ICT « dernière bougie opposée avant displacement/BOS »).

**Conditions déclenchantes**
- **OB haussier** (`strategy_features.py:766-770`) :
  - bougie `i-1` baissière (`close[i-1] < open[i-1]`)
  - bougie `i` haussière (`close[i] > open[i]`)
  - nouveau plus-haut (`high[i] > high[i-1]`)
- **OB baissier** (772-776) : miroir (bougie `i-1` haussière, `i` baissière, `low[i] < low[i-1]`).

**Filtre FVG optionnel** (`OB_REQUIRE_FVG`, défaut **False**) : par défaut l'OB est détecté indépendamment d'un FVG ; si présent, bonus de force `OB_FVG_BONUS=0.2`.

**Zone OB** : `BULLISH_OB_HIGH/LOW` (resp. `BEARISH_*`) = high/low de la **bougie `i-1`** (`strategy_features.py:791-794`).

**Force** : `OB_STRENGTH_NORM = (taille zone i-1) / ATR (+0.2 si FVG adjacent)` — `strategy_features.py:797-815`.

**Mitigation** : ❌ **non suivie** dans le moteur. Aucune logique de remplissage/invalidation.

**Sorties** : `BULLISH_OB_HIGH/LOW`, `BEARISH_OB_HIGH/LOW`, `OB_STRENGTH_NORM`.

---

## 5. FVG — Fair Value Gap

- **Fonction** : `_add_smc_base_features()` — `strategy_features.py:646-679`.
- **Définition codée** : gap 3-bougies classique (imbalance entre la bougie `i` et la bougie `i-2`).

**Conditions déclenchantes**
- **FVG haussier** : `low[i] > high[i-2]` → taille = `low[i] − high[i-2]`.
- **FVG baissier** : `high[i] < low[i-2]` → taille = `low[i-2] − high[i]`.
- **Normalisation** : `FVG_SIZE_NORM = FVG_SIZE / ATR`.
- **Seuil de signal** : `FVG_SIGNAL = FVG_DIR` si `|FVG_SIZE_NORM| > FVG_THRESHOLD` (défaut **0.1**, soit 10 % de l'ATR), sinon 0.

**Distinction bull/bear** : `FVG_DIR` (+1/−1/0).

**Remplissage** : ❌ **non suivi** dans le moteur (le statut « filled/partially_filled » du schéma n'est jamais alimenté).

**Sorties** : `FVG_SIZE`, `FVG_DIR`, `FVG_SIZE_NORM`, `FVG_SIGNAL`.

---

## 6. RETEST (BOS retest state machine)

- **Fonction** : `_calculate_bos_retest_numba()` (`strategy_features.py:259-334`), wrapper `calculate_bos_retest_fast()`.
- **États** : 0 IDLE · ±1 AWAITING (BOS frais, attente pullback) · ±2 ARMED (retest confirmé).

**Transitions (cas haussier)**
- Invalidation si `close < level − invalid_tol_atr·ATR`.
- Retest détecté si `low ≤ level + retest_tol_atr·ATR` → ARMED.
- Timeout après `awaiting_timeout` bougies.
- ARMED expire après `armed_window` bougies.

**Paramètres**
| Param | Défaut |
|---|---|
| `RETEST_TOL_ATR` | 0.5 |
| `RETEST_INVALID_TOL_ATR` | 1.0 |
| `RETEST_AWAITING_TIMEOUT` | 20 |
| `RETEST_ARMED_WINDOW` | 30 |

**Sorties** : `BOS_RETEST_STATE` (−2..+2), `BOS_RETEST_ARMED` (−1/0/+1).

---

## 7. RÉGIME — trend / volatilité / phase de marché

> ⚠️ **Le régime du MarketReading n'utilise NI le moteur SMC NI les HMM.** Il est recalculé indépendamment, à partir des seuls closes/high/low de la fenêtre, dans `market_reading_mappers.py`.

### 7.1 Trend — `_derive_trend()` (`market_reading_mappers.py:210-223`)
- < 5 closes → `neutral`.
- `pct_move = |close[-1] − close[0]| / |close[0]|` ; `rng_pct = (max−min)/|close[0]|`.
- Si `pct_move < rng_pct · 0.3` → **`ranging`**.
- Sinon `bullish` si `close[-1] > close[0]`, sinon `bearish`.
- **Indicateur** : aucun. Pur momentum close-à-close sur la fenêtre (200 bougies par défaut).

### 7.2 Volatilité — `_derive_volatility()` (`market_reading_mappers.py:226-244`)
- < 14 bougies → `normal`.
- TR par bougie = `high − low`.
- `recent = moy(TR[-7:])`, `baseline = moy(TR[:-7])`, `ratio = recent/baseline`.
- `ratio < 0.7` → **`low`** · `0.7 ≤ ratio ≤ 1.3` → **`normal`** · `ratio > 1.3` → **`elevated`**.
- **Indicateur** : True Range relatif (récent vs historique), PAS un percentile ATR.

### 7.3 Phase — `_derive_market_phase()` (`market_reading_mappers.py:247-252`)
| Trend | Volatilité | Phase |
|---|---|---|
| bullish/bearish | elevated | `expansion` |
| bullish/bearish | normal/low | `trend` |
| ranging | toute | `ranging` |
| neutral | toute | `accumulation` |

- **Type** : table de correspondance pure (pas de HMM, pas de ML).
- **Note** : la valeur `distribution` existe au schéma mais **aucun chemin de code ne la produit**.

### 7.4 MTF confluence — `_derive_bias_from_candles()` (`market_reading_mappers.py:255-260`)
- Applique `_derive_trend()` à chaque TF supérieur fourni (`h1/h4/d1/w1`). Sortie : biais par TF.

### 7.5 Modèles ML présents mais NON câblés au MarketReading
| Modèle | Fichier | Statut vis-à-vis du MarketReading |
|---|---|---|
| `RegimeClassifier` (GaussianHMM 3 états) | `regime_classifier.py` | **Non câblé** — viz timeline webapp uniquement |
| `VolatilityForecaster` (HAR-RV + HMM) | `volatility_forecaster.py` | **Non câblé** — prévision ATR future |
| `RegimeFilter` (percentile ATR + session) | `regime_filter.py` | **Non câblé** — gate de signaux legacy |
| `RegimeGate` (BOCPD + bipower) | `regime_gate.py` | **Non câblé** — blocage d'entrées legacy |

---

## 8. MAPPING vers le `MarketReading` — ce qui est réellement publié

`confluence_signal_to_structure()` — `market_reading_mappers.py:91-198` lit la **dernière ligne** des features :

| Champ MarketReading | Source feature | Remarque |
|---|---|---|
| `structure.bos.direction` | signe de `BOS_SIGNAL` | état propagé (peut être « ancien ») |
| `structure.bos.validation_status` | `BOS_EVENT≠0` → `confirmed`, sinon `pending` | |
| `structure.bos.level` | `smc_features["BOS_PRICE_LEVEL"]`, défaut `current_price` | ⚠️ **F1** : l'engine ne produit PAS `BOS_PRICE_LEVEL` (il produit `BOS_BREAK_LEVEL`) → niveau = `current_price` **toujours** |
| `structure.choch.level` | `smc_features["CHOCH_PRICE_LEVEL"]`, défaut `current_price` | ⚠️ **F2** : clé inexistante → niveau = `current_price` **toujours** |
| `structure.order_blocks[0]` | si `OB_STRENGTH_NORM > 0` | ⚠️ **F3** : `level_high/low = current_price ± ATR/2` (proxy), PAS `BULLISH_OB_HIGH/LOW`. ⚠️ **F4** : 1 OB max (bougie courante), pas une liste de zones actives |
| `structure.fair_value_gaps[0]` | si `FVG_SIGNAL ≠ 0` | idem F3/F4 : niveaux proxy, 1 FVG max |
| `structure.*.status` | constante `"active"` | mitigation/fill jamais calculés |
| `regime.*` | `candles_to_regime()` (§7) | indépendant du moteur SMC |

**Description** : `tags_and_description()` (template) ou `HaikuDescriptionEngine.generate(tags, regime)`.
- ⚠️ **F5** : le moteur Haiku **ne reçoit que `tags` + `regime`** (`market_reading_assembler.py:265`, `haiku_description_engine.py:97-101`), jamais l'objet `structure`. L'info structurelle ne lui parvient que via les **tags** (`bos_recent_bullish`, `ob_active`, `fvg_active`, `choch_recent_*`, `retest_in_progress`, `mtf_aligned/mixed/divergent`). Les niveaux de prix ne sont jamais transmis au LLM.

---

## 9. Forbidden tokens — deux ensembles distincts

| Ensemble | Fichier | Usage | Taille |
|---|---|---|---|
| `FORBIDDEN_TOKENS` | `market_reading_mappers.py:45-60` | Garde post-génération du **moteur Haiku MarketReading** | 13 tokens |
| `ALL_FORBIDDEN_TOKENS` (4 catégories) | `chatbot/constants.py:82-149` | Garde du **chatbot** (système séparé) + patterns adversariaux | ~90 tokens + 36 patterns |

> Pour l'évaluation Phase 3, le contrôle « Test C » est fait contre l'ensemble **complet** (`ALL_FORBIDDEN_TOKENS`, plus strict que la garde réelle du moteur), avec exclusions homonymes documentées dans `constants.py:22-47`.

---

## 10. Synthèse des paramètres

| Composant | Paramètre | Défaut |
|---|---|---|
| Fractals | `FRACTAL_WINDOW` | 2 |
| FVG | `FVG_THRESHOLD` | 0.1 ×ATR |
| OB | `OB_REQUIRE_FVG` / `OB_FVG_BONUS` | False / 0.2 |
| Retest | `RETEST_TOL_ATR` / `INVALID_TOL_ATR` | 0.5 / 1.0 |
| Retest | `AWAITING_TIMEOUT` / `ARMED_WINDOW` | 20 / 30 |
| Trend | seuil ranging | `pct_move < 0.3 · rng_pct` |
| Volatilité | buckets ratio | 0.7 / 1.3 |

---

*Suite : `STRUCTURE_DEFINITIONS_AUDIT.md` (comparaison aux définitions canoniques SMC + classement des divergences).*
