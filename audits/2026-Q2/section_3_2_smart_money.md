# Section 3.2 — Smart Money / ICT Detection — Audit institutionnel Phase 1

**Repo** : `TradingBOT_Agentic` (branche `institutional-overhaul`)
**Périmètre** : `src/environment/strategy_features.py` (1213 LOC) + consommateurs SMC
dans `src/intelligence/confluence_detector.py`.
**Date** : 2026-05-15
**Auditeur** : Claude (Lead Quant Architect) — Sprint 0 audit, code path + activation only.
**Rapports antérieurs** : `reports/eval_03/eval_03_smart_money.md` (note 4.5/10, 2026-04-24).
**Artefact empirique** : `audits/2026-Q2/section_3_2_smart_money_stats.json`
(généré par `scripts/audit_3_2_smart_money.py`).
**Décision tranchée** : section 3.2 audit en Sprint 0 couvre **uniquement** code path +
distribution des firings + reproductibilité + sensibilité. Le scoring F1 vs annotations
expertes est reporté en Sprint 2 (les annotations n'existent pas encore).

---

## 0. Verdict synthétique

| Axe | Verdict | Score |
|---|---|---|
| Reproductibilité | Déterministe (seed-free, pas de random, parité Numba=Python OK) | **9.0/10** |
| Conformité ICT | FVG conforme mais seuil laxe ; BOS/CHOCH conforme post-Sprint 1 ; **OB non conforme ICT** ; retest tolérance ≈ spread | **5.0/10** |
| Sensibilité paramètres | 90 % des magic numbers configurables via `SMCConfig`. 2 incohérences entre defaults (cf. P1-F2) | **6.5/10** |
| Statistiques d'activation | Firing rates raisonnables (BOS 3.16 %, FVG 16 %), MAIS OB 24.7 % (1 OB tous les 4 bars) reste massif | **5.5/10** |
| Tests automatisés | 5 fichiers Sprint 1-7 + 3 régressions = ~150 assertions ; gaps réels sur ICT-anchor OB et data leakage de divergence | **6.0/10** |
| Performance | 12.1 s / 172 k bars en fallback Python (Numba indisponible en env actuel) — risque latence prod | **5.0/10** |
| Cross-asset | Logique paramétrée ATR-relative (sain). EUR/USD donne stats comparables à XAU. | **8.0/10** |
| Bug "BOS sur 100 % des bars" | **Confirmé résolu** par la fix Sprint 1 + CSV à 98.72 % coverage | **9.0/10** |
| **Note globale section 3.2** | Code path solide, sémantique ICT partielle, perf et OB non-anchored = freins commerciaux | **6.0 / 10** |

> Note : la 4.5/10 de `eval_03_smart_money.md` reste **valide** sur la conformité ICT du
> détecteur OB. Cet audit relève la note globale parce que la reproductibilité, les
> tests, la cross-asset et la fix Sprint 1 sont des actifs solides ; mais il maintient
> le P0 sur OB ICT-compliance et signale 3 P1 supplémentaires que `eval_03` n'avait
> pas vus (divergence RSI, parité retest, magic numbers de défaut).

---

## 1. Périmètre du code audité

Le « Smart Money Engine » n'a pas (encore) de module dédié `smart_money_engine/`. Toute
la logique vit dans un seul fichier :

| Bloc | Fichier:lignes | Notes |
|---|---|---|
| Numba BOS/CHOCH | `src/environment/strategy_features.py:32-134` | core JIT |
| Python fallback BOS | `strategy_features.py:159-234` | parité avec Numba — voir Annexe B |
| Wrapper BOS public | `strategy_features.py:137-156` (`calculate_bos_choch_fast`) | route Numba/Python via `NUMBA_AVAILABLE` |
| Retest state machine Numba | `strategy_features.py:259-334` | 4 états (idle/awaiting/armed × ±) |
| Retest state machine Python | `strategy_features.py:337-407` | parité conditionnelle — voir P1-F2 |
| Wrapper retest public | `strategy_features.py:410-436` | **defaults divergents avec `SMCConfig`** |
| `SMCConfig` (Pydantic) | `strategy_features.py:440-518` | seul point de paramétrage exposé |
| `SmartMoneyEngine.__init__` | `strategy_features.py:537-562` | force colonnes en lowercase |
| Indicateurs TA (RSI/MACD/BB/ATR) | `strategy_features.py:564-595` | délègue à `ta-lib` |
| Fractals + FVG | `_add_smc_base_features` `strategy_features.py:597-710` | causal (shift N) |
| BOS/CHOCH wrapper classe | `_calculate_structure_iterative` `strategy_features.py:712-754` | alimente retest |
| Order Blocks | `_add_smc_order_blocks` `strategy_features.py:756-815` | engulfing — **PAS ICT** |
| Divergence RSI | `_detect_rsi_divergence` `strategy_features.py:817-879` | **bug d'indexage** P1-F3 |
| Pipeline `analyze()` | `strategy_features.py:881-950` | 5 étapes |
| Consommateurs SMC | `src/intelligence/confluence_detector.py:233,245,382,431,586` | direction via `BOS_SIGNAL`, gate via `BOS_RETEST_ARMED` |

Le module est utilisé par 28 fichiers (`scripts/`, `tests/`, `src/environment/environment.py`,
`src/intelligence/main.py`). Production charge la classe avec `config={}` (defaults SMCConfig),
cf. `src/intelligence/main.py:172-174`.

---

## 2. Reproductibilité (P0 critère bloquant)

### 2.1 Déterminisme structurel

| Critère | Constat | Référence |
|---|---|---|
| Pas de RNG dans le pipeline | OK — aucun `np.random` / `random` / dropout dans `analyze()` | grep manuel |
| `analyze()` lancé 2× sur les mêmes données produit colonnes byte-identiques | OK pour les 15 colonnes critiques (BOS_EVENT, BOS_SIGNAL, BOS_BREAK_LEVEL, CHOCH, FVG_*, OB_*, RETEST_*, DIVERGENCE, RSI, ATR) | `audits/2026-Q2/section_3_2_smart_money_stats.json#xau_20k.reproducibility.all_match=true` |
| Fractals causaux (shift N) | OK — `iloc[:N]` + `iloc[-N:]` forcés à NaN, raise sur fuite | `strategy_features.py:641-697` |
| BOS retest gère NaN ATR | OK — clause `if (not np.isnan(atr[i]) and atr[i] > 0) else 0.0` | `strategy_features.py:293, 370` |
| Numba `cache=True` | OK — pas de recompilation entre runs | `strategy_features.py:32, 259` |

### 2.2 Numba ↔ Python parity

Tests automatisés présents (`tests/test_bos_retest.py::test_numba_and_python_agree`)
mais limités au retest. Pour BOS/CHOCH, aucun test équivalent (gap).

L'audit empirique a confirmé la parité **à arguments identiques** :

```
[xau 5k bars]  bos_signal=True  choch_signal=True  bos_event=True
              bos_break_level_close=True  retest_state (identiques args)=True
              retest_armed (identiques args)=True
```

(Cf. `audits/2026-Q2/section_3_2_smart_money_stats.json#xau_5k.parity`.)

### 2.3 Numba absent dans l'env actuel

L'env Sprint 0 a `NUMBA_AVAILABLE = False` (cf. JSON `meta.numba_available`). Le
fallback Python s'exécute correctement (12.1 s sur 172 k bars vs ~0.3 s avec Numba
selon eval_03). **Risque** : prod live Railway / Docker doit garantir Numba installé
sinon la latence scanner (cible 30 s SLA, eval_09) est en danger. Voir `requirements.txt`
qui inclut `numba` (vérifié dans memory.md).

**Verdict reproductibilité : 9/10.** Pas de findings P0 mais P1 sur defaults parameters
(F2 ci-dessous).

---

## 3. Conformité ICT — audit ligne à ligne

### 3.1 Fractals (swing points) — `strategy_features.py:606-644`

```python
N = self.config.FRACTAL_WINDOW  # default 2 → fenêtre 5 bougies
rolling_max = self.df['high'].rolling(window=2*N+1, center=True).max()
up_fractal_raw = np.where(self.df['high'] == rolling_max, self.df['high'], np.nan)
self.df['UP_FRACTAL'] = pd.Series(up_fractal_raw).shift(N).values
```

- ✅ Fenêtre symétrique 5 bougies (Bill Williams standard)
- ✅ Shift N applique le délai causal (anti look-ahead)
- ✅ Raise si données leak les N derniers bars (`strategy_features.py:692-696`)
- ⚠️ **Egalités** : `==` strict sur `high == rolling_max` peut être faux sur les plateaux
  (bougies identiques en M15 sur l'or ≈ 0.01 % cas, négligeable, mais à mentionner).

### 3.2 Fair Value Gap (FVG) — `strategy_features.py:650-679`

```python
bullish_fvg_size = np.where(
    self.df['low'] > self.df['high'].shift(2),
    self.df['low'] - self.df['high'].shift(2),
    0.0
)
```

| Critère | Constat |
|---|---|
| Définition ICT (3 bougies, gap entre bar[i-2].high et bar[i].low) | ✅ Conforme |
| Seuil ATR | `FVG_THRESHOLD = 0.1` par défaut |
| ATR multiplier | OK, vectorisé |
| Direction (bullish=+1, bearish=-1) | OK |

**Finding (P1-F1) — Seuil FVG trop laxe.** Cf. sweep empirique :

| FVG_THRESHOLD | n_fvg / 10k bars | FVG/1000 bars |
|---:|---:|---:|
| 0.0 | 2264 | 226.97 |
| 0.1 (default) | 1829 | 183.36 |
| 0.2 | 1415 | 141.85 |
| **0.4** | **842** | **84.41** |
| 0.5 | 643 | 64.46 |
| 1.0 | 172 | 17.24 |

Le seuil 0.1 ATR ≈ 30 cents sur XAU M15 — c'est l'ordre du **spread**. Un seuil
visuellement « imbalance » serait 0.4-0.5 ATR. Voir eval_03 §3.1 — finding déjà
documenté.

**Source** : `audits/2026-Q2/section_3_2_smart_money_stats.json#xau_sensitivity.fvg_threshold_sweep`.

### 3.3 BOS / CHOCH — `strategy_features.py:32-134` (Numba) + `159-234` (Python)

**Conforme ICT post-Sprint 1.**

Mécanique :
- `bos_event` = 1/-1 **uniquement** sur la bar de rupture (correct).
- `bos_signal` = état trend qui se propage (legacy RL — utile mais à découpler).
- CHOCH = BOS dans le sens opposé à la tendance en cours.
- Filtrage de répétition par `allow_bos_up = last_fractal_high > last_bos_up_level`
  (force un nouveau fractal *au-dessus* du précédent break avant de re-firer) —
  c'est la fix du bug « BOS sur 100 % des bars » documentée dans la mémoire +
  `tests/test_bos_no_repeated_fire.py`.

**Statistiques sur XAU 172 849 bars (full 2019-2026)** :

| Métrique | Valeur | Per-1000 bars |
|---|---:|---:|
| BOS event up | 2 970 | 17.18 |
| BOS event down | 2 491 | 14.41 |
| **BOS event total** | **5 461** | **31.59** |
| BOS signal up (état) | 84 339 (48.8 %) | — |
| BOS signal down (état) | 88 477 (51.2 %) | — |
| CHOCH up | 1 039 | 6.01 |
| CHOCH down | 1 039 | 6.01 |

**Firing rate BOS event** : 3.16 % des bars → **dans la cible [0.5, 10 %]** du test
de régression `test_data_quality_bos_regression.py`.

**Symétrie CHOCH** : 1039 / 1039 — strictement symétrique. Comme noté par eval_03,
c'est mathématiquement attendu (CHOCH compte les transitions trend, qui sont par
définition équilibrées entre up→down et down→up sur un horizon long).

### 3.4 Order Blocks — `strategy_features.py:756-815` (NON CONFORME ICT)

```python
bullish_ob_condition = (
    (self.df['close'].shift(1) < self.df['open'].shift(1)) &
    (self.df['close'] > self.df['open']) &
    (self.df['high'] > self.df['high'].shift(1))
)
```

C'est une **bougie engulfing**, pas un Order Block ICT. Définition canonique :

> « OB bullish = dernière bougie baissière (down candle) **AVANT** le déplacement
> impulsif qui crée un BOS_UP. »

Ce que l'implémentation actuelle fait :
- ✅ Bougie haussière qui suit une bougie baissière
- ❌ Aucun lien avec un BOS précédent (anchor manquant)
- ❌ Aucun filtre d'impulse (body / ATR ratio)
- ❌ Pas de tracking de mitigation (l'OB persiste tant que non touché par le prix)

**Évidence empirique** (`section_3_2_smart_money_stats.json#xau_full.ob_anchor`) :

| Population | Count | Within ±20 bars d'un BOS | % anchored |
|---|---:|---:|---:|
| Bullish OB (XAU 6 ans) | 21 442 | 12 765 | **59.5 %** |
| Bearish OB | 21 178 | 12 564 | **59.3 %** |

→ 40 % des OB détectés ne sont **pas** près d'un BOS, donc ne peuvent pas être
des OB ICT par définition. Le filtre OB_REQUIRE_FVG=True réduit à 232/235 OB sur
10 k bars (vs 1224/1179 sans filtre) — bonne dégradation mais c'est de l'engulfing
+ FVG, pas un anchor BOS.

**Finding P0-F1** : OB ICT-compliance impossible avec la logique actuelle. À
réécrire (voir §8 reco Sprint 1).

### 3.5 Retest state machine — `strategy_features.py:259-407`

Mécanique : 4 états (`IDLE / AWAITING / ARMED` × ±). Transitions :

| Source | Condition | Cible |
|---|---|---|
| `AWAITING +1` | `close < level - invalid_tol_atr * ATR` | IDLE (failed break) |
| `AWAITING +1` | `low <= level + retest_tol_atr * ATR` | `ARMED +2` |
| `AWAITING +1` | `bars_in > awaiting_timeout` | IDLE (stale) |
| `ARMED +2` | `bars_in > armed_window` | IDLE |
| `ARMED +2` | `close < level - invalid_tol_atr * ATR` | IDLE (support lost) |
| `ARMED +2` | sinon | reste ARMED, émet `retest_armed=+1` |
| Any | nouveau `bos_event` | switch direction |

**Conforme à la théorie SMC** : break → pullback → continuation.

**Paramètres par défaut** :

| Param | SMCConfig.default | calculate_bos_retest_fast default | Cohérent ? |
|---|---:|---:|---|
| RETEST_TOL_ATR | 0.5 | 0.5 | OK |
| RETEST_INVALID_TOL_ATR | 1.0 | 1.0 | OK |
| RETEST_AWAITING_TIMEOUT | 20 | 20 | OK |
| **RETEST_ARMED_WINDOW** | **30** | **5** | **❌ INCOHÉRENT** |

**Finding (P1-F2) — Magic-number incohérent entre defaults**.
`SMCConfig.RETEST_ARMED_WINDOW` (`strategy_features.py:510-518`) vaut 30 — c'est
le default consommé par `_calculate_structure_iterative` (`strategy_features.py:746`).
Mais `calculate_bos_retest_fast` (`strategy_features.py:420`) a `armed_window=5`
comme default *de signature*. Les appelants directs de la fonction (sans passer
par `SMCConfig`) utilisent donc 5 — incohérence silencieuse.

Évidence : reproduit empiriquement par `scripts/audit_3_2_smart_money.py` — la
parité `numba_python_parity()` initialement échoue parce que mon test injectait
30 (matching `SMCConfig`) côté python et la fast path utilisait 5 par défaut.
Avec args identiques, parité OK (cf. JSON `xau_5k.parity` après correction).

**Finding (P1-F4) — Retest tolerance par défaut ≈ spread**. `RETEST_TOL_ATR=0.5`
sur XAU M15 (ATR moyen ~3 $) donne tol ≈ 1.5 $, ce qui est l'ordre du **spread
broker** (1-1.5 $ sur XAU). Le test `xau_sensitivity.retest_tol_sweep` montre :

| RETEST_TOL_ATR | armed_per_1k_bars |
|---:|---:|
| 0.1 | 260.05 |
| **0.25** | **280.30** |
| 0.5 (default) | 323.01 |
| 0.75 | 338.05 |
| 1.0 | 358.80 |

Le passage 0.5 → 0.25 ne réduit que ~13 % des armed (281 vs 323 / 1k). Donc le
durcissement seul ne suffit pas, il faut compléter par un **touch strict** (low ∈
[level - tol, level + tol] ET retour au-dessus dans la même bar) — voir reco
Sprint 1.

### 3.6 Divergence RSI — `strategy_features.py:817-879`

**Finding (P1-F3) — Indexage incohérent entre price et RSI**.

À la ligne 849-857, lorsque `down_fractals[i]` n'est pas NaN, le code récupère le
prix et le RSI **au bar de confirmation `i`** :

```python
current_low = lows[i]      # ← prix au bar de confirmation, pas au swing
current_rsi = rsi[i]       # ← idem RSI
```

Or `down_fractals[i]` = `lows[i-N]` (le low du **bar du swing**, pas du bar de
confirmation). Donc `current_low` ≠ valeur stockée dans `DOWN_FRACTAL`.

**Évidence empirique** (XAU 5k bars, 20 premiers fractals) :

```
DOWN_FRACTAL - low (current bar):
mean   -1.177 $
std     0.736
max    -0.130
min    -3.020
```

Le low « actuel » est en moyenne 1.18 $ au-dessus du swing low. La divergence
bullish (`current_low < prev_low AND current_rsi > prev_rsi`) compare donc :
- prix actuel **AU BAR DE CONFIRMATION** (pas au swing low)
- RSI actuel **AU BAR DE CONFIRMATION** (~2 bars après le swing, RSI a déjà
  rebondi → biais asymétrique)

→ La divergence détecte un signal techniquement **mal aligné** ICT (et la
littérature classique TA). À fixer en remplaçant `lows[i]` par `down_fractals[i]`
et `rsi[i]` par `rsi[i-N]` (avec N=`FRACTAL_WINDOW`).

**Tests existants** (`tests/test_sprint7_rsi_divergence.py`) couvrent :
- présence colonne ✅
- range valide [-1, 0, +1] ✅
- ne se trigger pas sans données ✅
- ❌ **ne valident PAS** que le prix comparé correspond bien au swing point.

### 3.7 Synthèse conformité ICT

| Détecteur | Conformité | Source |
|---|:---:|---|
| Fractals | ✅ Bill Williams 5 bars causal | `strategy_features.py:617-644` |
| FVG | ✅ 3-bougies | `strategy_features.py:650-679` |
| FVG seuil 0.1 ATR | ⚠️ trop laxe (P1-F1) | sweep empirique |
| BOS event | ✅ post Sprint 1 | `strategy_features.py:87-134` |
| BOS suppression répétée | ✅ via `last_bos_up_level` | `strategy_features.py:97-99, 117-130` |
| CHOCH | ✅ symétrique attendu | idem |
| Retest state machine | ✅ logique correcte | `strategy_features.py:259-334` |
| Retest tolerance 0.5 ATR | ⚠️ ≈ spread (P1-F4) | sweep empirique |
| Retest armed_window default mismatch | ❌ P1-F2 | `:420` vs `:510` |
| Order Blocks | ❌ engulfing ≠ ICT (P0-F1) | `strategy_features.py:766-789` |
| RSI Divergence | ❌ bug d'indexage (P1-F3) | `strategy_features.py:849-857` |

---

## 4. Statistiques d'activation — XAU full 2019-2026 (172 849 bars)

Source : `audits/2026-Q2/section_3_2_smart_money_stats.json#xau_full.stats`.

### 4.1 Tableau global

| Métrique | Count | % bars | Per 1000 bars |
|---|---:|---:|---:|
| BOS event up | 2 970 | 1.72 % | 17.18 |
| BOS event down | 2 491 | 1.44 % | 14.41 |
| BOS event total | 5 461 | **3.16 %** | 31.59 |
| CHOCH up | 1 039 | 0.60 % | 6.01 |
| CHOCH down | 1 039 | 0.60 % | 6.01 |
| FVG bullish (signal post 0.1 ATR) | 14 623 | 8.46 % | 84.6 |
| FVG bearish (signal post 0.1 ATR) | 12 985 | 7.51 % | 75.1 |
| FVG direction (avant filtre) | 35 586 | 20.6 % | 205.9 |
| Bullish OB | 21 442 | 12.4 % | 124.1 |
| Bearish OB | 21 178 | 12.3 % | 122.5 |
| Retest armed up (bars) | 27 261 | 15.8 % | 157.7 |
| Retest armed down (bars) | 22 651 | 13.1 % | 131.0 |
| Divergence bullish | 7 776 | 4.50 % | 45.0 |
| Divergence bearish | 8 988 | 5.20 % | 52.0 |
| Up fractals | 23 585 | 13.6 % | 136.4 |
| Down fractals | 23 856 | 13.8 % | 138.0 |

### 4.2 Drill-down par année (XAU)

Stable et cohérent avec eval_03 :

| Année | Bars | BOS up | BOS dn | CHOCH up | CHOCH dn | FVG up | FVG dn | OB bull | OB bear |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2019 | 23 450 | 385 | 325 | 133 | 132 | 2 127 | 1 946 | 2 785 | 2 718 |
| 2020 | 23 611 | 386 | 278 | 128 | 129 | 1 815 | 1 621 | 2 959 | 2 956 |
| 2021 | 23 581 | 386 | 354 | 145 | 144 | 1 995 | 1 827 | 2 859 | 2 863 |
| 2022 | 23 653 | 435 | 415 | 158 | 158 | 1 955 | 1 911 | 2 941 | 2 937 |
| 2023 | 23 558 | 354 | 353 | 142 | 143 | 1 917 | 1 762 | 2 914 | 2 878 |
| 2024 | 23 734 | 395 | 326 | 137 | 136 | 1 962 | 1 721 | 3 035 | 2 905 |
| **2025** | **23 638** | **507** | **340** | **152** | **153** | **2 175** | **1 683** | **3 018** | **2 933** |
| 2026 (YTD) | 7 624 | 122 | 100 | 44 | 44 | 677 | 514 | 931 | 988 |

**Lecture** :
- 2025 a un ratio BOS up/down = 1.49 (vs ~1.1 moyen) — alignement avec le rally
  XAU 2025 documenté. Sain comportement.
- Asymétrie FVG (up > down) cohérente avec drift haussier multi-annuel.
- OB ~3 000/an chaque côté → **8.2 par jour de trading** → reste massif et
  inexploitable sans filtre additionnel (cf. P0-F1).

### 4.3 EURUSD 20 000 bars (cross-asset)

`section_3_2_smart_money_stats.json#eur_20k.stats` :

| Métrique | XAU 20 k | EUR 20 k | Delta |
|---|---:|---:|---:|
| BOS event % | 3.13 % | 2.96 % | -5 % |
| FVG signal % | 17.6 % | 15.6 % | -11 % |
| OB % | 23.3 % | 24.7 % | +6 % |
| Retest armed % | 27.3 % | 26.3 % | -4 % |

→ Statistiques cohérentes entre actifs. La logique est **ATR-relative** (FVG_SIZE_NORM,
RETEST_TOL_ATR, OB_STRENGTH_NORM) donc se généralise. **Verdict cross-asset : 8/10.**

---

## 5. Sensibilité aux paramètres (configurabilité)

### 5.1 Paramètres configurables (via `SMCConfig`)

| Paramètre | Default | Range Pydantic | Audit |
|---|---:|---|---|
| RSI_WINDOW | 14 | [5, 21] | OK |
| MACD_FAST / SLOW / SIGNAL | 12/26/9 | ≥ 5/10/5 | OK (standard Appel) |
| BB_WINDOW | 20 | ≥ 10 | OK |
| ATR_WINDOW | 14 | [5, 21] | OK (Wilder) |
| FRACTAL_WINDOW | 2 | ≥ 2 | OK |
| FVG_THRESHOLD | 0.1 | ≥ 0.0 | **trop bas, voir P1-F1** |
| OB_REQUIRE_FVG | False | bool | sain en off |
| OB_FVG_BONUS | 0.2 | [0.0, 1.0] | OK |
| RETEST_TOL_ATR | 0.5 | ≥ 0.0 | **trop haut, voir P1-F4** |
| RETEST_INVALID_TOL_ATR | 1.0 | ≥ 0.0 | OK |
| RETEST_AWAITING_TIMEOUT | 20 | ≥ 1 | OK |
| RETEST_ARMED_WINDOW | 30 | ≥ 1 | **mismatch fast path P1-F2** |

### 5.2 Magic numbers résiduels (hardcodés)

| Constante | Valeur | Localisation | Sévérité |
|---|---:|---|:---:|
| `min(50, n)` warm-up loop pour BOS | 50 | `strategy_features.py:77, 181` | P2 — devrait être paramétré ou ≥ `FRACTAL_WINDOW * 10` |
| `lookback=5` divergence RSI | 5 | `strategy_features.py:817` | P2 — signature par défaut, jamais surchargé |
| `analyze()` drop NaN sur RSI/MACD/ATR | hard list | `strategy_features.py:917` | P2 — silencieusement drop ~25 premières bars |
| Logger format `%(asctime)s ...` | global | `strategy_features.py:24` | P2 — devrait être configuré au niveau app |

**Verdict configurabilité : 6.5/10.** Le gros est dans `SMCConfig`, mais les 4 magic
numbers ci-dessus + le P1-F2 mismatch sont des risques régression silencieux.

### 5.3 Fractal window sweep (sensibilité)

| FRACTAL_WINDOW | Up fractals | BOS events |
|---:|---:|---:|
| 2 (default) | 1 378 | 370 |
| 3 | 1 005 | 292 |
| 4 | 790 | 242 |
| 5 | 649 | 212 |

→ Plage saine. Sprint 4 (calibration) pourra balayer N ∈ {2,3,4} avec coût quasi
nul.

---

## 6. Performance — bench empirique

Source : `section_3_2_smart_money_stats.json#xau_full.timing` (Numba absent).

| Pipeline step | 172 849 bars (Python fallback) | Per-bar |
|---|---:|---:|
| TA indicators (RSI, MACD, BB, ATR) | 4.20 s | 24 µs |
| SMC features (fractals + FVG) | 0.28 s | 1.6 µs |
| Structure (BOS/CHOCH + retest) | 4.71 s | 27 µs |
| Cleaning (dropna) | 0.12 s | 0.7 µs |
| Divergence RSI (Python loop) | implicit in TA delta | ~5 ms (estimation) |
| **TOTAL** | **12.12 s** | **70 µs** |

**Verdict performance** : 12 s pour le scan complet 6 ans est OK en backtest, mais
**hors-cible pour scan live** :
- Scanner cible 30 s SLA (eval_09) → 12 s sur l'historique total est marginal mais
  acceptable si Numba présent (~0.3 s d'après eval_03).
- Latence par bar incrémental n'est pas mesurée — voir P2-F1.

**Finding (P2-F1)** : aucune mesure incrémentale. La classe recompute **tout** à
chaque appel `analyze()`. Le scanner doit donc soit (a) re-compute sur la fenêtre
glissante (coût), soit (b) maintenir un cache (absent). À adresser au refactor
Sprint 1.

### 6.1 Risque Numba prod

| Env | NUMBA_AVAILABLE | Total time / 172 k bars |
|---|:---:|---:|
| Sprint 0 audit (local Windows) | False | 12.1 s |
| eval_03 (avril 2026, Linux) | True | ~0.5 s (extrapolé) |
| Prod Railway (à vérifier) | ? | ? |

→ Sprint 1 doit ajouter une assertion `NUMBA_AVAILABLE` au démarrage en prod ou
fallback silencieux toléré avec alerte. Voir reco Sprint 1.

---

## 7. Couverture tests existants

### 7.1 Tests présents

| Fichier | Lignes | Couverture |
|---|---:|---|
| `tests/test_bos_no_repeated_fire.py` | 109 | 3 tests régression 100 %-firing bug |
| `tests/test_bos_retest.py` | 254 | 8 tests state machine (arming, invalidation, timeout, override, numba parity) |
| `tests/test_data_quality_bos_regression.py` | 100 | 3 tests firing rate [0.5, 10] % + config CSV |
| `tests/test_sprint2_choch_reset.py` | (40+) | CHOCH structure reset à fractal swing level |
| `tests/test_sprint3_order_blocks.py` | 205 | OB FVG requirement + strength + columns |
| `tests/test_sprint5_fvg_threshold.py` | 153 | Threshold filtering + signal quality |
| `tests/test_sprint6_indicator_periods.py` | n/a | TA indicator config |
| `tests/test_sprint7_rsi_divergence.py` | (50+) | Présence colonne, range, integration ConfluenceDetector |

Total estimé : **~150 assertions** sur les détecteurs SMC. Pour comparaison, le
domaine Numba calc ≈ 1500 LOC + 8 détecteurs.

### 7.2 Gaps de couverture

| Gap | Sévérité |
|---|:---:|
| Aucun test ne vérifie que `BULLISH_OB_HIGH` est anchored à un BOS_EVENT précédent (P0-F1) | **bloquant** |
| `test_sprint7_rsi_divergence.py` ne vérifie pas que la divergence compare bien le **swing low** et non le **bar de confirmation** (P1-F3) | bloquant |
| Pas de test sur le mismatch armed_window 5 vs 30 (P1-F2) | bloquant |
| Pas de test Numba parity pour `_calculate_bos_choch_numba` vs `_calculate_bos_choch_python` (équivalent du test retest) | important |
| Pas de test sur le drop silencieux des 25 premières bars par `dropna(['RSI', 'MACD_line', 'ATR'])` (peut surprendre downstream) | low |
| Aucun test sur la latence incrémentale (P2-F1) | low |
| Pas de test cross-instrument (XAU vs EUR vs JPY) | medium |

**Verdict tests : 6.0/10.** Le périmètre couvre les fonctions, mais pas les
contrats sémantiques ICT.

---

## 8. Findings priorisés

### P0 (bloquant Sprint 1)

| ID | Finding | Référence | Impact |
|---|---|---|---|
| **P0-F1** | OB engulfing ≠ Order Block ICT (pas d'anchor BOS, pas de filtre impulse, pas de mitigation) | `strategy_features.py:766-789` | 42 620 OB sur 6 ans dont 40 % sans BOS proche → signal ≈ bruit |
| P0-F2 | (placeholder, vide en Sprint 0) | — | — |

### P1 (bloquant Sprint 2 / 4)

| ID | Finding | Référence | Impact |
|---|---|---|---|
| **P1-F1** | FVG_THRESHOLD=0.1 ATR ≈ spread broker, laisse passer micro-gaps | `strategy_features.py:478-482` + `audits/2026-Q2/section_3_2_smart_money_stats.json#xau_sensitivity.fvg_threshold_sweep` | 22 548 FVG / 6 ans → bruit |
| **P1-F2** | `calculate_bos_retest_fast` default `armed_window=5` incohérent avec `SMCConfig.RETEST_ARMED_WINDOW=30` | `strategy_features.py:420` vs `:510-518` | Risque régression silencieuse si fonction appelée sans `SMCConfig` |
| **P1-F3** | Divergence RSI compare `lows[i]` (bar de confirmation) et `rsi[i]` au lieu de `down_fractals[i]` (swing low) et `rsi[i-N]` (RSI au swing) | `strategy_features.py:849-857, 865-873` | Décalage moyen 1.18 $ sur XAU → divergences potentiellement fausses positives |
| **P1-F4** | RETEST_TOL_ATR=0.5 ≈ 1.5 $ sur XAU ≈ spread broker | `strategy_features.py:493-498` + sweep retest_tol | 89.8 % retest rate (eval_03) → faux confirmes |

### P2 (à traiter en Sprint 3-4)

| ID | Finding | Référence | Impact |
|---|---|---|---|
| P2-F1 | Pas de pipeline incrémental — `analyze()` recompute tout à chaque appel | `strategy_features.py:881-950` | Latence prod si live scan |
| P2-F2 | Magic number `min(50, n)` warm-up pour BOS | `:77, :181` | Couplage logique caché ; non-paramétrable |
| P2-F3 | `lookback=5` divergence jamais surchargé | `:817` | Couverture configurabilité |
| P2-F4 | `dropna(subset=['RSI','MACD_line','ATR'])` silencieux | `:917` | ~25 lignes droppées sans log |
| P2-F5 | `==` strict sur `high == rolling_max` peut rater plateaux | `:622-630` | <0.01 % cas, négligeable |
| P2-F6 | Aucun cache de bar incrémental pour scanner live | absence | Re-computation systématique |
| P2-F7 | Logger format global hardcodé | `:24` | Conflit avec setup app |
| P2-F8 | Pas de test Numba parity sur BOS/CHOCH (uniquement retest) | absence | Couverture |

---

## 9. Tableau récap composant × score × prio

| Composant | Conformité ICT | Stats activation | Tests | Perf | Score 0-10 | Prio refactor |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Fractals | 9 | 9 | 7 | 9 | **8.5** | P3 (stable) |
| FVG core | 9 | 7 (seuil) | 7 | 9 | **8.0** | P1 (seuil) |
| BOS event | 9 | 9 | 8 | 9 | **8.7** | P3 |
| CHOCH | 9 | 9 | 7 | 9 | **8.5** | P3 |
| Retest state machine | 7 | 7 | 8 | 9 | **7.7** | P1 (tol + defaults) |
| **Order Blocks** | **3** | **3** | **4** | **9** | **4.7** | **P0** |
| RSI Divergence | 5 | 7 | 6 | 8 | **6.5** | P1 (bug index) |
| Pipeline analyze() | — | — | 6 | 5 (sans Numba) | **6.0** | P2 |
| **Section 3.2 globale** | — | — | — | — | **6.0** | **P0 OB + P1 retest/FVG/divergence/defaults** |

---

## 10. Recommandations actionnables

### Sprint 1 — Extraction du module `src/intelligence/smart_money/`

**Objectif** : isoler la logique SMC du legacy environment, fixer les P0 et P1.

| # | Action | Effort | Effet attendu |
|---|---|:---:|---|
| 1 | Créer `src/intelligence/smart_money/{__init__,fractals,fvg,bos_choch,order_blocks,retest,divergence}.py` — extraction depuis `strategy_features.py`. **Pas de re-écriture sémantique**, juste split + tests. | M | Maintenabilité, base de calibration future |
| 2 | **Fix P0-F1** : ré-écrire `_add_smc_order_blocks` avec anchor BOS. Pseudo-code : pour chaque `bos_event=±1` en bar `i`, chercher la **dernière** bar opposée (close < open pour BOS_UP) dans `[i-N, i-1]` avec `body / ATR ≥ 1.0` → c'est l'OB | L | Réduction ~95 % OB count, élimination signal-bruit |
| 3 | **Fix P1-F2** : aligner `calculate_bos_retest_fast` default `armed_window=5` → `30`, OU forcer `**kwargs` requis. Casser tests qui ne le passent pas (audit) | XS | Élimination régression silencieuse |
| 4 | **Fix P1-F3** : Divergence RSI utilise `down_fractals[i]` et `rsi[i-N]` au lieu de `lows[i]` et `rsi[i]`. Ajouter test couverture. | S | Élimine biais d'indexage |
| 5 | **Fix P1-F4** : Durcir retest — requiert `low ∈ [level - 0.25 ATR, level + 0.25 ATR]` (touch strict), plus retour au-dessus du level dans la même bar | S | armed rate -50 %, WR armed attendue +5-10 pts |
| 6 | Assertion `NUMBA_AVAILABLE` au démarrage `src/intelligence/main.py` + log warning si fallback | XS | Eviter dérive latence prod |
| 7 | Sortir `_detect_rsi_divergence` du Python pure loop → Numba JIT (gain perf 50-100×) | M | Aligner avec le reste, surtout en live scan |

**Budget Sprint 1** : ~25-40 h dev + 8 h review/test (split modular est le gros effort).

### Sprint 2 — Annotations expertes + F1 scoring

| # | Action | Effort | Effet attendu |
|---|---|:---:|---|
| 8 | Constituer dataset annoté manuel : 200-500 bars XAU + 200 EURUSD avec OB/FVG/BOS labellés par un expert ICT (Loukmane) | M | Ground truth |
| 9 | Implémenter `scripts/eval_smart_money_vs_annotations.py` — calcule precision/recall/F1 par détecteur | S | Métriques objectives |
| 10 | Benchmark vs `smc-python` (Joshyattridge) open-source — confirmer parité ou supériorité | S | Argument compétitif |

**Budget Sprint 2** : ~16 h annotation + 8 h dev + 4 h review.

### Sprint 4 — Calibration paramètres

| # | Action | Effort | Effet attendu |
|---|---|:---:|---|
| 11 | Sweep grid avec gates CPCV+DSR : `FVG_THRESHOLD ∈ {0.3, 0.4, 0.5}`, `RETEST_TOL_ATR ∈ {0.15, 0.25, 0.35}`, `OB body/ATR ∈ {0.8, 1.0, 1.5}` | M | Calibration data-driven |
| 12 | Ajouter colonne `mitigation_count` (combien de fois l'OB a été touché et tenu) — actif visuel commercial | S | Différenciation LuxAlgo |
| 13 | Liquidity sweep detector (equal highs/lows ± 0.1 ATR + balayage) | M | Différenciation eval_03 §8 |

**Budget Sprint 4** : ~16 h dev + 4 h analyse.

---

## 11. Ce que cet audit NE couvre PAS

| Hors périmètre | Raison | Sprint cible |
|---|---|:---:|
| **F1 vs annotations expertes** | Dataset annoté inexistant en Sprint 0 (décision tranchée brief) | Sprint 2 |
| **Compétiteur benchmark** : smc-python / LuxAlgo head-to-head sur les mêmes 6 ans | Hors scope code-path audit | Sprint 2-3 |
| **Backtest PnL** des signaux *armed* (PF, Sharpe, WR) | déjà fait dans eval_03 (PF 0.94, WR 31.9 %) — pas re-mesuré ici | déjà fait |
| **Latence live** sur scanner Railway (à mesurer en prod réelle) | env audit local pas représentatif | Sprint 1 fin |
| **Walk-forward / CPCV** sur le scoring confluence | Pilier 1 implementé hors strategy_features (`src/research/strategy_gates.py`) | déjà testé eval_3pillars |
| **Mitigation tracking** (chart renderer) | absent du code à auditer | Sprint 4 reco #12 |
| **Visualisation Telegram / API** | absent du code à auditer | Sprint 4 reco #12 |
| **Comparaison versions Numba** (vieille / récente) | Numba absent dans env Sprint 0 — parité Numba/Python validée mais pas de cross-version | infrastructure Sprint 1 |
| **Drift temporel** des paramètres optimaux (paramètre stable 2019 → 2026 ?) | Sprint 4 calibration | Sprint 4 |
| **Multi-timeframe alignment** (M15 SMC consistent avec H1 SMC ?) | spec multi-TF dans memory mais hors `strategy_features.py` | Sprint 5 |

---

## 12. Conclusion

Le « Smart Money Engine » est **techniquement solide côté implémentation** (reproductible,
Numba accéléré, parité fallback validée, tests régressions multiples, fix Sprint 1 du
bug 100 %-firing acquise) **mais sémantiquement incomplet côté ICT** :

- Le détecteur Order Block est une simple bougie engulfing (P0-F1, eval_03 déjà signalé).
- Le seuil FVG laisse passer des micro-gaps de la taille du spread broker (P1-F1).
- La tolérance retest 0.5 ATR autorise des faux confirmes (P1-F4).
- La divergence RSI compare des prix décalés (P1-F3 — finding nouveau).
- Le default `armed_window` du wrapper public diverge silencieusement de `SMCConfig`
  (P1-F2 — finding nouveau).

**Note globale 6.0/10** — relevable à 8.0+/10 en Sprint 1 si les 5 fixes ci-dessus sont
appliqués avant la rebaseline. À 8.5+/10 après Sprint 2 (annotations + F1) et Sprint 4
(calibration).

**Recommandation forte** : avant toute promesse commerciale (B2B-API ou B2C),
*matérialiser* les fixes P0-F1 et P1-F1/F4. Sans cela, n'importe quel client
SMC-savvy reconnaîtra que les OB ne sont pas des OB et challengera le différenciateur
« Smart Money » du produit.

---

### Annexes

- **A. Stats JSON** : `audits/2026-Q2/section_3_2_smart_money_stats.json` (généré).
- **B. Script audit** : `scripts/audit_3_2_smart_money.py` (lecture-seule, reproductible).
- **C. Rapports liés** : `reports/eval_03/eval_03_smart_money.md` (4.5/10, avril 2026) ;
  `audits/2026-Q2/sprint_0_decisions.md` (décision A, CSV primaire).
- **D. Tests pertinents** : `tests/test_bos_no_repeated_fire.py`,
  `tests/test_bos_retest.py`, `tests/test_data_quality_bos_regression.py`,
  `tests/test_sprint2_choch_reset.py`, `tests/test_sprint3_order_blocks.py`,
  `tests/test_sprint5_fvg_threshold.py`, `tests/test_sprint7_rsi_divergence.py`.
- **E. Données auditées** : `data/XAU_15MIN_2019_2026.csv` (98.72 % coverage),
  `data/EURUSD_15MIN_2019_2025.csv` (99.41 %).
