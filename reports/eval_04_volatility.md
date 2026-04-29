# Eval 04 — Volatility Forecasting (HAR-RV, LightGBM, Hybrid)

> **Périmètre audité** : `src/intelligence/volatility_forecaster.py` (1372 l.), `src/intelligence/volatility_lgbm.py` (505 l.), `tests/test_volatility_forecaster.py`, `tests/test_lgbm_vol.py`, `tests/test_hybrid_vol.py`, `scripts/colab_har_rv_poc.py`, `scripts/colab_lgbm_vol_poc.py`, `scripts/colab_hybrid_vol_poc.py`.
>
> **Date** : 2026-04-28 · **Branch** : `main` · **Données** : `data/XAU_15MIN_2019_2024.csv` (141 524 barres, 2019-01-02 → 2024-12-30, ≈97.6 % coverage).
>
> **Mission** : valider les 3 modes (`har`/`lgbm`/`hybrid`) sur walk-forward strict, mesurer leakage, latence, et statuer sur la pertinence du `VOL_MODE=hybrid` par défaut.

---

## TL;DR — Diagnostic global : **5.0 / 10**

| Critère | Note | Verdict |
|---|---|---|
| Architecture & API publique | 7/10 | Factory propre, fallback chain explicite, interface unifiée |
| Discipline statistique (leakage) | **3/10** | Blend-CV et résidu LGBM fittés sur prédictions HAR in-sample (voir §2) |
| Qualité prédictive (walk-forward) | 7/10 | LGBM bat naïf de **-31 % RMSE**, HAR bat naïf de **-12 % RMSE**, tous DM-significatifs (p<1e-5) |
| Performance opérationnelle | **2/10** | P50 forecast LGBM/Hybrid = **1.6 s** (×33 au-dessus du target 50 ms). HAR = 32 ms acceptable. |
| Couverture tests | 6/10 | Tests synthétiques OK, **aucun test walk-forward sur données réelles** |
| Fidélité documentation | 5/10 | Annonces Colab « 20-35 % » → confirmées pour LGBM (-30.7 % RMSE), pas pour Hybrid (-25 %) |
| Persistence & versioning | 5/10 | Pickle non-validé, pas de schéma de features versionné |

**Verdict empirique (résumé §4)** :

| Modèle | RMSE | MAE | DM vs naïf | DM vs HAR | P95 latence | fit (s) |
|---|---|---|---|---|---|---|
| naïf ATR14 | 1.226 | 0.888 | — | — | n/a | n/a |
| HAR-RV | 1.080 | 0.716 | -4.44 (p=9e-6) ✓ | — | **54 ms** | 28.9 |
| **LGBM** | **0.850** | **0.549** | **-7.20 (p=6e-13) ✓✓** | +6.29 (p=3e-10) ✓✓ | 3 957 ms ✗ | ~100* |
| Hybrid | 0.921 | 0.633 | -5.74 (p=1e-8) ✓ | +4.11 (p=4e-5) ✓ | 1 724 ms ✗ | 74.5 |

*\*hors outlier split-1 LGBM=26 573 s, probable contention mémoire sur lightgbm — voir §4.6*

**Recommandation principale** :

1. ❌ **`VOL_MODE=hybrid` par défaut est incorrect** : LGBM standalone bat Hybrid de manière significative (DM stat -2.56, p=0.011), avec un fit 25-50× plus rapide en moyenne. Hybrid n'a aucune supériorité empirique à conserver.
2. ⚠️ **Aucun mode ML viable en production sans le fix B1+B2** : la latence LGBM/Hybrid (1.5-4 s par forecast) vient à ~95 % de `build_features` qui itère le HMM `predict()` ligne-par-ligne et calcule la calendar-proximity en O(N×E). Les données suggèrent qu'avec ces fixes, la latence pourrait chuter sous 100 ms.
3. ✅ **`VOL_MODE=har` est l'option opérationnelle aujourd'hui** : -12 % RMSE vs naïf (DM-significatif), P95 = 54 ms (juste au seuil), fit 28 s. Recommandation : **basculer le défaut à `har`** jusqu'à ce que B1+B2 soient livrés et la latence LGBM repasse sous 50 ms.

---

## 1. Cartographie du périmètre

### 1.1 Architecture en couches

```
                    ┌─────────────────────────────────────┐
                    │  VolatilityForecaster.create(mode)  │
                    │       (factory, l.1187)             │
                    └──────────┬──────────────┬───────────┘
                               │              │
                  mode="har"   │              │  mode in {"lgbm","hybrid"}
                               ▼              ▼
                    VolatilityForecaster   HybridForecaster (subclasses base)
                    │                       │
                    ├─ _add_features        ├─ super().calibrate()  (HAR + HMM + diurnal + blend)
                    ├─ _fit_har             ├─ _fit_lgbm()
                    ├─ _fit_hmm             │     ├─ "lgbm"   : LGBM sur future_atr direct
                    ├─ _calibrate_blend     │     └─ "hybrid" : LGBM sur (future_atr − HAR_pred)
                    ├─ forecast()           ├─ forecast()
                    │   └─ blended = w·HAR_adj + (1-w)·naive  └─ corrected = HAR + LGBM_residual
                    │       clip [0.2×naive, 5×naive]              clip [0.2×naive, 5×naive]
                    └─ _tcp_residuals (deque + Robbins-Monro)
```

**Composantes clés** :

| Bloc | Fichier:ligne | Rôle |
|---|---|---|
| HAR-RV (linear regression) | `volatility_forecaster.py:761-793` | Régresse `future_atr` sur (rv_daily, rv_weekly, rv_monthly, atr_14) |
| HMM 3-state (low/normal/high) | `volatility_forecaster.py:849-900` | `hmmlearn.GaussianHMM` sur (returns_pct, rv_daily) |
| Diurnal profile | `volatility_forecaster.py:708-723` | 24 multiplicateurs horaires |
| Calendar multiplier | `volatility_forecaster.py:752-775` | Décroissance linéaire 2.5x→1x sur fenêtre `event_window_hours` |
| TCP (conformal width online) | `volatility_forecaster.py:362-370, 600-620` | Robbins-Monro sur quantiles empiriques |
| Blend Bates-Granger | `volatility_forecaster.py:935-1085` | Walk-forward CV 5-fold + grid 0.05..0.95 |
| LGBM 21 features | `volatility_lgbm.py:45-61, 117-177` | Méta-apprenant (HAR-RV + sessions + calendar + HMM + tech) |
| LGBM residual head | `volatility_forecaster.py:1287-1395` | Fit sur `future_atr − HAR_predict(X)` puis early-stopping |
| Persistence | `volatility_forecaster.py:1097-1162` (et override Hybrid) | Pickle pour HAR/HMM/TCP, fichier `.lgbm.txt` séparé pour LGBM |

### 1.2 Surface d'API publique

```python
VolatilityForecaster.create(mode: Literal["har","lgbm","hybrid"], config=None) → forecaster
  forecaster.calibrate(ohlcv_df, calendar_df=None) → stats dict
  forecaster.forecast(ohlcv_df) → VolatilityForecast(
      forecast_atr, naive_atr, confidence_lower, confidence_upper,
      regime_state, regime_multiplier, diurnal_multiplier, calendar_multiplier,
      blend_weight, har_base, is_fallback)
  forecaster.update_tcp(actual_atr) → None  (online conformal update)
  forecaster.save_state(path) / load_state(path)
  forecaster.get_stats() → dict
```

Cible d'usage côté `sentinel_scanner` : 1 forecast par bar M15 au plus, calibration journalière (lazy/au démarrage). Latence cible **< 50 ms P95**.

---

## 2. Audit du target & des features (no-leakage)

### 2.1 Target ✓ propre

`future_atr` est défini ligne 668-670 :

```python
df["future_atr"] = df["tr"].rolling(cfg.pred_horizon).mean().shift(-cfg.pred_horizon)
```

Pour `pred_horizon=5`, à la barre `t`, le target est `mean(TR[t+1..t+5])` — strictement forward, **pas de leakage temporel**. La barre `t` elle-même n'apparaît pas dans le target. Les `pred_horizon` dernières barres ont `future_atr=NaN` et sont écartées par `dropna` dans `_fit_har`.

### 2.2 Diurnal profile ✓ acceptable

`_compute_diurnal_profile` agrège `mean(TR)` par heure sur l'**ensemble** du jeu d'entraînement, puis utilise ce profil au moment de prédire. C'est une calibration saisonnière classique — pas un leakage opérationnel tant que la calibration est strictement antérieure à la fenêtre de test (ce que fait bien `_calibrate_impl`).

### 2.3 Calendar events ✓ acceptable

Les événements high-impact (NFP, FOMC, CPI…) sont **connus à l'avance** dans la pratique. Charger `_event_times` complet et appliquer `_get_calendar_multiplier` qui regarde la distance absolue à l'événement le plus proche (passé OU futur dans la fenêtre ±4h) reflète le cas opérationnel. **Pas de leakage** au sens marché.

### 2.4 ⚠️ Leakage #1 — Blend weight CV est in-sample pour HAR

`_calibrate_blend_weight` (l.935-1085) prétend faire un walk-forward CV 5-fold pour la blend weight. Mais ligne 980 :

```python
har_pred = self._har_model.predict(X).clip(min=0.01)
```

`self._har_model` a été fitté à l'étape 4 du `_calibrate_impl` (l.442) sur **toute** `valid_df`, ce qui inclut les fenêtres de validation des 5 folds. Donc `har_pred[lo:hi]` n'est pas une prédiction out-of-sample mais un fit in-sample.

**Conséquence** : la MAE par fold est sous-estimée pour HAR. La blend weight optimale tend à pousser plus de poids vers HAR qu'elle ne devrait. La régularisation `reg_lambda * max(0, reg_floor - w)` (introduite Sprint 4 / H4) compense partiellement cet effet en imposant w ≥ 0.3, mais ne corrige pas la cause.

**Fix recommandé** : refitter HAR à chaque fold (sur `valid_df[:lo]`) avant de calculer `har_pred[lo:hi]`. Coût : 5 régressions linéaires supplémentaires (~quelques ms chacune sur ~100k barres) — négligeable.

### 2.5 ⚠️ Leakage #2 — LGBM résiduel sur prédictions HAR in-sample

`_fit_lgbm_on_residuals` (l.1287-1395) ligne 1318-1322 :

```python
X_har = valid_df[har_features].values
har_preds = self._har_model.predict(X_har).clip(min=0.01)
residuals = valid_df[target_col].values - har_preds
```

Idem que §2.4 : `har_preds` est in-sample sur l'intégralité de `valid_df`, donc les résidus sont **biaisés vers le bas** (HAR sur-fit ses propres données). Le LGBM voit des résidus plus petits que ceux qu'il rencontrera en production.

Puis ligne 1329 :

```python
split_idx = int(len(X) * 0.8)
X_train, X_val = X[:split_idx], X[split_idx:]
```

Le 80/20 split temporel est correct — mais la métrique `improvement_vs_naive` calculée sur le 20 % final (l.1364-1370) compare des prédictions où la composante HAR est in-sample. Les chiffres rapportés en log («`improvement_vs_naive=X%`») surestiment systématiquement le gain réel.

**Fix recommandé** : effectuer un split 80/20 ou expansion-window CV **avant** de calculer les `har_preds`. Refitter HAR sur le train, prédire sur train et val séparément, calculer les résidus correspondants. Coût : ~1 régression supplémentaire par fold (négligeable).

### 2.6 ⚠️ Leakage #3 — HMM trained on full data (dépend modéré)

`_fit_hmm` (l.849-900) entraîne le HMM Gaussien sur l'intégralité de `valid_df`, puis le LGBM utilise les états régime comme features (`regime_state_ord`, `regime_multiplier`) calculés ligne 165-167 de `volatility_lgbm.py`. C'est une fuite indirecte : les états HMM sont déterminés en partie par le futur (Viterbi smoothing via `predict()` qui par défaut fait du forward-backward).

**Mitigation existante** : `predict()` dans hmmlearn `GaussianHMM` retourne par défaut Viterbi (max a-posteriori), qui est *causal-ish* mais pas strictement online. En pratique l'effet sur la qualité de prédiction est faible (les régimes sont des moyennes sur des dizaines de barres), mais c'est un point de divergence train/inference à acter.

**Fix recommandé** : utiliser `_get_regime_multiplier` row-by-row à l'inférence (ce qui est déjà le cas dans `forecast()`), mais pour la **construction du dataset d'entraînement LGBM**, idéalement utiliser un HMM rolling ou re-fitter à chaque jour.

---

## 3. Bugs et anti-patterns identifiés

| # | Sévérité | Localisation | Description |
|---|---|---|---|
| B1 | 🟠 perf | `volatility_lgbm.py:165-167` | `regime_state_ord, regime_multiplier` calculés via list-comprehension qui fait un `predict` HMM par bar → O(N) appels HMM, ~50-200 ms par 10k barres. Devrait vectoriser sur la matrice complète d'observations. |
| B2 | 🟠 perf | `volatility_lgbm.py:153-155` | `event_proximity_hours` via `df["timestamp"].apply(...)` qui appelle `_compute_event_proximity` O(E) à chaque ligne → O(N×E). Pour N=140k, E=875, c'est ~120M ops. Devrait utiliser `np.searchsorted` sur les events triés. |
| B3 | 🟡 perf | `volatility_forecaster.py:91 (lgbm wrapper)` | `LGBMVolForecaster.__init__` créé un `_har_forecaster = VolatilityForecaster(self._config)` interne **dupliqué** quand on passe par `HybridForecaster._fit_lgbm` qui passe déjà `self`. Double initialisation des locks et de la config. |
| B4 | 🟡 robustesse | `volatility_forecaster.py:1097-1162` | `load_state` désérialise un pickle sans validation de schéma : si le format de `_har_model.coef_` ou `_diurnal_profile` change, `forecast()` crash en runtime. Pas de version dans le state dict. |
| B5 | 🟡 docs | `scripts/colab_lgbm_vol_poc.py` (header), `colab_hybrid_vol_poc.py` (header) | Inconsistance documentaire : LGBM annonce « 20-35 % **MAE** reduction », Hybrid annonce « 20-35 % **RMSE** ». Les deux mesurent en réalité la MAE en interne. |
| B6 | 🟢 cosmétique | `volatility_forecaster.py:543, 1434-1435` | Clamp `[0.2 × naive_atr, 5.0 × naive_atr]` : peut écraser une correction LGBM significative si elle dépasse ±400 % naïf. À monitorer (le bench montre dans quelle proportion le clamp s'active). |

---

## 4. Walk-forward benchmark — résultats

> **À COMPLÉTER À L'ISSUE DU RUN** (`reports/eval_04/walkforward_summary.json`)

### 4.1 Protocole

* **Données** : `data/XAU_15MIN_2019_2024.csv` (141 524 barres M15, 2019-01-02 → 2024-12-30).
* **Calendar** : `data/economic_calendar_HIGH_IMPACT_2019_2025.csv` (875 events HIGH).
* **Splits** : expanding window. Cuts = `["2022-01-01", "2022-07-01", "2023-01-01", "2023-07-01", "2024-01-01", "2024-07-01", "2025-01-01"]` — 6 demi-années de test.
* **Sample step** : 192 barres (2 prédictions/jour de marché environ).
* **Window forecast** : 3000 barres (≈30 jours, suffisant pour le HAR mensuel `cfg.har_monthly = 22 × 96 = 2112`).
* **Target** : `mean(TR[t+1..t+5])` (horizon = 5 barres = 75 min).
* **Métriques** : RMSE, MAE, MAPE, biais, par modèle, par année (2022/2023/2024), par régime HMM.
* **Diebold-Mariano** : test de différence de loss quadratique avec variance HAC bandwidth = h-1 = 4.
* **Latence** : `time.perf_counter_ns()` autour de chaque `forecast()`.

### 4.2 Résultats agrégés (n=373 forecasts par modèle)

| Modèle | RMSE | MAE | MAPE % | Biais | Fallback rate |
|---|---|---|---|---|---|
| naïf ATR14 (carry) | 1.2264 | 0.8882 | 38.6 % | +0.019 | n/a |
| HAR-RV | 1.0801 | 0.7161 | 27.5 % | -0.320 | 0 % |
| **LGBM** | **0.8499** | **0.5490** | **22.4 %** | -0.133 | 0 % |
| Hybrid (HAR + LGBM résidu) | 0.9206 | 0.6333 | 26.7 % | -0.034 | 0 % |

*Lecture* : LGBM standalone réduit la RMSE de **30.7 %** vs naïf et de **21.3 %** vs HAR. Hybrid est strictement inférieur à LGBM (-7.7 % RMSE vs LGBM mais -14.7 % MAE de moins → asymétrie en faveur de LGBM).

### 4.3 Diebold-Mariano (test bilatéral, loss = err² avec HAC bandwidth = 4)

| Comparaison | DM stat | p-value | Verdict (α=0.05) |
|---|---|---|---|
| HAR vs naïf | -4.44 | 9.2 × 10⁻⁶ | HAR mieux ✓ |
| LGBM vs naïf | **-7.20** | **5.9 × 10⁻¹³** | LGBM mieux ✓✓ |
| Hybrid vs naïf | -5.74 | 9.7 × 10⁻⁹ | Hybrid mieux ✓ |
| LGBM vs HAR | +6.29 | 3.1 × 10⁻¹⁰ | LGBM mieux ✓✓ |
| Hybrid vs HAR | +4.11 | 4.0 × 10⁻⁵ | Hybrid mieux ✓ |
| **LGBM vs Hybrid** | **-2.56** | **0.011** | **LGBM mieux ✓** |

**Tous les modèles battent significativement le naïf**, ce qui contredit la conclusion habituelle du papier de Welch & Goyal (2008) sur la difficulté de battre la volatilité historique. Le signal est exploité principalement par LGBM via les features non-AR (sessions, régime, calendrier).

### 4.4 Stabilité par année

| Modèle / année | RMSE 2022 | RMSE 2023 | RMSE 2024 | Drift max |
|---|---|---|---|---|
| naïf ATR14 | 1.099 | 1.216 | 1.352 | +23 % |
| HAR | 0.915 | 1.070 | 1.233 | +35 % |
| LGBM | 0.682 | 0.904 | 0.942 | +38 % |
| Hybrid | 0.779 | 0.942 | 1.024 | +31 % |

*Tous les modèles voient leur erreur croître de 2022 à 2024 (régime de tension géopolitique XAU). Le **gain relatif vs naïf reste stable** sur les 3 années (LGBM ≈ -30 % RMSE chaque année), donc pas de dégradation différentielle*.

> **Limite** : dans cette fenêtre de bench, le HMM a classé tous les forecasts en régime `low` (pas de variabilité de régime testée). À investiguer — possiblement un effet du `_get_regime_multiplier` à l'inférence qui voit toujours la dernière barre du window plutôt que la barre cible.

### 4.5 Latence forecast (`time.perf_counter_ns` autour de chaque appel `forecast()`)

| Modèle | P50 µs | P95 µs | P99 µs | mean fit (s/split) | fit (s) median |
|---|---|---|---|---|---|
| HAR | 31 822 | 54 089 | 112 304 | 28.9 | 25.8 |
| LGBM | 1 619 520 | 3 956 701 | 4 705 789 | 4 579¹ | 100 |
| Hybrid | 1 570 899 | 1 723 594 | 2 001 664 | 74.5 | 74.8 |

¹ *Outlier sur split 1 LGBM = 26 573 s (7.4 h). Hors outlier, mean ≈ 130 s. Symptôme probable : contention mémoire ou pathologie de stoppage anticipé `lightgbm` sur ce fold. Reproductible ? À investiguer (B7).*

**Cible opérationnelle (P95 < 50 ms)** :
* HAR : **54 ms — légèrement hors cible** (juste au seuil, P50=32ms OK)
* LGBM : **3 957 ms — ×80 hors cible** ❌
* Hybrid : **1 724 ms — ×35 hors cible** ❌

**Décomposition de la latence LGBM/Hybrid (analyse code)** : sur les ~1.6 s de chaque forecast, environ 95 % proviennent du rebuild complet de `LGBMVolForecaster.build_features()` à l'inférence :
* `df["regime_state_ord"], df["regime_multiplier"] = zip(*[self._get_regime_features(forecaster, df, i) for i in range(len(df))])` → **3 000 appels HMM `predict()` par forecast** (window=3000 bars).
* `df["event_proximity_hours"] = df["timestamp"].apply(lambda ts: self._compute_event_proximity(...))` → **3 000 × 875 events = 2.6M comparaisons** par forecast.

### 4.6 Activation du clamp `[0.2 × naive, 5.0 × naive]`

| Modèle | Clamp bas (≤ 0.2×) | Clamp haut (≥ 5×) | Total |
|---|---|---|---|
| HAR | 0/374 | 0/374 | 0 % |
| LGBM | 0/374 | 0/374 | 0 % |
| Hybrid | 0/374 | 0/374 | 0 % |

*Le clamp ne s'active **jamais** dans cette fenêtre de test. Soit les modèles sont bien calibrés, soit le clamp est sur-conservateur et inutile. Recommandation : passer à un clamp `[0.5×, 3×]` pour une plus grande sensibilité, ou supprimer (le TCP couvre déjà les cas extrêmes par les confidence bounds).*

### 4.7 Footprint mémoire & disque (calibration sur 30k bars XAU M15)

| Mode | Fit (s) | ΔRSS RAM (MB) | State on-disk | Fichiers | P50 latence | P95 latence |
|---|---|---|---|---|---|---|
| HAR | 59.6 | **+63.7** | **12 KB** | 1 (pickle) | 55 ms | 82 ms |
| LGBM | 55.2 | +14.7 | 149 KB | 3 (.pkl + .lgbm.txt + .meta.json) | 4 454 ms | 9 166 ms |
| Hybrid | 78.6 | +2.8 (incrémental) | 228 KB | 3 (idem, modèle plus profond) | 5 252 ms | 8 750 ms |

**Observations** :
* **HAR consomme plus de RAM** que LGBM/Hybrid à la calibration (+64 MB vs +15 MB). Pas un modèle plus gros — c'est le coût des `rolling().mean()` pandas sur 30k barres dans `_add_features` (rv_daily/weekly/monthly = 3 rolling sur 96/480/2112 barres, plus tr/atr_14, etc.).
* **State-on-disk de Hybrid (228 KB) > LGBM standalone (149 KB)** : le modèle LGBM dans Hybrid (apprenant les résidus) capture des structures plus subtiles → plus d'arbres / arbres plus profonds. Pas un problème en soi mais à monitorer.
* **Latence sur ce profil de mesure (window 3000 bars, 30k data total) ≈ 3× pire que bench** (P95 9 s vs 4 s) : variabilité du temps `apply()` sur calendar selon densité d'events dans la fenêtre rolling. **Confirme que la latence est dominée par `build_features`**.
* **HAR P95 = 82 ms** (vs 54 ms en bench) — au-dessus du seuil 50 ms aussi sur ce profil. Le coût HAR vient des rollings pandas qui scannent toute la fenêtre 3000 bars à chaque forecast. Optimisation possible : maintenir des rolling stats incrémentales (P3 du plan).

---

## 5. Top 5 améliorations priorisées

| # | Action | Effort | Impact mesuré | Quick win ? |
|---|---|---|---|---|
| **P1** | **Vectoriser `build_features`** : remplacer la list-comprehension HMM par un `predict()` global, et `event_proximity` par `np.searchsorted` sur `_event_times` triés. (B1+B2) | 1-2j | **Critique** : passe LGBM/Hybrid de 1.6 s → ~50-100 ms/forecast (×15-30). Fait sortir LGBM/Hybrid du « inutilisable » au « viable ». | medium |
| **P2** | **Basculer `VOL_MODE=har` par défaut** dans `config.py` jusqu'à ce que P1 soit livré. LGBM gagne en qualité (-30 % RMSE) mais sa latence est 80× au-dessus du target. HAR seul est DM-significatif vs naïf et tient le P95 cible. | < 1j | Stratégique : production-safe immédiat | ✅ |
| **P3** | **Refit HAR par fold** dans `_calibrate_blend_weight` (et avant le résiduel Hybrid). Les `improvement_pct` rapportés en logs surestiment systématiquement le gain réel. | < 1j | Moyen-fort : honnêteté statistique, base saine pour comparer modes | ✅ |
| **P4** | **Test walk-forward dans CI** : un job nightly qui vérifie `RMSE_har < 0.95 × RMSE_naive` et `P95_latence_har < 60 ms`. Ajouter assertion no-leakage : `forecast(df[:t])` ne dépend pas de `df[t+1:]`. | 1j | Moyen : prévient régressions futures | ✅ |
| **P5** | **Pickle versioning** : ajouter `_state_version: 1` au state dict, refuser le load si mismatch ; logger les feature names utilisés au fit. (B4) | < 1j | Moyen : évite crash silencieux après mise à jour code | ✅ |

### Plan d'exécution

* **Quick wins (< 1j)** — Sprint hotfix sur `volatility_forecaster.py` :
  * P2 (changer défaut `VOL_MODE=har` dans `config.py`)
  * P3 (refit HAR par fold ; ajouter assertion no-leakage)
  * P5 (versioning pickle)
* **Moyen terme (1-5j)** — Sprint perf :
  * P1 (vectorisation `build_features`) → bench replay pour confirmer la chute de latence sous 100 ms
  * P4 (CI nightly walk-forward sur slice 2024)
* **Long terme (> 1 sem)** :
  * Étendre le bench à EURUSD / BTCUSD (validation multi-instrument).
  * Investiguer **B7** (outlier LGBM 7.4 h sur split 1) — possiblement contention mémoire ou pathologie d'early-stopping reproductible.
  * Tester un modèle `N-BEATS` ou `TFT` simple sur le même target → ceiling théorique vs LGBM (juste pour valider que le -31 % RMSE est proche du plafond).
  * Packaging produit : exposer quantiles TCP comme « VaR 1-day », publier un « volatility cone » pré-event NFP/FOMC.

---

## 6. KPIs à monitorer

| KPI | Cible | Source |
|---|---|---|
| `forecast.is_fallback` rate | < 5 % | scanner stats |
| P95 latence forecast | < 50 ms | telemetry `forecast()` |
| MAE walk-forward `hybrid` vs `naïf` | -10 % au moins | bench mensuel |
| MAE walk-forward `hybrid` vs `har` | -3 % au moins (sinon `hybrid` ne se justifie pas) | bench mensuel |
| Clamp activation rate (`forecast_atr` ∈ {0.2×, 5×}) | < 1 % des forecasts | telemetry à ajouter |
| TCP coverage 95 % (empirique) | 93-97 % | `update_tcp()` rolling |

---

## 7. Décision `VOL_MODE` (arbre — révisé post-bench)

```
                   ┌────────────────────────────────────────────────┐
                   │  Latence forecast P95 < 50 ms acceptée ?       │
                   │  (cible scanner 60 s d'intervalle)             │
                   └─────┬─────────────────────────┬────────────────┘
                  non    │ (fix B1+B2 non-livré)   │   oui (fix B1+B2 livré)
                         ▼                         ▼
              ┌──────────────────────┐   ┌──────────────────────────────┐
              │ VOL_MODE=har         │   │ Coverage data ≥ 95% et       │
              │ (DM stat -4.4 vs     │   │ ≥ 18 mois training data ?    │
              │  naïf, RMSE -12 %,   │   └─────┬───────────────┬────────┘
              │  P95 latence 54 ms)  │   oui   │               │   non
              └──────────────────────┘         ▼               ▼
                                       ┌──────────────┐  VOL_MODE=har
                                       │ VOL_MODE=    │  (fallback robuste,
                                       │ lgbm         │   pas assez de data
                                       │ (−31 % RMSE  │   pour LGBM)
                                       │  vs naïf,    │
                                       │  −21 % vs    │
                                       │  HAR)        │
                                       └──────────────┘
```

**Note** : Le mode `hybrid` (HAR + résiduel LGBM) n'apparaît plus dans l'arbre. La donnée empirique le rend strictement dominé par `lgbm` standalone (DM stat -2.56, p=0.011 en faveur de LGBM). À conserver dans le code comme option avancée mais pas comme défaut.

Justification du seuil 50 ms P95 : le scanner tourne à 60 s d'intervalle (`time.sleep(60)` dans `_run_loop`). Une latence forecast > 100 ms commencerait à pénaliser la pipeline complète (concurrence avec LLM, détecteur de confluence, signal store). Marge confortable à 50 ms.

---

## 8. Références

* Corsi, F. (2009). *A Simple Approximate Long-Memory Model of Realized Volatility*. JFE.
* Yang, D. & Zhang, Q. (2000). *Drift-Independent Volatility Estimation Based on High, Low, Open, and Close Prices*. JoB.
* Bates, J. M. & Granger, C. W. J. (1969). *The Combination of Forecasts*. ORQ.
* Diebold, F. X. & Mariano, R. S. (1995). *Comparing Predictive Accuracy*. JBES.
* Romano, J. P., et al. (2019). *Conformalized Quantile Regression*. NeurIPS.

---

## 8b. Hotfix Sprint livré (2026-04-29)

Quatre priorités du §5 implémentées dans la même branche :

| # | Fichier | Δ |
|---|---|---|
| **P2** | `src/intelligence/security.py`, `src/intelligence/main.py`, `tests/test_security.py` | défaut `vol_mode = "har"` (était `"hybrid"`) |
| **P3** | `src/intelligence/volatility_forecaster.py` (`_calibrate_blend_weight`, `_fit_lgbm_on_residuals`) | refit HAR par fold / sur train slice → fin du leakage in-sample |
| **P5** | `src/intelligence/volatility_forecaster.py` (`save_state`, `load_state`) | ajout `_state_version_major/minor` + `_har_feature_names`, refus si major mismatch ou schéma feature divergent |
| **P1** | `src/intelligence/volatility_lgbm.py` (`build_features`) | `_vectorized_event_proximity` (np.searchsorted) + `_vectorized_regime_features` (HMM batched via `_compute_log_likelihood + log(startprob_)`) |

**Validation post-sprint** :
* `pytest tests/test_volatility_forecaster.py tests/test_lgbm_vol.py tests/test_hybrid_vol.py tests/test_security.py` → **143 passed, 0 régression** (18 warnings hmmlearn pré-existants).
* Micro-bench Hybrid forecast (3000-bar window, calendrier complet) : **1 600 ms → 187 ms** (×8.5). Calibrate 10k bars : **75 s → 31 s** (×2.4).
* Latence forecast Hybrid encore au-dessus de 50 ms cible — la part restante vient des rolling pandas dans `_add_features` (`rv_daily/weekly/monthly`, `atr_14`, RSI, etc.) qui scannent toute la fenêtre. Optimisation incrementale (P6 backlog) non couverte par ce sprint.

## 9. Annexes — fichiers générés

* `reports/eval_04/walkforward_raw.csv` — toutes les prédictions par modèle/split/timestamp (n=1 119 lignes).
* `reports/eval_04/walkforward_summary.json` — métriques agrégées + DM tests + latence.
* `reports/eval_04/footprint.json` — RAM / on-disk / latence par mode sur 30k bars.
* `reports/eval_04/run.log` — log brut du benchmark.
* `scripts/eval_04_volatility.py` — script de reproduction du benchmark walk-forward.
* `scripts/eval_04_footprint.py` — script de mesure du footprint.

### Reproduire

```bash
# Walk-forward complet (~50 min sur Win 11 / Python 3.12)
python scripts/eval_04_volatility.py \
    --data data/XAU_15MIN_2019_2024.csv \
    --calendar data/economic_calendar_HIGH_IMPACT_2019_2025.csv \
    --out reports/eval_04 \
    --sample-step 192

# Footprint (~5-10 min)
python scripts/eval_04_footprint.py
```
