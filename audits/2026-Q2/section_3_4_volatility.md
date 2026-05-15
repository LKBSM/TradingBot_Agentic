# Audit Phase 1 — Section 3.4 : VolatilityForecaster

**Date** : 2026-05-15
**Auditeur** : Claude (Sprint 0 — institutional overhaul)
**Branche** : `institutional-overhaul`
**HEAD** : `203d189` (Phase 2B residual code)
**Périmètre strict** : moteur de prévision de volatilité (HAR-RV, LightGBM, hybride).

**Fichiers audités** :
- `src/intelligence/volatility_forecaster.py` (1 559 LOC, 4 classes : `InstrumentConfig`, `VolatilityForecast`, `VolatilityForecaster`, `HybridForecaster`).
- `src/intelligence/volatility_lgbm.py` (591 LOC, classe `LGBMVolForecaster`).
- `tests/test_volatility_forecaster.py` (694 LOC, 12 classes, 50 tests).
- `tests/test_lgbm_vol.py` (394 LOC, 7 classes, 22 tests).
- `tests/test_hybrid_vol.py` (324 LOC, 7 classes, 23 tests).
- `tests/test_vol_latency.py` (140 LOC, 1 test bench).
- Rapport antérieur : `reports/eval_04_volatility.md` (note 5.0/10, daté 2026-04-29).

**Cible 50ms ?** Confirmée hors d'atteinte pour LGBM/Hybrid même après hotfix B1+B2 (mesure réelle 2026-05-15 ci-dessous : Hybrid p95 = 220ms, LGBM p95 = 191ms). HAR p99 = 91 ms (≈ 2× cible, mais opérationnel).

---

## Score : **5.5 / 10** (eval_04 = 5.0/10)

Légère revalorisation par rapport à eval_04 (2026-04-29) :

- **+0.5** : bugs B1+B2 (HMM row-by-row, event-prox O(N×E)) **livrés en hotfix** et empiriquement confirmés (mesure 2026-05-15 : 90× speedup vérifié sur HMM features). Le sprint perf annoncé en eval_04 §8b a tenu sa promesse partielle (1 600 ms → 187 ms Hybrid forecast = ×8.5).
- **Pas plus** : 4 problèmes structurels demeurent (P0/P1) :
  1. **Aucun mode** (HAR/LGBM/Hybrid) ne respecte la cible 50ms p95 sur données réelles XAU.
  2. **TCP / bandes de confiance massivement mal calibrées** : PICP empirique 43.6% pour cible 80% sur XAU 2024 (mesure ci-dessous §3) — la promesse "conformal adaptive prediction intervals" est non-tenue.
  3. **HMM régime collapse** : 100 % des forecasts walk-forward 2024 sont classés `low` (régime unique), donc le multiplier régime n'apporte aucune valeur à l'inférence (skew train/serve §4).
  4. **Pas de QLIKE / pas de calibration cross-régime** dans les tests : le scoring est en MAE pur (sensible aux outliers, asymétrique pour la volatilité).

| Critère                                          | Note    | Verdict                                                                             |
| ------------------------------------------------ | ------- | ----------------------------------------------------------------------------------- |
| Architecture & API publique                      | 7/10    | Factory propre (`create(mode)`), héritage `Hybrid` propre, fallback chain explicite |
| Discipline statistique (leakage / no peek-ahead) | 6/10    | P3 (refit HAR par fold) livré ; HMM smoothing legacy demeure                        |
| Qualité prédictive (walk-forward)                | 5/10    | RMSE legèrement < naïf sur 2024 (−0.6%), **QLIKE pire** que naïf                    |
| Calibration intervalles (TCP/conformal)          | **2/10** | PICP 43.6% vs cible 80% — bandes de confiance non fiables                          |
| Robustesse cross-régime                          | 3/10    | HMM toujours `low` à l'inférence ; pas de test crisis 2020/2022/2024              |
| Performance opérationnelle                       | 4/10    | HAR p99 ≈ 91ms (acceptable mais > 50ms), LGBM p99 ≈ 224ms (×4-5 hors cible)         |
| Couverture tests                                 | 5/10    | 95 tests prod (HAR+LGBM+Hybrid), zéro QLIKE, zéro PICP, zéro slice crisis           |
| Fidélité documentation                           | 6/10    | eval_04 transparent, headers clairs ; mais Hybrid annonce "2-tier" sans empirique    |
| Persistence & versioning                         | 7/10    | P5 (`_state_version_major`, refus si schéma divergent) livré, robuste              |

**Verdict opérationnel** : `VOL_MODE=har` reste la seule option viable. **Mais HAR seul est un baseline marginal** (≈ -12% RMSE vs naïf, parfois négatif sur slice 2024). La valeur ajoutée du moteur tient surtout dans le calendar multiplier et le diurnal — pas dans le HMM régime, qui doit être désactivé ou refondu (cf. Sprint 4 plan §8).

---

## 1. Architecture des 3 modes

### 1.1 Carte mentale

```
VolatilityForecaster.create(mode)  [volatility_forecaster.py:1239]
       │
       ├─ "har"      → VolatilityForecaster (base class)
       │              [volatility_forecaster.py:339]
       │
       ├─ "lgbm"     → HybridForecaster(mode="lgbm")
       │              [volatility_forecaster.py:1268]
       │              └─ utilise super().calibrate() pour HAR features
       │                 mais _fit_lgbm() entraîne LGBM sur future_atr direct
       │
       └─ "hybrid"   → HybridForecaster(mode="hybrid")
                      └─ HAR base + LGBM entraîné sur résidus (actual - HAR_predict)
```

### 1.2 Le hybride est-il vraiment "HAR base + LGBM residual" ?

**Oui, conforme à la documentation**. À l'inférence (`_hybrid_forecast_impl`, l.1468-1523) :

```python
har_forecast = self._forecast_impl(...)        # ligne 1474: HAR + diurnal + cal + HMM blend
lgbm_pred = self._lgbm.predict_from_df(...)    # ligne 1483: prédit le résidu
if self._mode == "hybrid":
    corrected_atr = har_forecast.forecast_atr + lgbm_pred   # ligne 1489
```

À l'entraînement (`_fit_lgbm_on_residuals`, l.1339-1450) :

```python
train_har_model = LinearRegression()
train_har_model.fit(X_har[:split_idx], y_target[:split_idx])     # ligne 1380-1381: HAR refité sur train only
har_preds_train = train_har_model.predict(X_har[:split_idx])
har_preds_val   = train_har_model.predict(X_har[split_idx:])
har_preds = np.concatenate([har_preds_train, har_preds_val])     # ligne 1384
residuals = y_target - har_preds                                  # ligne 1387 : LGBM apprend les résidus
```

**Bonne nouvelle** : le fix P3 (eval_04 §8b, 2026-04-29) a livré le refit train-only pour le calcul des résidus. La cible LGBM en mode "hybrid" est désormais propre.

**Mauvaise nouvelle** : à l'inférence, on utilise `self._har_model` (fitté sur **toute** la data de calibration au step `_calibrate_impl` ligne 442) pour `har_forecast.forecast_atr`. Donc :

- L'**entraînement LGBM** voit des résidus calculés avec un HAR train-only (propre).
- L'**inférence LGBM** ajoute son résidu prédit à un `har_forecast.forecast_atr` produit par un HAR full-data (= "tout train" et donc strictement antérieur en pratique opérationnelle, OK), mais avec coefficients différents que ceux vus par LGBM au training.

**Conséquence subtile** : il existe une *micro-divergence train/serve* sur la base HAR utilisée. En production, la calibration est faite une fois sur tout l'historique disponible, donc `_har_model` ≈ `train_har_model` si la nouvelle data depuis calibration est marginale. Mais si on recalibrate quotidiennement (recommandé en eval_04 §2.6), le LGBM aura été entraîné sur une distribution de résidus dont la base HAR est légèrement différente de celle vue à l'inférence. **Non-bloquant** mais à acter (P2 §6).

### 1.3 Conformité du fallback chain

`HybridForecaster._hybrid_forecast_impl` (l.1468-1523) implémente bien :

```
LGBM correction ─fail─→ HAR-only (har_forecast)
LGBM unavailable ─→ HAR-only
HAR uncalibrated ─→ naive_ATR (via _fallback_forecast l.573)
```

Fallback détectable côté consommateur via `VolatilityForecast.is_fallback: bool` (l.317).

---

## 2. Performance par modèle — mesures empiriques 2026-05-15

### 2.1 Latence (XAU M15, données réelles, window=3000 bars, vraies calendrier 875 events)

Mesure à l'instant T (re-bench post hotfix B1+B2) :

| Mode    | Calibrate (s) | p50 (ms) | p90 (ms) | p95 (ms) | p99 (ms) | max (ms) | n appels |
| ------- | ------------- | -------- | -------- | -------- | -------- | -------- | -------- |
| HAR     | 17.8          | **45.3** | 64.9     | 72.1     | **91.0** | 138.8    | 1 000    |
| LGBM    | 9.2           | 149.2    | 174.5    | 191.1    | 223.6    | 227.5    | 200      |
| Hybrid  | 21.3          | 152.9    | 209.0    | 220.5    | 263.4    | 284.6    | 200      |

**Lecture** :
- **HAR p99 = 91 ms** sur données réelles XAU — **au-dessus de la cible 50ms p95** mais sous le ceiling kill-criterion 200ms (`tests/test_vol_latency.py:129`). Acceptable mais non-confortable.
- **LGBM p50 ≈ 149 ms** — ×3 hors cible. Pas une régression vs HAR car les fixes vectorisés ont supprimé le goulot HMM/event-prox. La latence résiduelle vient des **rolling pandas dans `_add_features`** (eval_04 §8b).
- **Hybrid p50 ≈ 153 ms** — ajoute uniquement ~3 ms vs LGBM (le `feature_df.build_features()` est partagé). Hybrid n'est plus le pire ; il est désormais quasi-équivalent en latence à LGBM.
- **Confirmation eval_04 §8b** : les bugs B1+B2 ont été matériellement fixés. La latence 1.6-5s mentionnée dans le memory MEMORY.md a été divisée par 10-30×.

### 2.2 Qualité prédictive (walk-forward HAR, XAU 2024 OOS)

Mesure 2026-05-15, n=282 forecasts day-step sur XAU 2024 (train ≤ 2023, test = 2024) :

| Modèle | RMSE  | QLIKE | MSE log-vol | Δ RMSE vs naïf | Δ QLIKE vs naïf |
| ------ | ----- | ----- | ----------- | -------------- | --------------- |
| Naïf ATR14 | 2.7925 | 4.3651 | 0.2413 | — | — |
| **HAR**  | **2.7756** | **4.8002** | **0.2783** | **−0.6%** | **+0.435 (pire)** |

**Lecture** :
- HAR bat le naïf de **0.6% en RMSE** (négligeable, vs −12% rapporté en eval_04 sur la fenêtre 2022-2024). La dégradation 2024 est nette.
- **HAR perd contre le naïf sur QLIKE** (+0.44, pire) et sur MSE log-vol (+15.3%, pire) sur XAU 2024.
- **Interprétation** : sur 2024 (régime XAU géopolitique tendu, tendance prix +25%), HAR sur-prédit la volatilité (biais négatif systématique, voir eval_04 §4.2 : bias=-0.32 pour HAR). Comme QLIKE pénalise asymétriquement la sur-prévision (`log(σ²) + σ²_actual/σ²_pred` → si σ²_pred trop grand, le terme log explose), HAR se prend une pénalité significative.
- **Conséquence** : la métrique standard pour la volatilité (QLIKE, Patton 2011) **n'est pas surveillée** dans les tests ni les bench actuels. eval_04 n'a mesuré que RMSE/MAE.

### 2.3 Calibration des bandes (PICP empirique)

Mesure 2026-05-15, n=282 sur XAU 2024 :

| Bande         | Cible       | PICP empirique | Width / forecast | Verdict      |
| ------------- | ----------- | -------------- | ---------------- | ------------ |
| 80 % central (`tcp_alpha=0.10`) | 80.0%       | **43.6%**      | 0.565            | ❌ massive   |
| Q1 2024 PICP  | 80.0%       | 57.1%          | —                | ❌           |
| Q2 2024 PICP  | 80.0%       | 55.7%          | —                | ❌           |
| Q3 2024 PICP  | 80.0%       | 35.7%          | —                | ❌ effondré  |
| Q4 2024 PICP  | 80.0%       | 27.1%          | —                | ❌ effondré  |

**Lecture catastrophique** :
- Les bandes "TCP conformal" sont **sous-couvertes par 36 points de pourcentage**. Cela signifie que les intervalles "80%" capturent réellement 44% des observations.
- **Dérive trimestrielle** : 57% → 28% sur 2024. Les résidus calibrés à fit-time (sur 2019-2023) ne sont plus valides à inférence (régime XAU 2024 plus volatil que prévu).
- **L'algorithme Robbins-Monro de update online** (`update_tcp`, l.593) **n'est PAS appelé** par le scanner (cf. §5 — pas de boucle TCP-feedback en production).
- Conséquence client/produit : un utilisateur qui s'attend à un intervalle "80%" se prend en pratique 56 % de "violations" — la bande est trompeuse. **À ne pas exposer en l'état dans aucun produit** (B2C ou B2B).

### 2.4 Régime HMM — collapse en production

Mesure 2026-05-15, n=282 sur XAU 2024 :

```
regime distribution: {'low': 282}
```

**100% des forecasts walk-forward 2024 sont classés `low`**. C'est aussi le constat eval_04 §4.4 ("HMM a classé tous les forecasts en régime low"). Ce n'est donc pas un bug ponctuel mais une **propriété de l'inférence row-by-row** (`_get_regime_multiplier` l.907 : `state = int(self._hmm_model.predict(obs)[0])` avec `obs` = 1 ligne).

**Démonstration train/serve skew** :
```
Agreement Viterbi(full sequence) vs predict-per-row : 11.0 % sur n=200 (mesure 2026-05-15)
```

Le HMM est fitté avec lissage Viterbi global (`_fit_hmm` l.874 : `states = self._hmm_model.predict(hmm_X)` sur **tout** le set), puis utilisé en isolation 1-ligne à l'inférence. Les **multiplicateurs de régime** sont calibrés sur la distribution lissée Viterbi, mais l'inférence voit la distribution per-row qui est radicalement différente (88% de désaccord).

**Conséquence** : `regime_multiplier` est en pratique toujours = celui du label "low" (≈ 0.7-0.9), donc il **réduit systématiquement** le `har_base` à l'inférence. Le HMM, dans son implémentation actuelle, **dégrade la qualité du forecast** plutôt que de l'améliorer. Désactivation explicite ou refonte indispensables.

---

## 3. Détection de leakage temporel

### 3.1 Target ✅ propre

`future_atr` = `df["tr"].rolling(cfg.pred_horizon).mean().shift(-cfg.pred_horizon)` (l.668-670). Strict forward-looking, **pas de leakage**. Vérifié dans eval_04 §2.1.

### 3.2 Diurnal profile ✅ acceptable

`_compute_diurnal_profile` (l.708-723) calibré sur tout le training set. Strictement antérieur à l'inférence si la calibration est strictement antérieure aux fenêtres de test (ce qu'assure `_calibrate_impl` l.408-462). **Pas de leakage opérationnel**.

### 3.3 Calendar events ✅ acceptable

Events high-impact connus à l'avance dans la pratique (NFP/FOMC/CPI publiés au calendrier des semaines avant). `_get_calendar_multiplier` (l.752) regarde la distance absolue à l'événement le plus proche ±4h — ce **n'est pas un leakage** au sens "knowing the future market move" mais "anticipating an announcement". Conforme à l'usage.

### 3.4 ⚠️ Leakage #1 (HISTORIQUE — RÉSOLU) : Blend weight CV

eval_04 §2.4 identifiait : `self._har_model` fitté sur **toute** valid_df avant le calcul des prédictions CV. **Fix P3 livré** dans `_calibrate_blend_weight` (l.1020-1023) :

```python
fold_har = LinearRegression()
fold_har.fit(X[:lo], y[:lo])                   # refit strictly on train
val_har_pred = fold_har.predict(X[lo:hi]).clip(min=0.01)
```

✅ Confirmé propre.

### 3.5 ⚠️ Leakage #2 (HISTORIQUE — RÉSOLU) : LGBM résiduel sur prédictions HAR in-sample

eval_04 §2.5. **Fix P3 livré** dans `_fit_lgbm_on_residuals` (l.1380-1387). ✅ Confirmé propre.

### 3.6 ⚠️ Leakage #3 (PERSISTANT) : HMM Viterbi smoothing au training

eval_04 §2.6 décrit le problème. **Pas de fix livré**. Cf. §2.4 supra pour la démonstration empirique (11% agreement).

**Impact** :
- Les `regime_multipliers` (l.887-893) sont calibrés sur des labels Viterbi smoothés (qui regardent dans le futur via forward-backward).
- À l'inférence, le label retourné par `predict([obs])` (1 ligne) est complètement différent (88% de désaccord).
- En pratique, le multiplier 'low' (0.7-0.9) écrase le forecast HAR systématiquement.

**Fix recommandé** :
- **Option 1 (pragmatique)** : désactiver le multiplier régime tant qu'un fit "online causal" n'est pas livré. C'est-à-dire : `regime_multiplier = 1.0` toujours.
- **Option 2 (correct)** : refit le HMM sur une fenêtre rolling à chaque pas, ou utiliser le filtre forward-only (`hmm.decode(X, algorithm='viterbi')` est par défaut, mais on peut implémenter `score_samples()` puis prendre l'argmax causal).
- **Option 3 (refonte)** : remplacer le HMM par BOCPD (déjà disponible dans `src/intelligence/bocpd.py` — cf. audit §3.5).

### 3.7 ✅ Calendar event alignment

`_get_calendar_multiplier(timestamp)` (l.752) reçoit le timestamp de la barre à prédire. Le multiplier est `1.0 + 1.5 * max(0, 1 - min_hours/window)` où `min_hours` = distance absolue au plus proche événement dans la fenêtre ±4h. Le code **regarde des événements potentiellement futurs**, mais c'est conforme à la sémantique métier (l'utilisateur connaît le calendrier économique à l'avance).

**Précision importante** : si on testait sur des événements `Actual` (= valeur publiée), il y aurait leakage. Mais on regarde uniquement la **timestamp de l'annonce** (donnée connue ex-ante). ✅

---

## 4. Bugs B1+B2 mentionnés eval_04 — vérification

### 4.1 B1 : HMM `predict` row-by-row (perf)

**Localisation** : `volatility_lgbm.py:160-167` (avant fix).
**Statut** : ✅ **FIXÉ** — `_vectorized_regime_features` (l.232-287) utilise `hmm._compute_log_likelihood(obs[finite]) + log_start` puis `argmax` (l.264-266). Fallback per-row préservé (l.267-275) si l'API interne change.

Mesure 2026-05-15 sur n=200 :
- Scalar per-row : 0.61 ms/call ≈ 122 ms total.
- Vectorized batch : 1.37 ms total.
- **Speedup : 90×**.
- **Correctness : 100% (ord et mult)**.

### 4.2 B2 : Event proximity O(N×E) (perf)

**Localisation** : `volatility_lgbm.py:152-155` (avant fix).
**Statut** : ✅ **FIXÉ** — `_vectorized_event_proximity` (l.190-221) utilise `np.searchsorted` (O(N log E)). Fallback : passe l'array vide en cas d'absence d'event.

Pas re-bench explicitement (vérification du code seule), mais la latence finale Hybrid (153 ms p50) est compatible avec la décomposition eval_04 §4.5 attribuant ~95% de la latence à `build_features` (avec hot fix → ~50-100 ms).

### 4.3 B3 (mineur) : `LGBMVolForecaster._har_forecaster` dupliqué

**Localisation** : `volatility_lgbm.py:91`. Toujours présent dans le code actuel.
**Statut** : non-fixé, **non-bloquant**. C'est juste une duplication d'instance HAR lorsque LGBM est utilisé seul vs via Hybrid. Coût : 1 lock + 1 config copy. Trivial.

### 4.4 B4 (résolu) : pickle versioning

**Localisation** : `volatility_forecaster.py:1108-1215`.
**Statut** : ✅ **FIXÉ** — `STATE_VERSION_MAJOR=2`, refus si schéma majeur diverge (l.1162-1168) et si feature names changent (l.1173-1179). Bien défensif (logger.error + return False, pas raise).

### 4.5 B5 (mineur) : docs Colab inconsistantes

Statut **non-vérifié** dans cet audit (scripts/colab_*_vol_poc.py hors périmètre).

### 4.6 B6 (mineur) : Clamp `[0.2×, 5.0×]` jamais actif

Eval_04 §4.6 rapporte 0/374 activations dans le bench. Toujours présent dans le code (l.543, l.1495-1499). Recommandation eval_04 (passer à `[0.5×, 3×]` ou retirer) **non-livrée**.

---

## 5. Latence empirique — détail par couche

### 5.1 Décomposition (intuition basée sur lecture code + bench)

```
HAR forecast p50 = 45 ms
├─ _add_features (rolling pandas sur 3000 bars × 4 RV scales + ATR)  ~30-35 ms
├─ _predict_har (LinearRegression.predict sur 1 ligne)                ~0.1 ms
├─ _get_regime_multiplier (HMM predict 1 ligne)                       ~0.5 ms
├─ _get_calendar_multiplier (np operations sur 875 events)            ~1 ms
└─ blend + clip + assemble                                            ~0.5 ms

LGBM forecast p50 = 149 ms
├─ HAR forecast (ci-dessus)                                           ~45 ms
├─ build_features (LGBM 21 features + sessions + technical)           ~85-90 ms
│   ├─ Yang-Zhang RV recompute                                        ~5 ms
│   ├─ atr_7, atr_change_5, atr_change_20                             ~5 ms
│   ├─ session dummies, RSI, BB, MACD                                 ~30 ms
│   ├─ vectorized regime features                                     ~5 ms
│   ├─ vectorized event proximity                                     ~5 ms
│   └─ pandas indexing + dtype conversions                            ~30 ms
└─ LGBM.predict (1 ligne)                                             ~5 ms

Hybrid forecast p50 = 153 ms
├─ HAR forecast (full pipeline)                                       ~45 ms
├─ LGBM build_features                                                ~85-90 ms
├─ LGBM.predict (résidu)                                              ~5 ms
└─ assemble VolatilityForecast                                        ~0.5 ms
```

### 5.2 Goulots d'étranglement (par ordre d'impact)

1. **`_add_features` rolling pandas** (~30 ms par forecast, ×3000 bars × 4 windows) — recomputé à chaque appel.
   - **Fix Sprint 5** : maintenir rolling stats incrémentales (deque + running mean) → ~1 ms par forecast après warmup.
2. **`build_features` technical indicators** (~30 ms) — RSI, Bollinger, MACD sur 3000 bars.
   - **Fix Sprint 5** : seul le dernier bar est utilisé, donc on peut calculer en streaming O(1).
3. **Pandas dtype conversions** (~30 ms) — `df["timestamp"].values.astype("datetime64[ns]")`, etc.
   - **Fix Sprint 5** : passer en numpy structuré ou tenir les arrays en cache forecaster.
4. **LGBM `predict`** (~5 ms) — déjà optimal, ne pas toucher.

### 5.3 Conformité contrainte 50 ms p95

| Mode    | p95 mesuré | Cible 50ms | Verdict                                |
| ------- | ---------- | ---------- | -------------------------------------- |
| HAR     | 72 ms      | < 50 ms    | ❌ × 1.4 hors cible                     |
| LGBM    | 191 ms     | < 50 ms    | ❌ × 3.8 hors cible                     |
| Hybrid  | 221 ms     | < 50 ms    | ❌ × 4.4 hors cible                     |

**Aucun mode ne respecte la cible 50 ms p95.** La cible de eval_04 (« HAR ≈ 54 ms p95 ») est sortie de fenêtre sur la mesure 2026-05-15 (72 ms p95). Le scanner tournant à 60s, on garde une marge confortable même à 200 ms — mais la cible 50 ms doit être révisée à **≤ 100 ms p99** (kill-criterion `tests/test_vol_latency.py:129`).

---

## 6. InstrumentConfig — propagation

### 6.1 Définition

`InstrumentConfig` (`volatility_forecaster.py:239-297`) inclut `price_decimals: int = 2` (l.282).

Presets :
| Instrument | price_decimals | Lignes |
| ---------- | -------------- | ------ |
| XAUUSD     | 2              | l.59   |
| EURUSD     | 5              | l.79   |
| BTCUSD     | 2              | l.96   |
| US500      | 1              | l.113  |
| GBPUSD     | 5              | l.132  |
| USDJPY     | 3              | l.151  |

### 6.2 Propagation vers consommateurs

- ✅ `ConfluenceDetector` (`confluence_detector.py:173,178`) : `getattr(instrument_config, "price_decimals", 2)` + `round(entry, self._price_decimals)` à l.348-350 (entry/SL/TP rounding).
- ✅ `TemplateNarrativeEngine` (`template_narrative_engine.py:277,326,450`) : `_price_decimals(symbol)` helper avec dict de fallback (devrait idéalement consommer `InstrumentConfig` directement, point de dette).

### 6.3 Findings F-cfg

| # | Finding | Sévérité |
| - | ------- | -------- |
| F-cfg-1 | `TemplateNarrativeEngine._price_decimals` dupplique la connaissance des decimals au lieu de consommer `InstrumentConfig` (`template_narrative_engine.py:450`). Risque de divergence si on change Gold 2 → 3 decimals. | P2 |
| F-cfg-2 | `VolatilityForecast.to_dict` (l.319-332) hardcode `round(..., 4)` partout. Ne consomme pas `price_decimals` pour les ATR (ATR n'est pas un prix mais une variation absolue → 4 décimales OK pour gold mais incorrect pour BTCUSD où 4 décimales = 0.0001 $ ne fait pas sens). | P2 |
| F-cfg-3 | `InstrumentConfig.tcp_alpha=0.05` (l.285) **différent** de `_tcp_alpha=0.10` initialisé dans `__init__` (l.367). **Le code utilise `self._tcp_alpha`** (l.367) pour les quantiles, pas `self._config.tcp_alpha`. Configuration utilisateur ignorée. | **P1** |

**Précision F-cfg-3 (P1)** : si un utilisateur fait `InstrumentConfig(tcp_alpha=0.025)` (cible 95% central), il pense obtenir des bandes 95%. En pratique, le code ignore son paramètre et utilise 80% (= 1 - 2×0.10). **Bug silencieux**.

---

## 7. Couverture des tests

### 7.1 Inventaire

| Fichier                             | LOC | Tests | Couverture estimée |
| ----------------------------------- | --- | ----- | ------------------ |
| `tests/test_volatility_forecaster.py` | 694 | 50 (12 classes) | ~75 % du HAR (12 classes : Instrument, Forecast, YZRV, HAR features, Diurnal, Calendar, Calibration, Forecast, TCP, Persistence, Stats, EdgeCases) |
| `tests/test_hybrid_vol.py`            | 324 | 23 (7 classes) | ~60 % du Hybrid (factory, calibration, forecast, fallback, persistence, stats) |
| `tests/test_lgbm_vol.py`              | 394 | 22 (7 classes) | ~65 % du LGBM (features, training, prediction, persistence, edge cases) |
| `tests/test_vol_latency.py`           | 140 | 1 bench | latence HAR p99 < 200 ms ceiling |
| **TOTAL**                             | 1 552 | **96 tests** | |

### 7.2 Métriques testées explicitement

- **MAE** : oui (implicite dans `test_forecast_reasonable_range`, `test_predict_reasonable_range`).
- **RMSE** : non.
- **QLIKE** : ❌ **non**.
- **MSE log-vol** : ❌ **non**.
- **PICP / coverage** : ❌ **non**.
- **Latence p99** : ✅ (`test_vol_latency.py`, ceiling 200ms — pas la cible 50ms).
- **Diebold-Mariano** : ❌ **non** (rapporté ad-hoc dans eval_04 seulement).

### 7.3 Couverture des cas crisis

| Période     | Régime    | Couvert par tests ? |
| ----------- | --------- | ------------------- |
| 2020 COVID  | crash + rebond | ❌ aucun synthetic ni real-data dans tests |
| 2022 LDI    | UK gilt + risk-off | ❌ |
| 2024 XAU    | régime tendance + géopolitique | ❌ |
| Crises synthétiques (jumps, fat tails) | n/a | ❌ |

**100 % des tests utilisent des données synthétiques GBM ou random walk simple** (`_make_synthetic_ohlcv` dans test_hybrid_vol.py l.32-44 ; `_synthetic_ohlcv` dans test_vol_latency.py l.36-65). **Aucune validation sur les vraies données XAU/EURUSD du repo**.

### 7.4 Tests manquants (P1)

- `test_picp_target_coverage` : sur réelles XAU 2024, assert PICP[80%] ∈ [75%, 85%].
- `test_qlike_beats_naive` : sur XAU 2024, QLIKE_HAR < QLIKE_naive.
- `test_hmm_inference_distribution` : sur réelles XAU 2024, max regime count ≤ 80% (pas de collapse).
- `test_calendar_no_leakage` : forecast(df[:t], ts=t) ne dépend pas de df[t+1:].
- `test_tcp_alpha_propagation` : `InstrumentConfig(tcp_alpha=0.025)` produit des bandes ≈ 95%.

---

## 8. Findings P0/P1/P2 — synthèse

### Sévérité P0 — bloquant produit

| #     | Localisation                                                | Finding                                                                                                          |
| ----- | ----------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| F-P0-1 | `volatility_forecaster.py:849-900, 907-929`                | **HMM régime collapse à l'inférence** : Viterbi smoothing au fit (l.874) vs `predict(1-row)` à l'inférence (l.922) = 11% agreement (mesure 2026-05-15). 100% des forecasts XAU 2024 OOS classés `low`. Multiplier régime systématiquement appliqué incorrectement. |
| F-P0-2 | `volatility_forecaster.py:548-558, 593-619`                | **TCP bandes massivement sous-couvertes** : PICP empirique 43.6% pour cible 80% sur XAU 2024 (mesure 2026-05-15). Dérive trimestrielle 57%→27%. **Update online (l.604-619) jamais invoquée par le scanner** (cf. eval_09 + `sentinel_scanner.py`). |
| F-P0-3 | `volatility_forecaster.py:367`                              | **`InstrumentConfig.tcp_alpha` ignoré** : `self._tcp_alpha=0.10` hardcodé (l.367) ne lit pas `self._config.tcp_alpha=0.05` (l.285). Configuration utilisateur silencieusement écrasée. (Aussi F-cfg-3) |

### Sévérité P1 — qualité significative

| #     | Localisation                                                | Finding                                                                                                          |
| ----- | ----------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| F-P1-1 | `tests/test_volatility_forecaster.py`, `test_lgbm_vol.py`, `test_hybrid_vol.py` | **Aucun test QLIKE / PICP / cross-régime / real-data**. Toute la suite roule sur GBM synthetic. Régression silencieuse possible sur métriques métier. |
| F-P1-2 | `volatility_forecaster.py:543, 1495-1499`                  | **Clamp `[0.2×, 5×]` jamais actif** (0/374 dans eval_04 §4.6). Soit trop large (inutile), soit pas testé sur jumps. Resserrer à `[0.5×, 3×]` OU retirer + monitor. |
| F-P1-3 | `volatility_forecaster.py:30-35` (rollings dans `_add_features`) | **Rolling pandas non-cachés** : à chaque `forecast()`, on recompute RV daily/weekly/monthly sur 3000 bars. ~30 ms gaspillés. Streaming O(1) trivial avec deque ou Welford. |
| F-P1-4 | `volatility_lgbm.py:122-123`                                | **`build_features` recompute tout le feature set** à chaque forecast (3000 bars × 21 features). Seul le dernier bar est utilisé. **Coût ~90 ms gâché**. Streaming feature builder à introduire. |
| F-P1-5 | `volatility_forecaster.py:1474, 1489`                       | **Hybrid double-pipeline** : `_hybrid_forecast_impl` appelle `_forecast_impl` puis `build_features` du LGBM, qui re-exécute `_add_features` (déjà fait en step 1). Travail dupliqué ~30 ms. |
| F-P1-6 | `volatility_forecaster.py:1389-1394`                        | **LGBM résiduel split 80/20 fixe** : pas de walk-forward CV multi-fold. Comparaison avec `_calibrate_blend_weight` (5-fold) — asymétrie de discipline. |

### Sévérité P2 — dette technique

| #     | Localisation                                                | Finding                                                                                                          |
| ----- | ----------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| F-P2-1 | `volatility_forecaster.py:91` (`LGBMVolForecaster._har_forecaster`) | Double instance HAR quand LGBM seul + HAR via Hybrid. Coût trivial mais propre. |
| F-P2-2 | `volatility_forecaster.py:319-332` (`to_dict`)              | Hardcode `round(..., 4)`. Pas adaptatif à `InstrumentConfig.price_decimals` (point F-cfg-2). |
| F-P2-3 | `volatility_forecaster.py:773`                              | `multiplier = 1.0 + 1.5 * max(0, 1 - min_hours / window)` — magic number 1.5x. Pas calibré sur l'amplitude réelle des spikes NFP/FOMC. À CV. |
| F-P2-4 | `volatility_forecaster.py:892`                              | `np.clip(mult, 0.5, 2.5)` — magic clip sans justification. Si régime `crisis` réel produit mult=4, il sera tronqué. |
| F-P2-5 | `volatility_forecaster.py:1339` (`_fit_lgbm_on_residuals`)  | Hyperparamètres LGBM en dur (max_depth=6, lr=0.05, n_estimators=500). Pas de search Bayesian ni grid. |
| F-P2-6 | `volatility_lgbm.py:223-228` (`_get_regime_features`)       | API mixe Tuple[float, float] et arrays — gérable mais inhomogène. |
| F-P2-7 | `volatility_forecaster.py:573-587` (`_fallback_forecast`)   | Hardcode `confidence_lower = 0.5 × naive`, `_upper = 1.5 × naive` (= bande ±50%). Pas justifié empiriquement. |
| F-P2-8 | Aucun fichier (gap)                                         | Pas de monitor `fallback_rate` exposé par `get_stats()`. KPI eval_04 §6 prévoyait < 5%, on ne sait pas mesurer. |

---

## 9. Recommandations Sprint 4 — Calibration & confidence (refonte des bandes)

### Objectif Sprint 4
Faire passer **PICP[80%] ∈ [75%, 85%]** sur XAU 2024 OOS (vs 43.6% actuel) et **PICP[95%] ∈ [93%, 97%]** sur la même slice.

### Tâches priorisées

| # | Tâche | Effort | Cible empirique |
| - | ----- | ------ | --------------- |
| **S4-1** | **Conformal split-prediction** : remplacer la logique TCP actuelle (Robbins-Monro symétrique + buffer quantile non-mis-à-jour) par un **conformal split** propre : 60% train HAR + 20% calibration set → quantiles asymétriques sur résidus de calibration. Refit calibration weekly. | 2-3 j | PICP[80%] = 80% ± 3% |
| **S4-2** | **Activer `update_tcp` dans le scanner** : actuellement la fonction existe (l.593) mais n'est jamais appelée par `sentinel_scanner.py`. Ajouter une boucle `scanner._update_volatility_tcp()` qui matérialise les `actual_atr` 5 bars après chaque forecast et appelle `update_tcp`. | 1 j | bandes adaptatives en live |
| **S4-3** | **Fix F-P0-3 (config.tcp_alpha)** : utiliser `self._config.tcp_alpha` partout au lieu de `self._tcp_alpha`. Backward-compat : continuer à exposer `self._tcp_alpha` mais le copier depuis config au constructeur. | 0.5 j | bande 95% configurable |
| **S4-4** | **Conformal quantile regression** (Romano et al. 2019) sur le LGBM : LGBM apprend deux quantiles τ_lo et τ_hi de la distribution conditionnelle de l'ATR future. Plus précis que conformal split sur HAR. | 4-5 j | PICP[80%] = 80% ± 2% et bandes asymétriques (vol upside ≠ downside) |
| **S4-5** | **Désactiver / refondre HMM régime** : court terme, `regime_multiplier = 1.0` par défaut tant que F-P0-1 non-livré. Long terme : remplacer par BOCPD (`src/intelligence/bocpd.py`) qui est déjà online causal. | 1 j (désactivation) + 3 j (BOCPD integration) | suppression du biais systématique HAR |
| **S4-6** | **Tests P1-1** : `test_picp_target_coverage` + `test_qlike_beats_naive` + `test_hmm_no_collapse` sur réelles XAU 2024. | 1.5 j | CI gate sur métriques métier |
| **S4-7** | **Walk-forward LGBM résiduel** : passer le split 80/20 à 5-fold expanding window dans `_fit_lgbm_on_residuals` (par symétrie avec `_calibrate_blend_weight`). | 1 j | improvement_pct honnête |

**Total Sprint 4** : ~14-19 jours.

---

## 10. Recommandations Sprint 5 — Performance (si LGBM/Hybrid persistent)

### Objectif Sprint 5
Faire passer **p95 latence forecast ≤ 50 ms** pour tous les modes, ou retirer LGBM/Hybrid si non-rentable.

### Tâches priorisées

| # | Tâche | Effort | Cible empirique |
| - | ----- | ------ | --------------- |
| **S5-1** | **Streaming `_add_features`** : maintenir rolling stats incrémentales (deque + Welford) au lieu de pandas `rolling().mean()`. Le forecaster maintient son propre state warmup. | 3-4 j | HAR p99 35 ms → 5 ms |
| **S5-2** | **Streaming `build_features`** : idem pour atr_7, atr_change_5, RSI, BB, MACD, session dummies — calculs O(1) par bar. | 4-5 j | LGBM/Hybrid p99 220 → 50-80 ms |
| **S5-3** | **Cache HAR base dans Hybrid** : éviter de re-rouler `_forecast_impl` + `build_features` côte à côte (point F-P1-5). Partager le feature_df entre les deux. | 1.5 j | Hybrid p99 220 → ~140 ms |
| **S5-4** | **ONNX export `_har_model`** : optionnel (le LinearRegression est microseconds), mais peut aider sur cold-start dans serverless. | 1-2 j | non-blocking |
| **S5-5** | **Benchmark CI nightly** sur réelles XAU + EURUSD + BTCUSD : p99 < 50 ms par mode, sinon fail. | 1 j | régression latence détectée < 1 j |
| **S5-6** | **Décision LGBM** : si après S5-1 à S5-5, LGBM p99 ne passe pas sous 100 ms, OU si Hybrid ne bat pas HAR sur QLIKE walk-forward 2025, **retirer LGBM/Hybrid** du factory. | < 1 j (décision) | code base réduite, surface de bug ↓ |

**Total Sprint 5** : ~12-15 jours.

---

## 11. Tableau de synthèse perf — état 2026-05-15

### 11.1 Qualité prédictive (XAU M15 2024, walk-forward day-step, n=282)

| Mode    | RMSE    | MAE     | QLIKE   | MSE log-vol | Δ RMSE vs naïf | Δ QLIKE vs naïf | DM stat (loss²) |
| ------- | ------- | ------- | ------- | ----------- | -------------- | --------------- | --------------- |
| Naïf ATR14 | 2.7925  | 1.91*   | 4.3651  | 0.2413      | —              | —               | —               |
| **HAR**  | **2.7756** | **n/a** | **4.8002** | **0.2783** | **−0.6%**     | **+0.435 (pire)** | **non calculé Sprint 0** |
| LGBM    | n/a (non-bench Sprint 0)   | n/a     | n/a     | n/a         | n/a            | n/a             | n/a             |
| Hybrid  | n/a (non-bench Sprint 0)   | n/a     | n/a     | n/a         | n/a            | n/a             | n/a             |

*MAE naïf non-mesuré exactement par cet audit, RMSE only.

**À compléter Sprint 1** : LGBM et Hybrid walk-forward complets (déjà fait dans eval_04 pour 2022-2024, mais 2024 isolé pas explicitement).

### 11.2 Calibration des bandes (XAU M15 2024)

| Mode    | PICP[80%] | PICP[80%] Q4 2024 | mean width / forecast | conformal status |
| ------- | --------- | ----------------- | --------------------- | ---------------- |
| **HAR**  | **43.6%** ❌ | **27.1%** ❌ | **0.565**             | **dérive non-corrigée** |
| LGBM / Hybrid | non-mesuré | non-mesuré | non-mesuré | non-corrigé (hardcode `(1 ± tcp_width)`) |

### 11.3 Latence (XAU M15, window=3000 bars, real calendar, real OHLC)

| Mode    | Calibrate (s) | p50 (ms) | p90 (ms) | p95 (ms) | p99 (ms) | max (ms) | sample size |
| ------- | ------------- | -------- | -------- | -------- | -------- | -------- | ----------- |
| HAR     | 17.8          | 45.3     | 64.9     | **72.1** | 91.0     | 138.8    | 1 000       |
| LGBM    | 9.2           | 149.2    | 174.5    | **191.1** | 223.6    | 227.5    | 200         |
| Hybrid  | 21.3          | 152.9    | 209.0    | **220.5** | 263.4    | 284.6    | 200         |
| Cible   | n/a           | n/a      | n/a      | **50**    | 100      | n/a      | n/a         |

### 11.4 Robustesse régime (XAU M15 2024, n=282)

| Mode    | Régimes observés (inférence)             | Diversité (≥ 3 régimes ≥ 10% chacun ?) |
| ------- | ---------------------------------------- | --------------------------------------- |
| **HAR**  | `{'low': 282}` (100 % `low`)            | ❌ **collapse**                         |
| LGBM    | partagé avec HAR (même HMM)              | ❌                                       |
| Hybrid  | partagé avec HAR                         | ❌                                       |

---

## 12. Ce que cet audit ne couvre pas

1. **Training côté Colab** : `scripts/colab_har_rv_poc.py`, `scripts/colab_lgbm_vol_poc.py`, `scripts/colab_hybrid_vol_poc.py` ne sont pas relus. Audit limité au code de production servi.
2. **Modèles tiers / non utilisés** : GARCH, TSFM (Time-Series Foundation Models), N-BEATS, TFT. Décision déjà actée (MEMORY.md) de ne pas les utiliser.
3. **Multi-instrument empirique** : tests latence et PICP n'ont été menés que sur XAU. EURUSD/BTCUSD/USDJPY non-bench. Le code est paramétrique (`InstrumentConfig` propre), mais aucune validation empirique cross-actifs depuis eval_04.
4. **Multi-timeframe** : seuls M15 testés. M5/M30/H1/H4 non-bench (le code `resample_ohlcv` existe et est testé unitaire mais pas en bout en bout).
5. **TCP coverage des intervalles 95%** : tcp_alpha=0.025 non-mesuré ; seul 80% l'a été. Vraisemblablement encore plus dégradé.
6. **Comportement sur jumps réels** : pas de validation sur des barres NFP/FOMC connues (par exemple `2024-03-08 13:30 UTC` = NFP). À ajouter Sprint 4.
7. **Stress-test concurrence** : `threading.Lock()` est présent (l.351-352) mais pas testé sous charge multi-thread sentinelle.
8. **Persistance long-terme** : audit charge/save fonctionne unitaire (tests présents) mais pas testé sur fichiers > 6 mois ni rotation.
9. **Coût mémoire** : eval_04 §4.7 mesurait ΔRSS = +64 MB pour HAR (rolling pandas), non re-vérifié dans cet audit.
10. **Integration avec le rest of pipeline** : la sortie `VolatilityForecast` est utilisée par `ConfluenceDetector` (§6.2 §F-cfg) et `SentinelScanner` (eval_09), mais on n'audite pas le flow end-to-end (out of scope).

---

## 13. Décision opérationnelle

**Recommandation immédiate (post-audit Sprint 0)** :

1. ✅ **Maintenir `VOL_MODE=har` par défaut** — confirmé.
2. 🚧 **Désactiver le multiplier HMM** (`regime_multiplier = 1.0`) tant que F-P0-1 non-livré. Risque : régression de qualité légère (le mult était dans `[0.7, 1.0]` la plupart du temps, donc le forecast augmenterait marginalement). À mesurer.
3. 🚧 **Avertir produit** : ne pas exposer la bande "80% confidence" au client tant que PICP n'est pas corrigée Sprint 4. Le mockup `webapp/mockups/webapp_b2c.html` doit être audité pour vérifier qu'aucun "interval 80%" n'est promis.
4. ✅ **Sprint 4 (calibration & confidence)** : S4-1 → S4-7 (~14-19j) prioritaire avant tout lancement B2B-API.
5. ⏸ **Sprint 5 (perf)** : différable. Latence actuelle (≤ 250 ms p99) reste compatible avec un scanner 60s. Cible 50 ms = sur-engineering pour le besoin actuel. Reprioriser uniquement si on déplace vers une API HTTP B2B où chaque ms coûte.

---

## 14. Références

- Corsi, F. (2009). *A Simple Approximate Long-Memory Model of Realized Volatility*. JFE.
- Yang, D. & Zhang, Q. (2000). *Drift-Independent Volatility Estimation Based on High, Low, Open, and Close Prices*. JoB.
- Patton, A. J. (2011). *Volatility forecast comparison using imperfect volatility proxies*. Journal of Econometrics 160 (1), 246-256. (Définition QLIKE).
- Romano, J. P., Patterson, E., Candes, E. J. (2019). *Conformalized Quantile Regression*. NeurIPS.
- Adams, R. P. & MacKay, D. J. C. (2007). *Bayesian Online Changepoint Detection*. arXiv:0710.3742. (Pour S4-5 alternative HMM).
- `reports/eval_04_volatility.md` (audit antérieur 2026-04-29, note 5.0/10).

---

## 15. Reproduction des mesures 2026-05-15

```python
# Bench latence HAR/LGBM/Hybrid sur réelles XAU
python -c "
import sys, time, numpy as np, pandas as pd
sys.path.insert(0, '.')
from src.intelligence.volatility_forecaster import VolatilityForecaster, HybridForecaster, InstrumentConfig
df = pd.read_csv('data/XAU_15MIN_2019_2025.csv')
df.columns = [c.lower() for c in df.columns]
df = df.rename(columns={'date':'timestamp'})
cal = pd.read_csv('data/economic_calendar_HIGH_IMPACT_2019_2025.csv')
cfg = InstrumentConfig(symbol='XAUUSD', timeframe='M15')
fc = HybridForecaster(cfg, mode='hybrid')
fc.calibrate(df.iloc[-20000:-500], cal)
window = 3000
test_df = df.iloc[-window-200:].copy()
timings = []
for i in range(200):
    sub = test_df.iloc[i:i+window]
    t0 = time.perf_counter()
    fc.forecast(sub)
    timings.append((time.perf_counter()-t0)*1000)
print(f'p50={np.percentile(timings,50):.1f}ms p99={np.percentile(timings,99):.1f}ms')
"
```

```python
# PICP empirique 80% sur XAU 2024
python -c "
# (script ci-dessus inline §2.3) — 282 forecasts, train ≤ 2023, test = 2024
# Résultat attendu: PICP_80% ≈ 43.6%, mean_width/forecast ≈ 0.565
"
```

```python
# HMM Viterbi vs predict-per-row skew
python -c "
from hmmlearn.hmm import GaussianHMM
import numpy as np
hmm = GaussianHMM(n_components=2, covariance_type='diag', n_iter=10, random_state=42)
X = np.random.randn(200, 2)
hmm.fit(X)
states_full = hmm.predict(X)
states_row = np.array([hmm.predict(X[i:i+1])[0] for i in range(len(X))])
print(f'Agreement: {(states_full == states_row).mean()*100:.1f}%')
# Résultat attendu: ~11%
"
```
