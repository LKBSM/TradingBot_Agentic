# Plan de Commercialisation — Catégorie 4: Volatility Forecasting

> **Périmètre** : HAR-RV, LightGBM vol model, Hybrid (HAR + LGBM résidu), HMM 3 états, calibration `_blend_weight` Bates-Granger, TCP conformal width, intégration scanner.
> **Date** : 2026-05-21 · **Branch** : `institutional-overhaul` · **Auteur** : Catégorie 4 — Volatility.
> **Référence empirique** : `reports/eval_04_volatility.md` (5.0/10, walk-forward XAU 141k barres) + `reports/eval_04/walkforward_summary.json`.

---

## 0. TL;DR

| Axe | État | Cible commercialisation | Effort |
|---|---|---|---|
| Mode prod par défaut | `VOL_MODE=har` (`src/intelligence/main.py:537`, `src/intelligence/security.py:212-234`) | **conserver HAR** tant que P0-LATENCY non livré ; route LGBM-fast en option Tier 3 | 0h |
| Latence HAR forecast p99 | **108-130 ms** mesuré (`tests/test_vol_latency.py:118-130` accepte 200ms via kill-criterion) | **p99 < 50 ms** | 12-18h |
| Latence LGBM forecast p99 | **187 ms** post-hotfix P1 (`reports/eval_04_volatility.md:370`) vs 3 957 ms avant | **p99 < 50 ms** via ONNX/Booster.predict batched | 24-32h |
| Qualité RMSE LGBM vs naïf | **−30.7 %** (DM p=6e-13, `reports/eval_04/walkforward_summary.json`) | maintenir ≥ −25 % RMSE walk-forward + DM p<0.01 | 8h |
| Calibration intervals (TCP) | Quantile empirique post-Sprint 4 (`volatility_forecaster.py:1084-1097`) — coverage non monitoré en prod | **coverage 93-97 % rolling, monitoring** | 16h |
| Multi-instrument | 6 presets définis mais **5/6 sans CSV** (eval_08 + eval_20) | XAU + EUR pleinement calibrés, M15 + H1 | 24h |

**Top 3 P0** : (1) Optimiser latence HAR `_add_features` rolling → cible 50 ms (12-18h). (2) ONNX/quantization LGBM + feature cache → cible 50 ms (24-32h). (3) Fix bug HMM `_get_regime_multiplier` regardant la dernière barre du window au lieu de la barre cible — eval_04 §4.4 note tous les forecasts classés `low` (8h).

**Total commercialisation v1.0 vol forecasting** : **154-186h** (≈4-5 sem solo, parallélisable 3 sem).

---

## 1. État actuel (Audit)

### 1.1 Architecture

```
VolatilityForecaster.create(mode) — factory (volatility_forecaster.py:1241-1263)
   ├─ mode="har"    → VolatilityForecaster (HAR + diurnal + calendar + HMM + blend + TCP)
   ├─ mode="lgbm"   → HybridForecaster(mode="lgbm")       (LGBM direct sur future_atr)
   └─ mode="hybrid" → HybridForecaster(mode="hybrid")     (HAR base + LGBM résidu)
```

Wiring scanner : `src/intelligence/sentinel_scanner.py:405-419` appelle `_vol_forecaster.forecast(enriched, bar_ts)` à chaque bar M15 (60s polling). Le forecast feed (a) `confluence_detector` pour le scoring vol-aware, (b) `kill_switch.update_volatility(forecast_atr)` pour le circuit breaker vol-spike, (c) `signal.vol_forecast_atr` exposé via API.

### 1.2 Résultats empiriques walk-forward (XAU M15, 2022-2024, n=373, `reports/eval_04/walkforward_summary.json`)

| Modèle | RMSE | MAE | DM vs naïf | DM vs HAR | P95 latence (bench) |
|---|---|---|---|---|---|
| naïf ATR14 | 1.226 | 0.888 | — | — | n/a |
| HAR-RV | 1.080 | 0.716 | −4.44 (p=9e-6) ✓ | — | **54 ms** |
| **LGBM** | **0.850** | **0.549** | **−7.20 (p=6e-13) ✓✓** | +6.29 (p=3e-10) ✓✓ | 3 957 ms ✗ (187 ms post-P1) |
| Hybrid | 0.921 | 0.633 | −5.74 (p=1e-8) ✓ | +4.11 (p=4e-5) ✓ | 1 724 ms ✗ |

**Verdict empirique** :
- LGBM **domine statistiquement** HAR ET Hybrid (DM −2.56 vs Hybrid, p=0.011).
- Mais latence brute LGBM = ×33 au-dessus du target 50 ms ; après P1 hotfix (vectorisation HMM/event-prox) = ×3.7. Encore hors cible.
- HAR seul tient le P95 = 54 ms en bench, mais p99 ≈ 108-130 ms en mesure réelle (`tests/test_vol_latency.py:118-122`) → toujours hors cible.

### 1.3 Bugs et limitations actives

| # | Sévérité | Localisation | Description |
|---|---|---|---|
| B1 | 🟢 Fixé | `volatility_lgbm.py:233-287` | HMM batched + event-prox via `np.searchsorted` (hotfix 2026-04-29) |
| B2 | 🟢 Fixé | `volatility_lgbm.py:191-221` | Idem B1 |
| **B3** | 🔴 P0 perf | `volatility_forecaster.py:641-678` (`_add_features`) | À chaque `forecast()`, rolling `rv_daily/weekly/monthly`+`atr_14`+`rv_bar` recalculés sur toute la fenêtre 3000 bars (≥80 ms) |
| **B4** | 🔴 P0 correctness | `volatility_forecaster.py:909-931` (`_get_regime_multiplier`) | En forecast, regarde `idx = latest_idx` (la barre courante) au lieu de la barre cible → eval_04 §4.4 : **tous les forecasts classés `low` en bench** |
| B5 | 🟡 P1 leakage | `_fit_lgbm` mode=`lgbm` direct (`volatility_forecaster.py:1330-1335`) | Pas de refit HAR par fold pour le baseline ; idem `LGBMVolForecaster._train_impl:380-400` (split 80/20 simple, pas CPCV) |
| B6 | 🟡 P1 perf | `volatility_lgbm.py:171-172` | RSI/BB/MACD recalculés en pandas sur toute la fenêtre à chaque forecast — pas de cache incrémental |
| B7 | 🟡 P1 robustesse | `volatility_forecaster.py:1097-1162` | `_state_version_major=2` ajouté ; mais pickle reste non signé (vuln. arbitrary code execution si state corrompu) |
| B8 | 🟢 P2 cosmétique | `volatility_forecaster.py:543, 1497-1501` | Clamp `[0.2×naive, 5×naive]` jamais activé en bench (eval_04 §4.6) — peut être resserré à `[0.5×, 3×]` |
| B9 | 🟡 P1 outlier | bench split 1 LGBM = 26 573 s (7.4 h) | Pathologie reproductible non investiguée (eval_04 §4.5 note 1) |
| B10 | 🟢 P2 leakage | HMM trained on full data (eval_04 §2.6) | Effet faible empirique mais à corriger pour audit institutionnel |
| B11 | 🔴 P0 monitoring | absent | Aucun telemetry sur `forecast.is_fallback`, latence p99, coverage TCP, clamp activation, drift RMSE |
| B12 | 🟡 P1 data | seul `XAU_15MIN_2019_2024.csv` exploité bench. EURUSD 99.6% disponible (memory eval_05_09) | Pas de calibration EUR validée |

### 1.4 Couverture tests

- `tests/test_volatility_forecaster.py` (143 tests passants post-hotfix selon `reports/eval_04_volatility.md:369`)
- `tests/test_lgbm_vol.py`
- `tests/test_hybrid_vol.py`
- `tests/test_vol_latency.py:83-139` : asserte p99 < **200 ms** sur 1000 inférences synthétique (kill-criterion, pas KPI cible 50 ms)
- `tests/test_vol_narratives.py` : intégration narrative LLM
- **Aucun test walk-forward sur données réelles dans la CI** (eval_04 §1 Couverture 6/10)

---

## 2. Vision cible

Vol forecasting **commercialisable, institutional-grade**, défini par les invariants :

| Invariant | Cible | Justification |
|---|---|---|
| Latence forecast p99 | **< 50 ms** | Scanner polling 60 s, marge LLM+confluence+narrative concurrente |
| Latence forecast p50 | **< 20 ms** | Préserve headroom multi-symbole (6 presets) |
| RMSE walk-forward vs naïf | **< −20 % minimum** (objectif −30 %) | Mesure DM test bilatéral, HAC bandwidth h-1 |
| Stabilité année-sur-année | RMSE drift < 30 % entre années consécutives | eval_04 §4.4 montre LGBM +38 % drift 2022→2024, à monitorer |
| Coverage TCP empirique | **93-97 %** rolling 500-bar | Conformal valide si calibré ; sous-coverage = risk metric bidon |
| Fallback rate | **< 5 %** | telemetry scanner |
| Clamp activation | **< 1 %** | Sinon clamp = bandwidth-limiter, signal info perdue |
| Multi-instrument | **XAU + EUR pleinement calibrés** ; BTC + US500 + GBP + JPY en preset best-effort | eval_20 reco drop BTC+US500 ; on garde framework |
| Reproductibilité | Walk-forward CI nightly avec assertion `RMSE_har < 0.85 × RMSE_naive` | eval_17 |
| Drift detection | Alerte si RMSE rolling 30j > 1.5× RMSE training | Production ML hygiene |
| Audit trail | State pickle versionné + checksum + signature | Eval institutional (B7) |

**Mode prod cible** : double bandeau —
- **Tier hot-path (scanner 60s)** : `VOL_MODE=har` ou `lgbm-fast` (cf. P0-LATENCY-LGBM) selon résultat optim.
- **Tier batch (calibration nightly)** : Hybrid pour audit RMSE de référence, jamais en hot-path.

---

## 3. Gap Analysis

| Gap | État actuel | Cible | Effort | Bloqueur commercialisation ? |
|---|---|---|---|---|
| **G1. Latence HAR p99** | 108-130 ms | < 50 ms | 12-18h | Oui (perceived "lag") |
| **G2. Latence LGBM p99** | 187 ms post-hotfix | < 50 ms | 24-32h | Oui (sinon LGBM inutilisable scanner) |
| **G3. Bug HMM regime** (B4) | tous forecasts `low` en bench | Régime correct = barre cible | 8h | Oui (feature dead) |
| **G4. Calibration TCP monitoring** | jamais vérifié en prod | coverage rolling 95% | 16h | Oui (risque commercial : "VaR au pifomètre") |
| **G5. Multi-asset** | XAU only en CSV | XAU + EUR calibrés et benchés | 24h | Non (peut sortir XAU-only v1.0) |
| **G6. Walk-forward CI** | aucun | nightly + DM test | 12h | Non mais nécessaire post-v1.0 |
| **G7. Drift detection** | aucun | RMSE rolling alert | 12h | Non (post-launch) |
| **G8. Signed pickle** | non signé | HMAC + checksum + version | 6h | Non (sécurité défense en profondeur) |
| **G9. LGBM CPCV leak** (B5) | split 80/20 simple | Combinatorial Purged CV | 16h | Non (mais bloque audit institutional) |
| **G10. Conformal regime-aware** | TCP global | TCP par régime HMM | 20h | Non (P1 différenciation produit) |

---

## 4. Plan d'exécution

### P0 — Latence & correctness (bloquants go-live)

#### P0-LATENCY-HAR : Optimiser HAR forecast hot-path → p99 < 50 ms — **12-18h**

**Objectif** : faire descendre p99 de 110-130 ms à < 50 ms sur fenêtre 3 000 bars sans dégrader RMSE.

**Tâches** :
- **T1 (4h)** : profiler `VolatilityForecaster._forecast_impl` (`volatility_forecaster.py:487-573`) avec `cProfile` + `line_profiler`. Identifier les ~80 % temps passés dans `_add_features` rolling.
- **T2 (6-10h)** : refactor `_add_features` (`volatility_forecaster.py:641-678`) en **calcul incrémental** :
  - Maintenir `_rolling_state` (deque) au niveau forecaster : `rv_bar` exponentially-weighted, `atr_14` Welford-style, `rv_daily/weekly/monthly` via somme-glissante.
  - Mode "incremental forecast" : on passe seulement la dernière barre + le state.
  - Garder `_add_features` legacy pour `calibrate()`.
  - Fichier : `src/intelligence/volatility_forecaster.py` (nouvelle méthode `_forecast_incremental`).
- **T3 (2-4h)** : test `tests/test_vol_latency.py` re-asserter **p99 < 50 ms** au lieu de 200 ms.

**Acceptance criteria** :
- `pytest tests/test_vol_latency.py::test_har_forecaster_p99_latency_under_50ms` passe.
- `pytest tests/test_volatility_forecaster.py` aucune régression.
- RMSE walk-forward reproductible (`scripts/eval_04_volatility.py`) : delta < 0.5 % vs baseline.

**Dépendances** : aucune (autonome).

---

#### P0-LATENCY-LGBM : LGBM Booster en hot-path → p99 < 50 ms — **24-32h**

**Objectif** : passer LGBM de 187 ms → < 50 ms p99 pour rendre `VOL_MODE=lgbm` viable en production.

**Tâches** :
- **T1 (4h)** : profiler `LGBMVolForecaster.predict_from_df` (`volatility_lgbm.py:493-496` → `build_features` + `_model.predict`). Décomposer où vont les 187 ms (probablement 80 % `build_features` rolling pandas + RSI/BB/MACD).
- **T2 (8h)** : **incremental feature cache** :
  - Réutiliser `_rolling_state` de P0-LATENCY-HAR pour les features partagées (`rv_daily/weekly/monthly`, `atr_14`, `atr_7`).
  - Cache RSI / BB %B / MACD via état exponentiel (Wilder pour RSI, Welford pour BB).
  - Fichier : `src/intelligence/volatility_lgbm.py` (méthode `build_features_incremental`).
- **T3 (6h)** : **export Booster compact** :
  - LGBM `Booster.save_model()` produit déjà du binaire optimisé. Évaluer `treelite` si latence model.predict > 5 ms.
  - Tester ONNX export via `onnxmltools.convert_lightgbm` + `onnxruntime` inference. Mesurer si gain > 30 %.
  - Garder code path pure-Python comme fallback si ONNX dégrade RMSE > 1 %.
  - Fichier : `src/intelligence/volatility_lgbm.py:520-575` (`save_model` / `load_model` extensions).
- **T4 (4h)** : **batch inference** : si scanner est étendu multi-symboles, ajouter `predict_batch(rows: pd.DataFrame) -> np.ndarray`.
- **T5 (4h)** : nouveau test `tests/test_vol_latency.py::test_lgbm_forecaster_p99_under_50ms` (synthetic 5k bars).
- **T6 (4-6h)** : bench reproductibilité (RMSE delta < 1 % vs baseline post-hotfix) sur `data/XAU_15MIN_2019_2024.csv`.

**Acceptance criteria** :
- Bench `scripts/eval_04_volatility.py --mode lgbm` : p95 < 50 ms, p99 < 80 ms.
- RMSE walk-forward LGBM ≥ −25 % vs naïf (vs −30.7 % avant ; tolérance 5 % pour gains ONNX).
- Test p99 50 ms vert sur GHA CI runner (matrix Linux/Windows).

**Dépendances** : P0-LATENCY-HAR (rolling state partagé).

---

#### P0-BUG-HMM-REGIME : Fix B4 — régime classé en barre cible — **8h**

**Objectif** : `_get_regime_multiplier` (`volatility_forecaster.py:909-931`) doit retourner le régime au moment de la **dernière barre observée** (correctement vu) et non aboutir à "low" en permanence.

**Tâches** :
- **T1 (3h)** : investiguer pourquoi eval_04 §4.4 note "tous forecasts classés low". Hypothèses :
  - (a) HMM `predict(obs)` avec une seule obs renvoie l'état dont `startprob_ × emission` est max → toujours le même état si `startprob_` très inégal.
  - (b) `obs = [[ret, rv_daily]]` avec `rv_daily` = moyenne 96 barres → toujours valeur centrale.
- **T2 (3h)** : fix : passer un **mini-window** (e.g. 5-10 dernières barres) à `hmm.predict()` pour exploiter Viterbi sur séquence, pas observation isolée. Ou utiliser `hmm._compute_log_likelihood` + posterior smoothing.
- **T3 (2h)** : test `tests/test_volatility_forecaster.py::test_regime_classified_correctly_in_high_vol` (synthétique high-vol burst → assert `regime_state == "high"`).

**Acceptance criteria** :
- Bench replay XAU 2024 : distribution régime ≠ 100 % `low`. Cible : 60 % `normal`, 20 % `low`, 20 % `high` ±10pp.
- Pas de régression RMSE.

**Dépendances** : aucune.

---

#### P0-DECISION-VOL-MODE : Trancher mode prod par défaut — **2h**

**Décision conditionnelle** :
- **Si P0-LATENCY-LGBM atteint p99 < 50 ms ET RMSE −25 % vs naïf** → basculer `VOL_MODE=lgbm` défaut dans `src/intelligence/security.py:233` et `src/intelligence/main.py:537`.
- **Sinon** → maintenir `VOL_MODE=har`, exposer `lgbm` en Tier 3 explicite (header `X-Vol-Mode: lgbm` en API B2B, opt-in).

**Tâches** :
- **T1 (1h)** : décision documentée dans `reports/governance/kill_criteria_board.md` (REGIME-1.1 row).
- **T2 (1h)** : flip défaut si conditions réunies ; ajuster `tests/test_security.py`.

**Dépendances** : P0-LATENCY-LGBM.

---

#### P0-MONITORING : Telemetry vol forecaster (G4 + G7 partiel) — **16h**

**Tâches** :
- **T1 (4h)** : exporter via `/metrics` les compteurs :
  - `vol_forecast_latency_ms` (Histogram, buckets 1/5/10/20/50/100/500 ms).
  - `vol_forecast_fallback_total` (Counter, label `reason`).
  - `vol_forecast_clamp_activated_total` (Counter, label `direction=low|high`).
  - `vol_forecast_regime_total` (Counter, label `state=low|normal|high|unknown`).
  - `vol_tcp_coverage_rolling_500` (Gauge, calculé chaque 100 updates).
  - Fichier : `src/intelligence/volatility_forecaster.py` + `src/intelligence/sentinel_scanner.py:405-419`.
- **T2 (4h)** : dashboard Grafana (JSON dashboard committé dans `infrastructure/grafana/vol_forecaster.json`).
- **T3 (4h)** : alerting Prometheus (`vol_forecast_p99 > 80ms 5min`, `vol_tcp_coverage_rolling_500 < 0.90 OR > 0.99`, `vol_forecast_fallback_rate > 5%`).
- **T4 (4h)** : tests `tests/test_vol_telemetry.py` (mock registry, assertions).

**Acceptance criteria** :
- `/metrics` payload contient les 5 séries.
- Tests verts.
- Dashboard rendu sur Grafana local.

**Dépendances** : eval_16 metric registry (corrigée dans cat. observability).

---

### P1 — Calibration & qualité

#### P1-TCP-CONFORMAL-REGIME : Intervals conformes par régime — **20h**

**Objectif** : remplacer le TCP global par 3 buffers résiduels par régime HMM (low/normal/high) → coverage stratifié, intervals adaptés au régime.

**Tâches** :
- **T1 (8h)** : refactor `_tcp_residuals` (`volatility_forecaster.py:362-373`) en `Dict[str, deque]` indexé par `regime_state`. `_recompute_tcp_quantiles` (`:1084-1097`) calcule q_lower/q_upper par régime.
- **T2 (4h)** : `forecast()` choisit q_lower/q_upper selon le régime courant ; fallback global si buffer régime < 30.
- **T3 (4h)** : tests `tests/test_volatility_forecaster.py::test_conformal_intervals_per_regime`.
- **T4 (4h)** : bench coverage 2024 par régime : cible 93-97 % chaque régime.

**Acceptance criteria** :
- Coverage low / normal / high tous dans [0.93, 0.97].
- Width interval high > width interval low (sanité).

**Dépendances** : P0-BUG-HMM-REGIME.

---

#### P1-WALK-FORWARD-CI : Nightly walk-forward + DM test — **12h**

**Tâches** :
- **T1 (3h)** : GHA workflow `.github/workflows/vol_walkforward.yml` (nightly 03:00 UTC). Lance `scripts/eval_04_volatility.py --data data/XAU_15MIN_2019_2024.csv --sample-step 384` (≈25 min CI).
- **T2 (3h)** : assertion : `RMSE_har / RMSE_naive < 0.90` ET `DM p-value < 0.01`. Upload artefact `walkforward_summary.json`.
- **T3 (3h)** : badge README + slack-webhook si red.
- **T4 (3h)** : ajouter run sur EURUSD si CSV ≥ 95 % coverage.

**Acceptance criteria** :
- Workflow nightly vert 5 jours consécutifs.
- DM p-value < 0.01 sur 6 splits.

**Dépendances** : data EURUSD pour T4.

---

#### P1-CPCV-LGBM-LEAK : Combinatorial Purged CV pour LGBM — **16h**

**Objectif** : remplacer le split 80/20 simple (`volatility_lgbm.py:398-400` et `volatility_forecaster.py:1378-1396`) par CPCV (López de Prado 2018) pour audit institutional.

**Tâches** :
- **T1 (8h)** : implémenter `CombinatorialPurgedKFold(n_splits=6, embargo_pct=0.01)` (ou réutiliser depuis `src/research/` si déjà présent — cf. `reports/three_pillars_implementation_2026_05_13.md`).
- **T2 (4h)** : training LGBM via CPCV au lieu du 80/20 ; reporter mean ± std RMSE OOS, plus DSR (Deflated Sharpe Ratio) sur l'amélioration.
- **T3 (4h)** : test `tests/test_lgbm_vol.py::test_cpcv_paths_purged` (synthetic, vérifie embargo).

**Acceptance criteria** :
- CPCV produit 15 paths (C(6,2)=15).
- RMSE moyen CPCV ≤ 1.10 × RMSE 80/20 (perte tolérée pour gagner statistical validity).

**Dépendances** : aucune.

---

#### P1-DRIFT-DETECTION : Alerte drift RMSE rolling — **12h**

**Tâches** :
- **T1 (6h)** : nouveau composant `src/intelligence/vol_drift_monitor.py` : maintient `_actual_buffer` (deque(maxlen=2000)) et `_forecast_buffer`. Calcul RMSE rolling 500-bar toutes les 100 updates.
- **T2 (3h)** : si `rmse_rolling_500 > 1.5 × rmse_training` → log WARNING + métrique Prometheus `vol_drift_detected`.
- **T3 (3h)** : test `tests/test_vol_drift.py`.

**Acceptance criteria** :
- Drift simulé (changement de seed walk) déclenche alerte sous 200 updates.

**Dépendances** : P0-MONITORING (registry).

---

### P2 — Multi-asset & différenciation produit

#### P2-MULTI-ASSET-EUR : Calibrer & bencher EURUSD — **24h**

**Tâches** :
- **T1 (4h)** : vérifier `data/EURUSD_15MIN_*.csv` coverage (memory note 99.6 % depuis eval_05_09). Si OK → skip data work.
- **T2 (8h)** : adapter `InstrumentConfig` EURUSD (`volatility_forecaster.py:61-80`) : session_hours, `calendar_events` (Eurozone CPI, ECB Rate Decision), `sl_atr_mult=1.5`, `tp_atr_mult=3.0`.
- **T3 (6h)** : bench walk-forward EUR avec `scripts/eval_04_volatility.py --symbol EURUSD`.
- **T4 (4h)** : rapport `reports/eval_04/walkforward_eurusd.json` + comparatif XAU.
- **T5 (2h)** : intégrer EURUSD au CI nightly.

**Acceptance criteria** :
- EUR RMSE_har / RMSE_naive < 0.92 (cible moins ambitieuse que XAU car volatility plus modérée).
- DM p-value < 0.05.

**Dépendances** : P1-WALK-FORWARD-CI.

---

#### P2-VOL-CONE-PRE-EVENT : "Volatility cone" pré-NFP/FOMC — **16h**

**Objectif** : exposer un produit différenciant — vol cone affichant la dispersion historique des `forecast_atr` × `cal_mult` × régime sur les ±2h autour de NFP/FOMC/CPI. Utilisable B2B (Bloomberg-like) et B2C (Telegram pre-event alert).

**Tâches** :
- **T1 (6h)** : nouvelle API `VolatilityForecaster.cone(timestamp, hours=2, quantiles=(0.1,0.5,0.9))` → DataFrame.
- **T2 (4h)** : endpoint REST `/v1/vol/cone?symbol=XAUUSD&event=NFP&date=2026-06-07`.
- **T3 (3h)** : mockup Telegram (`mockups/vol_cone_telegram.md`).
- **T4 (3h)** : tests intégration.

**Acceptance criteria** :
- Endpoint répond p99 < 200 ms.
- Cone visuel cohérent (Q90 > Q50 > Q10, élargissement avant event).

**Dépendances** : P0-LATENCY-LGBM (pour cône lgbm).

---

#### P2-SIGNED-STATE : Signed pickle + checksum — **6h**

**Tâches** :
- **T1 (3h)** : `save_state` ajoute HMAC-SHA256 du payload sérialisé en queue. `load_state` vérifie avant `pickle.load`. Secret en `VOL_STATE_HMAC_KEY` env.
- **T2 (2h)** : refus chargement si mismatch.
- **T3 (1h)** : test `tests/test_volatility_forecaster.py::test_signed_state_tamper_detection`.

**Acceptance criteria** : tampered pickle refusé.

**Dépendances** : aucune.

---

#### P2-CLAMP-TIGHTEN : Resserrer clamp `[0.5×, 3×]` — **4h**

**Tâches** :
- **T1 (2h)** : changer `(0.2, 5.0)` → `(0.5, 3.0)` lignes 545 et 1497-1501.
- **T2 (2h)** : bench reproductibilité — vérifier que clamp activation reste < 1 %.

**Acceptance criteria** : pas de régression RMSE.

**Dépendances** : aucune.

---

## 5. Tests & validation

### 5.1 Unit tests (existants à conserver + étendre)
- `tests/test_volatility_forecaster.py` : ajouter B4 fix, conformal régime, signed state.
- `tests/test_lgbm_vol.py` : CPCV path tests, incremental features parity.
- `tests/test_hybrid_vol.py` : inchangé sauf retrait du défaut hybrid.
- `tests/test_vol_latency.py` : durcir asserts (p99 < 50 ms HAR ET LGBM).
- `tests/test_vol_telemetry.py` : nouveau, Prometheus registry.
- `tests/test_vol_drift.py` : nouveau.

### 5.2 Walk-forward
- `scripts/eval_04_volatility.py` (existe). Étendre : `--mode lgbm-fast` (post P0-LATENCY-LGBM), `--cpcv` flag.
- Sorties JSON dans `reports/eval_04/` : `walkforward_summary_v2.json` (post-fix B4).

### 5.3 Diebold-Mariano gating
- DM test bilatéral, loss = err², HAC bandwidth = h−1 = 4 (`np.cov` Newey-West).
- Gate CI : `DM(har vs naive) p-value < 0.01` ; `DM(lgbm vs har) p-value < 0.05` post-optim.

### 5.4 Régime stratification
- Bench RMSE / coverage stratifié par régime HMM (low/normal/high).
- Cible : RMSE_high < 1.3 × RMSE_normal (sinon modèle dégrade en stress).

### 5.5 Stress tests
- `tests/test_vol_stress.py` (nouveau) : (a) gap weekly XAU 2020-03 (COVID), (b) FOMC 2023-03 (SVB), (c) NFP 2022-07. Asserter `is_fallback=False`, latence p99 stable.

---

## 6. Sécurité

### 6.1 Model serving isolation
- Pickle load (`volatility_forecaster.py:1145-1217`) doit refuser tout chemin hors `state_dir` configurable (path traversal). Actuellement `Path(path).exists()` accepté ; ajouter `Path(path).resolve().is_relative_to(allowed_state_dir)`.
- Désactiver pickle (HMAC obligatoire post-P2-SIGNED-STATE).

### 6.2 Input validation OHLCV
- `forecast(ohlcv_df)` accepte n'importe quelle structure. Ajouter validateur Pydantic `OhlcvBar` : `high >= max(open, close, low)`, `low <= min(...)`, `volume >= 0`, monotonic timestamps.
- Reject si NaN > 5 % ou outlier price (>10σ rolling).
- Fichier : nouveau `src/intelligence/vol_input_validator.py`, appelé en tête de `_forecast_impl`.

### 6.3 Resource limits
- Cap window passé à `forecast()` : tronquer si `len(df) > 10 000` (sinon DoS via gros payload).
- Timeout par appel forecast (signal.alarm en POSIX, threading.Timer en Windows) : kill si > 500 ms et retour fallback.

### 6.4 Telemetry sans PII
- Vérifier qu'aucun chat_id Telegram, IP, API key ne fuit dans les labels Prometheus du vol_forecaster.

---

## 7. Métriques

### 7.1 Qualité
| KPI | Cible | Source |
|---|---|---|
| RMSE walk-forward HAR vs naïf | < 0.90 | nightly CI |
| RMSE walk-forward LGBM vs naïf | < 0.75 | nightly CI |
| DM p-value HAR vs naïf | < 0.01 | nightly CI |
| Coverage TCP empirique (par régime) | 93-97 % | telemetry rolling 500 |
| Drift RMSE rolling 30j | < 1.5 × RMSE training | telemetry |
| Stabilité année-sur-année | drift < 30 % | rapport annuel |

### 7.2 Performance
| KPI | Cible | Source |
|---|---|---|
| Latence forecast p50 | < 20 ms | Prometheus histogram |
| Latence forecast p95 | < 35 ms | idem |
| Latence forecast p99 | < 50 ms | idem |
| Latence calibrate (10k bars) | < 60 s | bench |
| RAM resident vol forecaster | < 200 MB | psutil |
| Disk state (par symbol) | < 500 KB | dir stat |

### 7.3 Robustesse
| KPI | Cible | Source |
|---|---|---|
| `forecast.is_fallback` rate | < 5 % | scanner |
| Clamp activation rate | < 1 % | telemetry |
| Régime distribution | ≠ 100 % `low` (post-fix B4) | telemetry |
| Pickle load failures | 0 | logs |

---

## 8. Risques & mitigations

| Risque | Probabilité | Impact | Mitigation |
|---|---|---|---|
| **R1**. Optimisation HAR incrémentale introduit divergence numérique vs batch | Moyenne | Élevé (RMSE +X%) | Test parity : `forecast_incremental(df[:i]) ≈ forecast(df[:i])` à 1e-6 près sur 1000 cas |
| **R2**. ONNX LGBM dégrade précision plancher | Moyenne | Élevé | Fallback pure-Python si delta RMSE > 1 %, kill-criterion documenté |
| **R3**. Fix B4 HMM change drastiquement coverage TCP | Moyenne | Moyen | Re-calibrer TCP post-fix, valider coverage rolling sur 6 mois OOS |
| **R4**. EURUSD calibration faillit (vol moins persistante) | Faible | Faible | Garder XAU comme produit phare v1.0, EUR en bêta |
| **R5**. Outlier LGBM split 1 (7.4 h fit) se reproduit en prod | Faible | Élevé (downtime) | Timeout calibration 1 h max ; bisect data si reproduit (B9) |
| **R6**. HMM pickle break entre versions hmmlearn | Élevé (1×/an) | Moyen | State_version_major += 1 à chaque bump hmmlearn ; re-calibrate auto si refusé |
| **R7**. Concurrent forecasts thread-safety | Faible (lock présent l.351) | Élevé | Audit `_lock` couverture ; ajouter tests `test_vol_concurrent_forecast.py` |
| **R8**. Drift régulatoire MiFID II "vol forecast = conseil en investissement" | Moyen | Élevé | Wrapper narrative compliance (eval_29) ; jamais exposer le forecast brut sans disclaimer |
| **R9**. CSV data Dukascopy zone grise commerciale | Élevé | Élevé | Pivot data source TwelveData/Polygon B2B-licensed (eval_08) |
| **R10**. Modèle HAR perd power post-2024 (régime géopolitique XAU) | Moyenne | Moyen | Re-calibrer mensuel ; drift monitor P1-DRIFT-DETECTION |

---

## 9. Dépendances

### 9.1 Data
- `data/XAU_15MIN_2019_2024.csv` (141 524 bars, 97.6 % coverage) — base actuelle.
- `data/economic_calendar_HIGH_IMPACT_2019_2025.csv` (875 events) — calendar.
- `data/EURUSD_15MIN_*.csv` (99.6 % coverage selon memory) — à valider P2.
- Pivot data source long terme : voir catégorie 8 (Data Providers, eval_08).

### 9.2 ML stack (`requirements.txt`)
- `lightgbm >= 4.0` (déjà installé)
- `hmmlearn >= 0.3` (déjà installé)
- `scikit-learn >= 1.3` (LinearRegression)
- `onnxmltools` + `onnxruntime` (NOUVEAU si P0-LATENCY-LGBM choisit ONNX path)
- `treelite` (NOUVEAU si LGBM Booster compact insuffisant)

### 9.3 Deployment
- Prometheus client lib (existe, eval_16 le réactive).
- Grafana dashboard hosting.
- GHA matrix runner Linux + Windows (test_vol_latency portable).

### 9.4 Inter-catégorie
- **Catégorie 1 (Architecture)** : SignalStore + threading model. Lock sur `_vol_forecaster` partagé scanner ↔ calibration.
- **Catégorie 5 (Observability)** : registry Prometheus instanciée (G4 G7 dépendent du fix eval_16).
- **Catégorie 8 (Data Providers)** : pivot données live (G5 EUR).
- **Catégorie 9 (Compliance)** : wrapping narrative légal autour des intervals TCP (R8).
- **Catégorie 7 (Testing/CI)** : workflow GHA nightly (P1-WALK-FORWARD-CI).

---

## 10. Estimation totale & timeline

### 10.1 Budget heures par phase

| Phase | Tâches | Heures min | Heures max |
|---|---|---|---|
| **P0 — bloquants go-live** | LATENCY-HAR, LATENCY-LGBM, BUG-HMM, DECISION, MONITORING | 62 | 76 |
| **P1 — qualité commercialisation** | TCP-REGIME, WALK-FWD-CI, CPCV, DRIFT | 60 | 60 |
| **P2 — multi-asset & diff produit** | MULTI-ASSET-EUR, VOL-CONE, SIGNED-STATE, CLAMP | 50 | 50 |
| **Sous-total** | | **172** | **186** |
| Buffer 15 % (review, regressions, doc) | | 26 | 28 |
| **Total v1.0 commercialisation vol** | | **198** | **214** |

### 10.2 Timeline solo (8h/sem dispo dev profond)

| Sem | Phase | Livrable |
|---|---|---|
| S1-S2 (32h) | P0-LATENCY-HAR + P0-BUG-HMM | HAR p99 < 50 ms ; bug régime fixé |
| S3-S5 (48h) | P0-LATENCY-LGBM + P0-DECISION | LGBM p99 < 50 ms ; mode défaut validé |
| S6 (16h) | P0-MONITORING | Telemetry vol forecast en prod |
| S7-S8 (32h) | P1-TCP-REGIME + P1-CPCV | Conformal stratifié + audit CPCV |
| S9 (12h) | P1-WALK-FWD-CI + P1-DRIFT | Nightly CI vert |
| S10-S12 (50h) | P2-MULTI-ASSET-EUR + P2-VOL-CONE + P2-SIGNED + P2-CLAMP | XAU+EUR ; produit cône ; sécurité durcie |
| S13 (8h) | Buffer / regression freeze | RC v1.0 vol forecaster |

**Calendrier réaliste** : **13 semaines = ~3 mois** (commercialisation MVP fin août 2026 si départ semaine 22).

### 10.3 Parallélisation possible
- 3 dev parallèles : P0-LATENCY-HAR (dev A), P0-BUG-HMM (dev B), P0-MONITORING (dev C) → 6 sem au lieu de 13.

### 10.4 Critères de release v1.0
- [ ] Tous les KPI section 7.1 et 7.2 verts.
- [ ] Walk-forward CI vert 7 jours consécutifs sur XAU.
- [ ] Couverture tests > 75 % sur `volatility_forecaster.py` et `volatility_lgbm.py` (eval_17 cible 80 % zones revenue).
- [ ] Stress tests passants (R5 R6 R7).
- [ ] Dashboard Grafana + alerting opérationnels.
- [ ] Audit pickle/HMAC + path validation.
- [ ] Reproductibilité walk-forward documentée (`scripts/eval_04_volatility.py` + seed fixed).
- [ ] Disclaimer compliance autour des intervals (eval_29 W4).

---

## Annexes

### A. Fichiers à modifier (récap)

| Fichier | Lignes ciblées | Tâches |
|---|---|---|
| `src/intelligence/volatility_forecaster.py` | 487-573 (`_forecast_impl`), 641-678 (`_add_features`), 909-931 (`_get_regime_multiplier`), 1084-1097 (`_recompute_tcp_quantiles`), 1110-1217 (persistence) | P0-LATENCY-HAR, P0-BUG-HMM, P1-TCP-REGIME, P2-SIGNED-STATE, P2-CLAMP |
| `src/intelligence/volatility_lgbm.py` | 105-174 (`build_features`), 354-462 (`_train_impl`), 480-502 (predict), 520-575 (persistence) | P0-LATENCY-LGBM, P1-CPCV |
| `src/intelligence/sentinel_scanner.py` | 405-419 (forecast call) | P0-MONITORING, R7 thread-safety |
| `src/intelligence/security.py` | 212-260 (vol_mode default) | P0-DECISION |
| `src/intelligence/main.py` | 537-557 (vol_mode env) | P0-DECISION |
| `scripts/eval_04_volatility.py` | tout | flags `--cpcv`, `--mode lgbm-fast` |
| `tests/test_vol_latency.py` | 83-139 | durcir asserts |
| Nouveaux | — | `src/intelligence/vol_input_validator.py`, `src/intelligence/vol_drift_monitor.py`, `tests/test_vol_telemetry.py`, `tests/test_vol_drift.py`, `tests/test_vol_stress.py`, `infrastructure/grafana/vol_forecaster.json`, `.github/workflows/vol_walkforward.yml` |

### B. Références internes
- `reports/eval_04_volatility.md` — audit complet (5.0/10).
- `reports/eval_04/walkforward_summary.json` — n=373 forecasts XAU 2022-2024.
- `reports/eval_04/footprint.json` — RAM / disk / latence.
- `reports/governance/kill_criteria_board.md` — KPI REGIME-1.1.
- `memory/MEMORY.md` — section "Volatility Forecasting" et eval_04 2026-04-29.

---

---

**Chemin** : `C:/MyPythonProjects/TradingBOT_Agentic/reports/commercialization_sprint/04_volatility_forecasting.md`

**Top 3 P0** :
1. P0-LATENCY-HAR — refactor `_add_features` incrémental → p99 < 50 ms (12-18h)
2. P0-LATENCY-LGBM — feature cache + ONNX/Booster compact → p99 < 50 ms (24-32h)
3. P0-BUG-HMM-REGIME — fix `_get_regime_multiplier` régime classé barre cible (8h)

**Heures totales v1.0 commercialisation vol** : **198-214h** (P0=62-76h, P1=60h, P2=50h, buffer 15%).
