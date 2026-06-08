# Backlog — Volatility Modeler

**Date** : 2026-05-15
**Owner** : Volatility Modeler

## Sprint 1 (S3-S4) — Data Layer Hardening (hotfixes HMM)

- [ ] **Fix P0-13** : HMM `predict()` refit-at-call. Audit `volatility_forecaster.py` ligne 874-922 ; ré-architecturer pour fit one-shot + cached predict (input Regime Scientist) — 4 h
- [ ] Test régression `tests/test_hmm_no_refit_at_call.py` : monkeypatch HMM `.fit()` doit être appelé ≤ 1× par session — 2 h
- [ ] Audit empirique : sur XAU 2024 OOS, vérifier que HMM ne réfait pas son `.fit` à chaque prédiction (logging timing) — 2 h
- [ ] Coordination Regime Scientist : décider qui possède le canonical HMM utility (probable : Regime Scientist, Vol Modeler en lit le label) — 1 h

## Sprint 2 (S5-S6) — Detection Validation (frozen)

- [ ] Pas de livrable nouveau Vol.

## Sprint 3 (S7-S8) — Edge Discovery (feature input)

- [ ] Exposer features volatility à Stat Validator pour Sprint 3 batch 3.1 : `vol_forecast_h1`, `vol_forecast_h4`, `vol_regime_label`, `vol_pct_change`, `realized_vol_last_20`, `event_prox` — 4 h
- [ ] Information Coefficient (IC) par feature vol vs returns OOS — 4 h
- [ ] Documenter le contrat `VolatilityFeatures` (Pydantic v2) — 2 h

## Sprint 4 (S9-S10) — Calibration & Confidence (rôle pivot)

- [ ] **Batch 4.1** — Refonte HAR-RV : recalibration weights par actif (XAU, EURUSD, BTCUSD si CSV) ; tester variantes HAR-RV-J (Andersen-Bollerslev jumps) et HAR-RV-CJ (continuous + jump components) (P0-21) — 8 h
- [ ] **Batch 4.2** — Validation OOS QLIKE : implémenter `qlike(y_true, y_pred)` proprement (vs naive ATR_14, naive last_realized), DM test (P0-21) — 5 h
- [ ] **Batch 4.3** — Fix P0-18 : `tcp_alpha` consommé depuis `InstrumentConfig` (remplacer le `0.10` hardcoded ligne 367). Test régression — 2 h
- [ ] **Batch 4.4** — Recalibration LightGBM résiduel : refit sur 2019-2023, OOS strict 2024 ; éviter contamination calendar bias — 8 h
- [ ] **Batch 4.5** — Fix P0-19 : HMM train/serve skew. Unifier la procédure (Viterbi smoothing ou online predict — pas les deux). Cible accord ≥ 95 % (input Regime Scientist) — 6 h
- [ ] **Slice crisis tests** : QLIKE / RMSE sur COVID 2020 (mars), LDI 2022 (sept), SVB 2023 (mars), yen 2024 (avril). Cible : ne pas dégrader > +30 % vs régime calme — 6 h
- [ ] Coopération Conformal Eng pour PICP 80 % : fournir résiduals propres (homoscédastiques après normalisation par forecast) — 4 h
- [ ] Rapport `audits/2026-Q3/volatility_recalibration.md` — 4 h
- [ ] Décision : `VOL_MODE=har` reste défaut OU Hybrid retrouve sa place si recalibrate atteint PICP ≥ 78 %. Document dans `sprint_4_decisions.md` — 1 h

## Sprint 5 (S11-S12) — Robustness & Stress Testing

- [ ] Stress vol forecasting : 4 slices crisis × 4 actifs (batch 5.2) — 6 h
- [ ] Sensibilité hyperparamètres ±20 % (batch 5.3) — 4 h
- [ ] Fuzz : injecter returns extrêmes (sigma 10×), vérifier que forecast ne diverge pas — 3 h
- [ ] Property-based : forecast(volatility) > 0 toujours ; monotonicité conditionnelle (input plus volatile → output plus volatile) — 3 h

## Sprint 6 (S13-S14) — Production Hardening

- [ ] **Batch 6.1** — Optimisation latence : profiler avec `cProfile` + `line_profiler` ; cible HAR p99 ≤ 50 ms, Hybrid ≤ 100 ms (vs 91/220 ms actuels). Vectoriser `event_prox` (B2 déjà partiellement fixé) — 6 h
- [ ] Validate Numba dans containers prod / Railway (input Data Quality + Backtest Infra) — 2 h
- [ ] Versioning modèles : `models/har_rv_v2_<asset>_<tf>_<commit>.pkl` (P0-16 support) — 2 h

## Sprint 7 (S15-S16) — Commercial Readiness

- [ ] Tear sheet vol par actif : forecast vs realized, PICP plot, QLIKE evolution, latence p99 (batch 7.2) — 5 h
- [ ] Documentation `docs/algo/volatility.md` (HAR-RV theorique, calibration, références Corsi 2009, Andersen-Bollerslev) — 4 h
- [ ] Fiche transparence client : "comment nous mesurons la volatilité" (batch 7.3) — 2 h

## Inbox (non priorisé)
- HAR-Q (Bollerslev-Patton-Quaedvlieg 2016) variante avec quarticity — comparer à HAR-RV.
- Realized kernels (Barndorff-Nielsen) si tick data disponible (différé Sprint 8+).
- BiPower variation vs realized variance pour décomposition jumps.
- Multivariate vol (DCC-MGARCH) pour cross-asset — hors périmètre.
- Online learning LightGBM (incremental) pour drift réel — réserve.
- Évaluer si VOL_MODE=ensemble (HAR + LGBM weighted) > Hybrid (residual stacking).
