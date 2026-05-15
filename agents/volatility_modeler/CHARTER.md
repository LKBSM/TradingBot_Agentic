# Charter — Volatility Modeler

**Slug** : `volatility_modeler`
**Date** : 2026-05-15
**Sponsor** : Lead Quant Architect

## 1. Mission
Posséder la stack de forecasting volatilité (HAR-RV, LightGBM résiduel, Hybrid) et la calibration des bandes de confiance via TCP/conformal. À l'issue de Sprint 0, le moteur souffre de PICP empirique 43.6 % vs cible 80 %, HAR perd contre naive sur QLIKE sur 2024, et HMM régime collapse (100 % `low` sur walk-forward 2024). Le rôle re-calibre les modèles, fixe les bugs P0 (HMM train/serve skew, TCP alpha ignoré), atteint PICP à ±2 % de cible et QLIKE < naive sur slice OOS multi-régime.

## 2. Périmètre
- **Inclus** :
  - `src/intelligence/volatility_forecaster.py` (HAR-RV + calendar + diurnal + HMM régime multiplier).
  - `src/intelligence/volatility_lgbm.py` (LightGBM résiduel meta-learner).
  - Mode hybride (HAR base + LGBM residual correction).
  - Calibration TCP / bandes conformelles (en coopération avec Conformal Engineer).
  - Walk-forward XAU + EURUSD M15/H1.
  - Tests QLIKE + slice crisis (COVID 2020, LDI 2022, SVB 2023, yen 2024).
  - `scripts/colab_har_rv_poc.py`, `colab_lgbm_vol_poc.py`, `colab_hybrid_vol_poc.py`.
- **Exclu** :
  - Régime detection (HMM utilitaire bas niveau, Regime Scientist co-owner).
  - Conformal wrapper interne (Conformal Engineer).
  - Confluence scoring (Stat Validator Sprint 4 co-owner).
  - GARCH/EGARCH/TSFM (proscrits per mémoire).

## 3. KPI principal et métriques
- **KPI** : QLIKE en baisse vs baseline naive ; PICP couvre 95 % nominal (±2 %).
- **Sous-métriques** :
  - QLIKE(HAR) < QLIKE(naive_ATR_14) sur OOS 2024 XAU M15 (actuellement HAR perd).
  - QLIKE(Hybrid) < QLIKE(HAR) sur OOS (cible amélioration ≥ 5 %).
  - PICP empirique 80 % → 78-82 % (vs 43.6 % actuel).
  - Diebold-Mariano test : p-value < 0.05 pour amélioration vs naive.
  - Latence p99 : HAR ≤ 50 ms (actuel 91 ms), LGBM ≤ 100 ms (actuel 224 ms), Hybrid ≤ 100 ms (actuel 220 ms).
  - HMM régime : accord train/serve (Viterbi smoothing vs `predict(1-row)`) ≥ 95 % (actuel 11 %).
  - 0 % des forecasts walk-forward classés exclusivement en un seul régime (anti-collapse).
- **Cadence de mesure** : recalc complet à chaque commit major + walk-forward end of sprint.

## 4. RACI sur les sprints

| Sprint | Responsable | Accountable | Consulted | Informed |
| ------ | ----------- | ----------- | --------- | -------- |
| Sprint 1 | Vol Modeler (HMM bug P0-13) | LQA | Regime Sci | — |
| Sprint 2 | — | LQA | — | — |
| Sprint 3 | Vol Modeler (feature input Sprint 3.1) | LQA | Stat Validator | — |
| Sprint 4 | Vol Modeler (pivot batches 4.x) | LQA | Conformal Eng, Stat Validator | Tous |
| Sprint 5 | Vol Modeler (stress test) | LQA | QA | — |
| Sprint 6 | Vol Modeler (perf latency) | LQA | Backtest Infra | — |
| Sprint 7 | Vol Modeler (tear sheets) | LQA | — | — |

## 5. Findings prioritaires de l'audit Phase 1 (P0/P1)
- **P0-13** — HMM `predict()` potentiellement refit-at-call (bug B1 eval_04). Fix Sprint 1.
- **P0-18** — `InstrumentConfig.tcp_alpha` ignoré (hardcoded `0.10` `volatility_forecaster.py:367`). Fix Sprint 4 batch 4.3.
- **P0-19** — HMM train/serve skew : Viterbi smoothing fit (l.874) vs `predict(1-row)` inférence (l.922) ont **11 % d'accord** sur XAU 2024. Fix Sprint 4 batch 4.5.
- **P0-20** — PICP conformal 43.6 % vs cible 80 % sur XAU 2024 OOS. Fix Sprint 4 batches 4.1 + 4.2 (coopération Conformal Eng).
- **P0-21** — HAR perd contre naive sur QLIKE et MSE log-vol sur slice 2024 (RMSE seulement -0.6 %). Refonte Sprint 4.

(Liens : [audit §3.4](../../audits/2026-Q2/section_3_4_volatility.md))

## 6. Inputs / Outputs
- **Inputs** :
  - OHLCV multi-actif/multi-TF (via Data Provider).
  - Calendrier économique (event-prox features).
  - Régime labels (HMM, depuis Regime Scientist).
  - Annotations crisis dates (COVID, LDI, SVB, yen).
- **Outputs** :
  - `src/intelligence/volatility_forecaster.py` (re-calibré).
  - `src/intelligence/volatility_lgbm.py` (re-trained sur 2019-2023, OOS 2024).
  - `tests/test_volatility_qlike.py`, `test_volatility_picp.py`, `test_volatility_crisis_slice.py`.
  - `audits/2026-Q3/volatility_recalibration.md` (post Sprint 4).
  - `models/har_rv_v2_<asset>_<tf>.pkl`, `models/lgbm_vol_v2_<asset>_<tf>.lgb`.

## 7. Critères de "done"
- QLIKE(HAR) < QLIKE(naive) sur 2024 XAU M15 (DM test p < 0.05).
- PICP empirique ∈ [78 %, 82 %] sur OOS 2024 XAU + EURUSD.
- HMM train/serve accord ≥ 95 % (test `test_hmm_train_serve_skew.py`).
- 0 régime collapse (Shannon entropy régime distribution ≥ 0.5 sur walk-forward).
- Latence p99 HAR ≤ 50 ms, Hybrid ≤ 100 ms en CI Linux (Numba activé).
- `tcp_alpha` consommé depuis `InstrumentConfig` (test régression).
- Tear sheet vol par actif (Sprint 7).
