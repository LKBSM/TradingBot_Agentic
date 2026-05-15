# Backlog — Statistical Validator

**Date** : 2026-05-15
**Owner** : Statistical Validator

## Sprint 1 (S3-S4) — Data Layer (prep)

- [ ] Audit `src/research/cpcv_harness.py` (507 LOC) + `strategy_gates.py` : documenter l'API actuelle, identifier le couplage manquant avec `src/backtest/` (P0-17) — 4 h
- [ ] Designer `src/backtest/validation/` interface : `validate_strategy(trades, returns, n_groups=10, n_test_groups=2) -> ValidationReport` — 3 h
- [ ] Lecture théorique : López de Prado AFML chap 7 (CPCV), chap 11 (DSR), chap 14 (PBO) — 4 h
- [ ] Coordination Data Quality : confirmer que `signal_id` reproductible (hash déterministe) est en place — 1 h

## Sprint 2 (S5-S6) — Detection Validation (support SMC)

- [ ] Calcul F1 / precision / recall vs annotations SMC (input SMC Lead) avec bootstrap CI (batch 2.2) — 5 h
- [ ] Mise en place reliability diagrams pour BOS/CHOCH detection (cible monotone) — 3 h
- [ ] Test statistique : SMC firing rate par régime — chi-squared independence test — 3 h

## Sprint 3 (S7-S8) — Statistical Edge Discovery (rôle pivot)

- [ ] **Batch 3.1 — Feature engineering exhaustif** (coordination tous owners) : recevoir features Data + SMC + Vol + Regime + Conformal, produire dataset unifié `features_<asset>_<tf>.parquet` — 6 h
- [ ] **Batch 3.2 — Information Coefficient par feature isolée** : calcul IC bootstrap CI sur OOS 2024, ranking top-50 features — 6 h
- [ ] **Batch 3.3 — Stacking + conditionnement par régime** : implémenter logistic L1 par stratum régime, comparer vs unstratified — 6 h
- [ ] **Batch 3.4 — Couplage CPCV/DSR/PBO** (P0-17) : créer `src/backtest/validation/` ; wrapper `scripts/run_backtest.py` pour invoquer la chaîne automatiquement — 10 h
- [ ] Tests `tests/test_validation_chain.py` : trace que chaque backtest run produit un `ValidationReport.json` avec DSR, PBO, PF_lo, DM_p — 4 h
- [ ] Coordination Backtest Infra : grille de sweep (state machine + features) doit s'exécuter en parallèle (joblib) — 3 h
- [ ] **Gate Sprint 3 signature** : analyse résultats CPCV+DSR+PBO sur 4 actifs × 4 TF ; verdict GO (PF_lo > 1.0 CI 95 % sur ≥ 1 cellule) OU pivot — 4 h
- [ ] Rapport `audits/2026-Q3/sprint_3_validation_report.md` — 4 h

## Sprint 4 (S9-S10) — Calibration & Confidence (rôle pivot)

- [ ] **Batch 4.2 — Refonte ConfluenceDetector** (P0-1, co-owner Vol Modeler + LQA) : logistic regression L1 sur 8 features pondérées + cross-val ; remplacer `confluence_detector.py` scoring fn — 12 h
- [ ] Cible Brier skill ≥ +0.03 sur OOS 2024 ; reliability diagram monotone (Spearman ≥ 0.7) — 4 h
- [ ] Feature selection L1 doit résoudre P1-5 (OB ↔ Retest dupliquée) : tester si coefficient de l'un des deux passe à 0 — 2 h
- [ ] Tests régression `tests/test_confluence_l1.py` : Brier ≥ +0.03, sklearn LR L1 prouvée monotone — 4 h
- [ ] Coordination Conformal Eng : conformal sur P(win|features) downstream du logistic L1 (input pour Mondrian) — 3 h
- [ ] Arbitrage P1-4 (double-gating 96.3 % rejection) : avec un score calibré L1, faut-il garder enter=75 OU baisser ? Sweep en coordination avec State Machine — 3 h
- [ ] Rapport `audits/2026-Q3/confluence_l1_refit.md` — 4 h

## Sprint 5 (S11-S12) — Robustness & Stress Testing

- [ ] Validation stress test multi-régime : DSR / PBO calcul sur slice COVID 2020, LDI 2022, SVB 2023, yen 2024 (batch 5.2) — 6 h
- [ ] Reality Check (White 2000 + Hansen SPA) : tester si edge survive après multiple testing correction (toutes les cellules de sweep Sprint 3) — 6 h
- [ ] Sensibilité hyperparams ±20 % sur logistic L1 (batch 5.3) — 4 h
- [ ] Bootstrap CI sur métriques OOS — 3 h

## Sprint 6 (S13-S14) — Production Hardening

- [ ] Versioning métriques : chaque tear sheet inclut commit_sha, data_hash, validation_version (P0-16 support) — 2 h
- [ ] Cache validation results : invalidation par hash dataset + commit (batch 6.1) — 3 h
- [ ] Documentation `docs/algo/validation.md` (CPCV theory, DSR formula, PBO méthodologie, références López de Prado) — 5 h

## Sprint 7 (S15-S16) — Commercial Readiness (rôle pivot)

- [ ] **Certification finale** : passer la chaîne CPCV+DSR+PBO+DM+RealityCheck sur les 6 actifs × 2 TF (batch 7.4) — 8 h
- [ ] Tear sheets validation par actif/TF : signed off avec verdict GO / PIVOT / NO-GO (batch 7.2) — 8 h
- [ ] Fiche transparence client : "comment nous validons statistiquement nos signaux" (batch 7.3) — 4 h
- [ ] Contribution `audits/2026-Q4/certification.md` — section validation (LQA consolide) — 4 h

## Inbox (non priorisé)
- Hansen SPA test (Superior Predictive Ability) — alternative à Reality Check.
- Romano-Wolf StepM pour multiple testing.
- Bonferroni-Holm corrections sur multiple strategies.
- Sharpe ratio non-IID correction (Lo 2002).
- Probabilistic Sharpe Ratio (PSR) — Bailey-López de Prado.
- Deflated Sharpe Ratio with skewness/kurtosis — extensions.
- Block bootstrap pour autocorrelation (Politis-Romano).
- Online IC monitoring (drift detection production).
