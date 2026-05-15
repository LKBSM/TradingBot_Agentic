# Backlog — QA & Robustness

**Date** : 2026-05-15
**Owner** : QA & Robustness

## Sprint 0 (livré, archive)

- [x] CI minimale `.github/workflows/algo_tests.yml` — done commit `7ff3180`
- [x] Test régression `test_data_quality_bos_regression.py` (3 tests verts) — done commit `66c1a53`

## Sprint 1 (S3-S4) — Data Layer Hardening (coverage baseline)

- [ ] Coverage measurement baseline : `pytest --cov=src --cov-branch --cov-report=html` ; rapport `audits/2026-Q3/coverage_sprint_1.md` — 2 h
- [ ] Property-based test resampling MTF (no look-ahead) — Hypothesis (coordination Data Quality P0-7) — 6 h
- [ ] Property-based test data provider contract (Pydantic v2 NaN/Inf/gaps) (coordination Data Quality P0-8) — 4 h
- [ ] Workflow `.github/workflows/coverage.yml` : upload Codecov + comment PR avec delta — 3 h
- [ ] Audit suite tests existants : marquer flaky avec `@pytest.mark.flaky(reruns=3)` ; documenter dans `tests/known_flaky.md` — 3 h
- [ ] Élever coverage `volatility_forecaster.py` 63 % → 80 % en parallèle des fix Vol Modeler — 6 h

## Sprint 2 (S5-S6) — Detection Engine Validation

- [ ] Test framework for annotations validation (F1 scoring) — `tests/test_smc_annotations_framework.py` (coordination SMC Lead batch 2.2) — 8 h
- [ ] Snapshot tests for visual SMC detection (audit visuel automatisé batch 2.4) : framework matplotlib snapshot diff — 10 h
- [ ] Coverage smart_money/ modules ≥ 90 % branche — 4 h
- [ ] Property-based : SMC detector invariance par scaling prix ×10, par timezone shift — 4 h

## Sprint 3 (S7-S8) — Statistical Edge Discovery

- [ ] Coverage `src/backtest/validation/` ≥ 90 % branche (coordination Stat Validator + Backtest Infra) — 6 h
- [ ] Test `tests/test_validation_chain_e2e.py` : assert que `run_backtest --validate` produit `ValidationReport.json` valide avec tous les champs DSR/PBO/PF_lo/DM — 4 h
- [ ] Property-based : CPCV split disjoint (group i et group j n'ont aucun timestamp en commun) — 3 h
- [ ] Test régression métriques (Calmar annualisé, Sharpe stdev, max_consec_losses, Lo 2002) avec fixtures contrôlées — 4 h

## Sprint 4 (S9-S10) — Calibration & Confidence

- [ ] Property tests Mondrian conformal (coverage holds across strata) (coordination Conformal Eng P1-9) — 4 h
- [ ] Tests `LogisticL1Scorer` fit/predict + sparsity invariants (coordination Stat Validator P0-1) — 4 h
- [ ] Test `tests/test_conformal_picp_regression.py` : fixture XAU 2024, PICP ∈ [78%, 82%] — 3 h
- [ ] Coverage `conformal_wrapper.py` ≥ 90 % branche — 3 h

## Sprint 5 (S11-S12) — Robustness & Stress Testing (rôle pivot)

- [ ] **Chaos tests state machine** (P1-11, coordination State Machine Eng) : oscillation, opposing flips, cooldown, max_age, lockout. Hypothesis 10⁶ tirages — 10 h
- [ ] **Fuzz harness** `tests/fuzz/` (batch 5.1) :
  - `test_data_provider_fuzz.py` (NaN, inf, gaps, spread spikes, timestamps désordonnés) — 5 h
  - `test_state_machine_fuzz.py` (scores adversariaux) — 4 h
  - `test_vol_forecaster_fuzz.py` (returns extrêmes, prix négatifs simulés) — 4 h
  - `test_backtest_fuzz.py` (leverage 1000×, equity négatif) — 4 h
- [ ] **Adversarial fake-out tests** (batch 5.4) : générer setups "near miss" (BOS d'un cent, FVG d'un spread) ; vérifier rejection — 6 h
- [ ] Stress test suite e2e : COVID 2020, LDI 2022, SVB 2023, yen 2024 sur XAU + EURUSD (batch 5.2 input) — 5 h
- [ ] Rapport `audits/2026-Q3/robustness_report_sprint_5.md` — 4 h

## Sprint 6 (S13-S14) — Production Hardening

- [ ] **Mutation testing setup** : `mutmut` config dans `setup.cfg` ; cible modules critiques (state_machine, smart_money, validation, vol_forecaster, conformal, regime) — 4 h
- [ ] Workflow `.github/workflows/mutation.yml` : hebdomadaire (cron) — 2 h
- [ ] Mutation campaign : run sur 6 modules critiques, atteindre ≥ 70 % score (itérations + ajouts de tests pour les survivants) — 16 h
- [ ] Coverage branch ≥ 90 % campaign : combler les gaps identifiés Sprints 1-4 — 8 h
- [ ] Type hints coverage 69 % → 90 % (P2) : ajouter `from __future__ import annotations` + types missing ; `mypy --strict` sur modules cœur — 8 h
- [ ] Rapport `audits/2026-Q4/mutation_score_sprint_6.md` — 3 h

## Sprint 7 (S15-S16) — Commercial Readiness

- [ ] Suite e2e 6 actifs × 2 TF — coordination Backtest Infra batch 7.4 — 4 h
- [ ] Documentation `docs/algo/testing_strategy.md` (architecture tests, pyramide, references) — 6 h
- [ ] Certification QA signoff : checklist 50 points (coverage, mutation, fuzz, property, e2e, doc) ; signed `audits/2026-Q4/qa_certification.md` — 4 h
- [ ] Contribution à `audits/2026-Q4/certification.md` section "QA & robustness" (LQA consolide) — 2 h

## Inbox (non priorisé)
- Cosmic-ray vs mutmut benchmark (alternative mutation tool).
- Bandit (security scanner) — sortie hors algo strict mais utile.
- Pre-commit hooks (`.pre-commit-config.yaml`) — coordination LQA.
- Coverage report visualization dans tear sheets clients.
- Stryker (JS mutation) si stack TypeScript ajoutée — hors périmètre.
- Performance regression tests automatisés (CI bench Backtest Infra) — coordination.
- Visual regression tests (PNG diff) pour snapshots SMC — Sprint 2 base.
