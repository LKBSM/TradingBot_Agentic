# Backlog — QA & Robustness

## Sprint 0
- [x] CI minimale `.github/workflows/algo_tests.yml` — done commit 7ff3180
- [x] Test régression BOS firing rate — done commit 66c1a53

## Sprint 1
- [ ] Property-based test resampling MTF (no look-ahead) (6 h)
- [ ] Coverage measurement baseline (pytest --cov, 2 h)

## Sprint 2
- [ ] Test framework for annotations validation (F1 scoring) (8 h)
- [ ] Snapshot tests for visual SMC detection (audit visuel automatisé) (10 h)

## Sprint 4
- [ ] Property tests Mondrian conformal (coverage holds across strata) (4 h)
- [ ] Tests `LogisticL1Scorer` fit/predict + sparsity invariants (4 h)

## Sprint 5
- [ ] Chaos tests state machine (oscillation, opposing flips, cooldown) (8 h)
- [ ] Fuzz inputs (NaN, inf, gaps, spread spikes) (10 h)
- [ ] Adversarial fake-out tests (6 h)

## Sprint 6
- [ ] Mutation testing setup (mutmut config) (4 h)
- [ ] Coverage branch ≥ 80 % campaign (8 h)
- [ ] Type hints coverage 69 % → 90 % (8 h)

## Sprint 7
- [ ] Suite e2e 6 actifs × 2 TF (4 h)
- [ ] Certification QA signoff (2 h)
