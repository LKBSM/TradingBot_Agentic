# Charter — QA & Robustness

**Slug** : `qa_robustness`
**Date** : 2026-05-15
**Sponsor** : Lead Quant Architect

## 1. Mission
Garantir que l'algorithme Smart Sentinel AI est robuste, testé en profondeur et résiste aux inputs adversariaux. À l'issue de Sprint 0, la suite compte 2 696 tests (eval_17 — 5.5/10) mais la couverture est inégale (telegram 10 %, vol_forecaster 63 %), il y a 0 mutation testing systématique, 0 fuzz harness, 0 property-based en CI. Le rôle livre Sprint 1-3 l'élévation de la couverture branche à ≥ 90 % sur modules critiques, Sprint 5 le harness fuzz + property-based + adversarial (batch 5.1, 5.4), Sprint 6 le mutation testing avec score ≥ 70 % sur les modules critiques (state machine, smart_money, validation chain). Le rôle est gardien de la non-régression : aucune PR ne merge si elle casse la CI.

## 2. Périmètre
- **Inclus** :
  - `tests/` complets (unitaires + intégration + property-based + fuzz + mutation).
  - `.github/workflows/algo_tests.yml` + extensions (coverage, mutmut).
  - Hypothesis (property-based).
  - `mutmut` ou `cosmic-ray` (mutation testing).
  - Fuzz harness inputs algo (NaN, infinis, gaps, spreads anormaux, leverage extrême).
  - Adversarial inputs (fake-out setups, manipulation tentatives).
  - Documentation tests par module (`docs/algo/testing_strategy.md`).
  - Type hints coverage (P2).
- **Exclu** :
  - Implémentation modules métier (chaque owner les teste, QA les passe en revue).
  - Performance benchmarks (Backtest Infra).
  - Statistical validation des outputs (Stat Validator).
  - Tests E2E delivery / Telegram / API (out of algo scope).

## 3. KPI principal et métriques
- **KPI** : couverture branche ≥ 90 % ; mutation score ≥ 70 %.
- **Sous-métriques** :
  - Coverage branche par module critique : `state_machine.py`, `smart_money/*`, `volatility_forecaster.py`, `conformal_wrapper.py`, `validation/*`, `regime_*` ≥ 90 %.
  - Coverage ligne global ≥ 85 %.
  - Mutation score (`mutmut`) sur modules critiques ≥ 70 %.
  - Property-based tests Hypothesis : 10⁶ tirages stables sur state machine.
  - Fuzz harness : 10⁴ inputs aléatoires sans crash (data provider, state machine, vol forecaster).
  - 0 test marqué `xfail` non documenté.
  - 0 régression test green-to-red sur 1 sprint (mesuré en fin de sprint).
  - CI GitHub Actions : runtime < 15 min, success rate ≥ 95 %.
  - 0 flaky test non-marqué.
  - Type hints coverage ≥ 90 % (P2, Sprint 6).
- **Cadence de mesure** : continue (CI sur chaque PR) + recalc coverage end of sprint, mutation hebdomadaire (lourd).

## 4. RACI sur les sprints

| Sprint | Responsable | Accountable | Consulted | Informed |
| ------ | ----------- | ----------- | --------- | -------- |
| Sprint 1 | QA (élévation coverage) | LQA | Data Quality, SMC | — |
| Sprint 2 | QA (validation tests SMC + snapshot tests) | LQA | SMC | — |
| Sprint 3 | QA (coverage backtest+validation) | LQA | Backtest Infra, Stat Validator | — |
| Sprint 4 | QA (tests conformal + score L1) | LQA | Conformal Eng, Stat Validator | — |
| Sprint 5 | QA (rôle pivot fuzz + property + adversarial) | LQA | Tous owners | Tous |
| Sprint 6 | QA (mutation testing + type hints) | LQA | State Machine, Stat Validator | — |
| Sprint 7 | QA (certification tests) | LQA | LQA | — |

## 5. Findings prioritaires de l'audit Phase 1 (P0/P1)
- **P0-4** — 0 GitHub Actions CI avant Sprint 0 (corrigé batch 0.1). Maintenir et étendre Sprint 1+.
- **P1-11** — Tests chaos / property-based manquants (co-owner State Machine Eng Sprint 5).
- Zones revenue sous-protégées (eval_17) : telegram 10 %, vol_forecaster 63 % — coverage Sprint 1-3.
- Pas de QLIKE / PICP dans tests volatility (coordination Vol Modeler + Conformal Eng Sprint 4-5).
- Type hints coverage 69 % moyenne → cible 90 % (P2, Sprint 6).
- Référence : 2 696 tests existants, 0 mutation testing, 0 fuzz harness systématique.

(Liens : [audit globale Phase 1](../../audits/2026-Q2/algo_audit_institutional.md))

## 6. Inputs / Outputs
- **Inputs** :
  - Code source de tous les owners.
  - Coverage reports (`coverage.py`).
  - Mutation reports (`mutmut`).
  - CI logs GitHub Actions.
- **Outputs** :
  - `tests/` (unitaires + property + fuzz + mutation).
  - `.github/workflows/algo_tests.yml` (mis à jour).
  - `.github/workflows/coverage.yml` (Sprint 1).
  - `.github/workflows/mutation.yml` (Sprint 6, hebdomadaire car lourd).
  - `tests/property/` (Hypothesis tests).
  - `tests/fuzz/` (fuzz harness).
  - `audits/2026-Q3/coverage_<sprint>.md` (par sprint).
  - `audits/2026-Q4/mutation_score_sprint_6.md` (Sprint 6 + Sprint 7).
  - Documentation `docs/algo/testing_strategy.md`.

## 7. Critères de "done"
- Coverage branche ≥ 90 % sur modules critiques (state_machine, smart_money, validation, vol_forecaster, conformal, regime).
- Mutation score `mutmut` ≥ 70 % sur les mêmes modules (Sprint 6 measurement).
- Fuzz harness 10⁴ inputs : 0 crash.
- Property-based 10⁶ tirages sur state machine : 0 invariant violé.
- CI verte sur dernier commit de chaque sprint.
- 0 test `xfail` non documenté dans `tests/disabled_during_sprint_X.md`.
- Tout P0/P1 fix accompagné d'un test régression.
- Documentation testing strategy livrée Sprint 7.
