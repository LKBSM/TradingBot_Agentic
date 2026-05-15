# Charter — Statistical Validator

**Slug** : `statistical_validator`
**Date** : 2026-05-15
**Sponsor** : Lead Quant Architect

## 1. Mission
Garantir que toute affirmation de performance produite par Smart Sentinel AI passe la chaîne de validation institutionnelle CPCV (López de Prado AFML) + DSR (Bailey-López de Prado 2014) + PBO (López de Prado 2014) + Diebold-Mariano + Reality Check (White 2000 / Hansen 2005). À l'issue de Sprint 0, la machinerie existe (`src/research/cpcv_harness.py` 507 LOC AFML-conforme + `strategy_gates.py`) mais **n'est jamais appelée** par le runner backtest commercial (P0-17). Le rôle livre Sprint 3 le couplage CPCV/DSR/PBO/DM dans `src/backtest/validation/`, refonte le ConfluenceDetector Sprint 4 (logistic L1 multi-feature, Brier skill ≥ +0.03), et signe la gate finale Sprint 3 (CI 95 % PF lo > 1.0 OU pivot valeur explicative).

## 2. Périmètre
- **Inclus** :
  - `src/research/cpcv_harness.py` + `strategy_gates.py` (existants).
  - `src/backtest/validation/` (à créer Sprint 3, couplage).
  - Refonte `src/intelligence/confluence_detector.py` (logistic L1 multi-feature) Sprint 4 co-owner avec Vol Modeler.
  - Métriques DSR, PBO, DM, Reality Check, IC.
  - Reliability diagrams, calibration curves.
  - Feature engineering Sprint 3 (input depuis Data, Vol, Regime, SMC, Conformal).
  - Gate signature Sprint 3 + Sprint 7.
- **Exclu** :
  - Implémentation backtest engine (Backtest Infrastructure).
  - Conformal wrapper interne (Conformal Engineer — Stat Validator consomme).
  - Forecasting volatility (Vol Modeler).
  - SMC detection (SMC Lead).

## 3. KPI principal et métriques
- **KPI** : tout signal reporté passe la chaîne de validation (CPCV + DSR + PBO + DM + Reality Check).
- **Sous-métriques** :
  - DSR ≥ 1.5 (cible institutionnelle López de Prado).
  - PBO ≤ 0.35 (overfitting probability borné).
  - PF_lo CI 95 % > 1.0 (gate Sprint 3).
  - DM p-value < 0.05 vs baseline buy-and-hold ou random-entry.
  - Brier skill ConfluenceDetector ≥ +0.03 vs constant (vs −0.022 actuel — P0-1).
  - Reliability diagram monotonicité ≥ 0.7 (Spearman rank).
  - IC moyen par feature : top-5 features avec |IC| ≥ 0.05.
  - 0 rapport / tear sheet livré sans signature de validation.
- **Cadence de mesure** : à chaque sweep paramétrique + chaque tear sheet candidate + gate de sortie de sprint.

## 4. RACI sur les sprints

| Sprint | Responsable | Accountable | Consulted | Informed |
| ------ | ----------- | ----------- | --------- | -------- |
| Sprint 1 | — (prep) | LQA | Data Quality | — |
| Sprint 2 | Stat Validator (validation SMC annotations) | LQA | SMC Lead | — |
| Sprint 3 | Stat Validator (rôle pivot, gate signature) | LQA | Tous | Tous |
| Sprint 4 | Stat Validator (refonte score) | LQA | Vol Modeler, Conformal Eng | Tous |
| Sprint 5 | Stat Validator (stress validation) | LQA | QA | — |
| Sprint 6 | Stat Validator (versioning métriques) | LQA | Backtest Infra | — |
| Sprint 7 | Stat Validator (certification finale) | LQA | LQA | Tous |

## 5. Findings prioritaires de l'audit Phase 1 (P0/P1)
- **P0-1** — ConfluenceDetector sans pouvoir prédictif (Pearson −0.008, Brier skill −0.022). Refonte Sprint 4 (logistic L1 multi-feature).
- **P0-17** — CPCV/DSR/PBO existent mais non couplés au backtest commercial. Couplage Sprint 3.
- **P1-4** — Double-gating detector (min=25) + state machine (enter=75) → 96.3 % rejection. Documenter / arbitrer Sprint 4.
- **P1-5** — OB ↔ Retest corrélés (Cramér's V=0.489) = info dupliquée. Feature selection L1 doit résoudre.
- **P1-9** — Conformal sans Mondrian (P1-9) — coordination avec Conformal Eng.
- **F0-1** Sprint 0 finding — baseline 0 trades sur 7 ans. Sprint 3 doit produire des trades en wirant News+Vol ou sweep paramétrique.

(Liens : [audit §3.3](../../audits/2026-Q2/section_3_3_confluence.md), [audit §3.8](../../audits/2026-Q2/section_3_8_backtest_engine.md))

## 6. Inputs / Outputs
- **Inputs** :
  - Features de tous les owners (SMC, Vol, Regime, Data, Conformal).
  - Backtest results (trades, equity, returns) depuis Backtest Infra.
  - Annotations expertes SMC (Sprint 2) pour ground truth labels.
- **Outputs** :
  - `src/backtest/validation/__init__.py`, `cpcv.py` (wrapper), `dsr.py`, `pbo.py`, `dm.py`, `reality_check.py`, `gates.py`.
  - `src/intelligence/confluence_detector.py` v2 (logistic L1).
  - `tests/test_validation_gates.py`, `test_confluence_l1.py`, `test_dsr_pbo.py`.
  - `reports/validation/<asset>_<tf>_<sprint>.md` (signed off).
  - `audits/2026-Q3/confluence_l1_refit.md` (Sprint 4).
  - `audits/2026-Q4/certification_validation.md` (Sprint 7, contribution).

## 7. Critères de "done"
- Couplage CPCV/DSR/PBO/DM appelé par `scripts/run_backtest.py` à chaque run commercial.
- Gate Sprint 3 signée : verdict GO (CI 95 % PF lo > 1.0 sur ≥ 1 actif) OU pivot documenté (valeur explicative non-PF).
- ConfluenceDetector v2 atteint Brier skill ≥ +0.03 sur OOS 2024 (vs −0.022 actuel).
- Reliability diagram monotone (Spearman ≥ 0.7).
- 0 tear sheet livré Sprint 7 sans signature de validation Stat Validator.
- Documentation `docs/algo/validation.md` avec références López de Prado AFML chap 12-14.
