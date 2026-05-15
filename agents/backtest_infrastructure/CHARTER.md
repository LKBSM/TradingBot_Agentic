# Charter — Backtest Infrastructure

**Slug** : `backtest_infrastructure`
**Date** : 2026-05-15
**Sponsor** : Lead Quant Architect

## 1. Mission
Posséder `src/backtest/state_machine_replay.py`, `scripts/run_backtest.py`, `scripts/audit_backtest.py` et la mécanique de reproductibilité bit-à-bit. À l'issue de Sprint 0, le backtest engine est noté 3.5/10 (vs 2.0 eval_18) : CPCV existe en R&D mais n'est pas couplé (P0-17), coûts transactionnels à $0 alors que les modèles existent (P0-6), look-ahead MTF latent (P0-7), `signal_id` change à chaque run cassant la reproductibilité, métriques bugguées (Calmar non annualisé, Sharpe pstdev/stdev incohérent). Le rôle livre Sprint 3 le wirage `DynamicSpreadModel + DynamicSlippageModel`, la vectorisation pour cible < 2 min / 7 ans, et la reproductibilité bit-à-bit prouvée. Le rôle est l'opérateur de la machinerie ; les décisions statistiques appartiennent au Statistical Validator.

## 2. Périmètre
- **Inclus** :
  - `src/backtest/state_machine_replay.py`, `metrics.py`, `report.py`, `news_replay.py`, `validation.py`, `snapshot_store.py`, `stress_tests.py`.
  - `scripts/run_backtest.py`, `scripts/audit_backtest.py`, `scripts/run_baseline_sprint0.py`, `scripts/run_stress.py` (à créer Sprint 5).
  - `src/environment/execution_model.py` (DynamicSpread/Slippage existants — à wirer).
  - Reproductibilité : seeds, hash codes, hash data, hash modèles.
  - Vectorisation / parallélisation (joblib, numpy, Numba).
  - Edge cases : gaps, halts, leverage extreme, slippage outlier.
- **Exclu** :
  - Nouvelles métriques statistiques (Statistical Validator).
  - Nouvelles stratégies (Lead Quant Architect / SMC Lead).
  - Risk engine (frozen à `src/environment/risk_manager.py` Sprint 0-4 par décision B).
  - Live trading (out of scope Sprint 1-7).

## 3. KPI principal et métriques
- **KPI** : backtest 7 ans / 1 paire en < 2 minutes ; reproductibilité bit-à-bit.
- **Sous-métriques** :
  - Wall-clock 7 ans XAU M15 < 120 s (actuel ~5-10 min estimé Sprint 0).
  - Wall-clock 7 ans EURUSD M15 < 120 s.
  - SHA256 `trades.csv` identique sur 3 runs successifs.
  - 0 NaN / Inf dans `equity.csv` post-run.
  - Costs transactionnels appliqués : spread + slippage + commission > $0 dans tous les rapports.
  - Parallélisation sweep state machine (Sprint 3) : speedup ≥ 8× sur 16 cœurs.
  - Métriques fix : Calmar annualisé, Sharpe stdev unique cohérent, max_consec_losses ignore breakevens, autocorrélation Lo 2002.
- **Cadence de mesure** : à chaque commit (CI bench) + recalc end of Sprint 3 et Sprint 6.

## 4. RACI sur les sprints

| Sprint | Responsable | Accountable | Consulted | Informed |
| ------ | ----------- | ----------- | --------- | -------- |
| Sprint 1 | Backtest Infra (P0-7 follow-up, fix UUID) | LQA | Data Quality | — |
| Sprint 2 | — | LQA | — | — |
| Sprint 3 | Backtest Infra (rôle pivot wiring + vectorize) | LQA | Stat Validator, State Machine | Tous |
| Sprint 4 | — | LQA | — | — |
| Sprint 5 | Backtest Infra (stress dispatch) | LQA | QA, Stat Validator | — |
| Sprint 6 | Backtest Infra (rôle pivot, replay nano, snapshot) | LQA | State Machine, QA | — |
| Sprint 7 | Backtest Infra (tear sheets renderer) | LQA | Stat Validator, LQA | — |

## 5. Findings prioritaires de l'audit Phase 1 (P0/P1)
- **P0-5** — Pas de walk-forward propre dans `state_machine_replay.py`. Sprint 3.
- **P0-6** — Coûts transactionnels $0 ; modèles existent mais non wirés. Sprint 3 (note : partial fix déjà commité, finaliser et tester).
- **P0-7** — Look-ahead MTF latent (co-owner Data Quality). Sprint 1.2.
- **P0-17** — CPCV/DSR/PBO non couplés (co-owner Stat Validator). Sprint 3.
- **P1 (audit §3.8)** :
  - `signal_id = uuid.uuid4()` casse reproductibilité (`confluence_detector.py:343`). Sprint 1 fix 1 h.
  - Métriques bugguées : Calmar non annualisé (`metrics.py:254`), Sharpe `pstdev` vs `stdev` incohérent, `max_consec_losses` compte breakeven, annualisation sans correction Lo 2002. Sprint 3.

(Liens : [audit §3.8](../../audits/2026-Q2/section_3_8_backtest_engine.md))

## 6. Inputs / Outputs
- **Inputs** :
  - OHLCV (via Data Provider).
  - Signaux (state machine output).
  - Costs models (`DynamicSpreadModel`, `DynamicSlippageModel` existants).
  - Sweep configurations (depuis State Machine Eng Sprint 3).
  - Modèles persistés (`models/`), config snapshots.
- **Outputs** :
  - `src/backtest/state_machine_replay.py` (vectorisé, wired costs).
  - `src/backtest/metrics.py` (fixes Calmar/Sharpe).
  - `scripts/run_backtest.py` (couplé à validation chain via `--validate`).
  - `tests/test_backtest_reproducibility.py`, `test_costs_wired.py`, `test_metrics_correctness.py`.
  - `reports/baseline/<asset>_<tf>_<sprint>.{md,json}` + checksums.
  - `audits/2026-Q3/backtest_engine_v2.md`.
  - Snapshots JSONL per-signal (Sprint 6).

## 7. Critères de "done"
- Backtest 7 ans / 1 paire < 120 s (XAU M15 baseline).
- SHA256 `trades.csv` identique sur 3 runs successifs (reproductibilité bit-à-bit).
- DynamicSpread + DynamicSlippage + commission > $0 dans tous les rapports.
- Métriques fixées (Calmar annualisé, Sharpe stdev cohérent, max_consec_losses ignore breakevens, Lo 2002 autocorrelation correction).
- Tout backtest documenté reproductible à SHA256 identique.
- Toute stratégie passe par `src.backtest.validation.evaluate_gates` avant publication.
- Aucun secret commiteé, aucun chemin local hardcoded.
- Replay nano (Sprint 6) : trace step-by-step disponible pour audit.
- 0 NaN/Inf dans equity sur 7 ans.
