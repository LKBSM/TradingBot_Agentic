# Backlog — Backtest Infrastructure

**Date** : 2026-05-15
**Owner** : Backtest Infrastructure

## Sprint 1 (S3-S4) — Data Layer Hardening (support + reproducibility)

- [x] Fix `signal_id = uuid.uuid4()` → SHA1 deterministic (`(bar_ts, symbol, strategy)`) (P1, 1 h) — done commit `714cecc`
- [x] Wire `DynamicSpreadModel + DynamicSlippageModel` in `run_backtest.py` (P0-6 partial, 2 h) — done commit `4310bab`
- [ ] Validation : tester reproductibilité bit-à-bit post-fix UUID (`tests/test_backtest_reproducibility.py` — SHA256 sur trades.csv × 3 runs) — 2 h
- [ ] Co-owner P0-7 avec Data Quality : test régression que le fix `<=` → `<` ne casse pas les baselines existantes (delta documenté dans `audits/2026-Q3/lookahead_fix_impact.md`) — 3 h
- [ ] Refactor `state_machine_replay.py` : extraire le hot loop pour vectorisation Sprint 3 — 4 h

## Sprint 2 (S5-S6) — Detection (frozen)

- [ ] Pas de livrable nouveau. Stand-by.

## Sprint 3 (S7-S8) — Statistical Edge Discovery (rôle pivot)

- [x] Couple CPCV/strategy_gates to `run_backtest.py` (P0-17, 4 h) — done commit `4310bab`
- [ ] Add `--validate` flag to `run_backtest.py` to auto-invoke gates and produce `ValidationReport.json` (1 h)
- [ ] Walk-forward harness wrapper around CPCV for time-series CV (P0-5, 8 h)
- [ ] **Vectorisation** state_machine_replay hot loop (cible < 120 s / 7 ans / paire) — 12 h
  - Profil baseline (`cProfile` + `line_profiler`) sur XAU M15 7 ans.
  - Vectoriser inner loop bar-by-bar via numpy + numba si nécessaire.
  - Bench avant/après documenté.
- [ ] **Parallélisation sweep** state machine 432 cellules × 4 actifs × 4 TF (coordination State Machine Eng) : joblib `Parallel(n_jobs=-1)` ; cible speedup ≥ 8× sur 16 cœurs — 6 h
- [ ] **Fix métriques** (`src/backtest/metrics.py`) :
  - Calmar non annualisé (ligne 254) → annualiser correctement — 1 h
  - Sharpe `pstdev` vs `stdev` incohérent → standardiser sur `stdev` (sample stdev, ddof=1) — 1 h
  - `max_consec_losses` compte breakeven → exclure trades à PnL ≈ 0 (|pnl| < 0.5×spread) — 1 h
  - Lo 2002 autocorrelation correction in Sharpe annualisation — 3 h
- [ ] Test `tests/test_metrics_correctness.py` (fixtures avec valeurs exactes attendues sur datasets contrôlés) — 4 h
- [ ] Test `tests/test_costs_wired.py` : assert spread + slippage + commission > 0 dans tout summary — 2 h

## Sprint 4 (S9-S10) — Calibration (frozen)

- [ ] Pas de livrable nouveau. Stand-by si refonte ConfluenceDetector (Stat Validator) change l'API output.

## Sprint 5 (S11-S12) — Robustness & Stress Testing

- [ ] Wire `stress_tests.run_stress_suite` to `scripts/run_stress.py` (6 h)
- [ ] Historical replays on 4 windows × 2 actifs : COVID 2020, LDI 2022, SVB 2023, yen 2024 × XAU + EURUSD (batch 5.2, 8 h compute + 4 h analysis)
- [ ] Fuzz harness with property-based assertions : `tests/fuzz/test_backtest_fuzz.py` (input NaN, gaps, leverage extrême, spread 10×) (batch 5.1, 6 h)
- [ ] Sensitivity sweep ±20 % runner (batch 5.3, 4 h)
- [ ] Documenter edge cases découverts dans `audits/2026-Q3/backtest_edge_cases.md` — 3 h

## Sprint 6 (S13-S14) — Production Hardening (rôle pivot)

- [x] Snapshot store scaffold — done commit `4fbd7a9`
- [ ] Wire `SnapshotStore.write()` into `ConfluenceDetector.analyze()` (P0-16 input State Machine Eng, 3 h)
- [ ] Wire `SnapshotStore.write()` into `SignalReplay` (3 h)
- [ ] Replay-from-snapshot CLI (`scripts/replay_from_snapshot.py`) (4 h)
- [ ] **Replay nano** : trace step-by-step à la nanoseconde pour audit (batch 6.2, 8 h)
- [ ] Profiling + vectorisation hot paths (cible < 250 ms / tick / paire si live mode, batch 6.1, 12-20 h)
- [ ] Versioning model files (`models/v_{major}.{minor}.{patch}/`) + compat check at load (batch 6.4, 4 h)
- [ ] Doc `docs/algo/backtest_engine.md` (architecture, costs, validation chain) — 4 h

## Sprint 7 (S15-S16) — Commercial Readiness

- [ ] `scripts/render_tear_sheet.py` from summary JSON + trades CSV (batch 7.2, 8 h)
- [ ] Pandoc template `docs/algo/tear_sheet_template.tex` (décision C, 4 h)
- [ ] Test e2e 6 actifs × 2 TF (batch 7.4, 12 h compute) : produire la matrice complète `reports/tear_sheets/<asset>_<tf>.md` + `.pdf`
- [ ] Coordination Stat Validator pour signature `ValidationReport.json` attaché à chaque tear sheet — 2 h

## Inbox (non priorisé)
- Live backtest mode (paper trading WebSocket) — réserve post-Sprint 7.
- Parquet vs CSV pour `trades.csv` (perf I/O) — évaluer Sprint 6.
- Distributed backtest (Dask, Ray) si > 100 cellules × 10 actifs. Hors périmètre.
- GPU acceleration pour features (cuDF, RAPIDS) — réserve.
- Backtest-as-a-service API endpoint — produit, hors algo.
- Time-travel debugger pour signaux (replay UI).
