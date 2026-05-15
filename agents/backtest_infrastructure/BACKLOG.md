# Backlog — Backtest Infrastructure

## Sprint 1
- [x] Fix signal_id UUID → sha1 deterministic (P1, 1 h) — done commit 714cecc
- [x] Wire DynamicSpread/Slippage in run_backtest.py (P0-6, 2 h) — done commit 4310bab

## Sprint 3
- [x] Couple CPCV/strategy_gates to run_backtest (P0-17, 4 h) — done commit 4310bab
- [ ] Add `--validate` flag to run_backtest.py auto-invoke gates (1 h)
- [ ] Walk-forward harness wrapper around CPCV for time-series CV (8 h)
- [ ] Fix bugs métriques (Calmar non annualisé, Sharpe stdev inconsistent, max_consec_losses counts breakeven) (3 h)
- [ ] Lo 2002 autocorrelation correction in Sharpe annualisation (2 h)

## Sprint 5
- [ ] Wire `stress_tests.run_stress_suite` to a `scripts/run_stress.py` (6 h)
- [ ] Historical replays on 4 windows × 2 actifs (8 h compute + 4 h analysis)
- [ ] Fuzz harness with property-based assertions (6 h)
- [ ] Sensitivity sweep ±20 % runner (4 h)

## Sprint 6
- [x] Snapshot store scaffold — done commit 4fbd7a9
- [ ] Wire SnapshotStore.write() into ConfluenceDetector.analyze() (3 h)
- [ ] Wire SnapshotStore.write() into SignalReplay (3 h)
- [ ] Replay-from-snapshot CLI (4 h)
- [ ] Profiling + vectorisation hot paths (12-20 h)
- [ ] Versioning model files (`models/v_{major}.{minor}.{patch}/`) (4 h)

## Sprint 7
- [ ] `scripts/render_tear_sheet.py` from summary JSON + trades CSV (8 h)
- [ ] Pandoc template `docs/algo/tear_sheet_template.tex` (4 h)
- [ ] e2e test 6 actifs × 2 TF (12 h compute)
