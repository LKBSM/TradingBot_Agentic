# Admission gates — EURUSD walk-forward

**Date** : 2026-05-16T10:52:43.935815
**n_trades** : 6402
**Refit horizon** : 30 days

## Gate verdict

**ALL GATES PASSED** : ❌ NO

| Gate | Threshold | Value | Pass |
| --- | --- | --- | --- |
| trades >= 30 | 30 | 6402 | ✅ |
| DSR >= 1.5 | 1.5 | -8.000 | ❌ |
| PBO <= 0.35 | 0.35 | 0.000 | ✅ |
| PF lo > 1.0 | 1.0 | 1.922 | ✅ |
| DM p < 0.05 | 0.05 | 0.0000 | ✅ |

**Sharpe** : 23.8047
**Profit factor** : 2.1677
**PF 95% CI** : [1.9215, 2.4653]
**DM statistic** : -12.2467

**Failure reasons** :
- DSR z-score -8.000 < 1.5 (prob=0.000)