# SL/TP Sweep — Filtered XAU M15 (skip NY + skip ATR_PCTL>0.75)

**Trades after filter**: 881 (482 train / 399 test)
**Simulation**: walk-forward 12 bars, SL/TP intrabar (high/low), tie → SL
**Baseline current**: SL=2.0× ATR, TP=4.0× ATR (R:R 1:2)
---

## PF_test heatmap (rows=SL, cols=TP)

| SL \ TP | 1.5 | 2.0 | 3.0 | 4.0 | 5.0 |
|---|---|---|---|---|---|
| 1.0 | 1.07 | 1.16 | 1.19 | 1.21 | 1.25 |
| 1.5 | — | 1.09 | 1.13 | 1.15 | 1.20 |
| 2.0 | — | — | 1.20 | 1.26 | 1.34✅ |
| 2.5 | — | — | 1.11 | 1.17 | 1.24 |
| 3.0 | — | — | — | 1.15 | 1.22 |

## Top-3 by PF_test (n≥100)

| rank | SL | TP | R:R | n_train | PF_train | exp_train | n_test | PF_test | win_test | exp_test | total_R_test | stable |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2.0 | 5.0 | 2.50 | 482 | 1.297 | 0.132 | 399 | 1.340 | 0.484 | 0.152 | 60.752 | ✅ |
| 2 | 2.0 | 4.0 | 2.00 | 482 | 1.226 | 0.100 | 399 | 1.259 | 0.484 | 0.116 | 46.306 |  |
| 3 | 1.0 | 5.0 | 5.00 | 482 | 1.418 | 0.272 | 399 | 1.249 | 0.326 | 0.166 | 66.351 |  |

## Baseline (SL=2.0, TP=4.0)

- TRAIN: n=482, PF=1.226, exp_R=0.100
- TEST:  n=399, PF=1.259, win=0.484, exp_R=0.116, total_R=46.306

## Decision

**MARGINAL**: best is SL=2.0/TP=5.0 (PF_test 1.340, uplift +0.081 vs baseline 1.259). Below 1.50 threshold — keep current config or adopt only if reproducible on a forward sample.