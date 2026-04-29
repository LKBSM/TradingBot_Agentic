# Scoring v2 — LightGBM on filtered XAU M15 subset

**Filter applied BEFORE training/eval**: `session != NY` AND `ATR_PCTL <= 0.75`
**Walk-forward**: train < 2023-01-01 (last 20% inner valid), test ≥ 2023-01-01
**Sample**: 2363 raw trades → 883 after filter → 484 train / 399 test
**Features**: 31 (sign-aligned SMC + unsigned + interactions + dead-score as input)

---

## Predictive power (OOS = test set)

| Model | Pearson_train | Pearson_test | Spearman_test | AUC_train | AUC_test |
|---|---|---|---|---|---|
| Regression (r̂) | 0.4778 | -0.0150 | -0.0364 | — | — |
| Classification (P(r>0)) | 0.4100 | 0.0057 | — | 0.7935 | 0.4974 |

> **Reference**: confluence_score Pearson_test = −0.0139 (dead). Anything above |0.05| OOS is real signal.

## Backtest by quantile — chosen model: classification (P(r>0))

| top % predicted | n | PF | win% | exp_R | total_R |
|---|---|---|---|---|---|
| 100% | 399 | 1.354 | 0.491 | 0.109 | 43.67 |
| 50% | 199 | 1.323 | 0.508 | 0.111 | 22.02 |
| 40% | 159 | 1.328 | 0.491 | 0.118 | 18.83 |
| 30% | 119 | 1.328 | 0.479 | 0.121 | 14.41 |
| 20% | 79 | 1.317 | 0.456 | 0.121 | 9.60 |
| 10% | 39 | 1.402 | 0.436 | 0.166 | 6.49 |

**Targets**: top-30% PF ≥ 1.50 (success), top-50% PF ≥ 1.40 (acceptable), else not shipping.

## Losing model sweep (for transparency)

_regression (r̂)_

| top % | n | PF | exp_R | total_R |
|---|---|---|---|---|
| 100% | 399 | 1.354 | 0.109 | 43.67 |
| 50% | 199 | 1.333 | 0.109 | 21.73 |
| 40% | 159 | 1.206 | 0.071 | 11.28 |
| 30% | 119 | 1.200 | 0.071 | 8.42 |
| 20% | 79 | 1.163 | 0.059 | 4.65 |
| 10% | 39 | 0.750 | -0.113 | -4.40 |

## Top 10 features by gain

| feature | gain | split |
|---|---|---|
| ATR_x_hour | 70 | 15 |
| BODY_SIZE | 65 | 13 |
| MACD_Diff_aligned | 54 | 10 |
| month | 53 | 11 |
| ATR_PCTL | 51 | 12 |
| hour | 48 | 7 |
| BODY_RATIO | 48 | 11 |
| BB_POS | 36 | 8 |
| confluence_score | 35 | 8 |
| FVG_SIZE_NORM | 32 | 6 |

## Overfit honesty check

- Regression Pearson gap (train−test): **+0.4927** (large overfit)
- Classification AUC gap (train−test): **+0.2961** (large overfit)

## Verdict

❌ **NOT WORTH SHIPPING** — model fails to beat both 1.50@30% and 1.40@50% targets. Filter-only baseline (PF 1.355, n=416) is the better product.
