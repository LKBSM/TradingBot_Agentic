# Failure-mode diagnostic: NY × Q4_high

Hypothesis under test: SL = 2× ATR is too tight in high-vol regime → stops whipsawed.
---

## Per-bucket comparison (full sample, train+test)

| Bucket | n | PF | win% | exp_R | med_R | med_bars | %SL-loss | %near-BE | %TP-win | exit_target | exit_invalid | exit_time | ATR_med |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **NY × Q4_high** | 585 | 0.992 | 0.484 | -0.002 | -0.022 | 4 | 0.094 | 0.347 | 0.031 | 0.024 | 0.092 | 0.285 | 4.47 |
| **London × Q3** | 134 | 1.670 | 0.493 | 0.194 | -0.003 | 8 | 0.209 | 0.187 | 0.164 | 0.157 | 0.209 | 0.172 | 3.03 |
| **Asian × Q2** | 132 | 1.081 | 0.492 | 0.028 | -0.026 | 11 | 0.242 | 0.212 | 0.106 | 0.083 | 0.242 | 0.318 | 2.32 |
| **London × Q4_high** | 161 | 0.911 | 0.491 | -0.021 | -0.001 | 1 | 0.137 | 0.360 | 0.050 | 0.031 | 0.137 | 0.075 | 5.34 |
| **NY × Q3** | 284 | 1.047 | 0.461 | 0.014 | -0.067 | 11 | 0.190 | 0.229 | 0.067 | 0.056 | 0.187 | 0.433 | 3.23 |
| **NY × Q2** | 117 | 1.153 | 0.453 | 0.042 | -0.049 | 11 | 0.162 | 0.239 | 0.085 | 0.068 | 0.162 | 0.427 | 2.98 |

## SL-hit rate by ATR quartile (all sessions)

| ATR_Q | n | %SL-loss (R<-0.95) | %TP-win (R>1.5) | PF | exp_R |
|---|---|---|---|---|---|
| Q1_low | 295 | 0.332 | 0.169 | 1.181 | 0.077 |
| Q2 | 433 | 0.238 | 0.115 | 1.179 | 0.061 |
| Q3 | 624 | 0.184 | 0.088 | 1.193 | 0.057 |
| Q4_high | 1011 | 0.092 | 0.030 | 1.013 | 0.003 |

## R-multiple histogram — NY × Q4_high losers

| R bucket | count | % of losers |
|---|---|---|
| (-1.501, -1.0] | 54 | 17.9% |
| (-1.0, -0.8] | 4 | 1.3% |
| (-0.8, -0.5] | 43 | 14.3% |
| (-0.5, -0.2] | 101 | 33.6% |
| (-0.2, 0.0] | 99 | 32.9% |

## ATR at entry per bucket (validates Q4_high IS high vol)

| Bucket | ATR median | ATR mean | ATR_PCTL median |
|---|---|---|---|
| NY × Q4_high | 4.47 | 5.46 | 0.93 |
| London × Q3 | 3.03 | 3.83 | 0.63 |
| Asian × Q2 | 2.32 | 2.85 | 0.34 |
| London × Q4_high | 5.34 | 6.19 | 0.91 |
| NY × Q3 | 3.23 | 3.62 | 0.67 |
| NY × Q2 | 2.98 | 3.73 | 0.41 |

## Verdict

- NY × Q4_high SL-hit rate: **9.4%**
- London × Q3 SL-hit rate (control winner): **20.9%**
- ATR median NY×Q4_high vs London×Q3: 4.47 vs 3.03

→ **Hypothèse rejetée** — le SL n'est pas le problème. Le saigneur a une autre cause (regime, direction, news...).