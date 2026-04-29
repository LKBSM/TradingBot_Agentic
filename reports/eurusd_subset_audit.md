# EURUSD M15 — Subset Edge Audit

**Trades**: 1255 | Train < 2023-01-01 (4.00 yr) | Test ≥ 2023-01-01 (2.98 yr)
**Acceptance**: PF_test ≥ 1.30 AND n_test ≥ 100 AND sign(PF_train-1) = sign(PF_test-1)

**NB**: EURUSD NY session = 12:00–21:00 UTC (XAU uses 13:00–21:00).

---

## Overall

- TRAIN: n=729, PF=0.010, win=0.005, exp=-0.076, total_R=-55.760
- TEST: n=526, PF=0.006, win=0.006, exp=-0.066, total_R=-34.471


## By direction

| direction | n_train | PF_train | n_test | PF_test | win_test | exp_test | total_R_test | ✅ |
|---|---|---|---|---|---|---|---|---|
| LONG | 367 | 0.000 | 273 | 0.014 | 0.011 | -0.054 | -14.823 |  |
| SHORT | 362 | 0.019 | 253 | 0.000 | 0.000 | -0.078 | -19.648 |  |

## By session (NY = NY_overlap + NY_afternoon)

| session | n_train | PF_train | n_test | PF_test | win_test | exp_test | total_R_test | ✅ |
|---|---|---|---|---|---|---|---|---|
| Asian | 128 | 0.000 | 102 | 0.010 | 0.010 | -0.058 | -5.939 |  |
| London | 209 | 0.000 | 165 | 0.010 | 0.006 | -0.056 | -9.295 |  |
| NY | 367 | 0.016 | 242 | 0.004 | 0.004 | -0.074 | -17.845 |  |
| OffHours | 25 | 0.000 | 17 | 0.000 | 0.000 | -0.082 | -1.392 |  |

## By session (fine: NY split into overlap/afternoon)

| session_fine | n_train | PF_train | n_test | PF_test | win_test | exp_test | total_R_test | ✅ |
|---|---|---|---|---|---|---|---|---|
| NY_afternoon | 142 | 0.020 | 87 | 0.011 | 0.011 | -0.067 | -5.801 |  |
| Asian | 128 | 0.000 | 102 | 0.010 | 0.010 | -0.058 | -5.939 |  |
| London | 209 | 0.000 | 165 | 0.010 | 0.006 | -0.056 | -9.295 |  |
| NY_overlap | 225 | 0.013 | 155 | 0.000 | 0.000 | -0.078 | -12.044 |  |
| OffHours | 25 | 0.000 | 17 | 0.000 | 0.000 | -0.082 | -1.392 |  |

## By ATR quartile

| ATR_Q | n_train | PF_train | n_test | PF_test | win_test | exp_test | total_R_test | ✅ |
|---|---|---|---|---|---|---|---|---|
| Q4_high | 351 | 0.016 | 244 | 0.010 | 0.012 | -0.085 | -20.834 |  |
| Q1_low | 73 | 0.000 | 58 | 0.000 | 0.000 | -0.034 | -1.957 |  |
| Q2 | 114 | 0.000 | 76 | 0.000 | 0.000 | -0.045 | -3.416 |  |
| Q3 | 191 | 0.000 | 148 | 0.000 | 0.000 | -0.056 | -8.264 |  |

## By day of week

| dow | n_train | PF_train | n_test | PF_test | win_test | exp_test | total_R_test | ✅ |
|---|---|---|---|---|---|---|---|---|
| 4 | 157 | 0.006 | 92 | 0.032 | 0.022 | -0.048 | -4.409 |  |
| 3 | 171 | 0.032 | 123 | 0.007 | 0.008 | -0.072 | -8.859 |  |
| 0 | 126 | 0.000 | 94 | 0.000 | 0.000 | -0.069 | -6.495 |  |
| 1 | 137 | 0.000 | 96 | 0.000 | 0.000 | -0.049 | -4.674 |  |
| 2 | 132 | 0.000 | 117 | 0.000 | 0.000 | -0.081 | -9.498 |  |
| 6 | 6 | 0.000 | 4 | 0.000 | 0.000 | -0.134 | -0.536 |  |

## By session × ATR_Q (the canonical XAU breakdown)

| session | ATR_Q | n_train | PF_train | n_test | PF_test | win_test | exp_test | total_R_test | ✅ |
|---|---|---|---|---|---|---|---|---|---|
| Asian | Q4_high | 14 | 0.000 | 14 | 0.032 | 0.071 | -0.127 | -1.776 |  |
| London | Q4_high | 52 | 0.000 | 44 | 0.023 | 0.023 | -0.086 | -3.772 |  |
| NY | Q4_high | 275 | 0.021 | 177 | 0.005 | 0.006 | -0.078 | -13.894 |  |
| Asian | Q1_low | 61 | 0.000 | 47 | 0.000 | 0.000 | -0.032 | -1.487 |  |
| Asian | Q2 | 35 | 0.000 | 35 | 0.000 | 0.000 | -0.076 | -2.676 |  |
| Asian | Q3 | 18 | 0.000 | 6 | — | 0.000 | 0.000 | 0.000 |  |
| London | Q1_low | 10 | 0.000 | 8 | 0.000 | 0.000 | -0.059 | -0.470 |  |
| London | Q2 | 62 | 0.000 | 32 | 0.000 | 0.000 | -0.023 | -0.740 |  |
| London | Q3 | 85 | 0.000 | 81 | 0.000 | 0.000 | -0.053 | -4.313 |  |
| NY | Q1_low | 1 | — | 2 | — | 0.000 | 0.000 | 0.000 |  |
| NY | Q2 | 10 | — | 6 | — | 0.000 | 0.000 | 0.000 |  |
| NY | Q3 | 81 | 0.000 | 57 | 0.000 | 0.000 | -0.069 | -3.951 |  |
| OffHours | Q1_low | 1 | 0.000 | 1 | — | 0.000 | 0.000 | 0.000 |  |
| OffHours | Q2 | 7 | 0.000 | 3 | — | 0.000 | 0.000 | 0.000 |  |
| OffHours | Q3 | 7 | 0.000 | 4 | — | 0.000 | 0.000 | 0.000 |  |
| OffHours | Q4_high | 10 | 0.000 | 9 | 0.000 | 0.000 | -0.155 | -1.392 |  |

## By session × direction

| session | direction | n_train | PF_train | n_test | PF_test | win_test | exp_test | total_R_test | ✅ |
|---|---|---|---|---|---|---|---|---|---|
| Asian | LONG | 67 | 0.000 | 58 | 0.030 | 0.017 | -0.032 | -1.860 |  |
| London | LONG | 104 | 0.000 | 85 | 0.018 | 0.012 | -0.059 | -5.009 |  |
| NY | LONG | 181 | 0.000 | 123 | 0.008 | 0.008 | -0.065 | -7.954 |  |
| Asian | SHORT | 61 | 0.000 | 44 | 0.000 | 0.000 | -0.093 | -4.079 |  |
| London | SHORT | 105 | 0.000 | 80 | 0.000 | 0.000 | -0.054 | -4.286 |  |
| NY | SHORT | 186 | 0.032 | 119 | 0.000 | 0.000 | -0.083 | -9.891 |  |
| OffHours | LONG | 15 | 0.000 | 7 | — | 0.000 | 0.000 | 0.000 |  |
| OffHours | SHORT | 10 | 0.000 | 10 | 0.000 | 0.000 | -0.139 | -1.392 |  |

## Filter combinations (sweep)

Acceptance: PF_test ≥ 1.30 AND n_test ≥ 100 AND PF stable train↔test

| Rules | n_train | PF_train | n_test | PF_test | win_test | exp_test | total_R_test | sig/yr_test | ✅ |
|---|---|---|---|---|---|---|---|---|---|
| R2_skip_NY + R4_long_only | 186 | 0.000 | 150 | 0.021 | 0.013 | -0.046 | -6.869 | 50 |  |
| R2_skip_NY + R3_skip_NY_Q4 + R4_long_only | 186 | 0.000 | 150 | 0.021 | 0.013 | -0.046 | -6.869 | 50 |  |
| R2_skip_NY + R4_long_only + R5_skip_OffHours | 171 | 0.000 | 143 | 0.021 | 0.014 | -0.048 | -6.869 | 48 |  |
| R2_skip_NY + R3_skip_NY_Q4 + R4_long_only + R5_skip_OffHours | 171 | 0.000 | 143 | 0.021 | 0.014 | -0.048 | -6.869 | 48 |  |
| R3_skip_NY_Q4 + R4_long_only | 230 | 0.000 | 180 | 0.015 | 0.011 | -0.053 | -9.621 | 60 |  |
| R3_skip_NY_Q4 + R4_long_only + R5_skip_OffHours | 215 | 0.000 | 173 | 0.015 | 0.012 | -0.056 | -9.621 | 58 |  |
| R4_long_only | 367 | 0.000 | 273 | 0.014 | 0.011 | -0.054 | -14.823 | 91 |  |
| R4_long_only + R5_skip_OffHours | 352 | 0.000 | 266 | 0.014 | 0.011 | -0.056 | -14.823 | 89 |  |
| R2_skip_NY + R5_skip_OffHours | 337 | 0.000 | 267 | 0.010 | 0.007 | -0.057 | -15.234 | 89 |  |
| R2_skip_NY + R3_skip_NY_Q4 + R5_skip_OffHours | 337 | 0.000 | 267 | 0.010 | 0.007 | -0.057 | -15.234 | 89 |  |
| R2_skip_NY | 362 | 0.000 | 284 | 0.009 | 0.007 | -0.059 | -16.626 | 95 |  |
| R2_skip_NY + R3_skip_NY_Q4 | 362 | 0.000 | 284 | 0.009 | 0.007 | -0.059 | -16.626 | 95 |  |
| R3_skip_NY_Q4 + R5_skip_OffHours | 429 | 0.000 | 332 | 0.008 | 0.006 | -0.058 | -19.185 | 111 |  |
| R3_skip_NY_Q4 | 454 | 0.000 | 349 | 0.007 | 0.006 | -0.059 | -20.577 | 117 |  |
| R5_skip_OffHours | 704 | 0.010 | 509 | 0.006 | 0.006 | -0.065 | -33.079 | 171 |  |
| ALL (no filter) | 729 | 0.010 | 526 | 0.006 | 0.006 | -0.066 | -34.471 | 176 |  |
| R1_skip_Q4_high | 378 | 0.000 | 282 | 0.000 | 0.000 | -0.048 | -13.637 | 94 |  |
| R1_skip_Q4_high + R2_skip_NY | 286 | 0.000 | 217 | 0.000 | 0.000 | -0.045 | -9.686 | 73 |  |
| R1_skip_Q4_high + R3_skip_NY_Q4 | 378 | 0.000 | 282 | 0.000 | 0.000 | -0.048 | -13.637 | 94 |  |
| R1_skip_Q4_high + R4_long_only | 190 | 0.000 | 150 | 0.000 | 0.000 | -0.051 | -7.671 | 50 |  |
| R1_skip_Q4_high + R5_skip_OffHours | 363 | 0.000 | 274 | 0.000 | 0.000 | -0.050 | -13.637 | 92 |  |
| R1_skip_Q4_high + R2_skip_NY + R3_skip_NY_Q4 | 286 | 0.000 | 217 | 0.000 | 0.000 | -0.045 | -9.686 | 73 |  |
| R1_skip_Q4_high + R2_skip_NY + R4_long_only | 146 | 0.000 | 120 | 0.000 | 0.000 | -0.041 | -4.919 | 40 |  |
| R1_skip_Q4_high + R2_skip_NY + R5_skip_OffHours | 271 | 0.000 | 209 | 0.000 | 0.000 | -0.046 | -9.686 | 70 |  |
| R1_skip_Q4_high + R3_skip_NY_Q4 + R4_long_only | 190 | 0.000 | 150 | 0.000 | 0.000 | -0.051 | -7.671 | 50 |  |
| R1_skip_Q4_high + R3_skip_NY_Q4 + R5_skip_OffHours | 363 | 0.000 | 274 | 0.000 | 0.000 | -0.050 | -13.637 | 92 |  |
| R1_skip_Q4_high + R4_long_only + R5_skip_OffHours | 180 | 0.000 | 146 | 0.000 | 0.000 | -0.053 | -7.671 | 49 |  |
| R1_skip_Q4_high + R2_skip_NY + R3_skip_NY_Q4 + R4_long_only | 146 | 0.000 | 120 | 0.000 | 0.000 | -0.041 | -4.919 | 40 |  |
| R1_skip_Q4_high + R2_skip_NY + R3_skip_NY_Q4 + R5_skip_OffHours | 271 | 0.000 | 209 | 0.000 | 0.000 | -0.046 | -9.686 | 70 |  |
| R1_skip_Q4_high + R2_skip_NY + R4_long_only + R5_skip_OffHours | 136 | 0.000 | 116 | 0.000 | 0.000 | -0.042 | -4.919 | 39 |  |
| R1_skip_Q4_high + R3_skip_NY_Q4 + R4_long_only + R5_skip_OffHours | 180 | 0.000 | 146 | 0.000 | 0.000 | -0.053 | -7.671 | 49 |  |
| R1_skip_Q4_high + R2_skip_NY + R3_skip_NY_Q4 + R4_long_only + R5_skip_OffHours | 136 | 0.000 | 116 | 0.000 | 0.000 | -0.042 | -4.919 | 39 |  |

## Top 3 stable filter combos

**No filter combo passes the stability gate.** EURUSD does NOT have a transferable XAU-style edge under this config.

## Verdict

❌ **REJECT** — no filter combination passes the stability gate. The XAU regime-filter edge does NOT transfer to EURUSD under this config (enter=40/exit=25, default SL=1.5×ATR, TP=3×ATR).

## NY × Q4_high specifically

- n_test=177, PF_test=0.005, total_R_test=-13.894, win=0.006
→ **Saigneur confirmé** (same pattern as XAU).