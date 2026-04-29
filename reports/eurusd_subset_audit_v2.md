# EURUSD M15 — Subset Edge Audit

**Trades**: 1225 | Train < 2023-01-01 (4.00 yr) | Test ≥ 2023-01-01 (2.98 yr)
**Acceptance**: PF_test ≥ 1.30 AND n_test ≥ 100 AND sign(PF_train-1) = sign(PF_test-1)

**NB**: EURUSD NY session = 12:00–21:00 UTC (XAU uses 13:00–21:00).

---

## Overall

- TRAIN: n=709, PF=0.967, win=0.426, exp=-0.014, total_R=-10.091
- TEST: n=516, PF=1.029, win=0.430, exp=0.013, total_R=6.515


## By direction

| direction | n_train | PF_train | n_test | PF_test | win_test | exp_test | total_R_test | ✅ |
|---|---|---|---|---|---|---|---|---|
| SHORT | 353 | 1.064 | 247 | 1.125 | 0.441 | 0.053 | 13.085 |  |
| LONG | 356 | 0.875 | 269 | 0.947 | 0.420 | -0.024 | -6.570 |  |

## By session (NY = NY_overlap + NY_afternoon)

| session | n_train | PF_train | n_test | PF_test | win_test | exp_test | total_R_test | ✅ |
|---|---|---|---|---|---|---|---|---|
| London | 200 | 1.095 | 164 | 1.169 | 0.427 | 0.080 | 13.146 |  |
| OffHours | 22 | 0.478 | 16 | 1.056 | 0.562 | 0.019 | 0.306 |  |
| Asian | 127 | 0.872 | 99 | 1.051 | 0.424 | 0.025 | 2.507 |  |
| NY | 360 | 0.944 | 237 | 0.902 | 0.426 | -0.040 | -9.444 |  |

## By session (fine: NY split into overlap/afternoon)

| session_fine | n_train | PF_train | n_test | PF_test | win_test | exp_test | total_R_test | ✅ |
|---|---|---|---|---|---|---|---|---|
| NY_afternoon | 138 | 0.611 | 82 | 1.191 | 0.512 | 0.049 | 4.000 |  |
| London | 200 | 1.095 | 164 | 1.169 | 0.427 | 0.080 | 13.146 |  |
| OffHours | 22 | 0.478 | 16 | 1.056 | 0.562 | 0.019 | 0.306 |  |
| Asian | 127 | 0.872 | 99 | 1.051 | 0.424 | 0.025 | 2.507 |  |
| NY_overlap | 222 | 1.092 | 155 | 0.821 | 0.381 | -0.087 | -13.444 |  |

## By ATR quartile

| ATR_Q | n_train | PF_train | n_test | PF_test | win_test | exp_test | total_R_test | ✅ |
|---|---|---|---|---|---|---|---|---|
| Q1_low | 70 | 0.790 | 56 | 1.290 | 0.429 | 0.153 | 8.560 |  |
| Q2 | 109 | 1.059 | 74 | 1.101 | 0.459 | 0.050 | 3.679 |  |
| Q4_high | 347 | 0.972 | 243 | 0.973 | 0.440 | -0.010 | -2.344 |  |
| Q3 | 183 | 0.979 | 143 | 0.955 | 0.399 | -0.024 | -3.380 |  |

## By day of week

| dow | n_train | PF_train | n_test | PF_test | win_test | exp_test | total_R_test | ✅ |
|---|---|---|---|---|---|---|---|---|
| 1 | 131 | 0.800 | 96 | 1.167 | 0.448 | 0.074 | 7.150 |  |
| 4 | 154 | 0.679 | 93 | 1.137 | 0.441 | 0.059 | 5.523 |  |
| 0 | 121 | 1.384 | 91 | 1.107 | 0.462 | 0.044 | 3.991 |  |
| 3 | 165 | 1.217 | 121 | 0.973 | 0.446 | -0.012 | -1.464 |  |
| 2 | 132 | 0.962 | 111 | 0.865 | 0.369 | -0.063 | -6.951 |  |
| 6 | 6 | 0.468 | 4 | 0.166 | 0.250 | -0.434 | -1.734 |  |

## By session × ATR_Q (the canonical XAU breakdown)

| session | ATR_Q | n_train | PF_train | n_test | PF_test | win_test | exp_test | total_R_test | ✅ |
|---|---|---|---|---|---|---|---|---|---|
| NY | Q1_low | 1 | 999.000 | 2 | 999.000 | 1.000 | 1.172 | 2.344 |  |
| OffHours | Q3 | 7 | 0.246 | 4 | 3.392 | 0.750 | 0.255 | 1.019 |  |
| London | Q2 | 59 | 1.392 | 32 | 1.645 | 0.531 | 0.264 | 8.456 |  |
| Asian | Q1_low | 58 | 0.857 | 46 | 1.442 | 0.457 | 0.225 | 10.330 |  |
| London | Q4_high | 51 | 0.627 | 45 | 1.428 | 0.467 | 0.147 | 6.630 |  |
| OffHours | Q4_high | 9 | 0.995 | 8 | 1.051 | 0.625 | 0.019 | 0.154 |  |
| London | Q3 | 80 | 1.280 | 80 | 1.049 | 0.388 | 0.026 | 2.096 |  |
| NY | Q2 | 10 | 0.874 | 6 | 1.042 | 0.500 | 0.015 | 0.089 |  |
| NY | Q4_high | 272 | 1.108 | 176 | 0.882 | 0.426 | -0.044 | -7.761 |  |
| NY | Q3 | 77 | 0.654 | 53 | 0.854 | 0.396 | -0.078 | -4.116 |  |
| Asian | Q2 | 35 | 0.724 | 33 | 0.787 | 0.394 | -0.124 | -4.077 |  |
| OffHours | Q2 | 5 | 0.385 | 3 | 0.606 | 0.333 | -0.263 | -0.789 |  |
| Asian | Q4_high | 15 | 0.634 | 14 | 0.540 | 0.429 | -0.098 | -1.367 |  |
| Asian | Q3 | 19 | 1.533 | 6 | 0.405 | 0.333 | -0.397 | -2.379 |  |
| London | Q1_low | 10 | 0.505 | 7 | 0.327 | 0.143 | -0.577 | -4.036 |  |
| OffHours | Q1_low | 1 | 0.000 | 1 | 0.000 | 0.000 | -0.078 | -0.078 |  |

## By session × direction

| session | direction | n_train | PF_train | n_test | PF_test | win_test | exp_test | total_R_test | ✅ |
|---|---|---|---|---|---|---|---|---|---|
| OffHours | LONG | 14 | 0.364 | 7 | 1.479 | 0.429 | 0.171 | 1.199 |  |
| London | SHORT | 103 | 1.036 | 80 | 1.375 | 0.463 | 0.165 | 13.185 |  |
| Asian | LONG | 67 | 0.584 | 57 | 1.055 | 0.421 | 0.028 | 1.602 |  |
| Asian | SHORT | 60 | 1.300 | 42 | 1.044 | 0.429 | 0.022 | 0.905 |  |
| London | LONG | 97 | 1.163 | 84 | 0.999 | 0.393 | -0.000 | -0.039 |  |
| NY | SHORT | 182 | 0.994 | 116 | 0.998 | 0.414 | -0.001 | -0.112 |  |
| NY | LONG | 178 | 0.893 | 121 | 0.813 | 0.438 | -0.077 | -9.332 |  |
| OffHours | SHORT | 8 | 0.759 | 9 | 0.702 | 0.667 | -0.099 | -0.893 |  |

## Filter combinations (sweep)

Acceptance: PF_test ≥ 1.30 AND n_test ≥ 100 AND PF stable train↔test

| Rules | n_train | PF_train | n_test | PF_test | win_test | exp_test | total_R_test | sig/yr_test | ✅ |
|---|---|---|---|---|---|---|---|---|---|
| R2_skip_NY + R5_skip_OffHours | 327 | 1.005 | 263 | 1.123 | 0.426 | 0.060 | 15.653 | 88 |  |
| R2_skip_NY + R3_skip_NY_Q4 + R5_skip_OffHours | 327 | 1.005 | 263 | 1.123 | 0.426 | 0.060 | 15.653 | 88 |  |
| R2_skip_NY | 349 | 0.982 | 279 | 1.120 | 0.434 | 0.057 | 15.959 | 93 |  |
| R2_skip_NY + R3_skip_NY_Q4 | 349 | 0.982 | 279 | 1.120 | 0.434 | 0.057 | 15.959 | 93 |  |
| R1_skip_Q4_high + R2_skip_NY + R5_skip_OffHours | 261 | 1.088 | 204 | 1.096 | 0.417 | 0.051 | 10.390 | 68 |  |
| R1_skip_Q4_high + R2_skip_NY + R3_skip_NY_Q4 + R5_skip_OffHours | 261 | 1.088 | 204 | 1.096 | 0.417 | 0.051 | 10.390 | 68 |  |
| R1_skip_Q4_high + R2_skip_NY | 274 | 1.057 | 212 | 1.095 | 0.420 | 0.050 | 10.542 | 71 |  |
| R1_skip_Q4_high + R2_skip_NY + R3_skip_NY_Q4 | 274 | 1.057 | 212 | 1.095 | 0.420 | 0.050 | 10.542 | 71 |  |
| R3_skip_NY_Q4 + R5_skip_OffHours | 415 | 0.933 | 324 | 1.089 | 0.426 | 0.043 | 13.970 | 109 |  |
| R3_skip_NY_Q4 | 437 | 0.918 | 340 | 1.088 | 0.432 | 0.042 | 14.276 | 114 |  |
| R1_skip_Q4_high + R5_skip_OffHours | 349 | 0.984 | 265 | 1.063 | 0.419 | 0.033 | 8.707 | 89 |  |
| R1_skip_Q4_high + R3_skip_NY_Q4 + R5_skip_OffHours | 349 | 0.984 | 265 | 1.063 | 0.419 | 0.033 | 8.707 | 89 |  |
| R1_skip_Q4_high | 362 | 0.964 | 273 | 1.063 | 0.421 | 0.032 | 8.859 | 91 |  |
| R1_skip_Q4_high + R3_skip_NY_Q4 | 362 | 0.964 | 273 | 1.063 | 0.421 | 0.032 | 8.859 | 91 |  |
| R2_skip_NY + R4_long_only | 178 | 0.863 | 148 | 1.037 | 0.405 | 0.019 | 2.762 | 50 |  |
| R2_skip_NY + R3_skip_NY_Q4 + R4_long_only | 178 | 0.863 | 148 | 1.037 | 0.405 | 0.019 | 2.762 | 50 |  |
| ALL (no filter) | 709 | 0.967 | 516 | 1.029 | 0.430 | 0.013 | 6.515 | 173 |  |
| R5_skip_OffHours | 687 | 0.979 | 500 | 1.028 | 0.426 | 0.012 | 6.209 | 168 |  |
| R2_skip_NY + R4_long_only + R5_skip_OffHours | 164 | 0.893 | 141 | 1.022 | 0.404 | 0.011 | 1.563 | 47 |  |
| R2_skip_NY + R3_skip_NY_Q4 + R4_long_only + R5_skip_OffHours | 164 | 0.893 | 141 | 1.022 | 0.404 | 0.011 | 1.563 | 47 |  |
| R1_skip_Q4_high + R2_skip_NY + R4_long_only + R5_skip_OffHours | 130 | 0.979 | 113 | 1.006 | 0.381 | 0.004 | 0.408 | 38 |  |
| R1_skip_Q4_high + R2_skip_NY + R3_skip_NY_Q4 + R4_long_only + R5_skip_OffHours | 130 | 0.979 | 113 | 1.006 | 0.381 | 0.004 | 0.408 | 38 |  |
| R1_skip_Q4_high + R2_skip_NY + R4_long_only | 139 | 0.953 | 117 | 0.995 | 0.376 | -0.002 | -0.291 | 39 |  |
| R1_skip_Q4_high + R2_skip_NY + R3_skip_NY_Q4 + R4_long_only | 139 | 0.953 | 117 | 0.995 | 0.376 | -0.002 | -0.291 | 39 |  |
| R3_skip_NY_Q4 + R4_long_only | 220 | 0.862 | 177 | 0.988 | 0.401 | -0.006 | -1.075 | 59 |  |
| R3_skip_NY_Q4 + R4_long_only + R5_skip_OffHours | 206 | 0.886 | 170 | 0.974 | 0.400 | -0.013 | -2.274 | 57 |  |
| R1_skip_Q4_high + R4_long_only + R5_skip_OffHours | 172 | 0.952 | 142 | 0.957 | 0.380 | -0.024 | -3.429 | 48 |  |
| R1_skip_Q4_high + R3_skip_NY_Q4 + R4_long_only + R5_skip_OffHours | 172 | 0.952 | 142 | 0.957 | 0.380 | -0.024 | -3.429 | 48 |  |
| R1_skip_Q4_high + R4_long_only | 181 | 0.932 | 146 | 0.949 | 0.377 | -0.028 | -4.128 | 49 |  |
| R1_skip_Q4_high + R3_skip_NY_Q4 + R4_long_only | 181 | 0.932 | 146 | 0.949 | 0.377 | -0.028 | -4.128 | 49 |  |
| R4_long_only | 356 | 0.875 | 269 | 0.947 | 0.420 | -0.024 | -6.570 | 90 |  |
| R4_long_only + R5_skip_OffHours | 342 | 0.893 | 262 | 0.936 | 0.420 | -0.030 | -7.769 | 88 |  |

## Top 3 stable filter combos

**No filter combo passes the stability gate.** EURUSD does NOT have a transferable XAU-style edge under this config.

## Verdict

❌ **REJECT** — no filter combination passes the stability gate. The XAU regime-filter edge does NOT transfer to EURUSD under this config (enter=40/exit=25, default SL=1.5×ATR, TP=3×ATR).

## NY × Q4_high specifically

- n_test=176, PF_test=0.882, total_R_test=-7.761, win=0.426
→ **Saigneur confirmé** (same pattern as XAU).