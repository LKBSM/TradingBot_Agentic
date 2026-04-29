# Filter Strategy Audit — XAU/USD M15 (Chantier 3 step 2)

Train years: 4.00 | Test years: 2.99
Acceptance: PF_test ≥ 1.30 AND n_test ≥ 100 AND PF_test stable vs PF_train (same sign of edge, |drop| < 0.20)
---

| Rules | n_train | PF_train | n_test | PF_test | win_test | exp_test | total_R_test | sig/yr_test | ✅ |
|---|---|---|---|---|---|---|---|---|---|
| R1_skip_Q4_high + R3_skip_Tuesday + R4_long_only | 305 | 1.132 | 315 | 1.472 | 0.514 | 0.131 | 41.134 | 105 |  |
| R1_skip_Q4_high + R2_skip_NY + R3_skip_Tuesday + R4_long_only | 205 | 1.204 | 212 | 1.462 | 0.500 | 0.135 | 28.592 | 71 |  |
| R2_skip_NY + R3_skip_Tuesday + R4_long_only | 274 | 1.263 | 332 | 1.433 | 0.509 | 0.106 | 35.261 | 111 | ✅ |
| R1_skip_Q4_high + R2_skip_NY + R4_long_only | 266 | 1.159 | 255 | 1.425 | 0.502 | 0.131 | 33.374 | 85 |  |
| R2_skip_NY + R4_long_only | 358 | 1.179 | 394 | 1.396 | 0.508 | 0.102 | 40.179 | 132 |  |
| R1_skip_Q4_high + R4_long_only | 399 | 1.068 | 389 | 1.385 | 0.504 | 0.110 | 42.945 | 130 |  |
| R1_skip_Q4_high + R2_skip_NY + R3_skip_Tuesday | 386 | 1.242 | 341 | 1.379 | 0.501 | 0.115 | 39.071 | 114 | ✅ |
| R1_skip_Q4_high + R2_skip_NY | 489 | 1.186 | 416 | 1.355 | 0.490 | 0.110 | 45.799 | 139 | ✅ |
| R1_skip_Q4_high + R3_skip_Tuesday | 551 | 1.143 | 516 | 1.352 | 0.500 | 0.105 | 54.243 | 172 |  |
| R2_skip_NY + R3_skip_Tuesday | 523 | 1.230 | 551 | 1.329 | 0.499 | 0.083 | 45.784 | 184 | ✅ |
| R1_skip_Q4_high | 705 | 1.098 | 647 | 1.304 | 0.491 | 0.091 | 58.821 | 216 |  |
| R3_skip_Tuesday + R4_long_only | 483 | 1.208 | 546 | 1.292 | 0.502 | 0.071 | 38.932 | 182 |  |
| R2_skip_NY | 669 | 1.158 | 662 | 1.281 | 0.488 | 0.073 | 48.583 | 221 |  |
| R4_long_only | 632 | 1.157 | 665 | 1.238 | 0.496 | 0.060 | 39.704 | 222 |  |
| R3_skip_Tuesday | 926 | 1.193 | 954 | 1.182 | 0.487 | 0.046 | 43.763 | 319 |  |
| ALL (no filter) | 1182 | 1.133 | 1181 | 1.126 | 0.479 | 0.032 | 38.170 | 395 |  |

## Top stable filters

- **R2_skip_NY + R3_skip_Tuesday + R4_long_only** — PF_test=1.433, n=332
- **R1_skip_Q4_high + R2_skip_NY + R3_skip_Tuesday** — PF_test=1.379, n=341
- **R1_skip_Q4_high + R2_skip_NY** — PF_test=1.355, n=416
- **R2_skip_NY + R3_skip_Tuesday** — PF_test=1.329, n=551