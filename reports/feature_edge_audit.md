# Feature Edge Audit — XAU/USD M15 (Chantier 1)

**Trades**: 2363  |  **Train**: 1182 (< 2023-01-01)  |  **Test**: 1181 (≥ 2023-01-01)

**Decision criterion**: stable_edge = ✅ if |Pearson_test| ≥ 0.05 AND sign(Pearson_train) = sign(Pearson_test) AND n_test ≥ 100


---

## Baseline — confluence_score (the current product)

- **Train**: n=1182, Pearson=+0.0035, Spearman=+0.0186
- **Test**:  n=1181, Pearson=-0.0139, Spearman=-0.0035
- **Verdict**: edge ❌ (-0.0139)


## Per-feature edge (sorted by |Pearson_test|)

| feature | n_train | n_test | pearson_train | pearson_test | spearman_test | mi_test | long_pearson_test | short_pearson_test | stable_edge |
|---|---|---|---|---|---|---|---|---|---|
| FVG_SIGNAL_aligned | 1182 | 1181 | -0.0195 | +0.0711 | +0.0493 | +0.0230 | +0.0501 | +0.0961 | — |
| ATR_PCTL | 1180 | 1181 | -0.0168 | -0.0599 | +0.0020 | +0.0915 | -0.0948 | -0.0047 | ✅ |
| BB_POS_aligned | 1182 | 1181 | -0.0150 | +0.0516 | +0.0233 | +0.0143 | +0.0487 | +0.0497 | — |
| RSI | 1182 | 1181 | +0.0084 | +0.0402 | +0.0307 | +0.0000 | +0.0187 | -0.0023 | — |
| ATR | 1182 | 1181 | -0.0305 | -0.0335 | +0.0719 | +0.1542 | -0.0548 | -0.0078 | — |
| hour | 1182 | 1181 | -0.0186 | -0.0306 | -0.0067 | +0.0354 | -0.0355 | -0.0201 | — |
| dow | 1182 | 1181 | +0.0070 | +0.0199 | +0.0195 | +0.0000 | -0.0167 | +0.0696 | — |
| BODY_RATIO | 1182 | 1181 | +0.0316 | -0.0174 | -0.0201 | +0.0166 | -0.0356 | +0.0099 | — |
| CHOCH_DIVERGENCE_aligned | 1182 | 1181 | -0.0259 | +0.0158 | +0.0145 | +0.0127 | +0.0218 | +0.0058 | — |
| FVG_aligned_size | 1182 | 1181 | +0.0343 | +0.0132 | +0.0437 | +0.0000 | +0.0194 | +0.0002 | — |
| OB_STRENGTH_NORM | 1182 | 1181 | +0.0045 | -0.0069 | -0.0466 | +0.0305 | -0.0124 | +0.0025 | — |
| MACD_Diff_aligned | 1182 | 1181 | -0.0145 | +0.0068 | +0.0107 | +0.0266 | +0.0011 | +0.0153 | — |
| FVG_SIZE_NORM | 1182 | 1181 | +0.0222 | +0.0014 | +0.0287 | +0.0000 | +0.0139 | -0.0192 | — |
| BOS_SIGNAL_aligned | 1182 | 1181 | — | — | — | — | — | — | — |
| BOS_EVENT_aligned | 1182 | 1181 | — | — | — | — | — | — | — |
| CHOCH_SIGNAL_aligned | 1182 | 1181 | — | — | — | — | — | — | — |
| BOS_RETEST_ARMED_aligned | 1182 | 1181 | — | — | — | — | — | — | — |
| RETEST_aligned_bool | 1182 | 1181 | — | — | — | — | — | — | — |

## Stable features (decision: keep for Chantier 2)

- **ATR_PCTL**: Pearson_test -0.0599 (train -0.0168), n=1181, MI=+0.0915