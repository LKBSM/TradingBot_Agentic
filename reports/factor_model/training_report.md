# Factor Model — institutional XAU predictor

**Date** : 2026-05-16T10:37:22.308177
**Mode** : FULL
**Horizon** : 96 M15 bars = 1440 min
**Total bars** : 172874, train 121011 / OOS 51863
**Features** : 24 (macro 11 + microstructure 13)

## OOS Performance

| Metric | Value | Verdict |
| --- | --- | --- |
| OOS R² | -0.044611 | ❌ noise |
| Directional accuracy | 0.4964 | ❌ not enough edge |
| IC Pearson | +0.042179 | ✅ signal |
| IC Spearman | -0.014659 | ❌ noise |
| Buy-and-Hold Sharpe (ann) | +8.512 | baseline |
| Strategy Sharpe (ann) | +1.594 | strat |
| **Information Ratio** | **-6.918** | ❌ no alpha vs BH |

## Top 15 features (LightGBM importance)

| Feature | Importance |
| --- | --- |
| `dxy_z` | 1343 |
| `real_10y_z` | 1081 |
| `rv_london_1d` | 1049 |
| `yield_curve_2s10s` | 980 |
| `rv_ny_1d` | 945 |
| `vix_level` | 937 |
| `real_10y` | 929 |
| `dxy_slope_20d` | 725 |
| `rv_asia_1d` | 661 |
| `gk_vol_1d` | 651 |
| `cot_mm_net_pct` | 593 |
| `dxy_level` | 563 |
| `cot_mm_net_z52` | 518 |
| `roll_spread_estimate` | 465 |
| `rv_1d` | 422 |

## Interpretation

❌ **Pas d'edge OOS** sur ces features. Possibles directions :
- Ajouter features cross-asset (gold vs silver, gold vs SPX).
- Régresser sur résidus de modèle macro (separate beta from alpha).
- Walk-forward refit (les facteurs macro driftent — refit mensuel).