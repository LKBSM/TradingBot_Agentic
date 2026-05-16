# Factor Model — institutional XAU predictor

**Date** : 2026-05-16T10:35:06.373161
**Mode** : QUICK (50k bars)
**Horizon** : 4 M15 bars = 60 min
**Total bars** : 50000, train 35000 / OOS 15000
**Features** : 24 (macro 11 + microstructure 13)

## OOS Performance

| Metric | Value | Verdict |
| --- | --- | --- |
| OOS R² | -0.009850 | ❌ noise |
| Directional accuracy | 0.5101 | ❌ not enough edge |
| IC Pearson | -0.002234 | ❌ noise |
| IC Spearman | +0.005200 | ❌ noise |
| Buy-and-Hold Sharpe (ann) | +1.121 | baseline |
| Strategy Sharpe (ann) | -0.454 | strat |
| **Information Ratio** | **-1.575** | ❌ no alpha vs BH |

## Top 15 features (LightGBM importance)

| Feature | Importance |
| --- | --- |
| `rv_london_1d` | 1027 |
| `rv_asia_1d` | 993 |
| `rv_ny_1d` | 968 |
| `volume_ratio` | 875 |
| `gk_vol_1d` | 868 |
| `dxy_z` | 849 |
| `rv_1d` | 789 |
| `volume_z` | 700 |
| `vix_level` | 652 |
| `roll_spread_estimate` | 610 |
| `real_10y_z` | 593 |
| `real_10y` | 507 |
| `dxy_slope_20d` | 478 |
| `dxy_level` | 429 |
| `cot_mm_net_pct` | 405 |

## Interpretation

🟡 **Edge marginal**. Directional > 51% mais R² faible. Tester en walk-forward.