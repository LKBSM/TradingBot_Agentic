# Walk-forward factor model — XAU

**Date** : 2026-05-16T10:44:47.035634
**Asset** : XAU
**Horizon** : 96 M15 bars (24.0 h)
**Train window** : 365 days, **refit every** 30 days
**Threshold quantile** : 0.60
**Total bars** : 172874

## Overall walk-forward

- Sharpe strategy : +0.399
- Sharpe B&H      : +1.018
- IR vs B&H       : **-0.619**
- Max DD strat    : -29.76%
- Max DD B&H      : -27.15%
- Final equity    : 1.401 vs B&H 3.190
- Exposed         : 63.8% of bars

## Per-period

| Period | Sharpe strat | Sharpe B&H | IR vs B&H | MaxDD strat | MaxDD B&H | Final eq strat | Final eq B&H |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2019-2021 (cyclical) | -0.317 | +0.755 | **-1.071** | -29.76% | -19.75% | 0.894 | 1.372 |
| 2022 (range-bound) | +1.166 | -0.014 | **+1.180** | -13.64% | -22.46% | 1.157 | 0.987 |
| 2023-2026 (bull) | +0.625 | +1.474 | **-0.849** | -24.53% | -27.15% | 1.356 | 2.356 |

## Verdict

🟡 **Edge conditionnel** : bat B&H sur ['2022 (range-bound)'].