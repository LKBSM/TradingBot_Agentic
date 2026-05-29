# Eval 03 — Smart Money Engine — Stats

- Période : 2019-01-02 12:15:00 → 2024-12-30 23:45:00
- Bars totaux : 141,499

## Compteurs d'événements

| Evénement | Up/Bull | Down/Bear | /1k bars up | /1k bars down |
|-----------|---------|-----------|-------------|---------------|
| BOS | 2340 | 2051 | 16.54 | 14.49 |
| CHOCH | 842 | 842 | 5.95 | 5.95 |
| FVG | 11763 | 10785 | 83.13 | 76.22 |
| Order Block | 17475 | 17247 | 123.5 | 121.89 |

## Retest state machine
- Total BOS events : **4391**
- Awaiting → Armed : **3941** (taux succès 89.8%)
- Invalidés/timeout : 245
- Armed up / down : 2104 / 1837

## Outcomes armed (TP=2R, SL=1R, fenêtre 50 bars)
- LONG armed : 695W / 1381L / 28amb — win rate 33.5%
- SHORT armed : 550W / 1267L / 20amb — win rate 30.3%

## Par année — BOS / Armed
| Année | Bars | BOS↑ | BOS↓ | CHOCH↑ | CHOCH↓ | OB↑ | OB↓ | FVG↑ | FVG↓ |
|-------|------|------|------|--------|--------|-----|-----|------|------|
| 2019 | 23,450 | 385 | 325 | 133 | 132 | 2785 | 2718 | 2127 | 1946 |
| 2020 | 23,611 | 386 | 278 | 128 | 129 | 2959 | 2956 | 1815 | 1621 |
| 2021 | 23,581 | 386 | 354 | 145 | 144 | 2859 | 2863 | 1995 | 1827 |
| 2022 | 23,653 | 435 | 415 | 158 | 158 | 2941 | 2937 | 1955 | 1911 |
| 2023 | 23,558 | 354 | 353 | 142 | 143 | 2914 | 2878 | 1917 | 1762 |
| 2024 | 23,646 | 394 | 326 | 136 | 136 | 3017 | 2895 | 1954 | 1718 |

## Forward post-BOS (close du break, MFE/MAE en ATR)
### h_5
| Dir | n | mean MFE | median MFE | mean MAE | median MAE |
|-----|---|----------|------------|----------|------------|
| up | 2340 | 1.398 | 0.973 | 1.372 | 1.022 |
| down | 2051 | 1.438 | 0.995 | 1.425 | 1.098 |
### h_20
| Dir | n | mean MFE | median MFE | mean MAE | median MAE |
|-----|---|----------|------------|----------|------------|
| up | 2340 | 2.763 | 1.971 | 2.654 | 1.925 |
| down | 2051 | 2.659 | 1.901 | 2.702 | 1.98 |
### h_50
| Dir | n | mean MFE | median MFE | mean MAE | median MAE |
|-----|---|----------|------------|----------|------------|
| up | 2340 | 4.528 | 3.134 | 4.045 | 2.892 |
| down | 2051 | 4.126 | 2.763 | 4.291 | 3.117 |

## Latence détection (bars fractal → BOS event)
- BOS↑ : median **4.0** bars, p25 2.0, p75 6.0 (n=2340)
- BOS↓ : median **4.0** bars, p25 2.0, p75 6.0 (n=2051)