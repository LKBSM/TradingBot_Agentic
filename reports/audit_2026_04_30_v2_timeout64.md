# Audit XAU/USD M15 v2 — quick-win timeout 24 → 64 bars
## Confirmation forensic L1
**Date :** 2026-04-30
**Changes :** `CFG["max_lifetime_bars"] = 24 → 64` dans `scripts/quant_audit_2026_04_30.py` ; default `signal_state_machine.StateMachineConfig.max_signal_age_bars = 12 → 64`.
**Données :** XAU_15MIN_2019_2026.csv (172 874 barres), seed=42 — bit-à-bit identique au v1 hors timeout
**Coûts :** spread 0.30 USD, slippage 0.10-0.20, commission 7 USD/lot RT, capital 10k USD, risk 1%/trade

---

## A. Comparatif v1 (timeout=24) vs v2 (timeout=64)

| Métrique | **v1 (timeout=24)** | **v2 (timeout=64)** | Δ |
|---|---:|---:|---:|
| Profit Factor | 0.786 | **1.037** | **+0.251** ✅ |
| Win rate | 44.18% | 41.02% | -3.16 pp |
| Sharpe (mensuel ann.) | -0.860 | **+0.274** | +1.134 |
| Sortino | -1.309 | **+0.454** | +1.763 |
| Calmar | -0.801 | **+0.309** | +1.110 |
| Max Drawdown | -77.92% | **-48.02%** | +29.9 pp ✅ |
| Durée DD max | 2 662 jours | 1 825 jours | -837 jours |
| Avg R:R réalisé | 0.993 | 1.492 | +0.499 |
| Expectancy / trade | -0.032 R | **+0.013 R** | +0.045 R ✅ |
| Pearson(score, R) | -0.0010 | -0.0075 | -0.0065 (idem 0) |
| Capital final | 3 754 USD | **11 482 USD** | **+7 728 USD** |
| Return total | -62.46% | **+14.82%** | **+77.28 pp** |
| Trades totaux | 2 363 | 1 753 | -610 |
| Buy & Hold XAU | +255.60% | +255.60% | identique |

**Le système passe de "destructeur de capital" à "modestement profitable" sur 7 ans.** Confirmation bit-à-bit de la prédiction L1 (PF 1.037 vs prédit 1.04 [CI 0.92, 1.17]).

---

## B. Décomposition par année — v2

| Année | n | WR | **PF** | PnL USD |
|:---|---:|---:|---:|---:|
| 2019 | 229 | 39.7% | 0.914 | -470 |
| **2020** | 238 | 44.1% | **1.195** | +938 |
| 2021 | 254 | 33.9% | 0.621 | -2 358 |
| 2022 | 225 | 39.6% | 0.746 | -1 130 |
| 2023 | 243 | 35.0% | 0.702 | -1 220 |
| **2024** | 231 | 46.3% | **1.441** | +1 340 |
| **2025** | 239 | 47.3% | **1.406** | +2 591 |
| **2026 YTD** | 94 | 45.7% | **1.350** | +1 791 |

**4 années profitables (2020, 2024, 2025, 2026)** vs 3 en v1. **2021 reste le pire année** (PF 0.62, -2358 USD) — corrélé au range XAU 2021 (qui n'a ni bull ni bear net).

---

## C. Décomposition par régime — v2

| Régime | n | WR | **PF** | PnL |
|:---|---:|---:|---:|---:|
| Bear | 679 | 33.7% | **0.722** | -5 005 |
| Bull | 836 | 45.3% | **1.301** | +5 082 |
| Range | 238 | 46.6% | **1.298** | +1 405 |

**Lecture :** Bear continue de saigner (-5 005 USD, PF 0.72). Bull et Range sont profitables. Le pattern reste asymétrique — **le système est foncièrement long-biased en bear market**.

## D. Side decomposition — v2

| Side | n | WR | **PF** | PnL |
|:---|---:|---:|---:|---:|
| **Long (+1)** | 898 | **45.4%** | **1.305** | **+5 595** |
| Short (-1) | 855 | 36.4% | 0.806 | -4 113 |

**v2 amplifie l'asymétrie long/short** : Long PF 1.31 (vs 0.95 en v1) ; Short PF 0.81 (vs 0.64 en v1). Le timeout long bénéficie surtout aux longs — confirmation de la thèse "système long-only sur XAU bull".

---

## E. Edges par composant — v2

| Composant | n_on | n_off | edge_R | Δ vs v1 |
|---|---:|---:|---:|---:|
| **fvg** | 1 414 | 339 | **+0.043** | +0.016 |
| bos | 1 680 | 73 | +0.031 | +0.016 |
| regime | 1 224 | 529 | +0.011 | -0.001 |
| retest | 1 612 | 141 | +0.002 | -0.020 |
| ob | 1 716 | 37 | -0.005 | +0.001 |
| rsi_div | 591 | 1 162 | -0.019 | -0.001 |
| **choch** | 794 | 959 | -0.028 | -0.010 |
| news_ok | 1 750 | 3 | -0.518 | (n_off=3, non-significatif) |

**FVG reste le seul composant à edge positif notable** (+0.043 R, en hausse vs v1). **CHOCH s'aggrave** (-0.028 vs -0.018 en v1). Le scoring reste structurellement aussi prédictif que zéro (Pearson -0.0075).

---

## F. Distribution des sorties — v2

Avec timeout 64 bars, on attend moins de SL (positions ont le temps de respirer) et plus de TP/timeout :

| Exit reason | v1 (timeout=24) | v2 (timeout=64) | Δ |
|:---|---:|---:|---:|
| timeout | 1 634 (69.1%) | 781 (44.6%) | -25 pp |
| sl | 314 (13.3%) | 218 (12.4%) | -1 pp |
| tp | 113 (4.8%) | 130 (7.4%) | **+2.6 pp** ✅ |
| opposite | 302 (12.8%) | 624 (35.6%) | **+22.8 pp** |

**Observation cruciale** : avec timeout 64, le **bug exit "opposite" devient dominant** (35.6 % des sorties vs 12.8 % en v1). Cela signifie qu'avec un timeout plus long, le système a plus d'opportunités pour qu'un signal opposé arrive et coupe la position prématurément. **Le bug "opposite" identifié dans le rapport initial est encore plus critique en v2.**

---

## G. Verdict v2 (200 mots)

**Le quick-win timeout 24→64 fait basculer le système au-dessus de PF 1.0** (1.037 sur 7 ans, return +14.82 % vs -62.46 % en v1). **C'est un changement de configuration de 5 minutes qui produit un gain de 77 points de pourcentage de return.** Confirmation bit-à-bit du finding forensic L1.

**Mais ne pas confondre cela avec une stratégie commercialisable :**
- PF 1.04 reste **bien sous la cible 1.20**.
- Max DD -48 % reste **rédhibitoire** pour facturer 29-149 USD/mois (cf eval_27).
- Bear regime reste cassé (PF 0.72) — le système continue de saigner sur bear/range.
- Pearson(score, R) = -0.0075 — **le scoring 8-composants n'a toujours aucun pouvoir prédictif**. Le gain vient du paramètre timeout, pas du score.
- **Bug "opposite" dominant à 35.6 %** → priorité absolue à corriger en Sprint 1.

**Recommandation :** v2 est un baseline plus crédible pour comparer XAU H1 et EURUSD M15 (Actions 2-3). À ce stade, **ne pas commercialiser v2** — c'est juste un meilleur point de départ pour la décision finale.

---

## ANNEXES

- `reports/audit_2026_04_30_v2_timeout64_trades.csv` — 1 753 trades v2
- `reports/audit_2026_04_30_v2_timeout64_summary.json` — métriques v2
- `reports/audit_2026_04_30_v2_timeout64_equity.csv` — equity curve v2
- `reports/audit_2026_04_30_v2_run.log` — console log
- `reports/audit_2026_04_30_*.{csv,json}` (sans suffix) — pointent vers v2 actuellement (canonical)

**Note traçabilité v1 :** le ledger et summary v1 (timeout=24) ont été écrasés par le run v2. Les 2 363 trades v1 sont préservés via le rapport `audit_2026_04_30_quant_senior.md` (synthèse) et toutes les analyses de la falsification (`reports/falsification/L1_D_*.csv`). Pour reproduire bit-à-bit v1, restaurer `CFG["max_lifetime_bars"] = 24` et re-runner le script.
