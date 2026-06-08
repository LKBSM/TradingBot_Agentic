# Comparatif XAU/USD H1 — pipeline SMC out-of-TF
## Action 2 — test du pipeline sur XAU H1 (mêmes seuils, agg M15→H1)
**Date :** 2026-04-30
**Sources :**  `scripts/comparatif_xau_h1.py`, `reports/comparatif_xau_h1_*.{csv,json}`
**Données :** XAU M15 re-aggrégé en H1 (43 243 barres, 2019-01-02 → 2026-04-29)
**Coûts :** spread 0.30 USD, slippage 0.10-0.20, commission 7 USD/lot RT, capital 10 000 USD, risk 1%/trade, seed=42
**Adaptations :** ATR_14 H1, SMA200 H1, max_lifetime=24 H1 bars (=1 jour), news ±60 min, HAR-RV recalibré (24 bars/jour)

> **Test out-of-domain honnête : aucun re-tuning des seuils du score sur H1.**

---

## A. Métriques globales XAU H1

| Métrique | Valeur | vs XAU M15 v2 | Verdict |
|---|---:|---:|:---:|
| **Profit Factor** | **0.946** | -0.091 | ❌ < 1.0 |
| **CI 95 % bootstrap PF** | **[0.758, 1.166]** | inclut 1.0 | — |
| Win rate | 45.68 % | +4.66 pp | ✓ |
| Sharpe (mensuel ann.) | -0.165 | -0.439 | ❌ |
| Sortino | -0.278 | -0.732 | ❌ |
| **Max Drawdown** | **-16.17 %** | **+31.85 pp** ✅ | ✓ |
| Return total | -8.30 % | -23.12 pp | ❌ |
| Trades totaux | 532 | -1 221 | — |
| Pearson(score, R) | **+0.0161** | +0.0236 | ✓ légère |
| Buy & Hold XAU | +255.29 % | identique | — |

**Lecture brute** : H1 fait **moins mal que M15 v2 sur PF** (0.95 vs 1.04 — différence non significative car les CI se chevauchent), mais **drastiquement mieux sur MaxDD** (-16 % vs -48 %). L'expérience Sharpe reste négative car le système perd en moyenne malgré PF proche de 1.

**Critère GO/NO-GO (CI 95 % PF lo > 1.0)** : **❌ NO-GO** — CI lo = 0.758, exclut clairement la zone profitable.

---

## B. Décomposition par année

| Année | n | WR | **PF** | CI 95 % | PnL USD |
|:---|---:|---:|---:|---|---:|
| 2019 | 71 | 45.1 % | 0.767 | [0.382, 1.410] | -438 |
| 2020 | 63 | 42.9 % | 0.950 | [0.459, 1.841] | -87 |
| 2021 | 73 | 41.1 % | 0.586 | [0.296, 1.076] | -772 |
| **2022** | 69 | 55.1 % | **1.307** | [0.695, 2.430] | **+398** |
| 2023 | 77 | 42.9 % | 0.936 | [0.529, 1.594] | -155 |
| 2024 | 75 | 48.0 % | 0.883 | [0.490, 1.537] | -220 |
| **2025** | 73 | 46.6 % | **1.054** | [0.608, 1.740] | **+149** |
| **2026** | 31 | 41.9 % | **1.180** | [0.568, 2.263] | **+296** |

**Observations :**
- **3 années profitables** (2022, 2025, 2026) sur 8 — moitié moins que M15 v2 (4/8).
- **Pas de pic 2024-2026** comme en M15 (où PF 1.20-1.32) → H1 ne capture **pas** le bull XAU intraday.
- Variance des PF par année plus faible (0.59 - 1.31) qu'en M15 (0.62 - 1.44). H1 est plus **stable** mais pas profitable.

---

## C. Décomposition par side

| Side | n | WR | **PF** | CI 95 % | PnL USD |
|---|---:|---:|---:|---|---:|
| **Long (+1)** | 291 | 47.4 % | **1.014** | **[0.760, 1.355]** | +122 |
| Short (-1) | 241 | 43.6 % | 0.862 | [0.617, 1.188] | -952 |

**Lecture cruciale :**
- **Long PF 1.014 [CI 0.76, 1.36]** — très près de breakeven. CI inclut 1.0.
- **Short PF 0.862** — toujours sous-breakeven mais bien meilleur que M15 v2 (0.81) et **bien meilleur que M15 v1** (0.64).
- Symétrie nettement meilleure : ratio Long/|Short| PnL = 122 / 952 ≈ 0.13 (vs 8.3× sur M15 v2 long-only). **H1 n'est PAS un système long-only déguisé** → meilleure robustesse à un changement de régime.

---

## D. Décomposition par régime

| Régime | n | WR | **PF** | CI 95 % | PnL USD |
|---|---:|---:|---:|---|---:|
| Bull | 290 | 49.3 % | 0.991 | [0.741, 1.308] | -79 |
| Bear | 179 | 41.9 % | 0.871 | [0.598, 1.236] | -654 |
| Range | 63 | 39.7 % | 0.942 | [0.436, 1.786] | -97 |

**Lecture :** Distribution des PF beaucoup plus serrée qu'en M15 (Bear M15 v2 = 0.72 ; H1 = 0.87). Le bear cesse d'être un trou noir mais **aucun régime n'est nettement profitable**.

---

## E. Décomposition pré/post 2024

| Période | n | WR | **PF** | CI 95 % | PnL |
|---|---:|---:|---:|---|---:|
| Pre 2024 (2019-2023) | 353 | 45.3 % | 0.885 | [0.666, 1.146] | -1 055 |
| **Post 2024 (2024-2026)** | 179 | 46.4 % | **1.036** | [0.721, 1.436] | **+225** |

**Lecture :** Amélioration pré→post mais beaucoup plus modeste qu'en M15 (0.68 → 1.28). **L'effet régime bull est ~5× moins prononcé sur H1 que sur M15** — confirmation que la "performance" 2024-2026 sur M15 était pour l'essentiel de la β-capture intraday qui se dilue à H1.

---

## F. Verdict Action 2 (200 mots)

**Le pipeline SMC sur XAU H1 ne capture pas un edge.** PF 0.946 [CI 0.76, 1.17] sur 7 ans. CI inclut 1.0 mais aussi 0.76 — non significativement profitable. **Critère GO "CI 95 % PF lo > 1.0" : ❌ FAIL.**

**Trois aspects positifs vs XAU M15 v2** :
1. **MaxDD -16 %** vs -48 %. La même "performance" est obtenue avec 3× moins de drawdown.
2. **Symétrie long/short retrouvée** (PF 1.01 vs 0.86) — pas de biais long pathologique.
3. **Pearson(score, R) légèrement positif** (+0.016 vs -0.008 en M15) — premier signe de pouvoir prédictif détectable, bien que non significatif.

**Trois aspects négatifs :**
1. **Aucune année profitable sustained** — 3/8 années > 1.0, jamais 2 années consécutives.
2. **Buy & Hold +255 %** — le système sous-performe de 263 points le passive holding (pire que M15 v2 qui underperform 240 points).
3. **CI très large** (0.40 sur le PF) → 532 trades insuffisants pour conclusion robuste.

**Implication décision finale :** XAU H1 est moins **destructeur** que XAU M15 mais pas plus **profitable**. Le passage M15→H1 n'est pas un pivot évident. À combiner avec EURUSD M15 (Action 3) pour décision globale.

---

## ANNEXES

- `reports/comparatif_xau_h1_trades.csv` — 532 trades H1 ledger
- `reports/comparatif_xau_h1_summary.json` — métriques + décompositions
- `reports/comparatif_xau_h1_equity.csv` — equity curve
- `reports/comparatif_xau_h1_run.log` — console log
- `scripts/comparatif_xau_h1.py` — script reproductible (seed=42)
