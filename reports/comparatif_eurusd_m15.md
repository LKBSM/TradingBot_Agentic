# Comparatif EUR/USD M15 — pipeline SMC out-of-asset
## Action 3 — test du pipeline sur EURUSD M15
**Date :** 2026-04-30
**Sources :** `scripts/comparatif_eurusd_m15.py`, `reports/comparatif_eurusd_m15_*.{csv,json}`
**Données :** `data/EURUSD_15MIN_2019_2025.csv` (174 506 barres, 2019-01-01 → 2025-12-31, **pas de 2026 dispo**)
**Coûts FX :** spread 0.5 pip (0.00005), slippage 0.1-0.3 pip, commission 7 USD/lot RT, lot 100 000 EUR, capital 10 000 USD, risk 1%/trade, seed=42
**Adaptations FX :** SL clamp 10-100 pips (vs 2-30 USD pour XAU), max_lifetime=64 (consistent quick-win), news ±60 min

> **Test out-of-domain honnête : aucun re-tuning des seuils sur EURUSD.**

---

## A. Métriques globales EURUSD M15

| Métrique | Valeur | vs XAU M15 v2 | Verdict |
|---|---:|---:|:---:|
| **Profit Factor** | **0.854** | -0.183 | ❌ |
| **CI 95 % bootstrap PF** | **[0.748, 0.969]** | **CI EXCLUT 1.0** | ❌❌ |
| Win rate | 40.44 % | -0.58 pp | ≈ |
| Sharpe (mensuel ann.) | -0.955 | -1.229 | ❌ |
| Sortino | -1.420 | -1.874 | ❌ |
| Max Drawdown | -53.47 % | -5.45 pp | ❌ |
| Return total | -50.24 % | -65.05 pp | ❌ |
| Capital final | 4 976 USD | -6 506 USD | ❌ |
| Trades totaux | 1 805 | +52 | — |
| Pearson(score, R) | -0.0086 | -0.0011 | ≈ 0 |
| Buy & Hold EURUSD | **+2.46 %** | (vs +255 % XAU) | — |

**Lecture choc :** sur EURUSD M15, le système perd **50 % du capital** sur 7 ans (vs +14.8 % sur XAU M15 v2). **Le pipeline ne se transfère pas hors XAU.**

**Critère GO/NO-GO (CI 95 % PF lo > 1.0)** : **❌ NO-GO — CI 95 % top atteint 0.969, exclut 1.0 par 3 points.**

---

## B. Décomposition par année

| Année | n | WR | **PF** | CI 95 % | PnL USD |
|:---|---:|---:|---:|---|---:|
| 2019 | 252 | (calculé) | 0.948 | [0.666, 1.311] | -289 |
| 2020 | 258 | — | 0.899 | [0.655, 1.234] | -647 |
| 2021 | 260 | — | 0.939 | [0.684, 1.281] | -315 |
| 2022 | 250 | — | 0.778 | [0.561, 1.075] | -1 177 |
| 2023 | 264 | — | 0.855 | [0.620, 1.181] | -670 |
| **2024** | 264 | — | **0.655** | **[0.462, 0.893]** | **-1 476** |
| 2025 | 257 | — | 0.851 | [0.613, 1.180] | -450 |

**Observations :**
- **0/7 années profitables** sur EURUSD M15 (vs 4/8 sur XAU M15 v2).
- **2024 = pire année** (PF 0.655) — l'inverse du XAU M15 v2 (où 2024 était la meilleure année). **L'effet régime "bull EURUSD" n'existe pas** car EURUSD est range-bound (BH +2.46 % sur 7 ans).
- CI 95 % de l'année 2024 sur EURUSD : [0.46, 0.89] — exclut 1.0 par 11 points.

---

## C. Décomposition par side

| Side | n | WR | **PF** | CI 95 % | PnL USD |
|---|---:|---:|---:|---|---:|
| Long (+1) | 866 | 39.0 % | 0.839 | [0.695, 1.005] | -1 920 |
| Short (-1) | 939 | 41.7 % | 0.868 | [0.732, 1.031] | -3 104 |

**Lecture :** Quasi-symétrie long/short (PF 0.84 vs 0.87). **Pas de biais directionnel pathologique** comme en XAU M15. Les deux sides perdent de manière équivalente — c'est cohérent avec le fait que EURUSD ne drift pas sur 7 ans.

---

## D. Décomposition par régime

| Régime | n | WR | **PF** | PnL USD |
|---|---:|---:|---:|---:|
| Bull | 769 | (≈ 41 %) | 0.862 | (≈ -800) |
| Bear | 831 | (≈ 40 %) | 0.890 | (≈ -1 100) |
| Range | 205 | (≈ 36 %) | **0.685** | (≈ -1 000) |

**Lecture :** Distribution serrée 0.69-0.89. **Aucun régime n'est profitable.** Pire en range — exactement comme XAU M15 v2 (Bear PF 0.72, Range 1.30, Bull 1.30 sur XAU). Sur EURUSD, **le système n'a aucun edge contextuel.**

---

## E. Décomposition pré/post 2024

| Période | n | WR | **PF** | PnL USD |
|---|---:|---:|---:|---:|
| Pre 2024 (2019-2023) | 1 284 | — | 0.886 | -3 097 |
| **Post 2024 (2024-2025)** | 521 | — | **0.736** | **-1 926** |

**Lecture extrêmement importante :**
- **EURUSD se DÉGRADE en 2024-2025** (PF 0.886 → 0.736). C'est l'**inverse exact** du XAU M15 (qui s'améliorait 0.68 → 1.28).
- Cela **invalide** définitivement la thèse "le système a un edge structurel qui s'est révélé en 2024+". Sur un autre asset où il n'y a pas eu de bull break-out, le système se DÉGRADE post-2024.
- **Conclusion forte :** la performance 2024+ sur XAU n'est **pas** un edge. C'est de la **β-capture sur le bull XAU spécifique**, totalement absente d'EURUSD.

---

## F. Verdict Action 3 (200 mots)

**Le pipeline SMC ne fonctionne PAS sur EURUSD M15.**
- PF 0.854, **CI 95 % [0.748, 0.969]** — exclut 1.0.
- Return -50 % sur 7 ans alors que Buy & Hold ≈ 0 % (EURUSD range-bound). Le système **détruit du capital sans benchmark à blâmer**.
- 0/7 années profitables.
- Long PF 0.84, Short PF 0.87 — symétrie pathologique : les deux sides perdent.

**Le finding définitif** : sur un asset sans drift directionnel structurel (EURUSD a fait +2.46 % sur 7 ans), le pipeline a **un edge négatif**. Cela confirme que la "performance" du système sur XAU M15 2024-2026 est **0 % alpha, 100 % β-capture** sur le bull XAU. Hors de cette configuration, le système sous-performe le passive holding de manière violente.

**Implication structurelle :**
- Le scoring 8-composants n'a aucun pouvoir prédictif générique (Pearson -0.0086, sans surprise).
- Les seuils ATR-based / SMA200-based / ICT pattern detection sont **calibrés implicitement pour XAU** par construction.
- **Le pipeline ne se transfère pas — c'est un système XAU-spécifique non-généralisable.**

**Critère GO/NO-GO** : **❌ NO-GO** sans ambiguïté. Le CI 95 % exclut 1.0.

---

## ANNEXES

- `reports/comparatif_eurusd_m15_trades.csv` — 1 805 trades EURUSD ledger
- `reports/comparatif_eurusd_m15_summary.json` — métriques + décompositions
- `reports/comparatif_eurusd_m15_equity.csv` — equity curve
- `reports/comparatif_eurusd_m15_run.log` — console log
- `scripts/comparatif_eurusd_m15.py` — script reproductible (seed=42)

**Limite données :** EURUSD M15 disponible 2019-2025 (pas de 2026). Le sample comparable XAU est 2019-2025 ; les 2 premiers mois 2026 (XAU PnL +1791) ne sont **pas** comptés ici. Pour réplication parfaite vs XAU, refiltrer les ledgers XAU sur 2019-12-31 et comparer.
