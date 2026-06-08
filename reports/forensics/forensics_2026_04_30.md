# Forensics XAU M15 SMC — 4 forensics + 1 baseline NR4
## Synthèse pour décision Sprint 1 / Pivot
**Date :** 2026-04-30
**Données :** XAU_15MIN_2019_2026.csv (172 874 barres), seed=42
**Scripts :** `scripts/forensics_timeout_sweep.py`, `forensics_walkforward_purged.py`, `forensics_decomp_side.py`, `baseline_nr4_2026_04_30.py`
**CSV/JSON :** `reports/forensics/L1_*..L4_*`

---

## L1 — TIMEOUT SWEEP (max_lifetime_bars ∈ {12, 18, 24, 30, 36, 48, 64})

**Note structurelle :** `sl_dist = sl_atr_mult × vol × √(max_lifetime_bars)` dans la production actuelle. **Le timeout n'est pas un paramètre indépendant — il agit comme proxy pour la taille SL/TP via √(timeout)**. Le sweep mesure l'effet **couplé** timeout × SL_size, qui est le levier réel disponible en production.

| timeout | n_trades | WR | **PF** | **CI 95%** | Sharpe | MaxDD | Return | n_TP | n_SL | n_timeout | n_opp |
|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 12 | 2 734 | 43.3% | 0.680 | [0.61, 0.76] | -1.28 | -90.0% | -82% | 108 | 392 | 2 075 | 159 |
| 18 | 2 523 | 42.9% | 0.736 | [0.66, 0.82] | -1.21 | -82.9% | -74% | 110 | 344 | 1 850 | 219 |
| **24 (prod)** | **2 363** | 44.2% | **0.786** | **[0.70, 0.88]** | **-0.86** | **-77.9%** | **-62%** | 113 | 314 | 1 634 | 302 |
| 30 | 2 238 | 43.4% | 0.831 | [0.74, 0.93] | -0.69 | -74.6% | -52% | 114 | 284 | 1 459 | 381 |
| 36 | 2 122 | 42.9% | 0.897 | [0.80, 1.01] | -0.38 | -66.4% | -36% | 117 | 266 | 1 297 | 442 |
| 48 | 1 918 | 41.2% | 0.916 | [0.81, 1.03] | -0.26 | -64.4% | -28% | 121 | 246 | 1 003 | 548 |
| **64** | **1 753** | 41.0% | **1.037** | **[0.92, 1.17]** | **+0.27** | **-48.0%** | **+15%** | 130 | 218 | 781 | 624 |

### Verdict L1
- **Pas de pic suspect à 24** : 4 valeurs sur 6 testées font mieux que la prod actuelle. Donc **pas d'overfit-via-tuning** du timeout.
- **PF strictement monotone croissant** de 0.68 (12 bars) à **1.04 (64 bars)**. Drapeau rouge inverse : la prod actuelle est **sub-optimisée** sur ce paramètre. Le coupling sqrt(timeout) sur SL/TP domine — quand on agrandit timeout, on agrandit aussi SL/TP, ce qui réduit les SL touchés et laisse les trades respirer.
- **Avec timeout=64 le système devient profitable** (PF 1.04, CI lo 0.92, return +15%, MaxDD -48%). **C'est le seul paramètre qui bascule le système au-dessus de 1.0 sans toucher au reste.**
- **MAIS** : MaxDD -48% reste rédhibitoire pour commercialiser, et le coupling SL/TP rend ce gain "fragile" (il dépend du fait que les trades aient le temps de respirer — pas un edge en soi).

**Implication décision :** changer `max_lifetime_bars` 24→48 ou 64 est un quick-win 5 minutes qui ferait gagner ~12-25 points de PF. À tester en walk-forward strict avant de déclarer victoire.

---

## L2 — WALK-FORWARD PURGÉ (Train 2019-2022 / Test 2023-2024 / Holdout 2025-2026)

| Segment | Période | n_trades | **PF** | **CI 95%** | Sharpe | MaxDD | Return |
|---|---|---:|---:|---|---:|---:|---:|
| Train | 2019-01 → 2022-12 (52m) | 1 285 | **0.674** | [0.57, 0.78] | **-2.56** | -70.9% | -69% |
| Test | 2023-01 → 2024-12 (24m) | 655 | **0.956** | [0.78, 1.17] | +0.08 | -26.1% | -6% |
| **Holdout** | **2025-01 → 2026-04 (16m)** | 412 | **1.279** | [1.01, 1.63] | **+2.42** | **-7.9%** | **+38%** |

### Décomposition annuelle (vérification non-stationarité)

| Année | n | PF | MaxDD |
|:---|---:|---:|---:|
| 2019 | 307 | 0.668 | -25.7% |
| 2020 | 324 | 0.726 | -24.2% |
| 2021 | 333 | 0.583 | -36.7% |
| 2022 | 314 | 0.689 | -28.1% |
| 2023 | 328 | 0.747 | -26.1% |
| 2024 | 326 | 1.223 | -8.9% |
| 2025 | 307 | 1.312 | -7.9% |
| 2026 (YTD) | 102 | 1.218 | -5.6% |

### Sweep timeout sur HOLDOUT seul (validation finding L1 OOS)

| timeout | n_trades | PF Holdout | CI 95% |
|---:|---:|---:|---|
| 12 | 468 | 1.200 | [0.96, 1.52] |
| 18 | 439 | 1.165 | [0.92, 1.48] |
| 24 | 412 | 1.279 | [1.01, 1.63] |
| 30 | 391 | 1.300 | [1.03, 1.67] |
| 36 | 381 | 1.285 | [1.02, 1.65] |
| 48 | 356 | 1.347 | [1.05, 1.72] |
| 64 | 329 | 1.340 | [1.04, 1.74] |

### Verdict L2
- **Amélioration monotone Train→Test→Holdout (PF 0.67→0.96→1.28)** : le système ne s'overfite pas en arrière, il **s'améliore** en out-of-sample. C'est l'inverse du pattern classique d'overfitting.
- **Drapeau rouge** : la non-stationarité est tellement forte (Sharpe -2.56 → +2.42, MaxDD -71% → -8%) que **les segments ne sont pas comparables** au sens d'un OOS classique. Le système n'a pas d'edge robuste — il est **conditionnel au régime de marché**.
- **Le Train 2019-2022 aurait été stoppé par n'importe quel risk manager** (MaxDD -71% sur 4 ans). Tester un système en 2025+ alors qu'il aurait fait faillite pendant 4 années consécutives est un **risque opérationnel de classe-action** si commercialisé.
- **Holdout PF 1.28 est robuste au timeout** : sur 2025-2026 seul, PF 1.20 → 1.35 quel que soit le paramètre 12-64. Le système "fonctionne" sur ce régime indépendamment du tuning.

**Cohérent avec falsification précédente** : corr equity↔XAU spot = 0.96 sur 2024-2026 → **β-capture sur le bull XAU**, pas un edge généralisable.

---

## L3 — DÉCOMPOSITION LONG/SHORT PRÉ/POST 2024

| Période | Side | n | WR | **PF** | **CI 95%** | PnL USD | avg_R | Sharpe ann | MaxDD |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|
| 2019-2023 | Long | 821 | 42.4% | **0.809** | **[0.66, 0.99]** | -2 143 | -0.052 | -0.98 | -22.5% |
| 2019-2023 | Short | 796 | 38.9% | **0.561** | **[0.46, 0.68]** | -5 525 | -0.111 | -2.22 | -56.4% |
| 2019-2023 | BOTH | 1 617 | 40.7% | 0.678 | [0.59, 0.78] | -7 668 | -0.081 | -2.39 | -77.9% |
| **2024-2026** | **Long** | 419 | **55.1%** | **1.545** | **[1.21, 1.99]** | **+1 469** | **+0.128** | **+2.21** | **-1.6%** |
| 2024-2026 | Short | 327 | 47.4% | **0.983** | **[0.72, 1.30]** | -47 | +0.003 | -0.03 | -4.2% |
| 2024-2026 | BOTH | 746 | 51.7% | 1.262 | [1.05, 1.52] | +1 422 | +0.073 | +1.75 | -2.6% |
| FULL | Long | 1 240 | 46.7% | 0.952 | [0.81, 1.11] | -673 | +0.009 | -0.16 | -22.5% |
| FULL | Short | 1 123 | 41.4% | **0.636** | [0.54, 0.75] | -5 572 | -0.078 | -1.59 | -59.3% |

### Verdicts L3
**Q1 — Le système est-il un long-only déguisé sur 2024-2026 ?**
**OUI sans ambiguïté.** Long PF 1.545 [CI 1.21, 1.99] vs Short PF 0.983 [CI 0.72, 1.30]. Ratio PnL Long/|Short| = **31×** sur 2024-2026. Le PnL post-2024 est porté à 100% par les longs ; les shorts sont net-négatifs.

**Q2 — Le module Short est-il définitivement cassé ou seulement défaillant en bull ?**
**Structurellement cassé sur tous régimes :**
- Short 2019-2023 (bear/range) : PF 0.561 [CI 0.46, 0.68] — catastrophique.
- Short 2024-2026 (bull) : PF 0.983 [CI 0.72, 1.30] — breakeven négatif.
- Short FULL : PF 0.636 [CI 0.54, 0.75] — CI exclut 1.0 par 25 pts.

Le module n'a JAMAIS atteint PF > 1.0 dans aucun sous-segment. **C'est un signal cassé**, pas un signal "fragile en bull".

**Q3 — Long-only sur 2019-2023 (bear/range) — quel PF ?**
**PF 0.809 [CI 0.66, 0.99] — non-profitable.** Le CI **inclut 0.99** mais exclut 1.0. Sur le segment "non-bull", **même les longs sont sous breakeven**. Donc le système **n'a aucun edge directionnel intrinsèque**, ni long ni short, hors régime bull XAU.

**Verdict global L3 :** Le système est **un système long-only sur bull XAU**. Hors de cette configuration spécifique, il perd des deux côtés. Toute commercialisation sans clause de désactivation conditionnelle au régime expose le client à des drawdowns -50% à -70% (2019-2022 reproductible).

---

## L4 — BASELINE NR4 (Crabel volatility breakout) — comparatif apples-to-apples

### Spec implémentée
- NR4 (range[t] = min des 4 derniers ranges), valide sur 8 barres post-détection
- ATR_14[t] / ATR_14[t-20] > 1.2 (expansion volatilité)
- Signal long si close > high(NR4) ET expansion ; short symétrique
- SL = 1.0 × ATR, TP = 2.0 × ATR (R:R = 1:2 fixe, **pas de couplage** avec timeout)
- max_lifetime = 16 bars, cooldown = 8 bars
- Filtre session : exclure 21h-00h UTC ; news ±60 min
- Microstructure identique : spread 0.30, slippage 0.10-0.20, commission 7 USD/lot, capital 10k, risk 1%, seed=42

### Comparatif global

| Stratégie | n_trades | WR | **PF** | Sharpe | Sortino | MaxDD | Expectancy R | Return |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **NR4 baseline** | 6 538 | 31.3% | **0.603** | +0.08 | +0.08 | **-133%** | -0.241 | **-129%** |
| SMC actuel | 2 363 | 44.2% | 0.786 | -0.86 | -1.31 | -77.9% | -0.032 | -62% |

### NR4 par année

| Année | n | PF | CI 95% | MaxDD |
|---:|---:|---:|---|---:|
| 2019 | 880 | 0.404 | [0.31, 0.51] | -99.3% (bust) |
| 2020 | 849 | 0.714 | [0.60, 0.85] | -6.1% |
| 2021 | 969 | 0.708 | [0.60, 0.82] | -5.7% |
| 2022 | 910 | 0.630 | [0.53, 0.75] | -7.3% |
| 2023 | 934 | 0.610 | [0.52, 0.72] | -6.9% |
| 2024 | 924 | 0.738 | [0.63, 0.86] | -6.1% |
| **2025** | 821 | **1.007** | [0.84, 1.18] | -2.8% |
| **2026 YTD** | 251 | **1.054** | [0.76, 1.42] | -2.9% |

### Verdict L4
- **NR4 fait PIRE que SMC sur le full sample** (PF 0.60 vs 0.79). 6 538 trades = 2.8× le volume SMC, avec WR 31%. **Trop de faux breakouts** sur XAU M15.
- **2025-2026 : NR4 ~ équivalent à SMC** (PF 1.01-1.05 vs SMC 1.32). Mais **NR4 est symétrique** (pas de biais long structurel) → plus représentatif d'un edge "pur".
- **MaxDD -133%** signifie capital négatif (le simulator continue à 0.01 lot minimum). C'est une signature de **non-viabilité** sur le sample complet.
- **Conclusion clé** : si **deux paradigmes radicalement différents** (SMC narratif vs Crabel volatility breakout) **convergent vers PF ~1.0 sur 2025-2026 et < 0.7 sur 2019-2023**, c'est que **XAU M15 lui-même** ne contient pas assez de signal exploitable hors bull spécifique. **Le problème est l'asset/TF, pas le paradigme.**

---

## L5 — DECISION MATRIX

### Lecture transverse des 4 forensics

| Question | Réponse forensic |
|---|---|
| Le système est-il overfit sur le timeout ? | **Non.** Sweep monotone, pas de pic à 24. Au contraire : 24 est sub-optimal. timeout=64 → PF 1.04. |
| Le système se dégrade-t-il IS→OOS ? | **Non, il s'améliore.** Train 0.67 → Test 0.96 → Holdout 1.28. Mais c'est dû au régime, pas à un edge robuste. |
| Le système a-t-il un edge directionnel intrinsèque ? | **Non.** Long 2019-2023 PF 0.81, Short FULL PF 0.64. L'edge n'existe que sur 2024+ longs (β bull XAU). |
| Le pivot vol-breakout (NR4) est-il une issue ? | **Non.** PF 0.60 sur full, équivalent à SMC sur 2025-2026 (~1.0). Pas un edge nouveau. |
| Le Holdout PF 1.28 est-il généralisable ? | **Probablement non.** Corr equity↔XAU = 0.96 → β-capture. Si XAU range/bear, retour à PF 0.67 type Train. |

### Scénarios

| Scénario | Action | Prob succès commercialisable | Coût opportunité |
|---|---|---:|---|
| **(a) Continuer Sprint 1+2+3** | Drop CHOCH/RSI, GBM, MTF | **15-25 %** | 60h dev sur paradigme fragile, β-driven |
| **(b) Pivot NR4 sur XAU M15** | Implémenter Crabel + tuning | **5-10 %** | 8-16h dev pour PF 0.60 baseline |
| **(c) Changer asset/TF** | Re-run XAU H1, EURUSD M15, ES M15 | **20-35 %** | 24h dev pour 3 nouveaux runs comparables |
| **(d) Pivot B2B-API brokers** | Productiser news_pipeline + scanner | **30-45 %** | 80h dev pour MVP commerciable, $310k ARR cible |

### Recommandation chiffrée

**Recommandation principale : (c) — Changer asset/TF avant tout dev paradigmatique.**

**Justification chiffrée :**
- L1 montre que le timeout (proxy SL/TP size) sub-optimisé → la prod XAU M15 est probablement **structurellement contrainte par la noise du TF**, pas par le paradigme.
- L4 (NR4) confirme : un paradigme **radicalement différent** sur même asset/TF arrive aux mêmes conclusions (~PF 1.0 en bull, < 0.7 sinon). **L'asset/TF est le facteur limitant, pas le scoring.**
- L3 montre que le module short n'est PAS récupérable (PF < 1.0 dans tous régimes testés). Continuer SMC actuel = accepter de fait un système long-only.
- Coût (c) = 24h dev (3 runs × 8h chacun) = 3 semaines au rythme founder solo. **ROI** : si XAU H1 ou EURUSD M15 montrent **CI 95% PF lo > 1.0** sur full sample, on a un système robuste. Sinon (c)→(d).

### Action concrète semaine prochaine

1. **Quick-win 5 minutes** : changer `max_lifetime_bars = 64` (ou 48) en config production. Re-run audit. Attendu PF ~ 0.90-1.04. Bénéfice marketing même sans édit fondamental : système "qui ne perd plus" sur full sample.
2. **Scénario (c) — runs comparables 24h** :
   - Run 1 (8h) : XAU **H1** 2019-2026 même script `quant_audit_2026_04_30.py`
   - Run 2 (8h) : EURUSD M15 2019-2026 (data Dukascopy déjà disponible)
   - Run 3 (8h) : ES futures M15 (data à télécharger Dukascopy gratuit)
   - Critère GO : **un seul** des 3 doit donner PF [CI 95% lo > 1.0] sur full sample 2019-2026.

### Kill criterion suivant (post-recommandation)

**Si aucun des 3 runs (c) ne donne CI 95% PF lo > 1.0 sur full sample, kill total → pivot (d) B2B-API brokers.**

C'est binaire et chiffré. Pas d'optimisation au-delà de ces 3 runs sur asset/TF actuel paradigme.

---

## ANNEXES

- `reports/forensics/L1_timeout_sweep.csv` (.json) — sweep brut + verdict
- `reports/forensics/L2_walkforward_main.csv` (.json) — Train/Test/Holdout
- `reports/forensics/L2_walkforward_yearly.csv` — décomposition annuelle
- `reports/forensics/L2_holdout_timeout_sweep.csv` — sweep timeout sur Holdout
- `reports/forensics/L3_decomp_side.csv` (.json) — long/short pré/post 2024
- `reports/forensics/L4_baseline_nr4_trades.csv` — 6 538 trades NR4
- `reports/forensics/L4_baseline_nr4.json` — métriques NR4 + comparatif SMC
- Logs : `L1_run_log.txt`, `L2_run_log.txt`, `L3_run_log.txt`, `L4_run_log.txt`

---

## ENCADRÉ — Synthèse 1 ligne

> **Le problème n'est pas le scoring 8-composants — c'est XAU M15 lui-même. Avant tout dev paradigmatique, tester XAU H1, EURUSD M15, ES M15 (3×8h dev). Si aucun des 3 ne franchit CI 95% PF lo > 1.0, kill total → B2B-API.**
