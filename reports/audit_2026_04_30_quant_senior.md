# Audit Quant Senior - Smart Sentinel AI XAU/USD M15
## Backtest reproductible 2019-01-02 → 2026-04-29

**Date :** 2026-04-30
**Donnees :** XAU_15MIN_2019_2026.csv (172 874 barres reelles, 98.4% coverage Dukascopy)
**Calendrier :** 875 events high-impact USD/EUR
**Script :** `scripts/quant_audit_2026_04_30.py`
**Microstructure :** spread 0.30 USD, slippage 0.10-0.20 USD, commission 7 USD/lot RT, risk 1%/trade, capital 10 000 USD

---

## A. METRIQUES GLOBALES

| Metrique | Valeur | Cible | Verdict |
|---|---:|---:|:---:|
| **Profit Factor** | **0.786** | > 1.20 | **❌** |
| Win rate | 44.18% | > 50% | ❌ |
| Sharpe (mensuel annualise) | -0.860 | > 1.0 | ❌ |
| Sortino | -1.309 | > 1.5 | ❌ |
| Calmar | -0.801 | > 0.5 | ❌ |
| Max Drawdown | **-77.92%** (-7 810 USD) | < 25% | ❌❌ |
| Duree DD max | 2 662 jours | < 180 j | ❌❌ |
| Avg R:R realise | 0.993 | > 1.5 | ❌ |
| Expectancy | **-0.032 R / trade** | > +0.1 R | ❌ |
| Pearson(score, R) | **-0.0010** | > 0.10 | ❌❌ |
| Capital final | 3 754 USD | > 10 000 | ❌ |
| Return total | **-62.46%** | > +50% | ❌ |
| Trades totaux | 2 363 | - | - |
| Buy & Hold XAU (memes dates) | **+255.60%** | - | - |

**Lecture brute :** le systeme detruit 62% du capital sur 7 ans alors que le simple **buy & hold XAU** rapportait +256%. Underperformance de **~318 points** vs benchmark passif.

---

## B. DECOMPOSITION PAR ANNEE — pattern de retournement 2024

| Annee | n trades | WR | PF | PnL USD |
|:---|---:|---:|---:|---:|
| 2019 | 310 | 40.0% | 0.672 | **-2 468** |
| 2020 | 325 | 42.2% | 0.743 | -1 442 |
| 2021 | 336 | 35.7% | 0.599 | -1 995 |
| 2022 | 314 | 44.6% | 0.673 | -1 013 |
| 2023 | 332 | 41.3% | 0.710 | -748 |
| **2024** | 328 | 50.9% | **1.199** | **+370** |
| **2025** | 314 | 51.9% | **1.322** | **+733** |
| **2026 YTD** | 104 | 53.8% | **1.248** | **+319** |

**Observation cruciale :** les 3 dernieres annees sont a/au-dessus de la cible PF 1.20. Hypotheses :
1. **Regime change reel** : XAU passe de 2 000 a 3 300 USD/oz en 2024-25 → bull tres directionnel ou les longs gagnent.
2. **Biais long systemique** : voir section side ci-dessous (longs PF 0.95 vs shorts PF 0.64).
3. **Sample size insuffisant** : 746 trades sur 2024-2026 → t-stat sur PF differential ~1.8, **non significatif a 95%**.

**Conclusion section :** ne pas conclure que "le systeme est devenu profitable" — c'est probablement un biais long sur un super-bull XAU.

---

## C. DECOMPOSITION PAR REGIME

| Regime | n | WR | PF | PnL |
|:---|---:|---:|---:|---:|
| Bear (close < SMA200, slope<0) | 917 | 42.6% | **0.623** | **-4 743** |
| Bull (close > SMA200, slope>0) | 1 150 | 45.3% | 0.909 | -1 200 |
| Range | 296 | 44.6% | 0.914 | -302 |

**Le systeme saigne en bear** — il continue d'envoyer des shorts et des longs contre-tendance qui se font tuer. 76% du PnL negatif total vient du regime bear.

## DECOMPOSITION PAR SIDE — asymetrie pathologique

| Side | n | WR | PF | PnL |
|:---|---:|---:|---:|---:|
| Long (+1) | 1 240 | 46.7% | 0.952 | -674 |
| **Short (-1)** | 1 123 | 41.4% | **0.636** | **-5 572** |

Les shorts representent **89% des pertes nettes**. Soit la detection bearish est cassee, soit XAU a un drift haussier structurel sur la fenetre (cf BH +256%) qui penalise mecaniquement tout short non-finement chronometre.

## SESSION (UTC)

| Session | n | WR | PF | PnL |
|:---|---:|---:|---:|---:|
| Asia (00-07h) | 678 | 42.5% | 0.721 | -1 945 |
| London (07-13h) | 640 | 45.3% | 0.788 | -1 800 |
| NY (13-21h) | 921 | 45.5% | 0.835 | -2 043 |
| Off (21-24h) | 124 | 37.9% | 0.662 | -457 |

**Asie est le pire**, NY le moins mauvais — mais aucune session n'est rentable. Couper Asie + Off economiserait 2 402 USD (38% des pertes).

## JOUR DE SEMAINE

| DOW | n | WR | PF | PnL |
|:---|---:|---:|---:|---:|
| Mercredi | 508 | 41.9% | **0.689** | -2 106 |
| Mardi | 496 | 42.1% | 0.786 | -1 316 |
| Jeudi | 437 | 45.1% | 0.765 | -1 316 |
| Lundi | 431 | 48.0% | 0.889 | -509 |
| Vendredi | 482 | 44.6% | 0.842 | -949 |

Mercredi est le piege — souvent FOMC, CPI, NFP releases. Le blackout ±15min est trop court (cf section news).

---

## D. ANALYSE STATISTIQUE DES DEFAUTS

### D.1 Score 0-100 : c'est en realite un score 3-niveaux

| Score | n trades | WR | avg_R | PnL |
|:---|---:|---:|---:|---:|
| 75.0 | 2 048 (86.6%) | 44.3% | -0.031 | -5 398 |
| 87.5 | 309 (13.1%) | 42.7% | -0.046 | -917 |
| 100.0 | 6 (0.3%) | 66.7% | +0.272 | +70 |

**Le scoring 8 composants × 12.5 pts ne genere QUE 3 valeurs distinctes** au-dessus du seuil 65. Les "deciles score" demandes sont impossibles a calculer — `qcut` echoue. Le score n'est pas un score, c'est un compteur Booleen tres grossier.

→ **Pearson(score, R) = -0.0010** : zero pouvoir predictif, confirme l'eval anterieure (-0.023).

### D.2 Edge par composant (n_on / n_off / edge en R realise)

| Composant | n_on | n_off | wr_on | wr_off | edge_R |
|:---|---:|---:|---:|---:|---:|
| **fvg** | 1 901 | 462 | 44.6% | 42.6% | **+0.027** ✓ |
| **retest** | 2 186 | 177 | 44.5% | 40.7% | **+0.022** ✓ |
| bos | 2 270 | 93 | 44.1% | 45.2% | +0.015 |
| regime | 1 671 | 692 | 45.2% | 41.6% | +0.012 |
| ob | 2 308 | 55 | 44.2% | 43.6% | -0.006 |
| **choch** | 995 | 1 368 | 42.7% | 45.2% | **-0.018** ❌ |
| **rsi_div** | 807 | 1 556 | 42.0% | 45.3% | **-0.020** ❌ |
| news_ok | 2 361 | 2 | 44.2% | 50.0% | -0.063 (n<2) |

**Verdict composants :**
- **FVG + retest** = les seuls a edge positif net. Ce sont les piliers a conserver.
- **CHOCH et RSI divergence sont anti-predictifs** : presents = ~3% de WR en moins. Ils auraient un edge **inverse** s'ils etaient retournes (a verifier en walk-forward).
- **OB ICT** : edge nul, sa formulation actuelle (engulfing + BOS dans 5 barres) est une approximation tres laxe de l'ICT pur.
- **News blackout est non-fonctionnel** : seulement 2 trades en blackout sur 7 ans. Fenetre ±15min trop etroite pour des releases NFP/CPI dont l'effet dure 1-3h.

### D.3 Distribution des sorties — l'aiguillage est casse

| Exit reason | n | WR | avg_R | PnL |
|:---|---:|---:|---:|---:|
| **timeout** | 1 634 (69.1%) | 54.8% | +0.092 | +4 685 |
| **sl** | 314 (13.3%) | 0% | -1.024 | **-13 579** |
| **tp** | 113 (4.8%) | 100% | +1.646 | +6 921 |
| **opposite** | 302 (12.8%) | 11.6% | -0.302 | **-4 273** |

**Trois pathologies critiques :**

1. **Le systeme atteint son TP 4.8% du temps mais son SL 13.3%** → ratio TP/SL = 36%. Avec R:R 1.65, il faut au minimum 38% pour breakeven hors couts → on est **a la frontiere mathematique** mais les commissions+slippage tuent l'edge.
2. **Les timeouts** (69% des sorties) sont leger-positifs (+0.09R) — c'est la, et seulement la, qu'on capture le faible edge des composants FVG/retest. Le SL/TP est trop large pour la duree de vie 24 barres.
3. **L'exit "opposite" detruit -0.30R en moyenne** : le state machine coupe des positions qui auraient pu mature. **Bug operationnel** : "score adverse >= 65" se declenche frequemment dans des regimes choppy et inverse les positions a perte.

### D.4 Combos de composants (FVG × Retest × Regime, top 3)

| (fvg,retest,regime) | n | WR | PF | PnL |
|:---|---:|---:|---:|---:|
| (1,1,1) | 1 094 (46%) | 46.4% | 0.816 | -2 544 |
| (1,1,0) | 633 | 42.5% | 0.783 | -1 674 |
| (1,0,1) | 161 | 42.9% | 0.755 | -476 |

Meme avec les **3 meilleurs filtres allumes**, PF reste 0.82. Ajouter du filtrage ne suffit pas — il faut un classifieur reel, pas une moyenne ponderee.

---

## E. RECOMMANDATIONS HIERARCHISEES (ROI / effort)

### P0 — Refondre le scoring (le KPI rouge n°1)

**1. Remplacer `confluence_score` par un GBM binaire (LightGBM)** trained sur (features composants × contexte) → label `1{R_5bars > 0}`.
- **Effort :** 6-8h dev + 2h walk-forward
- **Gain attendu :** +20-40% PF (de 0.79 a ~1.0-1.10) — base : eval_04 LGBM bat naïf -31% RMSE en walk-forward; correlation score↔R passerait de -0.001 a typiquement 0.06-0.12 pour ces features.
- **Risque :** overfitting si on entraine sur tout 2019-2026. **Mitigation :** purged k-fold avec embargo 5 jours, train pre-2023 / test 2023-2026.
- **Plan :** voir `scripts/colab_lgbm_vol_poc.py` pour template, adapter pour cible binaire R>0.

### P0 — Inverser ou supprimer CHOCH et RSI divergence

- **Edge_R negatif** (-0.018, -0.020) prouve sur 2 363 trades : ces composants **degradent** la decision.
- **Effort :** 30 min (changement de signe ou retrait dans `compute_confluence_score`)
- **Gain attendu :** +5-10% PF par retrait, **+15-25% si inverses** (sous reserve verification walk-forward sur OOS)
- **Risque :** retournement OOS — verifier sur EURUSD/GBPUSD avant deploiement.

### P0 — Fixer le bug exit "opposite"

- 302 trades sortis -0.30R en moyenne, soit -4 273 USD (24% des pertes nettes).
- **Effort :** 1h
- **Action :** elever le seuil opposite a 80 (vs 65 actuel) OU exiger 2 barres consecutives, OU ne JAMAIS sortir sur opposite — laisser SL/TP/timeout faire le job.
- **Gain attendu :** +500 a +1 500 USD sur 7 ans (recapture partielle des +0.092R timeout sur ces trades).

### P1 — Money management dynamique : Kelly fractionne

- **Effort :** 4-6h
- **Hypothese :** sizer par lot = `0.25 × kelly(p_win, R:R)` ou p_win vient du GBM P0.
- **Gain attendu :** -30% max DD a meme PF, mais **necessite GBM operationnel** (P0 prerequisite).
- **Risque :** fluctuation lot size si p_win volatil → cap a 1.5R/trade max.

### P1 — Sortie adaptive : trailing stop ATR + partial TP

- 1 634 timeouts (69%) avec avg_R +0.09 → on **laisse de l'argent sur la table**.
- **Effort :** 4h
- **Action :** TP1 a 1.0R (50% position), trailing stop 1.0×ATR sur le reste.
- **Gain attendu :** +0.05 a +0.10R par trade ≈ +1 000 a +2 000 USD sur 7 ans.

### P1 — Multi-timeframe confirmation (M15 × H1 × H4)

- **Effort :** 8h
- **Action :** entrer M15 seulement si bias H1 et H4 alignes (close > EMA50 H4 = OK long, etc.).
- **Gain attendu :** -50% trades, +30-50% PF par filtrage qualite (eval comparable strategies).
- **Risque :** lookahead si pas resample correct — utiliser only `closed bars` H1/H4.

### P2 — Rallonger le blackout news a ±60 minutes

- 875 events / 2 trades touches = blackout casse.
- **Effort :** 5 min
- **Gain :** -3-5% trades sur tout le sample, qualite + (eviter slippage extreme).

### P2 — Couper sessions Asie + Off

- Economie sur 7 ans : -802 trades, +2 402 USD.
- **Effort :** 5 min (filtre `hour in [7..21]`).
- **Gain :** PF passe de 0.79 a ~0.90 sans rien changer d'autre.

### P3 — Architecture : ajouter un layer "position sizing par qualite"

- Trader > signaleur : couplet `signal × confidence` au lieu de `signal binaire`.
- Confidence = probabilite GBM × accord MTF × distance au support/resistance recent.
- **Effort :** 12-16h (toolchain complete)
- **Gain :** +0.2 a +0.4 sur PF si toute la chaine est cohérente.
- **Risque :** complexite operationnelle, latence > cible 50ms (mais sur M15 c'est OK).

---

## F. ROADMAP CHIFFREE — atteindre PF > 1.20

### Sprint 1 (semaine 1, ~12h dev)
- [ ] Retirer CHOCH + RSI div du score (P0, 30 min)
- [ ] Bug exit opposite seuil 80 (P0, 1h)
- [ ] Filtre session 07-21h UTC (P2, 5 min)
- [ ] Blackout news ±60min (P2, 5 min)
- [ ] Re-run audit → PF estime **0.95-1.05**

### Sprint 2 (semaines 2-3, ~16h dev)
- [ ] LGBM binaire (FVG, retest, BOS, regime, ATR_pct, hour, dow, score_BH200) (P0, 8h)
- [ ] Walk-forward purged (2019-22 train / 2023-26 OOS) (3h)
- [ ] Re-run audit → PF estime **1.10-1.25**

### Sprint 3 (semaine 4, ~12h dev)
- [ ] Trailing TP partial (P1, 4h)
- [ ] MTF H1 + H4 confirmation (P1, 8h)
- [ ] Re-run audit → PF estime **1.25-1.45**

**Si Sprint 3 atteint PF >= 1.20 sur OOS** → GO commercial pour tier ANALYST/STRATEGIST.
**Sinon (Sprint 3 PF < 1.20)** → pivot necessaire (TF / asset / approche).

---

## G. VERDICT (200 mots max)

**Le systeme actuel n'est PAS commercialisable.** Profit factor 0.79 sur 7 ans avec un drawdown de 78% et un benchmark passif a +256% : facturer 29-1990 USD/mois pour ca exposerait a un risque de class-action — c'est economiquement et legalement intenable.

**Trois bons points** : (1) FVG et retest ont un edge positif faible mais reel (+0.022 a +0.027R) ; (2) le code execution est realiste (spread, slippage, commission corrects) ; (3) la performance 2024-26 (PF 1.20-1.32) suggere que la mecanique fonctionne quand le regime coopere — donc pas un cas perdu.

**La pathologie centrale** est le scoring : 8 composants en moyenne ponderee → 3 valeurs effectives → Pearson 0 vs PnL futur. **Sans GBM/classifieur**, aucun reglage de seuils ne peut sauver ce score. C'est un probleme d'**architecture de decision**, pas de tuning.

**Probabilite de PF > 1.20 OOS apres 4-6 semaines R&D structuree** : 60-70% sur XAU M15. Probabilite que XAU M15 soit fondamentalement non profitable : faible (les 3 dernieres annees prouvent qu'un edge existe). **Si tu devais ne garder QUE 3 features** : `FVG`, `retest`, `regime SMA200`. Drop le reste, mets un GBM dessus, walk-forward, et reviens.

---

## ANNEXES

- **Trade ledger complet :** `reports/audit_2026_04_30_trades.csv` (2 363 lignes)
- **Equity curve :** `reports/audit_2026_04_30_equity.csv`
- **Summary JSON :** `reports/audit_2026_04_30_summary.json`
- **Script :** `scripts/quant_audit_2026_04_30.py` (~530 LOC, reproductible avec `random_seed=42`)
- **Console log :** `reports/audit_2026_04_30_console.log`
