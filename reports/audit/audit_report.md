# Smart Sentinel AI — Audit Backtest Complet XAU/USD

**Symbole** : XAUUSD  
**Timeframe** : M15  
**Fenêtre** : 2019-01-03 23:15:00 → 2025-12-31 21:45:00  
**Bars totales** : 106,618  

Objectif : identifier les blocages à la commercialisation et cartographier le comportement de la stratégie à tous les seuils.

---

## 1. Distribution des scores de confluence

Le détecteur a tourné avec `min_score=0` pour capturer **tous** les scores (pas seulement ceux au-dessus du seuil).

_Aucun bar scoré. Détecteur ne produit rien._

### Interprétation

Le ConfluenceDetector utilise 8 composants totalisant 100 points :
- BOS 15  |  FVG 15  |  OrderBlock 10  |  Régime 25  |  **News 20**  |  **Volume 10**  |  Momentum 3  |  RSI div 2

**En backtest, News et Volume sont structurellement à 0** (pas de flux historique news, pas de `volume_ma`). Score max théorique atteignable = **70/100**.

→ Le seuil de production `enter=75` est **mathématiquement impossible** à atteindre en backtest, et même en live il exige que News+Volume contribuent quasi-parfaitement à chaque bar — ce qui n'arrive jamais.

---

## 2. Sweep multi-seuils (la vraie mesure de l'edge)

| Config | enter/exit | Trades | Win% | Expectancy R | Total R | PF | Max DD R | Sharpe ann. | Trades/an |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| production_default | 75/55 | 1 | 0.0% | -0.042 | -0.04 | 0.00 | 0.00 | — | 0 |
| relaxed_55 | 55/40 | 136 | 36.0% | -0.056 | -7.55 | 0.80 | 9.25 | -0.43 | 31 |
| relaxed_50 | 50/35 | 421 | 40.1% | -0.053 | -22.37 | 0.83 | 32.58 | -0.71 | 96 |
| relaxed_45 | 45/30 | 750 | 42.9% | -0.026 | -19.84 | 0.91 | 33.74 | -0.48 | 170 |
| relaxed_40 | 40/25 | 1582 | 46.6% | +0.024 | +38.67 | 1.09 | 18.80 | +0.59 | 359 |
| relaxed_35 | 35/20 | 2363 | 46.8% | +0.037 | +87.38 | 1.13 | 21.05 | +1.06 | 537 |
| relaxed_30 | 30/15 | 2556 | 46.5% | +0.036 | +92.44 | 1.13 | 16.86 | +1.07 | 581 |

### Lecture

- **Profit factor** : < 1 = stratégie perd de l'argent. >= 1.3 minimum pour un produit payant. >= 1.5 pour rassurer un acheteur.
- **Expectancy R** : gain moyen (en R = risque initial) par trade. Doit être positif.
- **Sharpe annualisé** : > 1 correct, > 2 excellent. < 0.5 = inexploitable commercialement.

---

## 3. Analyse détaillée — configuration `relaxed_35` (enter=35, exit=20)

Cette config est sélectionnée car elle a le meilleur PF parmi celles ayant au moins 30 trades (échantillon statistiquement pertinent).

### 3.1 Résumé

| Métrique | Valeur |
|---|---:|
| Trades | 2363 |
| Wins | 1105 (46.8%) |
| Losses | 1253 (53.0%) |
| Breakeven | 5 |
| Expectancy / trade | **+0.037 R** |
| Total R | +87.38 R |
| Best / Worst | +2.00 R / -1.00 R |
| Profit factor | **1.13** |
| Payoff ratio (avgWin/avgLoss) | 1.28 |
| Max drawdown | 21.05 R |
| Max consec losses | 10 |
| Sharpe per trade / annualised | +0.05 / +1.06 |
| Sortino annualised | +1.81 |
| Calmar | 4.15 |

### 3.2 LONG vs SHORT

| Direction | Trades | Win% | Expectancy R | Total R | PF |
|---|---:|---:|---:|---:|---:|
| LONG | 1297 | 48.3% | +0.054 | +70.42 | 1.19 |
| SHORT | 1066 | 44.9% | +0.016 | +16.96 | 1.05 |

**P&L en $ (price-space, 1 unit de gold par trade)** :

- LONG  : $+399.84
- SHORT : $-1.37
- **Total** : $+398.47

_Note : en ajoutant un sizing fixe (ex: 0.1 lot), multiplier par 10. Ne comprend pas spread/commissions._

### 3.3 Performance par année

| Année | Trades | Win% | Expectancy R | Total R | PF |
|---|---:|---:|---:|---:|---:|
| 2019 | 275 | 47.3% | +0.028 | +7.65 | 1.09 |
| 2020 | 334 | 47.6% | +0.059 | +19.63 | 1.21 |
| 2021 | 246 | 45.5% | +0.068 | +16.78 | 1.22 |
| 2022 | 327 | 42.2% | +0.016 | +5.16 | 1.04 |
| 2023 | 220 | 39.5% | -0.032 | -7.10 | 0.91 |
| 2024 | 366 | 50.5% | +0.088 | +32.19 | 1.32 |
| 2025 | 595 | 49.4% | +0.022 | +13.07 | 1.11 |

### 3.4 Performance par tier

| Tier | Trades | Win% | Expectancy R | Total R | PF |
|---|---:|---:|---:|---:|---:|
| STANDARD | 35 | 37.1% | -0.009 | -0.33 | 0.97 |
| WEAK | 1526 | 46.9% | +0.038 | +57.72 | 1.14 |
| INVALID | 802 | 46.9% | +0.037 | +29.99 | 1.12 |

### 3.5 Performance par heure d'entrée (UTC)

| Heure | Trades | Win% | Expectancy R | Total R |
|---:|---:|---:|---:|---:|
| 00h | 97 | 47.4% | +0.017 | +1.61 |
| 01h | 112 | 46.4% | +0.050 | +5.60 |
| 02h | 97 | 47.4% | +0.068 | +6.60 |
| 03h | 72 | 45.8% | +0.066 | +4.74 |
| 04h | 60 | 43.3% | -0.054 | -3.22 |
| 05h | 76 | 42.1% | -0.021 | -1.57 |
| 06h | 101 | 52.5% | +0.159 | +16.01 |
| 07h | 96 | 54.2% | +0.104 | +10.03 |
| 08h | 74 | 40.5% | -0.129 | -9.58 |
| 09h | 79 | 50.6% | +0.161 | +12.70 |
| 10h | 88 | 48.9% | +0.156 | +13.75 |
| 11h | 94 | 50.0% | +0.267 | +25.14 |
| 12h | 88 | 45.5% | +0.082 | +7.25 |
| 13h | 140 | 42.1% | -0.032 | -4.53 |
| 14h | 117 | 42.7% | -0.076 | -8.89 |
| 15h | 137 | 55.5% | +0.077 | +10.62 |
| 16h | 128 | 53.1% | -0.007 | -0.87 |
| 17h | 135 | 50.4% | -0.006 | -0.81 |
| 18h | 118 | 45.8% | +0.002 | +0.23 |
| 19h | 138 | 41.3% | +0.006 | +0.80 |
| 20h | 119 | 42.9% | +0.043 | +5.07 |
| 21h | 31 | 32.3% | -0.133 | -4.12 |
| 22h | 67 | 34.3% | -0.170 | -11.40 |
| 23h | 99 | 49.5% | +0.123 | +12.22 |

### 3.6 Raisons de sortie

| Raison | Nombre | % |
|---|---:|---:|
| regime_shifted | 781 | 33.1% |
| time_expired | 631 | 26.7% |
| invalidated | 406 | 17.2% |
| score_decayed | 380 | 16.1% |
| target_reached | 157 | 6.6% |
| opposing_signal | 8 | 0.3% |

### 3.7 Cadence et machine à états

- Signaux/jour : 0.93
- Bars moyennes en position : 14.6 (~219 min)
- Taux de confirmation : 49.3% (2363/4793 arms confirmés, 2430 abandonnés)
- Signaux générés par le détecteur : 31,799
- Score max observé : 82.5

---

## 4. Problèmes identifiés & priorisation

### P1 — [BLOQUANT] Seuil de production inatteignable

**Constat** : la config par défaut `enter=75, exit=55` produit **0 trades sur 7 ans** parce que News (20pts) et Volume (10pts) sont toujours nuls en replay, plafonnant le score à ~70 max. En live, sans un flux news continu, le même plafond existe.

**Impact commercial** : un client abonné aujourd'hui ne reçoit aucun signal. 

**Solutions** :
1. Baisser `enter` à 50-55 pour une config production réaliste.
2. Re-normaliser les poids (retirer News/Volume des composants quand les données sont absentes, re-répartir sur les autres composants pour que le score conserve sa plage 0-100).
3. Rendre News/Volume **optionnels** : si absents, score recalé sur les 5 composants restants (BOS 15, FVG 15, OB 10, Régime 25, Momentum+RSI 5 = 70pts). Diviser par 70, multiplier par 100 pour normaliser.

### P2 — BOS_SIGNAL = trend-state, pas event-only

**Constat** : le détecteur utilise `BOS_SIGNAL` (le state propagé après une cassure, vrai 100% du temps après la 1ère cassure) comme gate de direction. Chaque bar peut donc potentiellement produire un signal, au lieu de seulement les bars où une cassure se produit. C'est documenté dans le code comme intentionnel (continuation signals) mais cela **dilue fortement l'edge** : les signaux "frais" (BOS_EVENT) scorent 85% du poids BOS, les continuations 50%.

**Solutions** :
1. Ajouter un composant "pullback" : entrer seulement quand le prix retouche la structure (OB ou FVG) après la cassure.
2. Augmenter la différence quality (continuation 0.3 au lieu de 0.5) pour pénaliser plus fort les signaux non-frais.
3. Ne trader que les `BOS_EVENT` frais + confirmation OB/FVG.

### P3 — News bypassée en backtest

**Constat** : le replay ne charge pas le calendrier économique (`economic_calendar_HIGH_IMPACT_2019_2025.csv` existe pourtant dans `/data/`). Tous les signaux pendant NFP / FOMC / CPI sont évalués comme s'il n'y avait pas d'événement.

**Solutions** :
1. Charger le CSV economic_calendar et construire un `NewsAssessment` synthétique par bar (blocking window ±30 min autour high-impact).
2. Comparer le edge "news-aware" vs "news-blind" — c'est un gros argument marketing.

### P4 — Coûts de transaction non modélisés

**Constat** : les PnL sont calculés `exit_price - entry_price` sans spread, commission ni slippage. Sur XAU/USD M15 avec ~50 trades/an, spread moyen 20-30 pips × 0.01 = ~$0.25/trade, slippage similaire. Coût réel ~$0.50-1.00 par round-trip.

**Solutions** :
1. Ajouter `commission_per_trade` et `spread_pips` en paramètres du harness. Soustraire du PnL.
2. Refaire le sweep avec frais réalistes — certaines configs positives deviendront négatives.

### P5 — Pas de segmentation par régime

**Constat** : le rapport actuel ne ventile pas la performance par régime (trending vs ranging, high-vol vs low-vol). Or la stratégie peut être profitable en trend et perdante en range, ou inversement.

**Solutions** :
1. Tagger chaque trade avec le régime+vol_regime à l'entrée, ventiler les métriques par (trend×vol).
2. Désactiver automatiquement les trades dans les régimes où l'edge est négatif.

### P6 — RR fixe 2:1 peu adaptatif

**Constat** : SL = 2×ATR, TP = 4×ATR (ratio 2:1) quelle que soit la situation. En fort trend, TP trop proche (rate les moves de 5-8 ATR). En range, TP trop loin (rarement atteint, sorties par time-expiry).

**Solutions** :
1. RR adaptatif : 1.5:1 en ranging, 3:1 en strong trend.
2. Trailing stop après +1R : lock in profit sans plafonner le run.

---

## 5. Verdict commercialisation

### ⚠️ **Marginal — ne pas commercialiser tel quel**

La stratégie est légèrement profitable hors frais, mais le profit factor trop bas ne laisse aucune marge pour spread + slippage. Il faut resserrer l'entry (pullback filter P2) avant de pouvoir prendre un client payant.

**Plan d'action recommandé avant déploiement :**
1. **[P1]** Re-normaliser le scoring pour tenir compte des composants absents — sinon aucun client ne recevra jamais rien.
2. **[P4]** Ajouter la modélisation des coûts et re-tester.
3. **[P2]** Ajouter un filtre pullback : n'entrer que sur retest de structure après cassure.
4. **[P3]** Intégrer le calendrier économique (déjà en `/data/`).
5. **[P5]** Ventiler par régime, désactiver les trades dans les régimes négatifs.
6. Re-lancer cet audit après chaque correction pour mesurer le gain.

---

_Rapport généré automatiquement. Fichiers associés :_
- `reports/audit/sweep_results.json` — données brutes du sweep
- `reports/audit/trades_combined.csv` — trades de la meilleure config
- `reports/audit/score_distribution.csv` — échantillons de scores
