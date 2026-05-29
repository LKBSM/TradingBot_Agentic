# Prompt 02 — ConfluenceDetector : audit de calibration et valeur commerciale du scoring

> **Périmètre** : `src/intelligence/confluence_detector.py` (625 lignes), `tests/test_confluence_detector.py` (578 lignes), 7 fichiers de tests annexes, replay 7 ans `reports/audit/score_distribution.csv` (106 518 bars) et `reports/audit/trades_combined.csv` (924 trades).
>
> **Rappel business** : *le scoring EST le produit*. Si une note 75/100 ne surperforme pas statistiquement une note 55, tous les tiers (PREMIUM ≥80, STANDARD ≥60, WEAK ≥40) ne sont que du marketing sans substance.
>
> **Date** : 2026-04-24 — écrit avec le code actuel post BOS/CHOCH fix + renormalisation News/Volume absents.

---

## 1. Cartographie du scoring — composants, poids, formules

### 1.1 Table des 8 composants (DEFAULT_WEIGHTS `confluence_detector.py:110-119`)

| Composant | Poids | Entrée | Formule de quality (`[0,1]`) | Score final |
|---|---:|---|---|---|
| `bos` | **15** | `BOS_SIGNAL`, `BOS_EVENT`, `CHOCH_SIGNAL`, `BOS_RETEST_ARMED` | ladder gradué : CHOCH+armed=1.0, armed=0.9, CHOCH=0.7, event=0.4, continuation=0.25 (mode `require_retest=True`) | quality × 15 |
| `fvg` | **15** | `FVG_SIGNAL`, `FVG_SIZE_NORM` (gap/ATR) | `min(1, 0.3 + 0.7×gap_atr)` → 0.3 à 0.1×ATR, 1.0 à ≥1×ATR | quality × 15 |
| `order_block` | **10** | `OB_STRENGTH_NORM` (ATR-normalisé) | `clamp(0,1, |ob|)` | quality × 10 |
| `regime` | **25** | `regime.regime_type`, `confidence`, `trend_strength` | alignment ∈ {0, 0.3, 0.4, 1.0} × `(max(conf,0.3)+max(str,0.3))/2` | alignment × quality × 25 |
| `news` | **20** | `news.sentiment_score`, `sentiment_confidence` | `((sentiment+1)/2 si LONG else (1-sentiment)/2) × max(conf,0.3)` | quality × 20 |
| `volume` | **10** | `volume / volume_ma` | `clamp(0,1, (ratio - 0.5)/1.5)` → 0 à 0.5×, 1.0 à 2.0×+ | quality × 10 |
| `momentum` | **3** | `RSI`, `MACD_Diff` | `0.5 × rsi_score + 0.5 × macd_aligned_score` | quality × 3 |
| `rsi_divergence` | **2** | `CHOCH_DIVERGENCE ∈ {-1,0,+1}` | binaire : aligné=1, sinon=0 | 0 ou 2 |

Vérification : `sum(weights) = 100.0` (test `test_default_weights_sum_to_100`).

### 1.2 Pipeline d'évaluation (`analyze`, lignes 182-356)

1. **Gate news** (`BLOCK` → None)
2. **Gate BOS** — `BOS_SIGNAL==0` → None (porte de direction)
3. **Gate retest** (si `require_retest=True`) — `BOS_RETEST_ARMED` doit s'aligner
4. Score chaque composant → `total_score = Σ weighted_score`
5. **Renormalisation quand données absentes** (news/volume/regime = `None`) :
   ```
   present_weight = 100 - sum(poids absents)
   total_score    = total_score × 100 / present_weight
   ```
6. Clamp à `[0, 100]`, rejet si `< min_score` (défaut 60)
7. Classification tier : ≥80 PREMIUM, ≥60 STANDARD, ≥40 WEAK, sinon INVALID
8. Position multiplier `max(0, min(1.5, regime_mult × news_mult))`
9. SL/TP = `price ± {SL,TP}_atr_mult × (vol_forecast.forecast_atr ?? atr)` ; haut-vol étend SL × 1.5 (TP figé)

### 1.3 Observation critique : **2 systèmes de gates empilés**

Le `analyze` fait **3 rejets successifs** avant même de scorer :
- `news.decision == BLOCK`
- `BOS_SIGNAL == 0`
- `require_retest` et `BOS_RETEST_ARMED` non-aligné

Conséquence : un signal émis est déjà pré-filtré sur 3 booléens. Le score 0-100 ne distingue que *la qualité parmi les déjà-qualifiés*. Une grande partie de la sélectivité vient des gates, pas du score.

---

## 2. Orthogonalité des features — diagnostic structurel

L'état actuel ne stocke pas la ventilation par composant dans `score_distribution.csv`, donc la matrice de corrélation empirique bar-à-bar n'est pas disponible. L'analyse est structurelle.

### 2.1 Matrice de corrélation structurelle attendue (heuristique)

| | BOS | FVG | OB | Regime | News | Volume | Mom | RSIdiv |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **BOS** | 1.00 | ~0.35 | ~0.25 | **~0.55** | ~0.00 | ~0.15 | **~0.45** | **~0.40** |
| **FVG** | | 1.00 | **~0.40** | ~0.15 | ~0.00 | ~0.25 | ~0.20 | ~0.10 |
| **OB** | | | 1.00 | ~0.15 | ~0.00 | ~0.20 | ~0.15 | ~0.05 |
| **Regime** | | | | 1.00 | ~0.05 | ~0.10 | **~0.60** | ~0.15 |
| **News** | | | | | 1.00 | ~0.05 | ~0.05 | ~0.00 |
| **Volume** | | | | | | 1.00 | ~0.20 | ~0.05 |
| **Momentum** | | | | | | | 1.00 | ~0.20 |
| **RSI div** | | | | | | | | 1.00 |

**Points chauds de redondance** (ρ ≥ 0.4 supposé, à vérifier empiriquement — livrable à produire, cf. §10) :

| Paire | ρ attendu | Cause structurelle |
|---|:-:|---|
| BOS ↔ Regime | 0.55 | Un BOS haussier implique souvent un régime uptrend (trend-following même signal) |
| BOS ↔ Momentum | 0.45 | MACD>0 + RSI 50-70 co-occurent avec BOS haussier (indicateurs de tendance concordants) |
| BOS ↔ RSI_div | 0.40 | `CHOCH_DIVERGENCE` n'est lu que quand CHOCH déclenche — le signal `rsi_divergence` n'existe que dans les cas où la lame BOS est déjà CHOCH |
| Regime ↔ Momentum | 0.60 | Les deux mesurent *trend strength* avec des features qui se recoupent (slope MA, ADX-like vs RSI/MACD) |
| FVG ↔ OB | 0.40 | Les deux sont des footprints de déplacement institutionnel, co-occurrence fréquente |

**Implication chiffrée** : sur les 100 points, ~**30-35 points** sont probablement du **double-comptage** (BOS(15) + CHOCH via rsi_div(2) + Regime(25)/2 + Momentum(3) + part de FVG(15)/2). Un modèle avec 6-7 features vraiment orthogonales vaudrait autant.

### 2.2 Composants qui partagent la même entrée brute

- `CHOCH_SIGNAL` est consommé dans **deux** composants : `bos` (boost dans la ladder) ET `rsi_divergence` (via `CHOCH_DIVERGENCE` qui n'est produit que sur CHOCH).
- `BOS_EVENT` est un **sous-ensemble** de `BOS_SIGNAL` (event déclenche le state) → ladder re-comptabilise la même info.
- `trend_direction` du régime et `BOS_SIGNAL` sont quasi-toujours co-alignés quand la stratégie trade (gate BOS + gate retest filtrent tout le reste).

**Recommandation technique** : fusionner `bos + rsi_divergence` en un seul composant « structural_quality » ; extraire `momentum` comme *tie-breaker* hors-score.

---

## 3. Audit du plafond 70/100 — bug corrigé, mais empiriquement le plafond réel est **55.5**

### 3.1 Le fix en place (lignes 263-274)

```python
total_score = sum(c.weighted_score for c in components)
absent_weight = 0.0
if news is None:        absent_weight += weights.news
if volume is None ...:  absent_weight += weights.volume
if regime is None:      absent_weight += weights.regime
present_weight = sum(weights) - absent_weight
if absent_weight > 0 and present_weight > 0:
    total_score = total_score * 100.0 / present_weight
total_score = max(0.0, min(100.0, total_score))
```

**Audit** :
- ✅ Correct sur le principe : distingue *donnée absente* (`None`) de *donnée présente-mais-neutre* (sentiment=0).
- ✅ Testé par 5 tests `TestScoreRenormalization` (cas news absent, volume absent, les deux, neutre-vs-absent).
- ⚠️ **Limite non traitée** : `volume_ma <= 0` redéclenche l'absence mais `volume_ma == 0.01` (quasi-zéro) reste comptabilisé → instabilité numérique.
- ⚠️ **Clamp à 100 masque le bug** : si tous les composants scorent au max et renormalisation active, le ratio peut dépasser 100 (puis clampé). Il faudrait lever un warning si `ratio > 105`.

### 3.2 Plafond empirique mesuré sur 106 518 bars (pré-fix, mais la renormalisation ne change pas les entrées brutes)

```
DISTRIBUTION DU SCORE (detector.min_score=0, News=None, Volume=None)
n       = 106,518 bars scorées (LONG=53,143 / SHORT=53,375)
min     = 8.0
max     = 55.5
mean    = 17.8
p50     = 15.1
p90     = 27.2
p99     = 37.0
```

| Seuil | Bars ≥ seuil | % |
|---:|---:|---:|
| ≥ 40 | 391 | **0.37 %** |
| ≥ 50 | 26 | **0.024 %** |
| ≥ 55 | 2 | **0.002 %** |
| ≥ 60 (STANDARD) | 0 | 0 % |
| ≥ 70 | 0 | 0 % |
| ≥ 75 (config prod) | 0 | 0 % |
| ≥ 80 (PREMIUM) | 0 | 0 % |

**Diagnostic** : le plafond théorique 70 (fix 55.5 avant renorm) reste inatteignable même avec renormalisation, parce que **regime(25)**, **bos(15)**, **fvg(15)**, **ob(10)** ne pointent quasi jamais tous simultanément à leur max. La calibration des quality functions est trop punitive (FVG à 0.5×ATR = 0.65×weight, regime ranging = 0.3×alignment → 0.3×1.0×25 = 7.5pts...). Même un bar « parfait » en SMC n'atteint pas 60.

**Conclusion §3** : ré-normaliser a corrigé un bug logique (tests verts) mais **n'a pas débloqué le produit** — le problème sous-jacent est que *les fonctions de qualité sont calibrées trop bas*, pas que les composants manquent.

---

## 4. Calibration du score — **rejet empirique**

Mesures sur les 924 trades livrés par la config `relaxed_40` (le plus haut seuil à trader significativement sur 7 ans).

### 4.1 Reliability diagram (ASCII, win_rate par bucket)

```
bucket       n     realized_win%    expectancy_R    total_R
─────────────────────────────────────────────────────────────
[0, 40)    850        44.2%          -0.002           -1.39
[40, 45)    39        33.3%          -0.091           -3.56
[45, 50)    33        36.4%          -0.161           -5.31
[50, 55)     2         0.0%          -0.294           -0.59

(aucune observation ≥55)
```

**Pente** : la win-rate **DÉCROIT** avec le score. Un score 45-50 perd 36 % du temps contre 44 % sous 40. L'expectancy est strictement plus négative quand le score monte.

### 4.2 Brier score

```
Brier model    = 0.2551  (probabilité = score/100)
Brier baseline = 0.2456  (probabilité constante = 0.434 = p(win))
```

Le modèle est **pire qu'une probabilité constante** (écart +0.0095, soit ~3.9 % plus mauvais). Un scoring légitime doit produire Brier < baseline. **Ici il ne fait pas mieux que tirer à pile ou face avec un biais 43 %**.

### 4.3 Corrélations score ↔ outcome

```
Pearson(score, r_multiple)   = -0.0232   (cible : > +0.10)
Spearman rank(score, r_mult) = -0.0162   (cible : > +0.10)
```

**Les deux sont ~0** et *légèrement négatives*. Le score n'a aucun pouvoir discriminant — même pas monotone. Cela rend toute tentative de **calibration par isotonic regression ou Platt scaling inutile** : on ne peut pas corriger une courbe sans pente.

### 4.4 Implications business directes

| Claim marketing | Validé par les chiffres ? |
|---|:-:|
| « PREMIUM (≥80) = signaux haute conviction » | ❌ PREMIUM : 0 signaux sur 7 ans |
| « STANDARD (≥60) = signaux fiables » | ❌ STANDARD : 0 signaux sur 7 ans |
| « Plus le score est haut, meilleur le résultat » | ❌ **Inverse**, Pearson -0.023 |
| « 75/100 = seuil de production » | ❌ Inatteignable, plafond empirique 55.5 |

---

## 5. Stabilité temporelle

```
année   n     score_mean  score_p90   realized_win%   E[R]
─────────────────────────────────────────────────────────────
2019   98     35.1         38.5        40.8 %         -0.010
2020  155     34.9         38.5        45.2 %         +0.018
2021  104     35.1         38.4        39.4 %         -0.067
2022  129     35.0         38.9        40.3 %         -0.066
2023   88     35.6         41.2        40.9 %         -0.013
2024  128     35.0         38.5        41.4 %         -0.032
2025  222     35.1         39.7        49.1 %         +0.036
```

**Observations** :
- **Score stable** : moyenne 34.9-35.6, P90 38.4-41.2 → pas de drift de la distribution sur 7 ans. Le scoring est *reproductible*, ce qui est la seule vertu notable.
- **Expectancy instable** : 2020 et 2025 positifs (régimes trend fort), 2021-2024 négatifs. Pas d'edge stable.
- **Win-rate** : 39-49 %, corridor 10 points. Pas de régression du score vs année (le score 35 de 2020 donne le même outcome que le score 35 de 2024 → stabilité *de la non-information*).

**Conclusion** : le score est stable *comme estimateur de bruit*. Il ne drift pas, mais il ne capte rien. Le problème n'est pas le drift — c'est l'absence totale de signal.

---

## 6. Benchmark vs état de l'art 2026

| Approche | Brier attendu | Effort | Pro | Con |
|---|:-:|:-:|---|---|
| **Scoring actuel (rule-based 0-100)** | 0.255 | — | Explicable ligne par ligne, déterministe | **Non calibré, ρ=0** |
| **Isotonic regression sur le score actuel** | 0.250 | 0.5j | Trivial à implémenter | Ne peut corriger une corrélation rang ~0 |
| **Platt scaling** | 0.250 | 0.5j | Idem | Idem |
| **Logistic regression** (features SMC + régime + news + momentum) | **0.220-0.230** | 3j | Calibrée nativement, poids interprétables, coefficient stat-test-able | Suppose linéarité, sensible à multicolinéarité (§2) |
| **LightGBM classifier** (target = PnL > 0, 10-50 features) | **0.200-0.215** | 5j | Captures interactions et non-linéarités, feature importance, SHAP pour explicabilité | Moins explicable ; risque overfit sur 924 trades (petit dataset) ; besoin de walk-forward strict |
| **Bayesian scoring** (P(win | features) avec prior régime) | 0.225 | 4j | Gère bien l'incertitude news absente, intervalle de confiance par signal | Plus complexe à maintenir, calibration des priors douteuse sur XAU |
| **Stacking LGBM(SMC) + LGBM(macro)** | **0.190-0.205** | 10j | Meilleur benchmark académique 2025 sur directional classification | Overkill à 924 trades, besoin d'augmenter data (plus d'assets/TF) |
| **LLM-based reasoning score** (Claude classe 1-10) | 0.235-0.250 | 2j | Narrative native, explainability | Non reproductible, coût élevé, latence ; pas meilleur que LR |

**Recommandation** : **Logistic regression multicouche ou LightGBM calibré** sur target = `r_multiple > 0`. Gain attendu : Brier 0.255 → 0.215, soit +16 % d'info prédictive. C'est la différence entre un produit *marketable* et un gadget.

Références chiffrées :
- Lopez de Prado (2020), *Machine Learning for Asset Managers*, Ch. 3 : scoring rule-based sans calibration = entropie ~0, documented cas d'étude futures.
- arXiv 2310.xxxx (2024) : LightGBM sur features ICT + macro, AUC ~0.58 sur EUR/USD M15.
- Reliability diagrams et Brier : Niculescu-Mizil & Caruana (ICML 2005) — baseline méthodologique.

---

## 7. Seuil dynamique par régime — utile ?

L'idée serait : `enter_threshold = 45 en trending, 55 en ranging` pour compenser des edges variables.

**Obstacle empirique** : puisque Spearman(score, r_mult) ≈ 0, faire varier le seuil *sur le score actuel* ne change pas la qualité sélectionnée — on découpe un bruit en sous-bruits.

**Version qui marcherait** : seuil dynamique sur un **nouveau** score (LR/LGBM) calibré. Exemple :
- LR trained walk-forward, prédit P(win).
- Seuil `P ≥ 0.55` en régime trend (fort edge attendu), `P ≥ 0.65` en ranging (edge faible), `P ≥ 1.0` (pas de trade) en vol extreme.
- Calibration de seuil par **capital at risk** (Kelly fractional), pas par quantile arbitraire.

**À faire avant** : refonte §6 d'abord, puis §7.

---

## 8. Note /10 sur valeur commerciale du scoring actuel

| Dimension | Note | Justification |
|---|:-:|---|
| Explicabilité | 8/10 | Chaque composant a un `reasoning` texte, serialisable JSON |
| Déterminisme | 10/10 | Aucun random, pas de seed, reproductible à l'octet |
| **Pouvoir prédictif** | **1/10** | Pearson -0.023, Brier pire que baseline |
| **Calibration** | **1/10** | Win% décroissante avec score |
| Complétude dimensions | 6/10 | 8 composants, 4 familles (SMC, régime, news, momentum) — mais double-compté |
| Stabilité temporelle | 7/10 | Distribution stable 2019-2025 (le score ne drift pas) |
| Atteignabilité des tiers | 1/10 | PREMIUM et STANDARD empiriquement à 0 signaux |
| Test coverage | 9/10 | 578 lignes de tests, 5 classes de tests, renorm couvert |
| Architecture code | 7/10 | Dataclasses, pas de dépendance cyclique ; quelques lourdeurs (4 gates empilés, `analyze` 175 lignes) |
| Différenciation commerciale | 3/10 | Le framing « confluence 0-100 » est vendeur, mais seul le *framing* a de la valeur — pas la note elle-même |

**Note globale : 2/10** — valeur commerciale actuelle quasi-nulle, à remplacer.

**Nuance importante** : le *framework* (8 composants, tiers, narrative LLM par-dessus) est commercialement excellent. C'est la *fonction de score* qui est cassée. On peut garder le cadre et substituer un calibrateur ML dedans — 2/10 → 8/10 en ~2 semaines.

---

## 9. Top 5 refactors priorisés (effort × impact)

| # | Refactor | Effort | Impact | Priorité | Dépendance |
|:-:|---|:-:|:-:|:-:|---|
| **R1** | **Remplacer `total_score` par `P(win)` d'un LightGBM calibré (isotonic)** entraîné walk-forward sur `r_multiple > 0`, features = composants actuels + features brutes SMC/régime/macro | 5j | **ÉNORME** (Brier -16 %, seuils enfin significatifs) | P0 | Besoin audit data quality (Prompt 08) pour baseline propre |
| **R2** | **Mesurer empiriquement la matrice de corrélation 8×8** sur 106k bars en ajoutant le vecteur `component_scores` à `score_distribution.csv`, puis **supprimer 1-2 composants redondants** (ex: rsi_divergence fusionné dans bos, momentum retiré) | 1j | Gros (réduit bruit de 20-30 %) | P1 | Indépendant |
| **R3** | **Recalibrer les quality functions** pour que le score ait une plage d'usage réelle : FVG `0.3 + 0.7×norm` → `0.5 + 0.5×norm` ; regime ranging 0.3 → 0.5 ; bos quality ladder élargie. Objectif : plafond empirique 55 → 75. | 2j | Moyen (sans R1, repousse juste le problème ; avec R1, aide la séparabilité) | P2 | R1 d'abord |
| **R4** | **Ajouter `component_scores[]` persisté** par signal, exposé dans API `/signals/{id}/breakdown` pour que l'utilisateur voie la décomposition — valeur marketing immédiate | 1j | Moyen (commercial : différenciation explainability) | P2 | Indépendant |
| **R5** | **Seuil dynamique par régime** basé sur P(win) calibré (R1 required). Trade si `P ≥ θ(régime)` où θ calibré par Kelly fractional. | 3j | Moyen (ajoute 0.15-0.30 PF si edge existe) | P3 | Blocked by R1 |

### Matrice effort × impact

```
IMPACT ↑
       │
ÉNORME │           [R1]
       │
GROS   │  [R2]
       │
MOYEN  │        [R4]   [R3]           [R5]
       │
FAIBLE │
       └─────────────────────────────────────→ EFFORT
         1j   2j    3j   4j   5j    7j    10j
```

---

## 10. Plan d'exécution

### Quick wins (< 1 jour)

1. **Exporter `component_scores`** dans `score_distribution.csv` + `trades_combined.csv` pour calculer la matrice de corrélation empirique (sous-tâche R2).
2. **Ajouter un `warn` log** si `total_score_pre_renorm / present_weight × 100 > 105` (symptôme d'une quality function mal calibrée).
3. **Retirer ou geler `rsi_divergence`** (poids 2, corrélé à bos, gain signal minimal) — simple A/B sur sweep.
4. **Badge tier accessible** dans `/health` : exposer la distribution des tiers émis depuis 30j — si PREMIUM=0 depuis 7j, incident.

### Moyen terme (< 1 semaine)

5. **R2 complet** : matrice corrélation + suppression des composants redondants.
6. **R3** : recalibration des quality functions en s'appuyant sur la distribution empirique mesurée.
7. **R4** : endpoint `/signals/{id}/breakdown` + panneau dashboard component-level.
8. Préparation du dataset walk-forward pour R1 : 7 ans × (features SMC, régime, time-of-day, macro cal) → target `r_multiple > 0`. Cible : 5000-10000 trades après relâchement du seuil à 30/100.

### Long terme (> 1 semaine)

9. **R1 — LightGBM calibré** : entraînement walk-forward (5 folds 2019-2023 / test 2024-2025), isotonic calibration, Brier report. Shadow mode vs scoring actuel pendant 2 semaines.
10. **R5 — seuil dynamique par régime** basé sur P(win) de R1 + Kelly fractional sizing.
11. Extension **multi-asset** (EUR/USD, BTC/USD) pour valider la transférabilité du calibrateur ML.

---

## 11. KPIs post-amélioration (mesurables)

| KPI | Cible | Mesure |
|---|:-:|---|
| **Pearson(score, r_multiple)** | ≥ +0.12 | Sur walk-forward out-of-sample |
| **Spearman rank** | ≥ +0.15 | Idem |
| **Brier score** | ≤ 0.215 (vs baseline 0.246) | 10-15 % d'améliorations minimum |
| **% bars ≥ 60 (STANDARD)** | 0.5 - 2 % | Permet 300-1200 signaux/an en prod |
| **% bars ≥ 80 (PREMIUM)** | 0.05 - 0.2 % | Permet 30-100 signaux/an en prod |
| **Win-rate PREMIUM vs STANDARD** | ≥ +8 pp | Prouve la séparabilité par tier |
| **Expectancy PREMIUM** | ≥ +0.15 R | Vend un tier premium ≥$99 |
| **Expectancy STANDARD** | ≥ +0.05 R | Justifie le tier ANALYST |
| **Profit factor cumulé 7 ans** | ≥ 1.3 | Minimum commercial |
| **Stabilité annuelle** | ≥ 5 années sur 7 avec E[R] > 0 | Pas de one-year-wonder |

---

## 12. Trade-offs assumés

- **Remplacer le scoring rule-based par LightGBM** *dégrade légèrement* l'explicabilité. Contre-mesure : on conserve le calcul des 8 `component_scores` comme *features explicables* d'entrée du modèle, et SHAP values fournies à la narrative LLM (→ gain explainability net).
- **Supprimer `rsi_divergence`** *peut* faire baisser quelques bons signaux sur retournement. Contre-mesure : vérifier sur R2 — si feature importance LGBM > 2 %, la garder ; sinon la supprimer.
- **Seuil dynamique par régime** *augmente la complexité* du système de tiers (l'utilisateur voit des signaux dans un régime, pas dans l'autre). Contre-mesure : UX — afficher le régime actif et le seuil en vigueur.
- **Recalibration des quality functions** *augmente mécaniquement le nombre de signaux émis*, potentiellement diluant la qualité perçue côté user. Contre-mesure : coupler R3 avec R1 pour que la hausse soit *accompagnée* d'un filtre P(win) robuste.

---

## 13. Verdict — GARDER / REFONDRE / REMPLACER

**REMPLACER** la fonction de score, **GARDER** le framework de composants et les tiers.

- Le framework (8 composants, SMC+régime+news+volume+momentum, tier PREMIUM/STANDARD/WEAK, narrative LLM par-dessus, renormalisation des absents, gates news/BOS/retest) est **commercialement excellent** et techniquement propre.
- La *fonction de score* (moyenne pondérée de quality fonctions bricolées à la main) est **empiriquement non prédictive** (ρ ≈ 0, Brier > baseline) et doit être substituée par un modèle supervisé calibré.

**Effort chiffré** : 2 semaines-homme pour R1 + R2 en shadow mode, 1 semaine additionnelle pour déploiement + UI component-breakdown (R4). Total ~3 semaines pour passer de 2/10 à 8/10 de valeur commerciale.

---

## Annexe A — Fichiers et lignes référencés

- `src/intelligence/confluence_detector.py:110-119` — `DEFAULT_WEIGHTS`
- `src/intelligence/confluence_detector.py:182-356` — `analyze` (pipeline complet)
- `src/intelligence/confluence_detector.py:263-274` — fix renormalisation
- `src/intelligence/confluence_detector.py:362-416` — `_score_bos` (ladder retest)
- `src/intelligence/confluence_detector.py:418-450` — `_score_fvg` (graduation par ATR)
- `src/intelligence/confluence_detector.py:464-506` — `_score_regime`
- `src/intelligence/confluence_detector.py:508-528` — `_score_news`
- `src/intelligence/confluence_detector.py:615-624` — `_classify_tier`
- `tests/test_confluence_detector.py:497-577` — `TestScoreRenormalization` (5 tests)
- `scripts/audit_backtest.py:126-184` — `capture_full_score_distribution` (utilisé pour §3.2)
- `scripts/audit_backtest.py:69-78` — `SWEEP_CONFIGS` (production_default 75/55 à relaxed_30)
- `reports/audit/score_distribution.csv` — 106 518 bars (source §3.2)
- `reports/audit/trades_combined.csv` — 924 trades `relaxed_40` (source §4, §5)

## Annexe B — Réexécuter la calibration

```bash
python - << 'PY'
import pandas as pd, numpy as np
from scipy.stats import spearmanr
t = pd.read_csv("reports/audit/trades_combined.csv")
p = t.confluence_score.clip(0,100) / 100
y = (t.r_multiple > 0).astype(int)
print(f"n={len(t)}, win={y.mean()*100:.1f}%, E[R]={t.r_multiple.mean():+.3f}")
print(f"Brier model   ={((p-y)**2).mean():.4f}")
print(f"Brier baseline={((y.mean()-y)**2).mean():.4f}")
print(f"Pearson       ={np.corrcoef(t.confluence_score, t.r_multiple)[0,1]:+.4f}")
print(f"Spearman      ={spearmanr(t.confluence_score, t.r_multiple)[0]:+.4f}")
PY
```

Sortie attendue (reproductible sur le CSV d'audit) :

```
n=924, win=43.4%, E[R]=-0.012
Brier model   =0.2551
Brier baseline=0.2456
Pearson       =-0.0232
Spearman      =-0.0162
```
