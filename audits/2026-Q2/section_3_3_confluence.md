# Section 3.3 — Audit ConfluenceDetector (Sprint 0, institutional overhaul)

**Date** : 2026-05-15
**Auditeur** : Claude (Lead Quant Architect)
**Périmètre** : `src/intelligence/confluence_detector.py` (637 LOC), `tests/test_confluence_detector.py` (579 LOC), data empirique `reports/eval_02/`, `reports/audit_2026_04_30_trades.csv` (n=1753), `reports/audit/trades_combined.csv` (n=2363).
**Branche** : `institutional-overhaul`
**Note globale** : **3.0 / 10**

---

## 0. TL;DR

Le `ConfluenceDetector` est **architecturalement propre** (dataclasses, déterministe, testé à 579 LOC) mais **empiriquement non prédictif** :

- Pearson(score, R-multiple) ∈ {−0.023 (eval_02 ancien), −0.019 (eval_02 actuel n=1597), −0.008 (audit_2026_04_30 n=1753)} — **toutes nulles, toutes légèrement négatives**.
- Brier skill score = **−0.022** : le score, vu comme probabilité (score/100), est ~2 % **moins informatif** qu'une probabilité constante (base rate 46.4 %).
- Calibration **non monotone** : la win rate oscille entre 42.9 % et 50 % sur 7 buckets, sans pente (cf. `reports/eval_02/monotonicity.json:is_winrate_monotone_up=false`).
- Les **poids `DEFAULT_WEIGHTS` n'ont aucune justification empirique** (pas de grid search, pas de CV, pas de validation walk-forward). Choix « à dire d'expert » → finding **P0**.
- **Double-gating empilé** : `ConfluenceDetector.min_score=25` (`confluence_detector.py:151`) + `StateMachineConfig.enter_threshold=75` (`signal_state_machine.py:129`). Le score 0-100 ne discrimine que parmi les bars déjà filtrés par 4 gates booléens (news+BOS+retest+threshold) → la sélectivité vient des gates, pas du score.
- **Plafond empirique 77.1/100** observé sur le baseline 1597 trades (`reports/eval_02/summary.json`) ; 0 % des bars dépassent 80 → tier PREMIUM (≥55 post-recalibration) reste **inatteignable conceptuellement** pour l'ancien seuil 80, et **fortement comprimé** sur 0.5-3 % seulement.
- **Stack régime** : composante régime (poids 25 %, le plus lourd) co-bouge avec BOS (gate amont) → suspecté double-comptage.

**Décision Sprint 4** : recalibrer via **isotonic regression** sur le score actuel **ne suffira pas** : on ne peut isotoniser une corrélation rang ~0. **Recommandation forte** : remplacer la fonction de score (Sprint 5) par un calibrateur supervisé (LR ou LightGBM) **et** garder l'architecture 8-composantes comme feature engineering.

---

## 1. Architecture du scoring

### 1.1 Classe & flux d'évaluation

**Fichier** : `src/intelligence/confluence_detector.py`

| Élément | Localisation | Notes |
|---|---|---|
| `SignalType` enum (LONG/SHORT) | `:28-30` | OK |
| `SignalTier` enum (PREMIUM/STANDARD/WEAK/INVALID) | `:33-43` | Seuils recalibrés 2026-04-29 : ≥55 PREMIUM (était 80), ≥40 STANDARD, ≥25 WEAK |
| `ComponentScore` dataclass | `:46-54` | Champs `raw_value`, `weighted_score`, `weight`, `reasoning` — explainable |
| `ConfluenceSignal` dataclass | `:56-109` | Inclut `position_multiplier` (regime × news) post-2026-04 |
| `DEFAULT_WEIGHTS` | `:116-125` | **8 composantes, somme = 100** (assertion `:181`) |
| `ConfluenceDetector.__init__` | `:148-182` | Validation `sum(weights)==100`, support `instrument_config` |
| `analyze()` | `:188-362` | Pipeline 175 LOC, retourne `Optional[ConfluenceSignal]` |
| 8 component scorers | `:368-601` | `_score_bos`, `_score_fvg`, `_score_order_block`, `_score_regime`, `_score_news`, `_score_volume`, `_score_momentum`, `_score_rsi_divergence` |
| `_classify_tier` | `:626-637` | Cutpoints 55/40/25 |

**Pipeline `analyze` (lignes 188-362)** — séquence stricte :

1. Gate news → `BLOCK` (`:220-222`).
2. Gate BOS → si `BOS_SIGNAL == 0`, return None (`:233-235`) — la **direction est dérivée du *trend state* BOS**, pas de l'event flag (commenté lignes 224-232).
3. Gate retest (si `require_retest=True`) → `BOS_RETEST_ARMED` doit s'aligner avec la direction (`:245-250`).
4. Boucle de scoring des 8 composants (`:253-261`) → **somme pondérée additive**.
5. Renormalisation si données absentes (`:263-280`) — `present_weight = 100 − absent` puis `score × 100 / present_weight`.
6. Filtre `min_score` (défaut 25 depuis 2026-04-29, `:282-283`).
7. Classification tier (`:285`).
8. SL/TP via ATR ou vol_forecast (`:287-319`) — high-vol ⇒ SL × 1.5 (TP figé).
9. Position multiplier `max(0, min(1.5, regime_mult × news_mult))` (`:327-340`).
10. Return `ConfluenceSignal` (`:342-362`).

### 1.2 Mode de combinaison : SOMME PONDÉRÉE (additif)

```python
total_score = sum(c.weighted_score for c in components)   # :269
```

**Pas de produit, pas de moyenne géométrique, pas de min/max**. La somme additive a une conséquence forte : **n'importe quel sous-ensemble de 5-6 composants forts peut compenser les autres**. Il n'y a **pas de veto silencieux** côté score — la seule logique veto vient des gates upstream (news BLOCK, BOS=0, retest non aligné).

### 1.3 Poids `DEFAULT_WEIGHTS` (`:116-125`)

| Composant | Poids | Famille | Source de données |
|---|---:|---|---|
| `regime` | **25** | Macro/structure | `MarketRegimeAgent.analyze()` (HMM, ADX-like) |
| `news` | **20** | Macro | `NewsAnalysisAgent.evaluate_news_impact()` (calendrier + LLM sentiment) |
| `bos` | **15** | Smart Money | `SmartMoneyEngine` (BOS_SIGNAL, BOS_EVENT, CHOCH, BOS_RETEST_ARMED) |
| `fvg` | **15** | Smart Money | `SmartMoneyEngine` (FVG_SIGNAL, FVG_SIZE_NORM) |
| `order_block` | **10** | Smart Money | `SmartMoneyEngine` (OB_STRENGTH_NORM) |
| `volume` | **10** | Liquidité | volume / volume_ma |
| `momentum` | **3** | Indicateurs classiques | RSI, MACD_Diff |
| `rsi_divergence` | **2** | Indicateurs classiques | CHOCH_DIVERGENCE |

**Familles** :
- SMC pure : 40/100 (bos + fvg + ob)
- Régime + macro : 45/100 (regime + news)
- Indicateurs classiques : 5/100 (momentum + rsi_div)
- Liquidité : 10/100 (volume)

### 1.4 `_apply_weights` n'existe pas

Le brief mentionne `_apply_weights` mais cette méthode **n'existe pas** dans le fichier. La pondération est appliquée *inline* dans chaque component scorer (chaque `_score_*` retourne `quality × self.weights[name]`). Cela rend l'override centralisé impossible sans patcher les 8 méthodes ; toute recalibration future doit toucher chaque scorer individuellement.

**Finding code-quality** (P2) : le pattern « weight applied inline » empêche d'écrire un test unitaire qui vérifie `weighted = quality × weight` sans instancier la chaîne complète.

---

## 2. Justification empirique des poids — **finding P0**

### 2.1 Recherche dans le repo

- **Code source** : aucune référence à `grid_search`, `cross_validation`, `optuna`, `tune_weights`, `optimize_weights` (grep exhaustif sur `src/`, `scripts/`, `tests/`).
- **Scripts** : `scripts/audit_backtest.py` exécute un sweep sur le `min_score` (production_default 75 → relaxed_30) mais **ne tune pas les poids** — seul le seuil global est balayé.
- **Tests** : `tests/test_confluence_detector.py:test_default_weights_sum_to_100` vérifie uniquement la somme = 100, pas les valeurs individuelles.
- **Eval_02** : §1.1 du rapport eval_02 mentionne explicitement que les poids sont hardcodés sans CV.

### 2.2 Constat

Les valeurs `regime=25, news=20, bos=15, fvg=15, ob=10, volume=10, momentum=3, rsi_div=2` sont **choisies « à dire d'expert »** par l'auteur (cohérent avec le manifeste SMC : « régime > news > structure > momentum »). **Aucune** trace de validation empirique.

**Finding P0 — `confluence_detector.py:116-125`** :
> Les poids `DEFAULT_WEIGHTS` ne sont issus d'aucune méthodologie reproductible (grid search, CV walk-forward, optimisation sous contrainte). Ils encodent une intuition d'expert qui peut être contredite par les données (cf. §3 où la matrice de corrélation des composantes montre que `regime` et `bos` partagent une fraction substantielle d'information, ce qui rend le poids cumulé 25+15=40 sur-représenté). **À recalibrer en Sprint 5** sous double contrainte : (a) maximiser Brier skill score sur splits walk-forward 2019-2023 → 2024-2025, (b) régularisation Lasso pour éliminer composantes redondantes.

---

## 3. Décorrélation des 8 composantes — analyse empirique

### 3.1 Limite de l'analyse

Les composantes ne sont **pas persistées comme vecteur continu** dans les CSV disponibles. Deux sources permettent une analyse partielle :

- `reports/audit_2026_04_30_trades.csv` (n=1753) : contient `c_bos, c_choch, c_ob, c_fvg, c_retest, c_rsi_div, c_regime, c_news_ok` **mais en binaire 0/1** (présence/absence à l'entrée du trade) — ce n'est PAS le `weighted_score` continu.
- `reports/eval_02/score_distribution.csv` : ne contient que le score agrégé.

La matrice ci-dessous est donc une **corrélation des indicateurs binaires post-gate**, qui sous-estime la corrélation réelle des contributions au score (on perd la modulation continue par `quality()`).

### 3.2 Matrice de corrélation (Pearson, binaire post-gate, n=1753)

| | bos | choch | ob | fvg | retest | rsi_div | regime | news_ok |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **bos** | 1.000 | −0.046 | −0.031 | −0.023 | −0.009 | **−0.202** | −0.081 | −0.009 |
| **choch** | | 1.000 | **−0.161** | −0.091 | **−0.262** | **−0.266** | **−0.323** | −0.018 |
| **ob** | | | 1.000 | −0.072 | **+0.496** | **−0.206** | −0.097 | −0.006 |
| **fvg** | | | | 1.000 | −0.134 | **−0.302** | **−0.202** | −0.020 |
| **retest** | | | | | 1.000 | −0.122 | −0.140 | −0.012 |
| **rsi_div** | | | | | | 1.000 | −0.170 | −0.029 |
| **regime** | | | | | | | 1.000 | +0.003 |
| **news_ok** | | | | | | | | 1.000 |

**Activation rate par composante** (n=1753) :

| Composante | Activation | Commentaire |
|---|:-:|---|
| `c_bos` | 95.8 % | Gate amont : très peu de trades sans BOS |
| `c_ob` | 97.9 % | Quasi-saturée — peu d'info marginale |
| `c_news_ok` | 99.8 % | Quasi-saturée — peu d'info marginale |
| `c_retest` | 92.0 % | Gate amont quand `require_retest=True` |
| `c_fvg` | 80.7 % | OK |
| `c_regime` | 69.8 % | OK |
| `c_choch` | 45.3 % | OK |
| `c_rsi_div` | 33.7 % | OK |

### 3.3 Points chauds (ρ ≥ 0.2 en valeur absolue, ou Cramér's V ≥ 0.3)

| Paire | ρ Pearson | Cramér's V | Interprétation |
|---|:-:|:-:|---|
| **OB ↔ Retest** | +0.496 | **0.489** (χ² p<10⁻⁹⁰) | **Très forte dépendance** : presque tous les retests ont lieu sur un order block — c'est la définition SMC opérationnelle. Double-comptage net dans le score (10 + ladder bos avec retest = ~14/15). |
| Regime ↔ CHOCH | −0.323 | 0.323 | Anti-corrélation : CHOCH (reversal) inverse souvent le trend régime → score regime tombe quand CHOCH est présent. |
| FVG ↔ RSI_div | −0.302 | — | Anti-corrélation faible. |
| CHOCH ↔ RSI_div | −0.266 | — | Surprenant : on s'attendait à +ρ (RSI divergence devrait *confirmer* CHOCH). Le compteur indique que les deux composantes capturent des cas distincts. |
| CHOCH ↔ Retest | −0.262 | — | Anti-corrélation cohérente : CHOCH = retournement, Retest = continuation après break. |
| OB ↔ RSI_div | −0.206 | — | Faible. |
| BOS ↔ RSI_div | −0.202 | — | Anti-corrélation : RSI divergence est typique des reversals, pas des continuations. |

### 3.4 Lecture

- **OB et Retest sont structurellement redondants** dans cette implémentation. Le poids combiné `ob(10) + bos(retest boost dans la ladder ≈+4 à +6)` = ~14-16 points est partiellement du double-comptage. **À supprimer ou fusionner en Sprint 5**.
- **News est quasi-saturé à 99.8 %** : la composante apporte ~zéro info marginale (presque toutes les bars qui passent les gates ont news_ok=1). Effectivement, son contribution conditionnelle se réduit à `quality × confidence` mais la *présence* du facteur est constante.
- **Regime co-bouge avec BOS** via la direction du trend : un BOS haussier implique souvent un régime uptrend (le `MarketRegimeAgent` utilise des features similaires). À mesurer rigoureusement sur le `weighted_score` continu en Sprint 5 (P1).
- **L'eval_02 §2.1** propose une **matrice heuristique** anticipant ρ(BOS, Regime)=0.55, ρ(Regime, Momentum)=0.60 — la mesure empirique de §3.2 ici ne contredit pas, mais elle est limitée par la binarisation post-gate. La mesure continue reste **à produire en Sprint 5**.

**Finding P1 — `confluence_detector.py:368-601` + `audits/2026-Q2/section_3_3_confluence.md:§3`** :
> Sur 8 composantes, au minimum 2 (OB ↔ Retest) sont empiriquement très corrélées (Cramér's V=0.489) et 3 sont saturées en présence (news, ob, bos > 95 %). **À orthogonaliser** : (a) instrumenter le pipeline pour persister `component_scores[8]` continus dans le CSV de backtest, (b) recalculer la matrice ρ sur n>5000 trades, (c) supprimer ou pondérer à la baisse les composantes redondantes.

---

## 4. Calibration empirique — **finding P0**

Source : `reports/eval_02/reliability_baseline.csv` et `reports/eval_02/summary.json` (n=1597 trades, baseline post-renormalisation, 2019-2025 XAU M15).

### 4.1 Distribution du score

| Stat | Valeur |
|---|---:|
| n | 1597 |
| mean | 46.41 |
| min | 40.00 |
| p10 | 41.01 |
| p25 | 41.43 |
| p50 | 42.92 |
| p75 | 50.85 |
| p90 | 55.17 |
| p95 | 57.82 |
| p99 | 63.57 |
| **max** | **77.12** |
| share ≥ 60 | 3.69 % |
| share ≥ 70 | 0.50 % |
| share ≥ 80 | **0.00 %** |

**Conclusion** : sur 7 ans, **aucun** trade n'a atteint le tier PREMIUM original (≥80). Le plafond empirique 77.12 est très loin du 80 marketing initial. La recalibration 2026-04-29 (PREMIUM=55) corrige le bug commercial mais ne traite pas la cause profonde : les `quality()` fonctions saturent rarement à 1.0 et la composante `momentum` (poids 3) + `rsi_divergence` (poids 2) sont quasi-impossibles à maxer simultanément.

### 4.2 Reliability diagram par bucket de score (baseline post-renorm, n=1597)

```
bucket           n    win_rate    expectancy_R    total_R
─────────────────────────────────────────────────────────
[40, 43)        806     47.3 %        +0.037        +30.2
[43, 46)        112     45.5 %        −0.071        −7.9
[46, 50)        251     47.0 %        +0.061        +15.4
[50, 55)        257     43.6 %        −0.031        −8.1
[55, 60)        112     47.3 %        +0.097        +10.9
[60, 65)         49     42.9 %        −0.069        −3.4
[65, 80]         10     50.0 %        +0.194        +1.9
─────────────────────────────────────────────────────────
TOTAL          1597     46.4 %        +0.024        +38.9
```

**Diagnostic** :
- Win rate **non monotone** (oscille 42.9 % → 50 %, range 7.1 pp). `is_winrate_monotone_up=false` (`reports/eval_02/monotonicity.json`).
- Expectancy **non monotone** (oscille −0.071 → +0.194 R).
- **Le bucket [65, 80] (n=10) est statistiquement non significatif** : 10 trades ne permettent aucune conclusion. L'expectancy +0.194 R est du bruit, pas un signal de calibration.
- Le bucket [43, 46] (n=112) est le PIRE (E[R]=−0.071) tandis que [40, 43] (n=806) est positif → **le score moyen-bas est plus profitable que le score moyen** : antithèse d'un scoring calibré.

### 4.3 Brier skill score

Source : `reports/eval_02/brier.json` :

```json
{
  "n_trades": 1597,
  "base_rate_win": 0.464,
  "brier_score_as_prob_div100": 0.2542,
  "brier_naive_mean": 0.2487,
  "brier_skill_score": -0.0220
}
```

**Interprétation** :
- Brier modèle = 0.2542
- Brier baseline (prédire toujours 0.464) = 0.2487
- **Skill score = −0.022** : le score est **2.2 % MOINS informatif qu'une probabilité constante**.

Pour rappel, un scoring légitime doit produire skill score **positif** (typiquement +0.05 à +0.20 sur classification directionnelle financière).

### 4.4 Corrélations rang

Source : `reports/eval_02/rank_correlation.json` :

```
Spearman(score, R-multiple) baseline = −0.0192
Spearman(score, R-multiple) sweep    = −0.0162
```

**Pearson cross-vérifié sur `audit_2026_04_30_trades.csv` (n=1753) : −0.0075.**

Toutes les mesures convergent : **|ρ| < 0.025**, et le signe est légèrement négatif. **Le score n'a aucun pouvoir discriminant**.

### 4.5 Implication pour Sprint 4 (isotonic regression)

Le brief Sprint 4 mentionne « recalibration via isotonic regression sans nouvelle architecture ». **Cette stratégie est mathématiquement bornée** par |ρ_Spearman| ≈ 0 :

> L'isotonic regression cherche une transformation monotone `f: score → P(win)` qui minimise le Brier. **Si Spearman(score, win) ≈ 0**, la meilleure f isotone constante = base rate (0.464) → Brier identique à la baseline.

**Conclusion §4** : Sprint 4 doit soit (a) viser une **recalibration multi-feature** (ré-entraîner sur les composantes individuelles, pas le score agrégé), soit (b) explicitement acter que la recalibration **ne corrigera pas le manque de pouvoir prédictif**, et préparer Sprint 5 (refactor) en parallèle.

**Finding P0 — `confluence_detector.py:analyze()` + `reports/eval_02/`** :
> Le score 0-100 n'a aucun pouvoir prédictif (Pearson −0.008, Spearman −0.019, Brier skill −0.022, calibration non monotone). **Sprint 4 isotonic-only est insuffisant** : exiger un modèle supervisé multi-feature (LR L1 ou LightGBM calibré) entraîné walk-forward sur target `r_multiple > 0`. Cible : Brier skill ≥ +0.05, Spearman ≥ +0.10.

---

## 5. Double-gating empilé

### 5.1 Cartographie

Sur le chemin entrée → publication, un signal subit **6 filtres successifs** avant d'arriver à l'utilisateur :

| # | Filtre | Fichier:ligne | Type |
|:-:|---|---|---|
| 1 | News BLOCK | `confluence_detector.py:220-222` | Booléen (decision == BLOCK) |
| 2 | BOS_SIGNAL ≠ 0 | `confluence_detector.py:233-235` | Booléen |
| 3 | Retest aligné (si `require_retest=True`) | `confluence_detector.py:245-250` | Booléen |
| 4 | `total_score ≥ min_score` (défaut 25) | `confluence_detector.py:282-283` | Seuil score |
| 5 | `score ≥ enter_threshold` (défaut 75) | `signal_state_machine.py:129, :487, :511, :634` | Seuil score |
| 6 | Hysteresis `confirm_bars` (défaut 2) | `signal_state_machine.py:133` | Temporel |

### 5.2 Conséquence empirique

Le bug du **plafond 70/100** (corrigé par renormalisation `:263-280`) avait pour symptôme : **0 trades** sur 7 ans avec `min_score=75` (cf. memory `Audit Backtest 2026-04-24`). Le double-gating amplifie ce risque :

- `ConfluenceDetector.min_score` est fixé à 25 pour ne pas re-créer le bug du plafond ;
- mais `StateMachineConfig.enter_threshold` est resté à 75 → toute valeur entre 25 et 75 émet un `ConfluenceSignal` qui se fait **filtrer en aval par la state machine** sans observabilité fine côté detector.

**Test empirique facile** (à produire en Sprint 1 batch 1.0) : compter, sur les 1597 trades baseline, combien de signaux ont été *émis* par le detector mais *rejetés* par la state machine (score ∈ [25, 75)). L'estimation grossière depuis `score_distribution.csv` : **96.3 %** des signaux scorés ≥40 sont sous 60 (1−0.0369), donc ~75 % sont émis par le detector mais ne dépassent jamais l'enter_threshold de la state machine. La state machine fait **75 % du travail de sélection**, pas le detector.

### 5.3 Implication

- **Le score 0-100 du detector n'est pas l'output utilisable** ; c'est un input filtré par un seuil aval. Toute communication marketing sur le score (« 75/100 = haute conviction ») est **doublement trompeuse** : (a) ρ ≈ 0, (b) le 75 réel est un cutpoint de state machine, pas de detector.
- **Recommandation Sprint 5** : unifier les deux seuils. Soit `min_score = enter_threshold` (detector devient l'unique gate score), soit garder le decoupling mais le documenter dans la spec produit comme « detector candidate generation + state machine confirmation ».

**Finding P1 — `confluence_detector.py:151` + `signal_state_machine.py:129`** :
> Double seuil score : `min_score=25` (detector) et `enter_threshold=75` (state machine). 96.3 % des signaux émis par le detector sont sous 60 et donc rejetés par la state machine. Le score « 75 » de la spec produit appartient à la state machine, pas au detector. **Action Sprint 5** : unifier ou expliciter.

---

## 6. Stabilité temporelle

Source : `reports/eval_02/yearly_baseline.csv` (n=1597, 2019-2025).

| Année | n | win_rate | E[R] | mean | p50 | p90 | max | corr(score, R) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2019 | 172 | 50.0 % | +0.059 | 46.07 | 42.88 | 54.49 | 70.05 | **+0.071** |
| 2020 | 241 | 46.9 % | +0.006 | 46.76 | 43.68 | 55.54 | 71.37 | +0.033 |
| 2021 | 164 | 45.1 % | +0.006 | 45.88 | 42.14 | 54.48 | 62.86 | −0.036 |
| 2022 | 219 | 42.0 % | −0.025 | 46.46 | 42.30 | 55.54 | 77.12 | −0.034 |
| 2023 | 137 | 40.1 % | +0.017 | 46.00 | 42.14 | 54.95 | 63.57 | **−0.076** |
| 2024 | 256 | 48.4 % | +0.064 | 46.38 | 43.35 | 55.09 | 75.38 | −0.001 |
| 2025 | 408 | 48.3 % | +0.033 | 46.68 | 43.29 | 55.60 | 73.18 | −0.043 |

### 6.1 Lecture

- **Distribution du score remarquablement stable** : mean ∈ [45.88, 46.76], p50 ∈ [42.14, 43.68], p90 ∈ [54.48, 55.60] → drift inférieur à 2 % du score sur 7 ans. **Le scoring est reproductible** — c'est la seule vertu confirmée.
- **Corrélation par année oscille entre −0.076 et +0.071** : pas de stabilité de signe, pas de saisonnalité utile.
- **2019 (+0.071) et 2023 (−0.076)** sont les extrêmes : sur 7 ans, **aucun bias robuste**.
- **2024-2025 sont les meilleures années** (E[R]=+0.064 et +0.033) mais c'est **régime trend fort** (béta XAU 2024-2025 = +27 % et +13 %) — corrélation 0.96 avec le rallye Gold (cf. memory `Forensics 2026-04-30` : « sub 2024+ = β-capture XAU »). **Pas un edge structurel**.

### 6.2 Activation rate par composante stable sur 7 ans

Source : calcul ad hoc sur `audit_2026_04_30_trades.csv` (n=1753) :

| Année | c_bos | c_ob | c_fvg | c_retest | c_choch | c_regime | c_rsi_div |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2019 | 0.961 | 0.983 | 0.847 | 0.943 | 0.498 | 0.646 | 0.345 |
| 2020 | 0.958 | 0.996 | 0.761 | 0.945 | 0.429 | 0.731 | 0.328 |
| 2021 | 0.945 | 0.980 | 0.819 | 0.909 | 0.437 | 0.685 | 0.350 |
| 2022 | 0.969 | 0.960 | 0.822 | 0.916 | 0.404 | 0.724 | 0.342 |
| 2023 | 0.967 | 0.967 | 0.794 | 0.885 | 0.428 | 0.716 | 0.383 |
| 2024 | 0.939 | 0.983 | 0.801 | 0.926 | 0.446 | 0.697 | 0.377 |
| 2025 | 0.962 | 0.979 | 0.795 | 0.912 | 0.477 | 0.711 | 0.293 |
| 2026 | 0.979 | 0.989 | 0.830 | 0.926 | 0.585 | 0.638 | 0.191 |

Variabilité annuelle modeste (5-10 pp absolus). **Pas de drift structurel** — les composantes mesurent les mêmes choses dans les mêmes proportions année après année. C'est cohérent avec un *scoring bien implémenté mais sans pouvoir prédictif*.

**Conclusion §6** : Le scoring est **temporellement stable** (good news : pas de fix urgent pour drift), mais cette stabilité est **stabilité de l'absence d'information**. La distribution ne bouge pas parce que rien dans le score n'est sensible à des conditions marché qui distinguent réellement les bonnes des mauvaises configurations.

---

## 7. Plafond observé — explication mécanique

### 7.1 Plafond empirique 77.12 (n=1597 baseline)

D'après `reports/eval_02/score_distribution.csv` :

- baseline_post_renorm : max = **77.12**, p99 = 63.57
- sweep_relaxed30_pre_renorm : max = **50.61** (et p99 = 48.0)

Le score plafonne sous 80 même après le fix de renormalisation. Pourquoi ?

### 7.2 Cause mécanique — `quality()` saturées vers le bas

Pour atteindre 80/100 il faut **simultanément** :

| Composant | Quality cible pour atteindre 80 | Condition réelle |
|---|---:|---|
| bos (15) | 1.0 = CHOCH+armed | Possible mais rare (45 % CHOCH × probabilité retest) |
| fvg (15) | ~0.95 = gap ≥0.93 ATR | Très rare (FVG_SIZE_NORM ≥1 = institutional displacement) |
| ob (10) | 1.0 = OB_STRENGTH_NORM ≥1 | Plafond technique de OB_STRENGTH_NORM = 1 |
| regime (25) | ~0.85 = aligned + conf 0.7 + str 0.7 | Régime forte tendance + haute confiance |
| news (20) | ~0.85 = sentiment ≥0.7 et conf ≥0.7 | Sentiment fort confirmant |
| volume (10) | 1.0 = ratio ≥2 | Pic volume rare en M15 XAU |
| momentum (3) | 1.0 = RSI ≥70 ou ≤30, MACD aligné | Possible |
| rsi_div (2) | 1.0 = CHOCH_DIVERGENCE aligné | Possible si CHOCH actif |

**Toutes les fonctions de quality saturent vers le bas et lentement vers le haut** :
- `_score_fvg:444-446` : `quality = 0.3 + 0.7 × min(1, fvg_size_norm)` → quality ∈ [0.3, 1.0], jamais 0 si FVG aligné.
- `_score_regime:509` : `quality = (max(conf,0.3) + max(str,0.3)) / 2` → minimum 0.3, plafond 1.0. La clamp `max(...,0.3)` empêche les très basses qualités d'écraser le score.
- `_score_news:530` : `score = w × alignment × max(confidence, 0.3)` → même mécanisme.

→ **Le design « plancher à 0.3 + plafond mou »** rend le score `concentré vers le milieu` (mean = 46.4, std très basse). Excellent pour stabilité, **catastrophique pour discrimination**.

### 7.3 Note plafond 69 « premières vagues 2025-2026 »

Le brief mentionne « le score plafonne à 69 sur les premières vagues 2025-2026 ». Vérification croisée :
- `yearly_baseline.csv` 2024 : max = 75.38 ; 2025 : max = 73.18 ; 2026 (partiel via audit_2026_04_30) : pas explicite mais p99 ≈ 87.5 sur le format binaire qui n'est pas comparable.
- Le « 69 » du brief peut référer aux scores avant renormalisation ou à un sous-batch précis. **À confirmer** par le mainteneur — la donnée disponible montre des max ∈ [62.86 (2021), 77.12 (2022)].

**Finding P2 — `confluence_detector.py:_score_fvg:444, _score_regime:509, _score_news:530`** :
> Toutes les fonctions de quality ont un plancher 0.3 (clamp `max(x, 0.3)`) et un plafond à 1.0 atteint très rarement. La distribution agrégée a un std étroit (mean 46.4) qui empêche tout cutpoint discriminant. **Action Sprint 5** : remplacer les clamps par des transformations sigmoïdes calibrées ou directement remplacer par un modèle supervisé qui apprend les non-linéarités.

---

## 8. Code quality

### 8.1 Forces

- **Type hints** : 100 % des signatures de méthodes ont des type hints (`from __future__ import annotations`, `Optional`, `Dict`, etc.).
- **Docstrings** : présentes sur classes et méthodes principales (`analyze`, `_score_bos`, etc.). Commentaires inline justifient les décisions techniques (ex `:224-232` sur le BOS_SIGNAL vs BOS_EVENT).
- **Dataclasses** : `ComponentScore`, `ConfluenceSignal` immuables-friendly (sans frozen explicite cependant — improvement à noter).
- **Validation** : `__post_init__` n'est pas utilisé ici, mais validation de poids dans `__init__:181-182`.
- **Test coverage** : 579 LOC de tests pour 637 LOC de code (ratio ~0.91). 5 classes de tests, incluant `TestScoreRenormalization` (5 tests sur le fix bug renorm).
- **Renormalisation absente présente-mais-neutre** : commentaires `:263-269` documentent rigoureusement le rationale du fix.
- **Per-instrument config** : support `instrument_config` (`:170-179`) — propre pour multi-asset.

### 8.2 Faiblesses

| Issue | Localisation | Sévérité |
|---|---|:-:|
| `analyze` méthode de 175 LOC (lignes 188-362) | `:188-362` | P2 — devrait être décomposée |
| Pondération inline dans chaque `_score_*` empêche test centralisé `weighted = quality × weight` | `:368-601` | P2 |
| `_score_bos:368-422` mêle 2 modes (`require_retest=True/False`) avec branches dupliquées et 2 helper fns internes | `:386-402` | P2 — refactor lisibilité |
| `volume_ma <= 0` traité comme absence mais `volume_ma == 0.01` reste comptabilisé → instabilité numérique | `:540` | P2 (déjà noté eval_02 §3.1) |
| Clamp `total_score = max(0, min(100, total_score))` masque une éventuelle sur-pondération sans warning | `:280` | P2 (déjà noté eval_02) |
| `_score_regime:484` `str(regime_type)` peut produire des strings non-attendues si enum non valide | `:484` | P2 |
| Pas de docstring `"""…"""` sur `_classify_tier` (juste un commentaire) | `:626-637` | P3 cosmétique |
| Pas de logging structuré dans `analyze` — `logger.debug` unique sur news block | `:221` | P2 — pour observabilité production |
| `position_multiplier = max(0, min(1.5, regime_mult × news_mult))` — la borne 1.5 est arbitraire (justification commentée mais pas testée empiriquement) | `:337` | P2 |

### 8.3 Test gaps

Lecture rapide de `tests/test_confluence_detector.py` (579 LOC) :
- Couvre : gates news/BOS/retest, renormalisation, classification tier, position multiplier (regime × news), seuils.
- **Manque** :
  - Tests de **calibration** (golden tests : un input réel → un score attendu à 0.01 près).
  - Tests d'**invariants** (idempotence, monotonie d'un composant seul).
  - Tests de **fuzz** ou property-based (Hypothesis : générer des `smc_features` aléatoires, vérifier que `0 ≤ score ≤ 100` jamais violé).
  - Tests de **régression** sur baseline (n=1597 trades figés ; score recomputé identique).

**Finding P2 — `tests/test_confluence_detector.py`** :
> Tests fonctionnels présents (579 LOC) mais aucun **golden test de calibration** ni **property-based test**. Ajouter en Sprint 5 (a) 50 fixtures réels du baseline avec score attendu, (b) tests Hypothesis sur les invariants `0 ≤ score ≤ 100` et `monotone-in-component`.

---

## 9. Note 0-10 justifiée

| Dimension | Note | Justification ancrée |
|---|:-:|---|
| Architecture code | 7/10 | Dataclasses, type hints, per-instrument config (`:148-182`). `analyze` est trop longue (175 LOC, `:188-362`). |
| Explicabilité | 8/10 | `ComponentScore.reasoning` (`:53`) sérialisable JSON ; 8 composantes nommées. |
| Déterminisme | 10/10 | Aucun `random.*`, aucune dépendance temporelle interne. |
| Test coverage fonctionnelle | 7/10 | 579 LOC tests sur 637 LOC code, mais pas de golden/property-based (cf. §8.3). |
| Justification empirique des poids | **1/10** | Hardcodés, aucune CV/grid search (cf. §2). **P0** |
| Pouvoir prédictif | **1/10** | Pearson −0.008, Brier skill −0.022, Spearman −0.019 (`reports/eval_02/`). **P0** |
| Calibration monotone | **1/10** | Win rate non monotone (42.9 → 50 %), expectancy non monotone (`reports/eval_02/monotonicity.json:false`). **P0** |
| Décorrélation composantes | 4/10 | OB↔Retest Cramér's V=0.49, news quasi-saturé 99.8 %, 3 composantes >95 % activation. **P1** |
| Atteignabilité des tiers (post-recalibration 55) | 5/10 | PREMIUM (≥55) ≈ 3.7 % des signaux ; ancien PREMIUM ≥80 = 0 %. Recalibration 2026-04-29 corrige le bug commercial. |
| Stabilité temporelle | 7/10 | Distribution stable 7 ans (cf. §6.1). Stabilité de la non-information. |
| Double-gating empilé | 3/10 | 6 filtres successifs, score detector ≠ score state machine (`:151` + `signal_state_machine.py:129`). **P1** |
| Différenciation commerciale | 4/10 | Le framework « 8 composantes 0-100 » est vendeur. La fonction de score actuelle est cassée. |

**Note globale pondérée : 3.0 / 10**

Pondération utilisée : pouvoir prédictif (30 %), calibration (15 %), justification poids (10 %), code quality (15 %), tests (10 %), décorrélation (10 %), stabilité (5 %), double-gating (5 %).

Calcul : 0.30×1 + 0.15×1 + 0.10×1 + 0.15×7 + 0.10×7 + 0.10×4 + 0.05×7 + 0.05×3 = 0.30 + 0.15 + 0.10 + 1.05 + 0.70 + 0.40 + 0.35 + 0.15 = **3.20** → arrondi **3.0**.

---

## 10. Findings priorisés

### 10.1 P0 (bloquants, à traiter Sprint 4-5)

| # | Finding | Ancrage | Action |
|:-:|---|---|---|
| **P0-1** | Score sans pouvoir prédictif (Pearson −0.008, Brier skill −0.022) | `reports/eval_02/brier.json`, `reports/eval_02/rank_correlation.json` | **Sprint 5 refactor** : remplacer somme pondérée par modèle supervisé (LR L1 ou LightGBM) entraîné walk-forward. Cible Brier skill ≥ +0.05. |
| **P0-2** | Calibration non monotone (win rate 42.9-50 %) | `reports/eval_02/monotonicity.json:is_winrate_monotone_up=false` | **Sprint 4 recalibration** : isotonic regression sur le score actuel ne suffit pas (ρ rang ≈ 0). Recalibrer multi-feature. |
| **P0-3** | Poids `DEFAULT_WEIGHTS` hardcodés sans justification empirique | `confluence_detector.py:116-125` | **Sprint 5** : grid search + CV walk-forward, ou abandonner les poids fixes au profit d'un modèle appris. |

### 10.2 P1 (importants, Sprint 5-6)

| # | Finding | Ancrage | Action |
|:-:|---|---|---|
| **P1-1** | OB ↔ Retest redondants (Cramér's V=0.489) ; News saturé 99.8 % | `confluence_detector.py:458-468, :536-555` + audit §3.3 | Persister composantes continues dans le CSV de backtest ; recalculer matrice ρ ; fusionner ou supprimer composantes redondantes. |
| **P1-2** | Double-gating empilé `min_score=25` + `enter_threshold=75` | `confluence_detector.py:151` + `signal_state_machine.py:129` | Unifier en un seul seuil OU documenter explicitement le split « candidate generation » vs « confirmation ». |
| **P1-3** | Plafond mou des `quality()` fonctions (plancher 0.3, max 1.0 rarement atteint) | `confluence_detector.py:444, :509, :530` | Remplacer par sigmoïdes calibrées en Sprint 5 OU directement par modèle ML. |

### 10.3 P2 (qualité, Sprint 6-7)

| # | Finding | Ancrage | Action |
|:-:|---|---|---|
| **P2-1** | `analyze` méthode 175 LOC | `:188-362` | Décomposer en `_gate_news`, `_gate_bos_retest`, `_score_components`, `_renormalize`, `_build_signal`. |
| **P2-2** | Pondération inline empêche test centralisé | `:368-601` | Extraire `_apply_weight(quality, name)` helper testable. |
| **P2-3** | Pas de golden test / property-based | `tests/test_confluence_detector.py` | Ajouter 50 fixtures baseline + Hypothesis. |
| **P2-4** | Borne `position_multiplier ≤ 1.5` arbitraire | `:337` | Backtester sur baseline pour confirmer borne. |
| **P2-5** | `volume_ma == 0.01` n'est pas traité comme absence | `:540` | Ajouter seuil `volume_ma > 1e-6` ou similaire. |

### 10.4 P3 (cosmétique)

- Docstring manquante sur `_classify_tier:626-637`.
- Logging structuré absent dans `analyze` (un seul `logger.debug` ligne 221).

---

## 11. Recommandations actionnables — Sprint 4 / Sprint 5

### 11.1 Sprint 4 (recalibration — pas de nouvelle architecture)

Le brief Sprint 4 demande « recalibration via isotonic regression sans nouvelle architecture ». **Compte tenu de Spearman ≈ 0, isotonic seul ne corrigera rien**. Trois propositions par ordre de coût croissant :

| Option | Effort | Brier skill cible | Note |
|---|:-:|:-:|---|
| **A.** Isotonic regression sur le score agrégé seul | 0.5j | ≈ 0 | Échoue par construction. **Non recommandé** seul. |
| **B.** Platt scaling / sigmoid calibration sur le score | 0.5j | ≈ 0 | Idem A, échoue. |
| **C.** Recalibrage **multi-feature** : Logistic regression L1 sur les 8 `weighted_score` individuels → `P(win)` calibrée | **2-3j** | **+0.03 à +0.06** | **Recommandé Sprint 4** — pas de nouvelle architecture, on lit les composantes existantes et on remplace seulement la fonction d'agrégation. |

**Recommandation pratique Sprint 4** :
1. Instrumenter `confluence_detector.py` pour persister `components: List[ComponentScore]` continus dans `reports/audit/trades_combined.csv` (1 commit, +20 LOC).
2. Entraîner une logistic regression L1 walk-forward (2019-2023 train / 2024-2025 test) sur target `r_multiple > 0`, features = 8 `weighted_score`.
3. Comparer Brier skill / Spearman / monotonicité vs baseline actuelle.
4. Si Brier skill ≥ +0.03 : publier la calibration comme `ScoreCalibrator` séparé, **inchangé `confluence_detector.py`** côté production (juste ajout d'une couche `P(win) = calibrator(components)` aval).
5. Si Brier skill < +0.03 : escalader vers Sprint 5 refactor.

### 11.2 Sprint 5 (refactor — fonction de score remplacée)

Si Sprint 4 confirme l'insuffisance, refactor structurel :

1. **Conserver** : framework 8 composantes, tiers, narrative LLM par-dessus, gates news+BOS+retest, renormalisation absents.
2. **Remplacer** : la fonction `total_score = Σ weighted_score` par un modèle supervisé (LightGBM ou LR + features brutes SMC/régime/macro).
3. **Unifier** les 2 seuils (detector min_score = state machine enter_threshold).
4. **Persister** `component_scores[]` dans la base SQLite signals.
5. **Exposer** un endpoint `/signals/{id}/breakdown` (déjà demandé par eval_02 §10).
6. **Refactor `analyze`** en 5 sous-méthodes (cf. P2-1).
7. **Ajouter tests** golden + Hypothesis.

Effort total Sprint 5 ≈ **2 semaines-homme** (cf. eval_02 §9).

### 11.3 Sprint 6+ (raffinements conditionnels)

- Seuil dynamique par régime (Kelly fractional) **uniquement si** P(win) est calibrée (Sprint 4 C ou Sprint 5 sortie).
- Multi-asset transfer (EURUSD, BTCUSD) — tester si la calibration tient.

---

## 12. Ce que cet audit ne couvre PAS

- **Performance compute** : pas de benchmark microbenchmark sur `analyze()`. Latence supposée < 1 ms par appel (8 composants × opérations simples), mais non mesurée. À couvrir en eval_21 / Sprint 1.
- **Memory leaks** : pas d'analyse heap. Dataclasses Python = négligeable mais non mesuré.
- **Sécurité** : pas d'audit injection / désérialisation. La méthode `analyze` consomme des dicts/objets confiables (issus du scanner, pas user input direct). À couvrir si exposition future via API directe.
- **Multi-instrument** : audit fait sur XAU M15 uniquement. Les CSV `audit_2026_04_30` et `eval_02` ne couvrent pas EURUSD, BTCUSD, US500. Le score est-il transférable ? **Inconnu**.
- **Régime news en LIVE** : audit fait sur replay historique news. Latence et qualité du feed news live (forexfactory_live.py) sont auditées en eval_24 et hors périmètre ici.
- **Composantes continues** : la matrice de corrélation §3.2 est sur **binaires post-gate** (faute de données continues persistées). La vraie matrice ρ doit être recalculée Sprint 5 batch 5.1 (cf. P1-1).
- **Confidence intervals** : les Pearson/Spearman/Brier rapportés n'ont pas de bootstrap CI. À ajouter dans `reports/eval_02/` pour publication.
- **Comparaison vs baselines naïves** : pas de comparaison directe avec « toujours LONG en uptrend » ou « toujours en breakout structurel ». eval_02 §6 propose ces benchmarks mais ils ne sont pas implémentés.
- **Effet de la conformal wrapper / regime_gate** : `src/intelligence/conformal_wrapper.py` et `regime_gate.py` consomment le confluence_score en aval. Leur effet net sur la prédictivité finale du système n'est pas mesuré ici (cf. memory `3 Piliers IMPLÉMENTÉS 2026-05-13`).
- **Mockup recalibration** : aucun POC isotonic / LR n'est implémenté dans cet audit (audit = lecture, pas implémentation). Recommandation **proposée** mais non **prouvée**.

---

## 13. Annexes

### A. Fichiers et lignes référencés

- `src/intelligence/confluence_detector.py:28-43` — Enums
- `src/intelligence/confluence_detector.py:116-125` — `DEFAULT_WEIGHTS` **P0**
- `src/intelligence/confluence_detector.py:148-182` — `__init__` + validation poids
- `src/intelligence/confluence_detector.py:188-362` — `analyze` pipeline
- `src/intelligence/confluence_detector.py:220-250` — 3 gates upstream
- `src/intelligence/confluence_detector.py:263-280` — Fix renormalisation
- `src/intelligence/confluence_detector.py:368-601` — 8 component scorers
- `src/intelligence/confluence_detector.py:626-637` — `_classify_tier`
- `src/intelligence/signal_state_machine.py:120-185` — `StateMachineConfig`
- `src/intelligence/signal_state_machine.py:129` — `enter_threshold=75` **P1**
- `src/intelligence/score_calibration.py:1-110` — Score → narrative bucket (utilisé downstream LLM)
- `src/intelligence/sentinel_scanner.py:863-867` — Per-symbol detector instantiation
- `reports/eval_02_confluence.md` — Audit complet 2026-04-24
- `reports/eval_02/summary.json` — Distribution + reliability **P0 evidence**
- `reports/eval_02/brier.json` — Brier skill score = −0.022
- `reports/eval_02/monotonicity.json` — Calibration non monotone
- `reports/eval_02/rank_correlation.json` — Spearman ≈ −0.02
- `reports/eval_02/yearly_baseline.csv` — Stabilité 7 ans
- `reports/audit_2026_04_30_trades.csv` — n=1753 trades, composantes binaires (source §3.2)
- `reports/audit/trades_combined.csv` — n=2363 trades sweep
- `tests/test_confluence_detector.py` — 579 LOC tests

### B. Commandes de re-vérification

Pour reproduire les chiffres de l'audit :

```bash
# 1. Brier, calibration, monotonicité (déjà persistés)
cat reports/eval_02/summary.json | jq '.brier'
cat reports/eval_02/monotonicity.json
cat reports/eval_02/rank_correlation.json

# 2. Matrice de corrélation composantes binaires (§3.2)
python -c "
import pandas as pd
df = pd.read_csv('reports/audit_2026_04_30_trades.csv')
comps = ['c_bos','c_choch','c_ob','c_fvg','c_retest','c_rsi_div','c_regime','c_news_ok']
print(df[comps].corr().round(3).to_string())
print('Activation:', df[comps].mean().round(3).to_dict())
"

# 3. Cramér's V OB vs Retest
python -c "
import pandas as pd
from scipy.stats import chi2_contingency
df = pd.read_csv('reports/audit_2026_04_30_trades.csv')
ct = pd.crosstab(df.c_ob, df.c_retest)
chi, p, dof, exp = chi2_contingency(ct)
n = ct.sum().sum()
v = (chi/(n*(min(ct.shape)-1)))**0.5
print(f'Cramer V = {v:.3f}, p = {p:.3e}')
"

# 4. Pearson score vs r_realized
python -c "
import pandas as pd
df = pd.read_csv('reports/audit_2026_04_30_trades.csv')
print('Pearson:', round(df.score_in.corr(df.r_realized), 4))
"
```

### C. KPIs de sortie pour Sprint 4 et Sprint 5

| KPI | Baseline actuel | Cible Sprint 4 (recalibration C) | Cible Sprint 5 (refactor LGBM) |
|---|---:|---:|---:|
| Pearson(score, r_realized) | −0.008 | ≥ +0.10 | ≥ +0.15 |
| Spearman rank | −0.019 | ≥ +0.10 | ≥ +0.15 |
| Brier skill score | −0.022 | ≥ +0.03 | ≥ +0.05 |
| Monotonicité win rate (bucket) | false | true (≥6/7 buckets) | true (≥9/10 buckets) |
| Composantes corrélées (ρ ≥ 0.4) | 1 paire (OB-Retest) | 1 paire (mesure continue) | 0 paire |
| Couverture tier PREMIUM (≥55) | 3.7 % | 3-5 % | 5-10 % |
| Win rate PREMIUM vs STANDARD (gap) | ~0 pp | ≥ +5 pp | ≥ +8 pp |
| Expectancy PREMIUM | +0.097 R (n=112) | ≥ +0.10 R | ≥ +0.15 R |
| Brier CI 95 % | non calculé | bootstrap publié | bootstrap publié |

---

**Auteur** : Claude (Lead Quant Architect, Smart Sentinel AI)
**Statut** : Tranché — exécution Sprint 4/5 sous responsabilité du spécialiste.
**Prochain audit** : `section_3_4_signal_state_machine.md` (cf. signal_state_machine.py + double-gating §5).
