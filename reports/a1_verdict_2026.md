# A1 Verdict — 2026-05-01

> **Sprint QUANT-1.3 (Elena).** Pre-specified decision: stacked
> LightGBM on 19 features over CPCV-purged 28 paths,
> evaluated against constant baseline via Diebold-Mariano.

## Configuration

- Source matrix: `C:\MyPythonProjects\TradingBOT_Agentic\data\research\a1_matrix_2019_2026.parquet`
- Target: `r_forward_4` (forward log-return, 4 M15 bars = 1h)
- CPCV: N=8 folds, k=2 test ⇒ 28 paths
- Embargo: 16 bars (4h)
- Stack: 3 level-1 LightGBMs (price / macro / calendar+intra) ⇒ 1 meta
- Hyperparams: n_estimators=200, max_depth=5, lr=0.05, min_leaf=200

## Verdict mécanique

| Métrique | Cible | Observé | Status |
|---|---|---|---|
| **DSR** (Deflated Sharpe Ratio) | > 0.99 (probability) | 0.0000 | 🔴 |
| **PBO** (Probability of Backtest Overfitting) | < 0.3 | 0.5000 | 🔴 |
| **CPCV PF moyen** (28 paths) | > 1.20 | 1.008 | 🔴 |
| **CPCV PF p25** | > 1.05 | 0.994 | 🔴 |
| **Holm-significant features** (α=0.05) | ≥ 3 | 19 | 🟢 |
| **DM test vs constant baseline** p-value | < 0.05 | 0.0000 (stat=+46.684) | 🔴 |

**Score green: 1/6 critères**

## Décision automatique

### Décision : **GO Phase 2B** (edge non démontré, pivot narrative-first)

Edge non démontré aux seuils pré-spécifiés. Bascule Phase 2B (narrative-first + RAG sourcé). Aucune surprise — la probabilité a priori P(A1 succès) était estimée à 25-35% (falsification 2026-04-30, audit CIO 3.46/10).

## Holm-significant features

- `r_1`
- `r_4`
- `r_16`
- `atr_14_pct`
- `rsi_14`
- `macd_signal_diff`
- `bar_minute_of_day`
- `dow`
- `is_lunch_hour`
- `dgs10`
- `breakeven_10y`
- `dtwexbgs`
- `vix`
- `t10y2y`
- `cot_mm_net_pct_z52`
- `cot_producer_net_z52`
- `atr_ratio_14_50`
- `min_to_next_red_news`
- `min_since_last_red_news`

> **Note méthodologique** : ce test mesure si la *gain importance* LightGBM est consistently > 0 across folds. Pour LightGBM-gain, c'est presque toujours le cas (gain ne peut pas être négatif). Avoir 19/19 features Holm-significant ici signifie 'LightGBM utilise ces features', pas 'ces features ont un pouvoir prédictif'. La preuve d'edge réel est dans **DSR + PBO + DM-stat-direction**, qui montrent ici l'absence d'edge.

## Distribution des paths CPCV

| Métrique | Valeur |
|---|---|
| CPCV Sharpe mean | +0.3838 |
| CPCV PF mean | 1.0079 |
| CPCV PF p25 | 0.9937 |
| Train size typical | 114,665 |
| Test size typical | 38,240 |

## Per-path detail

| Path | Combo | Train | Test | Trades | Sharpe | PF | HitRate |
|---|---|---|---|---|---|---|---|
| 0 | (0, 1) | 114,704 | 38,241 | 38241 | +1.967 | 1.041 | 0.505 |
| 1 | (0, 2) | 114,668 | 38,241 | 38241 | -1.998 | 0.959 | 0.499 |
| 2 | (0, 3) | 114,668 | 38,241 | 38241 | -0.192 | 0.996 | 0.497 |
| 3 | (0, 4) | 114,668 | 38,241 | 38241 | -1.770 | 0.964 | 0.495 |
| 4 | (0, 5) | 114,668 | 38,241 | 38241 | +0.738 | 1.016 | 0.496 |
| 5 | (0, 6) | 114,668 | 38,241 | 38241 | -0.161 | 0.997 | 0.506 |
| 6 | (0, 7) | 114,684 | 38,241 | 38241 | +0.421 | 1.009 | 0.505 |
| 7 | (1, 2) | 114,685 | 38,240 | 38240 | +0.063 | 1.001 | 0.500 |
| 8 | (1, 3) | 114,649 | 38,240 | 38240 | -2.132 | 0.959 | 0.500 |
| 9 | (1, 4) | 114,649 | 38,240 | 38240 | +2.303 | 1.046 | 0.508 |
| 10 | (1, 5) | 114,649 | 38,240 | 38240 | +0.692 | 1.014 | 0.503 |
| 11 | (1, 6) | 114,649 | 38,240 | 38240 | -0.825 | 0.984 | 0.502 |
| 12 | (1, 7) | 114,665 | 38,240 | 38240 | +1.781 | 1.035 | 0.509 |
| 13 | (2, 3) | 114,685 | 38,240 | 38240 | -1.197 | 0.977 | 0.496 |
| 14 | (2, 4) | 114,649 | 38,240 | 38240 | +2.373 | 1.048 | 0.504 |
| 15 | (2, 5) | 114,649 | 38,240 | 38240 | +0.171 | 1.003 | 0.500 |
| 16 | (2, 6) | 114,649 | 38,240 | 38240 | +0.451 | 1.009 | 0.503 |
| 17 | (2, 7) | 114,665 | 38,240 | 38240 | +1.055 | 1.021 | 0.508 |
| 18 | (3, 4) | 114,685 | 38,240 | 38240 | +0.706 | 1.014 | 0.497 |
| 19 | (3, 5) | 114,649 | 38,240 | 38240 | -0.673 | 0.987 | 0.495 |
| 20 | (3, 6) | 114,649 | 38,240 | 38240 | +0.542 | 1.010 | 0.505 |
| 21 | (3, 7) | 114,665 | 38,240 | 38240 | +2.276 | 1.045 | 0.505 |
| 22 | (4, 5) | 114,685 | 38,240 | 38240 | +1.178 | 1.024 | 0.505 |
| 23 | (4, 6) | 114,649 | 38,240 | 38240 | -0.051 | 0.999 | 0.501 |
| 24 | (4, 7) | 114,665 | 38,240 | 38240 | +2.641 | 1.053 | 0.506 |
| 25 | (5, 6) | 114,685 | 38,240 | 38240 | -1.859 | 0.964 | 0.493 |
| 26 | (5, 7) | 114,665 | 38,240 | 38240 | +2.018 | 1.041 | 0.501 |
| 27 | (6, 7) | 114,701 | 38,240 | 38240 | +0.230 | 1.004 | 0.506 |

## Implications produit

Activer le brief `reports/positioning/positioning_2B_narrative_first.md`. Démarrer LLM-2B.1 (RAG architecture) et INFRA-2B.1 (webapp infra) en S9. Aisha (80h) devient l'agent central de Phase 2B.

## Engagement écrit (anti-rationalisation)

Je m'engage à exécuter Phase **2B** telle que définie dans `PLAN_12_MOIS.md`, sans rationaliser un retour vers la phase non-choisie pendant ≥ 90 jours, sauf incident kill criteria explicite documenté.

Signature solo founder : ___________________  Date : ___________________

Validation Sofia : ___________________