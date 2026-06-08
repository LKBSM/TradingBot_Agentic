# Action 3 — Scoring training report (L1 vs LightGBM)

**Input** : 97 trades aggregated from sweep cells.
**After feature build** : 94 trades (drop breakevens |R| < 0.05).
**Train / OOS split** : 61 / 33 (time-ordered).
**Base rate (P(win))** : 0.330
**Winner OOS** : `logistic_l1` (BS skill = -0.0044)

## A/B comparison

| Model | BS skill IS | BS skill OOS | Verdict |
| --- | --- | --- | --- |
| `logistic_l1` | -0.3796 | -0.0044 | ❌ weak OOS |
| `lgbm` | -0.1387 | -0.1014 | ❌ weak OOS |

## Logistic L1 coefficients

| Feature | Coef | Kept |
| --- | --- | --- |
| `score_z` | +0.0000 | ❌ dropped |
| `is_long` | +0.0000 | ❌ dropped |
| `hour_sin` | +0.2148 | ✅ |
| `hour_cos` | -0.0815 | ✅ |
| `bars_held_log` | +0.1638 | ✅ |
| `exit_natural` | +0.0000 | ❌ dropped |

## LightGBM feature importance

| Feature | Importance |
| --- | --- |
| `hour_sin` | 167 |
| `score_z` | 67 |
| `hour_cos` | 27 |
| `is_long` | 0 |
| `bars_held_log` | 0 |
| `exit_natural` | 0 |

## Interprétation

❌ **Modèle non prédictif OOS** : Brier skill ≤ +0.03. Soit le sweep n'a pas généré assez de variété, soit les features dérivées (score+hour+exit+holding) ne contiennent pas d'edge exploitable. Action requise : persister les 8 composantes au signal-time (refactor TradeRecord, Sprint 4 batch 4.2 vraie version).

**Model file** : `models/scoring_v3_logistic_l1.pkl`

**Reproducibility** : `python scripts/train_logistic_l1_on_sweep.py`