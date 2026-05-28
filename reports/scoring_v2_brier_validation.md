# DG-025 — Calibrated Conviction Pipeline Brier Validation

**Date** : 2026-05-27
**Sprint** : Phase 1 / Sprint 1 — Cœur algorithmique
**Référence** : `docs/governance/dev_focus_plan_2026_05_27.md` § Sprint Tech 1

---

## TL;DR

Pipeline LGBM → Isotonic → ACI **construit, trainé, sauvegardé**.
- Modèle prod : `models/calibrated_conviction_v1.pkl` (27.9 kB)
- Données : 133 trades XAU M15 2019-2026 (backtest E55/X35, --no-news)
- Base rate win = 29.3% (39 wins / 133 trades)

**Brier skill OOS = −0.251** (val Brier 0.2182 vs naive 0.1744)

⚠️ **Le gate `Brier skill ≥ +5 %` n'est PAS atteint** avec les composantes de scoring actuelles.

Ce résultat confirme empiriquement le verdict de `reports/certification/ACTIONS_1_2_3_RESULTS.md` (2026-05-16) : le scoring confluence à 8 composantes n'a pas de pouvoir prédictif extractible par LGBM sur l'historique XAU M15.

---

## Setup empirique

### Backtest source

```bash
python scripts/run_backtest.py \
    --csv data/XAU_15MIN_2019_2026.csv \
    --symbol XAUUSD --timeframe M15 \
    --enter 55 --exit 35 --no-news \
    --trades-csv reports/calibration/trades_xau_2019_2026.csv \
    --out reports/calibration/backtest_xau.json
```

Résultats backtest :
- 133 trades sur 7 ans
- WR 29.3 %, PF 0.36-0.39 selon le tier
- Score max observé : 81.2 (avec news off, plafond ~70-74 attendu)
- Sharpe annualisé : -1.544

### Pipeline d'entraînement

```bash
python scripts/train_calibrated_conviction_real.py \
    --trades-csv reports/calibration/trades_xau_2019_2026.csv \
    --output-pkl models/calibrated_conviction_v1.pkl \
    --val-fraction 0.30 --alpha 0.10 \
    --lgbm-leaves 7 --lgbm-estimators 50
```

- 93 train / 40 val (chronological split)
- LGBM : 50 trees × 7 leaves (conservatif vu n=93)
- Isotonic : 40 prédictions validation
- ACI alpha=0.10 (90 % coverage cible)

---

## Métriques

| Métrique | Train | Val |
|---|---|---|
| Accuracy | 0.602 | 0.725 |
| Brier | — | 0.2182 |
| Naive Brier | — | 0.1744 |
| **Brier skill** | — | **−0.251** |

Lecture : un modèle qui prédirait toujours la base rate (29.3 % win) atteint Brier 0.1744. Notre LGBM, en s'écartant de cette prédiction constante, fait **moins bien** que ce baseline.

### Pourquoi `val_acc` est-il élevé alors que Brier skill est négatif ?

Avec une base rate de 29.3 %, prédire systématiquement "loss" donne 70.7 % d'accuracy. Le modèle a appris à pencher vers cette classe majoritaire, ce qui gonfle l'accuracy mais ne reflète aucune capacité prédictive sur la classe minoritaire (wins).

---

## Mapping composantes utilisé

```
cmp_BOS              → score_bos
cmp_OrderBlock       → score_order_block
cmp_FVG              → score_fvg
cmp_Regime           → score_regime
cmp_News             → score_news
cmp_RSI_Divergence   → score_momentum_rsi_div
(absent du TradeRecord) → score_retest         := 0.0
(absent du TradeRecord) → score_vol_forecast   := 0.0
outcome              := int(r_multiple > 0)
pnl_r_multiple       := r_multiple
```

Les composantes `retest` et `vol_forecast` ne sont pas encore persistées dans `TradeRecord.components`. C'est une zone d'amélioration mais peu probable de changer le verdict global.

---

## Décision opérationnelle

### Ce qui est livré dans Sprint 1 (DG-025 audit)
- ✅ Pipeline LGBM → Isotonic → ACI **entraîné** et sauvegardé
- ✅ Artefact prod `models/calibrated_conviction_v1.pkl` créé
- ✅ Adapter `scripts/train_calibrated_conviction_real.py` pour re-train sur n'importe quel CSV de trades
- ⚠️ **Brier skill +5 % NON atteint** — gate empirique non franchi

### Conséquence sur le flag prod `SCORING_VERSION`

Décision : par défaut **`SCORING_VERSION=v1`** (pipeline calibré désactivé en prod).

Raisons :
1. Le modèle calibré ne bat pas la baseline → exposer des P(win) calibrées qui n'apportent rien.
2. L'interval conformal a été calibré sur 40 trades = trop peu pour une couverture fiable (ACI converge en plusieurs centaines d'observations).
3. La confluence v1 reste la source de vérité tant que le scoring v2 n'a pas démontré un edge.

Le pipeline reste **prêt à activer** dès que :
- (a) Brier skill OOS ≥ +5 % sur ≥ 500 trades (XAU + EURUSD combinés)
- (b) Couverture ACI empirique mesurée dans [0.85, 0.95] avec n ≥ 200

### Plan pour franchir le gate ultérieurement

1. **Persistance composantes `retest` + `vol_forecast`** dans `TradeRecord.components` (peu d'effort, retrouve 2 features).
2. **Combinaison XAU + EURUSD** pour augmenter le n train (XAU 133 + EUR ~? → ~250-300 trades).
3. **Features dérivées** :
   - hour-of-day sin/cos (le rapport `ACTIONS_1_2_3` montrait que `hour_sin` est la feature la plus prédictive disponible)
   - session label one-hot (london/ny/asia)
   - distance from VWAP / weekly pivot
   - regime probabilité plutôt que label
4. **Stratégie de re-train** :
   - Walk-forward CPCV (López de Prado 2018) — 5 folds embargo=24h
   - Brier skill computed per fold + mean ± std reporté
5. **Si Brier skill toujours < +5 %** après ces étapes → pivot stratégie (ex : event-driven macro, B2B-API) tel que recommandé dans `eval_00_synthesis.md` et `a1_verdict_2026_05_01.md`.

---

## Référencement

- Pipeline class : `src/intelligence/scoring/calibrated_conviction.py`
- LGBM scorer : `src/intelligence/scoring/lgbm_scorer.py`
- Isotonic recal : `src/intelligence/scoring/isotonic_recalibration.py`
- Conformal wrap : `src/intelligence/conformal_wrapper.py`
- Trainer script : `scripts/train_calibrated_conviction.py`
- Real-data adapter : `scripts/train_calibrated_conviction_real.py`
- Loader prod : `src.intelligence.scoring.calibrated_conviction.load_calibrated_pipeline`
- Wiring prod : `src/intelligence/main.py` (`build_system` — branche `SCORING_VERSION=v2`)

---

## Signature

2026-05-27 — Phase 1 / Sprint 1 — Cœur algorithmique
Auteur : assistant Claude (Auto mode), conforme `dev_focus_plan_2026_05_27.md`.
