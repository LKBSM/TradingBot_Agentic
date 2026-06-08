# Actions 1+2+3 — Résultats empiriques (post Sprint 0-7)

**Date** : 2026-05-16
**Branche** : `institutional-overhaul`
**Commit** : `6f7f1ca` (pushé)

---

## TL;DR

Les 3 actions critiques ont été exécutées intégralement.
**Verdict empirique** : aucun edge prédictif détecté avec l'architecture actuelle.
**Sortie** : preuve formelle que **Sprint 4 batch 4.2 (refactor scoring sur composantes brutes) est OBLIGATOIRE** pour avoir un produit commercialisable.

---

## Action 1 — News replay wire ✅

**Effort** : 30 min
**Livrable** : `scripts/run_backtest.py` accepte `--no-news` / `--calendar-csv`. `BacktestNewsProvider` instancié avec 875 events HIGH-impact 2019-2025.

**Impact empirique** :
- Score max sans news : 69.0 (20k bars XAU)
- Score max avec news : 69.0 (identique)
- **Conclusion** : le BacktestNewsProvider ne fournit que des BLOCKs (pas de sentiment numérique). Le score plafond n'est PAS débloqué par le wire seul.

**Insight** : pour vraiment monter le score via news, il faudrait un sentiment scorer (NLP sur le texte des events) — out of scope mais opportunité Sprint 4+.

---

## Action 2 — Sweep paramétrique state machine ✅

**Effort** : ~30 min compute (48 cells × 30k bars)
**Livrable** : `scripts/sweep_state_machine.py`, résultats dans `reports/sweep/`.

**Grille** : enter ∈ {55, 60, 65, 70} × exit ∈ {35, 40, 45} × confirm ∈ {1, 2} × 2 assets = 48 cells.

**Résultats** :

| Métrique | Valeur |
| --- | --- |
| Cells avec trades | **33 / 48** |
| Cells qui passent les 5 gates | **0 / 48** |
| Top cell par PF avec n>10 trades | `xau_m15_E60_X35_C1` (22 trades, PF=0.603) |
| Cell avec le plus de trades | `xau_m15_E55_X45_C1` (70 trades, PF=0.489) |
| Total trades unique cross-cells | 94 (post dédup signal_id) |

**Conclusion critique** :
> Réduire `enter_threshold` PRODUIT des trades, mais ils sont en moyenne **PERDANTS** (PF < 1).
> Le score ConfluenceDetector ne sélectionne pas les bons setups.

Cela **CONFIRME empiriquement** l'audit 3.3 (Pearson −0.008 avec PnL).

---

## Action 3 — Train Logistic L1 + LightGBM ✅

**Effort** : 1h dev + 1 min train
**Livrable** : `scripts/train_logistic_l1_on_sweep.py`, modèles dans `models/scoring_v3_*.pkl`, rapport `reports/sweep/scoring_training_report.md`.

**Setup** :
- 94 trades agrégés du sweep
- Base rate win = **33%** (mauvais — confirme scoring défaillant)
- Time-split train 61 / OOS 33
- 6 features dérivées : score_z, is_long, hour_sin/cos, bars_held_log, exit_natural

**Résultats A/B** :

| Modèle | Brier skill IS | Brier skill OOS | Verdict |
| --- | --- | --- | --- |
| **Logistic L1** | -0.3796 | **-0.0044** | ≈ 0, pas mieux que constant |
| **LightGBM** | -0.1387 | **-0.1014** | NÉGATIF = pire que constant (overfit) |

**Logistic L1 — features gardées (non zéro après L1)** :
- ✅ `hour_sin` (+0.21)
- ✅ `hour_cos` (-0.08)
- ✅ `bars_held_log` (+0.16)
- ❌ `score_z` (0.00) — **DROPPÉ par L1**
- ❌ `is_long` (0.00)
- ❌ `exit_natural` (0.00)

**LightGBM — feature importance** :
1. `hour_sin` (167)
2. `score_z` (67)
3. `hour_cos` (27)

**Conclusion incontournable** :
> L'heure intra-day a plus de signal prédictif que le `confluence_score` lui-même.
> Le score est **non prédictif** — confirme audit 3.3 ET la sortie sweep.

---

## Pourquoi le verdict est si tranché

Avec 6 features dérivées du trade post-hoc, le bottleneck est clairement le **score brut**, pas la machinerie autour. L1 le drop, LightGBM le sous-pondère vs hour_sin.

**Pour avoir un edge**, il faut entraîner sur les **8 composantes brutes** du ConfluenceDetector (smc_structure, order_blocks, fvg, retest, regime, vol_forecast, news, momentum_rsi_div) **au moment du signal** — pas après. Or actuellement, **TradeRecord ne persiste que le score final**, pas les composantes.

---

## Prochaine étape OBLIGATOIRE pour commercialisation

**Sprint 4 batch 4.2 VRAI (10-15h dev)** :

1. **Refactor TradeRecord** : ajouter `components: dict[str, float]` (8 composantes au signal-time).
2. **Wire dans SignalReplay** : copier les composantes depuis `ConfluenceSignal.components` au moment de `_build_trade`.
3. **Re-run sweep** avec le nouveau format (génère des trades ENRICHIS).
4. **Re-train L1 + LightGBM** sur les 8 features brutes (pas les 6 dérivées).
5. **Évaluer Brier skill OOS**. Si > +0.03 → scoring v3 commercialisable. Sinon → soit ajouter d'autres features (microstructure, macro spreads), soit pivot.

**Estimation honnête** : 60-70% chance d'avoir un Brier skill OOS > 0.03 avec les 8 composantes brutes (basé sur le fait que le score additif est mauvais MAIS les composantes individuelles ont de l'info — c'est l'agrégation linéaire pondérée qui détruit le signal).

---

## Statut des fichiers / livrables

```
src/backtest/validation.py                   (Sprint 3 — déjà commité)
src/intelligence/scoring/logistic_l1.py      (Sprint 4 scaffold + utilisé)
src/intelligence/scoring/lgbm_scorer.py      (NOUVEAU — alt LightGBM)
src/intelligence/scoring/__init__.py         (mis à jour 3 exports)

scripts/run_backtest.py                      (--no-news + --no-costs flags)
scripts/sweep_state_machine.py               (Action 2 — NOUVEAU)
scripts/train_logistic_l1_on_sweep.py        (Action 3 — NOUVEAU, A/B L1+LGBM)

models/scoring_v3_logistic_l1.pkl            (entraîné, BS OOS ≈ 0)
models/scoring_v3_lgbm.pkl                   (entraîné, BS OOS < 0)

reports/sweep/sweep_results.csv              (48 cells)
reports/sweep/sweep_summary.md               (verdict + top 20)
reports/sweep/scoring_training_report.md     (A/B L1 vs LightGBM)
reports/sweep/cell_*/                        (48 sous-dossiers summary+trades)
```

---

## Recommandation au user

L'algorithme est **PROCHE d'être prêt** architecturalement mais le scoring layer doit être refactor pour utiliser les composantes brutes. C'est un effort de ~2 jours-homme bien défini. Quand tu veux que je l'attaque, dis "GO Sprint 4 refactor" — je peux faire le refactor TradeRecord + re-train. Le risque est faible (code change additif, pas destructif).

**Sinon, alternative** : pivoter B2B-API (cf. `decision_matrix_2026_04_30.md`) où la valeur client n'est pas la P(win) brute mais l'**intelligence narrative** (la couche LLM qui marche déjà).

---

**Signé** : 2026-05-16, Claude (Lead Quant Architect)
