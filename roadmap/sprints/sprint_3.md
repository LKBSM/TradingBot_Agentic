# Sprint 3 — Statistical Edge Discovery

**Période** : Semaines 7-8 (S7-S8, ~2026-06-27 → 2026-07-11)
**Charge estimée totale** : **112 h** productives + buffer 14 h = 126 h
**Objectif** : LE sprint critique de la roadmap. Découvrir empiriquement où réside l'edge statistique (si edge il y a). Feature engineering exhaustif → Information Coefficient par feature → stacking avec conditionnement régime → industrialisation CPCV/DSR/PBO → sweep paramétrique state machine. **Gate binaire** : CI 95 % PF lo > 1.0 sur ≥ 1 actif/TF, OU pivot valeur explicative (information ratio, calibration probabiliste).
**Gate de sortie** : 1 actif/TF passe CI 95 % PF lo > 1.0, DSR ≥ 1.5, PBO ≤ 0.35, DM_p < 0.05 OU décision pivot tranchée et documentée.

---

## 0. Vue d'ensemble — 5 batches

| Batch | Titre                                                          | Heures | Critique chemin |
| ----- | -------------------------------------------------------------- | ------ | --------------- |
| 3.1   | Feature engineering exhaustif (microstructure, order flow, session, macro) | 24 h | ✅ |
| 3.2   | Information Coefficient + MIC + mutual information par feature | 16 h   | ✅              |
| 3.3   | Stacking + conditionnement par régime                          | 22 h   | ✅              |
| 3.4   | **CPCV/DSR/PBO industrialisés** (`src/research/` → `src/backtest/validation/`) | 28 h | ✅ PRIORITAIRE |
| 3.5   | Sweep paramétrique state machine × 4 actifs × 4 TF             | 22 h   | ✅              |
| —     | Buffer (debug, gate review, pivot decision)                    | 14 h   |                 |
| **TOTAL** |                                                            | **126 h** |             |

---

## Batch 3.1 — Feature engineering exhaustif (24 h)

### Objectif
Étendre la feature library de ~10 features actuelles (smart money + 1 vol forecast) vers ~60 features couvrant 4 familles : microstructure, order flow proxies, session/temporal, macro spreads. Aucun ML encore — juste génération + persistance.

### Steps
1. **Inventaire features actuelles** (2 h)
   - `Grep` features dans `strategy_features.py` + `multi_timeframe_features.py`.
   - Lister 10-15 features baseline.
   - Sortie : `audits/2026-Q2/features_inventory_baseline.md`.

2. **Features microstructure** (5 h)
   - `src/intelligence/features/microstructure.py` :
     - **Range / ATR ratio** (intra-bar volatility).
     - **Body / wick ratio** (rejection strength).
     - **Volume / dollar volume** (BTC quirks).
     - **Tick imbalance proxy** (close - vwap) / range.
     - **Realized volatility 5/15/60 bars** (HAR inputs).
     - **Bipower variation** (jump detection).
     - **Skew / kurtosis returns 20 bars**.
     - **Hurst exponent** (mean reversion vs trending).

3. **Features order flow proxies** (4 h)
   - `src/intelligence/features/order_flow.py` :
     - **Cumulative delta proxy** ((close-low) - (high-close)) × volume.
     - **Volume spike** : volume / EMA20(volume) > 2.
     - **Trapped traders** : long upper wick + close near low + next bar reversal.
     - **Sweep liquidity** : wick beyond recent swing high/low, close back inside.

4. **Features session/temporal** (4 h)
   - `src/intelligence/features/temporal.py` :
     - **Session flag** : Asian/London/NY/overlap (par actif, configurable).
     - **Hour of day** (sin/cos encoding pour ML).
     - **Day of week**.
     - **Bars since session open**.
     - **Bars since last NFP / FOMC / ECB / BOE** (depuis calendar Sprint 1.3).
     - **Time to next high-impact event**.

5. **Features macro spreads** (4 h)
   - `src/intelligence/features/macro.py` :
     - **DXY change 1/5/20 bars** (USD strength) — proxy nécessaire car pas de feed live.
     - **VIX level + change** (risk sentiment) — depuis `data/macro/`.
     - **US 10Y yield change** (rates) — depuis `data/macro/`.
     - **Gold/EUR spread** (XAU/EUR cross).
     - **BTC/Gold ratio** (risk-off proxy).

6. **Pipeline génération** (3 h)
   - `src/intelligence/features/pipeline.py` :
     - `generate_features(df, instrument) -> DataFrame` (60 colonnes).
     - Cache parquet `data/features/{symbol}_{tf}_features.parquet`.
     - Versioning par hash code source.

7. **Tests features** (2 h)
   - `tests/test_features_pipeline.py` :
     - Shape (n_bars, 60).
     - No NaN sauf warmup bars.
     - No look-ahead (chaque feature à t n'utilise que t' ≤ t).

### Critères d'acceptation
- ✅ 60+ features générées sur XAU + EURUSD + 4 autres MVP.
- ✅ Cache parquet versionné.
- ✅ Tests no look-ahead verts.
- ✅ Documentation `docs/algo/features.md`.

### Findings audit adressés
- Préparation P0-1 (ConfluenceDetector sans edge) — donner du carburant au scoring.

### Dépendances
- Sprint 1 batch 1.3 (calendrier économique).
- `data/macro/` (DXY, VIX, US10Y) — vérifier disponibilité, fallback Yahoo Finance sinon.

### Risques
- Macro feeds manquants → fallback Yahoo Finance daily (latence acceptable backtest, pas live).
- 60 features = curse of dimensionality. Mitigation : Sprint 3.2 sélection IC.

---

## Batch 3.2 — Information Coefficient par feature + MIC + MI (16 h)

### Objectif
Mesurer le pouvoir prédictif **isolé** de chaque feature sur les targets (return forward 1/5/20 bars). Quantifier via IC (Spearman), MIC (maximal info coefficient), mutual information. Identifier top 10-15 features → input Sprint 3.3.

### Steps
1. **Définition targets** (2 h)
   - Forward returns 1/5/20 bars (log-returns).
   - Forward Sharpe (return / vol future) 20 bars.
   - Binary target : `forward_return > 0` (classification).
   - Sortie : `src/intelligence/research/targets.py`.

2. **Calcul IC Spearman** (3 h)
   - Pour chaque feature × target × actif × TF :
     - Spearman rank correlation, p-value Benjamini-Hochberg corrected.
   - Sortie : `reports/sprint_3/ic_matrix.parquet`.

3. **Calcul MIC** (3 h)
   - `minepy` library (MIC).
   - Détecte relations non-linéaires.
   - Subsample 5 000 bars pour speed.

4. **Mutual Information** (2 h)
   - `sklearn.feature_selection.mutual_info_regression`.
   - Discretized targets (10 quantiles).

5. **Sélection top features** (2 h)
   - Aggregate score : `0.5 * IC + 0.3 * MIC + 0.2 * MI_normalized`.
   - Top 15 par actif.
   - Sortie : `config/feature_selection.json`.

6. **Visualisation** (2 h)
   - `scripts/plot_feature_ic.py` :
     - Heatmap IC matrix (feature × target × actif).
     - Barplot top 15 features.
   - Output : `reports/sprint_3/feature_ic_plots/`.

7. **Rapport** (2 h)
   - `reports/sprint_3/feature_selection_report.md` :
     - Top 15 features par actif.
     - Stabilité cross-actif (combien sont communs).
     - Hypothèses interprétatives (microstructure dominant ? session dominant ?).

### Critères d'acceptation
- ✅ IC matrix complète (60 features × 4 targets × 6 actifs).
- ✅ Top 15 features sélectionnées par actif.
- ✅ Au moins 5 features avec |IC| > 0.05 et p-corrected < 0.05.
- ✅ Visualisations + rapport.

### Findings audit adressés
- **P0-1** (ConfluenceDetector sans edge) — diagnostic empirique.

### Dépendances
- Batch 3.1 (features générées).

### Risques
- Si AUCUNE feature n'a |IC| > 0.05 → **GATE PIVOT précoce déclenché**. Sprint 3 ne peut pas converger sur PF. Pivot Sprint 3.5 vers calibration probabiliste only.

---

## Batch 3.3 — Stacking + conditionnement par régime (22 h)

### Objectif
Combiner les top 15 features par régime (HMM 3-state ou BOCPD) via stacking. Modèle linéaire L1 + RF + LightGBM blendés. Validation CPCV pour estimer edge réel.

### Steps
1. **Pipeline stacking** (4 h)
   - `src/intelligence/research/stacking.py` :
     - Level 0 : Logistic L1, Random Forest, LightGBM.
     - Level 1 : Logistic blending.
     - Conditionnement par régime (3 modèles séparés par régime).

2. **Régime canonique** (2 h)
   - `regime_filter.py` HMM 3-state (decision D Sprint 0).
   - Régimes : `low_vol_trending`, `high_vol_choppy`, `crisis`.

3. **CV temporelle stratifiée régime** (3 h)
   - Stratified K-Fold (5 splits) par régime.
   - Pas de leakage temporel (purged + embargoed).

4. **Training XAU M15** (4 h)
   - Top 15 features XAU.
   - 3 stacking models (1 par régime).
   - Target : `forward_return_5bars > 0`.
   - Output : probabilités prédites + AUC + Brier score.

5. **Training XAU H1 + EURUSD M15 + EURUSD H1** (5 h)
   - Idem, 3 autres configs.

6. **Backtest avec proba** (2 h)
   - Wire `stacking_proba` → state machine `enter_threshold`.
   - Cible : enter si `proba > 0.55`.
   - Quick backtest pour validation.

7. **Rapport stacking** (2 h)
   - `reports/sprint_3/stacking_report.md` :
     - AUC + Brier par config × régime.
     - Coefficients Logistic L1 (top features).
     - Comparaison vs ConfluenceDetector baseline.

### Critères d'acceptation
- ✅ Stacking entraîné sur 4 configs.
- ✅ AUC OOS > 0.55 sur ≥ 1 config (= edge stat positif).
- ✅ Brier skill > 0 (vs baseline `p=0.5`).
- ✅ Coefficients Logistic L1 interprétables (sparsité).

### Findings audit adressés
- **P0-1** (ConfluenceDetector sans edge) — remplace par stacking probabiliste.
- **P1-5** (OB ↔ Retest corrélés) — L1 force sparsité.

### Dépendances
- Batches 3.1 + 3.2.

### Risques
- Si AUC OOS ≤ 0.55 sur toutes configs → edge est faible. Possibilité : retenter avec features additionnelles ou pivoter Sprint 3.4 vers valeur explicative.

---

## Batch 3.4 — CPCV/DSR/PBO industrialisés (28 h) **PRIORITAIRE**

### Objectif
Wirer la machinerie de validation institutionnelle (CPCV + DSR + PBO) qui existe en R&D (`src/research/`) vers `src/backtest/validation/` et l'intégrer comme gate obligatoire pour tout signal en production. Référence : finding **P0-17** de l'audit.

### Steps
1. **Audit machinerie existante** (4 h)
   - Lire `src/research/cpcv_harness.py` (507 LOC) + `src/research/strategy_gates.py`.
   - Documenter API actuelle :
     - `CPCVHarness(returns, n_splits, k_test_groups, embargo)`.
     - `compute_dsr(sharpe_paths, n_trials)` (López de Prado AFML).
     - `compute_pbo(rank_matrix)` (probability of backtest overfitting).
     - `dm_test(returns_a, returns_b)` (Diebold-Mariano).
   - Sortie : `audits/2026-Q2/cpcv_machinery_audit.md`.

2. **Migration vers `src/backtest/validation/`** (4 h)
   - Créer `src/backtest/validation/__init__.py`.
   - Déplacer : `cpcv.py`, `dsr.py`, `pbo.py`, `dm_test.py`, `gates.py`.
   - Refactor imports legacy `src/research/` → backward-compat wrapper.
   - Tests migrés `tests/test_backtest_validation/`.

3. **Couplage `state_machine_replay.py`** (5 h)
   - Modifier `SignalReplay` pour exposer `paths_returns` (matrix `n_paths × n_bars`).
   - Wrapper `BacktestRunner` qui invoque CPCV après run :
     - `paths = cpcv_harness.split_and_predict(...)`.
     - `dsr = compute_dsr(paths)`.
     - `pbo = compute_pbo(rank_matrix)`.
     - `pf_lo, pf_hi = bootstrap_ci(returns, n=10_000)`.

4. **Coûts transactionnels wirés** (4 h)
   - Wire `DynamicSpreadModel` + `DynamicSlippageModel` (existants `src/environment/execution_model.py`) dans `SignalReplay`.
   - Commission paramétrable `InstrumentConfig.commission_per_trade`.
   - Test : XAU M15 baseline avec coûts → PF doit baisser de 5-15 %.

5. **Bugs métriques corrigés** (3 h)
   - Calmar annualisé (`metrics.py:254`).
   - Sharpe : utiliser `stdev` (sample) cohérent partout.
   - `max_consec_losses` exclure breakeven.
   - Annualisation : Lo 2002 autocorrelation correction.

6. **Reproductibilité bit-à-bit** (2 h)
   - Fix `uuid.uuid4()` → `uuid.uuid5(NAMESPACE, signal_key)` deterministe.
   - Test : 2 runs identiques produisent mêmes trades + mêmes SHA256.

7. **Gate `should_promote_to_prod(strategy)`** (3 h)
   - Fonction unifiée :
     - DSR ≥ 1.5 ✅
     - PBO ≤ 0.35 ✅
     - PF_lo > 1.0 (CI 95 % bootstrap) ✅
     - DM_p < 0.05 vs benchmark naïve ✅
     - All 4 → `should_promote = True`.
   - Sortie : `reports/sprint_3/strategy_gates.json` par config testée.

8. **Documentation + CHANGELOG** (3 h)
   - `docs/algo/backtest_validation.md` : guide d'usage CPCV + gates.

### Critères d'acceptation
- ✅ CPCV + DSR + PBO + DM exposés depuis `src/backtest/validation/`.
- ✅ Coûts transactionnels wirés.
- ✅ Reproductibilité bit-à-bit prouvée (test).
- ✅ Bugs métriques corrigés (Calmar, Sharpe, max_losses).
- ✅ Gate `should_promote_to_prod` testé sur 4 configs MVP.

### Findings audit adressés
- **P0-5** (Pas de walk-forward propre) — ✅ closed.
- **P0-6** (Coûts $0) — ✅ closed.
- **P0-17** (CPCV existe mais non couplée) — ✅ closed.
- Bugs métriques eval_18 — ✅ closed.

### Dépendances
- Sprint 0 batch 0.2 (baseline reproductible).
- Sprint 1 batch 1.1 (DataProvider contractuel).

### Risques
- CPCV cher computationnellement (10-30 min par run XAU 6 ans). Mitigation : cache résultats, parallélisation.
- Coûts wirés peuvent révéler que la baseline n'est plus profitable du tout. C'est OK — c'est ce qu'on cherche.

---

## Batch 3.5 — Sweep paramétrique state machine × 4 actifs × 4 TF (22 h)

### Objectif
Le sweep de 432 cellules pending depuis eval_07 (P0-12) sur 4 actifs × 4 TF avec hyperparams état machine + thresholds entry/exit. Décider per-config si edge réel via gates Sprint 3.4.

### Steps
1. **Grid définition** (2 h)
   - Hyperparams sweepés :
     - `enter_threshold` : [55, 60, 65, 70, 75].
     - `exit_threshold` : [40, 45, 50, 55].
     - `cooldown_bars` : [3, 5, 10, 15].
     - `confirm_bars` : [1, 2, 3].
     - `max_age_bars` : [10, 20, 30, 50].
     - `stacking_proba_thresh` : [0.50, 0.55, 0.60, 0.65].
   - Grid après filtering invalides : ~500 configs.

2. **Configs MVP** (1 h)
   - 4 actifs × 4 TF = 16 (XAU/EUR/BTC/US500/GBP/JPY × M15/H1/H4/D1, on garde 4 prioritaires : XAU M15, XAU H1, EUR M15, EUR H1).
   - Soit ~2 000 runs.

3. **Parallélisation** (3 h)
   - `scripts/sweep_state_machine.py` :
     - `joblib.Parallel(n_jobs=-1)`.
     - Cache par hash hyperparams.
     - Resume sur crash.

4. **Exécution sweep** (8 h compute, surveillance light)
   - ~2 000 runs × ~30 sec/run = ~16 h compute single-thread, ~2 h sur 8 cores.
   - Lance en background, batch monitoring 1×/h.

5. **Analyse résultats** (3 h)
   - Pour chaque config : PF, PF_lo CI95, Sharpe, DSR, PBO.
   - Filter `should_promote = True`.
   - Sortie : `reports/sprint_3/sweep_results.parquet`.

6. **Sélection finale per-config** (2 h)
   - Top config par actif/TF selon DSR puis PF_lo.
   - Sortie : `config/state_machine_optima.json`.

7. **Rapport gate** (3 h)
   - `reports/sprint_3/sweep_gate_report.md` :
     - Tableau : pour chaque actif/TF, best config + DSR + PF_lo CI95 + PBO.
     - Verdict : combien d'actifs passent le gate.
     - Si 0 actif passe → section "Pivot decision".

### Critères d'acceptation
- ✅ Sweep ~2 000 runs complet.
- ✅ Résultats persistés parquet.
- ✅ Best config par actif identifiée.
- ✅ Rapport gate signé.

### Findings audit adressés
- **P0-3** (0 trades avec defaults) — ✅ closed.
- **P0-12** (Defaults state machine non empiriques) — ✅ closed.
- **P1-10** (confirm_bars / max_age non paramétrés par TF) — ✅ closed.

### Dépendances
- Batches 3.3 + 3.4.

### Risques
- 2 000 runs × 30 sec = 16h compute. Si > 24h → réduire grid à 1 000.
- Si 0 actif passe gate → escalade pivot.

---

## Gate de sortie du Sprint 3 (checklist 14 items)

1. ✅ 60+ features générées et cachées.
2. ✅ IC matrix complète + top 15 par actif.
3. ✅ Au moins 5 features avec |IC| > 0.05 (p-corrected < 0.05).
4. ✅ Stacking entraîné 4 configs, AUC OOS > 0.55 sur ≥ 1.
5. ✅ CPCV + DSR + PBO + DM industrialisés `src/backtest/validation/`.
6. ✅ Coûts transactionnels wirés.
7. ✅ Reproductibilité bit-à-bit (test passing).
8. ✅ Bugs métriques fixés.
9. ✅ Sweep ~2 000 runs exécuté.
10. ✅ `config/state_machine_optima.json` signé.
11. **GATE BINAIRE** : ≥ 1 config passe DSR ≥ 1.5 + PBO ≤ 0.35 + PF_lo > 1.0 + DM_p < 0.05. **OU** décision pivot tranchée + documentée dans `roadmap/sprints/sprint_3_pivot_decision.md`.
12. ✅ Suite tests verte.
13. ✅ Doc `docs/algo/backtest_validation.md` + `features.md`.
14. ✅ `sprint_3_retrospective.md` rédigé.

---

## Livrables Sprint 3 (arborescence)

```
src/intelligence/features/
  ├── __init__.py
  ├── pipeline.py
  ├── microstructure.py
  ├── order_flow.py
  ├── temporal.py
  └── macro.py

src/intelligence/research/
  ├── targets.py
  └── stacking.py

src/backtest/validation/
  ├── __init__.py
  ├── cpcv.py                        # migré de src/research/
  ├── dsr.py
  ├── pbo.py
  ├── dm_test.py
  └── gates.py

config/
  ├── feature_selection.json
  └── state_machine_optima.json

data/features/
  └── {symbol}_{tf}_features.parquet (6 actifs × 1-4 TF)

scripts/
  ├── plot_feature_ic.py
  └── sweep_state_machine.py

tests/test_backtest_validation/
  ├── test_cpcv.py
  ├── test_dsr.py
  ├── test_pbo.py
  └── test_gates.py

reports/sprint_3/
  ├── ic_matrix.parquet
  ├── feature_selection_report.md
  ├── feature_ic_plots/
  ├── stacking_report.md
  ├── sweep_results.parquet
  ├── sweep_gate_report.md
  └── strategy_gates.json

docs/algo/
  ├── features.md
  └── backtest_validation.md

roadmap/sprints/
  ├── sprint_3.md
  ├── sprint_3_progress.md
  ├── sprint_3_retrospective.md
  └── sprint_3_pivot_decision.md (si gate fail)
```

---

## Décisions ouvertes pour user

1. **GATE BINAIRE** : si 0 actif passe → décision PIVOT tranchée en fin de Sprint 3. Options :
   - (a) Pivot valeur explicative (calibration probabiliste, IR par régime — Sprint 4-7 sur informational value).
   - (b) Pivot B2B-API (vendre les insights sans promesse PF — décision M2 roadmap).
   - (c) Reconnaître absence d'edge, arrêter projet (worst case).
2. **Budget compute CPCV** : si > 50 h compute Sprint 3 → louer EC2 GPU/CPU spot ~$50.
3. **Macro feeds** : si DXY/VIX/US10Y indisponibles localement, fallback Yahoo Finance OK ?

---

**Signé** : Claude, 2026-05-15
