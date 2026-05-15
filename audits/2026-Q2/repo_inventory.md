# Repo Inventory — Algo Layer Snapshot

**Date** : 2026-05-15
**Périmètre** : couche algorithmique uniquement (LLM / API / delivery / infra exclus)
**HEAD** : `203d189` (Phase 2B residual code)
**Branche** : `main`

---

## 0. Totaux quantitatifs

| Couche                     | Fichiers prod | LOC prod | Fichiers test | LOC test  | Ratio test/prod |
| -------------------------- | ------------- | -------- | ------------- | --------- | --------------- |
| `src/intelligence/` (algo) | 27            | 11 350   | 14            | ~3 500    | 0.31            |
| `src/backtest/`            | 4             | 1 714    | 4             | ~1 100    | 0.64            |
| `src/environment/` (features) | 6          | 5 431    | ~30           | ~7 400    | 1.36            |
| `src/agents/` (algo-relevant) | 4          | 3 567    | ~15           | ~2 700    | 0.76            |
| **TOTAL ALGO**             | **41**        | **22 062** | **~63**     | **~14 700** | **0.67**      |

**Config** : `config.py` = 850 LOC (instrument registry, paths, hyperparams).
**Données** : 7 CSV OHLCV/calendrier, 6 CSV macro FRED.

---

## 1. Inventaire `src/intelligence/` (couche cœur algo)

### 1.1 Détection Smart Money / Setup

| Fichier                       | LOC   | Rôle                                                    | Tests           | Dette visible                       |
| ----------------------------- | ----- | ------------------------------------------------------- | --------------- | ----------------------------------- |
| `confluence_detector.py`      | 637   | Scoring 0-100 (8 composantes pondérées)                 | ✅ 2 fichiers   | Aucune                              |
| `score_calibration.py`        | 190   | Platt scaling, isotonic regression                      | ✅              | Aucune                              |
| `stylized_facts.py`           | 250   | Validation faits stylisés (fat tails, vol clustering)   | partiel         | Aucune                              |
| `cross_asset_correlation.py`  | 159   | Corrélation multi-actifs                                | ❌              | Aucune                              |
| `feature_explainer.py`        | 165   | Attribution de signal (Shapley-like)                    | partiel         | Aucune                              |

🚨 **Gap critique** : **aucun module `smart_money_engine/`**. La logique BOS/CHOCH/OB/FVG/retest est dispersée :
- `src/environment/strategy_features.py` (1 213 LOC) → indicateurs techniques + OB/FVG/CHOCH
- `confluence_detector.py` → consomme les flags
- `agents/` → composantes orchestrées

➡️ Sprint 1 doit **extraire un module unifié `src/intelligence/smart_money/`** (BOS, CHOCH, OB ICT, FVG, retest) avec API contractuelle Pydantic v2.

---

### 1.2 Volatility forecasting

| Fichier                       | LOC   | Rôle                                          | Tests | Type hints | Dette                |
| ----------------------------- | ----- | --------------------------------------------- | ----- | ---------- | -------------------- |
| `volatility_forecaster.py`    | 1 559 | HAR-RV + diurnal + calendar + HMM blend       | ✅ ×3 | 61 %       | 3 lignes > 100 chars |
| `volatility_lgbm.py`          | 591   | LightGBM meta-learner                         | ✅    | 50 %       | Typage faible        |

**Note empirique (eval_04, 2026-04-29)** : LGBM/Hybrid latence 1.6-5 s/forecast vs cible 50 ms. `VOL_MODE=har` est le défaut acté.

---

### 1.3 Régime stack (4 modules concurrents)

| Fichier                  | LOC | Rôle                                              | Tests | Dette                    |
| ------------------------ | --- | ------------------------------------------------- | ----- | ------------------------ |
| `regime_classifier.py`   | 210 | HMM low/normal/high vol                           | ✅    | Aucune                   |
| `regime_filter.py`       | 189 | Filtrage signal par régime                        | ✅    | Aucune                   |
| `regime_gate.py`         | 335 | Gate conditionnel régime + BOCPD                  | ✅    | Aucune                   |
| `bocpd.py`               | 275 | Bayesian Online Changepoint Detection             | ✅    | Aucune                   |

🚨 **Risque** : 4 modules régime côté intelligence + `market_regime_agent.py` (887 LOC) + `regime_predictor.py` (1 051 LOC) côté `agents/`. **Au moins 6 implémentations parallèles de logique régime**. Sprint 0 audit doit identifier la canonique.

---

### 1.4 Conformal prediction

| Fichier                  | LOC | Rôle                                              | Tests | Dette  |
| ------------------------ | --- | ------------------------------------------------- | ----- | ------ |
| `conformal_wrapper.py`   | 384 | TCP (Transductive Conformal Prediction)           | ✅    | Aucune |

**Note A1 verdict (2026-05-01)** : sur stack actuelle, conformal rejette tout (correct sur weak edge).

---

### 1.5 Signal trust layer

| Fichier                       | LOC | Rôle                                                       | Tests        | Dette                  |
| ----------------------------- | --- | ---------------------------------------------------------- | ------------ | ---------------------- |
| `signal_state_machine.py`     | 922 | HOLD/BUY/SELL hystérésis, cooldown, lifetime, lockout      | ✅ ×4        | 1 TODO mineur, types 64 % |
| `state_persistence.py`        | 137 | Persistence JSON atomique + staleness guard                | ✅           | Aucune                 |
| `circuit_breaker.py`          | 231 | Pattern circuit breaker                                    | ✅           | Aucune                 |
| `notification_queue.py`       | 267 | File d'événements                                          | partiel      | Aucune                 |
| `semantic_cache.py`           | 251 | Cache hash (PAS sémantique malgré le nom)                  | ✅           | Hit rate prod 7.8 %    |

**Note eval_06** : cache hit rate empirique 7.8 % (vs 30-45 % estimé). Bump `SCORE_BUCKET_PTS 5→10` = ×4.3 hit déjà identifié.

---

### 1.6 Orchestration scanner

| Fichier                  | LOC   | Rôle                                                         | Tests | Dette                                |
| ------------------------ | ----- | ------------------------------------------------------------ | ----- | ------------------------------------ |
| `sentinel_scanner.py`    | 1 032 | Boucle scan tick-level (9 imports internes)                  | ✅ ×2 | Polling 60 s (SLA 30 s inatteignable) |
| `main.py`                | 696   | Entry point intelligence (`python -m src.intelligence.main`) | ❌    | Aucune                               |

---

### 1.7 Data layer

| Fichier                  | LOC | Rôle                                              | Tests | Dette                              |
| ------------------------ | --- | ------------------------------------------------- | ----- | ---------------------------------- |
| `data_providers.py`      | 375 | CSVDataProvider, MT5DataProvider                  | ✅    | Aucune                             |
| `data_quality.py`        | 191 | Validation OHLCV, gap detection, outliers         | ✅    | Aucune                             |
| `forward_test_paper.py`  | 276 | Walk-forward paper trading                        | partiel | Pas d'IC bootstrap                |

🚨 **Coverage CSV** (eval_08) : 5/6 presets sans CSV propre. Voir §5 ci-dessous.

---

### 1.8 Sécurité (algo seulement — pas auth)

| Fichier        | LOC | Rôle                              | Tests | Dette  |
| -------------- | --- | --------------------------------- | ----- | ------ |
| `security.py`  | 281 | Vérification cryptographique      | ✅    | Aucune |

---

### 1.9 EXCLUS de cet inventaire (LLM/Narrative)

- `llm_narrative_engine.py` (593 LOC)
- `template_narrative_engine.py` (468 LOC)
- `prompt_registry.py` (252 LOC)
- `narrative_quality.py` (176 LOC)
- `llm_cost_policy.py` (166 LOC)
- `regime_viz.py` (92 LOC) — visualisation uniquement
- `rag/` (sous-dossier complet)

---

## 2. Inventaire `src/backtest/`

| Fichier                       | LOC | Rôle                                                       | Tests | Dette                          |
| ----------------------------- | --- | ---------------------------------------------------------- | ----- | ------------------------------ |
| `state_machine_replay.py`     | 914 | Replay déterministe state machine sur historique           | ✅    | 1 `print()` ligne 390          |
| `metrics.py`                  | 383 | Sharpe, Sortino, drawdown, Kelly, PnL                      | ✅    | Aucune                         |
| `report.py`                   | 225 | Export HTML/JSON/CSV                                       | ✅    | Aucune                         |
| `news_replay.py`              | 192 | Replay impact news sur signaux                             | ✅    | Aucune                         |

🚨 **Backtest 2/10 (eval_18)** : aucun walk-forward propre, coûts transaction $0, look-ahead MTF possible, pas d'IC bootstrap. À rebâtir Sprint 3.

---

## 3. Inventaire `src/environment/` (features marché uniquement)

| Fichier                          | LOC   | Rôle                                                          | Tests        | Dette                       |
| -------------------------------- | ----- | ------------------------------------------------------------- | ------------ | --------------------------- |
| `environment.py`                 | 2 423 | Gym env RL (legacy mais hôte des features)                    | ✅ ×60+      | Gros fichier, typage 74 %   |
| `strategy_features.py`           | 1 213 | Indicateurs techniques + OB/FVG/CHOCH (cœur smart money réel) | ✅           | Aucune                      |
| `multi_timeframe_features.py`    | 667   | Aggregation M15/H1/H4/D1                                      | ✅           | Aucune                      |
| `risk_manager.py`                | 685   | Kelly, SL/TP, marge, liquidation                              | ✅           | 1 TODO ligne 682            |
| `feature_reducer.py`             | 325   | PCA / corrélation                                             | ✅           | Aucune                      |
| `execution_model.py`             | 118   | Slippage, spread, latence                                     | ✅           | Aucune                      |

🚨 **Décision Sprint 0** : `risk_manager.py` (685 LOC) est utilisé par le RL legacy. eval_19 (4.5/10) note **3 moteurs risk concurrents incohérents**. Question : oracle de backtest unique, OU refonte unifiée dans `src/intelligence/risk/` ?

---

## 4. Inventaire `src/agents/` (uniquement parties algo-relevant)

| Fichier                              | LOC   | Rôle                                                | Tests        | Dette         |
| ------------------------------------ | ----- | --------------------------------------------------- | ------------ | ------------- |
| `market_regime_agent.py`             | 887   | Détection régime technique (trend / vol / S-R)      | ✅ indirect  | Aucune        |
| `regime_predictor.py`                | 1 051 | HMM + regime switching                              | ✅           | Types 60 %    |
| `multi_timeframe.py`                 | 1 462 | Aggregation MTF signal-level                        | ✅           | Types 70 %    |
| `data/multi_instrument_quality.py`   | 167   | Monitoring qualité data multi-instrument            | ✅           | Aucune        |

**Tous les autres `src/agents/`** (~36 fichiers, ~20k LOC) sont **hors périmètre algo** (LLM, narrative, news scraping, scheduling, etc.).

---

## 5. Données OHLCV et calendrier

### 5.1 CSV principaux (`data/`)

| Fichier                              | Lignes  | Taille  | Période     | Coverage    | Statut MVP                     |
| ------------------------------------ | ------- | ------- | ----------- | ----------- | ------------------------------ |
| `XAU_15MIN_2019_2024.csv`            | 141 525 | 7.6 MB  | 2019-2024   | 97.6 %      | ✅ Recommandé                  |
| `XAU_15MIN_2019_2025.csv`            | 106 644 | 5.9 MB  | 2019-2025   | **63 %** ⚠️ | ❌ Bug data quality            |
| `XAU_15MIN_2019_2026.csv`            | 172 875 | 11 MB   | 2019-2026   | À vérifier  | ❓ À auditer Sprint 0          |
| `XAU_15MIN_2025_2026_dukascopy.csv`  | 31 443  | 2.6 MB  | 2025-2026   | À vérifier  | ❓ Dukascopy zone grise        |
| `EURUSD_15MIN_2019_2025.csv`         | 174 507 | 16 MB   | 2019-2025   | 99.6 %      | ✅ OK                          |
| `economic_calendar_2019_2025.csv`    | 1 380   | 71 KB   | 2019-2025   | All impacts | ✅ FF JSON validé              |
| `economic_calendar_HIGH_IMPACT_2019_2025.csv` | 876 | 44 KB | 2019-2025 | High only   | ✅ Filtré                      |

### 5.2 Macro FRED (`data/macro/`)

`cot_gold.csv`, `fred_DGS10.csv`, `fred_DFII10.csv`, `fred_DTWEXBGS.csv`, `fred_VIXCLS.csv`, `fred_T10Y2Y.csv`, `fred_BREAKEVEN_10Y.csv`.

### 5.3 Gaps identifiés

- **BTC, US500, GBPUSD, USDJPY** : aucun CSV M15 multi-an. Sprint 1 batch 1.5 doit télécharger (ou licencier).
- **XAU H1, EURUSD H1** : pas de CSV dédié (resampling à valider sans look-ahead).
- **Dukascopy** : licence commerciale ambiguë. À trancher Sprint 1 batch 1.4.

---

## 6. Scripts (`scripts/`) algo-relevant

### 6.1 Évaluations (`eval_*.py`)

`eval_02_confluence.py`, `eval_03_smart_money.py`, `eval_04_volatility.py`, `eval_04_footprint.py`, `eval_06_hit_rate_sim.py`, `eval_07_hysteresis_heatmap.py`, `eval_07_state_machine_sweep.py`, `eval_event_driven_macro.py`.

### 6.2 Backtest scripts

`audit_backtest.py` (993 LOC — sweep 7 ans), `backtest_combo_E.py`, `backtest_filter_modes.py`, `backtest_with_filter.py`.

### 6.3 Audits

`audit_data_quality.py`, `audit_feature_edge.py`, `audit_failure_mode.py`, `audit_filter_strategy.py`, `audit_subset_edge.py`, `audit_subset_eurusd.py`, `audit_ledger_snapshot.py`.

### 6.4 Téléchargement données

`download_economic_calendar.py` (579 LOC), `download_real_economic_data.py` (635 LOC), `download_xau_data.py`, `download_dukascopy_xau.py`, `fetch_forexfactory_live.py`, `crosscheck_mt5_calendar.py`, `export_mt5_history.py`.

### 6.5 Colab POC (preuves de concept entraînement)

`colab_har_rv_poc.py`, `colab_lgbm_vol_poc.py` (931 LOC), `colab_hybrid_vol_poc.py` (1 091 LOC), `colab_egarch_tcp_poc.py`, `colab_kronos_poc.py`, `colab_training_full.py`.

### 6.6 Falsification quant

`falsification_2026_04_30.py`, `falsification_complement.py`, `baseline_nr4_2026_04_30.py`.

---

## 7. Rapports existants pertinents

Tous dans `reports/`. Indexés en mémoire dans `MEMORY.md` (29 prompts d'éval). Highlights pour audit Phase 1 :

- `reports/eval_00_synthesis.md` — synthèse maître 29 notes (lecture obligatoire).
- `reports/eval_02_confluence.md` — score ConfluenceDetector PAS prédictif (Pearson −0.023).
- `reports/eval_03_smart_money.md` — détecteurs ICT/SMC 4.5/10, OB ≠ ICT correct.
- `reports/eval_04_volatility_findings.md` — verdict latence LGBM/Hybrid hors cible.
- `reports/eval_07_signal_state_machine.md` — code 8.0/10 mais defaults non empiriques.
- `reports/eval_08_data_providers.md` — 3.5/10 ❌ NO-GO commercial.
- `reports/eval_18_backtest.md` — 2/10 ❌ NON commercialisable.
- `reports/audit_2026_04_30_quant_senior.md` — falsification 60-70 % prob PF > 1.20.
- `reports/a1_verdict_2026.md` — verdict A1 DSR = 0.000, PBO = 0.50.
- `reports/3_pillars_implementation_2026_05_13.md` — gates fail sur 329 trades.
- `reports/institutional_quant_transformation_plan.md` — plan 250 h, 3 piliers.

---

## 8. Top 10 fichiers à dette technique (priorisé)

| # | Fichier                          | LOC   | Dette                                                | Prio |
| - | -------------------------------- | ----- | ---------------------------------------------------- | ---- |
| 1 | `environment.py`                 | 2 423 | Gros monolithe RL, doit être figé / encapsulé        | P1   |
| 2 | `multi_timeframe.py` (agents)    | 1 462 | 12 classes, types 70 % — duplication avec env       | P1   |
| 3 | `volatility_forecaster.py`       | 1 559 | Types 61 % + 3 long lines                            | P2   |
| 4 | `sentinel_scanner.py`            | 1 032 | 9 deps internes + polling 60 s (SLA 30 s impossible) | P0   |
| 5 | `regime_predictor.py`            | 1 051 | Types 60 %, duplication régime stack                 | P1   |
| 6 | `state_machine_replay.py`        | 914   | 1 `print()` debug                                    | P3   |
| 7 | `signal_state_machine.py`        | 922   | 1 TODO + types 64 %                                  | P3   |
| 8 | `confluence_detector.py`         | 637   | Score sans pouvoir prédictif (eval_02)               | P0   |
| 9 | `data_providers.py`              | 375   | Pas de contrat Pydantic v2                           | P1   |
| 10 | `volatility_lgbm.py`             | 591   | Types 50 %                                           | P2   |

**Légende prio** :
- **P0** : bloque go-live, à corriger Sprint 1-3
- **P1** : impact qualité fort, Sprint 4-5
- **P2** : confort / typage, Sprint 6
- **P3** : trivial, en pass-through

---

## 9. État des tests

- **1 366+ tests** au total (memoire).
- **0 GitHub Actions** (eval_17) — pas de CI automatisée.
- **Coverage** : ligne ~ inconnu (à mesurer Sprint 0 batch 0.2), branche jamais mesurée, mutation jamais lancée.
- **Connu cassé** : `tests/test_long_short_trading.py` (import) — déjà supprimé du working tree (statut `D` git).
- **Flaky connu** : `test_short_roundtrip_pnl` (passe isolé, échoue en suite).
- **Tests d'auth** patchent `TESTING_MODE=False` (hors périmètre algo).

---

## 10. Findings critiques pour l'audit Phase 1

| # | Finding                                                                            | Source            | Impact             |
| - | ---------------------------------------------------------------------------------- | ----------------- | ------------------ |
| F1 | **Aucun `smart_money_engine/` unifié** — logique éparpillée                        | inventaire        | Bloque audit ICT   |
| F2 | **6 implémentations parallèles** de régime stack                                   | inventaire        | Bloque audit régime |
| F3 | **CSV XAU 2019-2025 à 63 % coverage** mais référencé dans config                   | eval_08           | Bug data quality   |
| F4 | **ConfluenceDetector** : score sans pouvoir prédictif (Pearson −0.023)             | eval_02           | Refonte requise    |
| F5 | **Backtest 2/10** : pas de walk-forward, coûts $0, look-ahead MTF                  | eval_18           | Refonte engine     |
| F6 | **3 moteurs risk concurrents** dans le repo                                        | eval_19           | Unifier ou figer   |
| F7 | **Dukascopy** licence zone grise → bloquant commercial                             | eval_08           | Sprint 1 décision  |
| F8 | **Pas de CI / GitHub Actions**                                                     | eval_17           | Build infra Sprint 0 |
| F9 | **Sentinel polling 60 s** (SLA 30 s inatteignable)                                 | eval_09           | Sprint 6 perf      |
| F10 | **5/6 presets sans CSV propre** (BTC, US500, GBP, JPY, USOIL)                     | eval_20           | Sprint 1 data      |

---

## 11. Conclusion inventaire

L'algo layer est **conséquente (22k LOC prod)** et **outillée (14.7k LOC tests + 29 rapports d'éval)**, mais souffre de :

1. **Fragmentation** : smart money éparpillé, régime stack à 6 implémentations, 3 risk engines.
2. **Validation statistique faible** : pas de CPCV bit-à-bit reproductible, pas de DSR/PBO automatisé, walk-forward absent du backtest principal.
3. **Data quality non garantie** : 1 CSV référencé à 63 % de coverage, 5/6 presets vides, licence Dukascopy ambiguë.
4. **Pas de CI** ni de couverture mesurée systématiquement.

➡️ Le **Sprint 0** doit poser la baseline reproductible et flaguer ces zones avant tout build.

---

**Inventaire signé** : Claude (Lead Quant Architect par défaut)
**Date** : 2026-05-15
**Prochaine étape** : validation du plan Sprint 0 par le user.
