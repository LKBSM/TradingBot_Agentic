# Charter — Data Quality Engineer

**Slug** : `data_quality_engineer`
**Date** : 2026-05-15
**Sponsor** : Lead Quant Architect

## 1. Mission
Garantir que toutes les données consommées par le pipeline (OHLCV multi-actif/multi-TF, calendrier économique, sentiment) sont propres, contractualisées, et exemptes de look-ahead. Le rôle remplace l'historique "BOS firing on 100 % of bars" (eval_08) par un data layer institutionnel : contrat Pydantic v2, coverage prouvée ≥ 99 %, resampling MTF prouvé sans fuite, calendrier économique frais et versionné. Owner de `src/intelligence/data_providers.py`, du downloader Dukascopy, et du pipeline ForexFactory.

## 2. Périmètre
- **Inclus** :
  - CSV OHLCV : XAU, EURUSD, BTCUSD, US500, GBPUSD, USDJPY, USOIL (presets MVP).
  - Calendrier économique (`data/macro/`, ForexFactory JSON, FRED).
  - `DataProvider` API uniforme, contrats Pydantic v2 (Bar, Calendar Event, Tick).
  - Resampling MTF (`src/environment/multi_timeframe_features.py` + `src/intelligence/`).
  - Downloader Dukascopy (`scripts/download_dukascopy_xau.py`).
  - Test régression `test_data_quality_bos_regression.py` (existant) + extensions.
- **Exclu** :
  - Logique smart money / ICT (SMC Lead).
  - Volatility forecasting (Vol Modeler).
  - News sentiment downstream LLM (LLM eval suite hors périmètre Sprint 1-7).
  - Décision licence commerciale Polygon vs Databento (différée à Sprint 1.4 par décision A, arbitrée par LQA).

## 3. KPI principal et métriques
- **KPI** : coverage ≥ 99 % sur tous CSV MVP en session active (UTC 7h-21h Mon-Fri).
- **Sous-métriques** :
  - Coverage par actif : XAU (actuel 98.72 % → cible 99 %), EURUSD (99.41 % → 99.5 %), BTCUSD/US500/GBPUSD/USDJPY/USOIL (actuel <50 % ou inexistant → cible 99 %).
  - Nombre de gaps > 30 min en session active (cible : 0 non documenté).
  - Test property-based MTF look-ahead : 0 fuite sur 10⁵ tirages.
  - Fraîcheur calendrier économique : ≤ 7 jours d'écart vs date courante.
  - Conformité Pydantic v2 : 100 % des inputs `DataProvider` validés au boot.
- **Cadence de mesure** : à chaque commit data + recalc en début de sprint.

## 4. RACI sur les sprints

| Sprint | Responsable | Accountable | Consulted | Informed |
| ------ | ----------- | ----------- | --------- | -------- |
| Sprint 1 | Data Quality | LQA | SMC, Backtest Infra, QA | Vol, Regime |
| Sprint 2 | Data Quality (support) | LQA | SMC | Tous |
| Sprint 3 | — (frozen) | LQA | — | — |
| Sprint 4 | — | LQA | — | — |
| Sprint 5 | Data Quality (fuzz) | LQA | QA | — |
| Sprint 6 | Data Quality (versioning) | LQA | LQA | — |
| Sprint 7 | Data Quality (e2e) | LQA | LQA | — |

## 5. Findings prioritaires de l'audit Phase 1 (P0/P1)
- **P0-7** — Look-ahead MTF latent (`multi_timeframe_features.py:269` utilise `<=` au lieu de `<`). Fix Sprint 1.2.
- **P0-8** — Pas de contrat Pydantic v2 à l'ingestion data. Mise en place Sprint 1.1.
- **P0-14** — 5 / 6 presets sans CSV propre (BTC, US500, GBP, JPY, USOIL). Extension Sprint 1.5.
- **P1-13** — Réconciliation MTF intelligence/ vs environment/ (2 implémentations). Arbitrage LQA Sprint 1.
- **P1-14** — Calendrier économique fraîcheur 2025-12-31 (pas 2026). Refresh Sprint 1.3.
- Référence : décision A (XAU 2019_2026 adopté, 98.72 %) — actée Sprint 0.

(Liens : [audit Phase 1 §3.1](../../audits/2026-Q2/section_3_1_data_layer.md), [coverage audit](../../audits/2026-Q2/xau_coverage_audit.md))

## 6. Inputs / Outputs
- **Inputs** :
  - CSV bruts Dukascopy / fournisseurs.
  - ForexFactory JSON (`scripts/fetch_forexfactory_live.py`).
  - FRED API (sub-set macro).
  - Specs Pydantic v2 (issues de mémoire codebase `src/api/models.py`).
- **Outputs** :
  - `data/XAU_15MIN_2019_2026.csv` + autres presets MVP (versionnés via SHA256 dans `reports/baseline/`).
  - `src/intelligence/data_providers.py` (contrat Pydantic v2).
  - `src/intelligence/data_models.py` (Bar, Event, Tick).
  - `tests/test_data_provider_contract.py` (property-based).
  - `tests/test_mtf_no_lookahead.py` (property-based).
  - `audits/2026-Q*/data_coverage_<date>.md` à chaque sprint.

## 7. Critères de "done"
- Tous les CSV MVP coverage ≥ 99 % en session active.
- `pytest tests/test_data_provider_contract.py tests/test_mtf_no_lookahead.py tests/test_data_quality_bos_regression.py` vert sur CI.
- Calendrier économique daté ≤ 7 j.
- Tout signal généré pipeline référence un `data_version` traçable au commit + SHA256 CSV.
- Aucun fichier CSV à coverage < 95 % en source primaire dans `config.py`.
