# Data Layer Fix — Batch 0.4

**Date** : 2026-05-15
**Branche** : `institutional-overhaul`
**Décision source** : `audits/2026-Q2/sprint_0_decisions.md` (Décision A)
**Audit empirique** : `audits/2026-Q2/xau_coverage_audit.md`

---

## Problème

Le CSV `data/XAU_15MIN_2019_2025.csv` est à **63.71 %** de coverage en session active (audit empirique batch 0.0). Ses gaps créent des discontinuités qui font que le détecteur `BOS_EVENT` croit voir un break à quasi chaque bar — d'où le bug "BOS firing sur 100 % des bars" mentionné dans `MEMORY.md` (xau_replay_findings_2026_04_23, data_quality_audit_2026_04_23).

Avant ce batch :
- `config.py:42` → `XAU_15MIN_2019_2024.csv` (97.6 %, mais figé fin 2024).
- `src/core/config_loader.py:316` → `XAU_15MIN_2019_2025.csv` (CSV cassé en fallback prod).
- 22 scripts d'analyse pointaient vers le CSV cassé.

---

## Action

Switch vers `data/XAU_15MIN_2019_2026.csv` (**98.72 %** coverage 2019-2025, fraîcheur 15 jours).

### Fichiers patchés (in-scope Sprint 0)

| Fichier                                  | Avant                       | Après                       | Critique |
| ---------------------------------------- | --------------------------- | --------------------------- | -------- |
| `config.py:42`                            | `XAU_15MIN_2019_2024.csv` | `XAU_15MIN_2019_2026.csv` | ✅ P0    |
| `src/core/config_loader.py:316`           | `XAU_15MIN_2019_2025.csv` ❌ | `XAU_15MIN_2019_2026.csv` | ✅ P0    |
| `scripts/audit_backtest.py:61`            | `XAU_15MIN_2019_2025.csv` ❌ | `XAU_15MIN_2019_2026.csv` | ✅ P0 (utilisé batch 0.2) |
| `scripts/run_backtest.py:201, doc`        | `XAU_15MIN_2019_2025.csv` ❌ | `XAU_15MIN_2019_2026.csv` | ✅       |
| `scripts/replay_state_machine.py:144, doc` | `XAU_15MIN_2019_2025.csv` ❌ | `XAU_15MIN_2019_2026.csv` | ✅       |
| `scripts/audit_data_quality.py:102`       | liste cassée → liste étendue | (ajout du 2019_2026)      | ✅       |
| `scripts/audit_subset_edge.py:62`         | `XAU_15MIN_2019_2025.csv` ❌ | `XAU_15MIN_2019_2026.csv` | ⚠️      |
| `scripts/sweep_sl_tp.py:113`              | idem                        | idem                       | ⚠️      |
| `scripts/audit_failure_mode.py:70`        | idem                        | idem                       | ⚠️      |
| `scripts/audit_filter_strategy.py:57`     | idem                        | idem                       | ⚠️      |
| `scripts/audit_feature_edge.py:11, 273`   | idem                        | idem                       | ⚠️      |
| `scripts/backtest_combo_E.py:89`          | idem                        | idem                       | ⚠️      |
| `scripts/backtest_filter_modes.py:54`     | idem                        | idem                       | ⚠️      |
| `scripts/backtest_with_filter.py:35`      | idem                        | idem                       | ⚠️      |
| `scripts/train_lgbm_v2.py:30`             | idem                        | idem                       | ⚠️      |
| `scripts/prepare_training_data.py:26`     | idem                        | idem                       | ⚠️      |
| `scripts/poc_kronos_volatility.py:42`     | idem                        | idem                       | ⚠️      |
| `scripts/download_economic_calendar.py:569, 577` | idem                 | idem                       | ⚠️      |
| `reports/eval_18_walkforward_skeleton.py` | idem                        | idem                       | ⚠️      |

### Fichiers laissés tels quels (hors scope Sprint 0 — loggés dans `OUT_OF_SCOPE.md`)

- `colab_setup.py`, `colab_*.py`, `notebooks/Colab_Full_Training_Script.py` (training Colab).
- `parallel_training.py`, `examples/agentic_trading_demo.py` (RL legacy).
- `scripts/download_xau_data.py` (downloader, OUTPUT_FILE = nom de sortie, pas critique).
- `scripts/audit_xau_coverage.py` (référence intentionnelle pour cross-check).
- `scripts/audit_data_quality.py` (référence intentionnelle pour cross-check).

---

## Garde-fou : `tests/test_data_quality_bos_regression.py`

3 tests créés :

1. `test_primary_csv_coverage_is_sufficient` — sanity : le CSV charge ≥ 1800 bars.
2. `test_BOS_EVENT_firing_rate_is_reasonable` — vérifie que `BOS_EVENT != 0` firing rate ∈ [0.5 %, 10 %] sur 2000 bars échantillon. Bloque toute régression vers un CSV à coverage faible (~100 % firing) ou une dérive logique vers la sur-strictesse (< 0.5 %).
3. `test_config_points_to_primary_csv` — vérifie que `config.HISTORICAL_DATA_FILE` pointe bien vers `XAU_15MIN_2019_2026.csv`.

**Résultat** : 3/3 verts (avant le fix, 2/3 rouge — preuve que le garde-fou fonctionne).

---

## Vérification non-régression

| Suite                          | Avant fix    | Après fix      |
| ------------------------------ | ------------ | -------------- |
| `test_state_machine_replay.py` | ✅           | ✅ 24/24       |
| `test_signal_state_machine.py` | ✅           | ✅ 54/54       |
| `test_confluence_detector.py`  | ✅           | ✅             |
| `test_volatility_forecaster.py` | ✅          | ✅             |
| `test_regime_classifier.py`    | ✅           | ✅             |
| `test_sentinel_scanner.py`     | ✅           | ✅             |
| `test_data_quality.py`         | ✅           | ✅             |
| `test_data_quality_bos_regression.py` | n/a   | ✅ 3/3 NEW    |

**Core algo : 192 / 192 verts en 20 s.** Suite complète en cours (résultat à venir).

---

## Impact sur Batch 0.2 (baseline)

La baseline batch 0.2 doit désormais utiliser :
- `data/XAU_15MIN_2019_2026.csv` (172 875 bars, 2019-01-02 → 2026-04-29).
- `data/EURUSD_15MIN_2019_2025.csv` (174 506 bars, 2019-01-01 → 2025-12-31, 99.41 %).

Les chiffres PF/Sharpe peuvent légèrement différer des baselines historiques (eval_*) qui utilisaient le CSV cassé. **Ce delta est attendu et signe de la correction**.

---

**Signé** : 2026-05-15, Claude
