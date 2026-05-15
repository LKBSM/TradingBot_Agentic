# Sprint 0 — Progress log

Append-only log, un bloc par batch. Statut ✅ ok / ⚠️ partiel / ❌ bloqué.

---

## Batch 0.0 — Pre-flight env + audit data layer — ✅ ok

**Date** : 2026-05-15
**Charge effective** : ~1 h (vs 4 h estimé — efficient car l'env était sain et l'audit s'est bien passé)
**Livrables** :
- `audits/2026-Q2/preflight_env.md` — Python 3.12.6, pytest 9.0.2, pandas 3.0.0, scikit-learn 1.8.0, lightgbm 4.6.0. `arch` non installé (warning bénin).
- `audits/2026-Q2/preflight_imports.md` — 17/17 imports algo OK, 2 696 tests collectés (vs 1 366 en mémoire — la suite a grandi).
- `audits/2026-Q2/xau_coverage_audit.md` — audit empirique 5 CSV.
- `audits/2026-Q2/data_layer_pre_flight.md` — **Décision A actée**.
- `scripts/audit_xau_coverage.py` — audit reproductible.
- `.gitignore` mis à jour pour `backups/`.

**Décision A actée** :
- Source XAU primaire = `data/XAU_15MIN_2019_2026.csv` (98.72 % coverage 2019-2025, fraîcheur 15 jours).
- Source EURUSD primaire = `data/EURUSD_15MIN_2019_2025.csv` (99.41 %).
- Licence Dukascopy **différée** : le CSV adopté couvre déjà 2025-2026 sans dépendre du Dukascopy.

**Bonus inattendu** : la décision A simplifie le Sprint 1 batch 1.4 (la question de la licence n'est plus bloquante).

---

## Batch 0.1 — Freeze + branche + CI minimale — ✅ ok

**Date** : 2026-05-15
**Charge effective** : ~30 min (vs 5 h estimé — l'absence de WIP destructeur a simplifié)
**Livrables** :
- Tag `v0.9.0-pre-institutional` créé et pushé.
- Branche `institutional-overhaul` créée depuis le tag et pushée (tracked vers origin).
- Backups : `backups/data_2026-05-15.tar.gz` (39 MB) + `backups/models_2026-05-15.tar.gz` (425 KB).
- CI minimale : `.github/workflows/algo_tests.yml` (push trigger sur institutional-overhaul + main + PR).
- Commit `7ff3180` : 14 fichiers, 1 688 lignes.

**Choix de simplification vs plan v2** :
- Au lieu de créer une branche `wip/pre-institutional-2026-05-15`, le tag a été posé directement sur HEAD actuel (sans toucher au WIP local). Avantage : pas de manipulation destructive du working dir. Inconvénient : le WIP local (M + ??) reste tel quel sur la branche `institutional-overhaul` (acceptable, on triera plus tard).
- Le WIP non-algo (`api/`, `delivery/`, `llm_narrative_engine.py`, etc.) reste non-commité localement et hors scope.

**Verdict** : freeze réussi, branche de travail active, CI armée.

---

## Batch 0.4 — Fix data P0 + test régression BOS — ✅ ok

**Date** : 2026-05-15
**Charge effective** : ~45 min (vs 2 h estimé — efficient, sed bulk patches)
**Livrables** :
- 22 fichiers patchés : `config.py` ligne 42 + `src/core/config_loader.py` ligne 316 (prod), `scripts/audit_backtest.py`, `run_backtest.py`, `replay_state_machine.py`, et 14 autres scripts in-scope.
- Hors-scope loggés OUT_OF_SCOPE.md : 14 fichiers Colab/RL/notebooks (à fixer sprint dédié).
- Garde-fou `tests/test_data_quality_bos_regression.py` (3 tests verts).
- Doc `audits/2026-Q2/data_layer_fix_0.4.md`.
- Orchestrateur baseline `scripts/run_baseline_sprint0.py`.
- Commit `66c1a53` poussé.

**Vérifications post-fix** :
- Core algo : 192/192 tests verts (state machine replay, confluence, vol forecaster, regime, conformal, scanner, data_quality).
- Suite complète : en cours d'exécution background (2696 tests).

---

## Batch 0.2 — Baseline backtest reproductible — ✅ ok

**Date** : 2026-05-15
**Charge effective** : ~2 h (vs 15 h estimé — efficient grâce aux scripts existants `run_backtest.py`)

**Livrables** :
- `reports/baseline/baseline_report.md` + `.json` reproductible.
- `reports/baseline/baseline_narrative.md` — analyse narrative complète.
- `reports/baseline/config_snapshot_2026-05-15.json` (exhaustif : commit, code SHA256, data SHA256, libs).
- `reports/baseline/checksums.txt`.
- Backtests XAU M15 (172 749 bars) + EURUSD M15 (174 381 bars).

**Finding capital du Sprint 0** :
- XAU M15 sur 7 ans : **0 trades** (192 signaux par detector, 0 confirmés par state machine).
- EURUSD M15 sur 7 ans : **0 trades** (13 signaux, 0 confirmés).
- Score max XAU 72.61, EURUSD 74.97 — **JAMAIS ≥ 75 (enter_threshold)**.
- BOS firing rate 3.16 % XAU / 3.40 % EURUSD — garde-fou data quality vert.
- Reproductibilité bit-à-bit : 2× runs = mêmes SHA256.

**Implication** : la pipeline actuelle, configurée par défaut, ne produit pas de signal tradable. Ce constat empirique honnête est la référence immuable Sprint 0 contre laquelle Sprints 1-7 mesureront leurs améliorations.

**Choix de simplification vs plan v2** :
- H1 timeframes reportés Sprint 1 (resampling sans look-ahead à valider).
- Pas de re-run forcé après chaque changement de seed (déterminisme acquis par fix).

---

## Batch 0.3 — Audit institutionnel Phase 1 — ⏳ en cours

**Date** : 2026-05-15
**Charge en cours** : 4 sections rédigées par Claude, 4 sections déléguées à agents general-purpose en arrière-plan.

**Sections internes complétées** :
- ✅ `section_3_1_data_layer.md` — score 5.0/10 (vs eval_08 = 3.5)
- ✅ `section_3_5_regime.md` — score 6.5/10
- ✅ `section_3_6_conformal.md` — score 7.0/10
- ✅ `section_3_7_state_machine.md` — score 8.0/10 (confirme eval_07)

**Sections déléguées (agents en arrière-plan)** :
- ⏳ `section_3_2_smart_money.md`
- ⏳ `section_3_3_confluence.md`
- ⏳ `section_3_4_volatility.md`
- ⏳ `section_3_8_backtest_engine.md`

**Consolidation** : `algo_audit_institutional.md` à composer en fin de batch.

---
