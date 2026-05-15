# Sprint 0 — Stabilisation & Baseline (PLAN RÉVISÉ)

**Période** : 2026-05-15 → 2026-05-29 (2 semaines)
**Charge totale** : **66 h** productives (~33 h/sem)
**Objectif** : poser une baseline reproductible, figer le code, livrer l'audit institutionnel Phase 1, corriger les P0 data immédiats.
**Gate de sortie** : tous CSV MVP à coverage ≥ 95 %, baseline figée, audit Phase 1 signé.
**Décisions actées** : voir `audits/2026-Q2/sprint_0_decisions.md` (registre des 14 décisions tranchées A→N).

---

## 0. Vue d'ensemble — 5 batches

| Batch | Titre                                          | Heures | Critique chemin |
| ----- | ---------------------------------------------- | ------ | --------------- |
| 0.0   | Pre-flight env + audit data layer              | 4 h    | ✅              |
| 0.1   | Freeze de l'existant + branche dédiée + CI     | 5 h    | ✅              |
| 0.2   | Baseline backtest reproductible                | 15 h   | ✅              |
| 0.3   | Audit institutionnel Phase 1                   | 32 h   | ✅              |
| 0.4   | Correction P0 data + tests régression          | 2 h    | ✅              |
| —     | Buffer (imprévus, debug, reviews)              | 8 h    |                 |
| **TOTAL** |                                            | **66 h** |               |

Justification du re-séquencement (vs plan v1) : l'audit coverage XAU passe en batch 0.0, ce qui évite la re-run de la baseline en batch 0.4. Cf. décision N.

---

## Batch 0.0 — Pre-flight env + audit data layer (4 h)

### Objectif
Valider que l'environnement Python tourne, que les imports critiques chargent, et trancher la source XAU primaire avant tout autre travail.

### Steps

1. **Env audit** (30 min)
   - `python --version` + `pytest --version` + `pip freeze` filtré sur pandas/numpy/sklearn/lightgbm/hmmlearn.
   - Sauvegarde dans `audits/2026-Q2/preflight_env.md`.
   - **Déjà confirmé en pré-validation** : Python 3 + pytest 9.0.2 + pandas 3.0.0 + numpy 1.26.4 + sklearn 1.8.0. À écrire pour archive.

2. **Imports check** (30 min)
   - `python -c "from src.intelligence.sentinel_scanner import SentinelScanner"` etc. sur les 10 modules cœur.
   - `pytest --collect-only tests/` doit terminer sans erreur de chargement.
   - Liste des imports cassés dans `audits/2026-Q2/preflight_imports.md` (probablement vide).

3. **Audit coverage `XAU_15MIN_2019_2026.csv`** (2 h)
   - Script ad-hoc `scripts/audit_xau_coverage.py` (à créer) qui :
     - Charge le CSV (172 875 lignes).
     - Calcule bars attendus = ~98 bars/jour × ~250 jours/an × 7 ans ≈ 171 500 bars (cible).
     - Compte bars présents par an (2019, 2020, ..., 2026).
     - Détecte gaps > 30 min en session active (UTC 7h-21h Mon-Fri).
     - Compare à `XAU_15MIN_2019_2024.csv` (97.6 % connu) sur le sous-intervalle commun.
   - Sortie : `audits/2026-Q2/xau_coverage_audit.md` avec tableau coverage par an.

4. **Décision A finale** (15 min)
   - Si `2019_2026.csv` ≥ 95 % sur 2019-2025 ET frais sur 2026 → adopté primaire.
   - Sinon → fallback `2019_2024.csv` + extension Dukascopy (avec flag licence ouvert).
   - Décision tracée dans `audits/2026-Q2/data_layer_pre_flight.md`.

5. **Audit `.gitignore`** (15 min)
   - Confirmer que `data/*.csv` est ignoré (déjà vérifié).
   - Ajouter `backups/` à `.gitignore` si pas déjà présent.
   - Vérifier qu'aucun secret n'est track-able dans le repo.

6. **Audit EURUSD coverage** (30 min)
   - Quick check `EURUSD_15MIN_2019_2025.csv` (174 507 lignes, 99.6 % en mémoire) — confirmer ou nuancer.
   - Sortie ajoutée à `xau_coverage_audit.md`.

### Critères d'acceptation
- ✅ Env audit + imports check + decision A actée
- ✅ `audits/2026-Q2/preflight_env.md`, `preflight_imports.md`, `xau_coverage_audit.md`, `data_layer_pre_flight.md` créés
- ✅ Décision A actée dans le registre

### Livrables
```
audits/2026-Q2/
  ├── preflight_env.md
  ├── preflight_imports.md
  ├── xau_coverage_audit.md
  └── data_layer_pre_flight.md
scripts/audit_xau_coverage.py
```

### Dépendances
Aucune (premier batch).

### Risques
- `pytest --collect-only` peut échouer si un test fait un import lourd au chargement. → Marquer en `xfail` temporairement, documenter dans `tests/disabled_during_sprint_0.md`.
- Le CSV `2019_2026.csv` peut révéler des gaps importants → fallback B prêt.

---

## Batch 0.1 — Freeze + branche dédiée + CI minimale (5 h)

### Objectif
Geler l'état actuel sur un tag, créer la branche de travail, ajouter une CI minimale.

### Steps

1. **Préserver le WIP non-committé** (1 h) — décision J
   - `git checkout -b wip/pre-institutional-2026-05-15` (depuis l'état actuel `main` + WIP).
   - `git add -A && git commit -m "wip: snapshot pre-institutional avant freeze"`.
   - Cette branche reste accessible pour récupération.

2. **Retour main clean** (15 min)
   - `git checkout main`. Vérifier `git status` propre (tous les `M` et `??` ont basculé dans la branche wip).

3. **Backup local des données** (1 h)
   - `mkdir -p backups/data_2026-05-15 backups/models_2026-05-15`.
   - Copie : `cp data/*.csv data/macro/*.csv backups/data_2026-05-15/`.
   - Copie : `cp models/*.pkl models/*.lgb backups/models_2026-05-15/`.
   - Compression : `tar czf backups/data_2026-05-15.tar.gz backups/data_2026-05-15/ && rm -rf backups/data_2026-05-15/`.
   - Idem models.
   - **Ajout** : `backups/` dans `.gitignore`.

4. **Tag freeze** (15 min) — décision F
   - `git tag -a v0.9.0-pre-institutional -m "Pre-institutional baseline avant Sprint 0 — algo layer overhaul"`.
   - **Pas de push** du tag tant que la CI n'est pas en place (batch CI ci-dessous).

5. **Branche `institutional-overhaul`** (15 min)
   - `git checkout -b institutional-overhaul v0.9.0-pre-institutional`.
   - Toute la suite du sprint vit ici.

6. **CI minimale GitHub Actions** (1 h) — décision G
   - Création `.github/workflows/algo_tests.yml` :
     - Trigger : push sur `institutional-overhaul`, PR vers `main`.
     - OS : ubuntu-latest (pas Windows pour rester sur la free tier rapide).
     - Python 3.11 + cache pip.
     - `pip install -r requirements.txt`.
     - `pytest tests/ -m "not slow and not integration" --maxfail=10 -x` (pour ne pas exploser sur premiers échecs).
   - Si la suite a des tests Windows-only, marquer `@pytest.mark.skip_on_ci` (décision G).
   - Premier push pour valider que le workflow s'exécute.

7. **Init structure documentaire** (45 min)
   - `mkdir -p roadmap/sprints reports/baseline docs/algo agents`.
   - Templates vides pour `roadmap/sprints/sprint_0_progress.md` (rempli au fil des batches).
   - Update `CHANGELOG.md` section `[Unreleased]` avec les artefacts du batch.

8. **Commit + push** (15 min)
   - `git add -A && git commit -m "chore(sprint-0): freeze tag + CI minimale + structure docs"`.
   - `git push -u origin institutional-overhaul`.
   - `git push origin v0.9.0-pre-institutional`.

### Critères d'acceptation
- ✅ Branche `wip/pre-institutional-2026-05-15` créée et préserve le WIP
- ✅ Tag `v0.9.0-pre-institutional` créé sur main clean
- ✅ Branche `institutional-overhaul` active
- ✅ `.github/workflows/algo_tests.yml` créée, premier run vert (ou échecs documentés)
- ✅ Backups data + models compressés dans `backups/` (ignoré par git)
- ✅ `CHANGELOG.md` à jour
- ✅ `git status` clean

### Dépendances
- Batch 0.0 (imports check pour ne pas tag un état cassé).

### Risques
- Push d'un tag = visible publiquement. Si le repo est privé OK. **Pré-check** : `git config --get remote.origin.url` pour s'assurer du destinataire avant push.
- Premier run CI peut échouer sur des spécificités locales (paths Windows, fixtures absentes). → Documenter les échecs, ne pas fixer sauvagement en batch 0.1.

---

## Batch 0.2 — Baseline backtest reproductible (15 h)

### Objectif
Exécuter une baseline backtest sur 4 actifs/TF MVP (XAU M15, XAU H1, EURUSD M15, EURUSD H1) avec l'engine actuel, produire `reports/baseline/` figé et signé.

### Steps

1. **Audit engine actuel** (2 h)
   - Lecture `src/backtest/state_machine_replay.py` (914 LOC) + `scripts/audit_backtest.py` (993 LOC) + `scripts/comparatif_eurusd_m15.py` + `comparatif_xau_h1.py`.
   - Doc `audits/2026-Q2/backtest_engine_snapshot.md` :
     - Hypothèses (coûts, slippage, look-ahead).
     - Paramètres fixés vs sweepés.
     - Métriques calculées.

2. **Construction des datasets** (1 h)
   - XAU M15 = source primaire décidée batch 0.0.
   - EURUSD M15 = `EURUSD_15MIN_2019_2025.csv` (99.6 %).
   - XAU H1 + EURUSD H1 = resampling depuis M15 (script à valider absence look-ahead).
   - Sauvegarde des paths effectifs dans `reports/baseline/datasets_manifest.json`.

3. **Snapshot configuration** (2 h) — décision H
   - `scripts/snapshot_config.py` (à créer) :
     - Parse `config.py` → JSON.
     - Grep `os.getenv` / `os.environ` dans le périmètre algo → liste env vars consommées.
     - Hash SHA256 de tous les `.py` du périmètre (intelligence + backtest + environment features + agents algo-relevant).
     - Hash SHA256 des CSV utilisés.
     - `pip freeze` filtré → versions critiques.
     - `git rev-parse HEAD` → commit SHA.
   - Sortie : `reports/baseline/config_snapshot_2026-05-15.json`.

4. **Seed déterministe** (30 min)
   - Vérifier que `audit_backtest.py` et `state_machine_replay.py` initialisent `np.random.seed()`, `random.seed()`.
   - Si non → patcher avec un seed fixe `SEED=42` documenté.

5. **Exécution 4 backtests** (4 h, parallélisable Windows × 1, donc séquentiel ici)
   - XAU M15 : `python scripts/audit_backtest.py --asset XAU --tf M15 --start 2019-01-01 --end 2025-12-31 --output reports/baseline/xau_m15_*`.
   - XAU H1 : idem TF=H1.
   - EURUSD M15.
   - EURUSD H1.
   - Sortie par config : `_equity.csv`, `_trades.csv`, `_summary.json`.

6. **IC 95 % bootstrap** (2 h)
   - Script `scripts/baseline_bootstrap_ci.py` (à créer) :
     - Pour chaque (asset, tf), prendre les `_trades.csv`.
     - Bootstrap 10k re-samples des trade returns → distribution PF + Sharpe.
     - IC 95 % par percentile method + bias-corrected accelerated (BCa).
   - Sortie dans le `_summary.json` enrichi.

7. **Re-run reproductibility check** (1 h)
   - Relancer 1 backtest (XAU M15) avec la même seed.
   - Comparer SHA256 des `_equity.csv` et `_trades.csv` → doivent être identiques.
   - Si non → flag non-déterminisme, investigate, fix avant de signer la baseline.

8. **Checksums + signature** (30 min)
   - `sha256sum reports/baseline/*.csv reports/baseline/*.json > reports/baseline/checksums.txt`.
   - Append au `config_snapshot.json` : "baseline_signed_by": "Claude", "date": "2026-05-XX".

9. **Rapport baseline narratif** (2 h)
   - `reports/baseline/baseline_report.md` :
     - Tableau récap (asset, TF, PF, IC, Sharpe, max DD, nb trades).
     - Comparaison aux chiffres mémoire (XAU M15 v2 PF 1.04, EURUSD M15 PF 0.85, etc.) — flag tout écart.
     - Section "ce que la baseline ne dit pas" (pas de coûts réalistes, etc., préparant Sprint 3).
   - `reports/baseline/baseline_report.json` : structure machine-readable.

### Critères d'acceptation
- ✅ 4 backtests exécutés, sorties complètes par config
- ✅ IC 95 % bootstrap calculé sur chaque config
- ✅ Reproductibilité : 2e run XAU M15 = mêmes SHA256
- ✅ `baseline_report.md` + `.json` signés
- ✅ `config_snapshot_2026-05-15.json` exhaustif
- ✅ Cohérence ± 5 % vs chiffres mémoire (sinon section "Discrepancy" dans le rapport)

### Dépendances
- Batch 0.0 (source XAU décidée) + 0.1 (branche).

### Risques
- Resampling H1 depuis M15 peut introduire look-ahead silencieux. → Test sur 1 semaine isolée : bar H1 ne doit contenir aucune info > son timestamp.
- Engine actuel peut être non-déterministe (sentinel polling, threading). → Lancer **avec une seul thread** explicitement.
- Bootstrap sur < 50 trades = IC très larges, peu informatif. → Documenter dans le rapport, ne pas masquer.

---

## Batch 0.3 — Audit institutionnel Phase 1 (32 h)

### Objectif
Livrer `audits/2026-Q2/algo_audit_institutional.md` complet, 8 sections, scores 0-10, findings P0/P1/P2, recommandations actionnables.

### Découpe horaire

| Section                          | Temps | Sortie                                                          |
| -------------------------------- | ----- | --------------------------------------------------------------- |
| 3.1 — Audit data layer           | 4 h   | Coverage, fraîcheur, outliers, resampling, licence Dukascopy    |
| 3.2 — Audit Smart Money (code path) | 5 h | Reproductibilité BOS/CHOCH/OB/FVG + statistiques d'activation (sans annotations — décision I) |
| 3.3 — Audit ConfluenceDetector   | 4 h   | Justification poids, corrélation 8 composantes, monotonie       |
| 3.4 — Audit VolatilityForecaster | 4 h   | QLIKE, MSE log-vol, calibration bandes 50/80/95 %, cross-régime |
| 3.5 — Audit Régime stack         | 4 h   | 6 implémentations identifiées + acter canonique (décision D)    |
| 3.6 — Audit ConformalWrapper     | 3 h   | PICP / MPIW, exchangeabilité, gain réel                         |
| 3.7 — Audit SignalStateMachine   | 2 h   | Couverture transitions, chaos tests, cooldown/lockout           |
| 3.8 — Audit backtest engine      | 4 h   | Walk-forward / CPCV / DSR / PBO conformes ou pas                |
| Consolidation + synthèse         | 2 h   | Plan P0/P1/P2 + recommandations Sprints 1-7                     |

### Methodologie

- Chaque section produit un sous-fichier `audits/2026-Q2/section_3_X.md`, consolidé en fin de batch.
- Sources : code (`Read`/`Grep`), tests existants, rapports `reports/eval_*.md` (mémoire), baseline batch 0.2.
- **Pas de modification de code prod** pendant cette phase. Bugs → documentés, sévérité tagguée, pas fixés (sauf si P0 data → batch 0.4).
- Outils analytiques : scripts Python ad-hoc dans `scripts/audit_phase1/` (créés à la volée, conservés).
- Si une section nécessite des annotations expertes absentes (cas 3.2 ICT) → décision I : section limitée au code path + stats, le scoring F1 reporté Sprint 2.

### Critères d'acceptation
- ✅ 8 sous-fichiers `section_3_X.md` + 1 fichier consolidé `algo_audit_institutional.md`
- ✅ Chaque section : score 0-10 + findings + reco
- ✅ Consolidation : plan P0/P1/P2 priorisé pour Sprints 1-7
- ✅ Tous findings ancrés par référence `fichier:ligne` ou `reports/eval_XX.md`
- ✅ Format compatible avec le brief §3.1-3.8

### Dépendances
- Batch 0.2 (baseline disponible pour ancrer les chiffres).

### Risques
- Volume d'analyse important. → Si délai serré : livrer 5/8 sections complètes + 3 marquées "à compléter Sprint 1 batch 0.5". Mais cible = 8/8.
- Tentation de fixer pendant l'audit. **Interdit hors P0 data**.

---

## Batch 0.4 — Correction P0 data + tests régression (2 h)

### Objectif
Appliquer la décision A (source XAU primaire) au code et créer le test régression "BOS 100 % bars".

### Steps

1. **Fix `config.py`** (30 min)
   - Lecture ligne par ligne pour identifier `HISTORICAL_DATA_FILE`.
   - Switch vers la source décidée batch 0.0.
   - Commit dédié `fix(data): switch XAU primary source to <fichier> (coverage XX%)`.

2. **Test régression BOS** (45 min)
   - `tests/test_data_quality_bos_regression.py` :
     - Charge le CSV actuel via `data_providers`.
     - Lance le détecteur BOS sur 1 000 bars échantillon.
     - **Assertion** : firing rate BOS < 5 % (vs 100 % avec l'ancien CSV cassé).
     - Si on revient au mauvais CSV, le test tombe rouge → garde-fou structurel.

3. **Run suite complète** (30 min)
   - `pytest tests/ -x --maxfail=5`.
   - Tout doit rester vert. Si un test casse à cause du changement de CSV : fix immédiat OU revert + investigate.

4. **Doc** (15 min)
   - `audits/2026-Q2/data_layer_fix_0.4.md` : ce qui a changé, pourquoi, comment vérifier.
   - Update `CHANGELOG.md`.

### Critères d'acceptation
- ✅ `config.py` pointe vers CSV ≥ 95 % coverage
- ✅ `test_data_quality_bos_regression.py` créé et vert
- ✅ Suite complète verte (sauf flaky connu géré)
- ✅ Doc `data_layer_fix_0.4.md` créée

### Dépendances
- Batch 0.3 (l'audit a confirmé le P0).

### Risques
- Changer le CSV peut casser des fixtures qui référencent des timestamps précis. → Fixer les fixtures, pas le CSV.
- Le test BOS-regression peut être plus complexe à formuler proprement → s'aligner sur l'API actuelle de `strategy_features` + `data_providers`.

---

## Gate de sortie du Sprint 0

Tous ces critères doivent être verts pour démarrer Sprint 1 :

1. ✅ Tag `v0.9.0-pre-institutional` créé et pushé
2. ✅ Branche `institutional-overhaul` à jour avec tous les commits Sprint 0
3. ✅ CI minimale GitHub Actions verte sur dernier commit
4. ✅ `reports/baseline/baseline_report.{md,json}` reproductible (SHA256 stable)
5. ✅ `audits/2026-Q2/algo_audit_institutional.md` complet (8 sections, scores 0-10)
6. ✅ Plan P0/P1/P2 consolidé pour Sprints 1-7
7. ✅ Tous CSV MVP à coverage ≥ 95 %
8. ✅ Test régression `test_data_quality_bos_regression.py` vert
9. ✅ `CHANGELOG.md` à jour
10. ✅ `OUT_OF_SCOPE.md` enrichi des findings hors-périmètre découverts pendant Sprint 0
11. ✅ Suite complète verte (1 366+ tests, hors flaky connu)
12. ✅ `roadmap/sprints/sprint_0_retrospective.md` rédigé

---

## Livrables Sprint 0 — Arborescence cible

```
audits/2026-Q2/
  ├── sprint_0_decisions.md             (déjà fait — registre des décisions)
  ├── repo_inventory.md                  (déjà fait)
  ├── preflight_env.md                   (Batch 0.0)
  ├── preflight_imports.md               (Batch 0.0)
  ├── xau_coverage_audit.md              (Batch 0.0)
  ├── data_layer_pre_flight.md           (Batch 0.0)
  ├── backtest_engine_snapshot.md        (Batch 0.2)
  ├── algo_audit_institutional.md        (Batch 0.3 — consolidé)
  ├── section_3_1_data_layer.md          (Batch 0.3)
  ├── section_3_2_smart_money.md         (Batch 0.3)
  ├── section_3_3_confluence.md          (Batch 0.3)
  ├── section_3_4_volatility.md          (Batch 0.3)
  ├── section_3_5_regime.md              (Batch 0.3)
  ├── section_3_6_conformal.md           (Batch 0.3)
  ├── section_3_7_state_machine.md       (Batch 0.3)
  ├── section_3_8_backtest_engine.md     (Batch 0.3)
  └── data_layer_fix_0.4.md              (Batch 0.4)

reports/baseline/
  ├── baseline_report.md
  ├── baseline_report.json
  ├── datasets_manifest.json
  ├── config_snapshot_2026-05-15.json
  ├── checksums.txt
  ├── xau_m15_equity.csv
  ├── xau_m15_trades.csv
  ├── xau_m15_summary.json
  ├── xau_h1_equity.csv      (...idem...)
  ├── eurusd_m15_*.csv
  └── eurusd_h1_*.csv

roadmap/sprints/
  ├── sprint_0.md                        (ce fichier)
  ├── sprint_0_progress.md               (créé batch 0.1, append au fil de l'eau)
  └── sprint_0_retrospective.md          (fin de sprint)

scripts/
  ├── audit_xau_coverage.py              (Batch 0.0)
  ├── snapshot_config.py                 (Batch 0.2)
  ├── baseline_bootstrap_ci.py           (Batch 0.2)
  └── audit_phase1/                       (Batch 0.3, scripts ad-hoc)

tests/
  └── test_data_quality_bos_regression.py (Batch 0.4)

backups/
  ├── data_2026-05-15.tar.gz             (Batch 0.1, ignoré par git)
  └── models_2026-05-15.tar.gz

.github/workflows/
  └── algo_tests.yml                     (Batch 0.1)

(root)
  ├── MISSION_ACK.md                     (déjà fait)
  ├── OUT_OF_SCOPE.md                    (déjà fait)
  ├── CHANGELOG.md                       (déjà fait, mis à jour au fil)
  └── git tag: v0.9.0-pre-institutional  (Batch 0.1)

git branches:
  ├── main                                (intact, à jour avec v0.9.0)
  ├── wip/pre-institutional-2026-05-15    (préserve le WIP initial)
  └── institutional-overhaul              (branche de travail Sprint 0-7)
```

---

## Reporting fin de sprint (Gate)

À la clôture, je livre :

1. **`roadmap/sprints/sprint_0_retrospective.md`** :
   - Charge effective vs estimée par batch.
   - Gates passées / échouées.
   - Décisions ajustées (si l'audit a changé des choix tranchés a priori).
   - Backlog de dette identifiée pour les sprints suivants.
2. **Diff complet** : `git diff main..institutional-overhaul --stat` + lien aux commits.
3. **Premier signal au user** : "Sprint 0 clos, X gates passées, demande validation pour Sprint 1".

---

**Plan signé** : 2026-05-15, Claude (Lead Quant Architect)
**Statut** : décisions tranchées, exécution autorisée par l'auto-mode + instruction explicite du user. Démarrage Batch 0.0 dès que ce document est posé.
