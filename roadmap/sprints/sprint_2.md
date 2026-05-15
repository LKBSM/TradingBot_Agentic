# Sprint 2 — Detection Engine Validation

**Période** : Semaines 5-6 (S5-S6, ~2026-06-13 → 2026-06-27)
**Charge estimée totale** : **128 h** productives + buffer 16 h = 144 h
**Objectif** : sortir la détection ICT/SMC de l'auto-évaluation ("ça déclenche") vers une validation empirique (F1/Precision/Recall vs annotations expertes). Construire un dataset annoté ≥ 500 setups par actif, mesurer la précision réelle de chaque détecteur (BOS, CHOCH, OB, FVG, retest), tuner les hyperparamètres par actif via Bayesian optimization, et automatiser l'audit visuel par snapshots PNG.
**Gate de sortie** : F1 ≥ 0.65 sur ≥ 2 actifs pour BOS+OB, audit visuel reproductible, tuning bayésien convergé, gap honnête documenté entre détection actuelle et "ICT pure".

---

## 0. Vue d'ensemble — 4 batches

| Batch | Titre                                                    | Heures | Critique chemin |
| ----- | -------------------------------------------------------- | ------ | --------------- |
| 2.1   | Dataset annoté ≥ 500 setups par actif (XAU + EURUSD)     | 44 h   | ✅              |
| 2.2   | Régression SmartMoneyEngine vs annotations (F1/P/R)      | 28 h   | ✅              |
| 2.3   | Tuning bayésien hyperparams par actif                    | 36 h   | ✅              |
| 2.4   | Audit visuel automatisé (snapshots PNG)                  | 20 h   | ✅              |
| —     | Buffer (annotation tooling, debug, review)               | 16 h   |                 |
| **TOTAL** |                                                      | **144 h** |             |

---

## Batch 2.1 — Dataset annoté ≥ 500 setups par actif (44 h)

### Objectif
Créer le ground truth manquant (décision I Sprint 0). Annoter manuellement ≥ 500 setups par catégorie (BOS, OB, FVG) sur XAU + EURUSD M15. Définir règles d'annotation rigoureuses, valider inter-annotateur (auto-annotation par Claude + relecture user).

### Steps
1. **Règles d'annotation ICT pures** (6 h)
   - Lecture sources : *The Inner Circle Trader* (Michael Huddleston), Smart Money Concepts (TJR).
   - Définir formalisme :
     - **BOS** : cassure d'un swing high/low identifié par fractale 3-bar, confirmation par close beyond.
     - **CHOCH** : inversion de structure après BOS contraire.
     - **OB** : dernière bougie opposée AVANT impulsion qui casse structure, volume ≥ 1.5× MA20.
     - **FVG** : 3-bar imbalance, body bar2 dépasse high bar1 (ou low bar1), ratio ≥ 0.4 ATR.
     - **Retest** : prix revient ≤ 0.25 ATR du niveau OB/FVG sous armed_window=30 bars.
   - Sortie : `docs/algo/ict_annotation_rules.md`.

2. **Outil d'annotation** (8 h)
   - `scripts/annotate_ict.py` : CLI interactif basé sur plotly/matplotlib.
   - Affiche fenêtres 60 bars autour d'un candidat algo, demande validation `y/n/skip` par catégorie.
   - Sauvegarde : `data/annotations/{symbol}_{tf}_{category}_annotations.parquet`.
   - Format colonnes : `timestamp, symbol, tf, category, is_valid (bool), confidence (1-5), notes`.

3. **Pré-screening candidats algo** (4 h)
   - Lancer `SmartMoneyEngine` actuel sur XAU 2023-2024 + EURUSD 2023-2024.
   - Extraire ~2 000 candidats par catégorie par actif (BOS, OB, FVG).
   - Échantillonner 500 aléatoirement par catégorie pour annotation.

4. **Annotation XAU** (12 h)
   - 500 BOS + 500 OB + 500 FVG = 1 500 setups sur XAU M15.
   - Pace : ~30 sec/setup → 12 h.
   - Annotation par Claude (vision possible sur plots), relecture spot-check user 10 %.

5. **Annotation EURUSD** (8 h)
   - Idem, 1 500 setups EURUSD M15.
   - Comparaison rule-by-rule avec XAU (sensibilité ATR, sessions différentes).

6. **Audit inter-annotateur** (3 h)
   - Si user a annoté 10 % en parallèle → calculer Cohen's kappa.
   - Cible : kappa ≥ 0.6 (substantial agreement).
   - Sortie : `audits/2026-Q2/annotation_quality.md`.

7. **Documentation dataset** (3 h)
   - `docs/algo/annotation_dataset.md` : schéma, stats, limites (bias annotator, sample bias pre-screening).

### Critères d'acceptation
- ✅ 3 000 annotations (1 500 XAU + 1 500 EURUSD), 500 par catégorie.
- ✅ Cohen's kappa ≥ 0.6 sur sample inter-annotateur.
- ✅ Format parquet versionné, manifest checksums.
- ✅ Règles ICT documentées avec références.

### Findings audit adressés
- **P0-2** (OB ≠ ICT) — préparation au fix, validation empirique.
- **Décision I Sprint 0** (annotations reportées) — ✅ exécutée.

### Dépendances
- Sprint 1 batch 1.5 (EURUSD CSV coverage ≥ 99.4 %).

### Risques
- 30 sec/setup optimiste → si réel 60 sec → +12h budget. Mitigation : buffer 16h.
- Bias annotation Claude (training set ICT) → user spot-check critique. Mitigation : kappa ≥ 0.6 mandatory.
- Pre-screening introduit selection bias (on n'annote que ce que l'algo trouve). Mitigation : ajouter 50 setups "random bar" par catégorie pour estimer false negatives.

---

## Batch 2.2 — Régression SmartMoneyEngine vs annotations (28 h)

### Objectif
Mesurer la précision réelle du `SmartMoneyEngine` (façade Sprint 1.0 + extraction Sprint 6.5) contre les annotations. Quantifier F1, Precision, Recall par catégorie × actif. Identifier les bugs structurels révélés par l'écart algo vs ICT pure.

### Steps
1. **Harness de validation** (6 h)
   - `src/intelligence/smart_money/validation.py` :
     - `evaluate_detector(detector_fn, annotations) -> Metrics`.
     - Tolérance temporelle : prédiction valide si dans ±2 bars de l'annotation.
     - Métriques : F1, Precision, Recall, MCC, confusion matrix.

2. **Tests baseline (algo actuel)** (4 h)
   - Lancer `evaluate_detector` sur BOS, CHOCH, OB, FVG pour XAU + EURUSD.
   - Sortie : `reports/sprint_2/detector_baseline_metrics.json`.
   - Attendu : OB F1 < 0.4 (validation eval_03 = engulfing-only).

3. **Diagnostic erreurs OB** (4 h)
   - Confusion matrix OB algo vs OB annotations.
   - Analyser les 50 OB algo "False Positive" → confirmer absence de BOS dans ±20 bars.
   - Analyser les 50 OB annotations "False Negative" → confirmer pattern manqué.

4. **Diagnostic FVG threshold** (3 h)
   - Histogramme distribution `fvg_size / ATR` sur annotations vs algo.
   - Quantile 25 % annotations → nouveau threshold candidat.
   - Cible : FVG_THRESHOLD passe de 0.1 à 0.4 ATR (P1-2).

5. **Diagnostic retest tolérance** (3 h)
   - Histogramme `(price - level) / ATR` sur retests annotés.
   - Quantile 50 % → tolérance optimale.
   - Cible : RETEST_TOL_ATR de 0.5 à 0.25 (P1-3).

6. **Refactor candidats P0-2** (4 h)
   - Implémenter OB ICT-conforme : dernière bougie opposée AVANT impulsion qui casse structure ≥ 1×ATR.
   - Re-run `evaluate_detector` → F1 cible ≥ 0.65.
   - Si F1 < 0.65 → documenter gap, escalade Sprint 3 (le détecteur peut être OK même si l'edge est ailleurs).

7. **Rapport de section** (4 h)
   - `reports/sprint_2/smart_money_validation_report.md` :
     - Tableau F1/P/R par détecteur × actif, baseline vs refactored.
     - Confusion matrices.
     - Liste P0/P1 fixés et reportés Sprint 3.

### Critères d'acceptation
- ✅ Métriques baseline + refactored documentées.
- ✅ F1 BOS ≥ 0.65 sur XAU et EURUSD (cible institutionnelle).
- ✅ OB ICT-conforme implémenté.
- ✅ FVG_THRESHOLD + RETEST_TOL_ATR re-calibrés empiriquement.

### Findings audit adressés
- **P0-2** (OB ≠ ICT) — ✅ closed.
- **P1-2** (FVG_THRESHOLD trop laxe) — ✅ closed.
- **P1-3** (RETEST_TOL_ATR ≈ spread) — ✅ closed.

### Dépendances
- Batch 2.1 (annotations).
- Sprint 1 batch 1.0 (façade smart_money).

### Risques
- F1 < 0.65 même après refactor → l'algo est intrinsèquement trop laxe ou les annotations sont biaisées. Mitigation : re-annoter 100 cas litigieux avec user, décider si pivot SMC vers autre paradigme.

---

## Batch 2.3 — Tuning bayésien hyperparams par actif (36 h)

### Objectif
Optimiser les ~10 hyperparamètres ICT par actif (XAU, EURUSD) via Bayesian optimization (Optuna). Cible : maximiser F1 sur set validation, éviter overfit via CV temporelle.

### Steps
1. **Setup Optuna** (4 h)
   - `requirements.txt` : `optuna>=3.5`.
   - `src/intelligence/smart_money/tuning.py` : framework générique.
   - Hyperparams space :
     - `FVG_THRESHOLD_ATR` (0.1 - 1.0, log).
     - `RETEST_TOL_ATR` (0.05 - 0.5).
     - `ARMED_WINDOW` (10 - 60).
     - `OB_VOL_MULT` (1.0 - 3.0).
     - `BOS_CONFIRM_CLOSE_BUFFER_ATR` (0 - 0.2).
     - `SWING_FRACTAL_BARS` (2 - 5).
     - `RSI_DIV_LOOKBACK` (10 - 40).
     - `MOMENTUM_LOOKBACK` (3 - 15).

2. **CV temporelle** (4 h)
   - Split annotations 60/20/20 train/val/test par actif, stratifié par catégorie.
   - Critère anti-overfit : F1_train - F1_val < 0.10.

3. **Tuning XAU** (10 h)
   - 200 trials Optuna, TPE sampler, MedianPruner.
   - Objectif : maximiser F1 macro (mean BOS + OB + FVG).
   - Logs Optuna persistés `optuna_studies/xau_smc.db`.

4. **Tuning EURUSD** (8 h)
   - Idem, 200 trials.
   - Comparer hyperparams optimaux XAU vs EURUSD → quantifier domain shift.

5. **Validation OOS test set** (4 h)
   - Appliquer hyperparams optimaux sur test set (20 % annotations).
   - Si F1_test < F1_val - 0.05 → overfit, restart avec early stopping plus strict.

6. **Persistance per-actif** (3 h)
   - `config/smc_hyperparams.json` :
     ```json
     {
       "XAUUSD": {"FVG_THRESHOLD_ATR": 0.42, ...},
       "EURUSD": {"FVG_THRESHOLD_ATR": 0.38, ...}
     }
     ```
   - Loader dans `InstrumentConfig`.

7. **Rapport tuning** (3 h)
   - `reports/sprint_2/smc_tuning_report.md` :
     - Hyperparams optimaux par actif.
     - F1 train/val/test.
     - Parallel coordinates plot Optuna.
     - Recommandations cross-actif.

### Critères d'acceptation
- ✅ Optuna study XAU + EURUSD persistés, 200 trials chacun.
- ✅ F1 OOS test ≥ F1 baseline + 0.05.
- ✅ `smc_hyperparams.json` chargé par `InstrumentConfig`.
- ✅ Overfit gate : |F1_train - F1_test| < 0.10.

### Findings audit adressés
- **P0-2** (suite refactor OB) — tuning empirique.
- **P1-6** (Session NY hardcoded) — partiel via OB_VOL_MULT.

### Dépendances
- Batch 2.2 (refactor OB + metrics harness).

### Risques
- Optuna sur 8 hyperparams = espace 10^8 → 200 trials peut sous-explorer. Mitigation : warm-start avec defaults + restrict ranges.
- Hyperparams XAU et EURUSD divergent fort → suggère que l'edge est asset-specific, pas SMC-universel. C'est une **info**, pas un bug.

---

## Batch 2.4 — Audit visuel automatisé (snapshots PNG) (20 h)

### Objectif
Générer automatiquement des snapshots PNG pour chaque détection (BOS/OB/FVG), permettant audit visuel rapide et catch de régressions silencieuses entre versions.

### Steps
1. **Framework de plotting** (4 h)
   - `src/intelligence/smart_money/visualization.py` :
     - `plot_detection(df, detection, output_path)` : matplotlib OHLC candles + détection annotée.
     - Fenêtre 60 bars centrée sur la détection.
     - Annotations : BOS line (rouge), OB box (orange), FVG box (vert).

2. **Snapshot suite** (4 h)
   - `tests/visual/test_snapshots.py` :
     - Pour 20 détections fixées (10 XAU + 10 EURUSD), génère PNG.
     - Compare via `pixelmatch` (tolerance 1 % pixels).
     - Si différence → snapshot fail.

3. **Galerie audit** (3 h)
   - Script `scripts/generate_smc_gallery.py` :
     - Génère galerie HTML : 100 détections random par catégorie × actif.
     - Output : `reports/sprint_2/smc_gallery/index.html`.
   - User peut review visuellement les patterns.

4. **CI integration** (3 h)
   - Workflow `algo_tests.yml` : nightly snapshot tests sur subset.
   - Snapshots stockés `tests/visual/__snapshots__/` (git-tracked, < 1 MB).

5. **Régression visuelle entre commits** (3 h)
   - Hook pre-merge : si SmartMoneyEngine modifié → re-run snapshot suite.
   - Diff visuel pushed à PR comment.

6. **Documentation** (3 h)
   - `docs/algo/visual_audit.md` : guide review galerie, gestion faux positifs.

### Critères d'acceptation
- ✅ 20 snapshots tests verts.
- ✅ Galerie HTML 600 détections accessible.
- ✅ CI snapshot test nightly vert.
- ✅ Doc audit visuel.

### Findings audit adressés
- **Décision I Sprint 0** suite (validation visuelle complète).
- Renforce P0-2 + P1-2 + P1-3 fixes.

### Dépendances
- Batch 2.2 (détections refactored à visualiser).

### Risques
- Snapshots fragiles aux versions matplotlib → freeze version dans CI.
- Galerie HTML 600 PNG = ~50 MB → ne pas tracker git, generate à la demande.

---

## Gate de sortie du Sprint 2 (checklist 10 items)

1. ✅ 3 000 annotations XAU + EURUSD (BOS, OB, FVG), Cohen's kappa ≥ 0.6.
2. ✅ Métriques baseline `SmartMoneyEngine` documentées (F1/P/R par catégorie × actif).
3. ✅ OB ICT-conforme implémenté, F1 ≥ 0.65 BOS sur XAU + EURUSD.
4. ✅ FVG_THRESHOLD + RETEST_TOL_ATR re-calibrés empiriquement.
5. ✅ Tuning bayésien Optuna 200 trials XAU + EURUSD.
6. ✅ Hyperparams optimaux persistés `config/smc_hyperparams.json`.
7. ✅ Overfit gate vert (F1 OOS ≥ F1 val - 0.05).
8. ✅ Snapshot tests CI verts (20 détections fixées).
9. ✅ Galerie HTML 600 détections générée.
10. ✅ `sprint_2_retrospective.md` rédigé + diff cumulé vs `v0.9.0`.

---

## Livrables Sprint 2 (arborescence)

```
src/intelligence/smart_money/
  ├── validation.py                  # evaluate_detector + Metrics
  ├── tuning.py                      # Optuna framework
  └── visualization.py               # plot_detection

data/annotations/
  ├── XAUUSD_M15_BOS_annotations.parquet
  ├── XAUUSD_M15_OB_annotations.parquet
  ├── XAUUSD_M15_FVG_annotations.parquet
  └── EURUSD_M15_*.parquet (3 files)

config/
  └── smc_hyperparams.json

optuna_studies/
  ├── xau_smc.db
  └── eurusd_smc.db

scripts/
  ├── annotate_ict.py
  └── generate_smc_gallery.py

tests/
  ├── test_smart_money_validation.py     (15-20 tests)
  ├── test_smart_money_tuning.py         (8-10 tests)
  └── visual/
      ├── test_snapshots.py              (20 snapshots)
      └── __snapshots__/                 (PNG baselines)

reports/sprint_2/
  ├── detector_baseline_metrics.json
  ├── smart_money_validation_report.md
  ├── smc_tuning_report.md
  └── smc_gallery/index.html

docs/algo/
  ├── ict_annotation_rules.md
  ├── annotation_dataset.md
  └── visual_audit.md

audits/2026-Q2/
  └── annotation_quality.md

roadmap/sprints/
  ├── sprint_2.md
  ├── sprint_2_progress.md
  └── sprint_2_retrospective.md
```

---

## Décisions ouvertes pour user

1. **User annote 10 % spot-check** (~3 h sur 2 semaines) — confirmer disponibilité, sinon Claude annote seul avec confidence ≤ 4/5 flag.
2. **F1 cible 0.65 vs 0.75** : 0.65 = "production decent", 0.75 = "PhD-level". Trancher avant tuning.
3. **Si F1 reste < 0.65 même tuné** : pivot vers paradigme alternatif (order flow, vol breakout) → escalade Sprint 3.

---

**Signé** : Claude, 2026-05-15
