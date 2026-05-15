# Sprint 4 — Calibration & Confidence

**Période** : Semaines 9-10 (S9-S10, ~2026-07-11 → 2026-07-25)
**Charge estimée totale** : **76 h** productives + buffer 10 h = 86 h
**Objectif** : transformer les scores bruts (Sprint 3 stacking + ConfluenceDetector reweighting) en **probabilités calibrées** avec bandes de confiance OOS-validées. Mondrian conformal stratifié par régime (P1-9), Logistic L1 multi-feature pour remplacer le scoring linéaire actuel (P0-1), validation PICP OOS (P0-11, P0-20), documentation client-facing. Fix P0-18 (tcp_alpha ignoré) et P0-19 (HMM train/serve skew).
**Gate de sortie** : PICP empirique 78-82 % (cible 80 %), Brier skill ≥ +0.03 vs constant baseline, fiches client transparence rédigées.

---

## 0. Vue d'ensemble — 5 batches

| Batch | Titre                                                | Heures | Critique chemin |
| ----- | ---------------------------------------------------- | ------ | --------------- |
| 4.1   | Mondrian conformal stratifié par régime               | 22 h   | ✅              |
| 4.2   | Logistic regression L1 multi-feature (replace scoring) | 18 h | ✅              |
| 4.3   | Validation OOS bandes probabilité (PICP, MPIW)       | 14 h   | ✅              |
| 4.4   | Fix P0-18 (tcp_alpha) + P0-19 (HMM skew)             | 10 h   | ✅              |
| 4.5   | Documentation client-facing                          | 12 h   |                 |
| —     | Buffer                                                | 10 h   |                 |
| **TOTAL** |                                                  | **86 h** |               |

---

## Batch 4.1 — Mondrian conformal stratifié par régime (22 h)

### Objectif
Le ConformalWrapper actuel rejette tout (score 7.0/10 mais opérationnellement = 0 trades). PICP catastrophique 43.6 % vs 80 % cible (P0-20). Refonte Mondrian : un conformal **par régime** (low_vol / high_vol / crisis) pour adapter la nonconformity à la dispersion locale.

### Steps
1. **Audit ConformalWrapper actuel** (3 h)
   - Lecture `src/intelligence/conformal_wrapper.py`.
   - Identifier hardcoded `tcp_alpha=0.10` (P0-18).
   - Documenter API.

2. **Mondrian implementation** (6 h)
   - `src/intelligence/calibration/mondrian_conformal.py` :
     - `MondrianConformalRegressor(base_model, regime_fn, alpha=0.20)`.
     - Fit calibration set séparé par régime.
     - Predict : retourne `lower, upper` ajusté par régime à l'inférence.
   - Référence : Vovk 2003, Boström & Johansson 2020.

3. **Régime stratification** (2 h)
   - Utiliser `regime_filter.py` HMM 3-state (decision D).
   - Fallback : si régime collapse (Sprint 3 stack `low` 100 %) → revert ACI single.

4. **Calibration set splits** (2 h)
   - Train 60 % / calibration 20 % / test 20 % temporel.
   - Pour chaque actif/TF.

5. **Fit + Predict** (3 h)
   - Wrapper autour Stacking Sprint 3.3.
   - Output : `(proba_mean, proba_lower, proba_upper)` par signal.

6. **Tests** (3 h)
   - `tests/test_mondrian_conformal.py` :
     - Coverage empirique ≥ 1-alpha sur synthetic.
     - Mondrian split : 3 régimes → 3 jeux de quantiles.
     - Edge : régime non vu en calibration → fallback global.

7. **Rapport** (3 h)
   - `reports/sprint_4/mondrian_conformal_report.md` :
     - PICP par régime × actif.
     - MPIW (mean prediction interval width).
     - Comparaison Mondrian vs single ACI.

### Critères d'acceptation
- ✅ Mondrian implémenté + 15+ tests verts.
- ✅ PICP empirique ≥ 75 % sur ≥ 1 config (cible 80 %, tolerance ±5 %).
- ✅ MPIW raisonnable (pas dégénéré à [0,1]).

### Findings audit adressés
- **P0-11** (PICP non mesuré) — ✅ closed.
- **P0-20** (PICP catastrophique 43.6 %) — ✅ closed (devrait atteindre 75-82 %).
- **P1-9** (Conformal sans Mondrian) — ✅ closed.

### Dépendances
- Sprint 3 batch 3.3 (stacking models).

### Risques
- 3 régimes × 4 configs = 12 calibration sets. Chacun peut avoir < 100 samples → calibration instable. Mitigation : merger régimes faibles.

---

## Batch 4.2 — Logistic regression L1 multi-feature (18 h)

### Objectif
Remplacer le scoring ConfluenceDetector hardcoded (Pearson −0.008, Brier skill −0.022) par une **Logistic regression L1** sur 8 `weighted_scores` du detector + features Sprint 3.1. Cible Brier skill ≥ +0.03.

### Steps
1. **Extract `weighted_scores`** (2 h)
   - Modifier `ConfluenceDetector` pour exposer dict `weighted_scores = {regime, news, bos, fvg, ob, volume, momentum, rsi_div}`.
   - Persister par signal pour training.

2. **Dataset training** (3 h)
   - Réutiliser feature pipeline Sprint 3.1.
   - Target : `forward_return_5bars > 0` (proxy "good signal").
   - 70/15/15 train/val/test temporel.

3. **Logistic L1** (4 h)
   - `sklearn.linear_model.LogisticRegression(penalty='l1', solver='saga', C=tune)`.
   - GridSearch C ∈ {0.01, 0.1, 1.0, 10.0}.
   - Coefficient sparsité attendu : 3-5 features non-zero parmi 20 candidates.

4. **Wire dans ConfluenceDetector** (3 h)
   - Si `LOGISTIC_SCORING=true` env var → utiliser logistic proba × 100.
   - Sinon → legacy scoring (rollback safety).

5. **Tests** (3 h)
   - `tests/test_logistic_scoring.py` :
     - Score ∈ [0, 100].
     - Sparsité L1 vérifiée.
     - Régression baseline : Brier skill > 0 sur validation.

6. **Rapport calibration** (3 h)
   - `reports/sprint_4/logistic_l1_report.md` :
     - Brier score baseline (constant 0.5) vs Logistic.
     - Reliability diagram (binning 10 quantiles).
     - Top features non-zero.

### Critères d'acceptation
- ✅ Logistic L1 fit + persisté.
- ✅ Brier skill ≥ +0.03 vs constant baseline sur ≥ 1 config.
- ✅ Reliability diagram monotone.
- ✅ Coefficient sparsité < 50 % features non-zero.

### Findings audit adressés
- **P0-1** (ConfluenceDetector Pearson −0.008) — ✅ closed.
- **P1-4** (Double-gating 96.3 % rejection) — partial (logistic doit avoir threshold ajusté).
- **P1-5** (OB ↔ Retest corrélés) — ✅ closed (L1 force sparsité).

### Dépendances
- Sprint 3 batch 3.1 (features) + 3.2 (IC).

### Risques
- Si Brier skill < 0.03 → l'edge n'est pas dans les 8 weighted_scores. Escalade Sprint 3.5 pivot.

---

## Batch 4.3 — Validation OOS bandes probabilité (14 h)

### Objectif
Valider rigoureusement les bandes de probabilité sur un set OOS test (jamais vu en train/calib). Mesurer PICP, MPIW, CRPS, Hosmer-Lemeshow.

### Steps
1. **Define OOS test windows** (2 h)
   - Pour chaque actif : 2025-Q1 (out-of-sample du tuning Sprint 2-3).
   - Backtest avec bandes proba.

2. **Compute PICP** (3 h)
   - `picp = mean(y_true ∈ [lower, upper])`.
   - Cible 80 % avec tolerance ±5 %.

3. **Compute MPIW** (2 h)
   - `mpiw = mean(upper - lower)`.
   - Comparer Mondrian vs ACI single.

4. **Compute CRPS** (2 h)
   - Continuous Ranked Probability Score.
   - Plus bas = mieux. Compare baselines (constant, gaussian).

5. **Hosmer-Lemeshow** (2 h)
   - Test calibration goodness-of-fit (10 bins).
   - p-value > 0.05 = bien calibré.

6. **Rapport** (3 h)
   - `reports/sprint_4/calibration_validation_oos.md` :
     - Tableau PICP/MPIW/CRPS par config × régime.
     - Reliability diagrams.
     - Verdict : configs où PICP ∈ [75, 85] %.

### Critères d'acceptation
- ✅ PICP ∈ [75, 85] % sur ≥ 1 config OOS.
- ✅ Hosmer-Lemeshow p > 0.05 sur ≥ 1 config.
- ✅ Rapport signé.

### Findings audit adressés
- **P0-11** (PICP non mesuré) — ✅ closed.
- **P0-20** (PICP 43.6 %) — ✅ closed (fix Mondrian).

### Dépendances
- Batches 4.1 + 4.2.

### Risques
- Aucune config ne passe PICP 75-85 % → escalade Sprint 5 stress test pour identifier régimes problématiques.

---

## Batch 4.4 — Fix P0-18 (tcp_alpha) + P0-19 (HMM skew) (10 h)

### Objectif
Deux bugs P0 isolés de l'audit Section 3.4 (Volatility) :
- **P0-18** : `tcp_alpha` hardcoded à 0.10 dans `volatility_forecaster.py:367`, ignore `InstrumentConfig.tcp_alpha`.
- **P0-19** : HMM train/serve skew massif (11 % accord) entre Viterbi smoothing (l.874) et `predict(1-row)` inférence (l.922).

### Steps
1. **Fix tcp_alpha** (2 h)
   - `volatility_forecaster.py:367` : `tcp_alpha = self.config.tcp_alpha if self.config else 0.10`.
   - Test régression : 2 instruments différents avec tcp_alpha différents → quantiles différents.

2. **Fix HMM train/serve skew** (5 h)
   - Lire `volatility_forecaster.py:870-925` pour comprendre divergence.
   - Option 1 : utiliser **online filtering** (filter+predict, pas smoothing) en training pour matcher inférence.
   - Option 2 : utiliser smoothing en inférence aussi (latence acceptable ?).
   - Tester accord train/serve > 80 %.

3. **Tests régression** (2 h)
   - `tests/test_vol_forecaster_fixes.py` :
     - tcp_alpha config respecté.
     - HMM accord ≥ 80 %.

4. **Doc** (1 h)
   - Update `docs/algo/volatility.md`.

### Critères d'acceptation
- ✅ tcp_alpha config respecté.
- ✅ HMM accord train/serve ≥ 80 %.
- ✅ Tests régression verts.

### Findings audit adressés
- **P0-18** — ✅ closed.
- **P0-19** — ✅ closed.

### Dépendances
- Aucune.

### Risques
- Fix HMM peut modifier régimes prédits → impact downstream stacking. Mitigation : re-train Sprint 3 si nécessaire.

---

## Batch 4.5 — Documentation client-facing (12 h)

### Objectif
Rédiger documentation user-facing expliquant les bandes de probabilité, leur calibration et leur usage. Précurseur des tear sheets Sprint 7.

### Steps
1. **Guide "Probability Bands"** (4 h)
   - `docs/client/probability_bands.md` :
     - Que mesure PICP 80 % ?
     - Comment lire `proba=0.62 [0.51, 0.73]` ?
     - Différence point estimate vs interval.
   - Visuels (matplotlib examples).

2. **FAQ calibration** (3 h)
   - `docs/client/calibration_faq.md` :
     - "Pourquoi 80 % et pas 95 %" ?
     - "Que se passe-t-il en régime crisis" ?

3. **Glossaire** (2 h)
   - `docs/client/glossary.md` :
     - PICP, MPIW, Brier skill, Mondrian conformal.
     - Vulgarisation accessible (target B2C educated investor).

4. **Tear sheet preview** (3 h)
   - `docs/client/tear_sheet_preview.md` : maquette pour Sprint 7.

### Critères d'acceptation
- ✅ 4 docs client rédigées.
- ✅ Approuvées par user (review).

### Findings audit adressés
- Préparation Sprint 7 (transparence client).

### Dépendances
- Batches 4.1-4.3.

### Risques
- Vocabulaire trop technique → review user impérative.

---

## Gate de sortie du Sprint 4 (checklist 11 items)

1. ✅ Mondrian conformal implémenté + tests.
2. ✅ Logistic L1 fit + Brier skill ≥ +0.03.
3. ✅ Reliability diagram monotone sur ≥ 1 config.
4. ✅ PICP OOS ∈ [75, 85] % sur ≥ 1 config.
5. ✅ Hosmer-Lemeshow p > 0.05 sur ≥ 1 config.
6. ✅ tcp_alpha config respecté (P0-18).
7. ✅ HMM accord train/serve ≥ 80 % (P0-19).
8. ✅ 4 docs client rédigées.
9. ✅ Suite tests verte.
10. ✅ `LOGISTIC_SCORING=true` activable via env var.
11. ✅ `sprint_4_retrospective.md` rédigé.

---

## Livrables Sprint 4 (arborescence)

```
src/intelligence/calibration/
  ├── __init__.py
  ├── mondrian_conformal.py
  └── logistic_scoring.py

src/intelligence/
  ├── confluence_detector.py        # patched (weighted_scores exposed + logistic option)
  ├── conformal_wrapper.py          # patched (Mondrian wired)
  └── volatility_forecaster.py      # patched (tcp_alpha + HMM)

models/sprint_4/
  ├── logistic_l1_xau_m15.pkl
  ├── logistic_l1_eur_m15.pkl
  └── mondrian_calibration_*.pkl

tests/
  ├── test_mondrian_conformal.py     (15 tests)
  ├── test_logistic_scoring.py       (12 tests)
  └── test_vol_forecaster_fixes.py   (8 tests)

reports/sprint_4/
  ├── mondrian_conformal_report.md
  ├── logistic_l1_report.md
  └── calibration_validation_oos.md

docs/client/
  ├── probability_bands.md
  ├── calibration_faq.md
  ├── glossary.md
  └── tear_sheet_preview.md

docs/algo/
  └── volatility.md (patched)

roadmap/sprints/
  ├── sprint_4.md
  ├── sprint_4_progress.md
  └── sprint_4_retrospective.md
```

---

## Décisions ouvertes pour user

1. **PICP cible 80 % vs 90 %** : 80 % = bandes plus serrées, plus actionnable. 90 % = bandes plus larges, plus conservateur. Trancher.
2. **Logistic active par défaut** : si Brier skill ≥ +0.03 → activer `LOGISTIC_SCORING=true` par défaut ?
3. **Docs client review** : user disponible pour review 2h ?

---

**Signé** : Claude, 2026-05-15
