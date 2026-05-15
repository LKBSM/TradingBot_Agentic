# Backlog — Conformal Calibration Engineer

**Date** : 2026-05-15
**Owner** : Conformal Calibration Engineer

## Sprint 1 (S3-S4) — Data Layer (frozen, prep)

- [ ] Audit code `src/intelligence/conformal_wrapper.py` ligne par ligne : documenter l'API actuelle (Split + ACI), versionner les hyperparams (`alpha`, `gamma`, window_size) en `dataclass` Pydantic v2 — 4 h
- [ ] Identifier les call sites du wrapper (`grep -r ConformalWrapper src/`) pour cartographier les downstream consumers — 2 h

## Sprint 2 (S5-S6) — Detection (frozen)

- [ ] Lecture théorique : Angelopoulos & Bates 2024 "Gentle Introduction to Conformal Prediction" + Vovk 2005 "Algorithmic Learning in a Random World" + Gibbs & Candès 2021 "Adaptive Conformal Inference" — 4 h
- [ ] Notebook exploratoire `notebooks/conformal_exploration.ipynb` : tester Split conformal sur vol forecast XAU avec différents alpha (0.05, 0.10, 0.20) et fenêtres (250, 500, 1000) — 6 h

## Sprint 3 (S7-S8) — Edge Discovery (prep)

- [ ] Préparer le contrat d'interface avec Stat Validator : si Sprint 3 produit un edge probabiliste (P(win|features)), spécifier comment conformal wrapper s'applique au score logistic — 3 h
- [ ] Implémenter test exchangeability préliminaire : `tests/test_conformal_exchangeability.py` property-based (Hypothesis) — 4 h

## Sprint 4 (S9-S10) — Calibration & Confidence (rôle pivot)

- [ ] **Batch 4.1** — Implémenter Mondrian conformal stratifié par régime (P1-9) : `src/intelligence/conformal_mondrian.py`. Cible : conditional coverage par stratum ±5 % — 12 h
- [ ] **Batch 4.2** — Mesurer PICP empirique OOS 2024 XAU + EURUSD (P0-11). Cibles : nominal 80 % → empirique [78 %, 82 %], nominal 90 % → [88 %, 92 %] — 6 h
- [ ] **Fix P0-20** — PICP catastrophique 43.6 % : root cause analysis (résiduals non-exchangeables ? fenêtre cal trop courte ? alpha mal câblé via P0-18 ?) — 8 h
- [ ] **Batch 4.3** — Validation OOS bandes de probabilité : produire `audits/2026-Q3/conformal_calibration.md` avec plots PICP par alpha + Mondrian par régime — 5 h
- [ ] Coopérer avec Vol Modeler pour P0-18 (`tcp_alpha` ignoré) : confirmer que une fois consommé, alpha=0.20 (cible 80 %) atteint la cible — 2 h
- [ ] Test régression `tests/test_conformal_picp_regression.py` : à chaque commit, calcul PICP sur fixture XAU 2024 ; fail si hors [78 %, 82 %] — 3 h
- [ ] Documentation `docs/algo/conformal.md` v1 — 4 h

## Sprint 5 (S11-S12) — Robustness & Stress Testing

- [ ] Stress test conformal sur 4 crisis : vérifier que ACI adapte la quantile estimate dans ≤ 100 obs après crisis (batch 5.2) — 5 h
- [ ] Property-based : invariance par permutation aléatoire de la fenêtre de calibration (test exchangeability formel) — 4 h
- [ ] Adversarial : injecter outliers extrêmes en cal window, vérifier robustesse (batch 5.4) — 3 h
- [ ] Coordination avec LQA pour décision : fenêtre ACI doit-elle être adaptative par régime ? — 2 h
- [ ] Sensibilité hyperparams ±20 % (batch 5.3) — 3 h

## Sprint 6 (S13-S14) — Production Hardening

- [ ] Profiling conformal wrapper : cible < 5 ms / forecast (batch 6.1) — 3 h
- [ ] Versioning : ajouter `conformal_version` dans les outputs (P0-16 support) — 1 h
- [ ] Snapshot store : sauvegarder l'historique des intervalles produits par signal (input State Machine Eng) — 2 h
- [ ] Cache calibration set : invalidation par hash + commit_sha — 2 h

## Sprint 7 (S15-S16) — Commercial Readiness

- [ ] Tear sheet conformal par actif : PICP plot, interval width évolution, Mondrian conditional coverage (batch 7.2) — 5 h
- [ ] Fiche transparence client : "comment nous calibrons nos bandes de confiance" (batch 7.3) — 3 h
- [ ] Documentation `docs/algo/conformal.md` v2 finale avec proofs sketches — 4 h

## Inbox (non priorisé)
- Cross-conformal (CV+ : Barber et al. 2021) — alternative à Split.
- Online conformal (Foygel-Barber 2022) — adaptation continue.
- Weighted conformal (Tibshirani et al. 2019) pour covariate shift.
- Conformal sur classifier multi-classe (P(BUY|features), P(SELL|features), P(HOLD)) — pertinent si Sprint 4 produit classifier.
- Jackknife+ pour intervalles régression robustes.
- Conformal sur risk metrics directement (VaR conformal Romano et al. 2019).
