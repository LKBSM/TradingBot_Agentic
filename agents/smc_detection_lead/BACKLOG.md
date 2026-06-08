# Backlog — SMC/ICT Detection Lead

**Date** : 2026-05-15
**Owner** : SMC Detection Lead

## Sprint 1 (S3-S4) — Extraction + fix bugs P0/P1 (batch 1.0, critique chemin)

- [ ] Créer arborescence `src/intelligence/smart_money/` (`__init__.py`, `bos.py`, `choch.py`, `order_block.py`, `fvg.py`, `retest.py`, `smart_money_engine.py`) — 2 h
- [ ] Extraire la logique BOS/CHOCH depuis `src/environment/strategy_features.py` vers `bos.py` + `choch.py` ; conserver l'API publique pour backward-compat via re-export (P0-9, décision E) — 6 h
- [ ] Extraire la logique OB / FVG / retest vers les modules dédiés — 5 h
- [ ] **Fix P0-2** : redéfinir OB ICT : OB = last opposite candle AVANT un BOS confirmé ; reject 40 % des OB actuels qui n'ont pas de BOS associé dans ±20 bars — 6 h
- [ ] **Fix P0-15** : bug RSI Divergence indexage (`strategy_features.py:849-857` → version corrigée dans `src/intelligence/smart_money/divergence.py` ou utility module). Écrire `tests/test_rsi_divergence_indexing.py` régression — 3 h
- [ ] **Fix P1-1** : harmoniser `armed_window=5` / `RETEST_ARMED_WINDOW=30` ; documenter le bon défaut par actif/TF (input décision LQA) — 2 h
- [ ] Écrire contrat Pydantic v2 pour `SmartMoneyEvent` (type, bar_ts, price_low, price_high, direction, confidence) — 2 h
- [ ] Écrire tests unitaires `tests/test_smart_money_bos.py`, `..._choch.py`, `..._order_block.py`, `..._fvg.py`, `..._retest.py` (≥ 30 tests par module) — 8 h
- [ ] Activer Numba pour les hot loops (BOS detector, OB scanner) ; bench avant/après — 3 h
- [ ] Mettre à jour `sentinel_scanner.py` pour consommer `smart_money_engine.SmartMoneyEngine` (au lieu de `strategy_features`) — 2 h

## Sprint 2 (S5-S6) — Detection Engine Validation (rôle pivot)

- [ ] **Définir guidelines d'annotation** (`docs/algo/annotation_guidelines_smc.md`) : critères BOS, CHOCH, OB, FVG ICT-stricts — 4 h
- [ ] **Annoter manuellement** XAU M15 : 500 BOS + 500 CHOCH + 500 OB + 500 FVG (batch 2.1, décision I) — 17 h
- [ ] **Annoter manuellement** EURUSD M15 : idem — 17 h
- [ ] Implémenter `tests/test_smc_vs_annotations.py` : charge annotations + calcule F1/P/R par event type + asset (batch 2.2) — 6 h
- [ ] Tuning bayésien hyperparams (Optuna ou scikit-optimize) sur grille : `BOS_LOOKBACK ∈ [10,50]`, `FVG_THRESHOLD ∈ [0.1, 1.0] ATR`, `RETEST_TOL ∈ [0.1, 0.8] ATR`, `OB_LOOKBACK ∈ [5, 30]` (batch 2.3, fix P1-2, P1-3) — 10 h
- [ ] Audit visuel automatisé (batch 2.4) : pour chaque event détecté, générer un PNG matplotlib avec ±50 bars contexte → `reports/smc/snapshots_xau_m15/` — 6 h
- [ ] Validation cross-asset : F1 EURUSD M15 + BTCUSD H1 si CSV dispo — 4 h
- [ ] Rapport `audits/2026-Q3/smc_validation_phase2.md` — 3 h

## Sprint 3 (S7-S8) — Edge Discovery (support feature)

- [ ] Exposer features SMC pour Stat Validator (`SmartMoneyFeatures` : count BOS last 20 bars, distance to last OB, time since last CHOCH, etc.) — 5 h
- [ ] Tester si les annotations expertes (label binaire "setup valide") fournissent meilleur signal qu'un detector pur règle — 4 h

## Sprint 4 (S9-S10) — Calibration

- [ ] Pas de livrable nouveau. Support Stat Validator pour incorporer features SMC dans logistic L1.

## Sprint 5 (S11-S12) — Robustness

- [ ] Fuzz adversarial : générer setups synthétiques "near miss" (BOS de 1 cent, FVG de 1 spread) et vérifier que le detector reject (batch 5.4) — 6 h
- [ ] Stress test : refiring rate par régime crisis (COVID 2020 mars) — 3 h
- [ ] Property-based : invariance par scaling prix (×10), par décalage temporel (timezone shift) — 4 h

## Sprint 6 (S13-S14) — Production Hardening

- [ ] Profiling Numba : confirmer < 50 ms / 1000 bars sur CI Linux + Railway prod (batch 6.1) — 4 h
- [ ] Versioning détecteur : ajouter `smart_money_version` dans `SmartMoneyEvent` (P0-16 support) — 1 h
- [ ] Documentation `docs/algo/smart_money.md` avec définitions, paramètres, références ICT — 4 h

## Sprint 7 (S15-S16) — Commercial Readiness

- [ ] Tear sheet SMC par actif : taux d'activation, F1, distribution OB/FVG, exemple PNG (batch 7.2) — 5 h
- [ ] Fiche transparence client : "comment fonctionne notre détection ICT" (batch 7.3) — 3 h
- [ ] E2E 6 actifs × 2 TF (XAU/EUR M15+H1, BTC H1, US500 H1, etc.) (batch 7.4 input) — 4 h

## Inbox (non priorisé)
- Détection liquidity sweeps (au-delà du périmètre BOS/CHOCH).
- Mitigation blocks ICT (sous-catégorie OB).
- Premium / Discount (équilibrium ICT) — feature additionnelle.
- ML-based OB/FVG classifier (vs règle pure) — réserve Sprint 8+.
- Multi-timeframe BOS cascade (M15 BOS dans H1 OB) — feature additionnelle.
