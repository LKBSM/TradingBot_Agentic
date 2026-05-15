# Backlog — Data Quality Engineer

**Date** : 2026-05-15
**Owner** : Data Quality Engineer

## Sprint 1 (S3-S4) — Data Layer Hardening (rôle pivot)

- [ ] Écrire `src/intelligence/data_models.py` : Pydantic v2 `Bar`, `CalendarEvent`, `Tick`, `OHLCVDataset` avec validators (P0-8) — 4 h
- [ ] Refactor `src/intelligence/data_providers.py` pour exposer `DataProvider` ABC uniforme + impl CSV + ForexFactory (P0-8) — 6 h
- [ ] Fix look-ahead MTF : `multi_timeframe_features.py:269` `<=` → `<` (P0-7, batch 1.2) — 1 h
- [ ] Écrire `tests/test_mtf_no_lookahead.py` : property-based Hypothesis (10⁵ tirages, vérifier qu'à tout instant t la barre H1 contient uniquement des M15 de timestamp < t) (P0-7) — 4 h
- [ ] Arbitrage avec LQA : MTF canonique entre `src/intelligence/` et `src/environment/` (P1-13), archiver l'autre — 2 h
- [ ] Pipeline ForexFactory → CSV → `EconomicCalendarFetcher` end-to-end : refresh des données 2026 (P1-14, batch 1.3) — 5 h
- [ ] Étendre presets CSV : BTCUSD, US500, GBPUSD, USDJPY, USOIL via Dukascopy downloader (P0-14, batch 1.5). Priorité : USDJPY + GBPUSD (FX MVP), USOIL P2 — 12 h
- [ ] Audit coverage par actif post-extension, sortie `audits/2026-Q3/data_coverage_sprint_1.md` — 3 h
- [ ] Écrire `tests/test_data_provider_contract.py` : property-based Pydantic v2 (NaN, infinis, gaps, OHLC inconsistencies, monotonicité timestamp) — 4 h
- [ ] Fix `signal_id = uuid.uuid4()` → hash déterministe `(bar_ts, symbol, strategy)` pour reproductibilité bit-à-bit (P1 backtest ref) — 2 h

## Sprint 2 (S5-S6) — Detection Engine Validation (support)

- [ ] Pour chaque actif annoté SMC, produire un export "annotation-ready CSV" : OHLCV + ATR + session + régime (input dataset SMC) — 3 h
- [ ] Tester impact du switch décision A (XAU 2019_2026) sur le runtime SMC : recalculer firing rates et confirmer garde-fou BOS toujours vert — 2 h
- [ ] Documenter `data_version` dans tear sheets (lien hash CSV → résultats backtest) — 2 h

## Sprint 3 (S7-S8) — Edge Discovery (frozen, support léger)

- [ ] Préparer datasets feature engineering Sprint 3 batch 3.1 : exposer microstructure (spread M1 si dispo) + order flow proxy (volume delta) + session indicators (Tokyo/London/NY) + macro features (NFP days, FOMC) — 8 h
- [ ] Documenter contrats `FeatureDataset` pour Stat Validator — 2 h

## Sprint 4 (S9-S10) — Calibration (frozen)

- [ ] Pas de livrable nouveau. Stand-by support si Vol Modeler / Conformal Eng identifient un gap data.

## Sprint 5 (S11-S12) — Robustness & Stress Testing

- [ ] Fuzz dataset generator : `tests/fuzz/test_data_provider_fuzz.py` — injecter NaN, infinis, gaps, spreads anormaux, timestamps désordonnés (batch 5.1) — 5 h
- [ ] Stress slice datasets : prép spéciale COVID 2020 (mars-avril), LDI 2022 (sept-oct), SVB 2023 (mars), BoJ yen 2024 (avril) (batch 5.2 input) — 4 h
- [ ] Sensibilité à corruption data : test que le pipeline détecte ≥ 95 % des corruptions injectées avant atteindre la state machine — 4 h

## Sprint 6 (S13-S14) — Production Hardening

- [ ] Versioning data : ajouter SHA256 + commit_sha dans chaque tear sheet (`reports/baseline/datasets_manifest.json`) (P0-16 support) — 3 h
- [ ] Implémenter cache CSV avec invalidation par hash (perf batch 6.1) — 3 h
- [ ] Documenter procédure refresh data (`docs/algo/data_refresh.md`) — 2 h

## Sprint 7 (S15-S16) — Commercial Readiness

- [ ] Pipeline live data sketch (batch 7.4 e2e support) : valider que les CSV peuvent être remplacés par un feed live sans changer le contrat Pydantic — 4 h
- [ ] Tear sheet "data quality" par actif (coverage, gaps, fraîcheur) — 3 h
- [ ] Recette licence : statuer sur Dukascopy commercial (input décision LQA + juridique externe) — 2 h

## Inbox (non priorisé)
- Migration Polygon ou Databento (différée par décision A, à acter Sprint 1.4 par LQA).
- Crypto BTCUSD : décider tier session (24/7 vs filtré US trading hours).
- Évaluer impact passage M1 → M15 par downsampling vs CSV M15 natif (qualité).
- Sentiment / news textuelle : pipeline NLP différé (hors Sprint 1-7).
- Test régression "data dérive" : alerte si nouveau hash CSV change de plus de X % les firing rates SMC.
