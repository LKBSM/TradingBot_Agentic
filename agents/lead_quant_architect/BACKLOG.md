# Backlog — Lead Quant Architect

**Date** : 2026-05-15
**Owner** : LQA (Claude)

## Sprint 1 (S3-S4) — Data Layer Hardening
- [ ] Reviewer la PR d'extraction `src/intelligence/smart_money/` (P0-9, batch 1.0) — 3 h
- [ ] Arbitrage MTF intelligence/ vs environment/ (P1-13) : trancher une implémentation canonique, documenter dans `sprint_1_decisions.md` — 2 h
- [ ] Reviewer la PR contrat Pydantic v2 DataProvider (P0-8) — 2 h
- [ ] Reviewer fix look-ahead MTF `<=` → `<` (P0-7, batch 1.2) + property-test — 2 h
- [ ] Préparer `sprint_1_decisions.md` (template + 5 décisions initiales pressenties : MTF canonique, FF cadence, contrat Pydantic, presets CSV priorisés, `signal_id` reproductible) — 2 h
- [ ] Rédiger `sprint_1_retrospective.md` à la gate de sortie — 3 h

## Sprint 2 (S5-S6) — Detection Engine Validation
- [ ] Reviewer la PR du dataset annoté SMC (≥ 500 par actif, batch 2.1) — 4 h
- [ ] Arbitrage métriques F1 / precision / recall : trancher le seuil de "done" pour OB ICT-conforme (cible F1 ≥ 0.85, fallback 0.75 ?) — 2 h
- [ ] Reviewer tuning bayésien hyperparams SMC (batch 2.3) — 3 h
- [ ] Rédiger `sprint_2_retrospective.md` — 3 h

## Sprint 3 (S7-S8) — Statistical Edge Discovery
- [ ] Reviewer feature engineering exhaustif (batch 3.1) — 4 h
- [ ] **Arbitrage GATE Sprint 3** : si CI 95 % PF lo > 1.0 ÉCHEC sur tous actifs, trancher GO pivot "valeur explicative non-PF" vs CONTINUE Sprint 4 — 4 h (critique chemin)
- [ ] Reviewer couplage CPCV/DSR/PBO + `src/backtest/validation/` (P0-17) — 4 h
- [ ] Reviewer sweep paramétrique state machine 432 cellules × 4 actifs × 4 TF (P0-12, batch 3.5) — 3 h
- [ ] Coordination Stat Validator + Backtest Infra pour packaging final gates — 2 h
- [ ] Rédiger `sprint_3_retrospective.md` + verdict GATE — 4 h

## Sprint 4 (S9-S10) — Calibration & Confidence
- [ ] **Co-owner refonte ConfluenceDetector** (P0-1) avec Vol Modeler + Stat Validator : logistic L1 multi-feature, Brier skill cible ≥ +0.03 — 8 h
- [ ] Reviewer Mondrian conformal stratifié par régime (P1-9) — 3 h
- [ ] Reviewer validation OOS bandes de probabilité (P0-20, P0-11) — 3 h
- [ ] Lancer **Audit Phase 2** (8 sections rescorrées) post-Sprint 4 : produire `audits/2026-Q3/algo_audit_phase2.md` — 12 h
- [ ] Rédiger `sprint_4_retrospective.md` — 3 h

## Sprint 5 (S11-S12) — Robustness & Stress Testing
- [ ] Reviewer plan fuzz testing (batch 5.1, P1-11) — 2 h
- [ ] Reviewer stress test multi-régime (COVID 2020, LDI 2022, SVB 2023, yen 2024) — 3 h
- [ ] Arbitrer si fenêtre conformal ACI doit être adaptée par régime (input Conformal Eng) — 2 h
- [ ] Rédiger `sprint_5_retrospective.md` — 3 h

## Sprint 6 (S13-S14) — Production Hardening
- [ ] Reviewer profiling + vectorisation (cible <250 ms / tick / paire, batch 6.1) — 3 h
- [ ] Reviewer snapshot store API per-signal (P0-16, batch 6.3) — 3 h
- [ ] Reviewer versioning JSON state machine (P1-12) + compat ascendante modèles — 2 h
- [ ] Archiver legacy `agents/market_regime_*` + `regime_predictor` (P1-15) — 2 h
- [ ] Rédiger `sprint_6_retrospective.md` — 3 h

## Sprint 7 (S15-S16) — Commercial Readiness
- [ ] Reviewer tear sheets par actif/TF (batch 7.2) — 4 h
- [ ] Test e2e 6 actifs × 2 TF (batch 7.4) — 4 h
- [ ] **Audit Phase 3 finale** : produire `audits/2026-Q4/certification.md` avec scores 8 sections et verdict commercial — 16 h
- [ ] Signer certification interne + verdict B2C/B2B — 4 h
- [ ] Rédiger `sprint_7_retrospective.md` + plan post-S7 — 4 h

## Inbox (non priorisé)
- Réflexion sur séparation `src/intelligence/` (online prod) vs `src/research/` (offline R&D) — formaliser un contrat d'interface.
- Évaluer migration TypedDict → Pydantic v2 sur tout le pipeline (cohérence avec P0-8).
- Documenter le SLA latence cible par module (currently implicite à 50 ms vol, 250 ms tick total).
- Politique de versioning sémantique pour les modèles (`models/a1_stack_v1.pkl` → schéma `<module>_v<major>.<minor>.pkl`).
- Préparer un kit "agent handover" si un rôle devient inactif (transfert de propriété ordonné).
