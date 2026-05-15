# Backlog — Regime Detection Scientist

**Date** : 2026-05-15
**Owner** : Regime Detection Scientist

## Sprint 1 (S3-S4) — Calibration empirique seuils

- [ ] **Fix P0-10** — Calibration empirique seuils RegimeGate sur 7 ans XAU + EURUSD : trouver les percentiles (vol_regime, trend_regime) qui maximisent un proxy d'edge (PF, Sharpe non-déflaté) — 8 h
- [ ] **Fix P1-6** — Refactor sessions Tokyo/London/NY : ajouter `InstrumentConfig.session_tokyo / session_london / session_ny` (tuples UTC) ; retirer hardcoded 13-21. Tests régression — 4 h
- [ ] Co-owner P0-13 avec Vol Modeler : confirmer fit one-shot HMM (rôle utilitaire) — 3 h
- [ ] Logging structuré régime : `regime_distribution.json` par session backtest, format `{timestamp, regime_id, regime_label, prob}` — 3 h
- [ ] Test `tests/test_regime_thresholds_calibrated.py` : assert que `regime_gate.thresholds` charge depuis JSON empirique (pas hardcoded) — 2 h

## Sprint 2 (S5-S6) — Validation macro

- [ ] Ingester VIX (FRED `VIXCLS`) et NBER recession dates (FRED `USREC`) en `data/macro/` — 3 h
- [ ] Test corrélation : régime "high vol" vs VIX percentile > 80 ; cible corr ≥ 0.6 — 4 h
- [ ] Test corrélation : régime "low trend" vs NBER recession periods — 3 h
- [ ] **Fix P1-7** — Ajouter gate Tokyo/London symétrique (actuellement seulement NY) ; tester sur EURUSD (session London = mouvement principal) — 5 h
- [ ] Audit visuel : régime timeline overlaid sur XAU equity 2019-2026 (matplotlib) — 3 h

## Sprint 3 (S7-S8) — Edge Discovery (feature input)

- [ ] Exposer features régime à Stat Validator : `regime_label` (one-hot 3 states), `regime_persistence` (bars since flip), `bocpd_changepoint_prob`, `vol_regime`, `trend_regime` (batch 3.1) — 4 h
- [ ] Stacking + conditionnement par régime (batch 3.3 input) : fournir les datasets stratifiés par régime à Stat Validator — 5 h
- [ ] Information Coefficient feature régime par stratum — 3 h

## Sprint 4 (S9-S10) — Calibration & Confidence (HMM fix)

- [ ] **Fix P0-19** — HMM train/serve skew (co-owner Vol Modeler). Décision : unifier en utilisant `forward-filter` only (pas Viterbi smoothing) en inférence ; refit weekly seulement. Cible accord ≥ 95 % — 8 h
- [ ] Test `tests/test_hmm_train_serve_accord.py` : sur XAU 2024 OOS, accord Viterbi(full)|predict(online) ≥ 95 % — 4 h
- [ ] Coopération Conformal Eng pour **Mondrian stratifié par régime** (P1-9) : fournir labels régime stables (post-fix) pour stratification — 3 h
- [ ] Audit empirique : Shannon entropy régime distribution ≥ 0.5 sur OOS (anti-collapse Vol P0-20) — 2 h
- [ ] Rapport `audits/2026-Q3/regime_calibration_sprint_4.md` — 3 h

## Sprint 5 (S11-S12) — Robustness & Stress Testing

- [ ] Stress test régime stack sur 4 crisis (COVID 2020, LDI 2022, SVB 2023, yen 2024) : vérifier non-divergence BOCPD, HMM ne crash pas (batch 5.2) — 6 h
- [ ] Coordination State Machine Eng pour P1-8 : implémenter consommation `REDUCE` state (réduction position size en regime "stress") — 4 h
- [ ] Property-based : invariance HMM par scaling prix ×10 — 3 h
- [ ] Stabilité BOCPD : overflow log-probs (P2 audit) — 2 h

## Sprint 6 (S13-S14) — Production Hardening

- [ ] **Fix P1-15** — Archiver legacy : `git mv src/agents/market_regime_agent.py src/agents/_legacy/` + `regime_predictor.py` idem. README de gel : "Frozen 2026-05-XX par décision D, ne pas modifier". Tests existants migrés ou skip — 5 h
- [ ] Versioning HMM modèles : `models/hmm_<asset>_<tf>_v<n>.pkl` + load avec validation version (P0-16 support) — 3 h
- [ ] Cleanup numérique BOCPD : passer en log-space pour les products, éviter underflow (P2) — 3 h

## Sprint 7 (S15-S16) — Commercial Readiness

- [ ] Tear sheet régime par actif : distribution régime sur 7 ans, transitions matrix, performance par régime (PF par régime) (batch 7.2) — 5 h
- [ ] Documentation `docs/algo/regime_stack.md` (HMM choice, BOCPD theory, references Adams-MacKay 2007) — 4 h
- [ ] Fiche transparence client : "comment nous détectons les régimes" (batch 7.3) — 2 h

## Inbox (non priorisé)
- HSMM (Hidden Semi-Markov Model) — durée explicite des régimes (vs HMM géométrique).
- Online HMM (Stochastic EM) pour adaptation continue — réserve Sprint 8+.
- Multi-asset régime joint (BTC corrélation, USD index DXY).
- Regime forecasting (probabilité de flip dans N bars) — feature additionnelle.
- BOCPD avec hazard function variable (vs constante).
- EPU index (Baker-Bloom-Davis) ajout au stack — différé.
