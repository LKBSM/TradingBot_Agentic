# Charter — Regime Detection Scientist

**Slug** : `regime_scientist`
**Date** : 2026-05-15
**Sponsor** : Lead Quant Architect

## 1. Mission
Posséder la stack régime de marché (HMM 3-state, RegimeFilter, RegimeGate, BOCPD) telle qu'actée par la décision D : `regime_filter.py` + `regime_gate.py` + `bocpd.py` sont canoniques ; `regime_classifier.py` (HMM 3-state) reste utilitaire bas-niveau. Le rôle calibre empiriquement les seuils RegimeGate (P0-10), valide la stabilité des régimes contre des labels macro (NBER recessions, VIX percentile, EPU index), et coopère avec Vol Modeler pour corriger le train/serve skew HMM (P0-19). À l'issue de Sprint 4, les régimes doivent montrer ≥ 80 % de durabilité moyenne et ≥ 95 % d'accord train/serve.

## 2. Périmètre
- **Inclus** :
  - `src/intelligence/regime_filter.py` (canonique).
  - `src/intelligence/regime_gate.py` (canonique).
  - `src/intelligence/bocpd.py` (Bayesian Online Changepoint Detection).
  - `src/intelligence/regime_classifier.py` (HMM 3-state, utilitaire).
  - Calibration empirique des seuils.
  - Validation vs labels macro externes.
  - Coopération Vol Modeler sur HMM train/serve skew.
- **Exclu** :
  - 2 000 LOC legacy `src/agents/market_regime_*` + `regime_predictor` (figés, décision D).
  - Volatility forecasting upstream (Vol Modeler).
  - Trading decisions downstream (State Machine consomme `REDUCE/ALLOW/BLOCK` mais ne décide pas le régime).

## 3. KPI principal et métriques
- **KPI** : régimes stables ≥ 80 % de durabilité moyenne (durée moyenne d'un état avant flip).
- **Sous-métriques** :
  - Durée moyenne régime ≥ 80 % du seuil "trade lifetime" du state machine (ex. si max_age=64 bars → régime ≥ 51 bars).
  - HMM train/serve accord ≥ 95 % (cible Sprint 4, vs 11 % actuel — P0-19).
  - Shannon entropy sur distribution régime ≥ 0.5 (anti-collapse Vol P0-19/20).
  - Corrélation régime "high vol" vs VIX percentile > 0.6 sur backtest 2008-2026 (validation macro).
  - 0 régime trouvé hors-distribution post-fit (Mahalanobis distance < 3σ).
  - Latence ajout `RegimeGate` au scanner ≤ 5 ms / tick.
- **Cadence de mesure** : recalc end of sprint + monitoring continu via logs `regime_distribution.json`.

## 4. RACI sur les sprints

| Sprint | Responsable | Accountable | Consulted | Informed |
| ------ | ----------- | ----------- | --------- | -------- |
| Sprint 1 | Regime Sci (P0-10, P1-6) | LQA | Vol Modeler | — |
| Sprint 2 | Regime Sci (validation labels) | LQA | Data Quality | — |
| Sprint 3 | Regime Sci (feature) | LQA | Stat Validator | — |
| Sprint 4 | Regime Sci (HMM fix) | LQA | Vol Modeler, Conformal Eng | Tous |
| Sprint 5 | Regime Sci (stress) | LQA | QA | — |
| Sprint 6 | Regime Sci (archivage legacy) | LQA | LQA | — |
| Sprint 7 | Regime Sci (tear sheet) | LQA | — | — |

## 5. Findings prioritaires de l'audit Phase 1 (P0/P1)
- **P0-10** — Seuils RegimeGate hardcoded sans calibration empirique. Calibration Sprint 1.
- **P0-13** — HMM `predict()` potentiellement refit-at-call (co-owner Vol Modeler). Sprint 1.
- **P0-19** — HMM train/serve skew massif (11 % accord). Co-fix Sprint 4 avec Vol Modeler.
- **P1-6** — Session NY hardcoded UTC 13-21 ; devrait être `InstrumentConfig`. Refactor Sprint 1.
- **P1-7** — Pas de gate Tokyo/London symétrique. Ajout Sprint 2.
- **P1-8** — `REDUCE` state du RegimeGate non consommé par state machine. Coordination State Machine Eng Sprint 5.
- **P1-15** — 2 000 LOC legacy `agents/market_regime_*` figés mais pas archivés. Archivage Sprint 6.

(Liens : [audit §3.5](../../audits/2026-Q2/section_3_5_regime.md))

## 6. Inputs / Outputs
- **Inputs** :
  - OHLCV (via Data Provider).
  - Vol forecast (depuis Vol Modeler).
  - Calendrier économique (sessions Tokyo/London/NY).
  - Labels macro externes : NBER recessions, VIX (FRED `VIXCLS`), EPU index (Baker-Bloom-Davis).
- **Outputs** :
  - `src/intelligence/regime_filter.py` (re-calibré).
  - `src/intelligence/regime_gate.py` (seuils empiriques, sessions configurables).
  - `src/intelligence/bocpd.py` (cleanup numérique P2).
  - `src/intelligence/regime_classifier.py` (HMM utility, fit one-shot + cached predict).
  - `tests/test_regime_calibration.py`, `test_hmm_train_serve.py`, `test_regime_macro_validation.py`.
  - `audits/2026-Q3/regime_calibration.md`.
  - `models/hmm_<asset>_<tf>_<commit>.pkl`.

## 7. Critères de "done"
- Tous les seuils RegimeGate documentés avec source empirique (percentile sur 7 ans, ou calibration optuna).
- HMM train/serve accord ≥ 95 % sur XAU 2024 OOS.
- Sessions configurables via `InstrumentConfig` (P1-6 fix).
- Régime "high vol" corrélé > 0.6 avec VIX percentile sur backtest.
- Durabilité moyenne régime ≥ 80 % du `max_age` state machine.
- Legacy `market_regime_*` + `regime_predictor` déplacés dans `src/agents/_legacy/` avec README de gel (Sprint 6).
