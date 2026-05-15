# Agents Roster — Institutional Overhaul Team

**Date d'instanciation** : 2026-05-15
**Sprint courant** : Sprint 0 (livré) → Sprint 1 (à démarrer après validation user)
**Branche** : `institutional-overhaul` (depuis tag `v0.9.0-pre-institutional`)
**Note audit Phase 1** : 5.61 / 10 pondérée (21 P0 + 15 P1 + 7 P2)
**Référence findings** : `audits/2026-Q2/algo_audit_institutional.md`
**Décisions actées** : `audits/2026-Q2/sprint_0_decisions.md` (14 décisions A→N)

---

## Table récapitulative

| # | Rôle | Slug | KPI principal | Charter | Backlog |
| - | ---- | ---- | ------------- | ------- | ------- |
| 1 | Lead Quant Architect | `lead_quant_architect` | Cohérence pipeline end-to-end (zéro contradiction inter-modules en revue PR) | [Charter](lead_quant_architect/CHARTER.md) | [Backlog](lead_quant_architect/BACKLOG.md) |
| 2 | Data Quality Engineer | `data_quality_engineer` | Coverage ≥ 99 % sur tous CSV MVP, look-ahead = 0 prouvé par property-test | [Charter](data_quality_engineer/CHARTER.md) | [Backlog](data_quality_engineer/BACKLOG.md) |
| 3 | SMC/ICT Detection Lead | `smc_detection_lead` | F1 ≥ 0.85 BOS/CHOCH vs annotations expertes, OB ICT-conforme | [Charter](smc_detection_lead/CHARTER.md) | [Backlog](smc_detection_lead/BACKLOG.md) |
| 4 | Volatility Modeler | `volatility_modeler` | QLIKE en baisse vs naive ; PICP 80 % cible atteint ±2 % OOS | [Charter](volatility_modeler/CHARTER.md) | [Backlog](volatility_modeler/BACKLOG.md) |
| 5 | Regime Detection Scientist | `regime_scientist` | Durée moyenne régime ≥ 80 % stable ; HMM train/serve accord ≥ 95 % | [Charter](regime_scientist/CHARTER.md) | [Backlog](regime_scientist/BACKLOG.md) |
| 6 | Conformal Calibration Engineer | `conformal_engineer` | Couverture marginale = nominal ± 2 % sur OOS 2024 | [Charter](conformal_engineer/CHARTER.md) | [Backlog](conformal_engineer/BACKLOG.md) |
| 7 | State Machine Engineer | `state_machine_engineer` | 0 oscillation parasite sur 10⁶ tirages property-based | [Charter](state_machine_engineer/CHARTER.md) | [Backlog](state_machine_engineer/BACKLOG.md) |
| 8 | Statistical Validator | `statistical_validator` | 100 % des signaux reportés passent CPCV + DSR + PBO + DM | [Charter](statistical_validator/CHARTER.md) | [Backlog](statistical_validator/BACKLOG.md) |
| 9 | Backtest Infrastructure | `backtest_infrastructure` | Backtest 7 ans / 1 paire < 2 min ; reproductibilité bit-à-bit | [Charter](backtest_infrastructure/CHARTER.md) | [Backlog](backtest_infrastructure/BACKLOG.md) |
| 10 | QA & Robustness | `qa_robustness` | Couverture branche ≥ 90 % ; mutation score ≥ 70 % | [Charter](qa_robustness/CHARTER.md) | [Backlog](qa_robustness/BACKLOG.md) |

---

## Périmètre et frontières

| Sujet | Owner principal | Reviewer obligatoire |
| ----- | --------------- | -------------------- |
| `src/intelligence/data_providers.py`, CSV, calendrier économique | Data Quality | LQA |
| `src/intelligence/smart_money/` (à créer Sprint 1.0) | SMC Lead | LQA, QA |
| `src/intelligence/volatility_forecaster.py`, `volatility_lgbm.py` | Vol Modeler | Stat Validator, LQA |
| `src/intelligence/regime_filter.py`, `regime_gate.py`, `bocpd.py`, HMM | Regime Scientist | Vol Modeler, LQA |
| `src/intelligence/conformal_wrapper.py` | Conformal Engineer | Stat Validator, LQA |
| `src/intelligence/signal_state_machine.py`, `state_persistence.py` | State Machine Eng | QA, LQA |
| `src/research/cpcv_harness.py`, `strategy_gates.py` | Stat Validator | LQA |
| `src/backtest/`, `scripts/run_backtest.py`, `state_machine_replay.py` | Backtest Infra | Stat Validator, QA |
| `tests/`, CI workflows, fuzz, mutation | QA & Robustness | LQA |
| `src/intelligence/confluence_detector.py` (refonte Sprint 4) | Stat Validator + Vol Modeler (co-owners) | LQA |

Toute modification cross-domain (≥ 2 modules d'owners différents) requiert revue du LQA.

---

## Plan de communication

1. **Daily async** (texte) — chaque agent dépose un statut en fin de batch dans `roadmap/sprints/sprint_X_progress.md` : ✅ ok / 🟡 partiel / ❌ bloqué + 3-5 lignes.
2. **Reviews PR** — toute PR cross-domaine est tagguée par l'owner pour le LQA + 1 reviewer obligatoire (cf. matrice ci-dessus). Pas de merge auto.
3. **Escalade bloquage** — un agent bloqué > 4 h ouvre un fichier `agents/<slug>/BLOCKERS.md` (créé à la demande) avec la nature du blocage et le module amont attendu. Le LQA arbitre dans les 24 h.
4. **Décisions techniques** — toute décision qui dévie de `audits/2026-Q2/sprint_0_decisions.md` requiert un amendement dans ce registre (PR dédiée).
5. **Pas de communication user spontanée** — auto-mode : interruption user uniquement aux gates de sortie de sprint (cf. décision L).

---

## Cadence des reviews

| Cadence | Type | Owner | Sortie |
| ------- | ---- | ----- | ------ |
| Fin de batch (~2-15 h) | Statut + commit | Agent owner | Append `sprint_X_progress.md` |
| Fin de sprint (~2 semaines / ~60-80 h) | Rétrospective + gate | LQA (consolide) | `sprint_X_retrospective.md` + GO/NO-GO Sprint X+1 |
| Mi-sprint (S+1 sem) | Health check inter-agents | LQA | Note inline dans `sprint_X_progress.md` |
| Audit Phase 2 (entre Sprint 4 et 5) | Re-scoring 0-10 sur 8 sections | LQA + sub-agents par section | `audits/2026-Q3/algo_audit_phase2.md` |
| Audit Phase 3 (post-Sprint 7) | Certification commerciale | LQA + Stat Validator + QA | `audits/2026-Q4/certification.md` + gate finale |

Les gates de sortie de sprint suivent le brief §5 :
- **Sprint 3** : CI 95 % PF lo > 1.0 sur ≥ 1 actif (sinon pivot valeur explicative documenté).
- **Sprint 7** : tear sheets démontrables B2C / B2B, certification interne signée.

---

## RACI cross-sprint (matrice condensée)

R = Responsible (exécute) · A = Accountable (signe) · C = Consulted · I = Informed

| Sprint | LQA | Data | SMC | Vol | Regime | Conformal | StateM | StatVal | Backtest | QA |
| ------ | --- | ---- | --- | --- | ------ | --------- | ------ | ------- | -------- | --- |
| 1 (Data Layer) | A,C | R | C | I | I | I | I | I | C | C |
| 2 (Detection) | A,C | C | R | I | C | I | I | C | I | C |
| 3 (Edge Discovery) | A,C | I | C | C | C | I | C | R | R | C |
| 4 (Calibration) | A,C | I | I | R | C | R | I | C | I | C |
| 5 (Robustness) | A,C | I | I | C | I | I | C | I | C | R |
| 6 (Prod Hardening) | A,R | I | I | I | I | I | C | I | R | C |
| 7 (Commercial) | A,R | I | C | I | I | I | I | R | C | C |

---

**Signé** : 2026-05-15, Claude (Lead Quant Architect)
**Version** : v1.0 (instanciation)
