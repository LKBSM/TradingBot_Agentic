# Charter — Conformal Calibration Engineer

**Slug** : `conformal_engineer`
**Date** : 2026-05-15
**Sponsor** : Lead Quant Architect

## 1. Mission
Posséder le `ConformalWrapper` (Split conformal + ACI — Gibbs & Candès 2021) et garantir que les bandes de confiance produites par l'algorithme respectent les garanties théoriques de couverture marginale. À l'issue de Sprint 0, le wrapper est codé proprement (7.0/10) mais le PICP empirique est non mesuré (P0-11) et l'écosystème vol affiche un PICP catastrophique 43.6 % vs cible 80 % (P0-20). Le rôle livre Sprint 4 une couverture marginale = nominal ± 2 % sur OOS XAU + EURUSD, ajoute Mondrian stratifié par régime (P1-9), et vérifie l'exchangeabilité empiriquement.

## 2. Périmètre
- **Inclus** :
  - `src/intelligence/conformal_wrapper.py` (Split + ACI).
  - Mondrian conformal stratifié par régime (Sprint 4 batch 4.1).
  - PICP measurement OOS (Sprint 4 batch 4.3).
  - Bandes de probabilité downstream LLM (input pour delivery).
  - Documentation théorique (Angelopoulos & Bates 2024 reference).
  - Property-based tests d'exchangeabilité.
- **Exclu** :
  - Modèle base (Vol Modeler — fournit forecasts).
  - Calibration des résiduals (Vol Modeler ou Stat Validator).
  - Régime labels (Regime Scientist — fournit pour stratification Mondrian).

## 3. KPI principal et métriques
- **KPI** : couverture marginale = nominal ± 2 % sur OOS XAU + EURUSD 2024.
- **Sous-métriques** :
  - PICP empirique 80 % cible → [78 %, 82 %] (actuel 43.6 % sur XAU 2024).
  - PICP empirique 90 % cible → [88 %, 92 %].
  - Conditional coverage par régime (Mondrian) : chaque stratum ∈ [nominal − 5 %, nominal + 5 %].
  - Interval width median : pas plus large que naive ±2σ + 20 % (efficiency).
  - ACI adaptation latency : converge en ≤ 100 obs après changement de distribution (régime flip).
  - 0 NaN / Inf dans les intervalles produits sur 7 ans XAU.
- **Cadence de mesure** : recalc end of Sprint 4 + monitoring continu post-Sprint 4.

## 4. RACI sur les sprints

| Sprint | Responsable | Accountable | Consulted | Informed |
| ------ | ----------- | ----------- | --------- | -------- |
| Sprint 1 | — (frozen) | LQA | — | — |
| Sprint 2 | — | LQA | — | — |
| Sprint 3 | Conformal Eng (prep) | LQA | Stat Validator | — |
| Sprint 4 | Conformal Eng (pivot batches 4.1, 4.3) | LQA | Vol Modeler, Regime Sci | Tous |
| Sprint 5 | Conformal Eng (stress) | LQA | QA | — |
| Sprint 6 | Conformal Eng (perf) | LQA | Backtest Infra | — |
| Sprint 7 | Conformal Eng (tear sheet) | LQA | — | — |

## 5. Findings prioritaires de l'audit Phase 1 (P0/P1)
- **P0-11** — PICP conformelle non mesurée OOS. Mesure Sprint 4 batch 4.2.
- **P0-20** — PICP conformal catastrophique 43.6 % vs cible 80 % sur XAU 2024 OOS. Fix Sprint 4 batches 4.1 + 4.2.
- **P1-9** — Conformal sans Mondrian (stratification par régime). Ajout Sprint 4.1.
- Référence : sur stack actuel, le wrapper **rejette tout** (correct sur weak edge) → opérationnellement = 0 trade. Le wrapper attend qu'un edge prédictif existe (Sprint 3 doit précéder Sprint 4).

(Liens : [audit §3.6](../../audits/2026-Q2/section_3_6_conformal.md))

## 6. Inputs / Outputs
- **Inputs** :
  - Forecasts vol (depuis Vol Modeler).
  - Résiduals OOS (Vol Modeler + Stat Validator).
  - Labels régime (depuis Regime Scientist) pour Mondrian.
  - Score logistic Sprint 4 (depuis Stat Validator) pour conformal prediction sur probabilité de win.
- **Outputs** :
  - `src/intelligence/conformal_wrapper.py` (re-calibré + Mondrian).
  - `src/intelligence/conformal_mondrian.py` (nouveau, Sprint 4).
  - `tests/test_conformal_picp.py`, `test_conformal_exchangeability.py`, `test_conformal_mondrian.py`.
  - `audits/2026-Q3/conformal_calibration.md`.
  - Documentation `docs/algo/conformal.md`.

## 7. Critères de "done"
- PICP empirique nominal 80 % atteint à ±2 % sur OOS 2024 XAU + EURUSD.
- Mondrian stratifié par régime (3 strata) : conditional coverage à ±5 % par stratum.
- Property-based test d'exchangeabilité (`tests/test_conformal_exchangeability.py`) : permutations aléatoires des observations dans la fenêtre de calibration n'altèrent pas significativement les intervalles.
- ACI adaptation prouvée : convergence ≤ 100 obs après régime flip artificiel.
- Documentation théorique avec références (Angelopoulos & Bates 2024, Gibbs & Candès 2021, Vovk 2005).
- Tear sheet conformal (Sprint 7).
