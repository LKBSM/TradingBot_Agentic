# Charter — SMC/ICT Detection Lead

**Slug** : `smc_detection_lead`
**Date** : 2026-05-15
**Sponsor** : Lead Quant Architect

## 1. Mission
Posséder la détection Smart Money Concepts / ICT (BOS, CHOCH, Order Blocks, Fair Value Gaps, retest) et la rendre conforme à la définition expert ICT. À l'issue de Sprint 0, la détection est éparpillée dans `src/environment/strategy_features.py` (1 213 LOC mélangées) avec deux bugs logiques actés et un bug structurel (OB ≠ ICT — engulfing seule, 40 % des OB sans BOS dans ±20 bars). Le rôle livre Sprint 1 le module `src/intelligence/smart_money/` autonome et contractualisé, Sprint 2 un dataset annoté manuellement (≥ 500 setups par actif) et atteint F1 ≥ 0.85 vs annotations expertes.

## 2. Périmètre
- **Inclus** :
  - Détection BOS, CHOCH, Order Block (def ICT — réaction post-BOS), Fair Value Gap, retest (entry).
  - Module `src/intelligence/smart_money/` (à créer Sprint 1 batch 1.0).
  - Dataset d'annotations expertes (Sprint 2).
  - Métriques F1 / precision / recall vs annotations.
  - Tuning bayésien hyperparams par actif (Sprint 2 batch 2.3).
  - Audit visuel automatisé (snapshots PNG par setup, Sprint 2 batch 2.4).
- **Exclu** :
  - Indicateurs techniques classiques (ADX, RSI, MACD) — restent dans `strategy_features.py`.
  - Régime / volatility — autres owners.
  - Stratégie d'entry/exit autour du retest — délégué à State Machine.
  - Scoring confluence — Stat Validator (refonte Sprint 4).

## 3. KPI principal et métriques
- **KPI** : F1 ≥ 0.85 vs annotations expertes pour BOS et CHOCH (XAU + EURUSD M15 minimum).
- **Sous-métriques** :
  - F1 BOS (XAU M15) ≥ 0.85
  - F1 CHOCH (XAU M15) ≥ 0.85
  - F1 Order Block (def ICT post-BOS) ≥ 0.75 (plus difficile car définition stricte)
  - F1 FVG ≥ 0.80
  - Firing rate BOS/CHOCH cohérent (XAU M15 actuel 3.16 % → cible 2-4 %, EURUSD 2.96 %).
  - 0 % des OB sans BOS associé dans ±20 bars (actuel 40 %).
  - Performance : détection < 50 ms / 1000 bars (avec Numba).
- **Cadence de mesure** : recalc à chaque commit module, audit consolidé fin Sprint 2.

## 4. RACI sur les sprints

| Sprint | Responsable | Accountable | Consulted | Informed |
| ------ | ----------- | ----------- | --------- | -------- |
| Sprint 1 | SMC Lead (batch 1.0) | LQA | Data Quality, QA | Tous |
| Sprint 2 | SMC Lead | LQA | Data Quality, QA, Stat Validator | Tous |
| Sprint 3 | SMC Lead (support feature) | LQA | Stat Validator | — |
| Sprint 4 | — | LQA | Stat Validator (refonte score) | — |
| Sprint 5 | SMC Lead (fuzz adversarial) | LQA | QA | — |
| Sprint 6 | SMC Lead (perf Numba) | LQA | Backtest Infra | — |
| Sprint 7 | SMC Lead (tear sheets SMC) | LQA | LQA | — |

## 5. Findings prioritaires de l'audit Phase 1 (P0/P1)
- **P0-2** — Order Blocks ≠ définition ICT (engulfing seule, 40 % OB sans BOS dans ±20 bars). Refonte def OB Sprint 1.0 / 2.1.
- **P0-9** — Smart Money pas extrait en module dédié. Création `src/intelligence/smart_money/` Sprint 1.0 (décision E).
- **P0-15** — Bug RSI Divergence : compare wrong bar index (`strategy_features.py:849-857`). Fix Sprint 1.0.
- **P1-1** — Magic number incohérent : `armed_window=5` vs `RETEST_ARMED_WINDOW=30`. Fix Sprint 1.0.
- **P1-2** — FVG_THRESHOLD=0.1 ATR ≈ spread XAU (trop laxe). Tuning Sprint 2.
- **P1-3** — RETEST_TOL_ATR=0.5 ATR ≈ 1.5 $ ≈ spread XAU. Tuning Sprint 2.

(Liens : [audit §3.2](../../audits/2026-Q2/section_3_2_smart_money.md), [stats SMC](../../audits/2026-Q2/section_3_2_smart_money_stats.json))

## 6. Inputs / Outputs
- **Inputs** :
  - Dataset OHLCV (via Data Provider Pydantic v2).
  - Annotations expertes (créées Sprint 2 par owner).
  - Référence ICT : Inner Circle Trader Mentorship, Market Maker Buy/Sell Model.
- **Outputs** :
  - `src/intelligence/smart_money/__init__.py`, `bos.py`, `choch.py`, `order_block.py`, `fvg.py`, `retest.py`, `smart_money_engine.py`.
  - `data/annotations/smc_<asset>_<tf>.json` (annotations expertes, ≥ 500/actif).
  - `tests/test_smart_money_*.py` (unitaires + property-based + régression annotations).
  - `audits/2026-Q3/smc_validation_<asset>.md` (F1/P/R par actif).
  - `reports/smc/snapshots_<asset>/` (PNG audit visuel batch 2.4).

## 7. Critères de "done"
- Module `src/intelligence/smart_money/` autonome (zéro import depuis `src/environment/strategy_features.py`).
- 2 bugs P0/P1 corrigés et couverts par tests régression (`test_smart_money_retest_armed_window.py`, `test_smart_money_rsi_div_indexing.py`).
- F1 ≥ 0.85 BOS+CHOCH XAU M15 + EURUSD M15 vs annotations.
- OB def ICT-conforme : ≥ 95 % des OB ont un BOS associé dans ±20 bars (vs 60 % actuel).
- Detection time < 50 ms / 1000 bars (avec Numba activé en CI).
- Audit visuel batch 2.4 disponible (≥ 50 PNG par actif).
