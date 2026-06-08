# Sprint 5 — Robustness & Stress Testing

**Période** : Semaines 11-12 (S11-S12, ~2026-07-25 → 2026-08-08)
**Charge estimée totale** : **64 h** productives + buffer 10 h = 74 h
**Objectif** : prouver que le système ne casse pas sous conditions extrêmes. Fuzz testing inputs (NaN/inf/gaps/spreads anormaux), stress test multi-régime (COVID 2020, LDI 2022, SVB 2023, yen 2024), sensibilité ±20 % hyperparamètres, adversarial inputs (fake-out setups). Sortie : un système qui dégrade gracefully au lieu de crash silencieux.
**Gate de sortie** : 0 crash sur 100 000 fuzz cases, dégradation PF ≤ 30 % sur crisis windows, robustesse hyperparams ≤ 25 % PF degradation sur ±20 %, REDUCE state state machine consommé.

---

## 0. Vue d'ensemble — 4 batches

| Batch | Titre                                                | Heures | Critique chemin |
| ----- | ---------------------------------------------------- | ------ | --------------- |
| 5.1   | Fuzz testing inputs (NaN/inf/gaps/spreads anormaux)  | 16 h   | ✅              |
| 5.2   | Stress test multi-régime (COVID/LDI/SVB/yen)         | 22 h   | ✅              |
| 5.3   | Sensibilité hyperparamètres ±20 %                    | 14 h   | ✅              |
| 5.4   | Adversarial inputs + REDUCE state state machine      | 12 h   | ✅              |
| —     | Buffer                                                | 10 h   |                 |
| **TOTAL** |                                                  | **74 h** |               |

---

## Batch 5.1 — Fuzz testing inputs (16 h)

### Objectif
Garantir que chaque entrée pathologique (NaN, infini, gap > 1h, spread 10×normal, volume=0, OHLC inversé, timestamps dupliqués) est soit rejetée par le contrat Pydantic v2 (Sprint 1.1), soit traitée gracefully sans crash. 100 000 fuzz cases via Hypothesis.

### Steps
1. **Catégorisation inputs pathologiques** (2 h)
   - `docs/algo/fuzz_taxonomy.md` :
     - **Numerical** : NaN, +inf, -inf, OHLC=0.
     - **Structural** : duplicate timestamps, non-monotonic, missing bars.
     - **Semantic** : low > high, close hors [low, high], volume<0.
     - **Adversarial** : spread soudain 10×, gap weekend non-anticipé, flash crash.

2. **Hypothesis strategies** (4 h)
   - `tests/fuzz/strategies.py` :
     - `ohlcv_strategy()` génère DF valides + invalides selon proba.
     - `pathological_strategy()` injecte NaN/inf à freq paramétrable.
   - 5 strategies différentes.

3. **Fuzz suite par module** (6 h)
   - `tests/fuzz/test_fuzz_data_layer.py` : OHLCVFrame contract.
   - `tests/fuzz/test_fuzz_smart_money.py` : detector ne crash pas.
   - `tests/fuzz/test_fuzz_vol_forecaster.py` : HAR-RV ne return pas NaN.
   - `tests/fuzz/test_fuzz_state_machine.py` : transitions valides.
   - `tests/fuzz/test_fuzz_logistic_scoring.py` : score ∈ [0, 100].
   - Chaque test : 10 000+ cases via `@given(max_examples=10_000)`.

4. **Collection bugs** (2 h)
   - Run suite, collect failures.
   - Trier par sévérité : crash (P0) vs warning (P1) vs ok (P2).

5. **Fix critical bugs** (1 h)
   - Patch P0 trouvés : raise `ValueError` cleanly, log warning, return safe default.

6. **Rapport** (1 h)
   - `reports/sprint_5/fuzz_test_report.md` :
     - Nombre cases / module.
     - Bugs trouvés (par sévérité).
     - Coverage robustesse.

### Critères d'acceptation
- ✅ 50 000+ fuzz cases exécutées au total.
- ✅ 0 crash (uncaught exception).
- ✅ Tous warnings loggés.
- ✅ Rapport signé.

### Findings audit adressés
- **P1-11** (Tests chaos / property-based manquants state machine) — ✅ closed.
- Renforce stabilité numérique BOCPD (P2).

### Dépendances
- Sprint 1 batch 1.1 (Pydantic contract).
- Sprint 1 batch 1.2 (no look-ahead).

### Risques
- Hypothesis génère cas pathologiques trop éloignés du réel → biaise les bugs. Mitigation : restreindre ranges aux distributions empiriques.

---

## Batch 5.2 — Stress test multi-régime (22 h)

### Objectif
Tester le pipeline sur 4 fenêtres historiques de stress majeur :
- **COVID 2020-03** (XAU crash + rebound, volatilité ×3).
- **LDI 2022-09** (GBP collapse, pension fund crisis).
- **SVB 2023-03** (US bank run, flight to safety).
- **Yen carry 2024-08** (JPY 13% en 3 jours).

### Steps
1. **Window definitions** (2 h)
   - `tests/stress/windows.py` :
     - COVID : 2020-02-15 → 2020-04-15 (XAU + EURUSD).
     - LDI : 2022-09-01 → 2022-10-31 (GBPUSD).
     - SVB : 2023-03-01 → 2023-03-31 (US500 + XAU).
     - Yen : 2024-08-01 → 2024-08-15 (USDJPY).

2. **Baseline metrics par window** (4 h)
   - Run pipeline complet sur chaque window.
   - Sortie : PF, Sharpe, max DD, nb trades, win rate.
   - Reference : `reports/sprint_5/stress_baseline.parquet`.

3. **Régime classification check** (3 h)
   - Vérifier que HMM identifie bien chaque crisis comme `high_vol` ou `crisis`.
   - Si non → bug regime detection à fixer.

4. **Calibration check** (3 h)
   - PICP sur chaque crisis window.
   - Si PICP < 50 % en crisis → bandes complètement OOS, ajouter régime crisis aux 3 HMM states.

5. **Behavior under stress** (4 h)
   - Vérifier que :
     - State machine entre en `REDUCE` ou `CLOSE_ALL` en crisis.
     - Position sizing diminue (Kelly fraction).
     - LLM narratives ne hallucine pas (templates en fallback).
   - Si comportement OK → robust.

6. **Comparison normal vs crisis** (3 h)
   - PF crisis / PF normal ratio.
   - Cible : ratio ≥ 0.7 (dégradation ≤ 30 %).

7. **Rapport** (3 h)
   - `reports/sprint_5/stress_test_report.md` :
     - Tableau par crisis : PF, Sharpe, regime, PICP.
     - Verdict par crisis (robust / dégradé / cassé).
     - Recommandations.

### Critères d'acceptation
- ✅ 4 stress windows exécutées.
- ✅ Régime identifié comme high_vol/crisis dans ≥ 3/4 windows.
- ✅ PF ratio crisis/normal ≥ 0.7 sur ≥ 2/4 windows.
- ✅ PICP crisis ≥ 65 % (tolerance crise).

### Findings audit adressés
- Renforce Sprint 4 calibration sur OOS crisis.
- Indirect P1-7 (Pas de gate Tokyo/London) si overlap crisis détecté.

### Dépendances
- Sprint 4 (calibration en place).
- Sprint 1 batch 1.5 (CSV BTC/US500/GBP/JPY pour LDI/SVB/yen).

### Risques
- CSV coverage insuffisante sur crisis windows (GBP 2022-09 spécifique) → fallback Investing.com 1h.
- PF effondre en crisis → soit normal (no edge in crisis = no trade), soit bug.

---

## Batch 5.3 — Sensibilité hyperparamètres ±20 % (14 h)

### Objectif
Pour chaque hyperparam clé (~15), tester ±10 % et ±20 % autour de l'optimum Sprint 3. PF doit dégrader ≤ 25 %. Si effondre → l'optimum est un peak instable, refit ou abandonner.

### Steps
1. **Hyperparams list** (1 h)
   - `enter_threshold`, `exit_threshold`, `cooldown_bars`, `max_age_bars`, `confirm_bars`, `FVG_THRESHOLD_ATR`, `RETEST_TOL_ATR`, `ARMED_WINDOW`, `OB_VOL_MULT`, `mondrian_alpha`, `logistic_C`, `vol_forecast_alpha`, `regime_threshold_vol`, `regime_threshold_trend`, `position_size_fraction`.

2. **Grid ±10 % / ±20 %** (1 h)
   - Pour chaque hyperparam, 4 perturbations (-20 %, -10 %, +10 %, +20 %).
   - 15 × 4 = 60 runs par config × 4 configs = 240 runs.

3. **Backtest grid** (6 h compute)
   - Run sur XAU M15, EUR M15 (priorité).
   - Cache hash-based.

4. **Compute PF degradation** (2 h)
   - Pour chaque (config, hyperparam, perturbation) : `PF_perturbed / PF_baseline`.
   - Sortie : `reports/sprint_5/sensitivity_matrix.parquet`.

5. **Robustness score** (2 h)
   - Pour chaque hyperparam : worst PF degradation sur 4 perturbations.
   - Cible : ≥ 0.75 (PF dégrade ≤ 25 %).
   - Flag hyperparams "fragiles" (< 0.75).

6. **Rapport** (2 h)
   - `reports/sprint_5/sensitivity_report.md` :
     - Heatmap PF degradation par hyperparam × perturbation.
     - Liste hyperparams fragiles.
     - Recommandations refit / widening grids Sprint 3.

### Critères d'acceptation
- ✅ 240 runs exécutés.
- ✅ ≥ 80 % hyperparams ont robustness ≥ 0.75.
- ✅ Rapport signé.

### Findings audit adressés
- Renforce Sprint 3 batch 3.5 (sweep) en validant robustesse de l'optimum.

### Dépendances
- Sprint 3 batch 3.5 (sweep optima identifiés).

### Risques
- > 5 hyperparams fragiles → optimum surfit, escalade pivot Sprint 6 refit.

---

## Batch 5.4 — Adversarial inputs + REDUCE state state machine (12 h)

### Objectif
Tester contre setups fake-out (faux BOS, OB sans flow, stop-runs) et activer le REDUCE state du RegimeGate dans la state machine (P1-8 actuellement non consommé).

### Steps
1. **Catalog fake-out patterns** (2 h)
   - `docs/algo/adversarial_patterns.md` :
     - **Fake BOS** : cassure suivie de reversal en < 5 bars.
     - **Stop-run** : wick beyond level, close back.
     - **Liquidity sweep** : sweep highs/lows session.
     - **News spike fake** : spike post-news avec mean reversion.

2. **Synthetic adversarial dataset** (3 h)
   - Générer 200 cas synthétiques de chaque pattern.
   - Sauvegarder `data/adversarial/fake_setups.parquet`.

3. **Test detector behavior** (2 h)
   - Lancer `SmartMoneyEngine` sur dataset.
   - Mesurer taux rejection (cible ≥ 70 % fake-outs rejetés).

4. **Wire REDUCE state** (3 h) — P1-8
   - `signal_state_machine.py` :
     - Consume `RegimeGate.signal_action ∈ {ENTER, REDUCE, EXIT}`.
     - Si REDUCE : reduce position size 50 %, pas exit complet.
     - Si EXIT : close immédiat.

5. **Tests REDUCE** (1 h)
   - `tests/test_state_machine_reduce.py` :
     - Transition normal → REDUCE → normal après crisis.
     - Position size diminue puis remonte.

6. **Rapport** (1 h)
   - `reports/sprint_5/adversarial_report.md` :
     - Taux rejection par pattern.
     - REDUCE wiring validation.

### Critères d'acceptation
- ✅ ≥ 70 % fake-outs rejetés.
- ✅ REDUCE state consommé par state machine.
- ✅ Tests REDUCE verts.

### Findings audit adressés
- **P1-8** (REDUCE state non consommé) — ✅ closed.
- Préparation Sprint 6 production hardening.

### Dépendances
- Sprint 2 (detector tuné).

### Risques
- Synthetic adversarial trop simple → detector passe tout. Mitigation : utiliser real cases sweep liquidity XAU 2024.

---

## Gate de sortie du Sprint 5 (checklist 10 items)

1. ✅ 50 000+ fuzz cases, 0 crash.
2. ✅ 4 stress windows exécutées (COVID/LDI/SVB/yen).
3. ✅ Régime identifié crisis dans ≥ 3/4 windows.
4. ✅ PF ratio crisis/normal ≥ 0.7 sur ≥ 2/4.
5. ✅ Sensibilité ±20 % : ≥ 80 % hyperparams robustes.
6. ✅ Adversarial rejection ≥ 70 %.
7. ✅ REDUCE state wiré + tests.
8. ✅ Suite tests verte.
9. ✅ Rapports stress + sensitivity + adversarial signés.
10. ✅ `sprint_5_retrospective.md` rédigé.

---

## Livrables Sprint 5 (arborescence)

```
tests/fuzz/
  ├── strategies.py
  ├── test_fuzz_data_layer.py
  ├── test_fuzz_smart_money.py
  ├── test_fuzz_vol_forecaster.py
  ├── test_fuzz_state_machine.py
  └── test_fuzz_logistic_scoring.py

tests/stress/
  ├── windows.py
  ├── test_stress_covid_2020.py
  ├── test_stress_ldi_2022.py
  ├── test_stress_svb_2023.py
  └── test_stress_yen_2024.py

tests/
  └── test_state_machine_reduce.py

data/adversarial/
  └── fake_setups.parquet

src/intelligence/state_machine/
  └── signal_state_machine.py  # patched (REDUCE wiring)

reports/sprint_5/
  ├── fuzz_test_report.md
  ├── stress_baseline.parquet
  ├── stress_test_report.md
  ├── sensitivity_matrix.parquet
  ├── sensitivity_report.md
  └── adversarial_report.md

docs/algo/
  ├── fuzz_taxonomy.md
  └── adversarial_patterns.md

roadmap/sprints/
  ├── sprint_5.md
  ├── sprint_5_progress.md
  └── sprint_5_retrospective.md
```

---

## Décisions ouvertes pour user

1. **Tolérance dégradation PF crisis** : 30 % acceptable ? Si non → reconcevoir crisis handling.
2. **Hyperparams fragiles** : si > 5 fragiles → refit Sprint 3 avec regularization forte ?
3. **Synthetic vs real adversarial** : compléter avec real cases sweep liquidity XAU 2024 ?

---

**Signé** : Claude, 2026-05-15
