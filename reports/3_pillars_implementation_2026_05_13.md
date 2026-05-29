# 3-Pillar Implementation Results — 2026-05-13

**Status**: Pillars 1+2+3 BUILT, TESTED, and EVALUATED against real XAU M15 data + 7 years of HIGH-impact events.

**Plan reference**: `reports/institutional_quant_transformation_plan.md` §4-§6.

---

## TL;DR

| | Built | Tests | Real-data eval |
|---|---|---|---|
| Sprint 0 — strategy gates | ✅ | 12/12 pass | CPCV+DSR+PBO+DM+PF-lo integrated |
| Pillar 1 — Event-Driven Macro | ✅ | 9/9 pass | **FAILS gates honestly** (DSR=0.65, PF_lo=0.83) |
| Pillar 2 — Conformal Wrapper (Split + ACI) | ✅ | 14/14 pass | Correctly rejects weak-edge signal |
| Pillar 3 — Regime Gate (BOCPD + bipower jumps) | ✅ | 11/11 pass | +0.16 DSR improvement on Pillar 1 |
| **Total new tests** | — | **46/46 pass** | — |

**Empirical verdict**: the naive event-driven strategy does NOT pass admission gates on real XAU+FF data. This is consistent with A1 verdict (DSR=0) and confirms the institutional plan's mid-Sprint 3 condition: **0/3 pillars passing → pivot B2B-API** is on the table.

The methodology (gates + pillars) is now installed correctly. What is missing is a strategy that actually produces edge — and the framework is now able to detect, with academic rigor, when one does.

---

## Files created

### Sprint 0
- `src/research/strategy_gates.py` — unified CPCV+DSR+PBO+DM+PF-lo gate, `evaluate_gates()` API and `assert_passes_gates()` strict variant
- `tests/test_strategy_gates.py` — 12 tests including profitable-strategy-passes, zero-edge-fails, PBO with explicit paths

### Pillar 1 — Event-Driven Macro
- `src/strategies/event_driven_macro.py` — `EventDrivenMacroStrategy` (momentum breakout T..T+30min around HIGH-impact events; SL/TP/TIME exit; R-multiple output)
- `tests/test_event_driven_macro.py` — 9 tests including synthetic event, ATR computation, integration with gates
- `scripts/eval_event_driven_macro.py` — end-to-end eval script, 4 variants (naive / +conformal / +regime / +both)

### Pillar 2 — Conformal Prediction Wrapper
- `src/intelligence/conformal_wrapper.py` — `SplitConformalScorer` + `AdaptiveConformalScorer` (Gibbs-Candès 2021 ACI) + `apply_conformal_filter` reject-option
- `tests/test_conformal_wrapper.py` — 14 tests including coverage validation under i.i.d. and shift, reject-option logic, batch filter

### Pillar 3 — Regime Gate
- `src/intelligence/regime_gate.py` — `RegimeGate` combining the existing BOCPD with new bipower-variation jump detector (Barndorff-Nielsen-Shephard 2004)
- `tests/test_regime_gate.py` — 11 tests including BV/RV agreement under no-jumps, jump detection on synthetic spike, changepoint reaction

---

## Empirical results on real data

**Data**: 172,875 M15 XAU bars (2019-2026) + 875 HIGH-impact FF events (2019-2025), filtered to USD + curated keyword list → 329 candidate trades.

| Variant | n trades | Win-rate | PF | PF_lo (95%) | DSR z | PBO | DM p | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| A — naive baseline | 329 | 46.5% | 1.090 | 0.831 | 0.65 | 0.50 | 0.515 | **FAIL** |
| B — + Conformal (Pilier 2) | 0 | — | — | — | — | — | — | FAIL (rejected all) |
| C — + Regime Gate (Pilier 3) | 308 | — | 1.117 | 0.864 | 0.81 | 0.50 | 0.418 | **FAIL** |
| D — + both | n/a | — | — | — | — | — | — | SKIPPED (B killed sample) |

**Avg R per trade** : +0.031 (small positive but indistinguishable from zero per DM).
**Cumulative R over 7 years** : +10.2 → mediocre absolute return.

### Interpretation

1. **Variant A naive**: PF 1.09 *looks* ok at the point estimate, but bootstrap CI lower bound 0.83 fails the > 1.00 commercial-grade survival criterion. DSR z=0.65 is far below 1.5. This is what A1 looked like — and it's the same outcome the institutional plan predicted (P(naive M15 retail edge) ≤ 25-35%).

2. **Variant B conformal**: With a 50-trade calibration window of mean R ≈ 0.03, the 90% conformal lower bound is negative → reject everything. This is the CORRECT conservative behavior when the calibration set itself has no robust positive expectancy.

3. **Variant C regime gate**: Dropped 21 trades in BLOCK regime (BOCPD changepoint or bipower jump). Marginal improvement in PF (1.09→1.12) and DSR (0.65→0.81) — directionally right but nowhere near gate threshold.

4. **Variant D combined**: Not evaluated because B killed the sample. This is informative: with weak base edge, conformal reject is too aggressive. Would need more data and/or a stronger base strategy to be useful.

---

## What the framework proves

- The gates correctly reject a strategy that A1's standard-of-care methodology would also reject.
- The gates do NOT fall victim to data mining ("PF 1.09 looks fine, ship it!") — they bootstrap the CI and force PF_lo > 1.0.
- The conformal layer correctly degrades gracefully on weak edge (rejects everything) rather than letting through false signals.
- The regime gate provides a small but measurable improvement that proves the BOCPD+bipower wiring is working.

## What the framework does NOT prove

- That ANY edge exists at M15 XAU retail. Same conclusion as A1.
- That refining the event-driven strategy (different event subset, surprise data when available, multi-asset macro context) would pass. That requires **new data inputs** (Bloomberg consensus / actual surprise feed) and an investment in macro-factor features (Pillar 4 quick-win recommended).

---

## Recommendation

**Per the institutional plan §6, Sprint 3 gate decision**:

Given 0/3 pillars currently pass admission gates on the implemented strategy (which is the naive event-driven baseline), the plan calls for:

> "Gate global mi-parcours (fin Sprint 3) : si aucun pilier n'a passé son gate, **trigger pivot B2B-API**."

However, the *framework* is now in place — every future strategy attempt will be evaluated through the SAME gates. The right next moves before pivoting:

1. **Acquire Bloomberg/Reuters consensus data** to compute actual surprise scores (currently 0 rows have Forecast). Test surprise-conditioned variant.
2. **Add macro-factor features** (TIPS 10Y real-rates, DXY, COT positioning) → use cross_asset_correlation.py which already exists.
3. **Test event windows over multi-asset reactions** (XAU + DXY + UST10Y) — Hayashi-Yoshida lead-lag.

If after these 3 enhancements still 0/3 pillars pass → pivot B2B-API as planned (decision_matrix_2026_04_30.md, $310k ARR cible).

---

## How to reproduce

```bash
# Full pillar test suite
PYTHONPATH=. python -m pytest tests/test_strategy_gates.py \
                                  tests/test_conformal_wrapper.py \
                                  tests/test_regime_gate.py \
                                  tests/test_event_driven_macro.py -q

# Real-data eval
PYTHONPATH=. python scripts/eval_event_driven_macro.py
# Output → reports/event_driven_macro_eval_<DATE>.json
```

---

## References anchored in this implementation

- López de Prado (2018) AFML ch. 7 — CPCV
- Bailey & López de Prado (2014) — DSR
- Bailey-Borwein-LdP-Zhu (2014) — PBO
- Diebold-Mariano (1995) — equal forecast accuracy test
- Holm (1979) — sequential Bonferroni
- Adams & MacKay (2007) arXiv:0710.3742 — BOCPD
- Barndorff-Nielsen & Shephard (2004) JoFEM — bipower variation / jumps
- Angelopoulos & Bates (2024) arXiv:2411.11824 — Conformal Prediction foundations
- Gibbs & Candès (2021) arXiv:2106.00170 — Adaptive Conformal Inference
- Andersen, Bollerslev, Diebold, Vega (2003) AER — Micro effects of macro announcements
- Andersen, Bollerslev, Diebold (2007) ReStud — Roughing it up: jumps in vol modeling
- Corsi (2009) JoFEM — HAR-RV
