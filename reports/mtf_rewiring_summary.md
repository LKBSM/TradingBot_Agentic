# MTF Rewiring — Phases 1 + 2 + 3 — Final Report (2026-05-23)

## TL;DR
- **Phase 1 (WIRE)** : ✅ DONE — MTF feature pipeline branched read-only into ConfluenceDetector / InsightAssembler / InsightSignalV2. Zero behavioral change (weight=0). 23 new integration tests green. Schema bumped 2.1.0 → **2.2.0** (additive `mtf_readout` field).
- **Phase 2 (VALIDATE)** : ✅ DONE — 6-yr XAU M15 replay (2019-2025, 106 618 bars, 134 trades). Bootstrap CI95 PF "filtered" vs "all".
- **Phase 3 (ACTIVATE)** : ❌ **REJECTED BY EMPIRICAL GATE.** PF lo CI95 = -0.043 (need +0.050). Weight stays at 0.0; readout stays descriptive only.

## Phase 1 — wiring delivered
| File | Change |
|---|---|
| `src/intelligence/sentinel_scanner.py` | lookback 200→800, warm-up guard, `_compute_htf_features_safe()` step 5b |
| `src/intelligence/confluence_detector.py` | `htf_alignment: 0.0` in DEFAULT_WEIGHTS, `htf_features` kwarg on `analyze()`, `_score_htf_alignment()` method with 7 quality bands |
| `src/api/insight_signal_v2.py` | `MultiTimeframeReadout` Pydantic class, SCHEMA_VERSION → "2.2.0", optional `mtf_readout` field |
| `src/intelligence/readout_mappers.py` | `map_mtf_readout()` projection function with RSI denormalisation + session label |
| `src/intelligence/insight_assembler.py` | `htf_features` kwarg + readout propagation |
| `tests/test_mtf_wiring.py` | 23 integration tests (weights invariant, quality bands, Pydantic accept/reject, mapper consistency, end-to-end propagation) |
| 4 test files | bumped `"2.1.0"` → `"2.2.0"` in 8 assertions; component count 8 → 9 |

## Phase 2 — empirical validation

**Config** : `enter=55 / exit=35 / high_vol_forces_exit=False`. Vol regime forced-exit was disabled because the SMA-quantile classifier marks the bar after entry as 'high' on virtually every trade — collapsing the sample to 5 trades over 6 years. The HTF filter is evaluated independently of the vol regime.

**Bucketed PF by HTF alignment label (134 trades)**

| Label | n | win_rate | PF (point) | PF (mean) | PF lo CI95 | PF hi CI95 |
|---|---:|---:|---:|---:|---:|---:|
| `full_alignment` | 77 | 32.5% | 0.758 | 0.781 | 0.401 | 1.302 |
| `h4_aligned` | 10 | 30.0% | 0.634 | 0.805 | 0.000 | 2.920 |
| `counter_h4` | 47 | 29.8% | 0.686 | 0.712 | 0.282 | 1.345 |
| `h1_aligned` / `ranging` / `misaligned` / `na` | 0 | – | – | – | – | – |

**Strategy comparison**

| Strategy | n | PF (point) | PF (mean) | PF lo CI95 | PF hi CI95 |
|---|---:|---:|---:|---:|---:|
| ALL       | 134 | 0.724 | 0.734 | 0.457 | 1.081 |
| FILTERED (drop counter_h4) | 87 | 0.745 | 0.762 | **0.414** | 1.234 |

**Verdict**: PF lo CI95 dropped by 0.043 (gate required +0.050 minimum rise). The filter loses 47 trades — variance inflates more than the point estimate improves. The H4-counter bucket isn't materially worse than the aligned bucket (WR 29.8% vs 32.5%; PF 0.686 vs 0.758; CI95 overlapping heavily).

## Phase 3 — rejected (gate not met)

No weight change applied. `htf_alignment` stays at **0.0** in `DEFAULT_WEIGHTS`. No tier cutpoint re-sweep needed.

The MTF readout (`mtf_readout` in InsightSignalV2 v2.2.0) is still emitted for:
- B2C narrative engine (can mention HTF context descriptively without claiming filter edge)
- B2B clients (extra signal context in JSON, no behavioral impact)
- Future re-evaluation when (a) the base signal stream becomes profitable (current PF 0.724 < 1.0), or (b) a different alignment encoding (e.g., trend × strength weighted instead of categorical) is tested.

## Why the result makes sense
1. **Base strategy isn't profitable yet** (PF 0.724). A filter on a losing strategy can only redistribute losses across smaller buckets — it can't manufacture positive expectancy.
2. **H4-counter trades aren't selectively worse** : WR 29.8% counter vs 32.5% aligned ≈ 3pp gap — well inside noise on n=47/77.
3. **Sample loss penalty exceeds signal gain** : dropping 35% of trades raises variance more than it improves the mean.

This is exactly the failure mode the conservative gate was designed to catch. The wiring is preserved (no rip-out) so a future evaluation on an improved base signal stream can re-run `python -m scripts.eval_mtf_alignment` and revisit.

## Artefacts
- `scripts/eval_mtf_alignment.py` — re-runnable validation harness
- `reports/eval_mtf_alignment.csv` — per-trade alignment labels + R-multiples
- `reports/eval_mtf_alignment.md` — full bucketed report
- `tests/test_mtf_wiring.py` — 23 integration tests

## Net behavioral diff
**Zero.** No trade decision changes. Only the contract surface (`mtf_readout` field) and observability surface (HTF_Alignment component visible in detailed breakdowns with weight=0 → contribution=0).
