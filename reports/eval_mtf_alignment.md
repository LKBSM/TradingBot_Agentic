# Phase 2 — HTF Alignment Empirical Validation

- CSV  : `C:\MyPythonProjects\TradingBOT_Agentic\data\XAU_15MIN_2019_2025.csv`
- Bars : 106,618 (2019-01-02 10:15:00 → 2025-12-31 21:45:00)
- State machine: enter=55.0, exit=35.0
- Total trades (no filter): **134**
- Bootstrap iters: 5000

## Bucketed PF by HTF alignment label

| Label | n | win_rate | PF (point) | PF (mean) | PF lo CI95 | PF hi CI95 |
|---|---:|---:|---:|---:|---:|---:|
| `full_alignment` | 77 | 32.47% | 0.758 | 0.781 | 0.401 | 1.302 |
| `h4_aligned` | 10 | 30.00% | 0.634 | 0.805 | 0.000 | 2.920 |
| `h1_aligned` | 0 | 0.00% | 0.000 | 0.000 | 0.000 | 0.000 |
| `ranging` | 0 | 0.00% | 0.000 | 0.000 | 0.000 | 0.000 |
| `misaligned` | 0 | 0.00% | 0.000 | 0.000 | 0.000 | 0.000 |
| `counter_h4` | 47 | 29.79% | 0.686 | 0.712 | 0.282 | 1.345 |
| `na` | 0 | 0.00% | 0.000 | 0.000 | 0.000 | 0.000 |

## Strategy comparison: ALL vs FILTERED (drop counter_h4)

| Strategy | n | PF (point) | PF (mean) | PF lo CI95 | PF hi CI95 |
|---|---:|---:|---:|---:|---:|
| ALL       | 134 | 0.724 | 0.734 | 0.457 | 1.081 |
| FILTERED  | 87 | 0.745 | 0.762 | 0.414 | 1.234 |

## Decision gate

- PF lo CI95 rise: **-0.043** (gate: ≥ +0.050)
- Sample loss (counter_h4 dropped): 47 trades
- **Verdict: NO-GO — keep weight=0**

→ Keep `htf_alignment` at weight=0; the readout remains descriptive only. No behavioral change. Phase 3 is rejected by the empirical gate.