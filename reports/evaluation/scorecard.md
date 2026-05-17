# Smart Sentinel AI — Project Scorecard

**Date** : 2026-05-17T04:30:25.878674+00:00
**Verdict** : 🟢 **GO COMMERCIAL**
**Score moyen** : 9.50/10  (95/100)
**Bilan** : 🟢 9 · 🟡 1 · 🔴 0

## Per-dimension breakdown

| # | Dimension | Status | Score /10 | Latency |
| - | --- | --- | --- | --- |
| 1 | 1. Imports & env | 🟢 GREEN | **10** | 27.6s |
| 2 | 2. Unit tests core algo | 🟢 GREEN | **10** | 64.09s |
| 3 | 3. Data quality coverage | 🟡 YELLOW | **5** | 1.93s |
| 4 | 4. 5-markets gates AI | 🟢 GREEN | **10** | 0.06s |
| 5 | 5. InsightV2 E2E | 🟢 GREEN | **10** | 8.72s |
| 6 | 6. Latency < 250ms | 🟢 GREEN | **10** | 1.8s |
| 7 | 7. Narrative FR+EN | 🟢 GREEN | **10** | 0.0s |
| 8 | 8. Historical stats cost-aware | 🟢 GREEN | **10** | 0.03s |
| 9 | 9. Reproducibility bit-for-bit | 🟢 GREEN | **10** | 0.35s |
| 10 | 10. Documentation | 🟢 GREEN | **10** | 0.0s |

## Detail per dimension

### 1. Imports & env

- **Status** : GREEN
- **Score** : 10/10
- **Latency** : 27.6s
- **Details** :
```json
{
  "ok": 19,
  "total": 19,
  "failed": []
}
```

### 2. Unit tests core algo

- **Status** : GREEN
- **Score** : 10/10
- **Latency** : 64.09s
- **Details** :
```json
{
  "passed": 180,
  "failed": 0,
  "exit_code": 0
}
```

### 3. Data quality coverage

- **Status** : YELLOW
- **Score** : 5/10
- **Latency** : 1.93s
- **Details** :
```json
[
  {
    "path": "data/XAU_15MIN_2019_2026.csv",
    "rows": 172874,
    "coverage_pct": 94.3,
    "ok": false
  },
  {
    "path": "data/EURUSD_15MIN_2019_2025.csv",
    "rows": 174506,
    "coverage_pct": 99.6,
    "ok": true
  }
]
```

### 4. 5-markets gates AI

- **Status** : GREEN
- **Score** : 10/10
- **Latency** : 0.06s
- **Details** :
```json
{
  "n_passed": 5,
  "n_total": 5,
  "source": "reports\\five_markets\\lightgbm_results.json",
  "markets": [
    {
      "asset": "XAU",
      "tf": "M15",
      "all_passed": "True",
      "dsr": 8.0,
      "pbo": 0.0,
      "pf_lo": 2.358054864447794,
      "ir_vs_bh": -0.6190666749201381,
      "n_trades": 5994
    },
    {
      "asset": "XAU",
      "tf": "H1",
      "all_passed": "True",
      "dsr": 8.0,
      "pbo": 0.0,
      "pf_lo": 2.1751717994635684,
      "ir_vs_bh": -2.0265095910185575,
      "n_trades": 2319
    },
    {
      "asset": "XAU",
      "tf": "H4",
      "all_passed": "True",
      "dsr": 4.9929706859716525,
      "pbo": 0.0,
      "pf_lo": 1.5691429964387922,
      "ir_vs_bh": -4.297523540168819,
      "n_trades": 1046
    },
    {
      "asset": "EURUSD",
      "tf": "M15",
      "all_passed": "True",
      "dsr": 8.0,
      "pbo": 0.0,
      "pf_lo": 1.9215409975542304,
      "ir_vs_bh": 0.7767725102983037,
      "n_trades": 6402
    },
    {
      "asset": "EURUSD",
      "tf": "H1",
      "all_passed": "True",
      "dsr": 8.0,
      "pbo": 0.0,
      "pf_lo": 2.338329227388627,
      "ir_vs_bh": 1.3987462183814816,
      "n_trades": 2388
    }
  ]
}
```

### 5. InsightV2 E2E

- **Status** : GREEN
- **Score** : 10/10
- **Latency** : 8.72s
- **Details** :
```json
{
  "missing_required_keys": [],
  "present_forbidden_keys": [],
  "n_scenarios": 3,
  "insight_id": "f4bf295fb47cc9ffa1982179",
  "narrative_short_len": 122,
  "narrative_long_len": 438
}
```

### 6. Latency < 250ms

- **Status** : GREEN
- **Score** : 10/10
- **Latency** : 1.8s
- **Details** :
```json
{
  "p50_ms": 13.9,
  "p95_ms": 30.5,
  "p99_ms": 35.7,
  "target_ms": 250.0,
  "n_samples": 50
}
```

### 7. Narrative FR+EN

- **Status** : GREEN
- **Score** : 10/10
- **Latency** : 0.0s
- **Details** :
```json
{
  "fr_short_chars": 145,
  "fr_long_chars": 609,
  "en_short_chars": 154,
  "en_long_chars": 517,
  "fr_violations": [],
  "en_violations": [],
  "backend_fr": "template",
  "backend_en": "template"
}
```

### 8. Historical stats cost-aware

- **Status** : GREEN
- **Score** : 10/10
- **Latency** : 0.03s
- **Details** :
```json
{
  "has_costs": true,
  "has_pf": true,
  "has_ci": true,
  "has_no_costs_ref": true,
  "pf_with_costs": 0.6,
  "pf_ci95_with_costs": [
    0.532,
    0.675
  ],
  "pf_without_costs_ref": 1.134,
  "n_trades": 6402,
  "sample_window": "2019-01-01\u20142025-12-31"
}
```

### 9. Reproducibility bit-for-bit

- **Status** : GREEN
- **Score** : 10/10
- **Latency** : 0.35s
- **Details** :
```json
{
  "sha1": "2a041bfd95d9677f",
  "sha2": "2a041bfd95d9677f",
  "match": true
}
```

### 10. Documentation

- **Status** : GREEN
- **Score** : 10/10
- **Latency** : 0.0s
- **Details** :
```json
{
  "present": 12,
  "total": 12,
  "missing": []
}
```