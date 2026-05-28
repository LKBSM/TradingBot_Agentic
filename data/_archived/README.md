# Archived data feeds

⚠️ **Do not load files from this folder in production pipelines.**

They are kept solely for reproducibility of historical reports and audit
trails. They have known structural problems that silently corrupt
strategy backtests and live decisioning.

---

## XAU_15MIN_2019_2025_63pct_corrupt.csv

**Archived** : 2026-05-27 (DG-004, Sprint Tech 1)
**Issue**    : 63 % bar coverage versus the expected ~97 % for XAU M15
2019-2025. Identified in `docs/governance/...` and described in
`reports/data_quality_audit_2026_04_23.md`.

**Symptom** : pattern detectors (BOS, CHOCH, FVG) fired on essentially
100 % of bars because the missing bars created spurious gaps that the
SMC engine interpreted as structural breakouts.

**Production replacement** : `data/XAU_15MIN_2019_2026.csv` (~98.7 %
coverage, Dukascopy-derived; this is the file referenced by
`scripts/run_backtest.py` defaults).

**Reload command** (do NOT run unless explicitly required) :

```bash
# This file is intentionally kept outside the production data dir.
# Re-copying it back into data/ will be caught by the boot fail-fast
# guard (DG-053, COVERAGE_GATE=on, threshold 95 %).
```

---

## Why the archive is kept at all

- A few diagnostic notebooks reference rows from this CSV by index
  for the 2026-04-23 incident analysis (BOS-on-100 %-of-bars).
- Reproducing historical PROGRESS reports requires reading the
  original file.
- The file size (~6 MB) does not justify the compliance overhead
  of deletion.

If you need to consume historical XAU bars, prefer
`data/XAU_15MIN_2019_2026.csv` or download a fresh feed via
`scripts/download_dukascopy_xau.py`.
