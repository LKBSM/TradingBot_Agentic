# Baseline Sprint 0 — Report

**Date** : 2026-05-15T20:18:15.407077+00:00
**Mode** : `quick`
**Commit** : `714cecc3d596bab8d1311bbbed2f0d40372e53d2` (`institutional-overhaul`)
**Seed** : 42

## Configurations

| Name | Symbol | TF | CSV | Status |
| --- | --- | --- | --- | --- |
| xau_m15 | XAUUSD | M15 | `data/XAU_15MIN_2019_2026.csv` | ✅ ok |
| eurusd_m15 | EURUSD | M15 | `data/EURUSD_15MIN_2019_2025.csv` | ✅ ok |

## Métriques par configuration

### xau_m15 — XAUUSD M15
- bars_processed : `19875` (2025-06-26 02:45:00 → 2026-04-29 23:45:00)
- total_trades   : `0` (wins=0, losses=0)
- profit_factor  : `0.0000`
- sharpe_per_trade  : `0.0000`
- sharpe_annualised : `n/a`
- sortino_per_trade : `0.0000`
- max_drawdown_r : `0.0000`
- win_rate       : `0.0000`
- expectancy_r   : `0.0000`
- score_max      : `69.0100`
- arms_started/confirmed/aborted : `0` / `0` / `0`
- PF 95% CI bootstrap : ⚠️ no trades on this window (score max=69.01)
- JSON: `reports\baseline\xau_m15_summary.json` (sha256 `1274b59b6a399ef6…`)
- Trades CSV: `reports\baseline\xau_m15_trades.csv` (sha256 `(no file)…`)

### eurusd_m15 — EURUSD M15
- bars_processed : `19875` (2025-03-17 05:15:00 → 2025-12-31 21:45:00)
- total_trades   : `0` (wins=0, losses=0)
- profit_factor  : `0.0000`
- sharpe_per_trade  : `0.0000`
- sharpe_annualised : `n/a`
- sortino_per_trade : `0.0000`
- max_drawdown_r : `0.0000`
- win_rate       : `0.0000`
- expectancy_r   : `0.0000`
- score_max      : `74.9700`
- arms_started/confirmed/aborted : `0` / `0` / `0`
- PF 95% CI bootstrap : ⚠️ no trades on this window (score max=74.97)
- JSON: `reports\baseline\eurusd_m15_summary.json` (sha256 `47cb3e7e7fab122b…`)
- Trades CSV: `reports\baseline\eurusd_m15_trades.csv` (sha256 `(no file)…`)

## Lib snapshot
- `hmmlearn` = `0.3.3`
- `lightgbm` = `4.6.0`
- `matplotlib` = `3.10.8`
- `numpy` = `1.26.4`
- `pandas` = `3.0.0`
- `pydantic` = `2.11.7`
- `pytest` = `9.0.2`
- `scikit-learn` = `1.8.0`
- `scipy` = `1.15.3`

## Config
- HISTORICAL_DATA_FILE = `C:\MyPythonProjects\TradingBOT_Agentic\data\XAU_15MIN_2019_2026.csv`

## Reproductibilité

Pour rejouer cette baseline à l'identique :

```bash
git checkout 714cecc3d596bab8d1311bbbed2f0d40372e53d2
python scripts/run_baseline_sprint0.py  --quick
```

Toute différence dans les SHA256 des fichiers ci-dessus indique une dérive (env, data, ou code).