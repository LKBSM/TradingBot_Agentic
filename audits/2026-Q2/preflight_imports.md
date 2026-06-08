# Pre-flight — Imports modules cœur

**Date** : 2026-05-15
**Batch** : Sprint 0 — 0.0

## Modules importés sans erreur (17 / 17)

```
+ src.intelligence.sentinel_scanner
+ src.intelligence.confluence_detector
+ src.intelligence.volatility_forecaster
+ src.intelligence.signal_state_machine
+ src.intelligence.regime_filter
+ src.intelligence.regime_gate
+ src.intelligence.bocpd
+ src.intelligence.conformal_wrapper
+ src.intelligence.data_providers
+ src.intelligence.data_quality
+ src.intelligence.semantic_cache
+ src.intelligence.circuit_breaker
+ src.backtest.state_machine_replay
+ src.backtest.metrics
+ src.environment.strategy_features
+ src.environment.multi_timeframe_features
+ src.environment.risk_manager  (⚠️ UserWarning: arch not installed)
```

## Collecte pytest

- **2 696 tests collectés** en 9.25 s (mémoire MEMORY.md évoquait 1366+, la suite a grandi de ~1 300 tests depuis).
- 0 erreur de collecte.
- 1 warning : `arch` non installé (déjà loggé dans preflight_env.md).

## Imports cassés

Aucun.

## Verdict

✅ Tous les imports cœur passent. Pas besoin de marquer des tests `xfail` en pré-batch.
