"""Smart Money / ICT detection module — institutional facade.

Sprint 1 batch 1.0 (decision E acted in
``audits/2026-Q2/sprint_0_decisions.md``).

Context
-------
Historically the BOS / CHOCH / Order Block / FVG / retest detection logic
lives in ``src/environment/strategy_features.py`` (1213 LOC,
``SmartMoneyEngine`` class). The audit Phase 1 (section 3.2,
``audits/2026-Q2/section_3_2_smart_money.md``) recommends extracting it
into a dedicated module ``src/intelligence/smart_money/`` so the rest of
the algo pipeline can depend on it independently of the RL environment.

The full physical extraction (moving 1000 LOC of detection code) is
deferred to Sprint 6 (production hardening) for two reasons:

1. **Stability risk** : ``SmartMoneyEngine`` is used by ~20 backtest /
   scanner / training entrypoints. A full refactor without coverage
   from Sprint 2 (annotations + F1) and Sprint 3 (edge confirmation)
   risks introducing silent regressions that the current test suite
   cannot catch (the audit shows F1 not measured).
2. **Premature work** : we don't yet know which detectors will survive
   Sprint 2 validation. Extracting OB-as-engulfing into a clean module
   is wasted effort if Sprint 2 mandates OB-as-ICT (P0-2).

This file is therefore a **facade** : it re-exports the public API from
``strategy_features.py`` so new code can import from
``src.intelligence.smart_money`` (the institutionally correct path)
without us having to physically move the source yet.

Post Sprint 3, when (a) annotations exist, (b) the ICT-correct OB
detector replaces engulfing, and (c) the 5 P0/P1 bugs from section 3.2
are fixed, the actual code moves here and ``strategy_features.py`` keeps
only the technical indicators (RSI, MACD, etc.) — the smart money logic
lives here permanently.

Backlog
-------
See ``agents/smc_detection/BACKLOG.md`` for the full task list.

Quick contract
--------------
>>> from src.intelligence.smart_money import SmartMoneyEngine
>>> engine = SmartMoneyEngine(data=ohlcv_df, config={}, verbose=False)
>>> enriched = engine.analyze()  # df enriched with BOS_*, CHOCH_*, OB_*, FVG_*

Audit reference
---------------
- ``audits/2026-Q2/section_3_2_smart_money.md`` (score 6.0/10)
- ``audits/2026-Q2/sprint_0_decisions.md`` (Decision E)
- Findings to address before full extraction:
  - P0-2  Order Blocks must be ICT-defined (anchor to BOS), not engulfing.
  - P0-15 RSI Divergence bug at ``strategy_features.py:849-857``.
  - P1-1  Magic number incoherence ``armed_window=5`` vs ``RETEST_ARMED_WINDOW=30``.
  - P1-2  ``FVG_THRESHOLD=0.1 ATR`` ≈ XAU spread.
  - P1-3  ``RETEST_TOL_ATR=0.5 ATR`` ≈ XAU spread.
"""
from __future__ import annotations

# Re-export the public surface from the legacy module.
# These imports are stable across Sprint 1-3; the facade contract holds.
from src.environment.strategy_features import (  # noqa: F401
    SmartMoneyEngine,
    SMCConfig,
    calculate_bos_choch_fast,
    calculate_bos_retest_fast,
)

__all__ = [
    "SmartMoneyEngine",
    "SMCConfig",
    "calculate_bos_choch_fast",
    "calculate_bos_retest_fast",
]

# Versioning marker so downstream modules can detect facade-vs-physical-extraction.
__facade_version__ = "1.0"
__physical_extraction_target_sprint__ = 6
