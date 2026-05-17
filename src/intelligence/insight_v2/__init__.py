"""InsightV2 — full read-out contract aligned with mockups/v2/client_view_full.html.

Aggregates outputs from every pipeline layer into the unified
``InsightSignalV2`` payload :

- Structure (BOS / FVG / OB / retest / invalidation / liquidity)
- Conviction (LightGBM-backed P(win) + conformal interval, NOT additive)
- Regime (HMM + BOCPD + jump ratio + gate decision)
- Volatility (HAR-RV forecast + conformal interval)
- Event (news blackout, next event, sentiment, session)
- Historical stats (hit rate, PF with costs, conformal coverage)
- Narrative (FR/EN/DE/ES — produced by LLM, hors algo strict)
- Scenarios (principal + 2 alternatives — descriptives)
- Compliance (edge_claim, jurisdiction_blocked, disclaimer)

Critical design choice (post Sprint 0-7 + user pivot 2026-05-16) :
**conviction_0_100 is NOT the additive ConfluenceDetector score** (Pearson
−0.008). It is the LightGBM-backed P(win) × 100, where the LightGBM
passed 5/5 institutional gates (DSR/PBO/PF_lo/DM/n_trades).
"""

from src.intelligence.insight_v2.builder import InsightV2Builder  # noqa: F401
from src.intelligence.insight_v2.contract import InsightSignalV2  # noqa: F401
from src.intelligence.insight_v2.scenarios import build_scenarios  # noqa: F401
from src.intelligence.insight_v2.narrative import (  # noqa: F401
    InsightV2NarrativeGenerator,
    NarrativeOutput,
)

__all__ = [
    "InsightV2Builder",
    "InsightSignalV2",
    "build_scenarios",
    "InsightV2NarrativeGenerator",
    "NarrativeOutput",
]
