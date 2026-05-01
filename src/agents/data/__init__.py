"""
Data ingestion subsystem for Smart Sentinel AI.

Sprint DATA-1.1+: macro/news/sentiment data providers with vintage-aware timestamps
to prevent look-ahead bias in backtest and live forecasting.

See `reports/roadmap_2026_2027/PLAN_12_MOIS.md` Partie II.2 Agent 1 (Marwan).
"""

from src.agents.data.cot_provider import CotProvider
from src.agents.data.fred_provider import FredProvider

__all__ = ["FredProvider", "CotProvider"]
