"""Persistent storage layer for MIA Markets V2.

Inaugurates the ``src/storage/`` namespace (cf. architecture doc §3.4).
Future stores (candles_cache, news_cache, …) should land here as siblings.
"""

from src.storage.candles_cache_store import CandlesCacheStore
from src.storage.haiku_description_cache_store import HaikuDescriptionCacheStore
from src.storage.market_readings_store import MarketReadingsStore
from src.storage.news_cache_store import NewsCacheStore, NormalizedNewsEvent

__all__ = [
    "MarketReadingsStore",
    "CandlesCacheStore",
    "HaikuDescriptionCacheStore",
    "NewsCacheStore",
    "NormalizedNewsEvent",
]
