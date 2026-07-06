"""Twelve Data — REFERENCE provider (etalon) for the benchmark.

Free tier: 8 credits/min, 800/day -> min_interval 7.5s.
timezone=UTC is forced on every request (known +10h tz bug when the exchange
timezone is left implicit — cf. detection quality review 2026-06-12).
"""
from datetime import datetime

import pandas as pd

from symbols import Sym
from .base import ProviderBase, ProviderError, NotCovered

INTERVALS = {"M5": "5min", "M15": "15min", "H1": "1h", "H4": "4h", "D1": "1day"}

INDEX_MAP = {
    "US30": "DJI", "NAS100": "NDX", "SPX500": "SPX", "GER40": "GDAXI",
    "UK100": "FTSE", "FRA40": "FCHI", "EU50": "STOXX50E", "JP225": "N225",
    "AUS200": "AXJO", "HK50": "HSI", "US2000": "RUT",
}
ENERGY_MAP = {"WTI": "WTI/USD", "BRENT": "XBR/USD", "NATGAS": None}


class TwelveDataProvider(ProviderBase):
    name = "twelve_data"
    env_key = "TWELVE_DATA_API_KEY"
    min_interval_s = 7.5
    max_bars_per_request = 5000
    native_tfs = {tf: tf for tf in INTERVALS}

    def map_symbol(self, sym: Sym):
        if sym.cls.startswith("fx") or sym.cls in ("metal", "crypto"):
            return f"{sym.name[:3]}/{sym.name[3:]}"
        if sym.cls == "energy":
            return ENERGY_MAP.get(sym.name)
        if sym.cls == "index":
            return INDEX_MAP.get(sym.name)
        return None

    def _fetch_window(self, ticker, sym, tf, start: datetime, end: datetime):
        data = self.get_json("https://api.twelvedata.com/time_series", params={
            "symbol": ticker,
            "interval": INTERVALS[tf],
            "start_date": start.strftime("%Y-%m-%d %H:%M:%S"),
            "end_date": end.strftime("%Y-%m-%d %H:%M:%S"),
            "timezone": "UTC",
            "outputsize": 5000,
            "apikey": self.api_key,
        })
        if isinstance(data, dict) and data.get("status") == "error":
            code, msg = data.get("code"), str(data.get("message", ""))[:200]
            if code in (400, 404) or "not found" in msg.lower() or "symbol" in msg.lower():
                raise NotCovered(f"TD {code}: {msg}")
            raise ProviderError(f"TD {code}: {msg}")
        values = data.get("values") or []
        if not values:
            return pd.DataFrame(columns=["open", "high", "low", "close"])
        df = pd.DataFrame(values)
        df.index = pd.to_datetime(df["datetime"], utc=True)
        return df[["open", "high", "low", "close"]]
