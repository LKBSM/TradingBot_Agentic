"""FMP (Financial Modeling Prep) — FX + indices cash (proxy) + crypto.

Metaux et energie = FUTURES chez FMP (GCUSD/CLUSD...), PAS du spot : marques
"non couverts" (mission : ne jamais substituer un proxy futures a du spot).
ATTENTION timezone : les charts intraday FMP sont en heure de New York ->
conversion America/New_York -> UTC appliquee ici.
"""
from datetime import datetime

import pandas as pd

from symbols import Sym
from .base import ProviderBase, ProviderError

INTERVALS = {"M5": "5min", "M15": "15min", "H1": "1hour", "H4": "4hour"}
INDEX_MAP = {
    "US30": "^DJI", "NAS100": "^NDX", "SPX500": "^GSPC", "GER40": "^GDAXI",
    "UK100": "^FTSE", "FRA40": "^FCHI", "EU50": "^STOXX50E", "JP225": "^N225",
    "AUS200": "^AXJO", "HK50": "^HSI", "US2000": "^RUT",
}


class FmpProvider(ProviderBase):
    name = "fmp"
    env_key = "FMP_API_KEY"
    min_interval_s = 0.5
    max_bars_per_request = 3000
    native_tfs = {tf: tf for tf in ["M5", "M15", "H1", "H4", "D1"]}

    def map_symbol(self, sym: Sym):
        if sym.cls.startswith("fx") or sym.cls == "crypto":
            return sym.name
        if sym.cls == "index":
            return INDEX_MAP.get(sym.name)
        return None  # metal/energy: futures only chez FMP -> non couvert (spot exige)

    def _fetch_window(self, ticker, sym, tf, start: datetime, end: datetime):
        base = "https://financialmodelingprep.com/api/v3"
        if tf == "D1":
            data = self.get_json(f"{base}/historical-price-full/{ticker}", params={
                "from": start.strftime("%Y-%m-%d"), "to": end.strftime("%Y-%m-%d"),
                "apikey": self.api_key})
            rows = data.get("historical", []) if isinstance(data, dict) else []
            if not rows:
                return pd.DataFrame(columns=["open", "high", "low", "close"])
            df = pd.DataFrame(rows)
            df.index = pd.to_datetime(df["date"], utc=True)
            return df[["open", "high", "low", "close"]]
        rows = self.get_json(f"{base}/historical-chart/{INTERVALS[tf]}/{ticker}", params={
            "from": start.strftime("%Y-%m-%d"), "to": end.strftime("%Y-%m-%d"),
            "apikey": self.api_key})
        if isinstance(rows, dict):
            raise ProviderError(f"FMP: {str(rows)[:200]}")
        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close"])
        df = pd.DataFrame(rows)
        idx = pd.to_datetime(df["date"])
        df.index = idx.dt.tz_localize("America/New_York",
                                      ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
        df = df[df.index.notna()]
        return df[["open", "high", "low", "close"]]
