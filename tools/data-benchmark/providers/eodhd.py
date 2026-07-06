"""EODHD — FX/metaux via .FOREX, crypto via .CC, indices .INDX en D1 seulement.

Intervalles natifs limites a 1m/5m/1h : M15 est DERIVE de 5m, H4 DERIVE de 1h
(flag derived=True, penalise dans le rapport comme "non natif").
Pas d'energie intraday (leur API commodities = series FRED daily).
"""
from datetime import datetime

import pandas as pd

from symbols import Sym
from .base import ProviderBase, NotCovered

INDEX_MAP = {
    "US30": "DJI.INDX", "NAS100": "NDX.INDX", "SPX500": "GSPC.INDX",
    "GER40": "GDAXI.INDX", "UK100": "FTSE.INDX", "FRA40": "FCHI.INDX",
    "EU50": "STOXX50E.INDX", "JP225": "N225.INDX", "AUS200": "AXJO.INDX",
    "HK50": "HSI.INDX", "US2000": "RUT.INDX",
}


class EodhdProvider(ProviderBase):
    name = "eodhd"
    env_key = "EODHD_API_KEY"
    min_interval_s = 1.0
    max_bars_per_request = 9000
    native_tfs = {"M5": "M5", "M15": "derive:M5", "H1": "H1",
                  "H4": "derive:H1", "D1": "D1"}

    def map_symbol(self, sym: Sym):
        if sym.cls.startswith("fx") or sym.cls == "metal":
            return f"{sym.name}.FOREX"
        if sym.cls == "crypto":
            return f"{sym.name[:3]}-USD.CC"
        if sym.cls == "index":
            return INDEX_MAP.get(sym.name)
        return None  # energy: pas d'intraday OHLC

    def _fetch_window(self, ticker, sym, tf, start: datetime, end: datetime):
        if sym.cls == "index" and tf != "D1":
            raise NotCovered("EODHD: intraday indices non documente (D1 seulement)")
        if tf == "D1":
            rows = self.get_json(f"https://eodhd.com/api/eod/{ticker}", params={
                "from": start.strftime("%Y-%m-%d"), "to": end.strftime("%Y-%m-%d"),
                "api_token": self.api_key, "fmt": "json"})
            if not rows:
                return pd.DataFrame(columns=["open", "high", "low", "close"])
            df = pd.DataFrame(rows)
            df.index = pd.to_datetime(df["date"], utc=True)
        else:
            interval = {"M5": "5m", "H1": "1h"}[tf]
            rows = self.get_json(f"https://eodhd.com/api/intraday/{ticker}", params={
                "interval": interval, "from": int(start.timestamp()),
                "to": int(end.timestamp()), "api_token": self.api_key, "fmt": "json"})
            if not rows:
                return pd.DataFrame(columns=["open", "high", "low", "close"])
            df = pd.DataFrame(rows)
            df.index = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        return df[["open", "high", "low", "close"]]
