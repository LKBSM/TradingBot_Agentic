"""Finnhub — candles via symboles broker OANDA:/BINANCE:. Premium requis
(candles = 403 en cle gratuite, verifie 2026-07-05). H4 derive de H1.
"""
from datetime import datetime

import pandas as pd

from symbols import Sym
from .base import ProviderBase, ProviderError

RES = {"M5": "5", "M15": "15", "H1": "60", "D1": "D"}
SPECIAL = {
    "WTI": "OANDA:WTICO_USD", "BRENT": "OANDA:BCO_USD", "NATGAS": "OANDA:NATGAS_USD",
    "US30": "OANDA:US30_USD", "NAS100": "OANDA:NAS100_USD", "SPX500": "OANDA:SPX500_USD",
    "GER40": "OANDA:DE30_EUR", "UK100": "OANDA:UK100_GBP", "FRA40": "OANDA:FR40_EUR",
    "EU50": "OANDA:EU50_EUR", "JP225": "OANDA:JP225_USD", "AUS200": "OANDA:AU200_AUD",
    "HK50": "OANDA:HK33_HKD", "US2000": "OANDA:US2000_USD",
}


class FinnhubProvider(ProviderBase):
    name = "finnhub"
    env_key = "FINNHUB_API_KEY"
    min_interval_s = 1.1
    max_bars_per_request = 9000
    native_tfs = {"M5": "M5", "M15": "M15", "H1": "H1", "H4": "derive:H1", "D1": "D1"}

    def map_symbol(self, sym: Sym):
        if sym.cls == "crypto":
            return f"BINANCE:{sym.name[:3]}USDT"
        if sym.cls in ("energy", "index"):
            return SPECIAL.get(sym.name)
        return f"OANDA:{sym.name[:3]}_{sym.name[3:]}"  # fx + metaux

    def _fetch_window(self, ticker, sym, tf, start: datetime, end: datetime):
        endpoint = "crypto/candle" if ticker.startswith("BINANCE:") else "forex/candle"
        data = self.get_json(f"https://finnhub.io/api/v1/{endpoint}", params={
            "symbol": ticker, "resolution": RES[tf],
            "from": int(start.timestamp()), "to": int(end.timestamp()),
            "token": self.api_key})
        if data.get("s") == "no_data":
            return pd.DataFrame(columns=["open", "high", "low", "close"])
        if data.get("s") != "ok":
            raise ProviderError(f"Finnhub s={data.get('s')}: {str(data)[:150]}")
        df = pd.DataFrame({"open": data["o"], "high": data["h"],
                           "low": data["l"], "close": data["c"]},
                          index=pd.to_datetime(data["t"], unit="s", utc=True))
        return df
