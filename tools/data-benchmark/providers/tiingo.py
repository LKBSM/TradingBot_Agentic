"""Tiingo — FX (+ or/argent/platine spot) et crypto. Pas d'indices ni d'energie.

Free tier: 50 req/h, 1000 req/j -> min_interval 72s. Un run complet FX+crypto
prend plusieurs heures en gratuit : lancer avec --providers tiingo en tache de
fond, le cache de reprise fait le reste.
"""
from datetime import datetime

import pandas as pd

from symbols import Sym
from .base import ProviderBase, ProviderError

FREQ = {"M5": "5min", "M15": "15min", "H1": "1hour", "H4": "4hour", "D1": "1day"}
METALS_OK = {"XAUUSD", "XAGUSD", "XPTUSD", "XAUEUR", "XAUGBP"}  # XPD absent (recherche 2026-07-05)


class TiingoProvider(ProviderBase):
    name = "tiingo"
    env_key = "TIINGO_API_KEY"
    min_interval_s = 72.0
    max_bars_per_request = 9500
    native_tfs = {tf: tf for tf in FREQ}

    def map_symbol(self, sym: Sym):
        if sym.cls.startswith("fx"):
            return sym.name.lower()
        if sym.cls == "metal":
            return sym.name.lower() if sym.name in METALS_OK else None
        if sym.cls == "crypto":
            return sym.name.lower()
        return None  # index / energy: non couverts

    def _fetch_window(self, ticker, sym, tf, start: datetime, end: datetime):
        params = {
            "startDate": start.strftime("%Y-%m-%d"),
            "endDate": end.strftime("%Y-%m-%d"),
            "resampleFreq": FREQ[tf],
            "token": self.api_key,
        }
        if sym.cls == "crypto":
            data = self.get_json("https://api.tiingo.com/tiingo/crypto/prices",
                                 params={**params, "tickers": ticker})
            rows = data[0].get("priceData", []) if isinstance(data, list) and data else []
        else:
            rows = self.get_json(
                f"https://api.tiingo.com/tiingo/fx/{ticker}/prices", params=params)
            if isinstance(rows, dict):  # error payload
                raise ProviderError(f"Tiingo: {str(rows)[:200]}")
        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close"])
        df = pd.DataFrame(rows)
        # champ close parfois nomme "close", parfois "last" sur le feed FX
        if "close" not in df.columns and "last" in df.columns:
            df["close"] = df["last"]
        df.index = pd.to_datetime(df["date"], utc=True)
        return df[["open", "high", "low", "close"]]
