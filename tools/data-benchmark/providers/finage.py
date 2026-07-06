"""Finage — agregats forex/crypto (format Polygon-like). Essai 3 jours
seulement ; prix production 599-1450 $/mois et redistribution interdite par
disclaimer (recherche 2026-07-05). Adaptateur pret si cle d'essai fournie.
"""
from datetime import datetime

import pandas as pd

from symbols import Sym
from .base import ProviderBase, NotCovered

RANGE = {"M5": (5, "minute"), "M15": (15, "minute"), "H1": (1, "hour"),
         "H4": (4, "hour"), "D1": (1, "day")}


class FinageProvider(ProviderBase):
    name = "finage"
    env_key = "FINAGE_API_KEY"
    min_interval_s = 1.0
    max_bars_per_request = 20000
    native_tfs = {tf: tf for tf in RANGE}

    def map_symbol(self, sym: Sym):
        if sym.cls.startswith("fx") or sym.cls == "metal":
            return ("forex", sym.name)
        if sym.cls == "crypto":
            return ("crypto", sym.name)
        return None  # indices/energie: symbologie CFD Finage non publiee -> a cabler avec la cle

    def _fetch_window(self, ticker, sym, tf, start: datetime, end: datetime):
        kind, code = ticker
        mult, span = RANGE[tf]
        data = self.get_json(
            f"https://api.finage.co.uk/agg/{kind}/{code}/{mult}/{span}/"
            f"{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}",
            params={"apikey": self.api_key, "limit": 20000})
        rows = data.get("results") or []
        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close"])
        df = pd.DataFrame(rows)
        df.index = pd.to_datetime(df["t"], unit="ms", utc=True)
        return df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close"})[
            ["open", "high", "low", "close"]]
