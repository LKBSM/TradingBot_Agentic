"""Massive (ex-Polygon.io) — forex + crypto + 4 metaux spot via paires forex.

Free tier: 5 req/min -> min_interval 12.5s. Indices: seuls I:SPX/I:DJI/I:NDX/
I:RUT existent (cash US, plan Indices requis) — tentes, echec = note honnete.
Pas d'energie spot ni d'indices CFD monde.
"""
from datetime import datetime

import pandas as pd

from symbols import Sym
from .base import ProviderBase, NotCovered

RANGE = {"M5": (5, "minute"), "M15": (15, "minute"), "H1": (1, "hour"),
         "H4": (4, "hour"), "D1": (1, "day")}
INDEX_MAP = {"SPX500": "I:SPX", "US30": "I:DJI", "NAS100": "I:NDX", "US2000": "I:RUT"}


class MassivePolygonProvider(ProviderBase):
    name = "massive_polygon"
    env_key = "MASSIVE_API_KEY"          # alias POLYGON_API_KEY accepte (runner)
    min_interval_s = 12.5
    max_bars_per_request = 40000
    native_tfs = {tf: tf for tf in RANGE}

    def map_symbol(self, sym: Sym):
        if sym.cls.startswith("fx") or sym.cls == "metal":
            return f"C:{sym.name}"
        if sym.cls == "crypto":
            return f"X:{sym.name}"
        if sym.cls == "index":
            return INDEX_MAP.get(sym.name)
        return None

    def _fetch_window(self, ticker, sym, tf, start: datetime, end: datetime):
        mult, span = RANGE[tf]
        f, t = int(start.timestamp() * 1000), int(end.timestamp() * 1000)
        data = self.get_json(
            f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{mult}/{span}/{f}/{t}",
            params={"adjusted": "true", "sort": "asc", "limit": 50000,
                    "apiKey": self.api_key})
        if data.get("status") == "ERROR":
            raise NotCovered(f"Massive: {str(data.get('error', ''))[:150]}")
        rows = data.get("results") or []
        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close"])
        df = pd.DataFrame(rows)
        df.index = pd.to_datetime(df["t"], unit="ms", utc=True)
        return df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close"})[
            ["open", "high", "low", "close"]]
