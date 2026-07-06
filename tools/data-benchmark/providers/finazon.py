"""Finazon — dataset forex consolide (100+ instruments, historique depuis
2023-07). Licence la plus permissive du marche (redistribution des 19$/mois,
recherche 2026-07-05). Couverture metaux/crypto a confirmer avec une cle.
"""
from datetime import datetime

import pandas as pd

from symbols import Sym
from .base import ProviderBase, ProviderError

INTERVALS = {"M5": "5m", "M15": "15m", "H1": "1h", "H4": "4h", "D1": "1d"}


class FinazonProvider(ProviderBase):
    name = "finazon"
    env_key = "FINAZON_API_KEY"
    min_interval_s = 1.5
    max_bars_per_request = 1000
    native_tfs = {tf: tf for tf in INTERVALS}

    def map_symbol(self, sym: Sym):
        if sym.cls.startswith("fx") or sym.cls == "metal":
            return f"{sym.name[:3]}/{sym.name[3:]}"
        return None  # crypto: dataset distinct non cable en v1 ; indices/energie absents

    def _fetch_window(self, ticker, sym, tf, start: datetime, end: datetime):
        data = self.get_json("https://api.finazon.io/latest/finazon/forex/time_series",
                             params={"ticker": ticker, "interval": INTERVALS[tf],
                                     "start_at": int(start.timestamp()),
                                     "end_at": int(end.timestamp()),
                                     "page_size": 1000, "order": "asc"},
                             headers={"Authorization": f"apikey {self.api_key}"})
        if isinstance(data, dict) and data.get("error"):
            raise ProviderError(f"Finazon: {str(data)[:200]}")
        rows = data.get("data") or []
        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close"])
        df = pd.DataFrame(rows)
        df.index = pd.to_datetime(df["t"], unit="s", utc=True)
        return df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close"})[
            ["open", "high", "low", "close"]]
