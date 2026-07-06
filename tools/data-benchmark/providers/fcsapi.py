"""FCS API — forex/metaux (paires) + crypto. Indices: symbologie de leur API
stock non validee sans cle -> non couverts en v1 (a cabler si cle fournie).
ATTENTION recherche 2026-07-05 : cache 10 min sur les plans bas.
"""
from datetime import datetime

import pandas as pd

from symbols import Sym
from .base import ProviderBase, ProviderError, NotCovered

PERIOD = {"M5": "5m", "M15": "15m", "H1": "1h", "H4": "4h", "D1": "1d"}


class FcsApiProvider(ProviderBase):
    name = "fcsapi"
    env_key = "FCSAPI_API_KEY"
    min_interval_s = 2.0
    max_bars_per_request = 10**9  # l'API renvoie sa fenetre max par period
    native_tfs = {tf: tf for tf in PERIOD}

    def map_symbol(self, sym: Sym):
        pair = f"{sym.name[:3]}/{sym.name[3:]}"
        if sym.cls.startswith("fx") or sym.cls == "metal":
            return ("forex", pair)
        if sym.cls == "crypto":
            return ("crypto", pair)
        return None

    def windows(self, tf, start, end):
        yield start, end

    def _fetch_window(self, ticker, sym, tf, start: datetime, end: datetime):
        kind, pair = ticker
        data = self.get_json(f"https://fcsapi.com/api-v3/{kind}/history", params={
            "symbol": pair, "period": PERIOD[tf], "access_key": self.api_key,
            "from": start.strftime("%Y-%m-%d"), "to": end.strftime("%Y-%m-%d"),
            "level": 3})
        if not data.get("status"):
            msg = str(data.get("msg", data))[:200]
            if "invalid symbol" in msg.lower() or "not found" in msg.lower():
                raise NotCovered(f"FCS: {msg}")
            raise ProviderError(f"FCS: {msg}")
        resp = data.get("response") or {}
        rows = list(resp.values()) if isinstance(resp, dict) else resp
        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close"])
        df = pd.DataFrame(rows)
        df.index = pd.to_datetime(pd.to_numeric(df["t"]), unit="s", utc=True)
        return df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close"})[
            ["open", "high", "low", "close"]]
