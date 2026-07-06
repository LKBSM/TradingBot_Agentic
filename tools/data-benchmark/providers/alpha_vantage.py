"""Alpha Vantage — FX + crypto intraday (premium only) ; pas de metaux/indices/
energie en bougies intraday (verifie 2026-07-05). H4 derive de 60min.
Free = 25 req/jour : inutilisable pour le banc, adaptateur pret si cle premium.
"""
from datetime import datetime

import pandas as pd

from symbols import Sym
from .base import ProviderBase, ProviderError

INTERVALS = {"M5": "5min", "M15": "15min", "H1": "60min"}


class AlphaVantageProvider(ProviderBase):
    name = "alpha_vantage"
    env_key = "ALPHAVANTAGE_API_KEY"
    min_interval_s = 1.0
    max_bars_per_request = 10**9  # outputsize=full, une requete par fenetre
    native_tfs = {"M5": "M5", "M15": "M15", "H1": "H1", "H4": "derive:H1", "D1": "D1"}

    def map_symbol(self, sym: Sym):
        if sym.cls.startswith("fx") or sym.cls == "crypto":
            return sym.name
        return None  # metal/energy/index: pas de bougies intraday chez AV

    def windows(self, tf, start, end):
        yield start, end  # outputsize=full couvre la fenetre

    def _fetch_window(self, ticker, sym, tf, start: datetime, end: datetime):
        base, to = ticker[:-3], ticker[-3:]
        if sym.cls == "crypto":
            params = {"function": "CRYPTO_INTRADAY" if tf != "D1" else "DIGITAL_CURRENCY_DAILY",
                      "symbol": base, "market": to}
        else:
            params = {"function": "FX_INTRADAY" if tf != "D1" else "FX_DAILY",
                      "from_symbol": base, "to_symbol": to}
        if tf != "D1":
            params["interval"] = INTERVALS[tf]
        params.update({"outputsize": "full", "apikey": self.api_key})
        data = self.get_json("https://www.alphavantage.co/query", params=params)
        series = next((v for k, v in data.items() if "Time Series" in k), None)
        if series is None:
            raise ProviderError(f"AlphaVantage: {str(data)[:200]}")
        df = pd.DataFrame.from_dict(series, orient="index")
        df.columns = [c.split(". ")[-1].split(" ")[0] for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated()]
        df.index = pd.to_datetime(df.index, utc=True)  # AV intraday FX est en UTC
        return df[["open", "high", "low", "close"]]
