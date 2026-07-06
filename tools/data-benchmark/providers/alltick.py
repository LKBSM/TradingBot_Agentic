"""AllTick — nouvel entrant CFD-style (forex/metaux/energie/indices/crypto).

ADAPTATEUR NON VALIDE : ecrit d'apres la doc publique (quote.alltick.io,
kline_type), jamais execute faute de cle. Limite connue : profondeur kline par
requete faible -> le banc n'obtiendra peut-etre pas 30 jours complets en M5 ;
le rapport le notera via la completude mesuree.
"""
import json
from datetime import datetime

import pandas as pd

from symbols import Sym
from .base import ProviderBase, ProviderError

KLINE_TYPE = {"M5": 2, "M15": 3, "H1": 5, "H4": 6, "D1": 8}


class AllTickProvider(ProviderBase):
    name = "alltick"
    env_key = "ALLTICK_API_KEY"
    min_interval_s = 2.0
    native_tfs = {tf: tf for tf in KLINE_TYPE}

    def map_symbol(self, sym: Sym):
        if sym.cls == "crypto":
            return f"{sym.name}T"  # convention AllTick: BTCUSDT
        return sym.name

    def windows(self, tf, start, end):
        yield start, end

    def _fetch_window(self, ticker, sym, tf, start: datetime, end: datetime):
        query = {"trace": "bench", "data": {"code": ticker,
                                            "kline_type": KLINE_TYPE[tf],
                                            "kline_timestamp_end": 0,
                                            "query_kline_num": 1000,
                                            "adjust_type": 0}}
        data = self.get_json("https://quote.alltick.io/quote-b-api/kline",
                             params={"token": self.api_key,
                                     "query": json.dumps(query)})
        if data.get("ret") not in (200, 0):
            raise ProviderError(f"AllTick ret={data.get('ret')}: {str(data)[:150]}")
        rows = (data.get("data") or {}).get("kline_list") or []
        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close"])
        df = pd.DataFrame(rows)
        df.index = pd.to_datetime(pd.to_numeric(df["timestamp"]), unit="s", utc=True)
        df = df.rename(columns={"open_price": "open", "high_price": "high",
                                "low_price": "low", "close_price": "close"})
        return df[["open", "high", "low", "close"]]
