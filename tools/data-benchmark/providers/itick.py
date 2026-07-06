"""iTick — nouvel entrant CFD-style (forex/metaux/indices/crypto, 79-319$/mois).

ADAPTATEUR NON VALIDE : ecrit d'apres la doc publique (api.itick.org, kline),
jamais execute faute de cle. Toute erreur sera enregistree honnetement en
status=error ; ajuster kType/format apres premier run avec cle.
"""
from datetime import datetime

import pandas as pd

from symbols import Sym
from .base import ProviderBase, ProviderError

# kType d'apres la doc publique iTick (1=1m, 2=5m, 3=15m, 4=30m, 5=1h, 6=2h, 7=4h, 8=1d)
KTYPE = {"M5": 2, "M15": 3, "H1": 5, "H4": 7, "D1": 8}


class ITickProvider(ProviderBase):
    name = "itick"
    env_key = "ITICK_API_KEY"
    min_interval_s = 1.5
    max_bars_per_request = 1000
    native_tfs = {tf: tf for tf in KTYPE}

    def map_symbol(self, sym: Sym):
        if sym.cls in ("index", "energy"):
            return None  # symbologie indices/energie iTick non publiee -> a cabler avec la cle
        return sym.name

    def windows(self, tf, start, end):
        yield start, end

    def _fetch_window(self, ticker, sym, tf, start: datetime, end: datetime):
        kind = "crypto" if sym.cls == "crypto" else "forex"
        data = self.get_json(f"https://api.itick.org/{kind}/kline",
                             params={"region": "gb", "code": ticker,
                                     "kType": KTYPE[tf], "limit": 1000},
                             headers={"token": self.api_key})
        if data.get("code") not in (0, 200):
            raise ProviderError(f"iTick code={data.get('code')}: {str(data)[:150]}")
        rows = data.get("data") or []
        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close"])
        df = pd.DataFrame(rows)
        ts_key = "tu" if "tu" in df.columns else "t"
        unit = "ms" if df[ts_key].iloc[0] > 10**11 else "s"
        df.index = pd.to_datetime(df[ts_key], unit=unit, utc=True)
        return df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close"})[
            ["open", "high", "low", "close"]]
