"""TraderMade — FX/CFD/metaux. Fenetres API tres etroites (minute: 2 jours max,
hourly: 1 mois) et quota gratuit 1000 req/mois : le backfill 30j M5 depasse le
quota gratuit. Adaptateur pret ; a lancer avec une cle payante ou sur un
sous-ensemble (--symbols ... --tfs D1,H4).
"""
from datetime import datetime, timedelta

import pandas as pd

from symbols import Sym
from .base import ProviderBase, ProviderError, NotCovered

# interval/period natifs + fenetre max par requete
CFG = {"M5": ("minute", 5, timedelta(days=2)),
       "M15": ("minute", 15, timedelta(days=2)),
       "H1": ("hourly", 1, timedelta(days=28)),
       "H4": ("hourly", 4, timedelta(days=28)),
       "D1": ("daily", 1, timedelta(days=360))}
SPECIAL = {"WTI": "OIL", "BRENT": "UKOIL", "NATGAS": "NATGAS",
           "US30": "USA30", "NAS100": "NAS100", "SPX500": "SPX500",
           "GER40": "GER30", "UK100": "UK100", "FRA40": "FRA40",
           "JP225": "JPN225", "AUS200": "AUS200", "HK50": "HKG33"}
NOT_LISTED = {"USDCNH", "XPDUSD", "XAUGBP", "EU50", "US2000"}  # absents (2026-07-05)


class TraderMadeProvider(ProviderBase):
    name = "tradermade"
    env_key = "TRADERMADE_API_KEY"
    min_interval_s = 1.2
    native_tfs = {tf: tf for tf in CFG}

    def map_symbol(self, sym: Sym):
        if sym.name in NOT_LISTED:
            return None
        if sym.cls in ("energy", "index"):
            return SPECIAL.get(sym.name)
        return sym.name  # fx, metaux, crypto au format EURUSD/XAUUSD/BTCUSD

    def windows(self, tf, start, end):
        span = CFG[tf][2]
        cur = start
        while cur < end:
            nxt = min(end, cur + span)
            yield cur, nxt
            cur = nxt

    def _fetch_window(self, ticker, sym, tf, start: datetime, end: datetime):
        interval, period, _ = CFG[tf]
        data = self.get_json("https://marketdata.tradermade.com/api/v1/timeseries",
                             params={"currency": ticker, "api_key": self.api_key,
                                     "start_date": start.strftime("%Y-%m-%d-%H:%M"),
                                     "end_date": end.strftime("%Y-%m-%d-%H:%M"),
                                     "interval": interval, "period": period,
                                     "format": "records"})
        if isinstance(data, dict) and data.get("error"):
            msg = str(data)[:200]
            if "currency" in msg.lower() or "instrument" in msg.lower():
                raise NotCovered(f"TraderMade: {msg}")
            raise ProviderError(f"TraderMade: {msg}")
        rows = data.get("quotes", []) if isinstance(data, dict) else []
        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close"])
        df = pd.DataFrame(rows)
        df.index = pd.to_datetime(df["date"], utc=True)
        return df[["open", "high", "low", "close"]]
