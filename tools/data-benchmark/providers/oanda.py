"""OANDA v20 REST — free practice account, mid candles, ~68 CFD/FX instruments.

Env: OANDA_API_TOKEN (+ OANDA_ENV=practice|live, default practice).
Only candles with complete=true are kept (no partial-bar pollution).
"""
import os
from datetime import datetime

import pandas as pd

from symbols import Sym
from .base import ProviderBase, ProviderError, NotCovered

GRANULARITY = {"M5": "M5", "M15": "M15", "H1": "H1", "H4": "H4", "D1": "D"}

SPECIAL = {
    # energy
    "WTI": "WTICO_USD", "BRENT": "BCO_USD", "NATGAS": "NATGAS_USD",
    # indices (OANDA legacy instrument names)
    "US30": "US30_USD", "NAS100": "NAS100_USD", "SPX500": "SPX500_USD",
    "GER40": "DE30_EUR", "UK100": "UK100_GBP", "FRA40": "FR40_EUR",
    "EU50": "EU50_EUR", "JP225": "JP225_USD", "AUS200": "AU200_AUD",
    "HK50": "HK33_HKD", "US2000": "US2000_USD",
}
# OANDA only lists these cryptos; the other 16 are genuinely not covered.
CRYPTO_OK = {"BTCUSD", "ETHUSD", "LTCUSD", "BCHUSD"}
NOT_LISTED = {"USDBRL"}  # BRL not offered on OANDA


class OandaProvider(ProviderBase):
    name = "oanda"
    env_key = "OANDA_API_TOKEN"
    min_interval_s = 0.35
    max_bars_per_request = 4500
    native_tfs = {tf: tf for tf in GRANULARITY}

    def __init__(self, api_key):
        super().__init__(api_key)
        env = os.environ.get("OANDA_ENV", "practice").strip().lower()
        host = "api-fxtrade.oanda.com" if env == "live" else "api-fxpractice.oanda.com"
        self.base = f"https://{host}/v3"

    def map_symbol(self, sym: Sym):
        if sym.name in NOT_LISTED:
            return None
        if sym.cls == "crypto":
            return f"{sym.name[:3]}_{sym.name[3:]}" if sym.name in CRYPTO_OK else None
        if sym.cls in ("energy", "index"):
            return SPECIAL.get(sym.name)
        return f"{sym.name[:3]}_{sym.name[3:]}"  # fx + metals incl. XAU_EUR/XAU_GBP

    def _fetch_window(self, ticker, sym, tf, start: datetime, end: datetime):
        try:
            data = self.get_json(
                f"{self.base}/instruments/{ticker}/candles",
                params={
                    "granularity": GRANULARITY[tf],
                    "from": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "to": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "price": "M",
                    # aligner les D1 sur minuit UTC (defaut OANDA = 17h New York,
                    # qui desalignerait tous les timestamps vs la reference)
                    "alignmentTimezone": "UTC",
                    "dailyAlignment": 0,
                },
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
        except ProviderError as exc:
            if "Invalid value specified for 'instrument'" in str(exc):
                raise NotCovered(f"OANDA instrument inconnu: {ticker}")
            raise
        candles = [c for c in data.get("candles", []) if c.get("complete")]
        if not candles:
            return pd.DataFrame(columns=["open", "high", "low", "close"])
        df = pd.DataFrame(
            {k: [float(c["mid"][j]) for c in candles]
             for k, j in [("open", "o"), ("high", "h"), ("low", "l"), ("close", "c")]},
            index=pd.to_datetime([c["time"] for c in candles], utc=True),
        )
        return df
