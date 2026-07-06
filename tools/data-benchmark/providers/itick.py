"""iTick — nouvel entrant CFD-style. ADAPTATEUR CALIBRE SUR L'API REELLE
(sondes du 2026-07-06, cle fournie par l'utilisateur) :

- kType mesures par ecart de timestamps : 1=1m, 2=5m, 3=15m, 4=30m, 5=1h,
  8=1d, 9=1w, 10=1M ; kType 6/7 (2h/4h) renvoient vide -> H4 DERIVE du H1.
- 500 barres max par requete (limit> est plafonne) -> pagination a rebours
  via le parametre `et` (end time, epoch ms).
- Endpoints par classe : forex+metaux+energie = /forex/kline (region=gb),
  indices = /indices/kline (region=gb), crypto = /crypto/kline (region=ba,
  paires USDT Binance — CAVEAT : cote en USDT, pas USD, petit basis vs ref).
- Tickers verifies : USOIL (WTI), XNGUSD (NatGas), EUSTX50, HSI ;
  BRENT et US2000 introuvables -> non couverts.
- Champ `t` = timestamp ms (`tu` = turnover, piege du premier jet).
"""
from datetime import datetime

import pandas as pd

from symbols import Sym
from .base import ProviderBase, ProviderError

KTYPE = {"M5": 2, "M15": 3, "H1": 5, "D1": 8}
INDEX_MAP = {
    "US30": "US30", "NAS100": "NAS100", "SPX500": "SPX500", "GER40": "GER40",
    "UK100": "UK100", "FRA40": "FRA40", "EU50": "EUSTX50", "JP225": "JP225",
    "AUS200": "AUS200", "HK50": "HSI", "US2000": None,
}
ENERGY_MAP = {"WTI": "USOIL", "BRENT": None, "NATGAS": "XNGUSD"}
MAX_PAGES = 20


class ITickProvider(ProviderBase):
    name = "itick"
    env_key = "ITICK_API_KEY"
    min_interval_s = 1.3
    native_tfs = {"M5": "M5", "M15": "M15", "H1": "H1",
                  "H4": "derive:H1", "D1": "D1"}

    def map_symbol(self, sym: Sym):
        if sym.cls == "index":
            code = INDEX_MAP.get(sym.name)
            return ("indices", "gb", code) if code else None
        if sym.cls == "energy":
            code = ENERGY_MAP.get(sym.name)
            return ("forex", "gb", code) if code else None
        if sym.cls == "crypto":
            base = "POL" if sym.name == "MATICUSD" else sym.name[:-3]
            return ("crypto", "ba", f"{base}USDT")  # Binance, cote USDT
        return ("forex", "gb", sym.name)  # fx + metaux (XAUUSD verifie)

    def windows(self, tf, start, end):
        yield start, end  # pagination interne via `et`

    def _fetch_window(self, ticker, sym, tf, start: datetime, end: datetime):
        kind, region, code = ticker
        start_ms = int(start.timestamp() * 1000)
        et = int(end.timestamp() * 1000)
        all_rows = []
        for _ in range(MAX_PAGES):
            data = self.get_json(f"https://api.itick.org/{kind}/kline",
                                 params={"region": region, "code": code,
                                         "kType": KTYPE[tf], "limit": 500,
                                         "et": et},
                                 headers={"token": self.api_key})
            if data.get("code") not in (0, 200):
                raise ProviderError(f"iTick code={data.get('code')}: {str(data)[:150]}")
            rows = data.get("data") or []
            # payloads transitoires malformes observes sous rate-limiting :
            # ne garder que les dicts avec timestamp numerique
            rows = [r for r in rows
                    if isinstance(r, dict) and isinstance(r.get("t"), (int, float))]
            if not rows:
                break
            all_rows.extend(rows)
            oldest = min(r["t"] for r in rows)
            if oldest <= start_ms or len(rows) < 500:
                break
            et = oldest - 1
        if not all_rows:
            return pd.DataFrame(columns=["open", "high", "low", "close"])
        df = pd.DataFrame(all_rows)
        df.index = pd.to_datetime(df["t"], unit="ms", utc=True)
        return df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close"})[
            ["open", "high", "low", "close"]]
