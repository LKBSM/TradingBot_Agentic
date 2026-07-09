"""MetaTrader 5 (terminal local) — JUGE du banc, pas un candidat production.

C'est le feed broker que les clients du produit voient sur leurs charts :
l'etalon le plus pertinent pour la fidelite des meches SMC. Licence = usage
interne (donnee du broker du compte connecte dans le terminal).

Particularites gerees :
- Horodatage serveur (typiquement UTC+2/+3) : decalage detecte dynamiquement
  au demarrage (barre M5 la plus recente d'un marche 24/5 vs horloge UTC),
  puis soustrait de tous les timestamps.
- Bougies BID (convention MT5), la plupart des feeds API sont MID -> biais
  systematique ~demi-spread sur les extremes, note dans le rapport.
- Couverture MetaQuotes-Demo : forex + 6 metaux + 11 indices CFD ;
  PAS d'energie ni de crypto CFD (les tickers WTI/BTC/NG du catalogue sont
  des actions/ETF Nasdaq, exclus expres du mapping).
- Requiert le terminal installe et un compte (demo) connecte une fois.
"""
from datetime import datetime, timedelta, timezone

import pandas as pd

from symbols import Sym
from .base import ProviderBase, ProviderError, NotCovered

UTC = timezone.utc

INDEX_MAP = {
    "US30": "US30", "NAS100": "USTEC", "SPX500": "US500", "GER40": "DE40",
    "UK100": "UK100", "FRA40": "FRA40", "EU50": "EUSTX50", "JP225": "JPN225",
    "AUS200": "AUS200", "HK50": "HK50", "US2000": "US2000",
}


class Mt5LocalProvider(ProviderBase):
    name = "mt5"
    env_key = None            # pas de cle : session du terminal local
    min_interval_s = 0.05

    def __init__(self, api_key=None):
        super().__init__(api_key)
        import MetaTrader5 as mt5  # import differe : package optionnel
        self.mt5 = mt5
        if not mt5.initialize():
            raise ProviderError(f"MT5 initialize: {mt5.last_error()}")
        self.tf_map = {"M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
                       "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
                       "D1": mt5.TIMEFRAME_D1}
        self.native_tfs = {tf: tf for tf in self.tf_map}
        self.offset_s = self._detect_server_offset()

    def _detect_server_offset(self) -> int:
        """Decalage horodatage serveur vs UTC, arrondi a l'heure, via la
        barre M5 la plus recente d'un marche 24/5 actif."""
        now = datetime.now(UTC).timestamp()
        offsets = []
        for probe in ("EURUSD", "XAUUSD", "USDJPY"):
            r = self.mt5.copy_rates_from_pos(probe, self.tf_map["M5"], 0, 1)
            if r is not None and len(r):
                offsets.append(round((int(r[0]["time"]) - now) / 3600))
        if not offsets:
            return 0
        off = max(offsets, key=offsets.count) * 3600
        return off

    def map_symbol(self, sym: Sym):
        if sym.cls.startswith("fx") or sym.cls == "metal":
            return sym.name
        if sym.cls == "index":
            return INDEX_MAP.get(sym.name)
        return None  # energie/crypto : pas de CFD sur MetaQuotes-Demo

    def windows(self, tf, start, end):
        yield start, end

    def _fetch_window(self, ticker, sym, tf, start: datetime, end: datetime):
        mt5 = self.mt5
        info = mt5.symbol_info(ticker)
        if info is None:
            raise NotCovered(f"MT5: symbole {ticker} absent du broker")
        if not info.visible:
            mt5.symbol_select(ticker, True)
        # bornes exprimees en temps serveur
        s_srv = start + timedelta(seconds=self.offset_s)
        e_srv = end + timedelta(seconds=self.offset_s)
        rates = mt5.copy_rates_range(ticker, self.tf_map[tf], s_srv, e_srv)
        if rates is None or len(rates) == 0:
            # 1er acces : le terminal telecharge l'historique en tache de fond
            import time as _t
            _t.sleep(2.0)
            rates = mt5.copy_rates_range(ticker, self.tf_map[tf], s_srv, e_srv)
        self.stats.requests += 1
        if rates is None or len(rates) == 0:
            return pd.DataFrame(columns=["open", "high", "low", "close"])
        df = pd.DataFrame(rates)
        df.index = pd.to_datetime(df["time"] - self.offset_s, unit="s", utc=True)
        return df[["open", "high", "low", "close"]]
