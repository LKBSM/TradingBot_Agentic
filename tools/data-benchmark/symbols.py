"""Canonical symbol universe for the data-provider quality benchmark.

80 symbols x 5 timeframes. Each provider adapter maps canonical names to its
own tickers; a missing mapping means "not covered" (never substituted).

point : price increment used to express wick deviations in pips/points.
tol   : absolute tolerance (price units) for high/low concordance vs reference.
rel   : if True, tolerance is relative (tol = fraction, e.g. 0.0005 = 0.05%),
        used for crypto where absolute pips are meaningless across price scales.

Edit tolerances here; metrics and scoring pick them up automatically.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Sym:
    name: str
    cls: str          # fx_major | fx_cross | fx_exotic | metal | energy | index | crypto
    point: float      # pip/point size (crypto: 1.0, deviations reported in % only)
    tol: float        # concordance tolerance (price units, or fraction if rel)
    rel: bool = False


def _fx(name: str, cls: str, tol_pips: float) -> Sym:
    point = 0.01 if name.endswith("JPY") else 0.0001
    return Sym(name, cls, point, tol_pips * point)


FX_MAJORS = [_fx(s, "fx_major", 1.0) for s in
             ["EURUSD", "USDJPY", "GBPUSD", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]]

FX_CROSSES = [_fx(s, "fx_cross", 1.5) for s in
              ["EURGBP", "EURJPY", "GBPJPY", "EURCHF", "EURAUD", "EURCAD", "EURNZD",
               "GBPCHF", "GBPAUD", "GBPCAD", "GBPNZD", "AUDJPY", "AUDCAD", "AUDNZD",
               "AUDCHF", "CADJPY", "CADCHF", "CHFJPY", "NZDJPY", "NZDCAD", "NZDCHF"]]

FX_EXOTICS = [_fx(s, "fx_exotic", 3.0) for s in
              ["USDMXN", "USDZAR", "USDTRY", "USDCNH", "USDSGD", "USDNOK",
               "USDSEK", "USDPLN", "USDHKD", "USDBRL", "EURTRY", "EURPLN"]]

METALS = [
    Sym("XAUUSD", "metal", 0.1, 0.20),
    Sym("XAGUSD", "metal", 0.01, 0.02),
    Sym("XPTUSD", "metal", 0.1, 0.50),
    Sym("XPDUSD", "metal", 0.1, 0.50),
    Sym("XAUEUR", "metal", 0.1, 0.20),
    Sym("XAUGBP", "metal", 0.1, 0.20),
]

ENERGY = [
    Sym("WTI", "energy", 0.01, 0.05),
    Sym("BRENT", "energy", 0.01, 0.05),
    Sym("NATGAS", "energy", 0.001, 0.005),
]

INDICES = [
    Sym("US30", "index", 1.0, 5.0),
    Sym("NAS100", "index", 1.0, 5.0),
    Sym("SPX500", "index", 1.0, 1.0),
    Sym("GER40", "index", 1.0, 2.0),
    Sym("UK100", "index", 1.0, 2.0),
    Sym("FRA40", "index", 1.0, 2.0),
    Sym("EU50", "index", 1.0, 1.0),
    Sym("JP225", "index", 1.0, 10.0),
    Sym("AUS200", "index", 1.0, 2.0),
    Sym("HK50", "index", 1.0, 10.0),
    Sym("US2000", "index", 1.0, 1.0),
]

CRYPTO = [Sym(s, "crypto", 1.0, 0.0005, rel=True) for s in
          ["BTCUSD", "ETHUSD", "SOLUSD", "XRPUSD", "BNBUSD", "ADAUSD", "DOGEUSD",
           "AVAXUSD", "LINKUSD", "LTCUSD", "DOTUSD", "MATICUSD", "TRXUSD", "BCHUSD",
           "XLMUSD", "ATOMUSD", "UNIUSD", "ETCUSD", "FILUSD", "APTUSD"]]

ALL_SYMBOLS = FX_MAJORS + FX_CROSSES + FX_EXOTICS + METALS + ENERGY + INDICES + CRYPTO
assert len(ALL_SYMBOLS) == 80, f"expected 80 symbols, got {len(ALL_SYMBOLS)}"

SYM_BY_NAME = {s.name: s for s in ALL_SYMBOLS}

TIMEFRAMES = ["M5", "M15", "H1", "H4", "D1"]
TF_SECONDS = {"M5": 300, "M15": 900, "H1": 3600, "H4": 14400, "D1": 86400}
TF_PANDAS = {"M5": "5min", "M15": "15min", "H1": "1h", "H4": "4h", "D1": "1D"}
