"""Registre des adaptateurs fournisseurs.

REFERENCE = twelve_data (etalon des comparaisons de meches).
Un fournisseur sans cle dans l'environnement = "non teste" (jamais simule).
"""

REFERENCE = "twelve_data"


def build_registry():
    from .twelve_data import TwelveDataProvider
    from .oanda import OandaProvider
    from .tiingo import TiingoProvider
    from .massive_polygon import MassivePolygonProvider
    from .eodhd import EodhdProvider
    from .fmp import FmpProvider
    from .finnhub import FinnhubProvider
    from .alpha_vantage import AlphaVantageProvider
    from .tradermade import TraderMadeProvider
    from .finage import FinageProvider
    from .fcsapi import FcsApiProvider
    from .finazon import FinazonProvider
    from .itick import ITickProvider
    from .alltick import AllTickProvider

    classes = [TwelveDataProvider, OandaProvider, TiingoProvider,
               MassivePolygonProvider, EodhdProvider, FmpProvider,
               FinnhubProvider, AlphaVantageProvider, TraderMadeProvider,
               FinageProvider, FcsApiProvider, FinazonProvider,
               ITickProvider, AllTickProvider]
    return {cls.name: cls for cls in classes}


# Cles d'environnement alternatives acceptees (alias historiques)
ENV_ALIASES = {
    "MASSIVE_API_KEY": ["POLYGON_API_KEY"],
}

# Metadonnees affichees dans le rapport (statut licence / testabilite,
# recherche du 2026-07-05 — voir research_synthesis.md pour les sources).
PROVIDER_NOTES = {
    "twelve_data": "Reference du banc. Display commercial = Venture 499$/mois (414$ annuel).",
    "oanda": "Compte practice gratuit ; licence = usage interne (display client sur devis).",
    "tiingo": "Display publie ~250$/mois (startup). Pas d'indices/energie/XPD.",
    "massive_polygon": "Ex-Polygon.io. Display = plan Business non publie ; indices US only.",
    "eodhd": "M15/H4 derives (pas natifs). Display commercial sur devis (>=399$/mois).",
    "fmp": "Metaux/energie = futures (non spot) -> non couverts ici. Display sur devis.",
    "finnhub": "Candles premium-only (403 en gratuit). Display sur devis (ancre 3500$/mois).",
    "alpha_vantage": "Intraday premium-only ; pas de metaux/indices/energie en bougies.",
    "tradermade": "Quota gratuit 1000 req/mois insuffisant pour 30j M5. ~L599+/mois par feed.",
    "finage": "Essai 3 jours ; redistribution interdite par disclaimer ; 599-1450$/mois.",
    "fcsapi": "149-329$/mois all-markets, droit display a confirmer par ecrit. Cache 10min plans bas.",
    "finazon": "Redistribution incluse des 19$/mois (meilleure licence) ; historique FX depuis 2023-07.",
    "itick": "Calibre sur API reelle 2026-07-06. Crypto = paires USDT Binance "
             "(basis vs USD). BRENT et US2000 introuvables. H4 derive du H1. "
             "79-319$/mois mais droit display a confirmer par ecrit.",
    "alltick": "Adaptateur NON VALIDE (ecrit sur doc publique, jamais execute faute de cle).",
}
