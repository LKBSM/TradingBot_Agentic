"""DATA-4 — quels marches candidats Twelve Data sert-il REELLEMENT ?

``config/market_catalog_ux_test.json`` liste 100 marches populaires, et dit
lui-meme qu'ils ne sont **pas valides par un fournisseur** : « ni la couverture,
ni les symboles ne sont garantis ». Ajouter au registre un marche que le
fournisseur ne sert pas creerait un marche mort dans le produit.

Ce script interroge les points d'entree de REFERENCE de Twelve Data
(``/forex_pairs``, ``/cryptocurrencies``, ``/indices``, ``/commodities``,
``/symbol_search``), qui sont gratuits — aucun credit n'est consomme — et dit,
pour chaque candidat, s'il existe et sous quel ticker exact.

Usage :
    python tools/data_budget/check_provider_coverage.py --out couverture.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

import requests

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

BASE = "https://api.twelvedata.com"

#: Points d'entree de reference (catalogue), gratuits.
REFERENCE_ENDPOINTS = {
    "forex_pairs": "/forex_pairs",
    "cryptocurrencies": "/cryptocurrencies",
    "indices": "/indices",
    "commodities": "/commodities",
    "etf": "/etf",
}

#: Tickers non deductibles de l'id (un indice ne suit aucune convention).
#: Renseigne d'apres le catalogue Twelve Data, verifie par ce script.
INDEX_CANDIDATES: Dict[str, List[str]] = {
    "US500": ["SPX", "GSPC", "US500", "SPX500"],
    "US30": ["DJI", "US30", "DJIA"],
    "NAS100": ["IXIC", "NDX", "NAS100"],
    "US2000": ["RUT", "US2000"],
    "GER40": ["GDAXI", "DAX", "GER40"],
    "UK100": ["FTSE", "UKX", "UK100"],
    "FRA40": ["FCHI", "CAC", "FRA40"],
    "EU50": ["STOXX50E", "SX5E", "EU50"],
    "ESP35": ["IBEX", "ESP35"],
    "ITA40": ["FTSEMIB", "ITA40"],
    "SUI20": ["SSMI", "SMI", "SUI20"],
    "NED25": ["AEX", "NED25"],
    "JP225": ["N225", "JP225", "NI225"],
    "HK50": ["HSI", "HK50"],
    "AUS200": ["AXJO", "AUS200", "ASX200"],
    "CHN50": ["CHN50", "XIN9"],
    "IND50": ["NSEI", "NIFTY", "IND50"],
    "CAN60": ["GSPTSE", "TSX", "CAN60"],
    "VIX": ["VIX"],
    "USDX": ["DXY", "USDX"],
}


def _load_key() -> str:
    key = os.environ.get("TWELVE_DATA_API_KEY")
    if not key:
        for candidate in (REPO / ".env", Path(r"C:\MyPythonProjects\wt-run-main\.env")):
            if candidate.exists():
                for line in candidate.read_text(encoding="utf-8", errors="replace").splitlines():
                    if line.startswith("TWELVE_DATA_API_KEY="):
                        key = line.split("=", 1)[1].strip()
                        break
            if key:
                break
    if not key:
        raise SystemExit("TWELVE_DATA_API_KEY absente — verification impossible.")
    return key


def fetch_reference(session: requests.Session, key: str, path: str) -> List[dict]:
    """Un catalogue de reference. Gratuit : ne consomme aucun credit."""
    resp = session.get(f"{BASE}{path}", params={"apikey": key}, timeout=60)
    resp.raise_for_status()
    body = resp.json()
    if isinstance(body, dict) and body.get("status") == "error":
        raise SystemExit(f"{path}: {body.get('message')}")
    return list(body.get("data", []) if isinstance(body, dict) else body)


#: Devises de cotation connues, de la plus longue a la plus courte : c'est le
#: SUFFIXE qui tranche, pas une coupe au milieu. Un decoupage fixe a 3 lettres
#: rate toutes les cryptos a base longue (DOGEUSD, MATICUSD, AVAXUSD...).
from tools.data_budget.validate_markets import QUOTES  # une seule liste, pas deux


def _pair(raw: str) -> str:
    """``EURUSD`` -> ``EUR/USD``, ``DOGEUSD`` -> ``DOGE/USD``.

    Coupe sur la devise de COTATION (le suffixe), jamais au milieu : une base de
    4 ou 5 lettres est courante en crypto et un decoupage a 3 la casserait.
    """
    if "/" in raw:
        return raw
    up = raw.upper()
    for quote in QUOTES:
        if up.endswith(quote) and len(up) > len(quote):
            return f"{up[: -len(quote)]}/{quote}"
    return up


def _symbol_search(
    session: requests.Session, key: str, label: str, mid: str
) -> Optional[str]:
    """Recherche par nom (gratuite). Retourne le ticker le plus plausible."""
    for query in (mid, label.split("(")[0].strip()):
        if not query:
            continue
        try:
            resp = session.get(
                f"{BASE}/symbol_search",
                params={"symbol": query, "apikey": key, "outputsize": 10},
                timeout=30,
            )
            rows = resp.json().get("data", [])
        except Exception:  # noqa: BLE001 — une recherche qui echoue n'est pas fatale
            continue
        for row in rows:
            sym = str(row.get("symbol", "")).upper()
            if sym:
                return sym
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="", help="fichier JSON de sortie")
    args = ap.parse_args()

    key = _load_key()
    session = requests.Session()

    catalogue = json.loads(
        (REPO / "config" / "market_catalog_ux_test.json").read_text(encoding="utf-8")
    )

    print("Telechargement des catalogues de reference (gratuits)...")
    universes: Dict[str, Set[str]] = {}
    for name, path in REFERENCE_ENDPOINTS.items():
        try:
            rows = fetch_reference(session, key, path)
        except Exception as exc:  # noqa: BLE001 — un catalogue absent n'arrete pas le reste
            print(f"  {name:<16} INDISPONIBLE ({exc})")
            universes[name] = set()
            continue
        symbols = {str(r.get("symbol", "")).upper() for r in rows if r.get("symbol")}
        universes[name] = symbols
        print(f"  {name:<16} {len(symbols):>6} symboles")

    everything: Set[str] = set()
    for s in universes.values():
        everything |= s

    results = []
    for market in catalogue["markets"]:
        mid = market["id"]
        group = market["group"]
        if group == "index":
            candidates = INDEX_CANDIDATES.get(mid, [mid])
        else:
            candidates = [_pair(mid), mid]
        found = next((c for c in candidates if c.upper() in everything), None)
        searched = None
        if not found:
            # Dernier recours : la recherche par nom du fournisseur (gratuite),
            # pour ne pas conclure « absent » sur une convention de ticker.
            searched = _symbol_search(session, key, market.get("label", mid), mid)
            if searched and searched.upper() in everything:
                found = searched
        where = [n for n, s in universes.items() if found and found.upper() in s]
        results.append(
            {
                "id": mid,
                "group": group,
                "provider_symbol": found,
                "supported": bool(found),
                "catalogues": where,
                "tried": candidates,
                "found_by_search": searched,
            }
        )

    ok = [r for r in results if r["supported"]]
    ko = [r for r in results if not r["supported"]]
    print(f"\nCouverture : {len(ok)}/{len(results)} candidats servis par Twelve Data\n")
    for group in catalogue["groups"]:
        g_ok = [r for r in ok if r["group"] == group]
        g_ko = [r for r in ko if r["group"] == group]
        print(f"  {group:<12} {len(g_ok):>3} servis, {len(g_ko):>3} absents"
              + (f"  -> absents : {', '.join(r['id'] for r in g_ko)}" if g_ko else ""))

    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {"supported": ok, "unsupported": ko, "universe_sizes":
                 {k: len(v) for k, v in universes.items()}},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"\nRapport : {args.out}")


if __name__ == "__main__":
    main()
