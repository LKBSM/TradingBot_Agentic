"""DATA-4 — valide des marches candidats contre le VRAI flux, avant de les ajouter.

Figurer au catalogue de reference d'un fournisseur ne prouve rien : seul un appel
``time_series`` qui rend des bougies prouve qu'un marche est servi. Ce script
appelle le flux une fois par candidat (1 credit chacun), verifie que des bougies
reviennent, releve le dernier cours pour un controle de vraisemblance, et ecrit
les entrees de registre pretes a etre collees dans ``config/markets.json``.

Ne rien ajouter au registre sans etre passe par ici : un marche que le
fournisseur ne sert pas devient un marche MORT dans le produit (page vide,
scanner qui ne trouve rien, aucun message utile).

Usage :
    python tools/data_budget/validate_markets.py --groups fx-major,fx-minor,metal,crypto
    python tools/data_budget/validate_markets.py --ids EURGBP,XAGUSD --out valides.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

BASE = "https://api.twelvedata.com/time_series"

#: Devises de cotation, de la plus longue a la plus courte. Le SUFFIXE tranche :
#: une coupe fixe a 3 lettres casse toute crypto a base longue (DOGEUSD...).
QUOTES = (
    # stables d'abord : le suffixe le plus long doit gagner
    "USDT", "USDC",
    # majeures
    "USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD",
    # devises de cotation exotiques (paires USDxxx / EURxxx du catalogue)
    "TRY", "ZAR", "MXN", "SEK", "NOK", "DKK", "PLN", "HUF", "CZK",
    "SGD", "HKD", "THB", "CNH", "CNY", "INR", "BRL", "ILS", "KRW",
)

#: Precision d'affichage par classe / devise de cotation. Le JPY cote a 3
#: decimales, le FX a 5, les metaux et cryptos a 2 — meme convention que les deux
#: marches deja en production (XAUUSD 2, EURUSD 5).
def price_decimals_for(market_id: str, group: str) -> int:
    quote = _split(market_id)[1] if _split(market_id) else ""
    if group.startswith("fx"):
        return 3 if quote == "JPY" else 5
    if group in ("metal", "crypto"):
        return 2
    return 1


def _split(market_id: str) -> Optional[tuple]:
    up = market_id.upper()
    for quote in QUOTES:
        if up.endswith(quote) and len(up) > len(quote):
            return up[: -len(quote)], quote
    return None


def provider_symbol_for(market_id: str) -> Optional[str]:
    parts = _split(market_id)
    return f"{parts[0]}/{parts[1]}" if parts else None


#: group du catalogue UX -> type du registre (metal | fx | crypto | index)
GROUP_TO_TYPE = {
    "fx-major": "fx",
    "fx-minor": "fx",
    "fx-exotic": "fx",
    "metal": "metal",
    "crypto": "crypto",
    "index": "index",
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
        raise SystemExit("TWELVE_DATA_API_KEY absente — validation impossible.")
    return key


def probe(session: requests.Session, key: str, symbol: str) -> Dict:
    """Un appel reel. 1 credit. Rend le verdict et le dernier cours."""
    resp = session.get(
        BASE,
        params={
            "symbol": symbol,
            "interval": "15min",
            "outputsize": 2,
            "apikey": key,
            "format": "JSON",
            "timezone": "UTC",
        },
        timeout=30,
    )
    body = resp.json() if resp.content else {}
    values = body.get("values") or []
    return {
        "http": resp.status_code,
        "ok": bool(values),
        "last_close": float(values[0]["close"]) if values else None,
        "last_ts": values[0]["datetime"] if values else None,
        "message": (body.get("message") or "")[:120],
        "credits_left": resp.headers.get("Api-Credits-Left"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--groups",
        default="fx-major,fx-minor,fx-exotic,metal,crypto",
        help="groupes du catalogue UX a valider (defaut : tout sauf les indices)",
    )
    ap.add_argument("--ids", default="", help="liste d'ids explicite, prioritaire sur --groups")
    ap.add_argument("--out", default="", help="fichier JSON : entrees de registre pretes")
    ap.add_argument("--pause", type=float, default=1.2, help="secondes entre deux appels")
    args = ap.parse_args()

    key = _load_key()
    session = requests.Session()
    catalogue = json.loads(
        (REPO / "config" / "market_catalog_ux_test.json").read_text(encoding="utf-8")
    )

    if args.ids:
        wanted = {i.strip().upper() for i in args.ids.split(",") if i.strip()}
        rows = [m for m in catalogue["markets"] if m["id"].upper() in wanted]
    else:
        groups = {g.strip() for g in args.groups.split(",") if g.strip()}
        rows = [m for m in catalogue["markets"] if m["group"] in groups]

    print(f"{len(rows)} candidats a valider (1 credit chacun)\n")
    validated: List[Dict] = []
    rejected: List[Dict] = []

    for market in rows:
        mid = market["id"]
        symbol = provider_symbol_for(mid)
        if symbol is None:
            rejected.append({"id": mid, "reason": "devise de cotation non reconnue"})
            print(f"  {mid:<9} REJETE  devise de cotation non reconnue")
            continue
        result = probe(session, key, symbol)
        if result["ok"]:
            group = market["group"]
            validated.append(
                {
                    "id": mid,
                    "label": market["label"],
                    "symbol": mid,
                    "type": GROUP_TO_TYPE[group],
                    "priceDecimals": price_decimals_for(mid, group),
                    "glyph": market.get("glyph") or mid[:2],
                    "timeframes": ["M1", "M5", "M15", "H1", "H4", "D1"],
                    "providerSymbol": symbol,
                    "_last_close": result["last_close"],
                    "_last_ts": result["last_ts"],
                }
            )
            print(f"  {mid:<9} OK      {symbol:<12} dernier {result['last_close']} @ {result['last_ts']}")
        else:
            rejected.append(
                {"id": mid, "symbol": symbol, "http": result["http"], "reason": result["message"]}
            )
            print(f"  {mid:<9} REJETE  {symbol:<12} HTTP {result['http']} {result['message']}")
        time.sleep(args.pause)

    print(f"\n{len(validated)} valides, {len(rejected)} rejetes")
    if rejected:
        print("Rejetes :", ", ".join(r["id"] for r in rejected))

    if args.out:
        Path(args.out).write_text(
            json.dumps({"validated": validated, "rejected": rejected}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Entrees de registre : {args.out}")


if __name__ == "__main__":
    main()
