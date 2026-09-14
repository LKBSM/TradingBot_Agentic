"""DATA-4 — ecrit dans le registre les marches VALIDES contre le vrai flux.

Prend la sortie de ``validate_markets.py`` et l'applique a deux fichiers :

  - ``config/markets.json``       : identite + symbole fournisseur (MKT-1)
  - ``config/event_market_map.json`` : quelles devises de macro s'y rattachent

L'ecriture est TEXTUELLE et chirurgicale : pas de ``json.dump`` sur le fichier
entier, qui reformaterait tous les tableaux existants et noierait la revue.

Usage :
    python tools/data_budget/apply_validated_markets.py valides.json [valides2.json ...]
    python tools/data_budget/apply_validated_markets.py --dry-run valides.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.data_budget.validate_markets import _split  # noqa: E402 — une seule regle de coupe

MARKETS = REPO / "config" / "markets.json"
EVENT_MAP = REPO / "config" / "event_market_map.json"


def driver_currencies(market_id: str, mtype: str) -> List[str]:
    """Les devises dont la macro fait bouger ce marche.

    Une paire FX est mue par SES DEUX devises (EURUSD suit la macro EUR et USD —
    c'est deja la regle en place). Un metal ou une crypto n'a pas de devise de
    base macro-economique : seule sa devise de COTATION le meut.
    """
    parts = _split(market_id)
    if not parts:
        return ["USD"]
    base, quote = parts
    if mtype == "fx":
        return sorted({base, quote})
    return [quote]


def _entry_text(market: Dict) -> str:
    """Une entree, au format exact des entrees deja presentes (revue lisible)."""
    tfs = ", ".join(json.dumps(t) for t in market["timeframes"])
    return (
        "    {\n"
        f'      "id": {json.dumps(market["id"], ensure_ascii=False)},\n'
        f'      "label": {json.dumps(market["label"], ensure_ascii=False)},\n'
        f'      "symbol": {json.dumps(market["symbol"], ensure_ascii=False)},\n'
        f'      "type": {json.dumps(market["type"])},\n'
        f'      "priceDecimals": {int(market["priceDecimals"])},\n'
        f'      "glyph": {json.dumps(market["glyph"], ensure_ascii=False)},\n'
        f"      \"timeframes\": [{tfs}],\n"
        f'      "providerSymbol": {json.dumps(market["providerSymbol"], ensure_ascii=False)}\n'
        "    }"
    )


def apply_markets(new_markets: List[Dict], dry_run: bool) -> int:
    raw = MARKETS.open("r", encoding="utf-8", newline="").read()
    existing = {m["id"] for m in json.loads(raw)["markets"]}
    to_add = [m for m in new_markets if m["id"] not in existing]
    if not to_add:
        print("markets.json : rien a ajouter (tout est deja au registre)")
        return 0

    nl = "\r\n" if "\r\n" in raw else "\n"
    # point d'insertion : juste avant le "]" qui ferme le tableau "markets"
    close = raw.rindex("\n  ]")
    block = "," + nl + (("," + nl).join(_entry_text(m) for m in to_add))
    updated = raw[:close] + block.replace("\n", nl) + raw[close:]

    json.loads(updated)  # refuse d'ecrire un JSON casse
    if dry_run:
        print(f"markets.json : {len(to_add)} entrees seraient ajoutees (essai a blanc)")
    else:
        with MARKETS.open("w", encoding="utf-8", newline="") as fh:
            fh.write(updated)
        print(f"markets.json : {len(to_add)} entrees ajoutees ({len(existing)} -> {len(existing) + len(to_add)})")
    return len(to_add)


def apply_event_map(new_markets: List[Dict], dry_run: bool) -> int:
    raw = EVENT_MAP.open("r", encoding="utf-8", newline="").read()
    data = json.loads(raw)
    existing = data.get("markets", {})
    to_add = {
        m["id"]: {"driver_currencies": driver_currencies(m["id"], m["type"])}
        for m in new_markets
        if m["id"] not in existing
    }
    if not to_add:
        print("event_market_map.json : rien a ajouter")
        return 0

    nl = "\r\n" if "\r\n" in raw else "\n"
    close = raw.rindex("\n  }")
    lines = [
        f'    {json.dumps(mid)}: {{"driver_currencies": {json.dumps(v["driver_currencies"])}}}'
        for mid, v in to_add.items()
    ]
    block = "," + nl + (("," + nl).join(lines))
    updated = raw[:close] + block.replace("\n", nl) + raw[close:]

    json.loads(updated)
    if dry_run:
        print(f"event_market_map.json : {len(to_add)} entrees seraient ajoutees (essai a blanc)")
    else:
        with EVENT_MAP.open("w", encoding="utf-8", newline="") as fh:
            fh.write(updated)
        print(f"event_market_map.json : {len(to_add)} entrees ajoutees")
    return len(to_add)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", nargs="+", help="sorties JSON de validate_markets.py")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    new_markets: List[Dict] = []
    seen = set()
    for path in args.reports:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        for market in payload["validated"]:
            if market["id"] in seen:
                continue
            seen.add(market["id"])
            new_markets.append({k: v for k, v in market.items() if not k.startswith("_")})

    print(f"{len(new_markets)} marches valides a appliquer\n")
    apply_markets(new_markets, args.dry_run)
    apply_event_map(new_markets, args.dry_run)


if __name__ == "__main__":
    main()
