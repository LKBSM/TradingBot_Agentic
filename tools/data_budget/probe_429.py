"""DATA-3 (d) — TEST REEL UNIQUE : que renvoie Twelve Data quand on depasse ?

Le limiteur de DATA-2 doit etre concu sur le comportement REEL du fournisseur,
pas sur une supposition. Ce script depasse volontairement le plafond du plan
GRATUIT (8 credits/minute) UNE SEULE FOIS et enregistre, pour chaque reponse :
code HTTP, en-tetes de limite eventuels, corps, et le delai de reinitialisation
observe.

Garde-fous :
  - N'envoie jamais plus de ``--burst`` requetes (defaut 10, soit 2 de plus que
    le plafond) et ne recommence pas : une seule execution suffit.
  - Refuse de tourner sans ``--confirm`` pour qu'aucun appel ne parte par erreur.
  - Aucune cle n'est ecrite dans la sortie ni dans le depot.

Usage :
    python tools/data_budget/probe_429.py --confirm            # burst puis sonde
    python tools/data_budget/probe_429.py --confirm --burst 10 --out rapport.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[2]

URL = "https://api.twelvedata.com/time_series"
INTERESTING_HEADERS = (
    "api-credits-used",
    "api-credits-left",
    "ratelimit-limit",
    "ratelimit-remaining",
    "ratelimit-reset",
    "x-ratelimit-limit",
    "x-ratelimit-remaining",
    "x-ratelimit-reset",
    "retry-after",
)


def _load_key() -> str:
    key = os.environ.get("TWELVE_DATA_API_KEY")
    if not key:
        env_file = REPO / ".env"
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.startswith("TWELVE_DATA_API_KEY="):
                    key = line.split("=", 1)[1].strip()
                    break
    if not key:
        raise SystemExit(
            "TWELVE_DATA_API_KEY absente (environnement ou .env) — test reel impossible."
        )
    return key


def _one_call(session: requests.Session, key: str, idx: int) -> dict:
    t0 = time.monotonic()
    resp = session.get(
        URL,
        params={
            "symbol": "XAU/USD",
            "interval": "15min",
            # la plus petite fenetre possible : le cout est 1 credit quoi qu'il arrive,
            # autant ne pas transferer 3000 bougies pour rien
            "outputsize": 1,
            "apikey": key,
            "format": "JSON",
            "timezone": "UTC",
        },
        timeout=30,
    )
    elapsed = time.monotonic() - t0
    try:
        body = resp.json()
    except ValueError:
        body = {"_raw": resp.text[:400]}
    # un envelope d'erreur Twelve Data vit DANS le corps, pas toujours dans le code HTTP
    env_code = body.get("code") if isinstance(body, dict) else None
    env_status = body.get("status") if isinstance(body, dict) else None
    return {
        "n": idx,
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "http_status": resp.status_code,
        "elapsed_s": round(elapsed, 3),
        "envelope_status": env_status,
        "envelope_code": env_code,
        "envelope_message": (body.get("message") if isinstance(body, dict) else None),
        "values_returned": len(body.get("values", [])) if isinstance(body, dict) else 0,
        "headers": {
            h: resp.headers[h] for h in INTERESTING_HEADERS if h in resp.headers
        },
        "all_headers": dict(resp.headers),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm", action="store_true", help="requis : consomme des credits reels")
    ap.add_argument("--burst", type=int, default=10, help="requetes envoyees d'affilee (defaut 10)")
    ap.add_argument("--out", default="", help="fichier JSON de sortie")
    args = ap.parse_args()

    if not args.confirm:
        print(
            "Refus : ce script consomme des credits Twelve Data reels.\n"
            "Relancer avec --confirm pour l'executer UNE fois.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if args.burst > 15:
        raise SystemExit("--burst plafonne a 15 : le but est de franchir 8, pas d'epuiser la cle.")

    key = _load_key()
    session = requests.Session()
    results = []
    first_rejected_at = None

    for i in range(1, args.burst + 1):
        r = _one_call(session, key, i)
        results.append(r)
        rejected = r["envelope_code"] in (429, 400) or r["http_status"] == 429
        print(
            f"  #{i:>2} HTTP {r['http_status']} | envelope {r['envelope_status']}/{r['envelope_code']}"
            f" | bougies {r['values_returned']} | {r['elapsed_s']}s"
        )
        if rejected and first_rejected_at is None:
            first_rejected_at = time.monotonic()
            print(f"  -> premier refus a la requete #{i}")
            print(f"  -> message : {r['envelope_message']}")
            print(f"  -> en-tetes de limite : {r['headers'] or 'AUCUN'}")
            break

    recovery = None
    if first_rejected_at is not None:
        # une seule sonde de reprise, apres la fenetre glissante d'une minute
        print("  attente de 62 s puis UNE sonde de reprise...")
        time.sleep(62)
        probe = _one_call(session, key, 0)
        recovery = {
            "seconds_waited": round(time.monotonic() - first_rejected_at, 1),
            "recovered": probe["values_returned"] > 0,
            "http_status": probe["http_status"],
            "envelope_code": probe["envelope_code"],
        }
        print(f"  -> reprise apres {recovery['seconds_waited']}s : {recovery['recovered']}")

    report = {
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "plan": "free (8 credits/min annonces)",
        "requests_sent": len(results),
        "first_rejected_request_index": next(
            (r["n"] for r in results if r["envelope_code"] in (429, 400) or r["http_status"] == 429),
            None,
        ),
        "rate_limit_headers_present": any(r["headers"] for r in results),
        "results": results,
        "recovery_probe": recovery,
    }
    if args.out:
        Path(args.out).write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"  rapport ecrit : {args.out}")
    else:
        print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
