"""MT-D1 audit — fetch & cache real XAUUSD candles (Twelve Data).

READ-ONLY on the engine. This only fetches market data via the user's own
configured TWELVE_DATA_API_KEY and caches it to JSONL so the rest of the audit
is reproducible offline without re-hitting the provider. No engine code touched.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

CACHE = Path("docs/audits/ECHANTILLON-DETECTION-2026-07-29/_cache")
CACHE.mkdir(parents=True, exist_ok=True)


def _key() -> str:
    k = os.environ.get("TWELVE_DATA_API_KEY")
    if k:
        return k
    for line in Path(".env").read_text(encoding="utf-8").splitlines():
        if line.startswith("TWELVE_DATA_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("no TWELVE_DATA_API_KEY")


def fetch(interval: str, start: str, end: str, key: str) -> list[dict]:
    params = {
        "symbol": "XAU/USD",
        "interval": interval,
        "start_date": start,
        "end_date": end,
        "outputsize": 5000,
        "order": "ASC",
        "apikey": key,
    }
    url = "https://api.twelvedata.com/time_series?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=60) as r:
        d = json.load(r)
    if d.get("status") != "ok":
        raise SystemExit(f"provider error {interval} {start}..{end}: {str(d)[:300]}")
    vals = d.get("values", [])
    print(f"  {interval} {start}..{end}: {len(vals)} bars")
    return vals


def main() -> None:
    key = _key()
    # H1: one call covers ~100 days of trading hours (< 5000 bars).
    h1 = fetch("1h", "2026-04-18 00:00:00", "2026-07-29 23:00:00", key)
    (CACHE / "XAUUSD_H1.jsonl").write_text(
        "\n".join(json.dumps(v) for v in h1), encoding="utf-8"
    )
    time.sleep(8)
    # M15: two ~35-day chunks (each < 5000 bars), merged & de-duped.
    m15a = fetch("15min", "2026-05-18 00:00:00", "2026-06-23 23:45:00", key)
    time.sleep(8)
    m15b = fetch("15min", "2026-06-23 00:00:00", "2026-07-29 23:45:00", key)
    seen = {}
    for v in m15a + m15b:
        seen[v["datetime"]] = v
    m15 = [seen[k] for k in sorted(seen)]
    print(f"  M15 merged: {len(m15)} bars")
    (CACHE / "XAUUSD_M15.jsonl").write_text(
        "\n".join(json.dumps(v) for v in m15), encoding="utf-8"
    )
    # Report ranges
    for name, arr in (("H1", h1), ("M15", m15)):
        if arr:
            print(f"{name}: {arr[0]['datetime']} .. {arr[-1]['datetime']}  n={len(arr)}")


if __name__ == "__main__":
    main()
