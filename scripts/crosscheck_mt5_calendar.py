"""Cross-check the MT5 terminal's economic calendar against Forex Factory.

Purpose: confirm that your broker's news feed aligns with the external source
the bot uses (Forex Factory). Divergences on Tier-1 events (NFP, CPI, FOMC,
ECB, BoE) are a red flag — either your broker is out of date, or the FF cache
is stale, and the bot should not trust its blackout windows until the mismatch
is resolved.

Design
------
* Pulls next N days of HIGH-impact events from both sources.
* Normalises event names (NFP aliases, CPI aliases) and currencies.
* Reports 3 classes of mismatch:
    - MISSING_IN_MT5:   FF has it, broker doesn't
    - MISSING_IN_FF:    broker has it, FF doesn't
    - TIME_DRIFT:       both list it but scheduled_time differs > tolerance

Exit code is 0 on clean match, 1 if any Tier-1 mismatch is found — so this
can be used as a startup gate in ``src/intelligence/main.py``.

Usage
-----
::

    # Standalone check (call before/after starting the bot)
    python scripts/crosscheck_mt5_calendar.py --days 7

    # Stricter tolerance (1 min) for HIGH-impact only
    python scripts/crosscheck_mt5_calendar.py --days 3 --tolerance-min 1

Prerequisites
-------------
* MT5 terminal running and logged in (Windows only).
* ``pip install MetaTrader5 pandas``.

Notes
-----
Some brokers disable the terminal's built-in calendar (returns empty).
If MT5 calendar is empty, this script prints a warning and exits with code 0 —
you should then rely on Forex Factory alone and consider switching brokers if
you need in-platform news.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import List, Tuple

FF_URLS = [
    "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
    "https://nfs.faireconomy.media/ff_calendar_nextweek.json",
]

# Canonical Tier-1 keywords — if either side mentions any of these, we treat
# the row as Tier-1 regardless of the provider's self-reported impact rating.
TIER1_KEYWORDS = {
    "non-farm payrolls", "non farm payrolls", "nfp",
    "fomc", "federal funds", "fed interest rate", "fed rate decision",
    "cpi", "consumer price index", "core cpi",
    "pce price index", "core pce",
    "ecb", "ecb rate decision", "ecb main refinancing",
    "boe", "bank of england rate", "boe rate decision",
    "boj", "bank of japan rate",
    "unemployment rate",
    "gdp", "gross domestic product",
}

# Canonical aliases for matching events across providers
CANONICAL_ALIASES = [
    ("non-farm payrolls", {"nfp", "non farm payrolls", "non-farm payrolls", "change in nonfarm payrolls"}),
    ("cpi", {"cpi m/m", "cpi y/y", "cpi", "consumer price index"}),
    ("core cpi", {"core cpi m/m", "core cpi y/y", "core cpi"}),
    ("fomc", {"fomc statement", "fomc press conf", "fomc meeting minutes", "federal funds rate", "fed interest rate", "fomc"}),
    ("ecb rate", {"ecb main refinancing rate", "ecb rate decision", "ecb interest rate"}),
    ("boe rate", {"boe official bank rate", "boe rate decision"}),
    ("pce", {"pce price index", "core pce price index", "pce"}),
    ("gdp", {"advance gdp q/q", "gdp q/q", "prelim gdp q/q"}),
]


def canonicalize(name: str) -> str:
    n = name.lower().strip()
    for canon, aliases in CANONICAL_ALIASES:
        if any(a in n for a in aliases):
            return canon
    return n


def is_tier1(name: str) -> bool:
    n = name.lower()
    return any(k in n for k in TIER1_KEYWORDS)


def fetch_ff(days: int) -> List[dict]:
    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(days=days)
    out: List[dict] = []
    for url in FF_URLS:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=20) as r:
                data = json.loads(r.read().decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError):
            continue
        for ev in data:
            date_str = ev.get("date") or ""
            try:
                dt = datetime.fromisoformat(date_str)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone(timedelta(hours=-5)))
                dt_utc = dt.astimezone(timezone.utc)
            except ValueError:
                continue
            if dt_utc < now or dt_utc > cutoff:
                continue
            out.append({
                "time": dt_utc,
                "currency": (ev.get("country") or "").upper(),
                "name": (ev.get("title") or "").strip(),
                "impact": (ev.get("impact") or "").lower(),
            })
    return out


def fetch_mt5(days: int) -> Tuple[List[dict], bool]:
    """Returns (events, calendar_available). calendar_available=False if broker
    disabled the feature or API unavailable."""
    try:
        import MetaTrader5 as mt5
    except ImportError:
        print("ERROR: MetaTrader5 package missing — pip install MetaTrader5",
              file=sys.stderr)
        return [], False

    if not mt5.initialize():
        print(f"ERROR: MT5 initialize failed: {mt5.last_error()}", file=sys.stderr)
        return [], False

    try:
        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(days=days)

        # The MT5 Python package exposes calendar_value_history(date_from, date_to)
        # — returns events in the window with their impacts and actuals.
        if not hasattr(mt5, "calendar_value_history"):
            print("WARN: this MetaTrader5 package build does not expose calendar_value_history",
                  file=sys.stderr)
            return [], False

        values = mt5.calendar_value_history(now, cutoff)
        if values is None:
            err = mt5.last_error()
            print(f"WARN: MT5 calendar_value_history returned None: {err}",
                  file=sys.stderr)
            return [], False

        events = []
        for v in values:
            # Each value has event_id → look up event metadata
            ev = mt5.calendar_event_by_id(v.event_id) if hasattr(mt5, "calendar_event_by_id") else None
            name = getattr(ev, "name", "") if ev else ""
            currency_id = getattr(ev, "country_id", 0) if ev else 0
            country = mt5.calendar_country_by_id(currency_id) if hasattr(mt5, "calendar_country_by_id") and currency_id else None
            currency = getattr(country, "currency", "") if country else ""
            importance = getattr(ev, "importance", 0) if ev else 0
            # MT5 importance: 0=None, 1=Low, 2=Medium, 3=High
            impact = {3: "high", 2: "medium", 1: "low"}.get(importance, "none")

            # Value timestamp is a datetime (timezone-naive in terminal TZ).
            # Terminal usually runs UTC for backtesting but this depends on broker.
            ts = v.time if isinstance(v.time, datetime) else datetime.fromtimestamp(v.time, tz=timezone.utc)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)

            events.append({
                "time": ts,
                "currency": currency,
                "name": name,
                "impact": impact,
            })

        return events, True
    finally:
        mt5.shutdown()


def match_events(ff: List[dict], mt: List[dict], tolerance_min: int):
    """Returns (missing_in_mt5, missing_in_ff, time_drift)."""
    tol = timedelta(minutes=tolerance_min)

    ff_t1 = [e for e in ff if is_tier1(e["name"])]
    mt_t1 = [e for e in mt if is_tier1(e["name"])]

    missing_mt5 = []
    matched_mt_idx = set()
    time_drift = []

    for f in ff_t1:
        f_key = (canonicalize(f["name"]), f["currency"])
        best = None
        best_delta = None
        for i, m in enumerate(mt_t1):
            if i in matched_mt_idx:
                continue
            m_key = (canonicalize(m["name"]), m["currency"])
            if f_key == m_key:
                delta = abs(f["time"] - m["time"])
                if best_delta is None or delta < best_delta:
                    best_delta = delta
                    best = i
        if best is None:
            missing_mt5.append(f)
        else:
            matched_mt_idx.add(best)
            if best_delta > tol:
                time_drift.append((f, mt_t1[best], best_delta))

    missing_ff = [m for i, m in enumerate(mt_t1) if i not in matched_mt_idx]
    return missing_mt5, missing_ff, time_drift


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--days", type=int, default=7, help="Look-ahead window (days)")
    p.add_argument("--tolerance-min", type=int, default=5,
                   help="Max allowed time drift (minutes) before a mismatch is flagged")
    args = p.parse_args()

    print(f"Fetching next {args.days} days from both sources...")
    ff = fetch_ff(args.days)
    print(f"  Forex Factory: {len(ff)} events ({sum(1 for e in ff if is_tier1(e['name']))} Tier-1)")

    mt, mt_ok = fetch_mt5(args.days)
    if not mt_ok:
        print(f"\nMT5 calendar unavailable — cannot cross-check.")
        print(f"Options: (1) enable calendar in MT5 terminal, (2) switch to a broker that exposes it,")
        print(f"         (3) accept FF-only mode (set CROSSCHECK_SKIP=1 in production).")
        return 0  # soft-fail so this can be used as startup gate without killing bot

    print(f"  MT5 broker:    {len(mt)} events ({sum(1 for e in mt if is_tier1(e['name']))} Tier-1)")

    missing_mt5, missing_ff, time_drift = match_events(ff, mt, args.tolerance_min)

    print(f"\n{'='*70}")
    print(f"  Tier-1 cross-check result (tolerance ±{args.tolerance_min} min)")
    print(f"{'='*70}")

    if not (missing_mt5 or missing_ff or time_drift):
        print("  ✅ All Tier-1 events match between Forex Factory and MT5 broker.")
        return 0

    if missing_mt5:
        print(f"\n❌ Missing in MT5 broker calendar ({len(missing_mt5)}):")
        for e in missing_mt5:
            print(f"   {e['time'].strftime('%Y-%m-%d %H:%M UTC')}  {e['currency']}  {e['name']}")

    if missing_ff:
        print(f"\n⚠  In MT5 but not Forex Factory ({len(missing_ff)}):")
        for e in missing_ff:
            print(f"   {e['time'].strftime('%Y-%m-%d %H:%M UTC')}  {e['currency']}  {e['name']}")

    if time_drift:
        print(f"\n⚠  Time drift > {args.tolerance_min} min ({len(time_drift)}):")
        for ff_e, mt_e, delta in time_drift:
            print(f"   {ff_e['currency']}  {ff_e['name']}")
            print(f"     FF : {ff_e['time'].strftime('%Y-%m-%d %H:%M UTC')}")
            print(f"     MT5: {mt_e['time'].strftime('%Y-%m-%d %H:%M UTC')}  (Δ = {delta})")

    print(f"\nAction: verify with the official source (bls.gov, federalreserve.gov, ecb.europa.eu)")
    print(f"and update whichever feed is wrong. Do not trust either source blindly.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
