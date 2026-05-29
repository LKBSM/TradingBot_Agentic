"""Fetch the live Forex Factory economic calendar (this week + next week).

Uses the stable public JSON feed that MT4/MT5 EAs have relied on for 10+ years:
    https://nfs.faireconomy.media/ff_calendar_thisweek.json
    https://nfs.faireconomy.media/ff_calendar_nextweek.json

Output is written in the same column order already used by
``data/economic_calendar_2019_2025.csv`` so the existing pipeline (the
``EconomicCalendarFetcher`` / ``NewsAnalysisAgent`` chain) can consume it:

    Date,Currency,Event,Impact,Actual,Forecast,Previous

Values are upserted into the target CSV by (Date, Currency, Event) key, so
running this script every 15–30 minutes (cron / Windows Task Scheduler) keeps
the historical file growing without duplicates.

Usage
-----
::

    # One-shot refresh (appends new events into the live file)
    python scripts/fetch_forexfactory_live.py \\
        --out data/economic_calendar_live.csv

    # Filter to Gold-relevant currencies only
    python scripts/fetch_forexfactory_live.py \\
        --currencies USD,EUR,XAU \\
        --out data/economic_calendar_live.csv

Scheduling (Windows Task Scheduler, every 30 min)
::

    schtasks /create /tn FF_Calendar /tr "python C:\\...\\fetch_forexfactory_live.py --out C:\\...\\data\\economic_calendar_live.csv" /sc minute /mo 30
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

FF_URLS = [
    "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
    "https://nfs.faireconomy.media/ff_calendar_nextweek.json",
]

IMPACT_MAP = {
    "high": "HIGH",
    "medium": "MEDIUM",
    "low": "LOW",
    "holiday": "HOLIDAY",
    "non-economic": "LOW",
}

CSV_COLS = ["Date", "Currency", "Event", "Impact", "Actual", "Forecast", "Previous"]


def fetch_url(url: str, timeout: int = 20) -> list:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def normalize_event(ev: dict) -> dict | None:
    """Convert one FF JSON event to our CSV row, or None if malformed."""
    title = (ev.get("title") or "").strip()
    country = (ev.get("country") or "").strip().upper()
    date_str = ev.get("date") or ""
    if not (title and country and date_str):
        return None

    try:
        # FF uses ISO with timezone offset (e.g. "2024-01-03T08:15:00-05:00"),
        # or "All Day" / empty time for holidays.
        dt = datetime.fromisoformat(date_str)
        if dt.tzinfo is None:
            # Assume ET (Forex Factory's default)
            from datetime import timedelta
            dt = dt.replace(tzinfo=timezone(timedelta(hours=-5)))
        dt_utc = dt.astimezone(timezone.utc).replace(tzinfo=None)
    except ValueError:
        return None

    impact = IMPACT_MAP.get((ev.get("impact") or "low").lower(), "LOW")

    return {
        "Date": dt_utc.strftime("%Y-%m-%d %H:%M:%S"),
        "Currency": country,
        "Event": title,
        "Impact": impact,
        "Actual": (ev.get("actual") or "").strip(),
        "Forecast": (ev.get("forecast") or "").strip(),
        "Previous": (ev.get("previous") or "").strip(),
    }


def fetch_all() -> pd.DataFrame:
    rows: list[dict] = []
    for url in FF_URLS:
        try:
            data = fetch_url(url)
        except urllib.error.HTTPError as e:
            print(f"  WARN: {url} → HTTP {e.code}", file=sys.stderr)
            continue
        except urllib.error.URLError as e:
            print(f"  WARN: {url} → {e}", file=sys.stderr)
            continue
        print(f"  {url.rsplit('/', 1)[-1]}: {len(data)} raw events")
        for ev in data:
            norm = normalize_event(ev)
            if norm is not None:
                rows.append(norm)
    if not rows:
        return pd.DataFrame(columns=CSV_COLS)
    df = pd.DataFrame(rows)[CSV_COLS]
    # Drop within-batch duplicates (FF occasionally double-lists)
    df = df.drop_duplicates(subset=["Date", "Currency", "Event"], keep="last")
    return df


def upsert_csv(new_df: pd.DataFrame, out_path: Path, currencies: set[str] | None) -> None:
    if currencies:
        new_df = new_df[new_df["Currency"].isin(currencies)]

    prior_count = 0
    if out_path.exists():
        existing = pd.read_csv(out_path, dtype=str, keep_default_na=False)
        for c in CSV_COLS:
            if c not in existing.columns:
                existing[c] = ""
        existing = existing[CSV_COLS]
        prior_count = len(existing)
        merged = pd.concat([existing, new_df], ignore_index=True)
    else:
        merged = new_df

    # Keep the latest record per (Date, Currency, Event) — so Actual/Forecast
    # released after the event overrides the earlier pending row.
    merged = merged.drop_duplicates(
        subset=["Date", "Currency", "Event"], keep="last"
    )
    merged = merged.sort_values("Date").reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)

    added = max(len(merged) - prior_count, 0)
    print(f"\nWrote {len(merged):,} total events to {out_path}")
    print(f"  (net new: {added}, current batch contributed {len(new_df)})")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--out", default="data/economic_calendar_live.csv")
    p.add_argument("--currencies", default="",
                   help="Comma-separated currency filter (e.g. USD,EUR,XAU). Empty = all.")
    args = p.parse_args()

    currencies = {c.strip().upper() for c in args.currencies.split(",") if c.strip()} or None

    print(f"Fetching Forex Factory calendar at {datetime.now(timezone.utc).isoformat()}Z")
    df = fetch_all()
    if df.empty:
        print("No events fetched — aborting (previous CSV left intact).")
        return 1

    upsert_csv(df, Path(args.out), currencies)

    # Quick preview: next 10 Tier-1 events
    t1 = df[df["Impact"] == "HIGH"].head(10)
    if not t1.empty:
        print("\nNext HIGH-impact events:")
        for _, r in t1.iterrows():
            print(f"  {r['Date']:<20}  {r['Currency']:<4}  {r['Event']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
