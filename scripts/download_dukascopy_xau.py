"""Download XAU/USD tick history from Dukascopy and aggregate to OHLCV.

Dukascopy serves free institutional-grade tick history at:
    https://datafeed.dukascopy.com/datafeed/{SYMBOL}/{YYYY}/{MM-1:02}/{DD:02}/{HH:02}h_ticks.bi5

Each .bi5 file is LZMA-compressed, 20 bytes per tick:
    >IIIff  =  ms_offset | ask_int | bid_int | vol_ask | vol_bid

Usage
-----
::

    # Full XAUUSD 2019-01-01 → today at M15
    python scripts/download_dukascopy_xau.py \\
        --symbol XAUUSD --start 2019-01-01 --end 2026-04-23 \\
        --timeframe 15min --out data/XAUUSD_15m_dukascopy.csv

    # Resume an interrupted download (skips days already written)
    python scripts/download_dukascopy_xau.py --resume \\
        --out data/XAUUSD_15m_dukascopy.csv

Notes
-----
* Months in the Dukascopy URL are 0-indexed (January = 00, December = 11).
* XAU/USD point value = 0.001 (3 decimals). Override with --point for other instruments.
* Timestamps are UTC. The script outputs Open/High/Low/Close from the mid price
  and Volume from bid+ask lot volume (the native Dukascopy convention).
* Weekend and holiday hours return 404 or empty payloads — those are skipped silently.
"""
from __future__ import annotations

import argparse
import lzma
import struct
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Tuple

import pandas as pd

BASE_URL = "https://datafeed.dukascopy.com/datafeed"
TICK_STRUCT = struct.Struct(">IIIff")  # ms, ask, bid, vol_ask, vol_bid
HOUR_URL = "{base}/{symbol}/{year:04d}/{month:02d}/{day:02d}/{hour:02d}h_ticks.bi5"

# (ts_utc, bid, ask, vol_bid, vol_ask)
Tick = Tuple[datetime, float, float, float, float]


def fetch_hour(symbol: str, dt: datetime, point: float,
               retries: int = 3, backoff: float = 2.0) -> List[Tick]:
    """Fetch one hour of ticks. Returns [] for weekend/holiday/missing hours."""
    url = HOUR_URL.format(
        base=BASE_URL, symbol=symbol,
        year=dt.year, month=dt.month - 1,  # 0-indexed!
        day=dt.day, hour=dt.hour,
    )
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            break
        except urllib.error.HTTPError as e:
            if e.code in (404, 416):
                return []  # no data for this hour (weekend/holiday)
            last_exc = e
        except (urllib.error.URLError, TimeoutError) as e:
            last_exc = e
        time.sleep(backoff * (attempt + 1))
    else:
        raise RuntimeError(f"Failed to fetch {url}: {last_exc}")

    if not data:
        return []

    try:
        raw = lzma.decompress(data)
    except lzma.LZMAError:
        return []

    ticks: List[Tick] = []
    for i in range(0, len(raw), TICK_STRUCT.size):
        ms, ask_i, bid_i, vol_ask, vol_bid = TICK_STRUCT.unpack_from(raw, i)
        ts = dt + timedelta(milliseconds=ms)
        ticks.append((ts, bid_i * point, ask_i * point, float(vol_bid), float(vol_ask)))
    return ticks


def fetch_day(symbol: str, date: datetime, point: float,
              max_workers: int = 8) -> List[Tick]:
    hours = [
        datetime(date.year, date.month, date.day, h, tzinfo=timezone.utc)
        for h in range(24)
    ]
    all_ticks: List[Tick] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(fetch_hour, symbol, h, point) for h in hours]
        for fut in as_completed(futures):
            all_ticks.extend(fut.result())
    all_ticks.sort(key=lambda t: t[0])
    return all_ticks


def aggregate_to_ohlcv(ticks: List[Tick], freq: str) -> pd.DataFrame:
    if not ticks:
        return pd.DataFrame()
    df = pd.DataFrame(ticks, columns=["ts", "bid", "ask", "vol_bid", "vol_ask"])
    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    df["vol"] = df["vol_bid"] + df["vol_ask"]
    df = df.set_index("ts")

    ohlc = df["mid"].resample(freq).ohlc()
    vol = df["vol"].resample(freq).sum()
    out = ohlc.copy()
    out["Volume"] = vol
    out.columns = ["Open", "High", "Low", "Close", "Volume"]
    out = out.dropna(subset=["Open"])
    out.index = out.index.tz_convert("UTC").tz_localize(None)
    return out


def iter_days(start: datetime, end: datetime):
    cur = datetime(start.year, start.month, start.day, tzinfo=timezone.utc)
    stop = datetime(end.year, end.month, end.day, tzinfo=timezone.utc)
    while cur <= stop:
        # Skip Saturday (Gold closed); Dukascopy still serves empty files but save round-trips
        if cur.weekday() != 5:
            yield cur
        cur += timedelta(days=1)


def load_existing(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "Date"
    return df


def save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df.index.name = "Date"
    df.to_csv(path)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--symbol", default="XAUUSD")
    p.add_argument("--point", type=float, default=0.001,
                   help="Price unit per int (0.001 for XAU/USD, 1e-5 for EURUSD, 0.001 for JPY pairs)")
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--end", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    p.add_argument("--timeframe", default="15min",
                   help="Pandas resample freq (15min, 1H, 1D, ...)")
    p.add_argument("--out", default="data/XAUUSD_15m_dukascopy.csv")
    p.add_argument("--resume", action="store_true",
                   help="Skip days already present in --out")
    p.add_argument("--workers", type=int, default=8,
                   help="Parallel hour fetches (be kind to Dukascopy — 8 is safe)")
    p.add_argument("--flush-every", type=int, default=30,
                   help="Write to disk every N days (resumability)")
    args = p.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    out_path = Path(args.out)

    existing = load_existing(out_path) if args.resume else pd.DataFrame()
    have_days: set = set()
    if not existing.empty:
        have_days = set(existing.index.normalize().unique().date)
        print(f"Resume: {len(existing):,} bars already in {out_path}, covering {len(have_days)} days")

    collected: List[pd.DataFrame] = [existing] if not existing.empty else []
    days = [d for d in iter_days(start, end) if d.date() not in have_days]
    print(f"Downloading {args.symbol} {args.timeframe} from {start.date()} to {end.date()}")
    print(f"{len(days)} days to fetch (skipping Saturdays and already-downloaded days)\n")

    t0 = time.time()
    empty_streak = 0
    for i, day in enumerate(days, 1):
        try:
            ticks = fetch_day(args.symbol, day, args.point, max_workers=args.workers)
        except Exception as e:
            print(f"  {day.date()}  FAIL: {e} — pausing 10s and retrying once")
            time.sleep(10)
            try:
                ticks = fetch_day(args.symbol, day, args.point, max_workers=args.workers)
            except Exception as e2:
                print(f"  {day.date()}  FAIL again: {e2} — skipping")
                continue

        day_df = aggregate_to_ohlcv(ticks, args.timeframe)
        if not day_df.empty:
            collected.append(day_df)
            empty_streak = 0
        else:
            empty_streak += 1

        if i % 20 == 0 or i == len(days):
            rate = i / max(time.time() - t0, 0.1)
            eta_s = (len(days) - i) / max(rate, 0.01)
            total_bars = sum(len(d) for d in collected)
            print(f"  [{i:4d}/{len(days)}] {day.date()}  "
                  f"ticks={len(ticks):>6}  day_bars={len(day_df):>3}  "
                  f"total_bars={total_bars:>7}  rate={rate:.1f}d/s  eta={eta_s/60:.1f}m")

        if i % args.flush_every == 0 and collected:
            merged = pd.concat(collected)
            save(merged, out_path)
            collected = [merged]  # keep in-memory copy deduplicated

    if not collected:
        print("No data downloaded.")
        return 1

    merged = pd.concat(collected)
    save(merged, out_path)
    print(f"\nWrote {len(merged):,} bars to {out_path}")
    print(f"Range: {merged.index.min()} -> {merged.index.max()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
