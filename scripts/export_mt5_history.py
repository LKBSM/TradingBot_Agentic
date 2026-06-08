"""Export historical OHLCV from your MetaTrader 5 broker to CSV.

Unlike `download_dukascopy_xau.py` (neutral ECN feed), this pulls history
**directly from your live MT5 terminal** — same quotes, spreads, and gaps
your bot will see when trading live. Use this to validate a backtest
against your production broker's data.

Prerequisites
-------------
1. MetaTrader 5 terminal installed and running (Windows only).
2. Logged into a broker account (demo or live).
3. Symbol visible in Market Watch (right-click → Show All if missing).
4. ``pip install MetaTrader5 pandas`` (already in requirements.txt).

Usage
-----
::

    # Full XAUUSD M15 history from 2019 to today (uses terminal's active session)
    python scripts/export_mt5_history.py \\
        --symbol XAUUSD --timeframe M15 \\
        --start 2019-01-01 --end 2026-04-23 \\
        --out data/XAUUSD_M15_mt5.csv

    # Explicit login (if terminal is not pre-authenticated)
    python scripts/export_mt5_history.py \\
        --symbol XAUUSD --timeframe M15 --start 2019-01-01 \\
        --login 12345678 --password "xxx" --server "ICMarkets-Demo" \\
        --out data/XAUUSD_M15_mt5.csv

Notes
-----
* MT5 caps each ``copy_rates_range`` call at the broker's "max bars in chart"
  setting (often 100k). This script chunks by year to stay safely under.
* If the broker only retains 2 years of M15 history, the output will start
  from wherever data actually begins — check the terminal's Symbol
  Specification for your broker's retention policy.
* Volume source: real_volume if the broker exposes it, else tick_volume.
  FX/Gold brokers usually only have tick_volume (no centralized exchange).
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd


TIMEFRAME_MAP = {
    "M1": "TIMEFRAME_M1",
    "M5": "TIMEFRAME_M5",
    "M15": "TIMEFRAME_M15",
    "M30": "TIMEFRAME_M30",
    "H1": "TIMEFRAME_H1",
    "H4": "TIMEFRAME_H4",
    "D1": "TIMEFRAME_D1",
    "W1": "TIMEFRAME_W1",
}


def connect_mt5(login: Optional[int], password: Optional[str],
                server: Optional[str], path: Optional[str]):
    try:
        import MetaTrader5 as mt5
    except ImportError:
        print("ERROR: MetaTrader5 package missing — pip install MetaTrader5",
              file=sys.stderr)
        sys.exit(1)

    kwargs = {}
    if login is not None:     kwargs["login"] = login
    if password is not None:  kwargs["password"] = password
    if server is not None:    kwargs["server"] = server
    if path is not None:      kwargs["path"] = path

    if not mt5.initialize(**kwargs):
        err = mt5.last_error()
        print(f"ERROR: MT5 initialize failed: {err}", file=sys.stderr)
        print("Make sure MT5 terminal is running and logged into a broker.",
              file=sys.stderr)
        sys.exit(2)

    term = mt5.terminal_info()
    acct = mt5.account_info()
    if term:
        print(f"Connected to: {term.company}  (build {term.build})")
    if acct:
        print(f"Account: #{acct.login} on {acct.server}  "
              f"balance={acct.balance:.2f} {acct.currency}")
    return mt5


def ensure_symbol(mt5, symbol: str) -> None:
    info = mt5.symbol_info(symbol)
    if info is None:
        avail = mt5.symbols_get()
        names = [s.name for s in avail[:20]] if avail else []
        raise ValueError(
            f"Symbol '{symbol}' not found on this broker.\n"
            f"First 20 available: {names}\n"
            f"Your broker may use a suffix like {symbol}.m, {symbol}-ecn, etc."
        )
    if not info.visible:
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"Cannot add {symbol} to Market Watch")
    print(f"Symbol OK: {symbol}  digits={info.digits}  "
          f"point={info.point}  spread={info.spread}")


def fetch_range(mt5, symbol: str, mt5_tf, start: datetime, end: datetime,
                chunk_days: int = 365) -> pd.DataFrame:
    """Pull bars in chunks to stay under MT5's max-bars-per-request cap."""
    frames = []
    cur = start
    while cur < end:
        chunk_end = min(cur + timedelta(days=chunk_days), end)
        rates = mt5.copy_rates_range(symbol, mt5_tf, cur, chunk_end)
        if rates is None:
            err = mt5.last_error()
            print(f"  {cur.date()} → {chunk_end.date()}  FAILED: {err}")
        elif len(rates) == 0:
            print(f"  {cur.date()} → {chunk_end.date()}  0 bars")
        else:
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            frames.append(df)
            print(f"  {cur.date()} → {chunk_end.date()}  {len(df):>6} bars")
        cur = chunk_end

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Convert MT5 columns to the project's (Date, OHLCV) convention."""
    # Prefer real_volume; fall back to tick_volume
    volume_col = "tick_volume"
    if "real_volume" in df.columns and (df["real_volume"] > 0).any():
        volume_col = "real_volume"
        print(f"Volume source: real_volume")
    else:
        print("Volume source: tick_volume (broker does not expose real_volume)")

    out = pd.DataFrame({
        "Date":   df["time"],
        "Open":   df["open"],
        "High":   df["high"],
        "Low":    df["low"],
        "Close":  df["close"],
        "Volume": df[volume_col],
    })
    out = out.drop_duplicates(subset="Date").sort_values("Date").reset_index(drop=True)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--symbol", required=True,
                   help="e.g., XAUUSD (check your broker — may be XAUUSD.m, GOLD, etc.)")
    p.add_argument("--timeframe", default="M15", choices=list(TIMEFRAME_MAP))
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--end", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    p.add_argument("--out", required=True, help="Output CSV path")
    p.add_argument("--chunk-days", type=int, default=365,
                   help="Year-sized chunks are safe for M15; reduce to 90 for M1")
    p.add_argument("--login", type=int, default=None)
    p.add_argument("--password", default=None)
    p.add_argument("--server", default=None)
    p.add_argument("--path", default=None,
                   help="Path to terminal64.exe if multiple MT5 installs")
    args = p.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    out_path = Path(args.out)

    mt5 = connect_mt5(args.login, args.password, args.server, args.path)
    try:
        ensure_symbol(mt5, args.symbol)
        mt5_tf = getattr(mt5, TIMEFRAME_MAP[args.timeframe])

        print(f"\nFetching {args.symbol} {args.timeframe} "
              f"{args.start} → {args.end} in {args.chunk_days}-day chunks\n")
        raw = fetch_range(mt5, args.symbol, mt5_tf, start, end, args.chunk_days)

        if raw.empty:
            print("No data returned. Check broker history retention policy.")
            return 1

        df = normalize(raw)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)

        print(f"\nWrote {len(df):,} bars to {out_path}")
        print(f"Range: {df['Date'].iloc[0]} → {df['Date'].iloc[-1]}")

        # Coverage sanity check
        expected_per_year = 96 * 252 if args.timeframe == "M15" else None
        if expected_per_year:
            span_years = (df['Date'].iloc[-1] - df['Date'].iloc[0]).days / 365.25
            rate = len(df) / max(span_years, 0.01)
            pct = rate / expected_per_year * 100
            print(f"Coverage: {rate:,.0f} bars/year  ({pct:.1f}% of ideal {expected_per_year:,}/yr)")
            if pct < 85:
                print("  ⚠ Below 85% — broker may have limited history retention or many weekend gaps.")
        return 0
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    sys.exit(main())
