"""
Download XAU/USD 15-minute OHLCV data from Dukascopy (2019-2025).
Fast version: downloads all hours in parallel with 32 workers.

Memory-efficient: aggregates ticks to M15 bars per-hour during download,
avoiding the need to hold hundreds of millions of ticks in memory.

NOTE: This script is for LOCAL data generation only.
      For Colab training, the data file (XAU_15MIN_2019_2025.csv) is
      downloaded from GitHub Releases instead. See:
        - scripts/colab_training_full.py
        - Script collab
      To publish data for training, upload the generated CSV to a GitHub
      release at: https://github.com/LKBSM/TradingBot_Agentic/releases

Usage:
    python scripts/download_xau_data.py
"""

import struct
import lzma
import os
import time
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_URL = "https://datafeed.dukascopy.com/datafeed"
INSTRUMENT = "XAUUSD"
PIPETTE = 1e3  # Gold: 3 decimal places

START_DATE = datetime(2019, 1, 1)
END_DATE = datetime(2025, 12, 31, 23, 59)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_FILE = OUTPUT_DIR / "XAU_15MIN_2019_2025.csv"

MAX_WORKERS = 32  # 32 is the sweet spot - more causes Dukascopy rate limiting
RETRY_ATTEMPTS = 3


def download_hour(dt: datetime) -> list:
    """Download and parse one hour of tick data, return pre-aggregated M15 bars.

    Returns a list of dicts with keys: datetime, Open, High, Low, Close, Volume
    Each hour produces 0-4 M15 bars (at :00, :15, :30, :45).
    """
    url = f"{BASE_URL}/{INSTRUMENT}/{dt.year}/{dt.month - 1:02d}/{dt.day:02d}/{dt.hour:02d}h_ticks.bi5"

    for attempt in range(RETRY_ATTEMPTS):
        try:
            resp = requests.get(url, timeout=10, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
            })

            if resp.status_code == 404 or len(resp.content) == 0:
                return []
            if resp.status_code != 200:
                time.sleep(1)
                continue

            try:
                raw = lzma.decompress(resp.content)
            except lzma.LZMAError:
                return []

            if len(raw) == 0:
                return []

            tick_size = 20
            n_ticks = len(raw) // tick_size
            if n_ticks == 0:
                return []

            base_ts = dt.timestamp()

            # Aggregate ticks into M15 bars in-place (no large list needed)
            # Each 15-min bucket: {open, high, low, close, count}
            bars = {}  # key = minute bucket (0, 15, 30, 45)

            for i in range(n_ticks):
                offset = i * tick_size
                ms_offset, ask_int, bid_int, ask_vol, bid_vol = struct.unpack(
                    '>IIIff', raw[offset:offset + tick_size]
                )
                bid_price = bid_int / PIPETTE
                # Determine which 15-min bucket this tick belongs to
                tick_minute = (ms_offset // 1000) // 60  # minute within the hour
                bucket = (tick_minute // 15) * 15  # 0, 15, 30, or 45

                if bucket not in bars:
                    bars[bucket] = {
                        'open': bid_price,
                        'high': bid_price,
                        'low': bid_price,
                        'close': bid_price,
                        'count': 1
                    }
                else:
                    bar = bars[bucket]
                    bar['high'] = max(bar['high'], bid_price)
                    bar['low'] = min(bar['low'], bid_price)
                    bar['close'] = bid_price  # Last tick = close
                    bar['count'] += 1

            # Convert to list of result dicts
            result = []
            for bucket in sorted(bars.keys()):
                bar = bars[bucket]
                bar_dt = dt.replace(minute=bucket, second=0, microsecond=0)
                result.append({
                    'Date': bar_dt.strftime('%Y-%m-%d %H:%M:%S'),
                    'Open': round(bar['open'], 2),
                    'High': round(bar['high'], 2),
                    'Low': round(bar['low'], 2),
                    'Close': round(bar['close'], 2),
                    'Volume': bar['count']
                })

            return result

        except requests.RequestException:
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(1)
            continue

    return []


def main():
    logger.info(f"Downloading XAU/USD M15 data from Dukascopy ({MAX_WORKERS} workers)")
    logger.info(f"Period: {START_DATE.date()} to {END_DATE.date()}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate all hours
    hours = []
    current = START_DATE.replace(minute=0, second=0, microsecond=0)
    while current <= END_DATE:
        hours.append(current)
        current += timedelta(hours=1)

    total = len(hours)
    logger.info(f"Total hours to download: {total:,}")

    # Download all hours in parallel, collecting pre-aggregated M15 bars
    all_bars = []  # Each entry is a dict {Date, Open, High, Low, Close, Volume}
    completed = 0
    empty = 0
    tick_estimate = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_hour = {executor.submit(download_hour, h): h for h in hours}

        for future in as_completed(future_to_hour):
            bars = future.result()
            if bars:
                all_bars.extend(bars)
                tick_estimate += sum(b['Volume'] for b in bars)
            else:
                empty += 1
            completed += 1

            if completed % 2000 == 0:
                pct = completed / total * 100
                logger.info(
                    f"Progress: {completed:,}/{total:,} ({pct:.1f}%) "
                    f"- {len(all_bars):,} M15 bars, ~{tick_estimate:,} ticks"
                )

    logger.info(f"Download complete: {len(all_bars):,} M15 bars from {completed - empty:,} hours")
    logger.info(f"Empty hours (weekends/holidays): {empty:,}")
    logger.info(f"Estimated ticks processed: {tick_estimate:,}")

    if not all_bars:
        logger.error("No data downloaded!")
        return

    # Convert to DataFrame - this is only ~240K rows (M15 bars), not 258M ticks
    logger.info("Building DataFrame from pre-aggregated M15 bars...")
    ohlcv = pd.DataFrame(all_bars)
    ohlcv.sort_values('Date', inplace=True)
    ohlcv.reset_index(drop=True, inplace=True)

    # Save
    ohlcv.to_csv(OUTPUT_FILE, index=False)

    logger.info(f"\n{'='*70}")
    logger.info(f"DOWNLOAD COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"Total M15 bars: {len(ohlcv):,}")
    logger.info(f"Period: {ohlcv['Date'].iloc[0]} to {ohlcv['Date'].iloc[-1]}")
    logger.info(f"File: {OUTPUT_FILE}")
    logger.info(f"Size: {OUTPUT_FILE.stat().st_size / 1024 / 1024:.1f} MB")
    logger.info(f"Price range: ${ohlcv['Close'].min():.2f} - ${ohlcv['Close'].max():.2f}")

    # Gap check
    dates = pd.to_datetime(ohlcv['Date'])
    gaps = dates.diff()[dates.diff() > pd.Timedelta(hours=6)]
    logger.info(f"Weekend/holiday gaps: {len(gaps)}")


if __name__ == "__main__":
    main()
