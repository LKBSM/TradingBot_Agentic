"""OHLCV data quality audit — full diagnostic report."""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

def audit(path: Path, expected_freq_min: int = 15):
    print(f"\n{'='*78}\n  AUDIT: {path.name}\n{'='*78}")
    df = pd.read_csv(path)
    print(f"Columns: {list(df.columns)}")
    print(f"Rows: {len(df):,}")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Range
    first, last = df["Date"].iloc[0], df["Date"].iloc[-1]
    span_days = (last - first).days
    print(f"Range: {first} → {last}  ({span_days} days / {span_days/365.25:.2f} years)")

    # Duplicates
    dupes = df["Date"].duplicated().sum()
    print(f"Duplicate timestamps: {dupes}")

    # OHLC integrity
    bad_hi = ((df["High"] < df["Open"]) | (df["High"] < df["Close"]) | (df["High"] < df["Low"])).sum()
    bad_lo = ((df["Low"] > df["Open"]) | (df["Low"] > df["Close"]) | (df["Low"] > df["High"])).sum()
    print(f"OHLC integrity — bad High: {bad_hi}  bad Low: {bad_lo}")

    # Flat bars (O==H==L==C)
    flat = ((df["Open"] == df["High"]) & (df["High"] == df["Low"]) & (df["Low"] == df["Close"])).sum()
    print(f"Flat bars (O=H=L=C): {flat}  ({flat/len(df)*100:.2f}%)")

    # Zero-range bars (H==L)
    zero_range = (df["High"] == df["Low"]).sum()
    print(f"Zero-range bars (H=L): {zero_range}  ({zero_range/len(df)*100:.2f}%)")

    # Volume
    zero_vol = (df["Volume"] == 0).sum()
    null_vol = df["Volume"].isna().sum()
    print(f"Zero volume: {zero_vol}  ({zero_vol/len(df)*100:.2f}%)")
    print(f"Null volume: {null_vol}")
    print(f"Volume stats — min: {df['Volume'].min()}  median: {df['Volume'].median():.0f}  "
          f"mean: {df['Volume'].mean():.0f}  max: {df['Volume'].max():,}")

    # Gap analysis
    diffs = df["Date"].diff().dt.total_seconds().div(60).dropna()
    expected = expected_freq_min
    print(f"\nExpected bar spacing: {expected} min")
    print(f"Actual spacing — min: {diffs.min():.0f} min  median: {diffs.median():.0f} min  "
          f"max: {diffs.max():.0f} min")
    normal = (diffs == expected).sum()
    print(f"Normal gaps ({expected} min): {normal:,}  ({normal/len(diffs)*100:.2f}%)")
    weekend_like = ((diffs > 60*24) & (diffs < 60*72)).sum()  # 1-3 day gaps
    print(f"Weekend-sized gaps (1-3 days): {weekend_like}")
    big_gaps = diffs[diffs > 60*72]  # > 3 days
    print(f"Suspicious gaps (>3 days): {len(big_gaps)}")
    if len(big_gaps) > 0 and len(big_gaps) <= 10:
        print(f"  Examples: {big_gaps.head(10).tolist()} min")

    # Weekend bars (XAU trades Sun 22:00 UTC to Fri 22:00 UTC — Saturdays should be empty)
    df["dow"] = df["Date"].dt.dayofweek  # 0=Mon .. 6=Sun
    weekend_bars = df[df["dow"] == 5]  # Saturday bars (should be rare/none for Gold)
    print(f"\nSaturday bars: {len(weekend_bars)}  (Gold market closed Saturdays)")
    sunday_bars = df[df["dow"] == 6]
    print(f"Sunday bars: {len(sunday_bars)}  (only evening Sunday is normal)")

    # Hourly coverage (detect missing sessions)
    df["hour"] = df["Date"].dt.hour
    hour_counts = df.groupby("hour").size()
    min_h, max_h = hour_counts.min(), hour_counts.max()
    print(f"Bars per hour — min: {min_h:,}  max: {max_h:,}  ratio: {max_h/max(min_h,1):.2f}")
    if min_h / max(max_h, 1) < 0.5:
        sparse_hours = hour_counts[hour_counts < hour_counts.median() * 0.5].index.tolist()
        print(f"  Sparse hours (possible missing sessions): {sparse_hours}")

    # Outlier returns
    df["ret"] = df["Close"].pct_change()
    extreme = df["ret"].abs() > 0.05  # > 5% move on 15-min bar is suspicious for Gold
    print(f"\nExtreme 15-min moves (>5%): {extreme.sum()}")
    if extreme.sum() > 0 and extreme.sum() <= 5:
        for i in df.index[extreme][:5]:
            print(f"  {df.loc[i,'Date']}  ret={df.loc[i,'ret']*100:+.2f}%  "
                  f"close {df.loc[max(i-1,0),'Close']} → {df.loc[i,'Close']}")

    # Price decimals (Gold should be 2 decimals; forex 5; JPY 3)
    sample = df["Close"].head(100).astype(str)
    decimals = sample.apply(lambda s: len(s.split(".")[1]) if "." in s else 0)
    print(f"Price decimals — median: {int(decimals.median())}  max: {decimals.max()}  "
          f"min: {decimals.min()}")

    # Expected coverage vs actual (for 15-min Gold: ~96 bars/day × ~252 trading days/yr)
    expected_per_year = 252 * 96
    actual_per_year = len(df) / (span_days / 365.25) if span_days > 0 else 0
    print(f"\nBars per year — expected ~{expected_per_year:,}  actual: {actual_per_year:,.0f}  "
          f"coverage: {actual_per_year/expected_per_year*100:.1f}%")

    return df

if __name__ == "__main__":
    data_dir = Path("data")
    for f in ["XAU_15MIN_2019_2026.csv", "XAU_15MIN_2019_2025.csv", "XAU_15MIN_2019_2024.csv"]:
        p = data_dir / f
        if p.exists():
            audit(p)
