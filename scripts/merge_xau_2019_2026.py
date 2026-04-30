"""Merge the clean XAU 2019-2024 file with the freshly downloaded 2025-2026 slice.

The 2019-2024 file (97.6 % coverage) ends 2024-12-30. The 2025-2026 Dukascopy
fetch starts 2024-12-30 and ends 2026-04-29. We drop the overlap day from the
new file (2024-12-30 belongs to the audited 2019-2024 file) and concatenate.

Output: data/XAU_15MIN_2019_2026.csv
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

OLD = Path("data/XAU_15MIN_2019_2024.csv")
NEW = Path("data/XAU_15MIN_2025_2026_dukascopy.csv")
OUT = Path("data/XAU_15MIN_2019_2026.csv")


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)


def main():
    print(f"Loading {OLD}")
    old = load(OLD)
    print(f"  {len(old):,} bars  range=[{old['Date'].min()} .. {old['Date'].max()}]")

    print(f"Loading {NEW}")
    new = load(NEW)
    print(f"  {len(new):,} bars  range=[{new['Date'].min()} .. {new['Date'].max()}]")

    cutoff = old["Date"].max()
    new_filtered = new[new["Date"] > cutoff]
    print(f"Dropping overlap ({len(new) - len(new_filtered):,} bars at <= {cutoff}); "
          f"keeping {len(new_filtered):,} new bars.")

    merged = pd.concat([old, new_filtered], ignore_index=True)
    merged = merged.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
    print(f"Merged: {len(merged):,} bars  range=[{merged['Date'].min()} .. {merged['Date'].max()}]")

    span_days = (merged["Date"].max() - merged["Date"].min()).days
    expected_24x5 = int(span_days * 5/7 * 23 * 4)  # 23h/day x 4 bars/h, 5 days/week
    coverage = len(merged) / max(expected_24x5, 1) * 100
    print(f"Estimated coverage vs 23x5: {coverage:.1f} %")

    merged.to_csv(OUT, index=False)
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
