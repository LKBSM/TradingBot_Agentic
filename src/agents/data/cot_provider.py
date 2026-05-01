"""
CFTC Disaggregated Commitments of Traders (COT) ingestion for Gold.

Sprint DATA-1.2 (Marwan, 4h). See `reports/roadmap_2026_2027/PLAN_12_MOIS.md`
Partie II.2 Agent 1.

Anti-leak guarantee
-------------------
The CFTC publication cycle has a critical pitfall:
  - Position data is captured at Tuesday's close (the "report_date")
  - The report is published the *following Friday at 15:30 ET*

A naive backtest joining COT data on observation date leaks 3 days of future
information (Tue close → Fri 15:30). This module computes a `vintage_date`
column = Friday 15:30 ET of the same week (converted to UTC), and `cot_at(t)`
guarantees only data with `vintage_date <= t` is ever returned.

Per DoD: at 14:30 ET Friday, cot_at returns the *previous* week's report
(this week's Tuesday is not yet published). At 16:00 ET Friday, the new
report becomes available.

Source URLs
-----------
- Per-year history (2019+ all stable):
    https://www.cftc.gov/files/dea/history/fut_disagg_txt_{year}.zip
  Each zip contains `f_year.txt` (CSV with 101 columns).

Gold code
---------
- 088691 = "GOLD - COMMODITY EXCHANGE INC." (COMEX 100 oz Gold futures)

Features computed
-----------------
- mm_net          = M_Money_Positions_Long_All - M_Money_Positions_Short_All
- mm_net_pct      = mm_net / Open_Interest_All
- mm_net_pct_z52  = (mm_net_pct - rolling_mean_52) / rolling_std_52
- producer_net    = Prod_Merc_Long - Prod_Merc_Short
- open_interest   = Open_Interest_All
"""

from __future__ import annotations

import io
import logging
import zipfile
from datetime import time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GOLD_CFTC_CODE = "088691"
"""COMEX 100 oz Gold futures contract market code."""

CFTC_DISAGG_URL = (
    "https://www.cftc.gov/files/dea/history/fut_disagg_txt_{year}.zip"
)
"""URL template for yearly Disaggregated Futures Only Reports."""

ET = ZoneInfo("America/New_York")
"""Eastern Time zone for CFTC publication schedule.

Note: handles US DST automatically (publication is in ET local time, so the
UTC offset varies seasonally — using ZoneInfo means we get this correctly
without hand-rolling DST logic).
"""

PUBLICATION_TIME_ET = time(15, 30)
"""CFTC publishes reports Friday 15:30 ET.

In rare cases (federal holidays) this slips by 1 day. We do not model that
exception in Phase 1 — it is conservative to assume on-time publication
because consumers checking earlier than the actual release will see
`vintage_date > t` and correctly get the previous week's report. The 1-day
slip case can leak by at most 1 day, which is acceptable for Phase 1.
"""

# Output schema for the post-processed COT frame.
OUTPUT_COLUMNS = [
    "report_date",
    "vintage_date",
    "mm_long",
    "mm_short",
    "mm_net",
    "open_interest",
    "mm_net_pct",
    "mm_net_pct_z52",
    "producer_long",
    "producer_short",
    "producer_net",
]


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class CotProvider:
    """Vintage-aware CFTC COT provider for Gold (COMEX 088691).

    Parameters
    ----------
    cache_dir : str | Path
        Directory where downloaded yearly ZIPs are cached. Default
        ``data/macro/cot_cache``. Re-runs reuse the cache (no re-download).
    request_timeout : int
        HTTP timeout in seconds for CFTC downloads.
    cftc_code : str
        Contract market code; default Gold COMEX.
    """

    def __init__(
        self,
        cache_dir: str | Path = "data/macro/cot_cache",
        request_timeout: int = 30,
        cftc_code: str = GOLD_CFTC_CODE,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._timeout = request_timeout
        self._code = cftc_code

    # ------------------------------------------------------------------
    # Network
    # ------------------------------------------------------------------

    def _download_yearly_zip(self, year: int, force: bool = False) -> Path:
        """Download the CFTC yearly ZIP if not already cached."""
        local = self.cache_dir / f"fut_disagg_txt_{year}.zip"
        if local.exists() and not force:
            logger.debug("Cache hit: %s", local)
            return local
        url = CFTC_DISAGG_URL.format(year=year)
        logger.info("Downloading %s", url)
        resp = requests.get(url, timeout=self._timeout)
        resp.raise_for_status()
        local.write_bytes(resp.content)
        return local

    # ------------------------------------------------------------------
    # Parse
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_zip(zip_path: Path) -> pd.DataFrame:
        """Extract and parse the f_year.txt CSV inside the CFTC ZIP.

        Note: ``CFTC_Contract_Market_Code`` is forced to string dtype because
        Gold's code "088691" otherwise loses its leading zero when pandas
        infers int (real CFTC files happen to have non-numeric codes too,
        masking this issue, but synthetic zips trigger it).
        """
        with zipfile.ZipFile(zip_path) as z:
            names = z.namelist()
            if not names:
                raise ValueError(f"{zip_path} is empty")
            # CFTC zip historically contains exactly one .txt file.
            txt_name = next((n for n in names if n.endswith(".txt")), names[0])
            with z.open(txt_name) as f:
                # Read all bytes (one year ~2-5MB) then parse — simpler than
                # streaming and dataset is small.
                df = pd.read_csv(
                    io.BytesIO(f.read()),
                    low_memory=False,
                    dtype={"CFTC_Contract_Market_Code": str},
                )
        return df

    def fetch_year(self, year: int) -> pd.DataFrame:
        """Download (if needed) + parse + filter to Gold for one year."""
        path = self._download_yearly_zip(year)
        raw = self._parse_zip(path)
        gold = raw[raw["CFTC_Contract_Market_Code"].astype(str) == self._code]
        if gold.empty:
            logger.warning("No %s rows in %s", self._code, path.name)
        return gold.copy()

    def fetch_range(self, start_year: int, end_year: int) -> pd.DataFrame:
        """Concatenate all years inclusive."""
        frames = []
        for y in range(start_year, end_year + 1):
            try:
                frames.append(self.fetch_year(y))
            except Exception as exc:  # noqa: BLE001 — log and skip year
                logger.warning("fetch_year(%d) failed: %s", y, exc)
        if not frames:
            return _empty_frame()
        return pd.concat(frames, ignore_index=True)

    # ------------------------------------------------------------------
    # Feature engineering (vintage-aware)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_features(raw: pd.DataFrame) -> pd.DataFrame:
        """Convert raw CFTC rows to the schema in OUTPUT_COLUMNS.

        Key step: ``vintage_date = friday_after(report_date) at 15:30 ET → UTC``.
        """
        if raw.empty:
            return _empty_frame()

        df = pd.DataFrame()
        df["report_date"] = pd.to_datetime(
            raw["Report_Date_as_YYYY-MM-DD"], utc=True
        )
        df["vintage_date"] = df["report_date"].apply(_vintage_date_for_report)

        df["mm_long"] = pd.to_numeric(raw["M_Money_Positions_Long_All"]).astype(
            "Int64"
        )
        df["mm_short"] = pd.to_numeric(
            raw["M_Money_Positions_Short_All"]
        ).astype("Int64")
        df["open_interest"] = pd.to_numeric(raw["Open_Interest_All"]).astype(
            "Int64"
        )
        df["producer_long"] = pd.to_numeric(
            raw["Prod_Merc_Positions_Long_All"]
        ).astype("Int64")
        df["producer_short"] = pd.to_numeric(
            raw["Prod_Merc_Positions_Short_All"]
        ).astype("Int64")

        df["mm_net"] = (df["mm_long"] - df["mm_short"]).astype("Int64")
        df["producer_net"] = (df["producer_long"] - df["producer_short"]).astype(
            "Int64"
        )

        df = df.sort_values("report_date").reset_index(drop=True)
        df["mm_net_pct"] = df["mm_net"].astype("float64") / df[
            "open_interest"
        ].astype("float64")

        # 52-week rolling z-score; min_periods=26 = ~half-year, before that NaN
        roll = df["mm_net_pct"].rolling(window=52, min_periods=26)
        df["mm_net_pct_z52"] = (df["mm_net_pct"] - roll.mean()) / roll.std(
            ddof=0
        )

        return df[OUTPUT_COLUMNS]

    # ------------------------------------------------------------------
    # Lookup (vintage-safe)
    # ------------------------------------------------------------------

    @staticmethod
    def cot_at(df: pd.DataFrame, t) -> pd.Series | None:
        """Return the COT row valid at time ``t`` (vintage-aware).

        Only rows with ``vintage_date <= t`` are eligible. The most-recent
        eligible row is returned. ``None`` if nothing is available yet.

        DoD test fixture: at 2024-03-01 14:30 ET (Friday before 15:30) we
        return the 2024-02-20 report. At 2024-03-01 16:00 ET we return the
        2024-02-27 report.
        """
        ts = _ensure_utc_timestamp(t)
        avail = df[df["vintage_date"] <= ts]
        if avail.empty:
            return None
        # Most-recent already-published report
        return avail.sort_values("vintage_date").iloc[-1]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @staticmethod
    def save_to_csv(df: pd.DataFrame, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False, date_format="%Y-%m-%dT%H:%M:%S%z")
        logger.info("Saved %d COT rows -> %s", len(df), out)
        return out

    @staticmethod
    def load_from_csv(path: str | Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        df["report_date"] = pd.to_datetime(df["report_date"], utc=True)
        df["vintage_date"] = pd.to_datetime(df["vintage_date"], utc=True)
        # Numeric columns
        for col in [
            "mm_long",
            "mm_short",
            "mm_net",
            "open_interest",
            "producer_long",
            "producer_short",
            "producer_net",
        ]:
            df[col] = pd.to_numeric(df[col]).astype("Int64")
        for col in ["mm_net_pct", "mm_net_pct_z52"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df[OUTPUT_COLUMNS]


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _empty_frame() -> pd.DataFrame:
    cols = {
        "report_date": pd.Series(dtype="datetime64[ns, UTC]"),
        "vintage_date": pd.Series(dtype="datetime64[ns, UTC]"),
        "mm_long": pd.Series(dtype="Int64"),
        "mm_short": pd.Series(dtype="Int64"),
        "mm_net": pd.Series(dtype="Int64"),
        "open_interest": pd.Series(dtype="Int64"),
        "mm_net_pct": pd.Series(dtype="float64"),
        "mm_net_pct_z52": pd.Series(dtype="float64"),
        "producer_long": pd.Series(dtype="Int64"),
        "producer_short": pd.Series(dtype="Int64"),
        "producer_net": pd.Series(dtype="Int64"),
    }
    return pd.DataFrame(cols)


def _vintage_date_for_report(report_date: pd.Timestamp) -> pd.Timestamp:
    """Compute publication vintage = Friday 15:30 ET after the report Tuesday.

    The CFTC reports positions taken at Tuesday's close, published the
    following Friday at 15:30 ET. We construct that Friday as a tz-aware
    timestamp in America/New_York then convert to UTC.

    Implementation note: CFTC's "Report_Date_as_YYYY-MM-DD" is a calendar
    date with no time component. When pandas parses it as UTC midnight, the
    *date-part* (year/month/day) is correct in any timezone interpretation.
    We extract those components directly rather than `tz_convert(ET)`
    (which would shift to ET 20:00 the day before and corrupt the date).

    Parameters
    ----------
    report_date : pd.Timestamp
        Tuesday close date (naive or UTC-aware). Only the calendar date matters.

    Returns
    -------
    pd.Timestamp
        Friday 15:30 ET expressed in UTC.
    """
    # Take the calendar date components — independent of tz interpretation.
    tuesday = pd.Timestamp(
        year=report_date.year, month=report_date.month, day=report_date.day
    )
    # Tuesday + 3 days = Friday.
    friday = tuesday + pd.Timedelta(days=3)
    # Build a tz-aware ET timestamp at 15:30 then convert to UTC.
    friday_naive = pd.Timestamp(
        year=friday.year,
        month=friday.month,
        day=friday.day,
        hour=PUBLICATION_TIME_ET.hour,
        minute=PUBLICATION_TIME_ET.minute,
    )
    return friday_naive.tz_localize(ET).tz_convert("UTC")


def _ensure_utc_timestamp(t) -> pd.Timestamp:
    ts = pd.Timestamp(t)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")
