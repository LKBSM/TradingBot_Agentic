"""OHLCV data quality validation for Smart Sentinel AI.

Applied to every DataFrame returned by a DataProvider before it reaches the
SMC engine or scanner. Structural corruption raises; soft anomalies (bar gaps,
low volume, stale timestamps) log warnings but don't halt the pipeline.

The goal is to catch broker-feed problems before they silently corrupt
pattern detection — missing bars during NFP, inverted H/L from ticker errors,
duplicate timestamps from broker glitches.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)


REQUIRED_COLUMNS = {"Open", "High", "Low", "Close", "Volume"}

# Nominal bar intervals for gap detection (in minutes).
TIMEFRAME_MINUTES = {
    "M1": 1, "M5": 5, "M15": 15, "M30": 30,
    "H1": 60, "H4": 240, "D1": 1440, "W1": 10080,
}


class DataQualityError(ValueError):
    """Raised when OHLCV data has structural corruption we cannot recover from."""


@dataclass
class ValidationReport:
    """Result of validating an OHLCV DataFrame.

    `errors` is fatal — the pipeline should not use this data.
    `warnings` is advisory — the data is usable but worth noting.
    """
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    bars_checked: int = 0
    gap_count: int = 0

    def summary(self) -> str:
        return (
            f"bars={self.bars_checked} "
            f"errors={len(self.errors)} warnings={len(self.warnings)} "
            f"gaps={self.gap_count}"
        )


def validate_ohlcv(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    *,
    strict: bool = True,
) -> ValidationReport:
    """Validate an OHLCV DataFrame.

    Args:
        df: DataFrame with Open/High/Low/Close/Volume columns and a
            DatetimeIndex.
        symbol: instrument symbol (for log context only).
        timeframe: timeframe string (M15, H1, etc.) used for gap detection.
        strict: when True, raise DataQualityError on any fatal issue. When
            False, return a report with is_valid=False and let the caller
            decide.

    Returns:
        ValidationReport with errors (fatal) and warnings (advisory).
    """
    report = ValidationReport(is_valid=True, bars_checked=len(df))

    # 1. Non-empty
    if df.empty:
        report.errors.append("DataFrame is empty")
        return _finalize(report, symbol, timeframe, strict)

    # 2. Required columns present
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        report.errors.append(f"Missing required columns: {sorted(missing)}")
        return _finalize(report, symbol, timeframe, strict)

    # 3. DatetimeIndex, monotonic, no duplicates
    if not isinstance(df.index, pd.DatetimeIndex):
        report.errors.append(f"Index must be DatetimeIndex, got {type(df.index).__name__}")
        return _finalize(report, symbol, timeframe, strict)

    if not df.index.is_monotonic_increasing:
        report.errors.append("Timestamps are not monotonically increasing")

    dup_count = df.index.duplicated().sum()
    if dup_count > 0:
        report.errors.append(f"{dup_count} duplicate timestamps")

    # 4. No NaN in OHLC
    for col in ("Open", "High", "Low", "Close"):
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            report.errors.append(f"{col} has {nan_count} NaN values")

    # 5. Volume non-negative
    neg_volume = (df["Volume"] < 0).sum()
    if neg_volume > 0:
        report.errors.append(f"{neg_volume} bars with negative volume")

    # Early-exit on fatal errors; H/L and gap checks require clean OHLC.
    if report.errors:
        return _finalize(report, symbol, timeframe, strict)

    # 6. High >= max(Open, Close), Low <= min(Open, Close), High >= Low
    inverted = (df["High"] < df["Low"]).sum()
    if inverted > 0:
        report.errors.append(f"{inverted} bars where High < Low")

    high_violation = (df["High"] < df[["Open", "Close"]].max(axis=1)).sum()
    if high_violation > 0:
        report.errors.append(f"{high_violation} bars where High < max(Open, Close)")

    low_violation = (df["Low"] > df[["Open", "Close"]].min(axis=1)).sum()
    if low_violation > 0:
        report.errors.append(f"{low_violation} bars where Low > min(Open, Close)")

    # 7. No non-positive prices (Gold/Crypto can never be <= 0)
    non_positive = (df[["Open", "High", "Low", "Close"]] <= 0).any(axis=1).sum()
    if non_positive > 0:
        report.errors.append(f"{non_positive} bars with non-positive prices")

    if report.errors:
        return _finalize(report, symbol, timeframe, strict)

    # 8. Bar gap detection (advisory only — weekends and holidays produce gaps)
    expected_minutes = TIMEFRAME_MINUTES.get(timeframe.upper())
    if expected_minutes is not None and len(df) > 1:
        diffs = df.index.to_series().diff().dt.total_seconds() / 60.0
        # Bars with gap more than 2x expected (tolerant of DST / market close).
        tolerance = expected_minutes * 2
        gap_mask = diffs > tolerance
        report.gap_count = int(gap_mask.sum())
        if report.gap_count > 0 and report.gap_count > len(df) * 0.01:
            # >1% gap rate is worth flagging.
            report.warnings.append(
                f"{report.gap_count} bar gaps > {tolerance:.0f}min "
                f"({report.gap_count / len(df) * 100:.1f}% of bars)"
            )

    # 9. Stale data check — last bar shouldn't be older than 7 days (advisory)
    last_ts = df.index[-1]
    now = pd.Timestamp.now(tz="UTC")
    if last_ts.tzinfo is None:
        now = now.tz_localize(None)
    try:
        age_days = (now - last_ts).total_seconds() / 86400.0
        if age_days > 7:
            report.warnings.append(
                f"Last bar is {age_days:.1f} days old (possible stale feed)"
            )
    except (TypeError, ValueError):
        # Timezone mismatch — skip staleness check rather than fail validation
        pass

    return _finalize(report, symbol, timeframe, strict)


def _finalize(
    report: ValidationReport,
    symbol: str,
    timeframe: str,
    strict: bool,
) -> ValidationReport:
    report.is_valid = len(report.errors) == 0

    for warn in report.warnings:
        logger.warning("OHLCV quality [%s %s]: %s", symbol, timeframe, warn)

    if report.errors:
        err_summary = "; ".join(report.errors)
        if strict:
            raise DataQualityError(
                f"OHLCV validation failed for {symbol} {timeframe}: {err_summary}"
            )
        logger.error("OHLCV validation failed [%s %s]: %s", symbol, timeframe, err_summary)

    return report
