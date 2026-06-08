"""Twelve Data REST API provider for OHLCV (XAU + EURUSD).

Free tier limits (enforced by the embedded rate limiter):
  - 8 requests per minute
  - 800 requests per day

The free tier is what MIA Markets V1 backend uses (cf. docs/architecture/
MIA_MARKETS_V2_VISION.md §3.2). Upgrade path : Twelve Data Basic ($79/mo,
50k req/day) when ~50+ active users.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Deque, Dict, List, Optional, Tuple

import pandas as pd
import requests

from src.intelligence.data_providers.base import DataProvider

logger = logging.getLogger(__name__)


_SYMBOL_MAP: Dict[str, str] = {
    "XAUUSD": "XAU/USD",
    "EURUSD": "EUR/USD",
}

_TIMEFRAME_MAP: Dict[str, str] = {
    "M1": "1min",
    "M5": "5min",
    "M15": "15min",
    "M30": "30min",
    "H1": "1h",
    "H4": "4h",
    "D1": "1day",
    "W1": "1week",
}


@dataclass(frozen=True)
class Candle:
    """A single OHLCV bar as a typed value object."""

    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class TwelveDataError(Exception):
    """Base error for the Twelve Data provider."""


class TwelveDataAuthError(TwelveDataError):
    """Authentication failure (HTTP 401/403 or API error envelope code 401/403)."""


class TwelveDataRateLimiter:
    """Sliding-window rate limiter for per-minute + per-day caps.

    Thread-safe. Blocks the calling thread until both windows have capacity.
    The clock and sleep functions are injectable for deterministic tests.
    """

    def __init__(
        self,
        per_minute: int = 8,
        per_day: int = 800,
        sleep_fn: Callable[[float], None] = time.sleep,
        now_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self._per_minute = per_minute
        self._per_day = per_day
        self._minute_window: Deque[float] = deque()
        self._day_window: Deque[float] = deque()
        self._lock = threading.Lock()
        self._sleep_fn = sleep_fn
        self._now_fn = now_fn

    def acquire(self) -> None:
        """Block until both per-minute and per-day windows have a free slot."""
        with self._lock:
            while True:
                now = self._now_fn()
                while self._minute_window and now - self._minute_window[0] >= 60.0:
                    self._minute_window.popleft()
                while self._day_window and now - self._day_window[0] >= 86400.0:
                    self._day_window.popleft()

                if (
                    len(self._minute_window) < self._per_minute
                    and len(self._day_window) < self._per_day
                ):
                    self._minute_window.append(now)
                    self._day_window.append(now)
                    return

                if len(self._minute_window) >= self._per_minute:
                    sleep_s = max(0.0, 60.0 - (now - self._minute_window[0])) + 0.05
                else:
                    sleep_s = max(0.0, 86400.0 - (now - self._day_window[0])) + 0.05

                logger.info(
                    "TwelveDataRateLimiter: sleeping %.2fs (minute=%d/%d, day=%d/%d)",
                    sleep_s,
                    len(self._minute_window),
                    self._per_minute,
                    len(self._day_window),
                    self._per_day,
                )
                self._sleep_fn(sleep_s)


class TwelveDataProvider(DataProvider):
    """OHLCV provider backed by the Twelve Data REST API.

    Behaviour:
      - Rate-limited (8/min, 800/day on free tier by default).
      - Exponential backoff retry on HTTP 429 and 5xx (up to MAX_RETRIES).
      - Fail-fast on 401/403 — raises ``TwelveDataAuthError`` without retry.
      - In-memory TTL cache on (symbol, timeframe, lookback) tuples.

    The provider inherits from ``DataProvider`` so it's interchangeable with
    ``CSVDataProvider`` and ``MT5DataProvider`` in the downstream pipeline.

    Usage:
        provider = TwelveDataProvider()  # reads TWELVE_DATA_API_KEY from env
        df = provider.get_ohlcv("XAUUSD", "M15", 100)
    """

    BASE_URL = "https://api.twelvedata.com"
    MAX_RETRIES = 4
    BASE_BACKOFF_S = 1.0
    REQUEST_TIMEOUT_S = 20.0

    def __init__(
        self,
        api_key: Optional[str] = None,
        per_minute: int = 8,
        per_day: int = 800,
        cache_ttl_s: float = 60.0,
        session: Optional[requests.Session] = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        api_key = api_key or os.environ.get("TWELVE_DATA_API_KEY")
        if not api_key:
            raise ValueError(
                "TwelveDataProvider requires api_key argument or "
                "TWELVE_DATA_API_KEY environment variable"
            )
        self._api_key = api_key
        self._session = session or requests.Session()
        self._rate_limiter = TwelveDataRateLimiter(
            per_minute=per_minute, per_day=per_day, sleep_fn=sleep_fn
        )
        self._cache: Dict[Tuple[str, str, int], Tuple[float, pd.DataFrame]] = {}
        self._cache_ttl_s = cache_ttl_s
        self._cache_lock = threading.Lock()
        self._sleep_fn = sleep_fn

    @staticmethod
    def _map_symbol(symbol: str) -> str:
        try:
            return _SYMBOL_MAP[symbol]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported symbol: {symbol!r}. Supported: {sorted(_SYMBOL_MAP)}"
            ) from exc

    @staticmethod
    def _map_timeframe(timeframe: str) -> str:
        try:
            return _TIMEFRAME_MAP[timeframe]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported timeframe: {timeframe!r}. Supported: {sorted(_TIMEFRAME_MAP)}"
            ) from exc

    def get_ohlcv(self, symbol: str, timeframe: str, lookback: int) -> pd.DataFrame:
        """Fetch OHLCV bars as a DataFrame (``DataProvider`` interface)."""
        cache_key = (symbol, timeframe, lookback)
        now = time.monotonic()
        with self._cache_lock:
            cached = self._cache.get(cache_key)
            if cached is not None:
                ts, cached_df = cached
                if now - ts < self._cache_ttl_s:
                    return cached_df.copy()

        df = self._fetch_dataframe(symbol, timeframe, lookback)
        with self._cache_lock:
            self._cache[cache_key] = (now, df)
        return df.copy()

    def fetch_candles(self, symbol: str, timeframe: str, count: int) -> List[Candle]:
        """Fetch OHLCV as a list of typed ``Candle`` objects."""
        df = self.get_ohlcv(symbol, timeframe, count)
        return [
            Candle(
                ts=ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=float(row["Volume"]),
            )
            for ts, row in df.iterrows()
        ]

    def available_symbols(self) -> List[str]:
        return list(_SYMBOL_MAP.keys())

    def _fetch_dataframe(
        self, symbol: str, timeframe: str, lookback: int
    ) -> pd.DataFrame:
        td_symbol = self._map_symbol(symbol)
        td_interval = self._map_timeframe(timeframe)

        params = {
            "symbol": td_symbol,
            "interval": td_interval,
            "outputsize": lookback,
            "apikey": self._api_key,
            "format": "JSON",
        }
        url = f"{self.BASE_URL}/time_series"

        backoff = self.BASE_BACKOFF_S
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            self._rate_limiter.acquire()
            try:
                resp = self._session.get(url, params=params, timeout=self.REQUEST_TIMEOUT_S)
            except requests.RequestException as exc:
                last_exc = exc
                logger.warning(
                    "TwelveData network error (attempt %d/%d): %s",
                    attempt, self.MAX_RETRIES, exc,
                )
                if attempt < self.MAX_RETRIES:
                    self._sleep_fn(backoff)
                    backoff *= 2
                    continue
                raise TwelveDataError(
                    f"Network error after {attempt} attempts: {exc}"
                ) from exc

            status = resp.status_code
            if status in (401, 403):
                raise TwelveDataAuthError(
                    f"Authentication failed (HTTP {status}). Check TWELVE_DATA_API_KEY."
                )

            if status == 429 or 500 <= status < 600:
                logger.warning(
                    "TwelveData HTTP %d (attempt %d/%d), backoff %.1fs",
                    status, attempt, self.MAX_RETRIES, backoff,
                )
                if attempt < self.MAX_RETRIES:
                    self._sleep_fn(backoff)
                    backoff *= 2
                    continue
                raise TwelveDataError(
                    f"HTTP {status} after {attempt} attempts: {resp.text[:200]}"
                )

            if not resp.ok:
                raise TwelveDataError(f"HTTP {status}: {resp.text[:200]}")

            body = resp.json()
            if body.get("status") == "error":
                code = body.get("code")
                msg = body.get("message", "unknown error")
                if code in (401, 403):
                    raise TwelveDataAuthError(f"API error {code}: {msg}")
                raise TwelveDataError(f"API error {code}: {msg}")

            return self._parse_time_series(body)

        raise TwelveDataError(f"Exhausted retries; last error: {last_exc}")

    @staticmethod
    def _parse_time_series(body: dict) -> pd.DataFrame:
        values = body.get("values")
        if not values:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

        df = pd.DataFrame(values)
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df = df.set_index("datetime").sort_index()

        for col in ("open", "high", "low", "close", "volume"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.rename(columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        })
        # Volume is not always reported for FX/metal feeds — default to 0.
        if "Volume" not in df.columns:
            df["Volume"] = 0.0
        return df[["Open", "High", "Low", "Close", "Volume"]]
