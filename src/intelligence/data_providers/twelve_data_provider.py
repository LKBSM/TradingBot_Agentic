"""Twelve Data REST API provider for OHLCV.

Rate limits are read from the environment so the plan can be changed without a
code change (audit DATA-3 measured that they were frozen at the free tier, which
made paying for a bigger plan pointless):

  - ``TWELVE_DATA_PER_MINUTE`` — credits per minute (default 8 = free tier)
  - ``TWELVE_DATA_PER_DAY``    — credits per day    (default 800 = free tier)

One request = one symbol = **one credit**, whatever the interval and whatever
``outputsize`` (verified live: ``Api-Credits-Request: 1`` on a 5 000-bar page).
So the only thing that lowers consumption is issuing FEWER requests — never
smaller ones.

Three measured credit leaks are closed here (audit DATA-3 §7) :
  1. a 429 used to be retried up to ``MAX_RETRIES`` times, and a refused request
     is billed all the same — one failing fetch cost 4 credits. 429 is terminal.
  2. every response carries the server's own counter (``Api-Credits-Used`` /
     ``Api-Credits-Left``); it is now the authority, so the client stops issuing
     requests it already knows will be refused instead of spending them to find out.
  3. the symbol table was hard-coded to two markets; it now derives from the
     market registry (MKT-1), like the interval table already derived from TF-1.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import os
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Deque, Dict, Iterator, List, Optional, Tuple

import pandas as pd
import requests

from src.intelligence.data_providers.base import DataProvider

logger = logging.getLogger(__name__)


# Timeframe → Twelve Data interval string, derived from the single timeframe
# registry (TF-1), not a second copy.
from src.intelligence import timeframe_registry as _tfreg

_TIMEFRAME_MAP: Dict[str, str] = _tfreg.provider_map()


# --------------------------------------------------------------------------- #
# Symbol table — derived from the market registry (MKT-1), never a second copy
# --------------------------------------------------------------------------- #
def _derive_provider_symbol(spec: object, market_id: str) -> str:
    """Twelve Data ticker for a market.

    Explicit wins: ``providerSymbol`` in ``config/markets.json`` is used verbatim
    when present. Otherwise a 6-letter FX/metal/crypto pair is split in the middle
    (``XAUUSD`` → ``XAU/USD``), the shape Twelve Data expects; anything else (an
    index ticker) passes through unchanged. Adding an 81st market is therefore one
    entry in the registry, with an explicit override available the day a ticker
    does not follow the convention.
    """
    explicit = getattr(spec, "provider_symbol", None)
    if explicit:
        return str(explicit)
    raw = str(getattr(spec, "symbol", None) or market_id).upper()
    if "/" in raw:
        return raw
    mtype = str(getattr(spec, "type", "") or "").lower()
    if len(raw) == 6 and mtype in ("fx", "metal", "crypto"):
        return f"{raw[:3]}/{raw[3:]}"
    return raw


def _symbol_map() -> Dict[str, str]:
    """market id → Twelve Data ticker, for every market in the registry."""
    from src.intelligence import market_registry as _mktreg

    return {
        spec.id: _derive_provider_symbol(spec, spec.id) for spec in _mktreg.all_specs()
    }


# --------------------------------------------------------------------------- #
# Credit attribution — which trigger spent the budget (audit DATA-3 §a.2)
# --------------------------------------------------------------------------- #
_purpose: contextvars.ContextVar[str] = contextvars.ContextVar(
    "twelve_data_purpose", default="unknown"
)


@contextlib.contextmanager
def credit_purpose(name: str) -> Iterator[None]:
    """Label every provider call made inside this block.

    ``/health`` reports credits per label, so a drift is attributable to the
    trigger that caused it instead of surfacing as one opaque total.
    """
    token = _purpose.set(str(name))
    try:
        yield
    finally:
        _purpose.reset(token)


def _env_int(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


#: Plan limits. Defaults = free tier, so an unconfigured deployment behaves
#: exactly as before; set the env vars when the plan changes.
_PER_MINUTE_ENV = "TWELVE_DATA_PER_MINUTE"
_PER_DAY_ENV = "TWELVE_DATA_PER_DAY"
DEFAULT_PER_MINUTE = 8
DEFAULT_PER_DAY = 800

#: How long to stand down once the server says the minute budget is gone. Twelve
#: Data publishes no ``Retry-After`` and no reset header (verified live); the
#: window is a rolling minute and a probe succeeded after 70 s.
_COOLDOWN_ENV = "TWELVE_DATA_COOLDOWN_S"
DEFAULT_COOLDOWN_S = 61.0


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


class TwelveDataRateLimited(TwelveDataError):
    """The per-minute credit budget is gone (HTTP 429, or envelope ``code: 429``).

    Terminal on purpose: a refused request is BILLED (measured — the server's
    ``Api-Credits-Used`` still increments on the 429), so retrying it spends more
    of the budget that is already exhausted. Callers back off and come back.
    """

    def __init__(self, message: str, cooldown_s: float = DEFAULT_COOLDOWN_S) -> None:
        super().__init__(message)
        self.cooldown_s = cooldown_s


class TwelveDataRateLimiter:
    """Sliding-window rate limiter for per-minute + per-day caps.

    Thread-safe. Blocks the calling thread until both windows have capacity.
    The clock and sleep functions are injectable for deterministic tests.

    DATA-3: the local sliding window is a *prediction*; the authority is the
    server, which returns its own counter on every response. ``sync_from_server``
    feeds that back in, so when the budget is really gone the limiter holds the
    line instead of letting callers discover it by spending a credit on a 429.
    """

    def __init__(
        self,
        per_minute: Optional[int] = None,
        per_day: Optional[int] = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        now_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self._per_minute = (
            per_minute if per_minute is not None
            else _env_int(_PER_MINUTE_ENV, DEFAULT_PER_MINUTE)
        )
        self._per_day = (
            per_day if per_day is not None
            else _env_int(_PER_DAY_ENV, DEFAULT_PER_DAY)
        )
        self._minute_window: Deque[float] = deque()
        self._day_window: Deque[float] = deque()
        self._lock = threading.Lock()
        self._sleep_fn = sleep_fn
        self._now_fn = now_fn
        #: monotonic instant before which no request may leave (server said stop)
        self._blocked_until: float = 0.0
        #: credits spent per trigger label, for /health attribution
        self._by_purpose: Dict[str, int] = defaultdict(int)
        #: last counter the server itself reported
        self._server_used: Optional[int] = None
        self._server_left: Optional[int] = None

    def acquire(self, purpose: str = "unknown") -> None:
        """Block until both per-minute and per-day windows have a free slot.

        PERF-3: the wait happens OUTSIDE the lock. This used to sleep while
        holding it, so a thread waiting on the DAILY cap — where ``sleep_s`` can
        reach ~86 400 s — froze every other caller in the process, including ones
        whose own window had capacity. The lock now only guards the state; it is
        re-checked after each wake, so the accounting is unchanged and two threads
        can never claim the same slot.
        """
        while True:
            with self._lock:
                now = self._now_fn()
                while self._minute_window and now - self._minute_window[0] >= 60.0:
                    self._minute_window.popleft()
                while self._day_window and now - self._day_window[0] >= 86400.0:
                    self._day_window.popleft()

                if now < self._blocked_until:
                    # The server told us the budget is gone — waiting costs nothing,
                    # issuing costs a credit AND still fails.
                    sleep_s = self._blocked_until - now + 0.05
                    minute_used, day_used = len(self._minute_window), len(self._day_window)
                elif (
                    len(self._minute_window) < self._per_minute
                    and len(self._day_window) < self._per_day
                ):
                    self._minute_window.append(now)
                    self._day_window.append(now)
                    self._by_purpose[str(purpose)] += 1
                    return
                elif len(self._minute_window) >= self._per_minute:
                    sleep_s = max(0.0, 60.0 - (now - self._minute_window[0])) + 0.05
                    minute_used, day_used = len(self._minute_window), len(self._day_window)
                else:
                    sleep_s = max(0.0, 86400.0 - (now - self._day_window[0])) + 0.05
                    minute_used, day_used = len(self._minute_window), len(self._day_window)

            logger.info(
                "TwelveDataRateLimiter: sleeping %.2fs (minute=%d/%d, day=%d/%d)",
                sleep_s, minute_used, self._per_minute, day_used, self._per_day,
            )
            self._sleep_fn(sleep_s)

    def sync_from_server(self, used: Optional[int], left: Optional[int]) -> None:
        """Adopt the server's own credit counter, reported on EVERY response.

        The local window can never be aligned with the server's (measured: the
        server already read ``used: 2`` on the first request of a fresh burst).
        When the server says zero credits are left, stand down for a cooldown
        rather than spend a request to be told again.
        """
        if used is None and left is None:
            return
        with self._lock:
            self._server_used = used
            self._server_left = left
            if left is not None and left <= 0:
                self._blocked_until = max(
                    self._blocked_until, self._now_fn() + self._cooldown()
                )

    def note_rate_limited(self, cooldown_s: Optional[float] = None) -> None:
        """A request was refused: hold every caller until the window resets."""
        with self._lock:
            self._blocked_until = max(
                self._blocked_until,
                self._now_fn() + (cooldown_s if cooldown_s is not None else self._cooldown()),
            )

    @staticmethod
    def _cooldown() -> float:
        try:
            value = float(os.environ.get(_COOLDOWN_ENV, DEFAULT_COOLDOWN_S))
        except (TypeError, ValueError):
            return DEFAULT_COOLDOWN_S
        return value if value > 0 else DEFAULT_COOLDOWN_S

    def snapshot(self) -> dict:
        """Current window usage — the credit counter the audit found missing.

        A pure read (it prunes expired entries first, like ``acquire``) so a
        health probe can report Twelve Data consumption without spending one.
        """
        with self._lock:
            now = self._now_fn()
            while self._minute_window and now - self._minute_window[0] >= 60.0:
                self._minute_window.popleft()
            while self._day_window and now - self._day_window[0] >= 86400.0:
                self._day_window.popleft()
            return {
                "minute_used": len(self._minute_window),
                "minute_limit": self._per_minute,
                "day_used": len(self._day_window),
                "day_limit": self._per_day,
                # what the provider itself last reported — the authority
                "server_minute_used": self._server_used,
                "server_minute_left": self._server_left,
                "cooling_down_s": max(0.0, round(self._blocked_until - now, 1)),
                "by_purpose": dict(self._by_purpose),
            }


class TwelveDataProvider(DataProvider):
    """OHLCV provider backed by the Twelve Data REST API.

    Behaviour:
      - Rate-limited from ``TWELVE_DATA_PER_MINUTE`` / ``TWELVE_DATA_PER_DAY``
        (defaults 8/min, 800/day = free tier), cross-checked against the server's
        own ``Api-Credits-Left`` counter.
      - Exponential backoff retry on network errors and 5xx (up to MAX_RETRIES).
      - Fail-fast, WITHOUT retry, on 401/403 (``TwelveDataAuthError``) and on
        429 (``TwelveDataRateLimited``) — a refused request is billed, so
        retrying it only deepens the overrun.
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
        per_minute: Optional[int] = None,
        per_day: Optional[int] = None,
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
        # A wider TTL de-duplicates repeated REST fetches of the same combo within
        # the window (env-tunable). Market data a few minutes stale is fine — the
        # freshness badge already flags any lag — and it cuts credit usage.
        try:
            env_ttl = float(os.environ.get("TWELVE_DATA_CACHE_TTL_S", "300"))
        except (TypeError, ValueError):
            env_ttl = 300.0
        self._cache_ttl_s = env_ttl if cache_ttl_s == 60.0 else cache_ttl_s
        self._cache_lock = threading.Lock()
        self._sleep_fn = sleep_fn

    @staticmethod
    def _map_symbol(symbol: str) -> str:
        mapping = _symbol_map()
        key = (symbol or "").upper()
        try:
            return mapping[key]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported symbol: {symbol!r}. Supported: {sorted(mapping)}"
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
        return list(_symbol_map().keys())

    def credit_snapshot(self) -> dict:
        """Requests consumed in the current sliding windows (PERF-3, A-9).

        The audit found no way at all to see Twelve Data consumption: no counter,
        no per-call log — the only signal was the limiter announcing a sleep, i.e.
        once the quota was already gone. This is a pure read of the limiter it
        already keeps, so /health can surface the budget before it runs out.
        """
        return self._rate_limiter.snapshot()

    def fetch_candles_until(
        self, symbol: str, timeframe: str, count: int, end_date: Optional[str] = None
    ) -> List[Candle]:
        """Up to ``count`` candles ENDING at ``end_date`` (UTC "YYYY-MM-DD HH:MM:SS";
        most recent when None). Used by the deep (paginated) backfill to walk the
        history backward one 5000-bar page at a time — bypasses the small live
        cache. Never uses more than the provider's per-request cap."""
        df = self._fetch_dataframe(symbol, timeframe, min(int(count), 5000), end_date=end_date)
        return [
            Candle(
                ts=ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
                open=float(row["Open"]), high=float(row["High"]),
                low=float(row["Low"]), close=float(row["Close"]),
                volume=float(row["Volume"]),
            )
            for ts, row in df.iterrows()
        ]

    def _fetch_dataframe(
        self, symbol: str, timeframe: str, lookback: int, end_date: Optional[str] = None
    ) -> pd.DataFrame:
        td_symbol = self._map_symbol(symbol)
        td_interval = self._map_timeframe(timeframe)

        params = {
            "symbol": td_symbol,
            "interval": td_interval,
            "outputsize": lookback,
            "apikey": self._api_key,
            "format": "JSON",
            # Without this, Twelve Data returns exchange-local timestamps
            # (observed +10h vs UTC on XAU/EUR) that _parse_time_series would
            # mislabel as UTC — audit DETECTION_QUALITY_REVIEW_2026_06_12 §T2.
            "timezone": "UTC",
        }
        if end_date:
            # Walk backward: TD returns the ``outputsize`` bars up to end_date.
            params["end_date"] = end_date
        url = f"{self.BASE_URL}/time_series"

        purpose = _purpose.get()
        backoff = self.BASE_BACKOFF_S
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            self._rate_limiter.acquire(purpose)
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

            # The server reports its own budget on EVERY response, success or
            # failure. Adopt it before deciding anything else.
            self._sync_credits(resp)

            status = resp.status_code
            if status in (401, 403):
                raise TwelveDataAuthError(
                    f"Authentication failed (HTTP {status}). Check TWELVE_DATA_API_KEY."
                )

            # DATA-3: a 429 is TERMINAL. The refused request was already billed;
            # retrying it spends more of a budget that is, by definition, gone.
            if status == 429:
                self._rate_limiter.note_rate_limited()
                raise TwelveDataRateLimited(
                    f"Rate limited by Twelve Data (HTTP 429): {resp.text[:200]}"
                )

            if 500 <= status < 600:
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
                # Twelve Data can report the overrun in the ENVELOPE rather than
                # the HTTP status; it is the same exhausted budget, not a config
                # bug, and it must not be mistaken for one.
                if code == 429:
                    self._rate_limiter.note_rate_limited()
                    raise TwelveDataRateLimited(f"API error 429: {msg}")
                raise TwelveDataError(f"API error {code}: {msg}")

            return self._parse_time_series(body)

        raise TwelveDataError(f"Exhausted retries; last error: {last_exc}")

    def _sync_credits(self, resp: object) -> None:
        """Feed the server's ``Api-Credits-*`` counters back into the limiter."""
        headers = getattr(resp, "headers", None)
        if not headers:
            return

        def _read(name: str) -> Optional[int]:
            try:
                raw = headers.get(name)
            except Exception:  # noqa: BLE001 — a header bag that misbehaves is not fatal
                return None
            if raw is None:
                return None
            try:
                return int(str(raw).strip())
            except (TypeError, ValueError):
                return None

        self._rate_limiter.sync_from_server(
            _read("Api-Credits-Used"), _read("Api-Credits-Left")
        )

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
