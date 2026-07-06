"""Shared plumbing for provider adapters.

Contract: every adapter subclasses ProviderBase and implements
    map_symbol(sym) -> provider ticker or None (None = not covered)
    _fetch_window(ticker, sym, tf, start, end) -> pd.DataFrame (UTC-indexed OHLC)

fetch() handles rate limiting, pagination, retries, timing and normalization.
No data is ever simulated: a failed provider returns status="error" with the
reason; a missing key returns status="no_key" upstream (runner-level).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd
import requests

from symbols import Sym, TF_SECONDS, TF_PANDAS

UTC = timezone.utc


@dataclass
class FetchResult:
    status: str                      # ok | not_covered | error | empty
    df: pd.DataFrame | None = None   # index: UTC DatetimeIndex; cols: open high low close
    error: str | None = None
    latency_ms: float = 0.0          # mean HTTP latency
    requests_made: int = 0
    errors_seen: int = 0             # HTTP/parse errors encountered (incl. retried)
    derived: bool = False            # True if TF was resampled from a lower native TF
    fetched_at: str = ""             # ISO timestamp of fetch completion (freshness anchor)


class RateLimiter:
    def __init__(self, min_interval_s: float):
        self.min_interval = min_interval_s
        self._last = 0.0

    def wait(self):
        delta = time.monotonic() - self._last
        if delta < self.min_interval:
            time.sleep(self.min_interval - delta)
        self._last = time.monotonic()


@dataclass
class HttpStats:
    requests: int = 0
    errors: int = 0
    latencies: list = field(default_factory=list)


class ProviderBase:
    name = "base"
    env_key: str | None = None       # env var holding the API key (None = keyless)
    min_interval_s = 1.0             # seconds between requests (free-tier friendly)
    max_bars_per_request = 5000
    native_tfs: dict[str, str] = {}  # canonical TF -> provider interval string

    def __init__(self, api_key: str | None):
        self.api_key = api_key
        self.limiter = RateLimiter(self.min_interval_s)
        self.stats = HttpStats()

    # ---- to implement -----------------------------------------------------
    def map_symbol(self, sym: Sym) -> str | None:
        raise NotImplementedError

    def _fetch_window(self, ticker: str, sym: Sym, tf: str,
                      start: datetime, end: datetime) -> pd.DataFrame:
        """Fetch one window at the provider's NATIVE interval for tf.
        Must raise NotCovered / ProviderError on failure."""
        raise NotImplementedError

    # ---- shared machinery --------------------------------------------------
    def get_json(self, url: str, params: dict | None = None,
                 headers: dict | None = None, timeout: int = 30):
        """Rate-limited GET with 2 retries and 429 backoff. Returns parsed JSON."""
        last_exc: Exception | None = None
        for attempt in range(3):
            self.limiter.wait()
            t0 = time.monotonic()
            try:
                r = requests.get(url, params=params, headers=headers, timeout=timeout)
                self.stats.requests += 1
                self.stats.latencies.append((time.monotonic() - t0) * 1000)
                if r.status_code == 429:
                    self.stats.errors += 1
                    time.sleep(min(60, 5 * (attempt + 1) * 4))
                    last_exc = ProviderError("HTTP 429 rate-limited")
                    continue
                if r.status_code in (401, 403):
                    raise ProviderError(f"HTTP {r.status_code} auth/plan refused: {r.text[:200]}")
                if r.status_code == 404:
                    raise NotCovered(f"HTTP 404: {r.text[:120]}")
                if r.status_code >= 400:
                    self.stats.errors += 1
                    last_exc = ProviderError(f"HTTP {r.status_code}: {r.text[:200]}")
                    time.sleep(2 * (attempt + 1))
                    continue
                return r.json()
            except (requests.Timeout, requests.ConnectionError) as exc:
                self.stats.requests += 1
                self.stats.errors += 1
                last_exc = ProviderError(f"network: {exc}")
                time.sleep(2 * (attempt + 1))
        raise last_exc or ProviderError("unknown fetch failure")

    def fetch(self, sym: Sym, tf: str, start: datetime, end: datetime) -> FetchResult:
        ticker = self.map_symbol(sym)
        if ticker is None or tf not in self.effective_tfs():
            return FetchResult(status="not_covered",
                               fetched_at=datetime.now(UTC).isoformat())
        req0, err0 = self.stats.requests, self.stats.errors
        try:
            native_tf, derived = self.resolve_tf(tf)
            frames = []
            for w_start, w_end in self.windows(native_tf, start, end):
                frames.append(self._fetch_window(ticker, sym, native_tf, w_start, w_end))
            df = normalize(pd.concat(frames) if frames else pd.DataFrame())
            if derived and not df.empty:
                df = resample_ohlc(df, tf)
            df = df[(df.index >= pd.Timestamp(start)) & (df.index < pd.Timestamp(end))]
            # exclure la bougie partielle en cours (ouverte il y a < 1 TF)
            cutoff = pd.Timestamp(datetime.now(UTC)) - pd.Timedelta(seconds=TF_SECONDS[tf])
            df = df[df.index <= cutoff]
            status = "ok" if not df.empty else "empty"
            return self._result(status, df=df if not df.empty else None,
                                derived=derived, req0=req0, err0=err0)
        except NotCovered as exc:
            return self._result("not_covered", error=str(exc), req0=req0, err0=err0)
        except ProviderError as exc:
            return self._result("error", error=str(exc), req0=req0, err0=err0)
        except Exception as exc:  # parse bugs etc. — recorded, never fatal
            return self._result("error", error=f"{type(exc).__name__}: {exc}",
                                req0=req0, err0=err0)

    def _result(self, status, df=None, error=None, derived=False, req0=0, err0=0):
        lats = self.stats.latencies
        return FetchResult(
            status=status, df=df, error=error, derived=derived,
            latency_ms=sum(lats) / len(lats) if lats else 0.0,
            requests_made=self.stats.requests - req0,
            errors_seen=self.stats.errors - err0,
            fetched_at=datetime.now(UTC).isoformat(),
        )

    def effective_tfs(self):
        return self.native_tfs

    def resolve_tf(self, tf: str) -> tuple[str, bool]:
        """Return (canonical TF to fetch natively, derived?). Adapters that
        resample declare it by mapping tf -> a LOWER canonical tf in native_tfs
        via the DERIVED_FROM marker."""
        target = self.native_tfs[tf]
        if target.startswith("derive:"):
            return target.split(":", 1)[1], True
        return tf, False

    def windows(self, tf: str, start: datetime, end: datetime):
        """Split [start, end) so each window stays under max_bars_per_request."""
        step = TF_SECONDS[tf] * self.max_bars_per_request
        cur = start
        while cur < end:
            nxt = min(end, cur + pd.Timedelta(seconds=step))
            yield cur, nxt.to_pydatetime() if hasattr(nxt, "to_pydatetime") else nxt
            cur = nxt


class ProviderError(Exception):
    pass


class NotCovered(Exception):
    pass


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Sort, dedupe, coerce numeric, force tz-aware UTC index."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close"])
    df = df[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
    if df.index.tz is None:
        df.index = df.index.tz_localize(UTC)
    else:
        df.index = df.index.tz_convert(UTC)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df.dropna(how="all")


def resample_ohlc(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    rule = TF_PANDAS[tf]
    out = df.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"})
    return out.dropna(how="all")


def parse_epoch_index(rows, ts_key, unit="s"):
    idx = pd.to_datetime([r[ts_key] for r in rows], unit=unit, utc=True)
    return idx
