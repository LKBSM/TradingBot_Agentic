"""Tests for TwelveDataProvider (Chantier 1)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.intelligence.data_providers import DataProvider, TwelveDataProvider
from src.intelligence.data_providers.twelve_data_provider import (
    Candle,
    TwelveDataAuthError,
    TwelveDataError,
    TwelveDataRateLimiter,
)


# =============================================================================
# Mapping tests
# =============================================================================

class TestSymbolMapping:
    def test_xauusd_maps_to_xau_slash_usd(self):
        assert TwelveDataProvider._map_symbol("XAUUSD") == "XAU/USD"

    def test_eurusd_maps_to_eur_slash_usd(self):
        assert TwelveDataProvider._map_symbol("EURUSD") == "EUR/USD"

    def test_unsupported_symbol_raises_value_error(self):
        with pytest.raises(ValueError, match="Unsupported symbol"):
            TwelveDataProvider._map_symbol("BTCUSD")


class TestTimeframeMapping:
    @pytest.mark.parametrize("mia_tf, td_interval", [
        ("M1", "1min"),
        ("M5", "5min"),
        ("M15", "15min"),
        ("M30", "30min"),
        ("H1", "1h"),
        ("H4", "4h"),
        ("D1", "1day"),
        ("W1", "1week"),
    ])
    def test_known_timeframes_map(self, mia_tf, td_interval):
        assert TwelveDataProvider._map_timeframe(mia_tf) == td_interval

    def test_unsupported_timeframe_raises_value_error(self):
        with pytest.raises(ValueError, match="Unsupported timeframe"):
            TwelveDataProvider._map_timeframe("M3")


# =============================================================================
# Rate limiter tests (deterministic via mocked time + sleep)
# =============================================================================

class TestRateLimiter:
    def test_under_limit_never_sleeps(self):
        sleep_calls: list[float] = []
        fake_now = [0.0]
        limiter = TwelveDataRateLimiter(
            per_minute=8,
            per_day=800,
            sleep_fn=lambda s: sleep_calls.append(s),
            now_fn=lambda: fake_now[0],
        )
        for _ in range(8):
            limiter.acquire()
            fake_now[0] += 0.1
        assert sleep_calls == []

    def test_ninth_call_within_minute_blocks(self):
        """The 9th request inside a 60s window must trigger a sleep."""
        sleep_calls: list[float] = []
        fake_now = [0.0]

        def fake_sleep(seconds: float) -> None:
            sleep_calls.append(seconds)
            fake_now[0] += seconds

        limiter = TwelveDataRateLimiter(
            per_minute=8,
            per_day=800,
            sleep_fn=fake_sleep,
            now_fn=lambda: fake_now[0],
        )
        for _ in range(8):
            limiter.acquire()
            fake_now[0] += 1.0  # 8 calls spaced 1s apart → 8s elapsed

        limiter.acquire()  # 9th call, should sleep ~52s to fall outside the window
        assert sleep_calls, "9th call did not sleep"
        assert sum(sleep_calls) >= 50.0


# =============================================================================
# HTTP integration tests (mocked requests.Session)
# =============================================================================

def _make_response(status_code: int, json_body=None, text: str = "") -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.ok = 200 <= status_code < 300
    resp.json.return_value = json_body if json_body is not None else {}
    resp.text = text
    return resp


_SAMPLE_BODY_XAU_2BARS = {
    "status": "ok",
    "values": [
        {"datetime": "2026-05-29 14:00:00", "open": "2378.5", "high": "2379.0",
         "low": "2378.0", "close": "2378.7", "volume": "100"},
        {"datetime": "2026-05-29 14:15:00", "open": "2378.7", "high": "2380.0",
         "low": "2378.5", "close": "2379.5", "volume": "120"},
    ],
}


class TestProviderHttpBehaviour:
    def test_fail_fast_on_401_no_retry(self):
        session = MagicMock()
        session.get.return_value = _make_response(401, text="unauthorized")
        provider = TwelveDataProvider(
            api_key="bad_key", session=session, sleep_fn=lambda _: None,
        )
        with pytest.raises(TwelveDataAuthError):
            provider.get_ohlcv("XAUUSD", "M15", 10)
        assert session.get.call_count == 1, "401 must not trigger retry"

    def test_fail_fast_on_403_no_retry(self):
        session = MagicMock()
        session.get.return_value = _make_response(403, text="forbidden")
        provider = TwelveDataProvider(
            api_key="dummy", session=session, sleep_fn=lambda _: None,
        )
        with pytest.raises(TwelveDataAuthError):
            provider.get_ohlcv("EURUSD", "H1", 10)
        assert session.get.call_count == 1

    def test_retry_on_429_then_success(self):
        session = MagicMock()
        session.get.side_effect = [
            _make_response(429, text="rate limited"),
            _make_response(200, json_body=_SAMPLE_BODY_XAU_2BARS),
        ]
        provider = TwelveDataProvider(
            api_key="dummy", session=session, sleep_fn=lambda _: None,
        )
        df = provider.get_ohlcv("XAUUSD", "M15", 2)
        assert len(df) == 2
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert session.get.call_count == 2

    def test_retry_on_503_then_success(self):
        session = MagicMock()
        session.get.side_effect = [
            _make_response(503, text="unavailable"),
            _make_response(200, json_body=_SAMPLE_BODY_XAU_2BARS),
        ]
        provider = TwelveDataProvider(
            api_key="dummy", session=session, sleep_fn=lambda _: None,
        )
        df = provider.get_ohlcv("XAUUSD", "M15", 2)
        assert len(df) == 2
        assert session.get.call_count == 2

    def test_api_error_envelope_401_raises_auth_error(self):
        """Twelve Data may return HTTP 200 + status=error code=401 for bad keys."""
        session = MagicMock()
        session.get.return_value = _make_response(200, json_body={
            "status": "error", "code": 401, "message": "Invalid API key",
        })
        provider = TwelveDataProvider(
            api_key="bad", session=session, sleep_fn=lambda _: None,
        )
        with pytest.raises(TwelveDataAuthError):
            provider.get_ohlcv("XAUUSD", "M15", 10)

    def test_api_error_envelope_other_code_raises_generic_error(self):
        session = MagicMock()
        session.get.return_value = _make_response(200, json_body={
            "status": "error", "code": 400, "message": "bad request",
        })
        provider = TwelveDataProvider(
            api_key="dummy", session=session, sleep_fn=lambda _: None,
        )
        with pytest.raises(TwelveDataError, match="API error 400"):
            provider.get_ohlcv("XAUUSD", "M15", 10)

    def test_cache_avoids_duplicate_call(self):
        session = MagicMock()
        session.get.return_value = _make_response(200, json_body=_SAMPLE_BODY_XAU_2BARS)
        provider = TwelveDataProvider(
            api_key="dummy", session=session, sleep_fn=lambda _: None,
            cache_ttl_s=60.0,
        )
        df1 = provider.get_ohlcv("XAUUSD", "M15", 2)
        df2 = provider.get_ohlcv("XAUUSD", "M15", 2)
        assert session.get.call_count == 1
        pd.testing.assert_frame_equal(df1, df2)

    def test_fetch_candles_returns_typed_list(self):
        session = MagicMock()
        session.get.return_value = _make_response(200, json_body=_SAMPLE_BODY_XAU_2BARS)
        provider = TwelveDataProvider(
            api_key="dummy", session=session, sleep_fn=lambda _: None,
        )
        candles = provider.fetch_candles("XAUUSD", "M15", 2)
        assert len(candles) == 2
        assert all(isinstance(c, Candle) for c in candles)
        assert candles[0].close == 2378.7
        assert candles[1].volume == 120.0

    def test_request_pins_timezone_utc(self):
        """Audit 2026-06-12 §T2: without an explicit timezone=UTC, Twelve Data
        returns exchange-local timestamps that get mislabelled as UTC."""
        session = MagicMock()
        session.get.return_value = _make_response(200, json_body=_SAMPLE_BODY_XAU_2BARS)
        provider = TwelveDataProvider(
            api_key="dummy", session=session, sleep_fn=lambda _: None,
        )
        provider.get_ohlcv("XAUUSD", "M15", 2)
        params = session.get.call_args.kwargs["params"]
        assert params["timezone"] == "UTC"


# =============================================================================
# Init + interface tests
# =============================================================================

class TestProviderInit:
    def test_init_without_api_key_or_env_raises(self, monkeypatch):
        monkeypatch.delenv("TWELVE_DATA_API_KEY", raising=False)
        with pytest.raises(ValueError, match="TWELVE_DATA_API_KEY"):
            TwelveDataProvider()

    def test_init_loads_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("TWELVE_DATA_API_KEY", "env_loaded_key")
        provider = TwelveDataProvider()
        assert provider._api_key == "env_loaded_key"

    def test_init_explicit_arg_overrides_env(self, monkeypatch):
        monkeypatch.setenv("TWELVE_DATA_API_KEY", "from_env")
        provider = TwelveDataProvider(api_key="explicit")
        assert provider._api_key == "explicit"

    def test_inherits_from_dataprovider_abc(self):
        provider = TwelveDataProvider(api_key="x")
        assert isinstance(provider, DataProvider)

    def test_available_symbols_returns_xau_and_eur(self):
        provider = TwelveDataProvider(api_key="x")
        assert set(provider.available_symbols()) == {"XAUUSD", "EURUSD"}
