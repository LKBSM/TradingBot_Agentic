"""Tests for the Twelve Data live-tick WS bridge (prototype, dev/free tier).

The network is never touched: message handling is tested directly, and the
connect/subscribe/consume lifecycle is driven with an in-memory fake WebSocket.
"""

from __future__ import annotations

import json

import pytest

from src.intelligence.data_providers.twelve_data_ws import (
    LiveTick,
    TwelveDataLiveTickBridge,
)


def _bridge(**kwargs) -> TwelveDataLiveTickBridge:
    return TwelveDataLiveTickBridge(api_key="k", instruments=["XAUUSD", "EURUSD"], **kwargs)


# ─── Construction / validation ────────────────────────────────────────────────


def test_requires_api_key():
    with pytest.raises(ValueError):
        TwelveDataLiveTickBridge(api_key="")


def test_rejects_unknown_instrument():
    with pytest.raises(ValueError):
        TwelveDataLiveTickBridge(api_key="k", instruments=["BTCUSD"])


# ─── Message handling (pure) ──────────────────────────────────────────────────


def test_price_frame_updates_latest_with_app_instrument():
    b = _bridge()
    b._handle_message(
        json.dumps({"event": "price", "symbol": "XAU/USD", "price": 4317.5, "timestamp": 1781750760})
    )
    tick = b.get_latest("XAUUSD")
    assert isinstance(tick, LiveTick)
    assert tick.instrument == "XAUUSD"
    assert tick.price == 4317.5
    assert tick.ts == 1781750760
    # EURUSD untouched.
    assert b.get_latest("EURUSD") is None


def test_latest_keeps_the_most_recent_tick():
    b = _bridge()
    b._handle_message(json.dumps({"event": "price", "symbol": "EUR/USD", "price": 1.152, "timestamp": 1}))
    b._handle_message(json.dumps({"event": "price", "symbol": "EUR/USD", "price": 1.153, "timestamp": 2}))
    assert b.get_latest("EURUSD").price == 1.153


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        json.dumps({"event": "subscribe-status", "status": "ok"}),
        json.dumps({"event": "heartbeat"}),
        json.dumps({"event": "price", "symbol": "BTC/USD", "price": 1.0}),  # unknown symbol
        json.dumps({"event": "price", "symbol": "XAU/USD"}),  # missing price
        json.dumps({"event": "price", "symbol": "XAU/USD", "price": "nan-ish"}),
        json.dumps({"event": "price", "symbol": "XAU/USD", "price": -5.0}),  # non-positive
        json.dumps([1, 2, 3]),  # not a dict
    ],
)
def test_bad_frames_are_dropped_not_raised(raw):
    b = _bridge()
    b._handle_message(raw)  # must not raise
    assert b.get_latest("XAUUSD") is None


def test_price_without_timestamp_defaults_ts_zero():
    b = _bridge()
    b._handle_message(json.dumps({"event": "price", "symbol": "XAU/USD", "price": 1.0}))
    assert b.get_latest("XAUUSD").ts == 0


# ─── Lifecycle with a fake WebSocket (no network) ─────────────────────────────


class _FakeWS:
    """Async-iterable fake websocket yielding a fixed list of frames once.

    ``on_drain`` fires AFTER the last frame is yielded — used to stop the bridge
    once every frame has been consumed (so the consume loop handles them all
    before the outer reconnect loop exits).
    """

    def __init__(self, frames, sent, on_drain=None):
        self._frames = frames
        self._sent = sent
        self._on_drain = on_drain

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def send(self, msg):
        self._sent.append(msg)

    def __aiter__(self):
        async def gen():
            for f in self._frames:
                yield f
            if self._on_drain:
                self._on_drain()

        return gen()


def test_run_async_subscribes_and_consumes():
    sent: list[str] = []
    frames = [
        json.dumps({"event": "subscribe-status", "status": "ok"}),
        json.dumps({"event": "price", "symbol": "XAU/USD", "price": 4000.0, "timestamp": 10}),
        json.dumps({"event": "price", "symbol": "EUR/USD", "price": 1.1, "timestamp": 11}),
    ]

    b = _bridge()

    def fake_connect(url, **kwargs):
        # Stop only AFTER the frames drain, so all of them are handled and the
        # subscribe message is sent exactly once.
        return _FakeWS(frames, sent, on_drain=b._stop_event.set)

    b._connect = fake_connect
    import asyncio

    asyncio.run(b._run_async())

    # Subscribed to both mapped symbols.
    assert len(sent) == 1
    params = json.loads(sent[0])
    assert params["action"] == "subscribe"
    assert set(params["params"]["symbols"].split(",")) == {"XAU/USD", "EUR/USD"}
    # Both ticks landed.
    assert b.get_latest("XAUUSD").price == 4000.0
    assert b.get_latest("EURUSD").price == 1.1


def test_start_stop_is_idempotent_and_safe():
    # No connect → the loop will fail fast and back off; we just want start/stop
    # to be safe to call without raising, and stop() to be idempotent.
    def fake_connect(url, **kwargs):
        raise RuntimeError("boom")

    b = _bridge(connect=fake_connect, base_backoff_s=0.01, max_backoff_s=0.01)
    b.start()
    assert b.running is True
    b.stop()
    b.stop()  # idempotent
    assert b.running is False
