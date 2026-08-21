"""MKT-1 — the single market registry.

Pins the registry contract and the invariants that make "add a market = one JSON
entry" safe: the supported-instrument perimeter derives from the registry, every
active market has a forecasting preset, and every market timeframe is a known
timeframe. The news-attachment rule stays in event_market_map.json (referenced,
not duplicated).
"""

from __future__ import annotations

import pytest

from src.intelligence import market_registry as mr
from src.intelligence import timeframe_registry as tr


@pytest.fixture(autouse=True)
def _reset():
    mr.reset_cache()
    tr.reset_cache()
    yield
    mr.reset_cache()
    tr.reset_cache()


def test_v1_perimeter():
    assert mr.all_ids() == ("XAUUSD", "EURUSD")


def test_specs_carry_identity_and_precision():
    gold = mr.spec("xauusd")  # case-insensitive
    assert gold.id == "XAUUSD"
    assert gold.label == "Or (XAU/USD)"
    assert gold.type == "metal"
    assert gold.price_decimals == 2
    eur = mr.spec("EURUSD")
    assert eur.type == "fx"
    assert eur.price_decimals == 5


def test_unknown_market_raises():
    assert not mr.has("BTCUSD")
    with pytest.raises(KeyError):
        mr.spec("BTCUSD")


def test_supported_instruments_derives_from_registry():
    """The LB-1 perimeter is the registry — no second source of truth."""
    from src.intelligence import lookback_config as lb

    lb.reset_cache()
    assert lb.supported_instruments() == mr.all_ids()


def test_every_active_market_has_a_forecasting_preset():
    """A market in the registry must have a volatility preset — otherwise a
    supported market would have no forecasting config (registry ⊆ presets)."""
    from src.intelligence.volatility_forecaster import get_instrument_registry

    presets = set(get_instrument_registry().keys())
    assert set(mr.all_ids()) <= presets


def test_every_market_timeframe_is_known():
    known = set(tr.all_ids())
    for m in mr.all_specs():
        assert set(m.timeframes) <= known, f"{m.id} has an unknown timeframe"


def test_driver_currencies_reference_event_market_map():
    """driver_currencies are READ from event_market_map.json, not stored here."""
    assert mr.driver_currencies("XAUUSD") == ("USD",)
    assert set(mr.driver_currencies("EURUSD")) == {"USD", "EUR"}
