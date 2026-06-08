"""Property-based tests (Hypothesis) PR-ready pour Smart Sentinel AI.

Six specs couvrant les 3 sous-systemes les plus critiques :
  * ConfluenceDetector  -> score borne [0,100], renormalisation correcte
  * SignalStateMachine  -> contrat HOLD/BUY/SELL inviolable (lockout, cooldown)
  * resample_ohlcv       -> invariants OHLC + conservation volume

Pour activer :
    pip install hypothesis pytest
    python -m pytest tests/test_property_based.py -v --hypothesis-show-statistics

Notes execution :
  * deadline=2000 ms (vol_forecaster lent sur grands DataFrames)
  * max_examples=50 sur SignalStateMachine (chaines longues couteuses)
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, assume, given, settings, strategies as st

from src.intelligence.confluence_detector import (
    ConfluenceDetector,
    DEFAULT_WEIGHTS,
    SignalTier,
)
from src.intelligence.signal_state_machine import (
    Direction,
    PublicState,
    SignalStateMachine,
    StateMachineConfig,
)
from src.intelligence.volatility_forecaster import resample_ohlcv


# =============================================================================
# Strategies partagees
# =============================================================================

# Prix realistes XAU (1500-3000) ; on evite NaN/inf et 0
positive_price = st.floats(min_value=100.0, max_value=10000.0,
                           allow_nan=False, allow_infinity=False)
positive_atr = st.floats(min_value=0.01, max_value=500.0,
                         allow_nan=False, allow_infinity=False)
score_0_100 = st.floats(min_value=0.0, max_value=100.0,
                        allow_nan=False, allow_infinity=False)
small_pos = st.floats(min_value=0.0, max_value=1.0,
                      allow_nan=False, allow_infinity=False)


def _smc_features_strategy():
    """Generate plausible SMC feature dicts (BOS / FVG / OB / RSI / MACD)."""
    return st.fixed_dictionaries({
        "BOS_SIGNAL": st.sampled_from([-1, 0, 1]),
        "FVG_SIGNAL": st.sampled_from([-1, 0, 1]),
        "OB_STRENGTH_NORM": small_pos,
        "RSI": st.floats(min_value=0.0, max_value=100.0,
                         allow_nan=False, allow_infinity=False),
        "MACD_Diff": st.floats(min_value=-50.0, max_value=50.0,
                               allow_nan=False, allow_infinity=False),
    })


# =============================================================================
# SPEC 1 - ConfluenceDetector : score toujours dans [0, 100]
# =============================================================================

@settings(max_examples=200, deadline=2000,
          suppress_health_check=[HealthCheck.too_slow])
@given(
    smc=_smc_features_strategy(),
    price=positive_price,
    atr=positive_atr,
    regime_strength=st.floats(min_value=0.0, max_value=1.0,
                              allow_nan=False),
)
def test_confluence_score_within_bounds(smc, price, atr, regime_strength):
    """Quel que soit l'input, score in [0, 100] et tier coherent."""
    detector = ConfluenceDetector(min_score=0.0, require_retest=False)

    # Mock regime + news avec attributs minimaux
    regime = type("R", (), {
        "regime": "TRENDING_UP",
        "strength": regime_strength,
        "value": "TRENDING_UP",
    })()
    news = type("N", (), {
        "blackout_active": False,
        "high_impact_count": 0,
        "sentiment_score": 0.0,
    })()

    signal = detector.analyze(smc, regime, news, price, atr)

    if signal is None:
        return  # Aucun signal genere, OK

    # INVARIANT 1 : borne dure du score
    assert 0.0 <= signal.confluence_score <= 100.0, (
        f"Score hors bornes: {signal.confluence_score}"
    )
    # INVARIANT 2 : tier coherent avec score
    if signal.confluence_score >= 80:
        assert signal.tier == SignalTier.PREMIUM
    elif signal.confluence_score >= 60:
        assert signal.tier == SignalTier.STANDARD
    elif signal.confluence_score >= 40:
        assert signal.tier == SignalTier.WEAK
    else:
        assert signal.tier == SignalTier.INVALID
    # INVARIANT 3 : SL/TP coherents avec direction
    if signal.signal_type.value == "LONG":
        assert signal.stop_loss < signal.entry_price < signal.take_profit
    else:
        assert signal.take_profit < signal.entry_price < signal.stop_loss
    # INVARIANT 4 : RR ratio strictement positif
    assert signal.rr_ratio > 0


# =============================================================================
# SPEC 2 - ConfluenceDetector : monotonie sur le score regime
# =============================================================================

@settings(max_examples=100, deadline=2000)
@given(
    smc=_smc_features_strategy(),
    price=positive_price,
    atr=positive_atr,
    s_low=st.floats(min_value=0.0, max_value=0.4),
    s_delta=st.floats(min_value=0.05, max_value=0.5),
)
def test_confluence_regime_monotonic(smc, price, atr, s_low, s_delta):
    """Augmenter regime.strength ne doit pas DIMINUER le score, toutes choses egales par ailleurs."""
    detector = ConfluenceDetector(min_score=0.0, require_retest=False)
    s_high = min(1.0, s_low + s_delta)
    assume(s_high > s_low)

    news = type("N", (), {
        "blackout_active": False, "high_impact_count": 0, "sentiment_score": 0.0,
    })()

    def _signal_with(strength):
        regime = type("R", (), {
            "regime": "TRENDING_UP", "strength": strength, "value": "TRENDING_UP",
        })()
        return detector.analyze(smc, regime, news, price, atr)

    sig_low = _signal_with(s_low)
    sig_high = _signal_with(s_high)

    if sig_low is None or sig_high is None:
        return  # Comparaison impossible
    # Memes inputs SMC + meme direction => score regime monotone croissant
    if sig_low.signal_type == sig_high.signal_type:
        assert sig_high.confluence_score >= sig_low.confluence_score - 1e-6, (
            f"Score chute en augmentant regime.strength: "
            f"{sig_low.confluence_score} -> {sig_high.confluence_score}"
        )


# =============================================================================
# SPEC 3 - SignalStateMachine : pas de transition BUY -> SELL sans HOLD
# =============================================================================

# Sequence (score, direction) realiste pour la machine
_directions = st.sampled_from([Direction.LONG, Direction.SHORT])
_score_event = st.tuples(score_0_100, _directions)


@settings(max_examples=50, deadline=3000,
          suppress_health_check=[HealthCheck.too_slow])
@given(events=st.lists(_score_event, min_size=10, max_size=200))
def test_state_machine_no_direct_buy_sell_flip(events):
    """Apres un BUY, on ne peut JAMAIS atteindre SELL au bar suivant.

    Invariant business critique : le client n'accepte aucun flip direct
    (cf. confiance + lockout, signal_state_machine.md regle 5).
    """
    cfg = StateMachineConfig(
        enter_threshold=75.0, exit_threshold=55.0,
        confirm_bars=2, cooldown_bars=2,
    )
    sm = SignalStateMachine(symbol="XAUUSD", config=cfg)

    base_ts = datetime(2025, 1, 1)
    prev_state = PublicState.HOLD
    for i, (score, direction) in enumerate(events):
        ts = (base_ts + timedelta(minutes=15 * i)).isoformat()
        # On feed un signal synthese minimal
        sm.process_bar(
            timestamp=ts,
            confluence_score=score,
            direction=direction,
            entry_price=2000.0, stop_loss=1990.0, take_profit=2020.0,
            current_price=2000.0,
        )
        snapshot = sm.get_snapshot()
        cur = PublicState(snapshot["public_state"])
        # INVARIANT : aucun flip direct BUY -> SELL ou SELL -> BUY
        if prev_state == PublicState.BUY:
            assert cur != PublicState.SELL, (
                f"Flip illegal BUY->SELL detecte au bar {i} (score={score})"
            )
        if prev_state == PublicState.SELL:
            assert cur != PublicState.BUY, (
                f"Flip illegal SELL->BUY detecte au bar {i} (score={score})"
            )
        prev_state = cur


# =============================================================================
# SPEC 4 - SignalStateMachine : confirm_bars respecte
# =============================================================================

@settings(max_examples=30, deadline=2000)
@given(
    confirm_bars=st.integers(min_value=1, max_value=5),
    extra=st.integers(min_value=0, max_value=10),
)
def test_state_machine_confirm_bars_respected(confirm_bars, extra):
    """Le passage HOLD -> BUY exige >= confirm_bars consecutifs au-dessus du seuil.

    Si on injecte (confirm_bars - 1) bars > 75 puis 1 bar < 75, on doit RESTER en HOLD.
    """
    cfg = StateMachineConfig(
        enter_threshold=75.0, exit_threshold=55.0,
        confirm_bars=confirm_bars, cooldown_bars=2,
    )
    sm = SignalStateMachine(symbol="XAUUSD", config=cfg)
    base_ts = datetime(2025, 1, 1)

    # Phase 1 : (confirm_bars - 1) bars armants
    for i in range(confirm_bars - 1):
        sm.process_bar(
            timestamp=(base_ts + timedelta(minutes=15 * i)).isoformat(),
            confluence_score=85.0,
            direction=Direction.LONG,
            entry_price=2000.0, stop_loss=1990.0, take_profit=2020.0,
            current_price=2000.0,
        )
        snap = sm.get_snapshot()
        assert snap["public_state"] == PublicState.HOLD.value, (
            f"BUY emis trop tot apres seulement {i + 1} bars (besoin {confirm_bars})"
        )

    # Phase 2 : 1 bar en-dessous casse la confirmation
    sm.process_bar(
        timestamp=(base_ts + timedelta(minutes=15 * confirm_bars)).isoformat(),
        confluence_score=50.0,
        direction=Direction.LONG,
        entry_price=2000.0, stop_loss=1990.0, take_profit=2020.0,
        current_price=2000.0,
    )
    snap = sm.get_snapshot()
    assert snap["public_state"] == PublicState.HOLD.value, (
        "Confirmation devrait etre cassee par le bar bas"
    )


# =============================================================================
# SPEC 5 - resample_ohlcv : invariants OHLC + conservation volume
# =============================================================================

@st.composite
def _ohlcv_minutes_strategy(draw):
    """Genere un DataFrame OHLCV M1 sur 4-12 heures avec OHLC bien forme."""
    n_bars = draw(st.integers(min_value=120, max_value=720))  # 2h a 12h en M1
    base_price = draw(st.floats(min_value=1500.0, max_value=3000.0,
                                 allow_nan=False, allow_infinity=False))
    # Genere closes via random walk borne
    rng = np.random.default_rng(draw(st.integers(0, 10**6)))
    closes = base_price + np.cumsum(rng.normal(0, 0.5, size=n_bars))
    closes = np.clip(closes, 100.0, 1e5)
    opens = np.concatenate([[base_price], closes[:-1]])
    spreads = np.abs(rng.normal(0, 0.3, size=n_bars))
    highs = np.maximum(opens, closes) + spreads
    lows = np.minimum(opens, closes) - spreads
    volumes = rng.integers(50, 500, size=n_bars).astype(float)

    timestamps = pd.date_range("2025-01-01", periods=n_bars, freq="1min")
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })


@settings(max_examples=50, deadline=3000)
@given(df=_ohlcv_minutes_strategy())
def test_resample_ohlcv_invariants(df):
    """Resample M1 -> M15 doit preserver les invariants OHLC + sum(volume).

    INVARIANTS :
      I1. high >= max(open, close) pour chaque bar resamplee
      I2. low  <= min(open, close)
      I3. low  <= high
      I4. sum(volume_M15) == sum(volume_M1) (conservation)
      I5. nb_bars_resampled <= ceil(nb_bars_source / 15)
    """
    src_total_volume = float(df["volume"].sum())
    n_src = len(df)

    out = resample_ohlcv(df, source_tf="M1", target_tf="M15")
    assume(len(out) > 0)

    # I1
    assert (out["high"] >= out[["open", "close"]].max(axis=1) - 1e-9).all(), (
        "high < max(open, close) detecte"
    )
    # I2
    assert (out["low"] <= out[["open", "close"]].min(axis=1) + 1e-9).all(), (
        "low > min(open, close) detecte"
    )
    # I3
    assert (out["low"] <= out["high"] + 1e-9).all(), "low > high"
    # I4 : conservation du volume (tolerance flottante)
    out_total_volume = float(out["volume"].sum())
    assert math.isclose(src_total_volume, out_total_volume, rel_tol=1e-6), (
        f"Volume non conserve : src={src_total_volume} vs out={out_total_volume}"
    )
    # I5
    assert len(out) <= math.ceil(n_src / 15) + 1, (
        f"Trop de bars resamplees : {len(out)} > ceil({n_src}/15)"
    )


# =============================================================================
# SPEC 6 - resample_ohlcv : refuse upsampling
# =============================================================================

@settings(max_examples=20, deadline=1000)
@given(df=_ohlcv_minutes_strategy())
def test_resample_ohlcv_rejects_upsampling(df):
    """Tenter de resampler M15 -> M1 doit lever ValueError (pas de upsampling)."""
    # On simule un input M15 en re-tagguant le timestep
    df_m15 = df.iloc[::15].copy().reset_index(drop=True)
    df_m15["timestamp"] = pd.date_range("2025-01-01", periods=len(df_m15), freq="15min")
    assume(len(df_m15) >= 2)
    with pytest.raises(ValueError, match="(?i)cannot upsample"):
        resample_ohlcv(df_m15, source_tf="M15", target_tf="M1")
