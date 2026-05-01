"""50 evaluation fixtures for LLM narrative quality scoring.

Sprint LLM-1.1 (Aisha). Generates fixtures programmatically with parametric
variation across direction, conviction, regime, and price levels — much
more maintainable than 50 hand-typed narratives.

Distribution:
- 15 BUY high conviction
- 15 SELL high conviction
- 10 HOLD (low conviction OR contradictory signals)
- 5 high-vol regime
- 5 with news events nearby

Each fixture has:
- `id`         : unique identifier
- `category`   : grouping label
- `input`      : payload that mimics ConfluenceSignal-shape input to the LLM
- `expected`   : ground-truth annotations the eval scorers test against
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


# ---------------------------------------------------------------------------
# Forbidden phrase taxonomy (compliance MiFID II 2024/2811)
# ---------------------------------------------------------------------------

FORBIDDEN_PHRASES_FR = [
    "achetez",
    "vendez",
    "100% sûr",
    "garanti",
    "garantie de gain",
    "résultats garantis",
    "sans risque",
]

FORBIDDEN_PHRASES_EN = [
    "buy now",
    "sell now",
    "100% sure",
    "guaranteed",
    "guaranteed profit",
    "no risk",
    "risk-free",
]

ALL_FORBIDDEN_PHRASES = FORBIDDEN_PHRASES_FR + FORBIDDEN_PHRASES_EN


# ---------------------------------------------------------------------------
# Fixture dataclass
# ---------------------------------------------------------------------------


@dataclass
class FixtureExpected:
    """Ground truth annotations for a single fixture."""

    must_mention_direction: str = ""  # "bullish" / "bearish" / "neutral"
    must_cite_entry: float | None = None
    must_cite_stop: float | None = None
    must_cite_target: float | None = None
    must_mention_components: list[str] = field(default_factory=list)
    max_chars: int = 400
    min_grade_level: float = 8.0
    max_grade_level: float = 18.0
    forbidden_phrases: list[str] = field(default_factory=lambda: list(ALL_FORBIDDEN_PHRASES))


@dataclass
class Fixture:
    id: str
    category: str
    input: dict[str, Any]
    expected: FixtureExpected

    def as_dict(self) -> dict:
        d = asdict(self)
        return d


# ---------------------------------------------------------------------------
# Fixture factory
# ---------------------------------------------------------------------------


def _make_buy_signal(
    idx: int,
    base_price: float,
    conviction: int,
    regime: str = "strong_uptrend",
    vol_regime: str = "normal",
    components: list[str] | None = None,
) -> Fixture:
    components = components or ["BOS", "OB", "FVG"]
    sl = round(base_price - 10.0, 2)
    tp1 = round(base_price + 20.0, 2)
    tp2 = round(base_price + 40.0, 2)
    return Fixture(
        id=f"buy_high_{idx:03d}",
        category="BUY high conviction",
        input={
            "symbol": "XAUUSD",
            "direction": "BUY",
            "conviction": conviction,
            "entry": base_price,
            "stop_loss": sl,
            "target_1": tp1,
            "target_2": tp2,
            "regime": regime,
            "vol_regime": vol_regime,
            "components_fired": components,
            "atr_14": 5.0 if vol_regime == "normal" else 8.0,
        },
        expected=FixtureExpected(
            must_mention_direction="bullish",
            must_cite_entry=base_price,
            must_cite_stop=sl,
            must_cite_target=tp1,
            must_mention_components=components,
        ),
    )


def _make_sell_signal(
    idx: int,
    base_price: float,
    conviction: int,
    regime: str = "strong_downtrend",
    vol_regime: str = "normal",
    components: list[str] | None = None,
) -> Fixture:
    components = components or ["BOS", "OB", "FVG"]
    sl = round(base_price + 10.0, 2)
    tp1 = round(base_price - 20.0, 2)
    tp2 = round(base_price - 40.0, 2)
    return Fixture(
        id=f"sell_high_{idx:03d}",
        category="SELL high conviction",
        input={
            "symbol": "XAUUSD",
            "direction": "SELL",
            "conviction": conviction,
            "entry": base_price,
            "stop_loss": sl,
            "target_1": tp1,
            "target_2": tp2,
            "regime": regime,
            "vol_regime": vol_regime,
            "components_fired": components,
            "atr_14": 5.0 if vol_regime == "normal" else 8.0,
        },
        expected=FixtureExpected(
            must_mention_direction="bearish",
            must_cite_entry=base_price,
            must_cite_stop=sl,
            must_cite_target=tp1,
            must_mention_components=components,
        ),
    )


def _make_hold_signal(idx: int, base_price: float, conviction: int = 35) -> Fixture:
    return Fixture(
        id=f"hold_{idx:03d}",
        category="HOLD",
        input={
            "symbol": "XAUUSD",
            "direction": "HOLD",
            "conviction": conviction,
            "entry": base_price,
            "stop_loss": None,
            "target_1": None,
            "target_2": None,
            "regime": "ranging",
            "vol_regime": "normal",
            "components_fired": ["FVG"] if idx % 2 == 0 else [],
            "atr_14": 4.0,
        },
        expected=FixtureExpected(
            must_mention_direction="neutral",
            # HOLD setups don't have entry/SL/TP — narrative should NOT cite levels
            must_cite_entry=None,
            must_cite_stop=None,
            must_cite_target=None,
            must_mention_components=[],
            max_chars=300,
        ),
    )


def _make_high_vol_signal(idx: int, base_price: float, direction: str = "BUY") -> Fixture:
    sign = 1 if direction == "BUY" else -1
    sl = round(base_price - sign * 15.0, 2)  # wider stop in high vol
    tp1 = round(base_price + sign * 30.0, 2)
    return Fixture(
        id=f"highvol_{direction.lower()}_{idx:03d}",
        category="High volatility regime",
        input={
            "symbol": "XAUUSD",
            "direction": direction,
            "conviction": 75,
            "entry": base_price,
            "stop_loss": sl,
            "target_1": tp1,
            "target_2": None,
            "regime": "transition",
            "vol_regime": "high",  # ← high vol
            "components_fired": ["BOS", "OB"],
            "atr_14": 12.0,
            "vol_forecast": 15.0,
            "vol_naive": 8.0,
        },
        expected=FixtureExpected(
            must_mention_direction="bullish" if direction == "BUY" else "bearish",
            must_cite_entry=base_price,
            must_cite_stop=sl,
            must_cite_target=tp1,
            must_mention_components=["BOS", "OB"],
            max_chars=450,  # high-vol narratives slightly longer (extra context)
        ),
    )


def _make_news_event_signal(idx: int, base_price: float, direction: str = "BUY") -> Fixture:
    sign = 1 if direction == "BUY" else -1
    sl = round(base_price - sign * 10.0, 2)
    tp1 = round(base_price + sign * 20.0, 2)
    events = ["FOMC", "NFP", "CPI", "Powell speech", "Retail Sales"]
    event = events[idx % len(events)]
    return Fixture(
        id=f"news_{direction.lower()}_{idx:03d}",
        category="News event nearby",
        input={
            "symbol": "XAUUSD",
            "direction": direction,
            "conviction": 65,
            "entry": base_price,
            "stop_loss": sl,
            "target_1": tp1,
            "target_2": None,
            "regime": "weak_uptrend" if direction == "BUY" else "weak_downtrend",
            "vol_regime": "normal",
            "components_fired": ["BOS", "FVG"],
            "atr_14": 6.0,
            "news_event": event,
            "news_decision": "blackout",  # signal blocked by news proximity
            "minutes_to_event": 12,
        },
        expected=FixtureExpected(
            must_mention_direction="bullish" if direction == "BUY" else "bearish",
            must_cite_entry=base_price,
            must_cite_stop=sl,
            must_cite_target=tp1,
            must_mention_components=["BOS", "FVG"],
            max_chars=500,  # news context allows extra explanation
        ),
    )


# ---------------------------------------------------------------------------
# Top-level fixture set assembly (50 fixtures total)
# ---------------------------------------------------------------------------


def build_fixtures() -> list[Fixture]:
    """Generate the full 50-fixture set in deterministic order."""
    fixtures: list[Fixture] = []

    # 15 BUY high conviction — variations across price, regime, vol, components
    base_prices_buy = [2050, 2100, 2150, 2200, 2250, 2280, 2310, 2340, 2370, 2400,
                      2430, 2460, 2490, 2520, 2550]
    convictions_buy = [70, 75, 78, 80, 82, 85, 87, 88, 90, 92, 75, 78, 80, 85, 90]
    components_variants = [
        ["BOS", "OB", "FVG"],
        ["BOS", "OB"],
        ["BOS", "FVG"],
        ["OB", "FVG"],
        ["BOS", "OB", "FVG", "Liquidity_Sweep"],
    ]
    for i, (price, conv) in enumerate(zip(base_prices_buy, convictions_buy)):
        fixtures.append(
            _make_buy_signal(
                idx=i + 1,
                base_price=float(price),
                conviction=conv,
                regime="strong_uptrend" if conv >= 80 else "weak_uptrend",
                components=components_variants[i % len(components_variants)],
            )
        )

    # 15 SELL high conviction — symmetric
    base_prices_sell = [2080, 2130, 2180, 2230, 2270, 2300, 2330, 2360, 2390, 2420,
                       2450, 2480, 2510, 2540, 2570]
    convictions_sell = [72, 76, 79, 82, 84, 86, 88, 89, 91, 93, 76, 79, 82, 86, 91]
    for i, (price, conv) in enumerate(zip(base_prices_sell, convictions_sell)):
        fixtures.append(
            _make_sell_signal(
                idx=i + 1,
                base_price=float(price),
                conviction=conv,
                regime="strong_downtrend" if conv >= 80 else "weak_downtrend",
                components=components_variants[i % len(components_variants)],
            )
        )

    # 10 HOLD — varied conviction, with/without partial components
    hold_prices = [2150, 2180, 2210, 2240, 2270, 2300, 2330, 2360, 2390, 2420]
    hold_convictions = [25, 30, 35, 40, 45, 38, 32, 28, 42, 36]
    for i, (price, conv) in enumerate(zip(hold_prices, hold_convictions)):
        fixtures.append(_make_hold_signal(idx=i + 1, base_price=float(price), conviction=conv))

    # 5 high-vol — alternating BUY/SELL
    highvol_prices = [2200, 2300, 2400, 2150, 2350]
    for i, price in enumerate(highvol_prices):
        direction = "BUY" if i % 2 == 0 else "SELL"
        fixtures.append(_make_high_vol_signal(idx=i + 1, base_price=float(price), direction=direction))

    # 5 news events — varied, alternating direction
    news_prices = [2250, 2350, 2150, 2450, 2300]
    for i, price in enumerate(news_prices):
        direction = "BUY" if i % 2 == 0 else "SELL"
        fixtures.append(
            _make_news_event_signal(idx=i + 1, base_price=float(price), direction=direction)
        )

    assert len(fixtures) == 50, f"expected 50 fixtures, got {len(fixtures)}"
    return fixtures
