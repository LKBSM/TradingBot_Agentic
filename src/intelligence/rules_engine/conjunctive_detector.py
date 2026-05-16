"""ConjunctiveDetector — pure algorithm scorer alternative.

Drop-in interface compatible avec :class:`ConfluenceDetector` mais sans
agrégation additive. Émet un signal **uniquement** si toutes les rules
de la RuleSet par défaut passent — pas de score 0-100 (binaire FIRE/NO).

Architecture
------------
Le détecteur consomme les mêmes features SMC enrichies par
:class:`SmartMoneyEngine`. Les rules sont écrites en pur Python
déterministe, faciles à lire et auditer.

Exemple de RuleSet par défaut (XAU M15) — basée sur ICT classique :

```
LONG_RULES = AND(
    BOS_event == 1                            # break of structure up
    AND retest_armed == 1                     # pullback into OB/FVG
    AND ATR_PCTL < 0.85                       # vol not extreme
    AND session ∈ {London, NY}                # liquid session
    AND regime_decision != BLOCK              # no regime gate block
)
```

Status
------
Scaffold + default ruleset for XAU. Real tuning happens via
:mod:`scripts/sweep_rules.py` (Sprint 4 alternative path, ~6-8h).
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.intelligence.confluence_detector import (
    ConfluenceSignal, SignalTier, SignalType, ComponentScore,
)
from src.intelligence.rules_engine.rules import Rule, RuleSet

logger = logging.getLogger(__name__)


def default_long_ruleset() -> RuleSet:
    """ICT-canonical LONG rules — XAU M15 starting point."""
    rs = RuleSet(name="LONG_ICT_canonical", mode="AND")
    rs.add_rule(Rule(
        "bos_up_recent",
        lambda f: int(f.get("BOS_EVENT", 0)) == 1 or int(f.get("BOS_SIGNAL", 0)) == 1,
        "Break of structure up within reach",
    ))
    rs.add_rule(Rule(
        "retest_armed",
        lambda f: int(f.get("BOS_RETEST_ARMED", 0)) == 1,
        "Pullback into OB/FVG zone armed",
    ))
    rs.add_rule(Rule(
        "atr_not_extreme",
        lambda f: float(f.get("ATR_PCTL", 0.5)) < 0.85,
        "Volatility percentile below 0.85 (avoid extremes)",
    ))
    rs.add_rule(Rule(
        "regime_not_block",
        lambda f: str(f.get("regime_decision", "TRADE")).upper() != "BLOCK",
        "Regime gate does not block entries",
    ))
    rs.add_rule(Rule(
        "session_liquid",
        lambda f: int(f.get("session_is_london", 0)) == 1
                  or int(f.get("session_is_ny", 0)) == 1,
        "London or NY session active",
    ))
    return rs


def default_short_ruleset() -> RuleSet:
    """ICT-canonical SHORT rules — mirror of LONG."""
    rs = RuleSet(name="SHORT_ICT_canonical", mode="AND")
    rs.add_rule(Rule(
        "bos_down_recent",
        lambda f: int(f.get("BOS_EVENT", 0)) == -1 or int(f.get("BOS_SIGNAL", 0)) == -1,
        "Break of structure down within reach",
    ))
    rs.add_rule(Rule(
        "retest_armed",
        lambda f: int(f.get("BOS_RETEST_ARMED", 0)) == 1,
        "Pullback into OB/FVG zone armed",
    ))
    rs.add_rule(Rule(
        "atr_not_extreme",
        lambda f: float(f.get("ATR_PCTL", 0.5)) < 0.85,
        "Volatility percentile below 0.85",
    ))
    rs.add_rule(Rule(
        "regime_not_block",
        lambda f: str(f.get("regime_decision", "TRADE")).upper() != "BLOCK",
        "Regime gate does not block entries",
    ))
    rs.add_rule(Rule(
        "session_liquid",
        lambda f: int(f.get("session_is_london", 0)) == 1
                  or int(f.get("session_is_ny", 0)) == 1,
        "London or NY session active",
    ))
    return rs


class ConjunctiveDetector:
    """Pure-algorithm detector — no score, no ML.

    Returns a :class:`ConfluenceSignal` with score=100 (binary fire) when
    the relevant ruleset passes, else None. The `components` list
    enumerates the rule names that fired (for auditability).
    """

    def __init__(
        self,
        symbol: str = "XAUUSD",
        long_ruleset: Optional[RuleSet] = None,
        short_ruleset: Optional[RuleSet] = None,
        instrument_config: Any = None,
        require_retest: bool = True,
    ):
        self.symbol = symbol
        self.long_ruleset = long_ruleset or default_long_ruleset()
        self.short_ruleset = short_ruleset or default_short_ruleset()
        self.instrument_config = instrument_config
        self.require_retest = bool(require_retest)
        # Property accessed by SignalReplay
        self.min_score = 100.0  # binary fire → score=100 if pass

    def analyze(
        self,
        smc_features: Dict[str, Any],
        regime: Any = None,
        news: Any = None,
        price: float = 0.0,
        atr: float = 0.0,
        volume: Optional[float] = None,
        volume_ma: Optional[float] = None,
        vol_forecast: Any = None,
        bar_timestamp: Optional[Any] = None,
    ) -> Optional[ConfluenceSignal]:
        """Return a binary-fire signal when the conjunctive ruleset passes."""
        # Inject regime decision into features dict so rules can read it
        feats = dict(smc_features)
        if regime is not None:
            feats["regime_decision"] = getattr(regime, "decision", "TRADE")

        # Try LONG first
        for direction, rs in [(SignalType.LONG, self.long_ruleset),
                              (SignalType.SHORT, self.short_ruleset)]:
            verdict, passed = rs.evaluate(feats)
            if verdict:
                # Build a synthetic ConfluenceSignal so downstream replay
                # works unchanged.
                components = [
                    ComponentScore(name=name, raw_value=1.0, weighted_score=20.0,
                                   weight=20.0, reasoning="Rule passed")
                    for name in passed
                ]
                bts = bar_timestamp.isoformat() if hasattr(bar_timestamp, "isoformat") else str(bar_timestamp or "na")
                sigid = hashlib.sha1(
                    f"conjunctive|{self.symbol}|{bts}|{direction.value}".encode("utf-8")
                ).hexdigest()[:12]
                # ICT defaults : SL=2×ATR, TP=4×ATR
                if direction == SignalType.LONG:
                    sl = price - 2.0 * atr
                    tp = price + 4.0 * atr
                else:
                    sl = price + 2.0 * atr
                    tp = price - 4.0 * atr
                rr = abs(tp - price) / max(1e-9, abs(price - sl))
                return ConfluenceSignal(
                    signal_id=sigid,
                    symbol=self.symbol,
                    signal_type=direction,
                    confluence_score=100.0,   # binary fire
                    tier=SignalTier.PREMIUM,  # binary fire = PREMIUM (top tier)
                    entry_price=price,
                    stop_loss=sl,
                    take_profit=tp,
                    rr_ratio=rr,
                    atr=atr,
                    components=components,
                    reasoning=[f"Conjunctive: {p}" for p in passed],
                    bar_timestamp=bts,
                    position_multiplier=1.0,
                )
        return None


__all__ = ["ConjunctiveDetector", "default_long_ruleset", "default_short_ruleset"]
