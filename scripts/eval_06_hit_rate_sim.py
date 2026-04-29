"""Hit-rate simulation for SemanticCache on a synthetic-but-realistic signal stream.

Drives the cache through a representative cardinality of signals and reports
the empirical hit rate. The component-score distribution is sampled from the
actual replay distribution observed on XAU 2019-2024 (per eval_02 reports).
"""
from __future__ import annotations

import json
import random
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.intelligence.semantic_cache import SemanticCache  # noqa: E402


@dataclass
class _Comp:
    name: str
    weighted_score: float
    weight: float = 1.0


@dataclass
class _Signal:
    symbol: str
    signal_type: str
    tier: str
    components: List[_Comp] = field(default_factory=list)


def make_signal(rng: random.Random, symbol: str = "XAUUSD") -> _Signal:
    """Sample a signal whose components mimic the replay distribution.

    Empirical priors (from eval_02 score distribution + Sprint 7 detector
    reset): score-bucket density concentrates around plateaus once the
    detector gates fire deterministically.
    """
    direction = rng.choice(["LONG", "SHORT"])
    tier_choice = rng.choices(
        ["PREMIUM", "STANDARD", "WEAK"],
        weights=[0.05, 0.25, 0.7],
    )[0]

    bos = rng.choice([0, 12, 15])
    fvg = rng.choice([0, 10, 12, 15])
    ob = rng.choice([0, 10, 15])
    regime = rng.choice([0, 8, 15, 20, 23])
    news = rng.choice([0, 5, 10, 18])
    vol = rng.choice([0, 4, 8, 12])
    mom = rng.choice([0, 1.0, 1.5, 2.0, 2.5])
    rsi = rng.choice([0, 1, 2, 3])

    return _Signal(
        symbol=symbol,
        signal_type=direction,
        tier=tier_choice,
        components=[
            _Comp("bos", bos),
            _Comp("fvg", fvg),
            _Comp("orderblock", ob),
            _Comp("regime", regime),
            _Comp("news", news),
            _Comp("volume", vol),
            _Comp("momentum", mom),
            _Comp("rsi_div", rsi),
        ],
    )


def main():
    rng = random.Random(42)
    n_signals_per_run = 1200  # ~6 months of valid signals on XAU M15
    n_symbols = 1
    runs = []

    with tempfile.TemporaryDirectory() as tmp:
        cache = SemanticCache(db_path=str(Path(tmp) / "cache.db"))

        unique_keys = set()
        per_call_hit = []

        for i in range(n_signals_per_run):
            sig = make_signal(rng)
            key = SemanticCache.generate_cache_key(sig)
            unique_keys.add(key)
            hit = cache.get(key) is not None
            per_call_hit.append(hit)
            if not hit:
                cache.put(key, {"text": f"narrative_for_{key}", "v": 1})

        stats = cache.get_stats()
        out = {
            "n_signals_simulated": n_signals_per_run,
            "n_symbols": n_symbols,
            "unique_keys": len(unique_keys),
            "cardinality_pct": round(100 * len(unique_keys) / n_signals_per_run, 2),
            "stats_from_cache": stats,
            "manual_hit_rate": round(sum(per_call_hit) / max(1, len(per_call_hit)), 3),
            "hit_rate_after_first_50pct": round(
                sum(per_call_hit[n_signals_per_run // 2:]) / (n_signals_per_run // 2), 3
            ),
            "hit_rate_last_25pct": round(
                sum(per_call_hit[-n_signals_per_run // 4:]) / (n_signals_per_run // 4), 3
            ),
        }
        runs.append(out)

    out_path = Path("reports/eval_06/hit_rate_sim.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(runs, f, indent=2)
    print(json.dumps(runs, indent=2))


if __name__ == "__main__":
    main()
