"""LLM-judge benchmark for Prompt 05 — narrative engine quality.

Samples N signals from a historical OHLCV file, generates narratives via
the LLMNarrativeEngine (Sonnet by default) AND the TemplateNarrativeEngine
(deterministic baseline), then asks an Opus judge to score each narrative
on a 25-point rubric:

    1. Faithfulness to payload (no hallucinations)             — 0–5
    2. Use of SMC framework (BOS / FVG / OB / regime cited)    — 0–5
    3. Risk frame completeness (SL price, R:R, invalidation)   — 0–5
    4. Tone & clarity (institutional, no marketing fluff)      — 0–5
    5. Actionability (a trader could act on this)              — 0–5

CI gate: mean LLM score must be ≥ ``--threshold`` (default 18/25). The
template baseline is reported alongside but does NOT gate CI — its purpose
is to expose whether the LLM is delivering measurable lift over the algo.

Usage
-----
    # Local with cached calls (no API):
    python scripts/eval_05_narratives.py \
        --data data/XAU_15MIN_2019_2024.csv --n 20 --offline

    # Real Anthropic calls (requires ANTHROPIC_API_KEY):
    python scripts/eval_05_narratives.py \
        --data data/XAU_15MIN_2019_2024.csv --n 20 --threshold 18

CI exit codes:
    0   mean LLM score >= threshold
    1   mean LLM score <  threshold (fail the build)
    2   not enough signals could be generated to evaluate

The script emits ``reports/eval_05/narratives_<timestamp>.json`` with per-
sample scores, judge rationales, and aggregate stats.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.intelligence.confluence_detector import (  # noqa: E402
    ConfluenceDetector,
    ConfluenceSignal,
)
from src.intelligence.llm_narrative_engine import (  # noqa: E402
    DEFAULT_INSTITUTIONAL_MODEL,
    DEFAULT_NARRATOR_MODEL,
    LLMNarrativeEngine,
    NarrativeTier,
)
from src.intelligence.template_narrative_engine import (  # noqa: E402
    TemplateNarrativeEngine,
)


# =============================================================================
# RUBRIC
# =============================================================================

JUDGE_SYSTEM_PROMPT = """You are an institutional trading analyst evaluating
the quality of automated signal narratives produced by Smart Sentinel AI.

You will receive:
  1. The raw signal payload (CSV) — the ground truth.
  2. The narrative the engine produced.

You must score the narrative on FIVE dimensions, each worth 0–5 points:

  1. FAITHFULNESS — Does the narrative cite ONLY information present in the
     payload? Penalize invented prices, news events, levels, percentages, or
     components that aren't in the CSV.
  2. SMC_FRAMEWORK — Does the narrative correctly use Smart Money Concepts
     vocabulary (BOS, FVG, Order Block, regime, CHoCH) where the payload
     supports it?
  3. RISK_FRAME — Does the narrative quote the SL price, the R:R ratio, AND
     a clear invalidation condition?
  4. TONE — Is the tone institutional? (No emojis, no exclamation marks, no
     superlatives, no marketing fluff like "must-trade" or "explosive".)
  5. ACTIONABILITY — Could a competent trader act on this narrative without
     additional information?

Reply with STRICT JSON, no preamble:

{
  "faithfulness": <int 0-5>,
  "smc_framework": <int 0-5>,
  "risk_frame": <int 0-5>,
  "tone": <int 0-5>,
  "actionability": <int 0-5>,
  "total": <int 0-25>,
  "rationale": "<≤200 char explanation of the lowest sub-score>"
}
"""


# =============================================================================
# DATA LOADING + SIGNAL SAMPLING
# =============================================================================

def load_ohlcv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    ts_col = "timestamp" if "timestamp" in df.columns else "date"
    df["timestamp"] = pd.to_datetime(df[ts_col])
    cols = ["timestamp", "open", "high", "low", "close", "volume"]
    return df[cols].sort_values("timestamp").reset_index(drop=True)


def sample_signals(df: pd.DataFrame, n: int, seed: int = 42) -> List[ConfluenceSignal]:
    """Generate ``n`` synthetic high-quality signals across the dataset.

    We don't need a full SMC pipeline here — we synthesise a payload directly
    from random rows so the eval focuses on narrative quality, not detector
    plumbing. The signals span LONG/SHORT and a range of scores/tiers so the
    judge sees real variety.
    """
    from src.intelligence.confluence_detector import (
        ComponentScore, SignalTier, SignalType,
    )

    rng = random.Random(seed)
    signals: List[ConfluenceSignal] = []
    if len(df) < 100:
        return signals

    indices = rng.sample(range(50, len(df) - 1), min(n, len(df) - 51))
    regimes = ["strong_uptrend", "weak_uptrend", "ranging", "weak_downtrend", "strong_downtrend"]
    tier_buckets: List[Tuple[float, SignalTier]] = [
        (82.0, SignalTier.PREMIUM),
        (76.0, SignalTier.STANDARD),
        (68.0, SignalTier.STANDARD),
        (62.0, SignalTier.STANDARD),
        (52.0, SignalTier.WEAK),
    ]

    for idx in indices:
        row = df.iloc[idx]
        price = float(row["close"])
        atr = max(float(row["high"]) - float(row["low"]), 0.5)
        direction = SignalType.LONG if rng.random() > 0.5 else SignalType.SHORT
        score, tier = rng.choice(tier_buckets)
        regime = rng.choice(regimes)
        if direction == SignalType.LONG:
            sl = price - 2 * atr
            tp = price + 4 * atr
        else:
            sl = price + 2 * atr
            tp = price - 4 * atr
        rr = 2.0

        components: List[ComponentScore] = [
            ComponentScore(
                name="BOS", raw_value=1.0 if direction == SignalType.LONG else -1.0,
                weighted_score=15.0, weight=15.0, reasoning="BOS aligned",
            ),
            ComponentScore(
                name="FVG", raw_value=1.0 if direction == SignalType.LONG else -1.0,
                weighted_score=12.0, weight=15.0, reasoning="FVG nearby",
            ),
            ComponentScore(
                name="OrderBlock", raw_value=0.7,
                weighted_score=7.0, weight=10.0, reasoning="OB rebound",
            ),
            ComponentScore(
                name="Regime", raw_value=0.7,
                weighted_score=18.0, weight=25.0, reasoning=f"Regime {regime}",
            ),
        ]

        sig = ConfluenceSignal(
            signal_id=f"eval_{idx:06d}",
            symbol="XAUUSD",
            signal_type=direction,
            confluence_score=score,
            tier=tier,
            entry_price=round(price, 2),
            stop_loss=round(sl, 2),
            take_profit=round(tp, 2),
            rr_ratio=rr,
            atr=round(atr, 2),
            components=components,
            created_at=pd.Timestamp(row["timestamp"]).to_pydatetime(),
            bar_timestamp=pd.Timestamp(row["timestamp"]).to_pydatetime(),
            vol_forecast_atr=None,
            vol_regime=None,
            vol_confidence_lower=None,
            vol_confidence_upper=None,
        )
        signals.append(sig)

    return signals


# =============================================================================
# JUDGE
# =============================================================================

JUDGE_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


@dataclass
class JudgeScore:
    faithfulness: int = 0
    smc_framework: int = 0
    risk_frame: int = 0
    tone: int = 0
    actionability: int = 0
    total: int = 0
    rationale: str = ""

    @classmethod
    def from_judge_text(cls, text: str) -> "JudgeScore":
        match = JUDGE_JSON_RE.search(text)
        if not match:
            return cls(rationale=f"unparseable: {text[:120]}")
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            return cls(rationale=f"json error: {exc}")
        score = cls(
            faithfulness=int(data.get("faithfulness", 0)),
            smc_framework=int(data.get("smc_framework", 0)),
            risk_frame=int(data.get("risk_frame", 0)),
            tone=int(data.get("tone", 0)),
            actionability=int(data.get("actionability", 0)),
            rationale=str(data.get("rationale", ""))[:200],
        )
        score.total = (
            score.faithfulness + score.smc_framework + score.risk_frame
            + score.tone + score.actionability
        )
        return score


def offline_score(payload: str, narrative: str) -> JudgeScore:
    """Heuristic offline scorer — used when --offline is set or no API key.

    This is a coarse sanity check, not a real evaluation. It rewards length,
    SMC vocabulary, risk-frame keywords, and absence of marketing words.
    """
    n = (narrative or "").lower()
    # FAITHFULNESS: penalize words that almost certainly aren't in payload.
    bad_invent = any(w in n for w in ("nfp", "cpi", "fomc", "bitcoin", "btc", "equities"))
    faithfulness = 5 if not bad_invent and narrative else 2

    # SMC vocab
    smc_hits = sum(1 for w in ("bos", "fvg", "order block", "regime", "choch") if w in n)
    smc_framework = min(5, smc_hits + 1)

    # Risk frame
    risk_hits = sum(1 for w in ("sl ", "stop", "r:r", "tp ", "invalidat") if w in n)
    risk_frame = min(5, risk_hits + 1)

    # Tone — penalise marketing words
    bad_tone = any(w in n for w in ("must-trade", "explosive", "incredible", "guarantee", "!"))
    tone = 2 if bad_tone else (4 if narrative else 0)

    # Actionability
    actionability = 4 if narrative and len(narrative.split()) > 30 else 2

    total = faithfulness + smc_framework + risk_frame + tone + actionability
    return JudgeScore(
        faithfulness=faithfulness,
        smc_framework=smc_framework,
        risk_frame=risk_frame,
        tone=tone,
        actionability=actionability,
        total=total,
        rationale="offline heuristic",
    )


def judge_narrative(
    client: Any,
    judge_model: str,
    payload: str,
    narrative: str,
    timeout_s: float = 30.0,
) -> JudgeScore:
    if not narrative:
        return JudgeScore(rationale="empty narrative")
    user = f"Signal payload:\n{payload}\n\nNarrative to score:\n{narrative}\n\nReturn ONLY the JSON object."
    try:
        response = client.messages.create(
            model=judge_model,
            max_tokens=512,
            system=[{"type": "text", "text": JUDGE_SYSTEM_PROMPT,
                     "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
            timeout=timeout_s,
        )
        return JudgeScore.from_judge_text(response.content[0].text)
    except Exception as exc:
        return JudgeScore(rationale=f"judge error: {exc}")


# =============================================================================
# MAIN
# =============================================================================

@dataclass
class SampleResult:
    signal_id: str
    direction: str
    score: float
    tier: str
    payload_csv: str = ""
    llm_narrative: str = ""
    template_narrative: str = ""
    llm_score: Dict[str, Any] = field(default_factory=dict)
    template_score: Dict[str, Any] = field(default_factory=dict)
    llm_latency_ms: float = 0.0


def run(args: argparse.Namespace) -> int:
    df = load_ohlcv(args.data)
    signals = sample_signals(df, args.n, seed=args.seed)
    if len(signals) < args.min_samples:
        print(
            f"[FAIL] only {len(signals)} signals generated, need >= {args.min_samples}",
            file=sys.stderr,
        )
        return 2

    # Engines
    api_key = os.environ.get("ANTHROPIC_API_KEY") if not args.offline else None
    llm_engine = LLMNarrativeEngine(
        api_key=api_key,
        narrator_model=args.narrator_model,
        institutional_model=args.judge_model,
    )
    template_engine = TemplateNarrativeEngine()

    # Judge client
    judge_client = None
    if not args.offline and api_key:
        try:
            import anthropic
            judge_client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            print("[WARN] anthropic SDK not installed — using offline heuristic", file=sys.stderr)

    results: List[SampleResult] = []
    for sig in signals:
        payload_csv = LLMNarrativeEngine._signal_to_csv(sig)

        # LLM narrative (Sonnet by default, falls back to visual-only if no client)
        t0 = time.time()
        llm_narrative = llm_engine.generate_narrative(sig, NarrativeTier.NARRATOR)
        llm_latency = (time.time() - t0) * 1000

        # Template narrative
        template_narrative = template_engine.generate_narrative(sig, NarrativeTier.NARRATOR)

        # Score both
        if judge_client is not None:
            llm_score = judge_narrative(
                judge_client, args.judge_model, payload_csv,
                llm_narrative.full_narrative, timeout_s=args.timeout,
            )
            template_score = judge_narrative(
                judge_client, args.judge_model, payload_csv,
                template_narrative.full_narrative, timeout_s=args.timeout,
            )
        else:
            llm_score = offline_score(payload_csv, llm_narrative.full_narrative)
            template_score = offline_score(payload_csv, template_narrative.full_narrative)

        results.append(SampleResult(
            signal_id=sig.signal_id,
            direction=sig.signal_type.value,
            score=sig.confluence_score,
            tier=sig.tier.value,
            payload_csv=payload_csv,
            llm_narrative=llm_narrative.full_narrative,
            template_narrative=template_narrative.full_narrative,
            llm_score=asdict(llm_score),
            template_score=asdict(template_score),
            llm_latency_ms=round(llm_latency, 1),
        ))

    # Aggregate
    llm_totals = [r.llm_score["total"] for r in results]
    tpl_totals = [r.template_score["total"] for r in results]
    aggregate = {
        "n": len(results),
        "llm": {
            "mean": round(statistics.mean(llm_totals), 2),
            "median": round(statistics.median(llm_totals), 2),
            "min": min(llm_totals),
            "max": max(llm_totals),
            "p25": round(np.percentile(llm_totals, 25), 2),
            "p75": round(np.percentile(llm_totals, 75), 2),
        },
        "template": {
            "mean": round(statistics.mean(tpl_totals), 2),
            "median": round(statistics.median(tpl_totals), 2),
            "min": min(tpl_totals),
            "max": max(tpl_totals),
        },
        "lift_llm_vs_template": round(
            statistics.mean(llm_totals) - statistics.mean(tpl_totals), 2,
        ),
        "threshold": args.threshold,
        "passed": statistics.mean(llm_totals) >= args.threshold,
        "judge_model": args.judge_model,
        "narrator_model": args.narrator_model,
        "offline": args.offline or judge_client is None,
        "llm_engine_stats": llm_engine.get_stats(),
    }

    # Persist
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"narratives_{ts}.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump({
            "aggregate": aggregate,
            "samples": [asdict(r) for r in results],
        }, fh, indent=2, default=str)

    # Console summary
    print(f"\n=== Eval 05 Narrative Quality ({aggregate['n']} samples) ===")
    print(f"LLM      : mean {aggregate['llm']['mean']}/25  median {aggregate['llm']['median']}/25  range [{aggregate['llm']['min']}-{aggregate['llm']['max']}]")
    print(f"Template : mean {aggregate['template']['mean']}/25  median {aggregate['template']['median']}/25")
    print(f"Lift     : {aggregate['lift_llm_vs_template']:+.2f} pts (LLM vs Template)")
    print(f"Threshold: {args.threshold}/25  ->  {'PASS' if aggregate['passed'] else 'FAIL'}")
    print(f"Report   : {out_path}")

    return 0 if aggregate["passed"] else 1


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, help="OHLCV CSV path")
    p.add_argument("--n", type=int, default=20, help="number of signals to evaluate")
    p.add_argument("--min-samples", type=int, default=10,
                   help="hard floor — fail with exit 2 if fewer signals generated")
    p.add_argument("--threshold", type=float, default=18.0,
                   help="mean LLM score floor (0-25) — exit 1 below this")
    p.add_argument("--narrator-model", default=DEFAULT_NARRATOR_MODEL)
    p.add_argument("--judge-model", default=DEFAULT_INSTITUTIONAL_MODEL,
                   help="model used as the rubric judge (Opus by default)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--offline", action="store_true",
                   help="skip Anthropic calls — use heuristic scorer only")
    p.add_argument("--out", default="reports/eval_05",
                   help="output directory for the JSON report")
    return p.parse_args(argv)


if __name__ == "__main__":
    sys.exit(run(parse_args()))
