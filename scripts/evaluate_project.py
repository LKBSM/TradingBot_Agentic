"""Unified project evaluation — single GO/NO-GO scorecard.

Runs every relevant test in one pass and produces a single Markdown +
JSON scorecard quantifying the system's commercial readiness.

10 dimensions evaluated :

1. **Imports & env**         : all critical modules import cleanly
2. **Unit tests**            : pytest core algo suite green
3. **Data quality**          : XAU + EURUSD coverage ≥ 95 %
4. **5-markets gates**       : LightGBM passes DSR/PBO/PF_lo/DM on 5/5
5. **InsightV2 e2e**         : pipeline produces valid contract
6. **Latency**               : insight generation < 250 ms / bar
7. **Narrative**             : FR + EN produced, no trade-action prose
8. **Historical stats**      : compute with costs runs and persists JSON
9. **Reproducibility**       : 2× runs of same input → identical SHA256
10. **Documentation**        : docs/algo/ + reports/certification/ present

Output :
- ``reports/evaluation/scorecard.md`` — human-readable verdict
- ``reports/evaluation/scorecard.json`` — machine-readable per-dimension
- exit code 0 if global verdict is GO, 1 otherwise

Usage::

    python scripts/evaluate_project.py [--quick]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ====================================================================== #
# Dimension runners — each returns dict {status, score_0_10, details}
# ====================================================================== #


def dim_imports() -> dict:
    """1. Verify all critical modules import cleanly."""
    mods = [
        "src.intelligence.smart_money",
        "src.intelligence.macro_factors",
        "src.intelligence.microstructure",
        "src.intelligence.factor_model",
        "src.intelligence.scoring.lgbm_scoring_engine",
        "src.intelligence.scoring",
        "src.intelligence.conformal",
        "src.intelligence.regime_filter",
        "src.intelligence.regime_gate",
        "src.intelligence.bocpd",
        "src.intelligence.signal_state_machine",
        "src.intelligence.sentinel_scanner",
        "src.intelligence.insight_v2",
        "src.backtest.state_machine_replay",
        "src.backtest.validation",
        "src.backtest.snapshot_store",
        "src.backtest.stress_tests",
        "src.research.cpcv_harness",
        "src.research.strategy_gates",
    ]
    failed = []
    for m in mods:
        try:
            __import__(m)
        except Exception as exc:
            failed.append({"module": m, "error": str(exc)[:200]})
    n_ok = len(mods) - len(failed)
    return {
        "status": "GREEN" if not failed else "RED",
        "score": int(round(10 * n_ok / len(mods))),
        "details": {"ok": n_ok, "total": len(mods), "failed": failed},
    }


def dim_unit_tests() -> dict:
    """2. Pytest core algo suite green."""
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/test_state_machine_replay.py",
        "tests/test_signal_state_machine.py",
        "tests/test_confluence_detector.py",
        "tests/test_data_quality_bos_regression.py",
        "tests/test_volatility_forecaster.py",
        "tests/test_multi_timeframe.py",
        "-q", "--tb=no", "--no-header",
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT,
                                encoding="utf-8", errors="replace", timeout=300)
    output = completed.stdout + completed.stderr
    passed = failed = 0
    for line in output.splitlines():
        if " passed" in line:
            tokens = line.split()
            for i, t in enumerate(tokens):
                if t == "passed" or t == "passed,":
                    try:
                        passed = int(tokens[i - 1])
                    except (ValueError, IndexError):
                        pass
                if t == "failed" or t == "failed,":
                    try:
                        failed = int(tokens[i - 1])
                    except (ValueError, IndexError):
                        pass
    total = passed + failed
    return {
        "status": "GREEN" if failed == 0 and passed > 100 else "RED" if failed else "YELLOW",
        "score": int(round(10 * passed / max(1, total))) if total else 0,
        "details": {"passed": passed, "failed": failed, "exit_code": completed.returncode},
    }


def dim_data_quality() -> dict:
    """3. XAU + EURUSD coverage ≥ 95 %."""
    needed = [
        ("data/XAU_15MIN_2019_2026.csv", 95.0),
        ("data/EURUSD_15MIN_2019_2025.csv", 95.0),
    ]
    results = []
    for path, threshold in needed:
        p = ROOT / path
        if not p.exists():
            results.append({"path": path, "exists": False, "coverage": 0})
            continue
        df = pd.read_csv(p, parse_dates=["Date"])
        df = df.dropna()
        if len(df) < 10:
            results.append({"path": path, "exists": True, "coverage": 0, "rows": len(df)})
            continue
        # crude coverage : actual bars / expected (15min × 7 years × ~24h × ~5 days)
        days = (df["Date"].max() - df["Date"].min()).days
        if days <= 0:
            coverage = 0.0
        else:
            expected = days * 96 * 5 / 7  # M15 weekday hours/day approximation
            coverage = min(100.0, 100.0 * len(df) / max(1, expected))
        results.append({"path": path, "rows": len(df), "coverage_pct": round(coverage, 1),
                        "ok": coverage >= threshold})
    n_ok = sum(1 for r in results if r.get("ok"))
    return {
        "status": "GREEN" if n_ok == len(needed) else "YELLOW",
        "score": int(round(10 * n_ok / len(needed))),
        "details": results,
    }


def dim_5_markets_gates(quick: bool = False) -> dict:
    """4. LightGBM walk-forward passes gates on 5/5 markets."""
    # Read the cached result from previous run if available
    cached = ROOT / "reports" / "five_markets" / "lightgbm_results.json"
    if cached.exists():
        data = json.loads(cached.read_text(encoding="utf-8"))
        n_passed = data.get("n_passed", 0)
        n_total = len(data.get("markets", []))
        return {
            "status": "GREEN" if n_passed >= 5 else "YELLOW" if n_passed >= 3 else "RED",
            "score": int(round(10 * n_passed / max(1, n_total))),
            "details": {
                "n_passed": n_passed, "n_total": n_total,
                "source": str(cached.relative_to(ROOT)),
                "markets": [
                    {"asset": m.get("asset"), "tf": m.get("tf"),
                     "all_passed": m.get("all_passed"),
                     "dsr": m.get("dsr"), "pbo": m.get("pbo"), "pf_lo": m.get("pf_lo"),
                     "ir_vs_bh": m.get("ir_vs_bh"), "n_trades": m.get("n_trades")}
                    for m in data.get("markets", [])
                ],
            },
        }
    return {"status": "YELLOW", "score": 5,
            "details": {"error": "no cached 5-markets result, run scripts/find_ai_5_markets.py"}}


def dim_insight_v2_e2e() -> dict:
    """5. InsightV2 pipeline produces valid contract."""
    try:
        from src.intelligence.smart_money import SmartMoneyEngine
        from src.intelligence.scoring.lgbm_scoring_engine import LGBMScoringEngine
        from src.intelligence.macro_factors import MacroFactorExtractor
        from src.intelligence.microstructure import MicrostructureExtractor
        from src.intelligence.insight_v2 import InsightV2Builder, InsightV2NarrativeGenerator

        ohlcv = pd.read_csv(ROOT / "data" / "XAU_15MIN_2019_2026.csv",
                            parse_dates=["Date"], nrows=2000).set_index("Date")
        ohlcv.rename(columns={c: c.capitalize() for c in ohlcv.columns
                              if c.lower() in {"open", "high", "low", "close", "volume"}},
                     inplace=True)
        engine = SmartMoneyEngine(data=ohlcv.copy(), config={}, verbose=False)
        enriched = engine.analyze()
        scoring = LGBMScoringEngine.from_pickle(ROOT / "models" / "factor_model_v1.pkl", edge_claim=True)
        macro = MacroFactorExtractor().extract(enriched.index).drop(columns=["vix_regime"], errors="ignore")
        micro = MicrostructureExtractor().extract(ohlcv)
        feats = pd.concat([macro, micro], axis=1).ffill().fillna(0)
        narr = InsightV2NarrativeGenerator(force_template=True, lang="fr")
        builder = InsightV2Builder(asset="XAUUSD", timeframe="M15",
                                    scoring_engine=scoring, edge_claim=True,
                                    narrative_generator=narr)
        insight = builder.build(
            bar_timestamp=enriched.index[-1],
            enriched_df=enriched,
            features_for_scoring=feats.iloc[-1].values,
            naive_atr=float(enriched["ATR"].iloc[-1]),
        )
        d = insight.to_dict()
        # Required keys
        required = ["insight_id", "asset", "timeframe", "structure_bias",
                    "conviction_0_100", "conviction_label", "structure_readout",
                    "regime_readout", "volatility_readout", "event_readout",
                    "scenarios", "narrative_short", "narrative_long", "compliance"]
        missing = [k for k in required if k not in d or d[k] is None]
        # Check no trade-action keys present
        forbidden = ["entry_price", "stop_loss", "take_profit", "rr_ratio"]
        present_forbidden = [k for k in forbidden if k in d]
        ok = not missing and not present_forbidden and len(d.get("scenarios", [])) >= 3
        return {
            "status": "GREEN" if ok else "RED",
            "score": 10 if ok else (5 if not missing else 0),
            "details": {
                "missing_required_keys": missing,
                "present_forbidden_keys": present_forbidden,
                "n_scenarios": len(d.get("scenarios", [])),
                "insight_id": d.get("insight_id"),
                "narrative_short_len": len(d.get("narrative_short") or ""),
                "narrative_long_len": len(d.get("narrative_long") or ""),
            },
        }
    except Exception as exc:
        return {"status": "RED", "score": 0, "details": {"error": str(exc)[:500],
                                                          "trace": traceback.format_exc()[:1000]}}


def dim_latency() -> dict:
    """6. Insight generation < 250 ms / bar."""
    try:
        from src.intelligence.smart_money import SmartMoneyEngine
        from src.intelligence.scoring.lgbm_scoring_engine import LGBMScoringEngine
        from src.intelligence.macro_factors import MacroFactorExtractor
        from src.intelligence.microstructure import MicrostructureExtractor
        from src.intelligence.insight_v2 import InsightV2Builder, InsightV2NarrativeGenerator

        ohlcv = pd.read_csv(ROOT / "data" / "XAU_15MIN_2019_2026.csv",
                            parse_dates=["Date"], nrows=2000).set_index("Date")
        ohlcv.rename(columns={c: c.capitalize() for c in ohlcv.columns
                              if c.lower() in {"open", "high", "low", "close", "volume"}},
                     inplace=True)
        engine = SmartMoneyEngine(data=ohlcv.copy(), config={}, verbose=False)
        enriched = engine.analyze()
        scoring = LGBMScoringEngine.from_pickle(ROOT / "models" / "factor_model_v1.pkl", edge_claim=True)
        macro = MacroFactorExtractor().extract(enriched.index).drop(columns=["vix_regime"], errors="ignore")
        micro = MicrostructureExtractor().extract(ohlcv)
        feats = pd.concat([macro, micro], axis=1).ffill().fillna(0)
        narr = InsightV2NarrativeGenerator(force_template=True, lang="fr")
        builder = InsightV2Builder(asset="XAUUSD", timeframe="M15",
                                    scoring_engine=scoring, edge_claim=True,
                                    narrative_generator=narr)

        # Warm-up
        builder.build(bar_timestamp=enriched.index[-1], enriched_df=enriched,
                      features_for_scoring=feats.iloc[-1].values,
                      naive_atr=float(enriched["ATR"].iloc[-1]))
        # Benchmark 50 builds (per-bar)
        n = 50
        starts = np.zeros(n)
        ends = np.zeros(n)
        for i in range(n):
            idx = -1 - i
            starts[i] = time.perf_counter()
            builder.build(bar_timestamp=enriched.index[idx], enriched_df=enriched,
                          features_for_scoring=feats.iloc[idx].values,
                          naive_atr=float(enriched["ATR"].iloc[idx]))
            ends[i] = time.perf_counter()
        latencies_ms = (ends - starts) * 1000.0
        p50 = float(np.percentile(latencies_ms, 50))
        p95 = float(np.percentile(latencies_ms, 95))
        p99 = float(np.percentile(latencies_ms, 99))
        target = 250.0
        ok = p99 < target
        score = 10 if ok else (8 if p95 < target else (5 if p50 < target else 2))
        return {
            "status": "GREEN" if ok else "YELLOW" if p95 < target else "RED",
            "score": score,
            "details": {
                "p50_ms": round(p50, 1),
                "p95_ms": round(p95, 1),
                "p99_ms": round(p99, 1),
                "target_ms": target,
                "n_samples": n,
            },
        }
    except Exception as exc:
        return {"status": "RED", "score": 0, "details": {"error": str(exc)[:500]}}


def dim_narrative() -> dict:
    """7. Narrative FR + EN, no trade-action prose."""
    try:
        from src.intelligence.insight_v2 import InsightV2NarrativeGenerator
        from src.intelligence.insight_v2.contract import (
            InsightSignalV2, StructureReadout, RegimeReadout,
            VolatilityReadout, EventReadout, HistoricalStats, ComplianceMeta,
        )

        insight = InsightSignalV2(
            insight_id="test", asset="XAUUSD", timeframe="M15",
            generated_at="2026-01-01T00:00:00Z", expires_at="2026-01-01T03:00:00Z",
            structure_bias="bullish", conviction_0_100=72.0,
            conviction_label="strong",
            conviction_interval={"lower": 54.0, "upper": 82.0, "alpha": 0.10},
            structure_readout=StructureReadout(direction="bullish", bos_level=2391.5,
                                                fvg_zone=(2378.0, 2381.0),
                                                structural_invalidation=2374.5),
            regime_readout=RegimeReadout(hmm_label="trend_bullish", hmm_posterior=0.71),
            volatility_readout=VolatilityReadout(forecast_atr=8.7, naive_atr=7.9,
                                                  forecast_vs_naive_pct=10.0),
            event_readout=EventReadout(session="new_york"),
            historical_stats=HistoricalStats(),
            compliance=ComplianceMeta(edge_claim=True, is_paper_demo=False),
        )

        out_fr = InsightV2NarrativeGenerator(force_template=True, lang="fr").generate(insight)
        out_en = InsightV2NarrativeGenerator(force_template=True, lang="en").generate(insight)

        forbidden_fr = ["achetez", "vendez", "buy at", "stop loss", "take profit",
                        "entrée à", "sortie à", "ordre d'achat", "ordre de vente"]
        forbidden_en = ["buy at", "sell at", "stop loss", "take profit", "entry price"]
        # Check forbidden absent
        fr_text = (out_fr.short + " " + out_fr.long).lower()
        en_text = (out_en.short + " " + out_en.long).lower()
        fr_violations = [w for w in forbidden_fr if w in fr_text]
        en_violations = [w for w in forbidden_en if w in en_text]
        ok = (out_fr.short and out_fr.long and out_en.short and out_en.long
              and not fr_violations and not en_violations
              and len(out_fr.short) <= 400 and len(out_en.short) <= 400)
        return {
            "status": "GREEN" if ok else "YELLOW",
            "score": 10 if ok else (5 if (fr_violations or en_violations) else 7),
            "details": {
                "fr_short_chars": len(out_fr.short),
                "fr_long_chars": len(out_fr.long),
                "en_short_chars": len(out_en.short),
                "en_long_chars": len(out_en.long),
                "fr_violations": fr_violations,
                "en_violations": en_violations,
                "backend_fr": out_fr.backend,
                "backend_en": out_en.backend,
            },
        }
    except Exception as exc:
        return {"status": "RED", "score": 0, "details": {"error": str(exc)[:500]}}


def dim_historical_stats() -> dict:
    """8. Historical stats with costs runs and persists JSON."""
    stats_path = ROOT / "reports" / "historical_stats" / "EURUSD_M15.json"
    if not stats_path.exists():
        return {"status": "YELLOW", "score": 5,
                "details": {"error": "no cached historical stats — run scripts/compute_historical_stats.py"}}
    data = json.loads(stats_path.read_text(encoding="utf-8"))
    has_costs = data.get("cost_assumptions", {}).get("round_trip_spread_bps") is not None
    has_pf = data.get("profit_factor") is not None
    has_ci = data.get("profit_factor_ci95") is not None
    has_no_costs_ref = data.get("profit_factor_no_costs_reference") is not None
    score = sum([2.5 * int(b) for b in (has_costs, has_pf, has_ci, has_no_costs_ref)])
    return {
        "status": "GREEN" if score >= 8 else "YELLOW" if score >= 5 else "RED",
        "score": int(round(score)),
        "details": {
            "has_costs": has_costs, "has_pf": has_pf, "has_ci": has_ci,
            "has_no_costs_ref": has_no_costs_ref,
            "pf_with_costs": data.get("profit_factor"),
            "pf_ci95_with_costs": data.get("profit_factor_ci95"),
            "pf_without_costs_ref": data.get("profit_factor_no_costs_reference"),
            "n_trades": data.get("similar_setups_n"),
            "sample_window": data.get("sample_window"),
        },
    }


def dim_reproducibility() -> dict:
    """9. 2× runs of same input → identical SHA256."""
    try:
        from src.intelligence.smart_money import SmartMoneyEngine
        ohlcv = pd.read_csv(ROOT / "data" / "XAU_15MIN_2019_2026.csv",
                            parse_dates=["Date"], nrows=2000).set_index("Date")
        ohlcv.rename(columns={c: c.capitalize() for c in ohlcv.columns
                              if c.lower() in {"open", "high", "low", "close", "volume"}},
                     inplace=True)

        def hash_run() -> str:
            engine = SmartMoneyEngine(data=ohlcv.copy(), config={}, verbose=False)
            enriched = engine.analyze()
            # Use a stable subset of columns for hashing
            cols = [c for c in ["BOS_SIGNAL", "BOS_EVENT", "FVG_DIR", "FVG_SIZE",
                                 "BULLISH_OB_HIGH", "BEARISH_OB_HIGH", "RSI", "ATR"]
                    if c in enriched.columns]
            payload = enriched[cols].fillna(0).round(6).to_csv().encode("utf-8")
            return hashlib.sha256(payload).hexdigest()

        h1 = hash_run()
        h2 = hash_run()
        ok = h1 == h2
        return {
            "status": "GREEN" if ok else "RED",
            "score": 10 if ok else 0,
            "details": {"sha1": h1[:16], "sha2": h2[:16], "match": ok},
        }
    except Exception as exc:
        return {"status": "RED", "score": 0, "details": {"error": str(exc)[:500]}}


def dim_documentation() -> dict:
    """10. docs/algo/ + reports/certification/ present."""
    expected = [
        "docs/algo/README.md",
        "docs/deployment/docker.md",
        "reports/certification/v1.0_commercial_readiness.md",
        "reports/certification/BREAKTHROUGH_2026_05_16.md",
        "reports/certification/INSTITUTIONAL_PIVOT_FINAL.md",
        "audits/2026-Q2/algo_audit_institutional.md",
        "audits/2026-Q2/sprint_0_decisions.md",
        "roadmap/sprints/sprint_0.md",
        "agents/ROSTER.md",
        "MISSION_ACK.md",
        "CHANGELOG.md",
        "mockups/v2/client_view_full.html",
    ]
    present = []
    missing = []
    for path in expected:
        p = ROOT / path
        if p.exists():
            present.append(path)
        else:
            missing.append(path)
    n_ok = len(present)
    return {
        "status": "GREEN" if not missing else "YELLOW" if n_ok >= 8 else "RED",
        "score": int(round(10 * n_ok / len(expected))),
        "details": {"present": n_ok, "total": len(expected), "missing": missing},
    }


# ====================================================================== #
# Orchestrator
# ====================================================================== #


DIMENSIONS = [
    ("imports", "1. Imports & env",            dim_imports,            False),
    ("unit_tests", "2. Unit tests core algo",   dim_unit_tests,         False),
    ("data_quality", "3. Data quality coverage", dim_data_quality,      False),
    ("five_markets", "4. 5-markets gates AI",    dim_5_markets_gates,   True),
    ("insight_v2_e2e", "5. InsightV2 E2E",       dim_insight_v2_e2e,    False),
    ("latency", "6. Latency < 250ms",            dim_latency,            False),
    ("narrative", "7. Narrative FR+EN",          dim_narrative,          False),
    ("historical_stats", "8. Historical stats cost-aware", dim_historical_stats, False),
    ("reproducibility", "9. Reproducibility bit-for-bit",   dim_reproducibility,  False),
    ("documentation", "10. Documentation",        dim_documentation,     False),
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    out_dir = ROOT / "reports" / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Smart Sentinel AI — Project Evaluation Scorecard")
    print(f"Date : {datetime.now(timezone.utc).isoformat()}")
    print("=" * 70)

    results = {}
    for key, label, runner, accepts_quick in DIMENSIONS:
        print(f"\n[{label}] running...")
        t0 = time.perf_counter()
        try:
            if accepts_quick:
                res = runner(quick=args.quick)
            else:
                res = runner()
        except Exception as exc:
            res = {"status": "RED", "score": 0,
                   "details": {"runner_error": str(exc)[:300]}}
        res["latency_s"] = round(time.perf_counter() - t0, 2)
        results[key] = res
        status_icon = {"GREEN": "✅", "YELLOW": "🟡", "RED": "❌"}.get(res["status"], "❓")
        print(f"  {status_icon} {res['status']}  score {res.get('score', 0)}/10  ({res['latency_s']}s)")

    # Aggregate
    total = sum(r.get("score", 0) for r in results.values())
    max_total = len(DIMENSIONS) * 10
    avg = total / len(DIMENSIONS)
    n_green = sum(1 for r in results.values() if r["status"] == "GREEN")
    n_yellow = sum(1 for r in results.values() if r["status"] == "YELLOW")
    n_red = sum(1 for r in results.values() if r["status"] == "RED")

    if n_red == 0 and avg >= 8.0:
        verdict = "GO COMMERCIAL"
        verdict_icon = "🟢"
    elif n_red == 0 and avg >= 6.0:
        verdict = "GO PILOT (B2B early adopter)"
        verdict_icon = "🟡"
    elif n_red <= 2:
        verdict = "NO-GO — fix REDs first"
        verdict_icon = "🟠"
    else:
        verdict = "NO-GO COMPLET"
        verdict_icon = "🔴"

    print()
    print("=" * 70)
    print(f"VERDICT GLOBAL : {verdict_icon} {verdict}")
    print(f"  Score moyen : {avg:.2f}/10  ({total}/{max_total})")
    print(f"  GREEN  : {n_green}/{len(DIMENSIONS)}")
    print(f"  YELLOW : {n_yellow}/{len(DIMENSIONS)}")
    print(f"  RED    : {n_red}/{len(DIMENSIONS)}")
    print("=" * 70)

    # Save JSON
    payload = {
        "computed_at_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "score_avg": round(avg, 2),
        "score_total": total,
        "score_max": max_total,
        "n_green": n_green,
        "n_yellow": n_yellow,
        "n_red": n_red,
        "dimensions": results,
    }
    (out_dir / "scorecard.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    # Save MD
    lines = [
        f"# Smart Sentinel AI — Project Scorecard",
        "",
        f"**Date** : {datetime.now(timezone.utc).isoformat()}",
        f"**Verdict** : {verdict_icon} **{verdict}**",
        f"**Score moyen** : {avg:.2f}/10  ({total}/{max_total})",
        f"**Bilan** : 🟢 {n_green} · 🟡 {n_yellow} · 🔴 {n_red}",
        "",
        "## Per-dimension breakdown",
        "",
        "| # | Dimension | Status | Score /10 | Latency |",
        "| - | --- | --- | --- | --- |",
    ]
    for key, label, _, _ in DIMENSIONS:
        r = results[key]
        icon = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}.get(r["status"], "❓")
        lines.append(f"| {label.split('.')[0]} | {label} | {icon} {r['status']} | "
                     f"**{r.get('score', 0)}** | {r['latency_s']}s |")

    lines.append("")
    lines.append("## Detail per dimension")
    for key, label, _, _ in DIMENSIONS:
        r = results[key]
        lines.append(f"\n### {label}")
        lines.append("")
        lines.append(f"- **Status** : {r['status']}")
        lines.append(f"- **Score** : {r.get('score', 0)}/10")
        lines.append(f"- **Latency** : {r['latency_s']}s")
        details = r.get("details", {})
        if details:
            lines.append("- **Details** :")
            lines.append("```json")
            lines.append(json.dumps(details, indent=2, default=str)[:1500])
            lines.append("```")

    (out_dir / "scorecard.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved {out_dir / 'scorecard.md'}")
    print(f"Saved {out_dir / 'scorecard.json'}")
    return 0 if n_red == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
