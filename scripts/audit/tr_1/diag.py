"""TR-1 diagnostic — READ ONLY. Compares two STRUCTURAL trend definitions
against real XAUUSD candles, measures the 'indeterminate' frequency, trend-flip
counts, window behaviour. The engine is NEVER modified: we only read its
enriched output (UP_FRACTAL/DOWN_FRACTAL/BOS_EVENT/CHOCH_SIGNAL/BOS_SIGNAL).

Run from the worktree root:  python scripts/audit/tr_1/diag.py
"""
from __future__ import annotations
import json, sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.audit.mt_d1.harness import load_candles, enrich  # reuse MT-D1 loader

WINDOW = 500  # MARKET_READING_LOOKBACK default


def rolling_defs(enriched, window=WINDOW):
    """For every bar, evaluate the two structural definitions using ONLY the
    trailing `window` bars (mirrors the live assembler's fixed lookback)."""
    idx = list(enriched.index)
    n = len(enriched)
    be = enriched["BOS_EVENT"].values
    ch = enriched["CHOCH_SIGNAL"].values
    upf = enriched["UP_FRACTAL"].values
    dnf = enriched["DOWN_FRACTAL"].values

    # Precompute event list (k, dir) and fractal lists.
    events = [(k, (1 if be[k] > 0 else -1)) for k in range(n)
              if (not _isnan(be[k]) and be[k] != 0) or (not _isnan(ch[k]) and ch[k] != 0)]
    # a CHOCH bar is also a BOS_EVENT bar; direction from whichever is nonzero
    events = []
    for k in range(n):
        d = 0
        if not _isnan(ch[k]) and ch[k] != 0:
            d = 1 if ch[k] > 0 else -1
        elif not _isnan(be[k]) and be[k] != 0:
            d = 1 if be[k] > 0 else -1
        if d != 0:
            events.append((k, d))
    highs = [(k, float(upf[k])) for k in range(n) if not _isnan(upf[k])]
    lows = [(k, float(dnf[k])) for k in range(n) if not _isnan(dnf[k])]

    out = []
    ei = hi = li = 0
    for i in range(n):
        lo = max(0, i - window + 1)
        # def (a): most recent event within [lo, i]
        a = "indeterminate"
        last_ev = None
        for (k, d) in reversed(events):
            if k > i:
                continue
            if k < lo:
                break
            last_ev = (k, d)
            break
        if last_ev is not None:
            a = "bullish" if last_ev[1] > 0 else "bearish"

        # def (b): sequence of last two swing highs and last two swing lows in window
        wh = [v for (k, v) in highs if lo <= k <= i]
        wl = [v for (k, v) in lows if lo <= k <= i]
        b = "indeterminate"
        if len(wh) >= 2 and len(wl) >= 2:
            hh = wh[-1] > wh[-2]
            hl = wl[-1] > wl[-2]
            if hh and hl:
                b = "bullish"
            elif (not hh) and (not hl):
                b = "bearish"
            else:
                b = "indeterminate"  # mixed (HH+LL or LH+HL)
        out.append({
            "i": i, "ts": idx[i], "a": a, "b": b,
            "last_event_k": last_ev[0] if last_ev else None,
        })
    return out


def _isnan(x):
    return x != x


def analyse(tf, days=60):
    candles = load_candles(tf)
    enriched = enrich(candles)          # REAL engine, default FRACTAL_WINDOW=2
    idx = list(enriched.index)
    cutoff = idx[-1] - timedelta(days=days)
    start_i = next(i for i, t in enumerate(idx) if t >= cutoff)

    rows = rolling_defs(enriched)
    ev = [r for r in rows if r["i"] >= start_i]  # evaluation window = last `days`
    total = len(ev)

    det = {"a": 0, "b": 0}
    indet = {"a": 0, "b": 0}
    for r in ev:
        for key in ("a", "b"):
            if r[key] == "indeterminate":
                indet[key] += 1
            else:
                det[key] += 1

    # Agreement where BOTH determinate
    both_det = [r for r in ev if r["a"] in ("bullish", "bearish") and r["b"] in ("bullish", "bearish")]
    agree = sum(1 for r in both_det if r["a"] == r["b"])

    # Flip counts (bullish<->bearish transitions, ignoring indeterminate gaps)
    def flips(key):
        prev = None
        c = 0
        for r in ev:
            v = r[key]
            if v not in ("bullish", "bearish"):
                continue
            if prev is not None and v != prev:
                c += 1
            prev = v
        return c

    # divergences: both determinate & opposite
    divs = [r for r in both_det if r["a"] != r["b"]]

    # window question: distance (bars) from each bar back to last event, over eval window
    dists = []
    for r in ev:
        if r["last_event_k"] is not None:
            dists.append(r["i"] - r["last_event_k"])
    dists.sort()

    def pct(x):
        return round(100.0 * x / total, 1) if total else 0.0

    print(f"\n===== {tf}  (last {days} days, {total} bars evaluated) =====")
    print(f"bars covered: {ev[0]['ts']} .. {ev[-1]['ts']}")
    print(f"  def(a) last-event : determinate {det['a']} ({pct(det['a'])}%)  indeterminate {indet['a']} ({pct(indet['a'])}%)")
    print(f"  def(b) swing-seq  : determinate {det['b']} ({pct(det['b'])}%)  indeterminate {indet['b']} ({pct(indet['b'])}%)")
    print(f"  agreement (both determinate, n={len(both_det)}): {agree} agree, {len(both_det)-agree} disagree  -> {round(100*agree/len(both_det),1) if both_det else 0}%")
    print(f"  trend FLIPS over period: def(a)={flips('a')}   def(b)={flips('b')}")
    if dists:
        import statistics
        print(f"  bars-since-last-event: min {dists[0]}  median {int(statistics.median(dists))}  p90 {dists[int(0.9*len(dists))-1]}  max {dists[-1]}")
    print(f"  divergences (both determinate & OPPOSITE): {len(divs)}")
    for r in divs[:6]:
        print(f"     {r['ts']}  a={r['a']:8s} b={r['b']:8s}")
    return {
        "tf": tf, "total": total,
        "indet_a_pct": pct(indet["a"]), "indet_b_pct": pct(indet["b"]),
        "agree_pct": round(100*agree/len(both_det),1) if both_det else None,
        "flips_a": flips("a"), "flips_b": flips("b"),
        "divergences": len(divs),
        "div_samples": [{"ts": str(r["ts"]), "a": r["a"], "b": r["b"]} for r in divs[:6]],
    }


if __name__ == "__main__":
    res = {}
    for tf in ("H1", "M15"):
        res[tf] = analyse(tf, days=60)
    Path("scripts/audit/tr_1").mkdir(parents=True, exist_ok=True)
    Path("scripts/audit/tr_1/diag_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("\nwrote scripts/audit/tr_1/diag_results.json")
