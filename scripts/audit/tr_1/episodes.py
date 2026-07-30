"""TR-1 — extract 3 dated divergence EPISODES with their structural cause.
READ ONLY. Reuses diag.rolling_defs on the real engine output.
"""
from __future__ import annotations
import sys
from datetime import timedelta
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.audit.mt_d1.harness import load_candles, enrich
from scripts.audit.tr_1.diag import rolling_defs, WINDOW, _isnan


def episodes(tf, days=60, want=3):
    candles = load_candles(tf)
    enriched = enrich(candles)
    idx = list(enriched.index)
    upf = enriched["UP_FRACTAL"].values
    dnf = enriched["DOWN_FRACTAL"].values
    be = enriched["BOS_EVENT"].values
    ch = enriched["CHOCH_SIGNAL"].values
    cutoff = idx[-1] - timedelta(days=days)
    start_i = next(i for i, t in enumerate(idx) if t >= cutoff)
    rows = rolling_defs(enriched)

    # group consecutive both-determinate-opposite bars into episodes
    eps = []
    cur = None
    for r in rows:
        if r["i"] < start_i:
            continue
        opp = (r["a"] in ("bullish", "bearish") and r["b"] in ("bullish", "bearish") and r["a"] != r["b"])
        if opp:
            if cur is None:
                cur = [r]
            else:
                cur.append(r)
        else:
            if cur:
                eps.append(cur); cur = None
    if cur:
        eps.append(cur)
    eps.sort(key=len, reverse=True)

    print(f"\n===== {tf}: {len(eps)} divergence episodes (both determinate & opposite), longest first =====")
    for ep in eps[:want]:
        mid = ep[len(ep)//2]
        i = mid["i"]
        lo = max(0, i - WINDOW + 1)
        # def(a) cause: last event
        la = None
        for k in range(i, lo-1, -1):
            d = 0
            if not _isnan(ch[k]) and ch[k] != 0:
                d = 1 if ch[k] > 0 else -1
            elif not _isnan(be[k]) and be[k] != 0:
                d = 1 if be[k] > 0 else -1
            if d != 0:
                la = (idx[k], "bullish" if d > 0 else "bearish"); break
        wh = [(idx[k], float(upf[k])) for k in range(lo, i+1) if not _isnan(upf[k])]
        wl = [(idx[k], float(dnf[k])) for k in range(lo, i+1) if not _isnan(dnf[k])]
        dur = ep[-1]["ts"] - ep[0]["ts"]
        print(f"\n  EPISODE {ep[0]['ts']} -> {ep[-1]['ts']}  ({len(ep)} bars, {dur})")
        print(f"    def(a)={mid['a']}  <- dernier evenement: {la[1] if la else '?'} le {la[0] if la else '?'}")
        print(f"    def(b)={mid['b']}  <- 2 derniers sommets: {[ (str(t)[:16], round(v,2)) for t,v in wh[-2:] ]}")
        print(f"                          2 derniers creux : {[ (str(t)[:16], round(v,2)) for t,v in wl[-2:] ]}")


if __name__ == "__main__":
    for tf in ("H1", "M15"):
        episodes(tf, days=60, want=3)
