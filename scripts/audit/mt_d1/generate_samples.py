"""MT-D1 Section 3/4/8 — build annotated sample set with context charts.

Engine READ-ONLY. Produces:
  images/EVT-*.png       (25 emitted events)
  images/NON-*.png       (25 swing-crossings that emitted NO event)
  images/EXTRA-*.png     (15 events a wick-crossing rule WOULD add)
  images/OPENCASE.png    (section 4)
  manifest.json          (machine index of every case)
"""
from __future__ import annotations

import json
import random
import sys
from datetime import timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, ".")
from scripts.audit.mt_d1 import harness as H
from scripts.audit.mt_d1 import variant_detect as V

OUT = Path("docs/audits/ECHANTILLON-DETECTION-2026-07-29")
IMG = OUT / "images"
IMG.mkdir(parents=True, exist_ok=True)
random.seed(20260729)

PRE, POST = 45, 18  # context bars before / after


def plot(enr, k, title, level=None, level2=None, mark_color="tab:blue", fname="", subtitle=""):
    lo = max(0, k - PRE); hi = min(len(enr), k + POST + 1)
    sub = enr.iloc[lo:hi]
    xs = mdates.date2num(sub.index.to_pydatetime())
    fig, ax = plt.subplots(figsize=(11, 5.2))
    w = (xs[1] - xs[0]) * 0.7 if len(xs) > 1 else 0.01
    for x, (_, r) in zip(xs, sub.iterrows()):
        up = r["close"] >= r["open"]
        col = "#26a69a" if up else "#ef5350"
        ax.plot([x, x], [r["low"], r["high"]], color=col, linewidth=0.8, zorder=1)
        ax.add_patch(plt.Rectangle((x - w / 2, min(r["open"], r["close"])), w,
                     max(abs(r["close"] - r["open"]), 1e-6), color=col, zorder=2))
    kx = mdates.date2num(enr.index[k].to_pydatetime())
    ax.axvline(kx, color=mark_color, linestyle="--", linewidth=1.3, alpha=0.9, zorder=3,
               label="bougie du cas")
    if level is not None:
        ax.axhline(level, color="#8e24aa", linewidth=1.1, alpha=0.8, label=f"niveau structurel {level:.2f}")
    if level2 is not None:
        ax.axhline(level2, color="#fb8c00", linewidth=1.0, alpha=0.7, linestyle=":", label=f"creux protégé {level2:.2f}")
    ax.set_title(title, fontsize=11, pad=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %Hh"))
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.2)
    fig.autofmt_xdate()
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    if subtitle:
        fig.text(0.5, 0.015, subtitle, ha="center", va="bottom", fontsize=8.5, color="#444")
    fig.savefig(IMG / fname, dpi=90)
    plt.close(fig)


def main():
    frames = {tf: H.enrich(H.load_candles(tf)) for tf in ("H1", "M15")}
    manifest = {"events": [], "non_events": [], "extra_events": [], "open_case": None}

    # ---------- Section 3A: 25 emitted events (stratified) ----------
    pool = []
    for tf, enr in frames.items():
        now = enr.index[-1]; cut = now - timedelta(days=60)
        for e in H.extract_events(enr):
            if enr.index[e["k"]] >= cut:
                pool.append((tf, e))
    # stratify by (type, direction) so both TFs and both signs appear
    random.shuffle(pool)
    buckets: dict = {}
    for tf, e in pool:
        buckets.setdefault((tf, e["type"], e["direction"]), []).append((tf, e))
    picked = []
    keys = list(buckets)
    random.shuffle(keys)
    i = 0
    while len(picked) < 25 and any(buckets.values()):
        kk = keys[i % len(keys)]
        if buckets[kk]:
            picked.append(buckets[kk].pop())
        i += 1
    for n, (tf, e) in enumerate(picked[:25], 1):
        enr = frames[tf]; cid = f"EVT-{n:02d}"
        fname = f"{cid}.png"
        ts = enr.index[e["k"]]
        plot(enr, e["k"],
             f"{cid} — {tf} — {e['type']} {e['direction']} @ {ts:%d/%m %Hh} UTC — niveau {e['level']:.2f}",
             level=e["level"], mark_color="tab:blue", fname=fname,
             subtitle="Événement ÉMIS par le moteur. Le niveau mauve est le creux/sommet structurel franchi (clôture).")
        manifest["events"].append({"id": cid, "tf": tf, "ts": str(ts), "type": e["type"],
                                   "direction": e["direction"], "level": round(e["level"], 2), "image": fname})

    # ---------- Section 3B: 25 non-events (blocked crossings) ----------
    npool = []
    for tf, enr in frames.items():
        c = enr["close"].values.astype(float); h = enr["high"].values.astype(float)
        l = enr["low"].values.astype(float)
        uf = enr["UP_FRACTAL"].values.astype(float); dfa = enr["DOWN_FRACTAL"].values.astype(float)
        atr = enr["ATR"].values.astype(float)
        res = V.detect(c, h, l, uf, dfa, atr)
        now = enr.index[-1]; cut = now - timedelta(days=60)
        for b in res["blocked"]:
            if enr.index[b["k"]] >= cut:
                npool.append((tf, b))
    random.shuffle(npool)
    nb = {}
    for tf, b in npool:
        nb.setdefault((b["reason"], b["side"]), []).append((tf, b))
    npick = []
    nkeys = list(nb); random.shuffle(nkeys); i = 0
    while len(npick) < 25 and any(nb.values()):
        kk = nkeys[i % len(nkeys)]
        if nb[kk]:
            npick.append(nb[kk].pop())
        i += 1
    reason_fr = {
        "wick_only_close_inside": "franchi en MÈCHE seulement — la clôture est restée en-deçà (règle : clôture requise)",
        "level_already_broken": "niveau déjà franchi — aucun nouveau creux/sommet formé depuis la dernière cassure (garde anti-répétition)",
    }
    for n, (tf, b) in enumerate(npick[:25], 1):
        enr = frames[tf]; cid = f"NON-{n:02d}"; fname = f"{cid}.png"
        ts = enr.index[b["k"]]
        lvl = b.get("structural_high", b.get("structural_low"))
        side = "sommet" if b["side"] == "up" else "creux"
        plot(enr, b["k"],
             f"{cid} — {tf} — franchissement de {side} SANS événement @ {ts:%d/%m %Hh} UTC",
             level=lvl, mark_color="tab:red", fname=fname,
             subtitle=f"Condition non remplie : {reason_fr[b['reason']]}")
        manifest["non_events"].append({"id": cid, "tf": tf, "ts": str(ts), "side": b["side"],
                                       "reason": b["reason"], "level": round(float(lvl), 2),
                                       "bar_high": b.get("bar_high"), "bar_low": b.get("bar_low"),
                                       "bar_close": b["bar_close"], "image": fname})

    # ---------- Section 8: 15 extra events under wick-crossing ----------
    # Most influential lever = cross_mode close→wick. Extra = bars where the wick
    # variant fires an event that the engine (close) does not.
    xpool = []
    for tf, enr in frames.items():
        c = enr["close"].values.astype(float); h = enr["high"].values.astype(float)
        l = enr["low"].values.astype(float)
        uf = enr["UP_FRACTAL"].values.astype(float); dfa = enr["DOWN_FRACTAL"].values.astype(float)
        atr = enr["ATR"].values.astype(float)
        base = V.detect(c, h, l, uf, dfa, atr, cross_mode="close")
        wick = V.detect(c, h, l, uf, dfa, atr, cross_mode="wick")
        base_bars = {k for k in range(len(enr)) if base["bos_event"][k] != 0 or base["choch_signal"][k] != 0}
        now = enr.index[-1]; cut = now - timedelta(days=60)
        for k in range(len(enr)):
            fired = wick["bos_event"][k] != 0 or wick["choch_signal"][k] != 0
            if fired and k not in base_bars and enr.index[k] >= cut:
                direction = "bullish" if (wick["bos_event"][k] > 0 or wick["choch_signal"][k] > 0) else "bearish"
                lvl = wick["bos_break_level"][k]
                xpool.append((tf, k, direction, float(lvl) if not np.isnan(lvl) else float(c[k])))
    random.shuffle(xpool)
    # spread across TFs
    xpool.sort(key=lambda t: (t[0],))  # group, then interleave
    inter = []
    by_tf = {"H1": [t for t in xpool if t[0] == "H1"], "M15": [t for t in xpool if t[0] == "M15"]}
    while len(inter) < 15 and (by_tf["H1"] or by_tf["M15"]):
        for tf in ("H1", "M15"):
            if by_tf[tf]:
                inter.append(by_tf[tf].pop())
            if len(inter) >= 15:
                break
    for n, (tf, k, direction, lvl) in enumerate(inter[:15], 1):
        enr = frames[tf]; cid = f"EXTRA-{n:02d}"; fname = f"{cid}.png"
        ts = enr.index[k]
        plot(enr, k,
             f"{cid} — {tf} — événement AJOUTÉ si franchissement en mèche — {direction} @ {ts:%d/%m %Hh} UTC",
             level=lvl, mark_color="tab:green", fname=fname,
             subtitle="N'EST PAS émis aujourd'hui (règle clôture). Le serait avec un franchissement en mèche. Vrai événement ou bruit ?")
        manifest["extra_events"].append({"id": cid, "tf": tf, "ts": str(ts), "direction": direction,
                                         "level": round(lvl, 2), "image": fname})

    # ---------- Section 4: open case ----------
    enr = frames["H1"]
    # mark the last BOS up on 24 Jul 23:00
    k_open = None
    for e in H.extract_events(enr):
        if str(enr.index[e["k"]]).startswith("2026-07-24 23"):
            k_open = e["k"]
    if k_open is None:
        k_open = list(enr.index).index(
            [t for t in enr.index if str(t).startswith("2026-07-27 08")][0])
    # widen context for the open case
    global PRE, POST
    PRE, POST = 20, 80
    plot(enr, k_open,
         "OPENCASE — XAUUSD H1 — BOS haussier 24/07 23h → dérive vers 28/07",
         level=4080.69, level2=4012.26, mark_color="tab:blue", fname="OPENCASE.png",
         subtitle="Creux protégé 4012.26 (orange) jamais clôturé sous → aucun CHOCH baissier (règle clôture). Plus bas clôturé 4043.43.")
    manifest["open_case"] = {"tf": "H1", "bos_ts": str(enr.index[k_open]),
                             "protected_low": 4012.26, "lowest_close": 4043.43,
                             "verdict": "drift_no_close_below_protected_low", "image": "OPENCASE.png"}

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print("events", len(manifest["events"]), "non_events", len(manifest["non_events"]),
          "extra", len(manifest["extra_events"]))


if __name__ == "__main__":
    main()
