"""Score de qualite agrege par fournisseur.

PONDERATIONS (modifiable ici, repris tel quel dans le rapport) :
  meches       35%  — fidelite high/low vs reference : 50% concordance aux
                      tolerances + 50% score MAE combine ou high/low pese 2x
                      open/close ; score MAE = 100*(1 - mae/(2*tol)), borne 0
  completude   25%  — % bougies attendues presentes
  validite     15%  — % bougies OHLC intrinsequement coherentes
  couverture   15%  — % des 80 symboles x 5 TF servis avec statut ok
  fraicheur     5%  — age de la derniere bougie: 100 si <= 2 TF, 0 a >= 20 TF
  fiabilite     5%  — 100*(1 - erreurs HTTP/requetes)

La reference (twelve_data) n'a pas de sous-score meches : son score global est
renormalise sur les composantes mesurables et marque "etalon" dans le rapport.
Les cellules "derived" (TF resample, ex. EODHD M15/H4) sont signalees mais pas
penalisees numeriquement — la penalite naturelle est dans les metriques.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from symbols import ALL_SYMBOLS, SYM_BY_NAME, TIMEFRAMES  # noqa: E402
from providers import REFERENCE  # noqa: E402

RESULTS_DIR = HERE / "results"

WEIGHTS = {
    "wick": 0.35,
    "completeness": 0.25,
    "validity": 0.15,
    "coverage": 0.15,
    "freshness": 0.05,
    "reliability": 0.05,
}
TOTAL_CELLS = len(ALL_SYMBOLS) * len(TIMEFRAMES)  # 400


def wick_cell_score(w, sym) -> float:
    tol_pts = (sym.tol / sym.point) if not sym.rel else None
    if sym.rel:
        # crypto: tolerance relative -> travailler en % (mae_pct vs tol%)
        tol_pct = sym.tol * 100
        mae_hl = (w["mae_high_pct"] + w["mae_low_pct"]) / 2
        mae_oc = (w["mae_open_pct"] + w["mae_close_pct"]) / 2
        mae = (2 * mae_hl + mae_oc) / 3
        mae_score = max(0.0, 100.0 * (1 - mae / (2 * tol_pct)))
    else:
        mae_hl = (w["mae_high_pts"] + w["mae_low_pts"]) / 2
        mae_oc = (w["mae_open_pts"] + w["mae_close_pts"]) / 2
        mae = (2 * mae_hl + mae_oc) / 3
        mae_score = max(0.0, 100.0 * (1 - mae / (2 * tol_pts)))
    return 0.5 * w["concordance_pct"] + 0.5 * mae_score


def freshness_score(bars: float) -> float:
    if bars <= 2:
        return 100.0
    if bars >= 20:
        return 0.0
    return 100.0 * (20 - bars) / 18


def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def main():
    metrics = json.loads((RESULTS_DIR / "metrics.json").read_text(encoding="utf-8"))
    scores = {}
    for name, cells in metrics.items():
        statuses = [c["status"] for c in cells.values()]
        n_ok = statuses.count("ok")
        tested = any(s not in ("no_key",) for s in statuses)
        if not tested or n_ok == 0:
            scores[name] = {"tested": n_ok > 0, "cells_ok": n_ok,
                            "status_counts": {s: statuses.count(s) for s in set(statuses)}}
            continue

        wick_scores, comp, valid, fresh = [], [], [], []
        reqs = errs = 0
        for cell, c in cells.items():
            sym = SYM_BY_NAME[cell.rsplit("_", 1)[0]]
            reqs += c.get("requests", 0)
            errs += c.get("errors_seen", 0)
            if c["status"] != "ok":
                continue
            comp.append(c.get("completeness_pct"))
            valid.append(c.get("validity_pct"))
            if c.get("freshness_bars") is not None:
                fresh.append(freshness_score(c["freshness_bars"]))
            if "wick" in c:
                wick_scores.append(wick_cell_score(c["wick"], sym))

        sub = {
            "wick": mean(wick_scores) if name != REFERENCE else None,
            "completeness": mean(comp),
            "validity": mean(valid),
            "coverage": 100.0 * n_ok / TOTAL_CELLS,
            "freshness": mean(fresh),
            "reliability": 100.0 * (1 - errs / reqs) if reqs else None,
        }
        avail = {k: v for k, v in sub.items() if v is not None}
        wsum = sum(WEIGHTS[k] for k in avail)
        total = sum(WEIGHTS[k] * v for k, v in avail.items()) / wsum if wsum else None
        scores[name] = {
            "tested": True,
            "is_reference": name == REFERENCE,
            "cells_ok": n_ok,
            "status_counts": {s: statuses.count(s) for s in set(statuses)},
            "subscores": {k: (round(v, 2) if v is not None else None)
                          for k, v in sub.items()},
            "wick_cells_scored": len(wick_scores),
            "total": round(total, 2) if total is not None else None,
        }

    out = {"weights": WEIGHTS, "total_cells": TOTAL_CELLS, "scores": scores}
    (RESULTS_DIR / "scores.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    for name, s in sorted(scores.items(),
                          key=lambda kv: -(kv[1].get("total") or -1)):
        print(f"  {name:16s} total={s.get('total')} ok={s.get('cells_ok', 0)}")
    print("-> results/scores.json")


if __name__ == "__main__":
    main()
