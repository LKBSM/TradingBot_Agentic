"""Calcul des metriques par fournisseur x symbole x TF.

Sorties: results/metrics.json

Familles:
  couverture   : statut de la cellule (ok / not_covered / error / empty / no_key)
  completude   : % de bougies attendues presentes + 5 plus gros trous
                 grille attendue: crypto = 24/7 theorique ; fx/metaux = grille
                 24/5 theorique (week-end exclu ven 22:00 -> dim 22:00 UTC,
                 approximation DST documentee) ; indices/energie = timestamps
                 de la reference (sessions CFD non modelisees)
  validite     : % bougies OHLC coherentes (high>=max(o,c), low<=min(o,c),
                 valeurs > 0, range non aberrant)
  meches       : vs reference (jointure par timestamp UTC) — MAE/mediane/p95
                 des ecarts |high| et |low| en points ET en % ; open/close
                 suivis mais ponderes moitie moins (cf. scoring)
  concordance  : % bougies avec |dHigh|<=tol ET |dLow|<=tol (tol par symbole)
  fraicheur    : age de la derniere bougie a l'instant du fetch, en nb de TF
  fiabilite    : erreurs HTTP / requetes (au niveau fournisseur)

La reference (twelve_data) n'a pas de metriques "meches vs reference" :
c'est l'etalon, pas une verite terrain (marche OTC, pas de prix officiel).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from symbols import ALL_SYMBOLS, SYM_BY_NAME, TIMEFRAMES, TF_SECONDS  # noqa: E402
from providers import build_registry, REFERENCE  # noqa: E402

RAW_DIR = HERE / "data" / "raw"
RESULTS_DIR = HERE / "results"
UTC = timezone.utc


def load_cell(provider: str, cell: str) -> pd.DataFrame | None:
    p = RAW_DIR / provider / f"{cell}.csv"
    if not p.is_file():
        return None
    df = pd.read_csv(p, index_col="ts", parse_dates=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize(UTC)
    return df


def is_weekend(ts: pd.Timestamp) -> bool:
    """Fenetre week-end forex approx: ven 22:00 UTC -> dim 22:00 UTC."""
    wd, h = ts.weekday(), ts.hour
    return (wd == 4 and h >= 22) or wd == 5 or (wd == 6 and h < 22)


def anchored_range(start, end, step, anchor_off):
    """Grille au pas `step` ancree sur anchor_off (secondes depuis epoque % step).

    Les ancres H4/D1 varient par fournisseur ET par classe d'actif (ex.
    Twelve Data: H4 forex ancre session NY 01/05/09h UTC, H4 crypto ancre
    epoque 00/04/08h). On infere donc l'ancre du fournisseur lui-meme pour
    mesurer SES trous internes, pas sa convention d'ancrage."""
    first = int(-(-(start.timestamp() - anchor_off) // step) * step + anchor_off)
    return pd.date_range(pd.Timestamp(first, unit="s", tz=UTC), end,
                         freq=f"{step}s", inclusive="left", tz=UTC)


def inferred_anchor(df, step) -> int:
    # (index - epoque).total_seconds() : independant de l'unite interne
    # de l'index (read_csv produit du datetime64[us], pas du [ns])
    secs = (df.index - pd.Timestamp(0, tz=UTC)).total_seconds()
    offs = pd.Series(secs % step)
    return int(offs.mode().iloc[0])


def expected_grid(sym, tf, df, start, end, fetched_at):
    """Grille attendue. None pour indices/energie (sessions CFD non modelisees:
    la completude y est remplacee par la liste des trous intrinseques)."""
    step = TF_SECONDS[tf]
    # ne jamais attendre la bougie encore en formation au moment du fetch
    cap = min(end, fetched_at - pd.Timedelta(seconds=step)) if fetched_at is not None else end
    if sym.cls == "crypto":
        return anchored_range(start, cap, step, 0)
    if sym.cls in ("index", "energy"):
        return None
    grid = anchored_range(start, cap, step, inferred_anchor(df, step))
    if tf == "D1":
        return grid[grid.weekday < 5]
    return grid[[not is_weekend(t) for t in grid]]


def intrinsic_gaps(df, step):
    """Trous internes bruts (diff entre bougies consecutives > 1 pas),
    week-ends exclus. Pour les classes sans grille attendue."""
    if len(df) < 2:
        return []
    diffs = df.index[1:] - df.index[:-1]
    gaps = []
    for i, d in enumerate(diffs):
        n_missing = int(d.total_seconds() // step) - 1
        if n_missing >= 1 and not is_weekend(df.index[i] + pd.Timedelta(seconds=step)):
            gaps.append({"from": df.index[i].isoformat(),
                         "to": df.index[i + 1].isoformat(), "bars": n_missing})
    gaps.sort(key=lambda g: -g["bars"])
    return gaps[:5]


def completeness(df, grid):
    if grid is None or len(grid) == 0:
        return None, []
    present = df.index.intersection(grid)
    missing = grid.difference(df.index)
    pct = 100.0 * len(present) / len(grid)
    gaps = []
    if len(missing):
        miss = missing.sort_values()
        # regrouper les timestamps manquants contigus en trous
        groups = []
        run_start = miss[0]
        prev = miss[0]
        step = pd.Timedelta((grid[1] - grid[0])) if len(grid) > 1 else pd.Timedelta(0)
        count = 1
        for t in miss[1:]:
            if step and (t - prev) <= step:
                count += 1
            else:
                groups.append((run_start, prev, count))
                run_start, count = t, 1
            prev = t
        groups.append((run_start, prev, count))
        groups.sort(key=lambda g: -g[2])
        gaps = [{"from": g[0].isoformat(), "to": g[1].isoformat(), "bars": int(g[2])}
                for g in groups[:5]]
    return round(pct, 2), gaps


def validity(df):
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    ok = (h >= np.maximum(o, c) - 1e-12) & (l <= np.minimum(o, c) + 1e-12) \
        & (o > 0) & (h > 0) & (l > 0) & (c > 0) & h.notna() & l.notna()
    rng = (h - l).abs()
    med = rng.median()
    if med and med > 0:
        ok &= rng <= 50 * med  # bougie aberrante: range > 50x la mediane
    n_bad = int((~ok).sum())
    worst = []
    if n_bad:
        bad = df[~ok].head(5)
        worst = [{"ts": t.isoformat()} for t in bad.index]
    return round(100.0 * ok.mean(), 3), n_bad, worst


def epoch_resample(df, tf):
    """Resample epoque-aligne (minuit UTC) — meme convention que le
    resample_ohlcv du produit. Utilise pour comparer les meches H4/D1 entre
    fournisseurs dont les ancres natives different."""
    rule = {"H4": "4h", "D1": "1D"}[tf]
    out = df.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"})
    return out.dropna(how="all")


def wick_vs_reference(df, ref_df, sym):
    joined = df.join(ref_df, how="inner", rsuffix="_ref")
    if len(joined) < 10:
        return None
    out = {"aligned_bars": int(len(joined))}
    ref_close = joined["close_ref"].abs()
    for col in ["high", "low", "open", "close"]:
        d = (joined[col] - joined[f"{col}_ref"]).abs()
        out[f"mae_{col}_pts"] = float((d / sym.point).mean())
        out[f"med_{col}_pts"] = float((d / sym.point).median())
        out[f"p95_{col}_pts"] = float((d / sym.point).quantile(0.95))
        out[f"mae_{col}_pct"] = float((d / ref_close * 100).mean())
    tol = sym.tol * ref_close if sym.rel else sym.tol
    dh = (joined["high"] - joined["high_ref"]).abs()
    dl = (joined["low"] - joined["low_ref"]).abs()
    out["concordance_pct"] = float(100.0 * ((dh <= tol) & (dl <= tol)).mean())
    worst_metric = np.maximum(dh, dl)
    top = worst_metric.nlargest(5)
    out["worst_cases"] = [
        {"ts": t.isoformat(),
         "d_high_pts": round(float(dh.loc[t] / sym.point), 2),
         "d_low_pts": round(float(dl.loc[t] / sym.point), 2),
         "ref_high": float(joined.loc[t, "high_ref"]),
         "prov_high": float(joined.loc[t, "high"]),
         "ref_low": float(joined.loc[t, "low_ref"]),
         "prov_low": float(joined.loc[t, "low"])}
        for t in top.index]
    return out


def main():
    run_meta = json.loads((RESULTS_DIR / "run_meta.json").read_text(encoding="utf-8"))
    start = pd.Timestamp(run_meta["start"])
    end = pd.Timestamp(run_meta["end"])
    registry = build_registry()
    all_metrics = {}

    ref_cache = {}

    def ref_df_for(cell):
        if cell not in ref_cache:
            ref_cache[cell] = load_cell(REFERENCE, cell)
        return ref_cache[cell]

    for name in registry:
        meta_p = RESULTS_DIR / f"fetch_meta_{name}.json"
        if not meta_p.is_file():
            continue
        fetch_meta = json.loads(meta_p.read_text(encoding="utf-8"))
        prov_metrics = {}
        for cell, m in fetch_meta.items():
            sym_name, tf = cell.rsplit("_", 1)
            if sym_name not in SYM_BY_NAME or tf not in TIMEFRAMES:
                continue
            sym = SYM_BY_NAME[sym_name]
            entry = {"status": m["status"], "error": m.get("error"),
                     "derived": m.get("derived", False),
                     "requests": m.get("requests", 0),
                     "errors_seen": m.get("errors_seen", 0)}
            if m["status"] == "ok":
                df = load_cell(name, cell)
                if df is None or df.empty:
                    entry["status"] = "empty"
                else:
                    fetched = pd.Timestamp(m["fetched_at"]) if m.get("fetched_at") else None
                    # fenetre par cellule : les reprises de cache ont des instants
                    # de fetch differents ; la grille attendue suit chaque cellule
                    cell_start = (fetched - pd.Timedelta(days=run_meta["days"])
                                  ) if fetched is not None else start
                    grid = expected_grid(sym, tf, df, cell_start, end, fetched)
                    if grid is None:
                        entry["completeness_pct"] = None
                        entry["top_gaps"] = intrinsic_gaps(df, TF_SECONDS[tf])
                    else:
                        comp, gaps = completeness(df, grid)
                        entry["completeness_pct"], entry["top_gaps"] = comp, gaps
                    v_pct, v_bad, v_worst = validity(df)
                    entry["validity_pct"], entry["invalid_bars"] = v_pct, v_bad
                    if v_worst:
                        entry["invalid_examples"] = v_worst
                    if m.get("last_ts") and m.get("fetched_at"):
                        age = (pd.Timestamp(m["fetched_at"]) -
                               pd.Timestamp(m["last_ts"])).total_seconds()
                        entry["freshness_bars"] = round(age / TF_SECONDS[tf], 2)
                    if name != REFERENCE:
                        w = None
                        if tf in ("H4", "D1"):
                            # ancres H4/D1 heterogenes -> comparaison sur bougies
                            # derivees des H1 des deux cotes, alignees epoque
                            prov_h1 = load_cell(name, f"{sym_name}_H1")
                            ref_h1 = ref_df_for(f"{sym_name}_H1")
                            if prov_h1 is not None and len(prov_h1) and \
                                    ref_h1 is not None and len(ref_h1):
                                w = wick_vs_reference(epoch_resample(prov_h1, tf),
                                                      epoch_resample(ref_h1, tf), sym)
                                if w:
                                    w["derived_epoch_aligned"] = True
                        if w is None:
                            ref_df = ref_df_for(cell)
                            if ref_df is not None and len(ref_df):
                                w = wick_vs_reference(df, ref_df, sym)
                        if w:
                            entry["wick"] = w
            prov_metrics[cell] = entry
        all_metrics[name] = prov_metrics
        n_ok = sum(1 for e in prov_metrics.values() if e["status"] == "ok")
        print(f"[{name}] metriques calculees ({n_ok} cellules ok)")

    (RESULTS_DIR / "metrics.json").write_text(
        json.dumps(all_metrics, indent=1, ensure_ascii=False), encoding="utf-8")
    print("-> results/metrics.json")


if __name__ == "__main__":
    main()
