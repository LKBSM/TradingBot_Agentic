"""Audit coverage CSV XAU (et EURUSD pour bonus) — Batch 0.0 pre-flight.

Calcule :
 - lignes totales, plage de dates, fraicheur
 - bars présents par année
 - bars attendus en heures de session active (Mon-Fri 07:00-21:00 UTC)
 - coverage % par année
 - gaps > 30 min en session active
 - comparaison 2019_2024 vs 2019_2026 sur intervalle commun

Sortie : tableau markdown sur stdout (capturé dans audits/2026-Q2/xau_coverage_audit.md).
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def df_to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = [
        "| " + " | ".join(str(v) for v in row) + " |"
        for row in df.itertuples(index=False, name=None)
    ]
    return "\n".join([header, sep, *rows])


CSV_FILES = {
    "XAU_2019_2024": DATA_DIR / "XAU_15MIN_2019_2024.csv",
    "XAU_2019_2025": DATA_DIR / "XAU_15MIN_2019_2025.csv",
    "XAU_2019_2026": DATA_DIR / "XAU_15MIN_2019_2026.csv",
    "XAU_Dukascopy_2025_2026": DATA_DIR / "XAU_15MIN_2025_2026_dukascopy.csv",
    "EURUSD_2019_2025": DATA_DIR / "EURUSD_15MIN_2019_2025.csv",
}


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    # Date column auto-detect
    date_col = next((c for c in df.columns if c.lower() in ("date", "datetime", "time", "timestamp")), df.columns[0])
    df["ts"] = pd.to_datetime(df[date_col], utc=False, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    return df


def expected_bars_session(start: pd.Timestamp, end: pd.Timestamp) -> int:
    """Count expected 15-min bars in active session (Mon-Fri 07:00-21:00 UTC).

    XAU effectively trades Sun 22:00 → Fri 21:00 UTC, but for coverage we
    benchmark against Mon-Fri 07:00-21:00 which is the active liquid window
    and matches eval_08 methodology.
    """
    idx = pd.date_range(start.floor("D"), end.ceil("D"), freq="15min", inclusive="left")
    mask = (idx.weekday < 5) & (idx.hour >= 7) & (idx.hour < 21)
    return int(mask.sum())


def coverage_by_year(df: pd.DataFrame, label: str) -> pd.DataFrame:
    df = df.copy()
    df["year"] = df["ts"].dt.year
    df["in_session"] = (df["ts"].dt.weekday < 5) & (df["ts"].dt.hour >= 7) & (df["ts"].dt.hour < 21)
    rows = []
    for year, g in df.groupby("year"):
        actual_session = int(g["in_session"].sum())
        year_start = pd.Timestamp(year=int(year), month=1, day=1)
        year_end = pd.Timestamp(year=int(year), month=12, day=31, hour=23, minute=59)
        expected_session = expected_bars_session(year_start, year_end)
        coverage = 100.0 * actual_session / expected_session if expected_session else float("nan")
        rows.append({
            "year": int(year),
            "total_bars": int(len(g)),
            "session_bars": actual_session,
            "expected_session": expected_session,
            "coverage_pct": round(coverage, 2),
        })
    return pd.DataFrame(rows)


def find_gaps(df: pd.DataFrame, max_gap_min: int = 30) -> int:
    """Count gaps > max_gap_min minutes in active session."""
    df = df[(df["ts"].dt.weekday < 5) & (df["ts"].dt.hour >= 7) & (df["ts"].dt.hour < 21)]
    diffs = df["ts"].diff().dt.total_seconds() / 60.0
    return int((diffs > max_gap_min).sum())


def main() -> int:
    print(f"# Audit coverage CSV — {pd.Timestamp.now('UTC').isoformat()}\n")
    print("Méthodologie : bars attendus = 15-min ticks dans la session active Mon-Fri 07:00-21:00 UTC.")
    print("Cible coverage MVP : ≥ 95 %.\n")

    summaries = {}
    for label, path in CSV_FILES.items():
        print(f"\n## {label} — `{path.name}`\n")
        if not path.exists():
            print(f"❌ Fichier absent : {path}\n")
            continue
        try:
            df = load(path)
        except Exception as exc:  # pragma: no cover - audit script
            print(f"❌ Erreur lecture : {exc}\n")
            continue

        print(f"- Lignes totales : **{len(df):,}**")
        print(f"- Plage : `{df['ts'].min()}` → `{df['ts'].max()}`")
        print(f"- Fraîcheur (jours depuis dernière bar) : {(pd.Timestamp.now('UTC').tz_localize(None) - df['ts'].max()).days}")

        cov = coverage_by_year(df, label)
        summaries[label] = cov
        print()
        print(df_to_markdown(cov))
        print()

        gaps = find_gaps(df)
        total_coverage_session = float(cov["session_bars"].sum()) / float(cov["expected_session"].sum()) * 100
        print(f"- Coverage agrégée (toutes années, session active) : **{total_coverage_session:.2f} %**")
        print(f"- Nb gaps > 30 min en session active : **{gaps}**")

    # Comparaison 2019_2024 vs 2019_2026 sur intervalle commun
    if "XAU_2019_2024" in summaries and "XAU_2019_2026" in summaries:
        print("\n## Comparaison XAU_2019_2024 vs XAU_2019_2026 sur 2019-2024\n")
        c24 = summaries["XAU_2019_2024"]
        c26 = summaries["XAU_2019_2026"]
        common = pd.merge(c24, c26, on="year", suffixes=("_2024", "_2026"))
        common = common[common["year"] <= 2024]
        print(df_to_markdown(common))

    print("\n---\n")
    print("**Décision A — Source XAU primaire** :")
    if "XAU_2019_2026" in summaries:
        agg = summaries["XAU_2019_2026"]
        coverage_2019_2025 = agg[agg["year"].between(2019, 2025)]
        if not coverage_2019_2025.empty:
            mean_cov = coverage_2019_2025["session_bars"].sum() / coverage_2019_2025["expected_session"].sum() * 100
            print(f"- XAU_2019_2026 coverage 2019-2025 = **{mean_cov:.2f} %**")
            if mean_cov >= 95.0:
                print("- ✅ ≥ 95 % → ADOPTÉ comme source XAU primaire.")
            else:
                print(f"- ⚠️ < 95 % → fallback `XAU_2019_2024.csv` + extension `Dukascopy 2025-2026` (licence flag).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
