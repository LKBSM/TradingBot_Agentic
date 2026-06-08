"""Aggregate per-block eval outputs into the single audit JSON contract."""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent.parent.parent / "docs" / "audits" / "data"
OUT = HERE.parent.parent.parent / "docs" / "audits" / "descriptive_quality_data.json"


BLOCKS = [
    ("bos", "BOS / break_level", 3),
    ("choch", "CHOCH", 3),
    ("fvg", "FVG", 3),
    ("ob", "OB", 3),
    ("retest", "Retest state machine", 3),
    ("calendar", "Calendar / blackout", 1),
    ("metadata", "Metadata sanity", 0.5),
    ("hmm", "HMM regime", 2),
    ("bocpd", "BOCPD cp_prob", 1),
    ("jump", "Jump ratio", 1),
    ("volatility", "HAR-RV + conformal CI", 2),
]


def main():
    out = {
        "audit_date": "2026-05-27",
        "audit_scope": "descriptive_quality",
        "train_window": "2019-01-01 .. 2023-12-31",
        "oos_window": "2024-01-01 .. 2026-04 (XAU) / 2025-12 (EUR)",
        "instruments": ["XAUUSD", "EURUSD"],
        "timeframe": "M15",
        "thresholds": {
            "f1_green": 0.85, "f1_yellow_min": 0.65,
            "ece_green": 0.05, "ece_yellow_max": 0.10,
            "picp_green_pp": 2, "picp_yellow_pp": 5,
        },
        "weights_by_block": {b[0]: b[2] for b in BLOCKS},
        "results": {},
    }
    for slug, name, weight in BLOCKS:
        p = DATA_DIR / f"{slug}_eval.json"
        if p.exists():
            out["results"][slug] = {
                "block_name": name,
                "weight": weight,
                "data": json.loads(p.read_text(encoding="utf-8")),
            }
        else:
            out["results"][slug] = {
                "block_name": name,
                "weight": weight,
                "data": {"missing": True},
            }
    OUT.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"Saved: {OUT}")
    print(f"Blocks consolidated: {len(out['results'])}")


if __name__ == "__main__":
    main()
