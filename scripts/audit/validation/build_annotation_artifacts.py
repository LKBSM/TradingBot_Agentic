"""Construit les artefacts d'annotation manuelle (Phase 2) depuis le dataset JSON.

Lit data/validation/marketreadings_2026_06_06.json et produit :
  - docs/audits/VALIDATION_DATASET_2026_06_06.md        (liste des 60 candles)
  - docs/audits/MANUAL_ANNOTATION_TEMPLATE_2026_06_06.md (blocs founder + résultats algo)
  - docs/audits/SCORING_TEMPLATE_2026_06_06.csv          (tableau de scoring)

Usage : python scripts/audit/validation/build_annotation_artifacts.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "data" / "validation" / "marketreadings_2026_06_06.json"
DOC_DATASET = ROOT / "docs" / "audits" / "VALIDATION_DATASET_2026_06_06.md"
DOC_TEMPLATE = ROOT / "docs" / "audits" / "MANUAL_ANNOTATION_TEMPLATE_2026_06_06.md"
CSV_SCORING = ROOT / "docs" / "audits" / "SCORING_TEMPLATE_2026_06_06.csv"


def _algo_summary(reading: dict) -> dict:
    s = reading["structure"]
    r = reading["regime"]
    bos = s.get("bos")
    choch = s.get("choch")
    obs = s.get("order_blocks") or []
    fvgs = s.get("fair_value_gaps") or []
    return {
        "bos_dir": bos["direction"] if bos else None,
        "bos_status": bos["validation_status"] if bos else None,
        "choch_dir": choch["direction"] if choch else None,
        "ob_count": len(obs),
        "ob_importance": [o.get("importance") for o in obs],
        "fvg_count": len(fvgs),
        "trend": r["trend"],
        "vol": r["volatility_observed"],
        "phase": r["market_phase"],
        "mtf": r.get("mtf_confluence", {}),
    }


def build():
    data = json.loads(SRC.read_text(encoding="utf-8"))
    readings = data["readings"]
    meta = data["metadata"]

    # ---- 1. Dataset list ----
    lines = [
        "# VALIDATION_DATASET_2026_06_06 — Liste des 60 candles échantillonnées",
        "",
        "> Audit de validation algorithmique — Phase 2.2",
        f"> Généré pour : {meta['generated_for']} · TF : {meta['timeframe']} · lookback : {meta['lookback']}",
        "",
        "## Note data (écart documenté)",
        "",
        f"- **XAUUSD** : période {meta['periods'].get('XAUUSD')} (jan-mars 2026, comme spécifié).",
        f"- **EURUSD** : période {meta['periods'].get('EURUSD')}. {meta['data_note']}",
        "",
        "## Méthode d'échantillonnage",
        "",
        "Stratification par **état détecté par l'algo** (6 strates × 5 candles × 2 instruments).",
        "Sélection déterministe (équi-espacée par strate, pas de RNG). La stratification utilise",
        "l'état-algo lui-même afin de couvrir des conditions variées à faire valider manuellement.",
        "",
        "Composition par strate :",
        "",
        "| Instrument | bos_bull | bos_bear | choch | range | high_vol | ordinary |",
        "|---|---|---|---|---|---|---|",
    ]
    for inst, st in meta["strata"].items():
        c = st["composition"]
        lines.append(
            f"| {inst} | {c['bos_bull']} | {c['bos_bear']} | {c['choch']} | "
            f"{c['range']} | {c['high_vol']} | {c['ordinary']} |"
        )
    lines += [
        "",
        f"Total : {meta['total_candles']} candles · {meta['errors']} erreurs · "
        f"Haiku live : {meta['haiku_live']}",
        "",
        "## Liste complète",
        "",
        "| # | Instrument | Bar open (UTC) | Close ts (UTC) | Close | Trend | Vol | Phase | BOS | CHOCH | OB | FVG |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for rec in readings:
        a = _algo_summary(rec["market_reading"])
        lines.append(
            f"| {rec['candle_id']} | {rec['instrument']} | {rec['bar_open_ts']} | "
            f"{rec['candle_close_ts']} | {rec['close_price']:.5f} | {a['trend']} | "
            f"{a['vol']} | {a['phase']} | {a['bos_dir'] or '—'} | {a['choch_dir'] or '—'} | "
            f"{a['ob_count']} | {a['fvg_count']} |"
        )
    DOC_DATASET.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- 2. Annotation template ----
    t = [
        "# MANUAL_ANNOTATION_TEMPLATE_2026_06_06 — Annotation manuelle (founder)",
        "",
        "> Audit de validation algorithmique — Phase 2.4",
        "> **À compléter par le founder en regardant TradingView** (H1, même instrument, même heure UTC).",
        "> Le bloc « Détection algo MIA Markets » est pré-rempli pour comparaison directe.",
        "",
        "## Mode d'emploi",
        "",
        "1. Ouvre TradingView sur l'instrument + H1, va à la bougie indiquée (heure d'OUVERTURE UTC).",
        "2. Annote ce que TU vois (BOS/CHOCH/OB/FVG/phase).",
        "3. Compare au bloc algo. Coche le verdict.",
        "4. Reporte le verdict dans `SCORING_TEMPLATE_2026_06_06.csv` (colonnes `manual_*` + `verdict`).",
        "",
        "**Rappel niveaux** : les niveaux de prix BOS/CHOCH/OB/FVG affichés par l'algo sont des",
        "**proxies** (cf. findings F1-F3 de `STRUCTURE_DEFINITIONS_AUDIT.md`) — ne valide PAS les",
        "niveaux chiffrés, valide la **présence/direction/phase**.",
        "",
        "---",
        "",
    ]
    for rec in readings:
        a = _algo_summary(rec["market_reading"])
        mtf_str = ", ".join(f"{k}:{v}" for k, v in a["mtf"].items()) or "—"
        t += [
            f"### Candle #{rec['candle_id']} — {rec['instrument']} H1 — "
            f"open {rec['bar_open_ts']} (close {rec['candle_close_ts']}) — Close: {rec['close_price']:.5f}",
            "",
            "**À annoter manuellement (founder)** :",
            "",
            "- [ ] BOS bullish à cette clôture ? (Oui/Non) — niveau : ________",
            "- [ ] BOS bearish à cette clôture ? (Oui/Non) — niveau : ________",
            "- [ ] CHOCH ? (Oui/Non + direction) : ________",
            "- [ ] Nb Order Blocks actifs : ____ — niveaux : ________",
            "- [ ] Nb FVG actifs (non comblés) : ____ — niveaux : ________",
            "- [ ] Phase : Trend (bull/bear/neutral) ____ / Volatility (low/normal/elevated) ____ / "
            "Phase (trend/range/volatile/accumulation/expansion) ____",
            "",
            "**Détection algo MIA Markets** :",
            "",
            f"- BOS : **{a['bos_dir'] or 'aucun'}**" + (f" ({a['bos_status']})" if a["bos_dir"] else ""),
            f"- CHOCH : **{a['choch_dir'] or 'aucun'}**",
            f"- Order Blocks actifs : **{a['ob_count']}**" + (f" (importance {a['ob_importance']})" if a["ob_count"] else ""),
            f"- FVG actifs : **{a['fvg_count']}**",
            f"- Régime : trend=**{a['trend']}**, volatilité=**{a['vol']}**, phase=**{a['phase']}**",
            f"- MTF : {mtf_str}",
            f"- Tags : `{rec['market_reading']['conditions']['tags']}`",
            f"- Description ({rec['description_source']}) : *{rec['haiku_description']}*",
            "",
            "**Verdict founder** (à compléter) :",
            "",
            "- [ ] Algo correct  · [ ] Hallucination (FP)  · [ ] Loupé (FN)  · [ ] Approximatif",
            "- Notes : ____________________________________________",
            "",
            "---",
            "",
        ]
    DOC_TEMPLATE.write_text("\n".join(t), encoding="utf-8")

    # ---- 3. Scoring CSV ----
    with CSV_SCORING.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([
            "candle_id", "instrument", "bar_open_ts", "close_price",
            "algo_bos_bullish", "manual_bos_bullish",
            "algo_bos_bearish", "manual_bos_bearish",
            "algo_choch", "manual_choch",
            "algo_ob_count", "manual_ob_count",
            "algo_fvg_count", "manual_fvg_count",
            "algo_phase", "manual_phase",
            "verdict", "notes",
        ])
        for rec in readings:
            a = _algo_summary(rec["market_reading"])
            phase_str = f"{a['trend']}/{a['vol']}/{a['phase']}"
            w.writerow([
                rec["candle_id"], rec["instrument"], rec["bar_open_ts"],
                f"{rec['close_price']:.5f}",
                "Oui" if a["bos_dir"] == "bullish" else "Non", "",
                "Oui" if a["bos_dir"] == "bearish" else "Non", "",
                a["choch_dir"] or "Non", "",
                a["ob_count"], "",
                a["fvg_count"], "",
                phase_str, "",
                "", "",
            ])

    print(f"OK : {DOC_DATASET.name}, {DOC_TEMPLATE.name}, {CSV_SCORING.name}")


if __name__ == "__main__":
    build()
