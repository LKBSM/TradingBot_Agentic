"""Genere docs/audits/data-benchmark-report.md a partir de results/*.json.

Honnetete du rapport :
- pas de "vrai prix" sur l'OTC : classement = proximite a la reference +
  completude + coherence, la reference etant elle-meme un agregat ;
- fournisseur sans cle = "non teste", jamais note ;
- les pires ecarts de meches sont listes avec timestamps pour verification
  visuelle.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from symbols import SYM_BY_NAME, TIMEFRAMES  # noqa: E402
from providers import PROVIDER_NOTES, REFERENCE  # noqa: E402

RESULTS_DIR = HERE / "results"
REPORT_PATH = HERE.parent.parent / "docs" / "audits" / "data-benchmark-report.md"

# Prix d'affichage commercial estimes (recherche 2026-07-05, cf. synthese) —
# utilises pour le croisement qualite x prix. "None" = sur devis / non publie.
DISPLAY_PRICE_USD = {
    "twelve_data": ("499 $/mois (414 $ annuel, plan Venture)", 499),
    "oanda": ("sur devis (licence API = interne only)", None),
    "tiingo": ("~250 $/mois (display startup publie)", 250),
    "fcsapi": ("149-329 $/mois SI display confirme par ecrit", 239),
    "itick": ("79-319 $/mois + avenant display ecrit requis", 199),
    "alltick": ("99-199 $/mois + avenant display ecrit requis", 149),
    "finazon": ("des 19 $/mois, redistribution incluse", 19),
    "eodhd": ("sur devis (>= 399 $/mois)", None),
    "fmp": ("sur devis (Data Display Agreement)", None),
    "massive_polygon": ("Business non publie (~4 chiffres/mois)", None),
    "tradermade": ("~L599+/mois par feed (~L1200 total)", None),
    "finage": ("599-1450 $/mois, redistribution interdite par disclaimer", None),
    "finnhub": ("sur devis (ancre 3500 $/mois)", None),
    "alpha_vantage": ("sur devis (tiers publies = personal use)", None),
}


def fmt(v, nd=1):
    return "—" if v is None else f"{v:.{nd}f}"


def main():
    scores = json.loads((RESULTS_DIR / "scores.json").read_text(encoding="utf-8"))
    metrics = json.loads((RESULTS_DIR / "metrics.json").read_text(encoding="utf-8"))
    run_meta = json.loads((RESULTS_DIR / "run_meta.json").read_text(encoding="utf-8"))
    research = (HERE / "research_synthesis.md").read_text(encoding="utf-8")

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    L = []
    L.append("# Banc d'essai qualité des fournisseurs de données de marché")
    L.append("")
    L.append(f"_Généré le {now} par `tools/data-benchmark/report.py` — relançable "
             f"(`python runner.py && python metrics.py && python scoring.py && "
             f"python report.py`)._")
    L.append("")
    L.append(f"Fenêtre testée : **{run_meta['days']} jours** "
             f"({run_meta['start'][:10]} → {run_meta['end'][:10]}), "
             f"80 symboles × {len(run_meta['tfs'])} TF ({', '.join(run_meta['tfs'])}).")
    L.append("")
    L.append("## Avertissement de méthode")
    L.append("")
    L.append("Sur les marchés OTC il n'existe **pas de prix officiel unique** : chaque feed "
             "est l'agrégat d'un panel de contributeurs ou le book d'un broker. La référence "
             f"du banc (`{REFERENCE}`) est elle-même un agrégat. Le classement se lit donc "
             "« le plus proche de ma référence + le plus complet + le plus cohérent », "
             "jamais « le vrai prix ». Un fournisseur sans clé API est marqué **non testé** — "
             "aucune donnée n'est simulée.")
    L.append("")

    # ---- classement ---------------------------------------------------------
    L.append("## Classement (fournisseurs testés)")
    L.append("")
    weights = scores["weights"]
    L.append("Pondérations (éditables dans `scoring.py`) : "
             + ", ".join(f"{k} {int(v*100)}%" for k, v in weights.items()) + ".")
    L.append("")
    L.append("| Rang | Fournisseur | Score global | Mèches | Complétude | Validité OHLC "
             "| Couverture | Fraîcheur | Fiabilité | Cellules OK/400 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    tested = {k: v for k, v in scores["scores"].items() if v.get("total") is not None}
    ranked = sorted(tested.items(), key=lambda kv: -kv[1]["total"])
    for i, (name, s) in enumerate(ranked, 1):
        sub = s["subscores"]
        label = f"**{name}**" + (" _(étalon)_" if s.get("is_reference") else "")
        L.append(f"| {i} | {label} | **{fmt(s['total'])}** | {fmt(sub['wick'])} | "
                 f"{fmt(sub['completeness'])} | {fmt(sub['validity'])} | "
                 f"{fmt(sub['coverage'])} | {fmt(sub['freshness'])} | "
                 f"{fmt(sub['reliability'])} | {s['cells_ok']} |")
    L.append("")
    not_tested = [k for k, v in scores["scores"].items() if v.get("total") is None]
    if not_tested:
        L.append("**Non testés (pas de clé API fournie — jamais simulés)** : "
                 + ", ".join(f"`{n}`" for n in sorted(not_tested)) + ".")
        L.append("")

    # ---- detail par fournisseur --------------------------------------------
    L.append("## Détail par fournisseur")
    L.append("")
    for name, s in ranked:
        L.append(f"### {name}")
        L.append("")
        L.append(f"_{PROVIDER_NOTES.get(name, '')}_")
        L.append("")
        L.append(f"Statuts cellules : `{s['status_counts']}`")
        cells = metrics.get(name, {})
        # non couverts, agreges par symbole
        nc_syms = sorted({c.rsplit('_', 1)[0] for c, e in cells.items()
                          if e["status"] == "not_covered"})
        if nc_syms:
            L.append("")
            L.append(f"Symboles non couverts ({len(nc_syms)}) : "
                     + ", ".join(nc_syms[:40])
                     + (" …" if len(nc_syms) > 40 else ""))
        errs = [(c, e.get("error", "")) for c, e in cells.items() if e["status"] == "error"]
        if errs:
            L.append("")
            L.append(f"Échecs ({len(errs)}) — 5 premiers :")
            for c, msg in errs[:5]:
                L.append(f"- `{c}` : {str(msg)[:140]}")
        derived = sorted({c for c, e in cells.items() if e.get("derived")})
        if derived:
            L.append("")
            L.append(f"⚠️ TF **dérivés par resampling** (non natifs) : {len(derived)} cellules "
                     f"(ex. {', '.join(derived[:6])}).")
        # pires ecarts de meches
        worst_all = []
        for c, e in cells.items():
            for w in e.get("wick", {}).get("worst_cases", []):
                worst_all.append((max(abs(w["d_high_pts"]), abs(w["d_low_pts"])), c, w))
        worst_all.sort(key=lambda x: -x[0])
        if worst_all:
            L.append("")
            L.append("Pires écarts de mèches vs référence (à vérifier à l'œil) :")
            L.append("")
            L.append("| Symbole×TF | Timestamp (UTC) | ΔHigh (pts) | ΔLow (pts) | "
                     "High réf → fournisseur | Low réf → fournisseur |")
            L.append("|---|---|---|---|---|---|")
            for _, c, w in worst_all[:8]:
                L.append(f"| {c} | {w['ts']} | {w['d_high_pts']} | {w['d_low_pts']} | "
                         f"{w['ref_high']} → {w['prov_high']} | "
                         f"{w['ref_low']} → {w['prov_low']} |")
        # plus gros trous
        gap_rows = []
        for c, e in cells.items():
            for g in e.get("top_gaps", [])[:2]:
                gap_rows.append((g["bars"], c, g))
        gap_rows.sort(key=lambda x: -x[0])
        if gap_rows:
            L.append("")
            L.append("Plus gros trous de complétude : "
                     + " ; ".join(f"`{c}` {g['bars']} barres dès {g['from'][:16]}"
                                  for _, c, g in gap_rows[:5]) + ".")
        L.append("")

    # ---- croisement qualite x prix -----------------------------------------
    L.append("## Croisement qualité × prix d'affichage commercial")
    L.append("")
    L.append("| Fournisseur | Score qualité | Prix display commercial (recherche 2026-07-05) |")
    L.append("|---|---|---|")
    order = ranked + [(n, scores["scores"][n]) for n in sorted(not_tested)]
    for name, s in order:
        price = DISPLAY_PRICE_USD.get(name, ("?", None))[0]
        total = fmt(s.get("total")) if s.get("total") is not None else "non testé"
        L.append(f"| {name} | {total} | {price} |")
    L.append("")
    L.append("_Les recommandations mono/bi-fournisseur sont rédigées dans la section "
             "conclusion du rapport au vu des scores mesurés — voir en bas de fichier._")
    L.append("")

    # ---- synthese recherche --------------------------------------------------
    L.append("---")
    L.append("")
    L.append(research)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(L), encoding="utf-8")
    print(f"-> {REPORT_PATH}")


if __name__ == "__main__":
    main()
