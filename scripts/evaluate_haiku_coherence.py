"""Évalue la cohérence des descriptions Haiku vs données structurées (Phase 3).

Lit data/validation/marketreadings_2026_06_06.json et applique 4 tests automatiques :
  Test A — Mention : les éléments importants (BOS/CHOCH/OB/FVG/news) sont-ils mentionnés ?
  Test B — Cohérence directionnelle : pas de contradiction haussier/baissier sur le trend.
  Test C — Forbidden tokens : 0 token interdit (OutputFilter chatbot canonique). DOIT être 100%.
  Test D — Pas d'invention : la description ne mentionne pas d'élément absent des données.

Sortie :
  - data/validation/haiku_coherence_results.json  (résultats détaillés)
  - impression console des métriques agrégées A/B/C/D + patterns

Usage : python scripts/evaluate_haiku_coherence.py
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.intelligence.chatbot.output_filter import OutputFilter  # noqa: E402

SRC = ROOT / "data" / "validation" / "marketreadings_2026_06_06.json"
OUT = ROOT / "data" / "validation" / "haiku_coherence_results.json"

# Lexiques de mention (formes normalisées, sans accent, minuscule)
MENTION = {
    "bos": ["bos", "cassure", "cassures", "casse", "rupture", "rompu", "break of structure",
            "structure cassee", "cassure de structure", "franchi", "franchissement"],
    "choch": ["choch", "changement de caractere", "change of character", "retournement",
              "renversement", "inversion"],
    "ob": ["order block", "order-block", "ob ", "bloc d'ordre", "zone de demande",
           "zone d'offre", "zone institutionnelle", "zone d'accumulation"],
    "fvg": ["fvg", "fair value", "fair-value", "desequilibre", "imbalance", "gap",
            "ecart de prix", "inefficience", "vide de liquidite"],
    "news": ["news", "evenement", "annonce", "publication", "economique", "calendrier"],
}

TREND_WORDS = {
    "bullish": ["haussier", "haussiere", "hausse", "bullish", "ascendant", "achat", "achete"],
    "bearish": ["baissier", "baissiere", "baisse", "bearish", "descendant", "vente"],
}


def _norm(text: str) -> str:
    low = text.lower().replace("'", "'")
    dec = unicodedata.normalize("NFKD", low)
    return "".join(ch for ch in dec if not unicodedata.combining(ch))


def _mentions(desc_norm: str, key: str) -> bool:
    return any(w in desc_norm for w in MENTION[key])


def _has_tag(tags: list[str], prefix: str) -> bool:
    return any(t.startswith(prefix) for t in tags)


def evaluate():
    data = json.loads(SRC.read_text(encoding="utf-8"))
    readings = data["readings"]
    of = OutputFilter()

    results = []
    # compteurs : (passed, applicable)
    A = {"bos": [0, 0], "choch": [0, 0], "ob": [0, 0], "fvg": [0, 0], "news": [0, 0]}
    B = [0, 0]   # [no_contradiction, applicable]
    C = [0, 0]   # [clean, total]
    D = [0, 0]   # [no_invention, total]

    for rec in readings:
        mr = rec["market_reading"]
        tags = mr["conditions"]["tags"]
        desc = rec["haiku_description"]
        dn = _norm(desc)
        regime = mr["regime"]
        events = mr["events"]
        rid = rec["candle_id"]

        flags = []

        # ---- Test A — mention ----
        a_detail = {}
        tag_map = {
            "bos": _has_tag(tags, "bos_recent_"),
            "choch": _has_tag(tags, "choch_recent_"),
            "ob": "ob_active" in tags,
            "fvg": "fvg_active" in tags,
            "news": bool(events.get("news_upcoming") or events.get("news_just_published")),
        }
        for key, present in tag_map.items():
            if present:
                A[key][1] += 1
                ok = _mentions(dn, key)
                A[key][0] += int(ok)
                a_detail[key] = "mentioned" if ok else "MISSING"
                if not ok:
                    flags.append(f"A:{key}_not_mentioned")

        # ---- Test B — cohérence directionnelle ----
        trend = regime["trend"]
        mtf_vals = set(regime.get("mtf_confluence", {}).values())
        mtf_conflict = ("bullish" in mtf_vals and "bearish" in mtf_vals)
        b_detail = "n/a"
        if trend in ("bullish", "bearish"):
            B[1] += 1
            opp = "bearish" if trend == "bullish" else "bullish"
            has_match = any(w in dn for w in TREND_WORDS[trend])
            has_opp = any(w in dn for w in TREND_WORDS[opp])
            # contradiction = mention du sens opposé sans le bon sens, hors contexte MTF mixte
            contradiction = has_opp and not has_match and not mtf_conflict
            B[0] += int(not contradiction)
            b_detail = "contradiction" if contradiction else "coherent"
            if contradiction:
                flags.append("B:directional_contradiction")

        # ---- Test C — forbidden tokens ----
        C[1] += 1
        res = of.check(desc)
        C[0] += int(not res.contaminated)
        c_detail = {"contaminated": res.contaminated,
                    "category": res.category,
                    "tokens": list(res.matched_tokens)}
        if res.contaminated:
            flags.append(f"C:forbidden:{res.category}:{','.join(res.matched_tokens)}")

        # ---- Test D — pas d'invention ----
        D[1] += 1
        invented = []
        for key in ("bos", "choch", "ob", "fvg"):
            if _mentions(dn, key) and not tag_map[key]:
                invented.append(key)
        # "gap"/"ecart" sont génériques → n'incriminer FVG que si le mot est spécifique
        D[0] += int(not invented)
        d_detail = invented
        if invented:
            flags.append("D:invented:" + ",".join(invented))

        results.append({
            "candle_id": rid, "instrument": rec["instrument"],
            "source": rec["description_source"], "description": desc,
            "tags": tags,
            "testA": a_detail, "testB": b_detail, "testC": c_detail, "testD": d_detail,
            "flags": flags,
        })

    def pct(p, n):
        return round(100.0 * p / n, 1) if n else None

    a_rate = {k: pct(v[0], v[1]) for k, v in A.items()}
    # taux A global = readings sans aucun élément manquant / readings avec ≥1 élément
    a_applicable = [r for r in results if any(
        v == "MISSING" or v == "mentioned" for v in r["testA"].values())]
    a_full = [r for r in a_applicable if "MISSING" not in r["testA"].values()]
    metrics = {
        "n": len(results),
        "A_per_element": a_rate,
        "A_full_coverage_pct": pct(len(a_full), len(a_applicable)) if a_applicable else None,
        "A_applicable": len(a_applicable),
        "B_coherence_pct": pct(B[0], B[1]), "B_applicable": B[1],
        "C_clean_pct": pct(C[0], C[1]),
        "D_no_invention_pct": pct(D[0], D[1]),
        "source_dist": {s: sum(1 for r in results if r["source"] == s)
                        for s in {r["source"] for r in results}},
    }

    OUT.write_text(json.dumps({"metrics": metrics, "results": results},
                              indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 64)
    print("PHASE 3 — COHÉRENCE HAIKU (n =", metrics["n"], ")")
    print("=" * 64)
    print("Test A — Mention par élément (passed/applicable %):")
    for k, v in A.items():
        print(f"  {k:6s}: {v[0]:2d}/{v[1]:2d}  ({a_rate[k]}%)")
    print(f"  → couverture COMPLÈTE (tous éléments mentionnés): "
          f"{metrics['A_full_coverage_pct']}% sur {metrics['A_applicable']} readings")
    print(f"Test B — Cohérence directionnelle: {C[0]}  {metrics['B_coherence_pct']}% "
          f"(sur {B[1]} applicables)")
    print(f"Test C — Sans forbidden token: {metrics['C_clean_pct']}%  (DOIT = 100%)")
    print(f"Test D — Sans invention: {metrics['D_no_invention_pct']}%")
    print("Source:", metrics["source_dist"])
    print()
    print("Readings flaggés:")
    for r in results:
        if r["flags"]:
            print(f"  #{r['candle_id']:2d} [{r['instrument']}/{r['source'][:6]}] {r['flags']}")
    print(f"\n→ {OUT.relative_to(ROOT)}")
    return metrics


if __name__ == "__main__":
    evaluate()
