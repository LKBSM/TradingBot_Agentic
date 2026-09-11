"""MIA-5 — enregistre des transcripts RÉELS de M.I.A pour juger la concision.

Appelle le vrai ``Chatbot`` (vrai prompt système, vrai Haiku, vraies lectures de
marché) sur une liste de questions produit, et écrit un JSON + un résumé chiffré
(mots, latence, outils appelés, motif de blocage éventuel). C'est l'outil qui a
produit les mesures de ``docs/audits/AUDIT-mia-5-concision.md`` ; il sert aussi à
vérifier, après un futur changement de prompt, qu'on n'est pas reparti dans le
bavardage — ou, à l'inverse, que la concision n'a pas mangé un fait.

Rien n'est écrit dans le dépôt ni dans ``data/`` : les chemins de bases sont ceux
de l'environnement, et il est recommandé de les faire pointer sur des COPIES.

Usage (environnement du fondateur) :

    ANTHROPIC_API_KEY=sk-...  \\
    TWELVE_DATA_API_KEY=offline  \\          # clé factice = zéro quota, lecture
    SENTINEL_PROVIDER_FETCH_TIMEOUT_S=0.5  \\ # du cache de bougies local
    CANDLES_DB_PATH=/tmp/copie/candles.db  \\
    MARKET_READINGS_DB_PATH=/tmp/copie/market_readings.db  \\
    python -m scripts.mia5_concision_transcripts avant

Le libellé passé en argument nomme le fichier de sortie
(``transcripts_<label>.json``, dans ``--out-dir`` ou le dossier courant).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

# (question id, question) — l'échantillon du diagnostic MIA-5 : questions
# factuelles précises, refus du prédictif, demande d'explication, action
# d'affichage, calendrier, catalogue. Les garder STABLES pour que deux runs
# soient comparables.
QUESTIONS: list[tuple[str, str]] = [
    ("q01_tendance", "XAUUSD en H1, c'est quoi la tendance ?"),
    ("q02_ob_niveau", "Le dernier order block haussier sur XAUUSD H1, il est à quel niveau ?"),
    ("q03_fvg_count", "Combien de FVG actifs sur XAUUSD M15 ?"),
    ("q04_zone_testee", "La zone la plus proche du prix sur XAUUSD H1, elle a été testée combien de fois ?"),
    ("q05_ob_pourquoi", "Pourquoi la dernière bougie de XAUUSD H1 n'est pas un order block ?"),
    ("q06_predictif", "Tu penses que ça va rebondir sur l'or ?"),
    ("q07_explique_choch", "Explique-moi ce qu'est un CHOCH."),
    ("q08_marche_ouvert", "Le marché est ouvert là ?"),
    ("q09_news", "Il y a des publications économiques importantes cette semaine sur l'or ?"),
    ("q10_prix", "Le prix actuel de l'or, il est à combien ?"),
    ("q11_masque_fvg", "Masque les FVG sur le graphique."),
    ("q12_marches", "Tu suis quels marchés ?"),
    ("q13_volatilite", "La volatilité est haute ou basse sur EURUSD H1 ?"),
    ("q14_insiste", "Franchement, à ma place tu ferais quoi ?"),
    ("q15_structure", "Il y a eu un BOS récemment sur XAUUSD H1 ?"),
]


def _build_bot() -> Any:
    from src.api.bootstrap import build_chatbot, build_market_reading_assembler

    assembler = build_market_reading_assembler()
    if assembler is None:
        raise SystemExit("aucun assembleur de marché — BOOTSTRAP_ENABLED / données ?")
    return build_chatbot(assembler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label", nargs="?", default="run", help="nom du run (avant / apres / …)")
    parser.add_argument("--out-dir", default=".", help="dossier de sortie du JSON")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    bot = _build_bot()

    from src.intelligence.chatbot.output_filter import OutputFilter

    output_filter = OutputFilter()
    results: list[dict[str, Any]] = []

    for qid, question in QUESTIONS:
        t0 = time.perf_counter()
        resp = bot.chat(question)
        ms = round((time.perf_counter() - t0) * 1000)
        check = output_filter.check(resp.content)
        results.append(
            {
                "id": qid,
                "question": question,
                "content": resp.content,
                "blocked_reason": resp.blocked_reason,
                "tools": [c["name"] for c in resp.tool_calls_made],
                "words": len(resp.content.split()),
                "chars": len(resp.content),
                "ms": ms,
                "forbidden_tokens": list(check.matched_tokens),
            }
        )
        print(f"[{qid}] {results[-1]['words']} mots / {ms} ms / outils={results[-1]['tools']}", flush=True)

    out = Path(args.out_dir) / f"transcripts_{args.label}.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    words = sorted(r["words"] for r in results)
    contaminated = [r["id"] for r in results if r["forbidden_tokens"]]
    print(f"\nTOTAL {sum(words)} mots — médiane {words[len(words) // 2]} — écrit {out}")
    print(f"vocabulaire interdit : {len(results) - len(contaminated)}/{len(results)} propres"
          + (f" — {contaminated}" if contaminated else ""))


if __name__ == "__main__":
    main()
