"""Generate the native-reviewer sheet for the Couche 1 language blocks.

The multilingual detection was machine-authored. That limit cannot be closed by
more machine work — it needs a native speaker per language. What CAN be done is
to make the review take minutes instead of an afternoon: a reviewer should never
have to read a regex.

So this emits, per language, the two lists that actually decide whether a block
is right — **the sentences we intercept** and **the sentences we deliberately let
through** — straight from the test corpora, so the sheet can never drift from
what the code does. A reviewer judges concrete sentences, and adds the ones a
real user of their language would type.

    python scripts/gen_adversarial_review.py

Writes docs/audits/couche1-revue-linguistique.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from src.intelligence.chatbot import adversarial_i18n as A  # noqa: E402
from src.intelligence.chatbot import templates_i18n as I18N  # noqa: E402
from test_adversarial_i18n import BENIGN, POSITIVES  # noqa: E402

LANGUAGE_NAMES = {
    "de": "Allemand", "es": "Espagnol", "it": "Italien", "nl": "Néerlandais",
    "pl": "Polonais", "pt": "Portugais", "ar": "Arabe",
}

BUCKET_INTENT = {
    "jailbreak": "faire sortir l'outil de ses règles",
    "trade_request": "demander quoi faire sur le marché (acheter / vendre / un signal)",
    "persona_hijack": "lui faire jouer un rôle (conseiller, trader…)",
    "financial_advice": "demander un avis personnalisé sur SA situation ou SON argent",
    "prediction": "demander ce que le prix VA faire",
}

OUT = REPO_ROOT / "docs" / "audits" / "couche1-revue-linguistique.md"


def main() -> None:
    lines: list[str] = [
        "# Couche 1 — fiche de revue linguistique",
        "",
        "> **Généré** par `scripts/gen_adversarial_review.py` depuis les corpus de test.",
        "> Ne pas éditer à la main : relancer le script.",
        "",
        "## Ce qu'on vous demande",
        "",
        "Les motifs de détection ont été **écrits par la machine**. Vous n'avez pas à lire",
        "de code : jugez des phrases.",
        "",
        "Pour votre langue, deux listes :",
        "",
        "1. **Interceptées** — la question est refusée immédiatement, sans appeler le modèle.",
        "2. **Laissées passer** — la question est légitime et doit recevoir une vraie réponse.",
        "",
        "Trois questions, dans cet ordre d'importance :",
        "",
        "- **Une phrase de la liste 2 vous paraît-elle devoir être refusée ?** (peu grave)",
        "- **Une phrase de la liste 1 vous paraît-elle légitime ?** ⚠️ **C'est le cas grave** :",
        "  cela veut dire qu'un client se fait refuser une question normale, sans recours.",
        "- **Quelles formulations manquent ?** Ajoutez les phrases qu'un vrai utilisateur de",
        "  votre langue taperait pour la même intention — c'est là que la couverture se gagne.",
        "",
        "Rappel utile : **rater une formulation n'est pas grave** (le modèle refuse quand même,",
        "cela coûte juste un appel). **Refuser à tort est grave.** En cas de doute, on laisse passer.",
        "",
        "---",
        "",
    ]

    for lang in A.EXTENDED_LOCALES:
        name = LANGUAGE_NAMES.get(lang, lang)
        block = A.PATTERNS_BY_LANG[lang]
        n_patterns = sum(len(p) for p in block.values())
        lines += [
            f"## {name} (`{lang}`) — {n_patterns} motifs",
            "",
            "### 1. Phrases INTERCEPTÉES (refus immédiat, sans appel au modèle)",
            "",
            "| Intention | Phrase testée |",
            "|---|---|",
        ]
        for text, bucket in POSITIVES.get(lang, []):
            lines.append(f"| {BUCKET_INTENT.get(bucket, bucket)} | {text} |")

        lines += [
            "",
            "### 2. Phrases LAISSÉES PASSER (questions légitimes)",
            "",
        ]
        for text in BENIGN.get(lang, []):
            lines.append(f"- {text}")

        refusal = I18N.REFUSAL.get(lang, "")
        prediction = I18N.PREDICTION_REFUSAL.get(lang, "")
        lines += [
            "",
            "### 3. Le texte du refus, dans votre langue",
            "",
            "Il est renvoyé tel quel, sans passer par le modèle — donc sa formulation compte.",
            "",
            "**Refus général :**",
            "",
            "> " + refusal.replace("\n\n", "\n>\n> "),
            "",
            "**Refus d'une demande de pronostic :**",
            "",
            "> " + prediction.replace("\n\n", "\n>\n> "),
            "",
            "---",
            "",
        ]

    lines += [
        "## Après votre relecture",
        "",
        "Les phrases que vous ajoutez ou contestez vont dans les corpus de test",
        "(`tests/test_adversarial_i18n.py`, `POSITIVES` et `BENIGN`) : les tests échouent",
        "alors tant que les motifs ne s'y conforment pas. C'est votre relecture qui devient",
        "la garantie, pas une promesse.",
        "",
    ]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"écrit : {OUT.relative_to(REPO_ROOT)} ({len(lines)} lignes)")


if __name__ == "__main__":
    main()
