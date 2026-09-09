"""MIA-5 — garde-fous de la consigne de concision du prompt système de M.I.A.

Ce fichier verrouille DEUX choses de nature différente :

1. **Ce que MIA-5 a ajouté** — la section ``FORMAT DE RÉPONSE``, et sa position
   en FIN de ``SYSTEM_PROMPT_STATIC`` (récence maximale : c'est la dernière
   consigne lue par le modèle, ce qui est précisément ce qui la rend efficace —
   la remonter au milieu du prompt annulerait l'effet mesuré).

2. **Ce que MIA-5 n'a pas le droit de retirer** — la phrase de refus mot pour
   mot, les règles d'ancrage (RÈGLE D'ABSENCE, diagnostic OB, état du marché,
   calendrier, catalogue) et le vocabulaire interdit. Réduire la longueur ne
   doit jamais réduire la précision factuelle ni affaiblir l'ancrage : si une
   future passe de « concision » supprime l'une de ces lignes, ce test casse.

Aucune couche de sécurité n'est testée ici (elles ont leurs propres fichiers) —
seulement le contrat du prompt.
"""

from __future__ import annotations

from src.intelligence.chatbot.chatbot import (
    SYSTEM_PROMPT_STATIC,
    SYSTEM_PROMPT_TEMPLATE,
)
from src.intelligence.chatbot.constants import (
    ALL_FORBIDDEN_TOKENS,
    INSIST_REDIRECT_TEMPLATE,
)

# La phrase de refus, mot pour mot. Elle est écrite à DEUX endroits (le prompt
# et le gabarit de la redirection ferme) et les deux doivent rester identiques.
REFUSAL_SENTENCE = (
    "Je décris les conditions du marché. La décision d'agir t'appartient."
)


# --------------------------------------------------------------------------- #
# 1 — la consigne de concision (MIA-5)
# --------------------------------------------------------------------------- #
def test_format_section_is_present() -> None:
    assert "FORMAT DE RÉPONSE (vaut pour toutes tes réponses) :" in SYSTEM_PROMPT_STATIC


def test_format_section_sits_at_the_very_end_of_the_static_prompt() -> None:
    """La section doit être la DERNIÈRE du bloc statique.

    C'est la position qui porte l'effet (mesuré : -43 % de mots sur 15 questions
    réelles). La déplacer au milieu la ferait perdre à la première règle
    d'ancrage qui demande d'« expliquer » ou de « rapporter la provenance ».
    """
    # index du TITRE de section (la puce des règles strictes y renvoie aussi).
    index = SYSTEM_PROMPT_STATIC.index("FORMAT DE RÉPONSE (vaut pour toutes tes réponses) :")
    tail = SYSTEM_PROMPT_STATIC[index:]
    # Aucune autre section (une section = un titre en capitales suivi de « : »)
    # ne doit venir après.
    for section in (
        "RÈGLES STRICTES",
        "CONTRÔLE DE L'AFFICHAGE",
        "DIAGNOSTIC ORDER BLOCK",
        "ÉTAT DU MARCHÉ",
        "CALENDRIER & PUBLICATIONS",
        "CATALOGUE DES MARCHÉS",
        "RÈGLE D'ABSENCE",
        "Tu as accès à 7 tools",
    ):
        assert section not in tail, f"{section} passe après FORMAT DE RÉPONSE"
    assert SYSTEM_PROMPT_STATIC.rstrip().endswith(
        "Une phrase vraie et courte, jamais une phrase courte et vague."
    )


def test_old_open_ended_length_rule_is_gone() -> None:
    """« 2-4 phrases » était une fourchette : le modèle visait le haut."""
    assert "2-4 phrases" not in SYSTEM_PROMPT_STATIC


def test_concision_rules_cover_the_four_padding_sources() -> None:
    """Les quatre sources de remplissage identifiées au diagnostic MIA-5."""
    # 1. préambule / reformulation
    assert "Aucun préambule, aucune reformulation de la question" in SYSTEM_PROMPT_STATIC
    # 2. conclusion générique / offre de service en fin de message
    assert "jamais une question ni une offre de service" in SYSTEM_PROMPT_STATIC
    # 3. contexte non demandé (y compris après un refus)
    assert "Tu n'ajoutes AUCUN élément que l'utilisateur n'a pas demandé" in SYSTEM_PROMPT_STATIC
    assert "Après une phrase de refus" in SYSTEM_PROMPT_STATIC
    # 4. fuite des rouages dans la prose
    assert "Tu ne cites pas tes rouages" in SYSTEM_PROMPT_STATIC


def test_explanations_stay_allowed_to_develop() -> None:
    """La concision vise le remplissage, jamais une question qui demande une
    explication (« explique-moi ce qu'est un CHOCH » reste développée)."""
    assert "développe autant qu'il faut" in SYSTEM_PROMPT_STATIC
    assert (
        "La concision porte sur le remplissage, jamais sur le fond demandé"
        in SYSTEM_PROMPT_STATIC
    )


# --------------------------------------------------------------------------- #
# 2 — ce que la concision n'a pas le droit de coûter
# --------------------------------------------------------------------------- #
def test_refusal_sentence_is_unchanged_word_for_word() -> None:
    assert REFUSAL_SENTENCE in SYSTEM_PROMPT_STATIC
    assert INSIST_REDIRECT_TEMPLATE == REFUSAL_SENTENCE


def test_being_short_never_drops_a_fact() -> None:
    """La règle explicite qui empêche « court » de devenir « vague »."""
    assert "Être concis ne retire JAMAIS un fait" in SYSTEM_PROMPT_STATIC
    for must_stay in ("niveaux exacts", "horodatages", "absence d'une donnée"):
        assert must_stay in SYSTEM_PROMPT_STATIC


def test_anchoring_rules_are_intact() -> None:
    """Les règles qui ancrent la réponse au moteur — jamais coupées au nom de la
    brièveté."""
    for rule in (
        # RÈGLE D'ABSENCE (MIA-3)
        "Un texte qui sonne juste sur une donnée que tu n'as pas est un mensonge.",
        # jamais de données inventées
        "Tu n'inventes jamais de données",
        # diagnostic OB : jamais la définition générale à la place du moteur
        "Tu n'expliques JAMAIS un rejet sans ce diagnostic",
        # état du marché fait autorité
        "Il fait autorité sur l'actualité des données.",
        # calendrier : jamais un chiffre sans l'outil
        "Tu ne cites JAMAIS un chiffre de publication sans l'avoir obtenu par ces outils.",
        # catalogue : jamais un marché absent du registre
        "Tu ne prétends jamais suivre un marché absent de ce catalogue",
        # ids réels seulement
        "n'utilise QUE des ids renvoyés par le moteur",
    ):
        assert rule in SYSTEM_PROMPT_STATIC, rule


def test_forbidden_vocabulary_rules_are_intact() -> None:
    """Le vocabulaire interdit (section 3 du dossier) reste interdit, qu'on soit
    concis ou pas — et les deux puces du prompt qui l'énoncent restent en place."""
    assert "Tu n'utilises jamais : achète, vends" in SYSTEM_PROMPT_STATIC
    assert "Tu n'utilises jamais : risqué, sûr, dangereux" in SYSTEM_PROMPT_STATIC
    # Le jeu de jetons de la Couche 3 n'est pas touché par MIA-5.
    assert len(ALL_FORBIDDEN_TOKENS) >= 115


def test_full_template_still_carries_the_format_section() -> None:
    """La vue concaténée (statique + contexte variable) reste cohérente."""
    assert "FORMAT DE RÉPONSE" in SYSTEM_PROMPT_TEMPLATE
    assert "CONTEXTE INITIAL (signal_summary)" in SYSTEM_PROMPT_TEMPLATE
