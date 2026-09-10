"""Chantier 4 — niveau 1.5 strict vocabulary: forbidden tokens + adversarial patterns.

This module is **pure data** (plus a tiny ``normalize_text`` helper). The actual
matching logic lives in:
  - ``adversarial_filter`` (Couche 1) — consumes ``ADVERSARIAL_PATTERNS_BY_CATEGORY``.
  - ``output_filter`` (Couche 3) — consumes ``ALL_FORBIDDEN_TOKENS``.

Design decisions
----------------
1. **Forbidden-token taxonomy follows doc §4.2** (the authoritative, legally-
   framed source) — Cat A *action de trading*, Cat B *recommandation*, Cat C
   *jugement de moment*, Cat D *jugement de valeur sur risque*. The brief's
   skeleton proposed alternate names (``URGENCE`` / ``ENGAGEMENT_EMOTIONNEL``);
   high-value guarantee/urgency tokens are folded into Cat D (a guarantee *is* a
   value judgement) so nothing is lost. Mapping is flagged in the Étape 2 report.

2. **Adversarial-pattern taxonomy follows the founder's validated 4-bucket
   mapping** (Étape 1 sign-off): JAILBREAK / TRADE_REQUEST / PERSONA_HIJACK /
   FINANCIAL_ADVICE. TRADE_REQUEST = doc Cat1 (action directe) + Cat4 (signal);
   FINANCIAL_ADVICE = doc Cat2 (personnalisation) + Cat3 (jugement de valeur).

3. **Homonym exclusions (deliberate).** French has high-frequency homonyms of
   trading verbs that would pathologically false-positive a *descriptive*
   chatbot. The following bare forms are intentionally EXCLUDED from the
   forbidden sets and handled via context / the input filter instead:
     - ``entre``  → preposition "between" ("FVG entre 2376 et 2378"). Kept:
       ``entrez`` / ``entrer`` / ``entry`` (unambiguous).
     - ``place`` / ``placer`` → "le support se place à…". Kept: none bare.
     - ``risque`` (noun, no accent) → "la gestion du risque". Kept: ``risqué``
       (accented adjective, always a judgement) + phrases ``c'est risqué``.
     - ``sûr`` (bare) → "bien sûr" = "of course". Kept: phrase ``c'est sûr``.
     - ``ouvre`` / ``ferme`` (bare) → "ferme" = firm/farm. Kept: ``ouvrez`` /
       ``ouvrir`` / ``fermez`` / ``fermer``.
   Matching in Couche 3 is accent-insensitive (via ``normalize_text``), so the
   accented and non-accented variants stored here both resolve correctly.

   Homonym exclusion table (kept for future traceability — extend if Couche 1
   surfaces more ambiguous pairs):

     Homonyme exclu (sens descriptif)   | Formes gardées (sens directif)
     -----------------------------------+--------------------------------------
     entre   (= between)                | entrez, entrer, entry
     place / placer (= se situe)        | placez (impératif)
     risque  (nom = risk noun)          | risqué (jugement), "high/low risk"
     sûr     (= certain, "bien sûr")    | "c'est sûr", "c'est safe"
     ouvre / ouvres (= se passe)        | ouvrez, ouvrir (impératif/infinitif)
     ferme / fermes (= ferme adj.)      | fermez, fermer (impératif/infinitif)
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

from src.intelligence.chatbot import adversarial_i18n as _adv_i18n
from src.intelligence.chatbot import templates_i18n as _i18n

# --------------------------------------------------------------------------- #
# Text normalisation (shared by Couche 1 and Couche 3)
# --------------------------------------------------------------------------- #


def normalize_text(text: str) -> str:
    """Lower-case, strip accents, and normalise apostrophes.

    Adversarial regex patterns are authored against the *normalised* form, and
    the output filter normalises both the LLM text and the forbidden tokens
    before comparing. This makes matching robust to ``é`` vs ``e`` and to curly
    vs straight apostrophes — the two most common evasion / noise vectors in
    French user input and LLM output.
    """
    decomposed = unicodedata.normalize("NFKD", fold_case(text))
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def fold_case(text: str) -> str:
    """Lower-case and normalise apostrophes, but KEEP the accents.

    Used by the output filter for the handful of tokens whose accent is the only
    thing that separates a value judgement from an ordinary French word
    (``risqué`` the adjective vs ``risque`` the noun) — see
    :data:`ACCENT_SENSITIVE_TOKENS`.
    """
    lowered = text.lower().strip()
    # Curly / modifier apostrophes → straight ASCII apostrophe.
    return lowered.replace("’", "'").replace("ʼ", "'").replace("‘", "'")


# --------------------------------------------------------------------------- #
# FORBIDDEN TOKENS — 4 categories (doc §4.2)
# --------------------------------------------------------------------------- #

# Catégorie A — Verbes d'action de trading.
FORBIDDEN_TOKENS_ACTION_TRADING: frozenset[str] = frozenset({
    "achète", "achete", "achetez", "achètes", "achetes", "acheter",
    "vends", "vendez", "vendre",
    "buy", "sell",
    "entrez", "entrer", "entry",          # bare "entre" excluded (homonym "between")
    "sors", "sortez", "sortir", "exit",
    "ouvrez", "ouvrir", "fermez", "fermer",  # bare "ouvre"/"ferme" excluded (homonyms)
    "trade", "trades", "tradez", "trader",
})

# Catégorie B — Verbes de recommandation.
# Bare verb forms are included so an intercalated pronoun does not bypass the
# filter ("je TE recommande", "on recommande", "il VOUS conseille"). A bare verb
# never matches its infinitive ("conseiller" / "recommander"), which is how a
# compliant refusal phrase like "je ne peux pas te conseiller" stays clean.
FORBIDDEN_TOKENS_RECOMMANDATION: frozenset[str] = frozenset({
    "je conseille", "je te conseille", "je vous conseille", "je conseillerais",
    "je déconseille", "je deconseille",
    "je recommande", "je ne recommande pas", "je recommanderais",
    "je suggère", "je suggere", "je préconise", "je preconise",
    "recommande", "recommandes", "recommandons",
    "conseille", "conseilles", "conseillons",
    "déconseille", "deconseille", "deconseilles",
    "suggère", "suggere", "suggères", "suggeres",
    "préconise", "preconise", "préconises", "preconises",
    "tu devrais", "vous devriez", "tu ferais mieux", "vous feriez mieux",
    "mieux vaut",
    "il faut", "il faudrait",
    "évite", "evite", "évitez", "evitez",
})

# Catégorie C — Jugements de moment.
FORBIDDEN_TOKENS_JUGEMENT_MOMENT: frozenset[str] = frozenset({
    "bon moment", "mauvais moment", "le bon moment",
    "c'est le moment", "c'est le bon moment", "ce n'est pas le moment",
    "timing parfait", "moment idéal", "moment ideal", "moment parfait",
    "opportunité", "opportunite", "occasion à saisir", "occasion a saisir",
    "setup parfait", "entrée idéale", "entree ideale",
    "point d'entrée idéal", "point d'entree ideal",
    "entrée parfaite", "entree parfaite",
})

# Catégorie D — Jugements de valeur sur risque (+ garanties / urgence repliées ici).
# NB (MIA-5) : le jeton nu ``risqué`` est ACCENT-SENSIBLE côté sortie (voir
# ``ACCENT_SENSITIVE_TOKENS``) — sans quoi « la gestion du risque » ou « le risque
# de volatilité », qui sont des noms communs descriptifs, faisaient tomber une
# réponse légitime sur le gabarit de contamination. L'adjectif reste bloqué sous
# toutes ses formes : accentué en tant que jeton nu, et NON accentué via les
# tournures adverbiales/comparatives ci-dessous, ajoutées pour ne rien perdre.
FORBIDDEN_TOKENS_JUGEMENT_RISQUE: frozenset[str] = frozenset({
    "c'est risqué", "c'est risque", "trop risqué", "trop risque",
    "peu risqué", "peu risque", "risqué",
    # Accords de l'adjectif : « une position risquée » n'était bloquée par AUCUN
    # jeton avant MIA-5 (le jeton nu ne couvrait que le masculin singulier).
    "risquée", "risqués", "risquées",
    "très risqué", "tres risque", "assez risqué", "assez risque",
    "plutôt risqué", "plutot risque", "moins risqué", "moins risque",
    "plus risqué", "plus risque",
    "dangereux", "dangereuse", "c'est dangereux",
    "c'est sûr", "c'est sur", "c'est sécurisé", "c'est securise",
    "sécurisé", "securise",
    "c'est safe", "safe", "low risk", "high risk",
    "sans risque", "risk-free",
    "garanti", "garantie", "garantis", "gain garanti", "profit garanti",
})

# Per-category mapping (used by the output filter to report which category leaked).
FORBIDDEN_TOKENS_BY_CATEGORY: dict[str, frozenset[str]] = {
    "action_trading": FORBIDDEN_TOKENS_ACTION_TRADING,
    "recommandation": FORBIDDEN_TOKENS_RECOMMANDATION,
    "jugement_moment": FORBIDDEN_TOKENS_JUGEMENT_MOMENT,
    "jugement_risque": FORBIDDEN_TOKENS_JUGEMENT_RISQUE,
}

ALL_FORBIDDEN_TOKENS: frozenset[str] = (
    FORBIDDEN_TOKENS_ACTION_TRADING
    | FORBIDDEN_TOKENS_RECOMMANDATION
    | FORBIDDEN_TOKENS_JUGEMENT_MOMENT
    | FORBIDDEN_TOKENS_JUGEMENT_RISQUE
)


# --------------------------------------------------------------------------- #
# MIA-5 — précision de la Couche 3 : deux faux positifs PROUVÉS
# --------------------------------------------------------------------------- #
# Aucun jeton n'est retiré : on retire uniquement DEUX collisions mesurées, qui
# faisaient remplacer une réponse factuelle correcte par le gabarit « Je ne peux
# pas formuler cette réponse de cette manière » (le client voyait un refus
# incompréhensible). Les deux cas ont été capturés sur des réponses réelles.

# 1. Jetons dont SEUL l'accent sépare le jugement de valeur du mot courant.
#    Ils sont cherchés dans le texte accentué (``fold_case``) au lieu du texte
#    normalisé. Même raisonnement que l'exclusion déjà faite pour ``sûr`` côté
#    entrée (« bien sûr ») — appliqué ici au couple ``risqué`` / ``risque`` :
#      bloqué   : « c'est risqué », « trop risqué », « très risque », « risqué »
#      laissé   : « la gestion du risque », « le risque de volatilité »
ACCENT_SENSITIVE_TOKENS: frozenset[str] = frozenset({
    "risqué", "risquée", "risqués", "risquées",
})

# 2. Expressions FIGÉES dans lesquelles un jeton interdit n'est pas un jugement.
#    « coût d'opportunité » est le terme macro-économique standard pour un actif
#    sans rendement (l'or) : le mesurer n'est pas juger d'un moment d'entrée.
#    Neutralisées AVANT la recherche, et rien d'autre ne l'est.
#    « c'est le moment OÙ … » est une subordonnée temporelle descriptive (« c'est
#    le moment où les ordres se manifestent ») : le jugement interdit est « c'est
#    le moment DE / POUR agir », qui reste bloqué, comme « c'est le bon moment ».
HOMONYM_SAFE_EXPRESSIONS: frozenset[str] = frozenset({
    "coût d'opportunité", "coûts d'opportunité",
    "c'est le moment où",
})


# --------------------------------------------------------------------------- #
# ADVERSARIAL PATTERNS — 4 buckets (founder-validated mapping)
# --------------------------------------------------------------------------- #
# Patterns are authored against ``normalize_text`` output (lower-case, accent-
# stripped, straight apostrophes). Compiled with IGNORECASE as defence in depth.

_JAILBREAK_RAW: list[str] = [
    # "ignore/oublie (toutes) tes/les/ces instructions/consignes/regles"
    r"(ignore|oublie|oubli)[a-z]*\s+((toutes?|tout)\s+)?(tes|les|ces)\s+"
    r"(instructions?|consignes?|regles?|directives?)",
    # "tu es maintenant / desormais ..."
    r"\btu\s+es\s+(maintenant|desormais)\b",
    # DAN-style jailbreak handles
    r"(mode\s+dan|\bdan\b\s+mode|do\s+anything\s+now|jailbreak)",
    # "override (le) system(e) (prompt)"
    r"\boverride\s+(le\s+)?(system|systeme)\b",
    # injected system prompt header
    r"\bsystem\s*prompt\s*[:=]",
    # "(nouvelles) instructions : ..." injected directive
    r"\b(nouvelles?\s+instructions?|new\s+instructions?)\s*[:=]",
    # "reponds sans (aucune) restriction/filtre/limite"
    r"\b(reponds?|repond)\s+sans\s+(aucune?\s+)?(restriction|filtre|limite|censure)",
    # "desactive/contourne tes filtres/restrictions/regles"
    r"\b(desactive|contourne|ignore)[a-z]*\s+(tes|les)\s+"
    r"(filtres?|restrictions?|regles?|securites?)",
]

_TRADE_REQUEST_RAW: list[str] = [
    # "(est-ce que) je devrais/dois/peux trader/acheter/vendre/entrer/..."
    r"(est-?ce\s+que\s+)?\bje\s+(devrais|dois|peux|vais)\s+"
    r"(trader|acheter|vendre|entrer|sortir|prendre|miser|investir)",
    # inverted interrogative "dois-je / devrais-je / puis-je acheter/vendre/..."
    r"\b(dois|devrais|puis)-?je\s+"
    r"(trader|acheter|vendre|entrer|sortir|prendre|miser|investir|y\s+aller)",
    # "donne-moi / envoie (un) signal/trade/setup/entree"
    r"\b(donne|envoie|file|balance|passe)\b[a-z-]*\s+(moi\s+)?(un\s+|une\s+)?"
    r"(signal|trade|setup|entree|position)",
    # "long ou short / bull ou bear / achat ou vente"
    r"\b(long\s+ou\s+short|short\s+ou\s+long|bull\s+ou\s+bear|bear\s+ou\s+bull|"
    r"achat\s+ou\s+vente|vente\s+ou\s+achat|acheter\s+ou\s+vendre)\b",
    # "j'entre ou pas / je trade ou pas"
    r"\bj'?\s*entre\s+ou\s+(pas|non)\b|\bje\s+trade\s+ou\s+(pas|non)\b",
    # "tu conseilles quoi / quel est ton avis|conseil / ton avis|conseil sur"
    r"(tu\s+conseilles?\s+quoi|quel\s+est\s+ton\s+(avis|conseil)|"
    r"ton\s+(avis|conseil)\s+sur)",
    # "(un) (bon) trade a prendre"
    r"\b(un\s+)?(bon\s+)?trade\s+a\s+prendre\b",
    # "c'est le (bon) moment pour/de ..."
    r"\bc'?est\s+le\s+(bon\s+)?moment\s+(pour|de|d')",
    # "quelle position/direction prendre / dois-je prendre"
    r"\bquelle?\s+(position|direction)\s+(prendre|dois-?je|devrais-?je)",
]

_PERSONA_HIJACK_RAW: list[str] = [
    # "agis comme un/une/si ..."
    r"\bagis?\s+comme\s+(un|une|si)\b",
    # "fais comme si / fais semblant d'etre / que tu ..."
    r"\b(fais|fait)\s+(comme\s+si|semblant)\b",
    # "comporte-toi comme ..."
    r"\b(comporte|comportes)-?toi\s+comme\b",
    # "reponds en tant qu' ..."
    r"\b(reponds?|repond)\s+en\s+tant\s+qu['e]",
    # "tu es un trader/expert/conseiller/analyste qui conseille|recommande"
    r"\btu\s+es\s+(un|une)\s+(trader|expert|conseiller|gourou|"
    r"analyste\s+qui\s+(conseille|recommande))",
    # "joue/incarne le role/un role/le personnage"
    r"\b(joue|incarne)\s+(le\s+role|un\s+role|le\s+personnage)",
    # English: "pretend you are/to be" / "act as a/an"
    r"\bpretend\s+(you\s+are|to\s+be)\b|\bact\s+as\s+(a|an)\b",
    # "imagine que tu es un trader/conseiller/expert"
    r"\bimagine\s+que\s+tu\s+es\s+(un|une)\s+(trader|conseiller|expert|pro)",
    # "oublie que tu es un outil/assistant/mia"
    r"\boublie\s+que\s+tu\s+es\s+(un\s+)?(outil|assistant|mia|bot)",
]

_FINANCIAL_ADVICE_RAW: list[str] = [
    # "avec 1000€ / 5 lots / 2k de capital ..." (no trailing \b: € / $ are
    # non-word chars, so a word-boundary after them never matches).
    r"\bavec\s+\d+\s*(k|euros?|eur|€|dollars?|usd|\$|lots?)",
    # "mon stop/sl/target/tp/entree/capital/budget est|sera a|de <n>"
    r"\bmon\s+(stop|sl|stop-?loss|target|tp|take-?profit|entree|capital|budget|levier)\b"
    r"[a-z\s]*\b(est|sera|se\s+situe|a|de)\b",
    # "j'ai 3 positions / 2 lots / trades ouverts"
    r"\bj'?ai\s+\d+\s+(positions?|lots?|trades?(\s+ouverts?)?)",
    # "si je perds/gagne/mise/investis <n>"
    r"\bsi\s+je\s+(perds?|gagne|mise|investis|risque)\s+\d+",
    # "c'est/est-ce risque|dangereux|safe|sur|securise|prudent de/pour"
    r"\b(c'?est|est-?ce)\s+(risque|dangereux|safe|sur|securise|prudent|raisonnable)\s+"
    r"(de|d'|pour)",
    # "est-ce sur/prudent/raisonnable/safe de ..."
    r"\best-?ce\s+(sur|prudent|raisonnable|safe)\s+(de|d')",
    # "combien (je) devrais/dois miser/investir/risquer"
    r"\b(combien|quelle\s+somme)\s+(je\s+)?(devrais|dois|peux)\s+"
    r"(miser|investir|risquer|mettre)",
    # "avec mon/un capital de ..."
    r"\bavec\s+(mon|un)\s+capital\b",
    # "quelle taille de position/lot/levier (je) dois/devrais prendre"
    r"\bquelle?\s+(taille\s+de\s+)?(position|lot|levier)\s+(je\s+)?"
    r"(dois|devrais|prendre)",
]


# Catégorie PREDICTION — demander à l'outil ce que le prix VA FAIRE.
#
# Ce seau intercepte la demande de pronostic, PAS le futur grammatical : « quand
# le marché va-t-il rouvrir ? » ou « y a-t-il une news bientôt ? » sont des
# questions FACTUELLES et doivent atteindre le modèle. Chaque motif exige donc
# deux choses — un cadre de pronostic ET un mot de direction de prix — ou une
# formule de prévision sans ambiguïté.
#
# Doctrine (même arbitrage que les homonymes plus haut) : sur l'ENTRÉE on préfère
# laisser passer un cas limite, que le prompt traite déjà, plutôt que de refuser
# une question descriptive. Le sur-blocage est réservé à la SORTIE (Couche 3).
_PREDICTION_RAW: list[str] = [
    # "ca / le prix / le marche va (-t-il) monter|baisser|rebondir…"
    r"\b(ca|cela|il|elle|le\s+prix|le\s+cours|le\s+marche|l'or|l'euro|le\s+niveau|"
    r"la\s+zone)\s+(va|vont|ira|iront)\s*-?\s*t?\s*-?\s*(il|elle|ils|elles)?\s+"
    r"(monter|baisser|descendre|remonter|redescendre|rebondir|chuter|grimper|"
    r"repartir|continuer|tenir|exploser|plonger|corriger|casser)",
    # forme inversée nue : "va-t-il monter ?"
    r"\bva\s*-?\s*t\s*-?\s*(il|elle)\s+(monter|baisser|descendre|remonter|rebondir|"
    r"chuter|grimper|repartir|continuer|tenir|casser)\b",
    # "tu penses / a ton avis / selon toi …" + un mot de DIRECTION. Jamais "va"
    # seul, qui attraperait « tu penses que ça va rouvrir quand ? ».
    r"\b(tu\s+penses?|penses?\s*-?\s*tu|tu\s+crois|crois\s*-?\s*tu|a\s+ton\s+avis|"
    r"selon\s+toi)\b[^?.!]{0,60}\b(monter|baisser|descendre|remonter|rebondir|chuter|"
    r"grimper|hausse|baisse|direction)\b",
    # verbes et noms de prévision adressés à l'outil. Le nom porte une exclusion :
    # le produit PRÉVOIT la volatilité (une amplitude), jamais une direction —
    # « quelle est la prévision de volatilité ? » est une question produit
    # légitime et doit atteindre le modèle.
    r"\b(predis|predit|predire|prevois|prevoir|anticipes?|anticiper|pronostique)\b"
    r"|\b(prediction|predictions|prevision|previsions|pronostic|anticipation)\b"
    + _adv_i18n.NO_VOLATILITY,
    # "quel est ton objectif / ta cible" · "objectif de prix" · "price target"
    r"\b(ton|ta|votre|vos)\s+(objectif|cible|prevision|pronostic|anticipation)\b|"
    r"\bobjectif\s+de\s+prix\b|\bprice\s+target\b",
    # "ou va le prix" / "jusqu'ou ca monte"
    r"\b(ou|jusqu'?\s*ou)\s+(va|ira|monte|montera|descend|descendra)\b",
    # anglais
    r"\bwill\s+(it|price|gold|the\s+market|eurusd|xauusd)\s+(go|rise|fall|drop|bounce|"
    r"break|continue|hold)\b|\bdo\s+you\s+think\s+[^?.!]{0,60}\b(will|going\s+to)\b|"
    r"\bwhat'?s?\s+your\s+(target|forecast|prediction)\b",
    # "hausse ou baisse ?" / "up or down ?"
    r"\b(hausse\s+ou\s+baisse|baisse\s+ou\s+hausse|up\s+or\s+down|down\s+or\s+up)\b",
]


def _compile(raw_patterns: list[str]) -> list[re.Pattern[str]]:
    return [re.compile(p, re.IGNORECASE) for p in raw_patterns]


# The FRENCH core, kept as its own object: it is the founder-validated set, it
# is what the "5-10 patterns per bucket" review rule applies to, and it must
# stay readable without scrolling past a hundred lines of other languages.
FRENCH_PATTERNS_BY_CATEGORY: dict[str, list[re.Pattern[str]]] = {
    "jailbreak": _compile(_JAILBREAK_RAW),
    "trade_request": _compile(_TRADE_REQUEST_RAW),
    "persona_hijack": _compile(_PERSONA_HIJACK_RAW),
    "financial_advice": _compile(_FINANCIAL_ADVICE_RAW),
    "prediction": _compile(_PREDICTION_RAW),
}


def _bucket(category: str) -> list[re.Pattern[str]]:
    """French core first, then the seven other locales (stable order).

    Every pattern runs against EVERY message: Couche 1 sees the text before
    anything identifies a locale, and a French user may type English. The
    extension is therefore written for precision — see adversarial_i18n.
    """
    return FRENCH_PATTERNS_BY_CATEGORY[category] + _compile(
        _adv_i18n.raw_patterns_for(category)
    )


ADVERSARIAL_PATTERNS_JAILBREAK: list[re.Pattern[str]] = _bucket("jailbreak")
ADVERSARIAL_PATTERNS_TRADE_REQUEST: list[re.Pattern[str]] = _bucket("trade_request")
ADVERSARIAL_PATTERNS_PERSONA_HIJACK: list[re.Pattern[str]] = _bucket("persona_hijack")
ADVERSARIAL_PATTERNS_FINANCIAL_ADVICE: list[re.Pattern[str]] = _bucket("financial_advice")
ADVERSARIAL_PATTERNS_PREDICTION: list[re.Pattern[str]] = _bucket("prediction")

# Ordered so the most security-critical bucket (jailbreak) is checked first, and
# the FIRST match wins. ``prediction`` is deliberately LAST: a message that is
# both a forecast request and a trade request ("tu penses que je devrais
# acheter ?") must keep reporting the graver category, and no existing message
# may see its reported category change because a bucket was added.
ADVERSARIAL_PATTERNS_BY_CATEGORY: dict[str, list[re.Pattern[str]]] = {
    "jailbreak": ADVERSARIAL_PATTERNS_JAILBREAK,
    "trade_request": ADVERSARIAL_PATTERNS_TRADE_REQUEST,
    "persona_hijack": ADVERSARIAL_PATTERNS_PERSONA_HIJACK,
    "financial_advice": ADVERSARIAL_PATTERNS_FINANCIAL_ADVICE,
    "prediction": ADVERSARIAL_PATTERNS_PREDICTION,
}

ALL_ADVERSARIAL_PATTERNS: list[re.Pattern[str]] = [
    p for patterns in ADVERSARIAL_PATTERNS_BY_CATEGORY.values() for p in patterns
]


# --------------------------------------------------------------------------- #
# Refusal / fallback templates (shared by Couche 1 and Couche 3)
# --------------------------------------------------------------------------- #
# NOTE: these templates are RETURNED verbatim as responses and are NEVER passed
# back through the output filter, so the literal "trade" inside REFUSAL_TEMPLATE
# (part of the refusal itself) does not trip Couche 3.

# Couche 1 — pedagogical refusal when an adversarial pattern is intercepted (doc §4.5).
# IMPORTANT: every template below is deliberately free of forbidden tokens so the
# strong invariant "the chatbot never returns a forbidden token" holds even for
# its own safety nets. (Earlier wording "recommandations de trade" / "tolérance
# au risque" tripped the output filter on "trade" / "risque" — reworded.)
#
# The TEXT lives in ``templates_i18n`` — one string per template per locale, so
# the nine locales cannot drift from each other and the French cannot be edited
# in two places. These names stay the French view of it, unchanged for callers.
REFUSAL_TEMPLATE: str = _i18n.REFUSAL[_i18n.DEFAULT_LOCALE]

# Couche 1 — refus dédié au seau ``prediction``. Le refus générique parle de
# recommandations ; ici la question porte sur l'AVENIR du prix, et la réponse
# honnête doit le dire.
PREDICTION_REFUSAL_TEMPLATE: str = _i18n.PREDICTION_REFUSAL[_i18n.DEFAULT_LOCALE]

# Couche 3 — replacement when the LLM output contains a forbidden token.
OUTPUT_CONTAMINATED_TEMPLATE: str = _i18n.OUTPUT_CONTAMINATED[_i18n.DEFAULT_LOCALE]

# Couche 2 — fail-safe when the Anthropic API errors (timeout / rate limit / network).
LLM_ERROR_TEMPLATE: str = _i18n.LLM_ERROR[_i18n.DEFAULT_LOCALE]

# Firm redirection used when the user insists on a recommendation (niveau 1.5
# rule). NOT localised on purpose: it is quoted INSIDE the system prompt as an
# example for the model, never returned verbatim, so the model already renders
# it in the visitor's language.
INSIST_REDIRECT_TEMPLATE: str = (
    "Je décris les conditions du marché. La décision d'agir t'appartient."
)

# Couche 1 — which refusal family a given bucket answers with. Absent from the
# map = the generic refusal, so adding a bucket never silently changes what the
# existing ones say.
REFUSAL_FAMILY_BY_CATEGORY: dict[str, str] = {
    "prediction": "PREDICTION_REFUSAL_TEMPLATE",
}

# Kept for callers that want the French mapping directly.
REFUSAL_TEMPLATE_BY_CATEGORY: dict[str, str] = {
    "prediction": PREDICTION_REFUSAL_TEMPLATE,
}


def refusal_for(category: Optional[str], locale: Optional[str] = None) -> str:
    """The refusal text Couche 1 returns for a matched bucket, in ``locale``.

    An unknown bucket falls back to the generic refusal; an unknown locale falls
    back to French. A defence layer must always have something to say.
    """
    family = REFUSAL_FAMILY_BY_CATEGORY.get(category or "", "REFUSAL_TEMPLATE")
    return _i18n.template_for(family, locale)


def localized(name: str, locale: Optional[str] = None) -> str:
    """Any verbatim template, in ``locale`` (French fallback)."""
    return _i18n.template_for(name, locale)


# Couche 4 — on-brand refusal when a chart-view action falls outside the
# display-only whitelist (e.g. inventing / moving / resizing a structure).
VIEW_ACTION_REFUSAL_TEMPLATE: str = _i18n.VIEW_ACTION_REFUSAL[_i18n.DEFAULT_LOCALE]

# Couche 4 — honest report when a category-targeted mask (« masque les SSL »)
# resolves to ZERO engine-emitted structures on the current reading. Nothing is
# hidden and nothing is invented; the model relays this factually.
VIEW_ACTION_EMPTY_CATEGORY_TEMPLATE: str = _i18n.VIEW_ACTION_EMPTY_CATEGORY[
    _i18n.DEFAULT_LOCALE
]


__all__ = [
    "ACCENT_SENSITIVE_TOKENS",
    "ADVERSARIAL_PATTERNS_BY_CATEGORY",
    "ADVERSARIAL_PATTERNS_FINANCIAL_ADVICE",
    "ADVERSARIAL_PATTERNS_JAILBREAK",
    "ADVERSARIAL_PATTERNS_PERSONA_HIJACK",
    "ADVERSARIAL_PATTERNS_PREDICTION",
    "ADVERSARIAL_PATTERNS_TRADE_REQUEST",
    "ALL_ADVERSARIAL_PATTERNS",
    "ALL_FORBIDDEN_TOKENS",
    "FRENCH_PATTERNS_BY_CATEGORY",
    "FORBIDDEN_TOKENS_ACTION_TRADING",
    "FORBIDDEN_TOKENS_BY_CATEGORY",
    "FORBIDDEN_TOKENS_JUGEMENT_MOMENT",
    "FORBIDDEN_TOKENS_JUGEMENT_RISQUE",
    "FORBIDDEN_TOKENS_RECOMMANDATION",
    "HOMONYM_SAFE_EXPRESSIONS",
    "INSIST_REDIRECT_TEMPLATE",
    "fold_case",
    "LLM_ERROR_TEMPLATE",
    "OUTPUT_CONTAMINATED_TEMPLATE",
    "PREDICTION_REFUSAL_TEMPLATE",
    "REFUSAL_TEMPLATE",
    "REFUSAL_FAMILY_BY_CATEGORY",
    "REFUSAL_TEMPLATE_BY_CATEGORY",
    "VIEW_ACTION_EMPTY_CATEGORY_TEMPLATE",
    "VIEW_ACTION_REFUSAL_TEMPLATE",
    "normalize_text",
    "localized",
    "refusal_for",
]
