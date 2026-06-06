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
    lowered = text.lower().strip()
    # Curly / modifier apostrophes → straight ASCII apostrophe.
    lowered = lowered.replace("’", "'").replace("ʼ", "'").replace("‘", "'")
    # Strip combining accents (NFKD decompose, drop combining marks).
    decomposed = unicodedata.normalize("NFKD", lowered)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


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
FORBIDDEN_TOKENS_RECOMMANDATION: frozenset[str] = frozenset({
    "je conseille", "je te conseille", "je vous conseille", "je conseillerais",
    "je déconseille", "je deconseille",
    "je recommande", "je ne recommande pas", "je recommanderais",
    "je suggère", "je suggere", "je préconise", "je preconise",
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
FORBIDDEN_TOKENS_JUGEMENT_RISQUE: frozenset[str] = frozenset({
    "c'est risqué", "c'est risque", "trop risqué", "trop risque",
    "peu risqué", "peu risque", "risqué",
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


def _compile(raw_patterns: list[str]) -> list[re.Pattern[str]]:
    return [re.compile(p, re.IGNORECASE) for p in raw_patterns]


ADVERSARIAL_PATTERNS_JAILBREAK: list[re.Pattern[str]] = _compile(_JAILBREAK_RAW)
ADVERSARIAL_PATTERNS_TRADE_REQUEST: list[re.Pattern[str]] = _compile(_TRADE_REQUEST_RAW)
ADVERSARIAL_PATTERNS_PERSONA_HIJACK: list[re.Pattern[str]] = _compile(_PERSONA_HIJACK_RAW)
ADVERSARIAL_PATTERNS_FINANCIAL_ADVICE: list[re.Pattern[str]] = _compile(_FINANCIAL_ADVICE_RAW)

# Ordered so the most security-critical bucket (jailbreak) is checked first.
ADVERSARIAL_PATTERNS_BY_CATEGORY: dict[str, list[re.Pattern[str]]] = {
    "jailbreak": ADVERSARIAL_PATTERNS_JAILBREAK,
    "trade_request": ADVERSARIAL_PATTERNS_TRADE_REQUEST,
    "persona_hijack": ADVERSARIAL_PATTERNS_PERSONA_HIJACK,
    "financial_advice": ADVERSARIAL_PATTERNS_FINANCIAL_ADVICE,
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
REFUSAL_TEMPLATE: str = (
    "Je suis un outil de description des conditions de marché. Je ne donne pas "
    "de recommandations de trade ni d'évaluations personnalisées. C'est à vous "
    "d'évaluer si les conditions actuelles correspondent à votre méthode et à "
    "votre tolérance au risque.\n\n"
    "Si vous voulez approfondir un élément précis (BOS, FVG, OB, news, régime), "
    "n'hésitez pas à me poser une question descriptive."
)

# Couche 3 — replacement when the LLM output contains a forbidden token.
OUTPUT_CONTAMINATED_TEMPLATE: str = (
    "Je ne peux pas formuler cette réponse de cette manière. Je peux te décrire "
    "les conditions actuelles du marché si tu veux."
)

# Couche 2 — fail-safe when the Anthropic API errors (timeout / rate limit / network).
LLM_ERROR_TEMPLATE: str = (
    "Je ne peux pas répondre pour le moment, le service de description est "
    "temporairement indisponible. Tu peux consulter directement les conditions "
    "du marché sur le dashboard."
)

# Firm redirection used when the user insists on a recommendation (niveau 1.5 rule).
INSIST_REDIRECT_TEMPLATE: str = (
    "Je décris les conditions du marché. La décision d'agir t'appartient."
)


__all__ = [
    "ADVERSARIAL_PATTERNS_BY_CATEGORY",
    "ADVERSARIAL_PATTERNS_FINANCIAL_ADVICE",
    "ADVERSARIAL_PATTERNS_JAILBREAK",
    "ADVERSARIAL_PATTERNS_PERSONA_HIJACK",
    "ADVERSARIAL_PATTERNS_TRADE_REQUEST",
    "ALL_ADVERSARIAL_PATTERNS",
    "ALL_FORBIDDEN_TOKENS",
    "FORBIDDEN_TOKENS_ACTION_TRADING",
    "FORBIDDEN_TOKENS_BY_CATEGORY",
    "FORBIDDEN_TOKENS_JUGEMENT_MOMENT",
    "FORBIDDEN_TOKENS_JUGEMENT_RISQUE",
    "FORBIDDEN_TOKENS_RECOMMANDATION",
    "INSIST_REDIRECT_TEMPLATE",
    "LLM_ERROR_TEMPLATE",
    "OUTPUT_CONTAMINATED_TEMPLATE",
    "REFUSAL_TEMPLATE",
    "normalize_text",
]
