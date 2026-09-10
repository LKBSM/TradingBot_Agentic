"""MIA-4S — the landing page's SIMULATED M.I.A agent.

Same orchestrator, same personality, same defences as production — a different
data surface. Nothing about the security chain is re-implemented here:

  · Couche 1 (adversarial input filter)  — the production ``AdversarialFilter``
  · Couche 2 (Haiku + tool use)          — the production ``Chatbot``
  · Couche 3 (forbidden-token output)    — the production ``OutputFilter``
  · Couche 4 (view-action whitelist)     — the production ``ViewActionValidator``

What this module narrows is the TOOL SURFACE and the tail of the system prompt,
through the three additive hooks on ``Chatbot`` (``tool_schemas``,
``tool_handlers``, ``extra_system_blocks``).

Two invariants this module exists to hold:

1. **It is a simulation, and it stays one.** The only market data reachable here
   is the frozen illustration scenario in ``config/demo_illustration.json``. The
   engine, the live provider, candles.db, the market registry and the economic
   calendar are NOT wired in — and the tools that read them are simply not
   declared, so the model cannot call them whatever a visitor talks it into. The
   lock is the code, not the instruction. Widening this needs an explicit
   decision, not a patch.
2. **Product knowledge is quoted, never invented.** Price, FAQ, terms and
   glossary are read at build time from their canonical files. Nothing about the
   product is retyped here — a retyped price is a price that goes stale and
   turns the showcase into a lie.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

from src.intelligence.chatbot.chatbot import TOOL_SCHEMAS, Chatbot

logger = logging.getLogger(__name__)

# src/intelligence/chatbot/demo_agent.py → repo root
REPO_ROOT = Path(__file__).resolve().parents[3]

ILLUSTRATION_PATH = REPO_ROOT / "config" / "demo_illustration.json"
PRICING_PATH = REPO_ROOT / "config" / "pricing.json"
TERMS_PATH = REPO_ROOT / "docs" / "legal" / "conditions-utilisation.md"
MESSAGES_DIR = REPO_ROOT / "webapp" / "messages"
GLOSSARY_PATH = REPO_ROOT / "webapp" / "lib" / "glossary.ts"

DEFAULT_LOCALE = "fr"

# The landing ships in nine languages; an agent that only speaks French would be
# a visible defect on eight of them. The FAQ and the price copy are ALREADY
# translated by the product itself (webapp/messages/<locale>.json), so the
# knowledge block is per-locale at no editorial cost. The glossary and the terms
# of use have a single canonical (French) source — quoted as such, never
# machine-translated here.
LOCALE_LANGUAGE: dict[str, str] = {
    "fr": "français",
    "en": "anglais",
    "de": "allemand",
    "es": "espagnol",
    "it": "italien",
    "nl": "néerlandais",
    "pl": "polonais",
    "pt": "portugais",
    "ar": "arabe",
}

# A parse that silently yields nothing would ship a showcase agent with no
# product knowledge and no way to notice. Below this floor we raise instead.
_MIN_GLOSSARY_ENTRIES = 8

# Chart-view actions the landing demo can actually RENDER. The whitelist itself
# (Couche 4) is unchanged and shared with production; this is the narrower set
# the demo's frozen chart can honour, applied to the model's OUTPUT so the demo
# never claims a change it did not make.
#
# Deliberately absent, because the demo chart cannot do them and a silent no-op
# would let M.I.A describe a move that never happened:
#   · focus_zone / highlight_zone   — the frozen SVG has no camera
#   · filter_zones                  — no zone filtering in the demo
#   · set_instrument_timeframe      — one combination only
# The demo chart toggles LAYERS, so a zone-targeting action is only honoured in
# its `category` form (see DEMO_VIEW_CATEGORY_ACTIONS).
DEMO_VIEW_ACTIONS: frozenset[str] = frozenset(
    {
        "set_layer_visibility",
        "hide_zones",
        "isolate_zones",
        "show_zones",
        "reset_view",
    }
)

# Of those, the ones that target zones. Couche 4 normalises a category mask
# («  masque les FVG ») into the plain `zone_ids` form, resolved against the ids
# the engine actually emitted — so what reaches the front is a list of REAL ids.
# The demo chart has layers, not addressable zones, so the ids are translated
# back to layers here, next to the scenario that defines them.
DEMO_VIEW_ZONE_ACTIONS: frozenset[str] = frozenset(
    {"hide_zones", "isolate_zones", "show_zones"}
)


def zone_layer_index(illustration: dict[str, Any]) -> dict[str, str]:
    """{zone id → demo layer} for the frozen scenario. Derived, never authored."""
    structure = illustration.get("structure", {})
    index: dict[str, str] = {}
    for key, layer in (("order_blocks", "ob"), ("fair_value_gaps", "fvg")):
        for zone in structure.get(key, []) or []:
            if isinstance(zone, dict) and zone.get("id"):
                index[str(zone["id"])] = layer
    for pool in structure.get("liquidity_pools", []) or []:
        if isinstance(pool, dict) and pool.get("id"):
            index[str(pool["id"])] = "liq"
    return index


class DemoKnowledgeError(RuntimeError):
    """A canonical product-knowledge source is missing or unreadable."""


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:  # missing file / unreadable — never a silent empty
        raise DemoKnowledgeError(f"cannot read {path}: {exc}") from exc


def load_illustration() -> dict[str, Any]:
    """The frozen scenario, read from its single source."""
    return json.loads(_read_text(ILLUSTRATION_PATH))


def _glossary_entries() -> list[tuple[str, str]]:
    """Extract (term, short definition) from the webapp's central glossary.

    The glossary is a TypeScript object literal; it is the single source of
    truth for the ⓘ tooltips and the /methodology page, so it is read rather
    than duplicated. A refactor that breaks this parse raises — it never
    degrades to an agent that quietly knows no vocabulary.
    """
    src = _read_text(GLOSSARY_PATH)
    pairs = re.findall(
        r"term:\s*'([^']*)',\s*short:\s*'([^']*)'",
        src,
        re.DOTALL,
    )
    if len(pairs) < _MIN_GLOSSARY_ENTRIES:
        raise DemoKnowledgeError(
            f"glossary parse yielded {len(pairs)} entries (< {_MIN_GLOSSARY_ENTRIES}); "
            f"{GLOSSARY_PATH} changed shape — fix the extraction, do not retype it"
        )
    return [(t.strip(), s.strip()) for t, s in pairs]


# Anything that looks like an e-mail address, so the demo cannot hand one out.
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")

_CONTACT_POINTER = "(coordonnées de l'exploitant : voir la page /conditions)"


def _redact_contacts(text: str) -> str:
    """Strip contact addresses from quoted legal text.

    The terms of use carry the operator's personal e-mail — legally required on
    the PAGE, and public there. But this block feeds a PUBLIC, unauthenticated
    LLM endpoint, and a live probe showed the agent volunteering that address to
    an anonymous visitor who had merely asked how volatility is computed. That
    turns the showcase into an address-harvesting endpoint.

    Redacting at the SOURCE rather than adding a prompt rule: a rule can be
    talked around, and the surest way not to repeat something is not to be told
    it. The information stays one click away, on the page that must display it.
    """
    return _EMAIL_RE.sub(_CONTACT_POINTER, text)


def load_product_knowledge(locale: str = DEFAULT_LOCALE) -> str:
    """Assemble the product-knowledge block from the canonical sources.

    Every line here is QUOTED from a file the site itself renders. The block is
    byte-stable between turns and between visitors of the same locale, so it
    caches. An unknown locale falls back to French rather than inventing copy.
    """
    # The price comes from the billing module, which reads config/pricing.json —
    # the same single source the /abonnement page and Stripe use. Never retyped.
    from src.billing import pricing as billing_pricing

    messages_path = MESSAGES_DIR / f"{locale}.json"
    if not messages_path.is_file():
        messages_path = MESSAGES_DIR / f"{DEFAULT_LOCALE}.json"
    messages = json.loads(_read_text(messages_path))
    home = messages.get("home", {})
    faq = home.get("faq", {})
    pricing_copy = home.get("pricing", {})

    parts: list[str] = [
        "CONNAISSANCE PRODUIT (reprise telle quelle du site — n'invente rien au-delà)",
        "",
        "## Tarif (source : config/pricing.json, via le module de facturation —",
        "le même prix que la page d'abonnement et que le paiement)",
    ]
    for plan in billing_pricing.list_paid_plans():
        amount = f"{plan.amount_usd:g}"
        per_month = f"{plan.monthly_equivalent_usd:g}"
        if plan.cadence == "annual":
            parts.append(
                f"- Abonnement annuel : {amount} {plan.currency} par an, "
                f"soit {per_month} {plan.currency} par mois."
            )
        else:
            parts.append(f"- Abonnement mensuel : {amount} {plan.currency} par mois.")
    parts += [
        "- Un seul abonnement, tout le produit. Résiliable à tout moment depuis le compte.",
        # The price's legal mention is deliberately NOT quoted here. It is
        # rendered next to the price by the page itself, and it contains
        # vocabulary the agent's own rules forbid it from writing — feeding it
        # in would only tempt the model into an answer Couche 3 then destroys.
        "- Les mentions légales du prix sont affichées à côté du prix sur la page "
        "de tarif : renvoie le visiteur vers la page, ne les récite pas.",
    ]

    paid = pricing_copy.get("paid", {})
    features = [paid.get(f"f{i}") for i in range(1, 7)]
    features = [re.sub(r"</?b>", "", f) for f in features if f]
    if features:
        parts += ["- Ce que l'abonnement contient : " + " ; ".join(features) + "."]

    parts += ["", "## Questions fréquentes (source : la FAQ de la page d'accueil)"]
    for i in range(1, 9):
        q, a = faq.get(f"q{i}"), faq.get(f"a{i}")
        if q and a:
            parts += [f"- Q : {q}", f"  R : {re.sub(r'</?b>', '', a)}"]

    parts += [
        "",
        "## Glossaire (source : le glossaire central du produit, celui des info-bulles)",
    ]
    parts += [f"- {term} : {short}" for term, short in _glossary_entries()]

    parts += [
        "",
        "## Conditions d'utilisation — version française canonique",
        "(source : docs/legal/conditions-utilisation.md, le document rendu tel quel",
        "par la page /conditions ; il n'existe pas d'autre version faisant foi)",
        "",
        _redact_contacts(_read_text(TERMS_PATH).strip()),
    ]
    # Belt and braces: the whole block, not just the terms, is swept — a future
    # source could bring an address in through another door.
    return _redact_contacts("\n".join(parts))


def build_scope_block(
    illustration: dict[str, Any], locale: str = DEFAULT_LOCALE
) -> str:
    """The demo's scope rules + the frozen scenario, as a stable system block.

    It sits AFTER the production rules and can only narrow them. It never edits
    a rule above it — and it could not: the tools it says do not exist are the
    tools that were never declared.
    """
    label = illustration.get("instrument_label", illustration.get("instrument"))
    return "\n".join(
        [
            "CADRE DE CETTE CONVERSATION — DÉMONSTRATION PUBLIQUE (simulation)",
            "",
            "- Tu es la même M.I.A que dans le produit, avec exactement les mêmes règles, "
            "sur la page d'accueil du site, face à un visiteur qui n'a pas de compte.",
            "- CORRECTION DE LA SECTION « Tu as accès à 7 tools » CI-DESSUS : dans cette "
            "démonstration tu n'as que 2 outils — get_illustration_reading() et "
            "apply_chart_view(). get_market_reading, get_signal_summary, "
            "get_ob_diagnostic, list_markets, get_economic_calendar et get_publication "
            "N'EXISTENT PAS ici. Partout où les règles ci-dessus te disent d'appeler "
            "get_market_reading, appelle get_illustration_reading à la place.",
            "- Les seules données de marché auxquelles tu as accès sont un SCÉNARIO "
            f"D'ILLUSTRATION FIGÉ ({label}, {illustration.get('timeframe')}). Tu n'as "
            "accès ni aux données en direct, ni aux marchés suivis en production, ni au "
            "calendrier économique, ni à la moindre lecture réelle.",
            "- Ce scénario n'a PAS de date : il n'est ni récent, ni en cours, ni "
            "d'aujourd'hui. Tu ne le décris jamais comme l'état actuel du marché, et tu "
            "n'emploies jamais « en ce moment », « actuellement » ou « vient de » à son sujet.",
            "- Dès qu'un visiteur pourrait confondre ce scénario avec du réel, tu le dis : "
            "« ce sont des données d'illustration, pas le marché en direct ». Si on te "
            "demande le prix actuel de l'or, l'état réel du marché, une actualité, un "
            "chiffre de publication ou un autre marché : tu réponds que la démonstration "
            "ne porte que sur ce scénario figé, et que le produit, lui, lit les marchés "
            "réels. Tu ne donnes AUCUN chiffre qui ne soit pas dans le scénario ci-dessous.",
            "- AVANT TOUTE ACTION D'AFFICHAGE, appelle get_illustration_reading DANS LE "
            "MÊME TOUR, même si le scénario t'a déjà été montré plus haut dans la "
            "conversation : les identifiants de zones ne sont reconnus que s'ils viennent "
            "d'une lecture faite CE tour-ci. Sans cela ton action est rejetée et tu "
            "annoncerais à tort qu'aucune zone n'existe — ce qui serait faux et visible.",
            "- Actions d'affichage disponibles ici, et AUCUNE autre : "
            + ", ".join(sorted(DEMO_VIEW_ACTIONS))
            + ". Le graphique de la démonstration est figé : il n'a ni caméra ni "
            "filtre, et ne montre qu'une combinaison. N'utilise donc jamais "
            "focus_zone, highlight_zone, filter_zones, focus_price, fit_chart ni "
            "set_instrument_timeframe — elles ne produiraient rien à l'écran, et tu "
            "décrirais un changement qui n'a pas eu lieu.",
            "- Toute ta connaissance du produit est dans le bloc CONNAISSANCE PRODUIT "
            "ci-dessous, repris tel quel du site. Si la réponse n'y est pas, tu le dis et "
            "tu renvoies vers la page concernée : tu n'inventes ni prix, ni "
            "fonctionnalité, ni engagement, ni date de disponibilité.",
            "- ATTENTION — ce bloc contient des CITATIONS (FAQ, conditions d'utilisation) "
            "qui emploient des mots que TES PROPRES RÈGLES t'interdisent d'écrire "
            "(acheter, vendre, trader, risqué, garantie…). Tu ne les recopies donc JAMAIS "
            "mot pour mot : tu en donnes le sens avec tes mots à toi, sans ce vocabulaire, "
            "et tu renvoies vers la page pour le texte exact (/abonnement, /conditions). "
            "C'est la page qui affiche ces textes, pas toi. Exemple : à « est-ce que MIA "
            "dit quand acheter ou vendre ? », tu réponds « non — M.I.A ne dit jamais quand "
            "intervenir sur le marché ; elle décrit ce qui a été détecté, la décision "
            "t'appartient », sans reprendre les verbes de la question.",
            "- Tu réponds à n'importe quelle question sur le produit ou sur ce scénario. "
            "Les questions proposées à l'écran ne sont que des amorces, pas une liste fermée.",
            "- LANGUE — CECI REMPLACE la règle « Tu réponds en français » ci-dessus : ce "
            f"visiteur lit le site en {LOCALE_LANGUAGE.get(locale, LOCALE_LANGUAGE[DEFAULT_LOCALE])}, "
            "réponds-lui dans cette langue. Les conditions d'utilisation citées plus bas "
            "n'existent qu'en français : tu peux les expliquer dans la langue du visiteur, "
            "en précisant que seule la version française fait foi.",
            "",
            "SCÉNARIO D'ILLUSTRATION (le même que celui dessiné à l'écran) :",
            json.dumps(illustration, ensure_ascii=False),
        ]
    )


def _apply_chart_view_schema() -> dict[str, Any]:
    """The PRODUCTION apply_chart_view schema, by reference — never a copy.

    Reusing the object means a change to the production description reaches the
    demo automatically; a duplicated description would drift on the first edit.
    """
    for schema in TOOL_SCHEMAS:
        if schema.get("name") == "apply_chart_view":
            return schema
    raise DemoKnowledgeError("apply_chart_view is no longer in TOOL_SCHEMAS")


def demo_tool_schemas() -> list[dict[str, Any]]:
    """The demo's entire tool surface. Anything absent here is unreachable."""
    return [
        {
            "name": "get_illustration_reading",
            "description": (
                "Le scénario d'illustration FIGÉ de la démonstration (Or, 15 minutes) : "
                "structure SMC (CHOCH, BOS), Order Blocks, Fair Value Gaps, poches de "
                "liquidité avec leurs identifiants réels, et le régime de volatilité. "
                "C'est la SEULE source de données de marché disponible ici, et elle ne "
                "décrit aucun marché réel ni aucune date. Appelle-le avant de décrire "
                "quoi que ce soit du graphique, et avant toute action d'affichage visant "
                "une zone précise (les identifiants viennent d'ici, jamais de ton cru)."
            ),
            "input_schema": {"type": "object", "properties": {}, "required": []},
        },
        _apply_chart_view_schema(),
    ]


def build_demo_chatbot(
    anthropic_client: Any,
    *,
    locale: str = DEFAULT_LOCALE,
    illustration: Optional[dict[str, Any]] = None,
) -> Chatbot:
    """Build the landing's simulated agent on the production orchestrator."""
    from src.intelligence.chatbot.adversarial_filter import AdversarialFilter
    from src.intelligence.chatbot.output_filter import OutputFilter

    scenario = illustration if illustration is not None else load_illustration()

    class _FrozenSummaryProvider:
        """Stands in for the live signal summary — the frozen scenario, nothing else."""

        @staticmethod
        def get() -> dict[str, Any]:
            return {
                "is_illustration": True,
                "note": (
                    "Démonstration : un seul scénario figé, aucune lecture réelle, "
                    "aucun marché suivi."
                ),
                "instruments_tracked": [],
            }

    chatbot = Chatbot(
        anthropic_client=anthropic_client,
        summary_provider=_FrozenSummaryProvider(),  # type: ignore[arg-type]
        assembler=None,  # no engine here, on purpose
        adversarial_filter=AdversarialFilter(),
        output_filter=OutputFilter(),
        tool_schemas=demo_tool_schemas(),
        tool_handlers={"get_illustration_reading": lambda _input: dict(scenario)},
        extra_system_blocks=[
            build_scope_block(scenario, locale),
            load_product_knowledge(locale),
        ],
        # The verbatim safety templates (Couches 1-4) answer in the visitor's
        # language too. A hard refusal short-circuits the model, so without this
        # an English visitor asking for a forecast got French.
        locale=locale,
    )
    logger.info(
        "Demo chatbot (MIA-4S, simulation) built — locale=%s, %d tools, scenario '%s'",
        locale,
        len(demo_tool_schemas()),
        scenario.get("scenario"),
    )
    return chatbot


class DemoAgentRegistry:
    """One simulated agent per locale, built on first use and kept.

    The nine agents share the production static prefix (identical bytes, so one
    cached prefix) and differ only in the scope + knowledge blocks. Building
    lazily means a landing nobody visits in Polish never pays for a Polish agent.
    """

    def __init__(self, anthropic_client: Any) -> None:
        self._client = anthropic_client
        self._agents: dict[str, Chatbot] = {}
        self._illustration = load_illustration()

    def for_locale(self, locale: Optional[str]) -> Chatbot:
        key = locale if locale in LOCALE_LANGUAGE else DEFAULT_LOCALE
        agent = self._agents.get(key)
        if agent is None:
            agent = build_demo_chatbot(
                self._client, locale=key, illustration=self._illustration
            )
            self._agents[key] = agent
        return agent


__all__ = [
    "DEFAULT_LOCALE",
    "DEMO_VIEW_ACTIONS",
    "LOCALE_LANGUAGE",
    "DemoAgentRegistry",
    "DemoKnowledgeError",
    "build_demo_chatbot",
    "build_scope_block",
    "demo_tool_schemas",
    "load_illustration",
    "load_product_knowledge",
]
