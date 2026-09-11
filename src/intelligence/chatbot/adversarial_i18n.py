"""Couche 1 — adversarial patterns for the seven non-French product locales.

WHY THIS FILE IS SEPARATE FROM ``constants``
--------------------------------------------
``constants`` holds the founder-validated FRENCH core: 5 to 10 patterns per
bucket, reviewed as a set. Folding seven languages into those same lists would
bury the core under a hundred lines of regex and make the "5-10 patterns"
review rule meaningless. Here the extension lives on its own, one block per
language, so a native reviewer can read exactly one block and judge it.

THE TRADE-OFF, STATED PLAINLY
-----------------------------
Every pattern here runs against EVERY message, whatever the visitor's language:
Couche 1 sees the text before anything identifies a locale, and a French user
may well type English. So a false positive in Polish hurts a French customer.

That asymmetry sets the design rule: **precision over recall**.

  · A missed adversarial phrasing costs one model call — and the prompt refuses
    it anyway, in the visitor's language (verified live in de and es before this
    file existed). The product stays correct; it just pays for the turn.
  · A false positive HARD-REFUSES a paying customer's legitimate descriptive
    question, with no model call and no way for the model to recover it. That is
    a visibly broken product.

So each pattern demands an unambiguous frame — a verb of asking-to-act, of
role-play, or of forecasting — plus its object. None fires on a lone keyword,
and none uses a short function word that another language might supply by
accident. Coverage is deliberately partial: the common phrasings, not every
phrasing.

Arabic is the safest block here (a different script cannot collide with the
Latin-language corpora) and uses plain substrings: ``\\b`` around Arabic runs is
unreliable in practice.

NOT NATIVELY REVIEWED. Same status as the translated templates: strictly better
than French-only detection, and shaped so a translator can audit one block.
"""

from __future__ import annotations

# The product FORECASTS VOLATILITY — an amplitude, never a direction — so
# "quelle est la prévision de volatilité ?" is a legitimate product question in
# every language, and the forecast-noun patterns must not swallow it.
#
# This guard is SHARED rather than written per language, because a per-language
# lookahead is a cross-language trap: the Italian pattern's "(?!\\s+di\\s+
# volatilita)" happily matched the FRENCH "prévision DE volatilité" and hard-
# refused it. Caught by the French negatives — the reason they run against every
# bucket, not just their own.
#
# One suffix, every language: they all share the root "volatil" (volatilité,
# volatilidad, volatilità, Volatilität, volatiliteit, volatilidade); Polish uses
# "zmienność" → "zmiennosc" after normalisation.
NO_VOLATILITY = r"(?![^?.!]{0,24}(volatil|zmiennosc))"

# {language: {bucket: [raw patterns, authored against normalize_text() output]}}
#
# normalize_text lower-cases, strips combining accents and normalises
# apostrophes. So "ä"→"a", "é"→"e", "ą"→"a" — but "ł" and "ß" survive, and
# Arabic keeps its letters while losing its harakat.
PATTERNS_BY_LANG: dict[str, dict[str, list[str]]] = {
    # ---------------------------------------------------------------- German #
    "de": {
        "jailbreak": [
            r"\b(ignoriere|vergiss)\s+(alle\s+)?(deine\s+|die\s+)?"
            r"(anweisungen|regeln|vorgaben|einschrankungen)",
            r"\bdu\s+bist\s+(jetzt|ab\s+sofort)\b",
            r"\bantworte\s+ohne\s+(jede\s+|jegliche\s+)?"
            r"(einschrankung|einschrankungen|filter|zensur)",
        ],
        "trade_request": [
            r"\bsoll(te)?\s+ich\s+(jetzt\s+)?"
            r"(kaufen|verkaufen|einsteigen|aussteigen|handeln)\b",
            r"\bgib\s+mir\s+(ein\s+)?(signal|setup|einstieg)\b",
            r"\blong\s+oder\s+short\b",
            r"\bwas\s+soll\s+ich\s+(jetzt\s+)?(tun|machen)\b",
        ],
        "persona_hijack": [
            r"\bverhalte\s+dich\s+wie\s+(ein|eine)\b",
            r"\btu\s+so,?\s+als\s+(ob|warst)\b",
            r"\bantworte\s+als\s+(ein|eine)\b",
            r"\bstell\s+dir\s+vor,?\s+du\s+bist\b",
        ],
        "financial_advice": [
            r"\bmit\s+\d+\s*(k|euro|eur|€|dollar|usd|\$|lots?)\b",
            r"\bmein\s+(stop|stopp|einstieg|kapital|budget|hebel)\b"
            r"[a-z\s]*\b(liegt|ist|betragt)\b",
            r"\bist\s+es\s+(riskant|sicher|gefahrlich)\s+(zu|jetzt|hier)\b",
            r"\bwie\s+viel\s+soll(te)?\s+ich\s+(einsetzen|investieren|riskieren)\b",
        ],
        "prediction": [
            r"\bwird\s+(der\s+|die\s+|das\s+)?(preis|kurs|gold|markt|es)\s+"
            r"(steigen|fallen|sinken|weiter\s+steigen|weiter\s+fallen)\b",
            r"\b(glaubst|denkst|meinst)\s+du\b[^?.!]{0,60}\b"
            r"(steigen|fallen|sinken|hoher|tiefer)\b",
            r"\b(kursziel|preisziel|vorhersage)\b",
            r"\bprognose\b" + NO_VOLATILITY,
            r"\bwohin\s+(geht|lauft)\s+(der\s+)?(preis|kurs|markt)\b",
        ],
    },
    # --------------------------------------------------------------- Spanish #
    "es": {
        "jailbreak": [
            r"\b(ignora|olvida)\s+(todas\s+)?(tus|las)\s+"
            r"(instrucciones|reglas|normas)",
            r"\bahora\s+eres\b",
            r"\bresponde\s+sin\s+(ninguna\s+)?(restriccion|filtro|censura)",
        ],
        "trade_request": [
            r"\b(deberia|debo|puedo)\s+(comprar|vender|entrar|salir|operar)\b",
            r"\bdame\s+(una\s+|un\s+)?(senal|entrada|posicion|setup)\b",
            r"\blargo\s+o\s+corto\b",
            r"\bque\s+hago\s+(ahora|aqui)\b",
        ],
        "persona_hijack": [
            r"\b(actua|comportate)\s+como\s+(un|una)\b",
            r"\bhaz\s+como\s+si\b",
            r"\bresponde\s+como\s+(un|una)\b",
            r"\bimagina\s+que\s+eres\b",
        ],
        "financial_advice": [
            r"\bcon\s+\d+\s*(k|euros?|eur|€|dolares?|usd|\$|lotes?)\b",
            r"\bmi\s+(stop|entrada|capital|presupuesto|apalancamiento)\b"
            r"[a-z\s]*\b(esta|es)\b",
            r"\bes\s+(arriesgado|seguro|peligroso)\s+(entrar|comprar|vender|ahora)\b",
            r"\bcuanto\s+(deberia|debo)\s+(invertir|arriesgar|poner)\b",
        ],
        "prediction": [
            r"\b(va|ira)\s+a\s+(subir|bajar|caer|rebotar|romper)\b",
            r"\b(crees|piensas)\s+que\b[^?.!]{0,60}\b"
            r"(subir|bajar|caer|rebotar|sube|baja)\b",
            r"\b(prediccion|pronostico)\b",
            r"\bprevision\b" + NO_VOLATILITY,
            r"\bobjetivo\s+de\s+precio\b",
            r"\bhasta\s+donde\s+(va|subira|bajara)\b",
        ],
    },
    # --------------------------------------------------------------- Italian #
    "it": {
        "jailbreak": [
            r"\b(ignora|dimentica)\s+(tutte\s+)?(le\s+)?(tue\s+)?(istruzioni|regole)",
            r"\bora\s+sei\b",
            r"\brispondi\s+senza\s+(alcun|alcuna)\s+(restrizione|filtro|censura)",
        ],
        "trade_request": [
            r"\b(dovrei|devo|posso)\s+(comprare|vendere|entrare|uscire|operare)\b",
            r"\bdammi\s+(un\s+)?(segnale|ingresso|setup)\b",
            r"\blungo\s+o\s+corto\b",
            r"\bche\s+faccio\s+(adesso|ora)\b",
        ],
        "persona_hijack": [
            r"\b(comportati|agisci)\s+come\s+(un|una)\b",
            r"\bfai\s+finta\s+di\b",
            r"\brispondi\s+come\s+(un|una)\b",
            r"\bimmagina\s+di\s+essere\b",
        ],
        "financial_advice": [
            r"\bcon\s+\d+\s*(k|euro|eur|€|dollari|usd|\$|lotti?)\b",
            r"\bil\s+mio\s+(stop|ingresso|capitale|budget|leva)\b",
            r"\be\s+(rischioso|sicuro|pericoloso)\s+(entrare|comprare|vendere|adesso)\b",
            r"\bquanto\s+(dovrei|devo)\s+(investire|rischiare|mettere)\b",
        ],
        "prediction": [
            r"\b(salira|scendera|rimbalzera)\b",
            r"\b(pensi|credi)\s+che\b[^?.!]{0,60}\b"
            r"(salira|scendera|sale|scende|rimbalza)\b",
            r"\bpronostico\b",
            r"\bprevision[ei]\b" + NO_VOLATILITY,
            r"\bobiettivo\s+di\s+prezzo\b",
        ],
    },
    # ----------------------------------------------------------------- Dutch #
    "nl": {
        "jailbreak": [
            r"\b(negeer|vergeet)\s+(al\s+)?(je|alle)\s+(instructies|regels)",
            r"\bje\s+bent\s+nu\b",
            r"\bantwoord\s+zonder\s+(enige\s+)?(beperking|filter|censuur)",
        ],
        "trade_request": [
            r"\bmoet\s+ik\s+(nu\s+)?(kopen|verkopen|instappen|uitstappen)\b",
            r"\bgeef\s+me\s+(een\s+)?(signaal|setup|instap)\b",
            r"\blong\s+of\s+short\b",
            r"\bwat\s+moet\s+ik\s+(nu\s+)?doen\b",
        ],
        "persona_hijack": [
            r"\bgedraag\s+je\s+als\s+(een)\b",
            r"\bdoe\s+alsof\s+je\b",
            r"\bantwoord\s+als\s+(een)\b",
            r"\bstel\s+je\s+voor\s+dat\s+je\b",
        ],
        "financial_advice": [
            r"\bmet\s+\d+\s*(k|euro|eur|€|dollar|usd|\$|lots?)\b",
            r"\bmijn\s+(stop|inleg|kapitaal|budget|hefboom)\b",
            r"\bis\s+het\s+(riskant|veilig|gevaarlijk)\s+(om|nu)\b",
            r"\bhoeveel\s+moet\s+ik\s+(inleggen|investeren|riskeren)\b",
        ],
        "prediction": [
            r"\bgaat\s+(het|de\s+prijs|de\s+koers|goud)\s+"
            r"(stijgen|dalen|zakken|omhoog|omlaag)\b",
            r"\b(denk|geloof)\s+je\s+dat\b[^?.!]{0,60}\b"
            r"(stijgt|daalt|stijgen|dalen|omhoog|omlaag)\b",
            r"\b(voorspelling|koersdoel)\b",
            r"\bprognose\b" + NO_VOLATILITY,
        ],
    },
    # ---------------------------------------------------------------- Polish #
    "pl": {
        "jailbreak": [
            r"\b(zignoruj|ignoruj|zapomnij)\s+(o\s+)?(wszystkie\s+|wszystkich\s+)?(swoje\s+|swoich\s+)?"
            r"(instrukcji|instrukcjach|instrukcje|zasadach|zasady)",
            r"\bteraz\s+jestes\b",
            r"\bodpowiadaj\s+bez\s+(zadnych\s+)?(ograniczen|filtrow|cenzury)",
        ],
        "trade_request": [
            r"\bczy\s+(powinienem|powinnam|mam|moge)\s+"
            r"(kupic|sprzedac|wejsc|wyjsc)\b",
            r"\bdaj\s+mi\s+(jakis\s+)?(sygnal|setup|wejscie)\b",
            r"\blong\s+czy\s+short\b",
            r"\bco\s+mam\s+(teraz\s+)?(robic|zrobic)\b",
        ],
        "persona_hijack": [
            r"\bzachowuj\s+sie\s+jak\b",
            r"\budawaj,?\s+ze\s+jestes\b",
            r"\bodpowiadaj\s+jako\b",
            r"\bwyobraz\s+sobie,?\s+ze\s+jestes\b",
        ],
        "financial_advice": [
            r"\bmam\s+\d+\s*(k|euro|eur|€|dolarow|usd|\$|lot(y|ow)?)\b",
            r"\bmoj\s+(stop|kapitał|kapital|budzet|depozyt)\b",
            r"\bczy\s+to\s+(ryzykowne|bezpieczne)\b",
            r"\bile\s+(powinienem|powinnam|mam)\s+"
            r"(zainwestowac|zaryzykowac|wlozyc|włozyc)\b",
        ],
        "prediction": [
            r"\bczy\s+(cena|kurs|złoto|zloto|rynek)\s+"
            r"(wzrosnie|spadnie|odbije|pojdzie)\b",
            r"\b(myslisz|sadzisz),?\s+ze\b[^?.!]{0,60}\b"
            r"(wzrosnie|spadnie|odbije|rosnie|spada)\b",
            r"\b(przewidywanie|przewidywania)\b",
            r"\bprognoz[ay]\b" + NO_VOLATILITY,
            r"\bcel\s+cenowy\b",
        ],
    },
    # ------------------------------------------------------------ Portuguese #
    "pt": {
        "jailbreak": [
            r"\b(ignora|ignore|esquece|esqueca)\s+(todas\s+)?(as\s+)?"
            r"(tuas\s+|suas\s+)?(instrucoes|regras)",
            r"\bagora\s+(es|voce\s+e)\b",
            r"\bresponde\s+sem\s+(qualquer\s+)?(restricao|filtro|censura)",
        ],
        "trade_request": [
            r"\b(devo|deveria|posso)\s+(comprar|vender|entrar|sair)\b",
            r"\bda-?\s?me\s+(um\s+|uma\s+)?(sinal|setup|entrada)\b",
            r"\blongo\s+ou\s+curto\b",
            r"\bo\s+que\s+faco\s+agora\b",
        ],
        "persona_hijack": [
            r"\b(age|comporta-?te)\s+como\s+(um|uma)\b",
            r"\bfaz\s+de\s+conta\s+que\b",
            r"\bresponde\s+como\s+(um|uma)\b",
            r"\bimagina\s+que\s+(es|voce\s+e)\b",
        ],
        "financial_advice": [
            r"\bcom\s+\d+\s*(k|euros?|eur|€|dolares|usd|\$|lotes?)\b",
            r"\bo\s+meu\s+(stop|capital|orcamento|alavancagem)\b",
            r"\be\s+(arriscado|seguro|perigoso)\s+(entrar|comprar|vender|agora)\b",
            r"\bquanto\s+(devo|deveria)\s+(investir|arriscar|meter)\b",
        ],
        "prediction": [
            r"\b(vai|ira)\s+(subir|descer|cair|romper)\b",
            r"\b(achas|acha|pensas)\s+que\b[^?.!]{0,60}\b"
            r"(sobe|desce|subir|descer|cair)\b",
            r"\bprognostico\b",
            r"\bprevisao\b" + NO_VOLATILITY,
            r"\bobjetivo\s+de\s+preco\b",
        ],
    },
    # ---------------------------------------------------------------- Arabic #
    # Plain substrings: word boundaries around Arabic runs are unreliable, and a
    # different script cannot collide with the Latin-language corpora.
    "ar": {
        "jailbreak": [
            r"تجاهل\s+(كل\s+)?(التعليمات|تعليماتك|القواعد)",
            r"انت\s+الان\b",
            r"اجب\s+(بدون|دون)\s+(أي\s+|اي\s+)?(قيود|قيد|رقابة)",
        ],
        "trade_request": [
            r"(هل\s+)?(اشتري|أشتري|ابيع|أبيع)\s+(الان|الآن|الذهب)",
            r"(اعطني|أعطني)\s+(اشارة|إشارة|صفقة|توصية)",
            r"شراء\s+ام\s+بيع",
            r"ماذا\s+(افعل|أفعل)\s+(الان|الآن)",
        ],
        "persona_hijack": [
            r"تصرف\s+(كأنك|كانك|مثل)",
            r"تظاهر\s+(بأنك|بانك)",
            r"(اجب|أجب)\s+(بصفتك|كأنك|كانك)",
            r"تخيل\s+(أنك|انك)\s+",
        ],
        "financial_advice": [
            r"(معي|لدي|لدى)\s+\d+\s*(الف|ألف|دولار|يورو)",
            r"(وقف\s+الخسارة|رأس\s+المال|راس\s+المال)\s+(عندي|لدي)",
            r"هل\s+(من\s+)?(الآمن|الامن|الخطر|المخاطرة)\s+",
            r"كم\s+(يجب|علي|عليّ)\s+(أن\s+|ان\s+)?(استثمر|أستثمر|اخاطر|أخاطر)",
        ],
        "prediction": [
            r"هل\s+(سيرتفع|سينخفض|سيصعد|سيهبط|سيرتد)",
            r"(هل\s+)?(تعتقد|تظن)\s+[^؟?.!]{0,60}(يرتفع|ينخفض|يصعد|يهبط)",
            r"(توقع|توقعاتك|تنبؤ|تنبؤات)\b",
            r"(الهدف\s+السعري|هدف\s+السعر)",
        ],
    },
}

#: The seven locales this file extends Couche 1 to (French lives in constants).
EXTENDED_LOCALES: tuple[str, ...] = tuple(PATTERNS_BY_LANG)

#: The buckets every language block must cover.
BUCKETS: tuple[str, ...] = (
    "jailbreak",
    "trade_request",
    "persona_hijack",
    "financial_advice",
    "prediction",
)


def raw_patterns_for(bucket: str) -> list[str]:
    """Every non-French pattern for ``bucket``, in a stable language order."""
    return [
        pattern
        for lang in EXTENDED_LOCALES
        for pattern in PATTERNS_BY_LANG[lang].get(bucket, [])
    ]


__all__ = ["BUCKETS", "EXTENDED_LOCALES", "PATTERNS_BY_LANG", "raw_patterns_for"]
