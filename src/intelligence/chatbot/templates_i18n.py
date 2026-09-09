"""Chantier 4 — the verbatim safety templates, in the nine product locales.

WHY THIS FILE EXISTS
--------------------
Couches 1, 2, 3 and 4 answer with a FIXED text, returned to the user without
passing through the model. Those texts were French-only, so an English visitor
who asked a forecast question — or whose answer got filtered — received French.
Nothing was wrong with the defence; the defence simply could not speak.

TWO RULES THAT MUST HOLD FOR EVERY TRANSLATION
----------------------------------------------
1. **Free of forbidden tokens.** These strings are returned WITHOUT passing
   through Couche 3, so the invariant "the chatbot never emits a forbidden
   token" holds for its own safety nets only if the texts are clean by
   construction. Enforced by a test over every locale × every template.
2. **Same promise, not a looser one.** A translation may not soften a refusal,
   add a hedge, or promise something the French does not. The French text
   remains AUTHORITATIVE: when the two disagree, the French is right.

Translations are machine-authored and have NOT been reviewed by native speakers.
They are strictly better than serving French to eight locales, and they are the
right thing to hand a translator: one file, one dict, one string per locale.

``INSIST_REDIRECT_TEMPLATE`` is deliberately absent: it is quoted INSIDE the
system prompt as an example for the model, never returned verbatim, so the model
already renders it in the visitor's language.
"""

from __future__ import annotations

from typing import Optional

DEFAULT_LOCALE = "fr"

#: Locales the product ships. Anything outside this falls back to French.
SUPPORTED_LOCALES: tuple[str, ...] = (
    "fr", "en", "de", "es", "it", "nl", "pl", "pt", "ar",
)

# --------------------------------------------------------------------------- #
# Couche 1 — generic refusal (jailbreak / trade_request / persona_hijack /
# financial_advice). Vouvoiement in French, matching the original.
# --------------------------------------------------------------------------- #
REFUSAL: dict[str, str] = {
    "fr": (
        "Je suis un outil de description des conditions de marché. Je ne donne pas "
        "de recommandations d'action ni d'évaluations personnalisées. C'est à vous "
        "d'évaluer si les conditions actuelles correspondent à votre méthode et à "
        "vos propres critères.\n\n"
        "Si vous voulez approfondir un élément précis (BOS, FVG, OB, news, régime), "
        "n'hésitez pas à me poser une question descriptive."
    ),
    "en": (
        "I am a tool that describes market conditions. I do not give action "
        "recommendations or personalised assessments. It is for you to judge "
        "whether current conditions match your own method and criteria.\n\n"
        "If you want to go deeper on a specific element (BOS, FVG, OB, news, "
        "regime), feel free to ask me a descriptive question."
    ),
    "de": (
        "Ich bin ein Werkzeug zur Beschreibung von Marktbedingungen. Ich gebe keine "
        "Handlungsempfehlungen und keine persönlichen Einschätzungen ab. Ob die "
        "aktuellen Bedingungen zu Ihrer Methode und Ihren eigenen Kriterien passen, "
        "beurteilen Sie selbst.\n\n"
        "Wenn Sie ein bestimmtes Element vertiefen möchten (BOS, FVG, OB, News, "
        "Regime), stellen Sie mir gern eine beschreibende Frage."
    ),
    "es": (
        "Soy una herramienta que describe las condiciones del mercado. No doy "
        "recomendaciones de actuación ni valoraciones personalizadas. Le "
        "corresponde a usted juzgar si las condiciones actuales encajan con su "
        "método y sus propios criterios.\n\n"
        "Si quiere profundizar en un elemento concreto (BOS, FVG, OB, noticias, "
        "régimen), pregúnteme de forma descriptiva."
    ),
    "it": (
        "Sono uno strumento che descrive le condizioni di mercato. Non fornisco "
        "raccomandazioni operative né valutazioni personalizzate. Sta a lei "
        "giudicare se le condizioni attuali corrispondono al suo metodo e ai suoi "
        "criteri.\n\n"
        "Se vuole approfondire un elemento preciso (BOS, FVG, OB, news, regime), "
        "mi faccia pure una domanda descrittiva."
    ),
    "nl": (
        "Ik ben een hulpmiddel dat marktomstandigheden beschrijft. Ik geef geen "
        "handelingsaanbevelingen en geen persoonlijke beoordelingen. Het is aan u "
        "om te beoordelen of de huidige omstandigheden bij uw eigen methode en "
        "criteria passen.\n\n"
        "Wilt u dieper ingaan op een specifiek element (BOS, FVG, OB, nieuws, "
        "regime), stel me dan gerust een beschrijvende vraag."
    ),
    "pl": (
        "Jestem narzędziem opisującym warunki rynkowe. Nie wydaję zaleceń "
        "dotyczących działań ani spersonalizowanych ocen. To Pan/Pani ocenia, czy "
        "obecne warunki odpowiadają Pana/Pani metodzie i własnym kryteriom.\n\n"
        "Jeśli chce Pan/Pani zgłębić konkretny element (BOS, FVG, OB, wiadomości, "
        "reżim), proszę zadać pytanie opisowe."
    ),
    "pt": (
        "Sou uma ferramenta que descreve as condições de mercado. Não dou "
        "recomendações de atuação nem avaliações personalizadas. Cabe-lhe a si "
        "avaliar se as condições atuais correspondem ao seu método e aos seus "
        "próprios critérios.\n\n"
        "Se quiser aprofundar um elemento específico (BOS, FVG, OB, notícias, "
        "regime), faça-me uma pergunta descritiva."
    ),
    "ar": (
        "أنا أداة لوصف أحوال السوق. لا أقدّم توصيات بالتصرّف ولا تقييمات شخصية. "
        "أنت من يقرّر ما إذا كانت الأحوال الحالية تناسب منهجك ومعاييرك الخاصة.\n\n"
        "إذا أردت التعمّق في عنصر محدّد (BOS، FVG، OB، الأخبار، النظام السعري)، "
        "اطرح عليّ سؤالاً وصفياً."
    ),
}

# --------------------------------------------------------------------------- #
# Couche 1 — the ``prediction`` bucket. Speaks about FORECASTING, not advice.
# --------------------------------------------------------------------------- #
PREDICTION_REFUSAL: dict[str, str] = {
    "fr": (
        "Je ne prédis pas les mouvements de prix. C'est un choix de conception, pas "
        "une limite technique : personne ne sait ce que le prix fera ensuite, et un "
        "outil qui prétendrait le savoir vous mentirait.\n\n"
        "Ce que je peux faire : décrire l'état d'une zone et son historique, "
        "expliquer un concept (BOS, CHOCH, Order Block, Fair Value Gap), ou vous "
        "montrer ce que le moteur a détecté."
    ),
    "en": (
        "I do not predict price moves. That is a design decision, not a technical "
        "limit: nobody knows what price will do next, and a tool that claimed to "
        "know would be lying to you.\n\n"
        "What I can do: describe the state of a zone and its history, explain a "
        "concept (BOS, CHOCH, Order Block, Fair Value Gap), or show you what the "
        "engine detected."
    ),
    "de": (
        "Ich sage Kursbewegungen nicht voraus. Das ist eine bewusste "
        "Design-Entscheidung, keine technische Grenze: niemand weiß, was der Kurs "
        "als Nächstes tut, und ein Werkzeug, das dies zu wissen vorgäbe, würde Sie "
        "belügen.\n\n"
        "Was ich kann: den Zustand einer Zone und ihre Historie beschreiben, ein "
        "Konzept erklären (BOS, CHOCH, Order Block, Fair Value Gap), oder Ihnen "
        "zeigen, was die Engine erkannt hat."
    ),
    "es": (
        "No predigo los movimientos del precio. Es una decisión de diseño, no un "
        "límite técnico: nadie sabe qué hará el precio a continuación, y una "
        "herramienta que dijera saberlo le estaría mintiendo.\n\n"
        "Lo que sí puedo hacer: describir el estado de una zona y su historial, "
        "explicar un concepto (BOS, CHOCH, Order Block, Fair Value Gap), o "
        "mostrarle lo que ha detectado el motor."
    ),
    "it": (
        "Non prevedo i movimenti del prezzo. È una scelta di progettazione, non un "
        "limite tecnico: nessuno sa cosa farà il prezzo dopo, e uno strumento che "
        "pretendesse di saperlo le starebbe mentendo.\n\n"
        "Quello che posso fare: descrivere lo stato di una zona e la sua storia, "
        "spiegare un concetto (BOS, CHOCH, Order Block, Fair Value Gap), o "
        "mostrarle ciò che il motore ha rilevato."
    ),
    "nl": (
        "Ik voorspel geen prijsbewegingen. Dat is een ontwerpkeuze, geen technische "
        "beperking: niemand weet wat de prijs hierna doet, en een hulpmiddel dat "
        "beweert het te weten, zou tegen u liegen.\n\n"
        "Wat ik wel kan: de toestand van een zone en haar geschiedenis beschrijven, "
        "een concept uitleggen (BOS, CHOCH, Order Block, Fair Value Gap), of u "
        "tonen wat de engine heeft gedetecteerd."
    ),
    "pl": (
        "Nie przewiduję ruchów ceny. To decyzja projektowa, a nie ograniczenie "
        "techniczne: nikt nie wie, co cena zrobi dalej, a narzędzie, które "
        "twierdziłoby inaczej, okłamywałoby Pana/Panią.\n\n"
        "Co mogę zrobić: opisać stan strefy i jej historię, wyjaśnić pojęcie (BOS, "
        "CHOCH, Order Block, Fair Value Gap) albo pokazać, co wykrył silnik."
    ),
    "pt": (
        "Não faço previsões dos movimentos do preço. É uma decisão de conceção, não "
        "um limite técnico: ninguém sabe o que o preço fará a seguir, e uma "
        "ferramenta que afirmasse sabê-lo estaria a mentir-lhe.\n\n"
        "O que posso fazer: descrever o estado de uma zona e o seu historial, "
        "explicar um conceito (BOS, CHOCH, Order Block, Fair Value Gap), ou "
        "mostrar-lhe o que o motor detetou."
    ),
    "ar": (
        "أنا لا أتنبّأ بحركة السعر. هذا اختيار في التصميم وليس حدّاً تقنياً: لا أحد "
        "يعرف ما سيفعله السعر بعد ذلك، وأي أداة تدّعي معرفة ذلك إنما تكذب عليك.\n\n"
        "ما يمكنني فعله: وصف حالة منطقة وتاريخها، أو شرح مفهوم (BOS، CHOCH، "
        "Order Block، Fair Value Gap)، أو عرض ما رصده المحرّك."
    ),
}

# --------------------------------------------------------------------------- #
# Couche 3 — the answer drifted into judgement and was replaced.
# --------------------------------------------------------------------------- #
OUTPUT_CONTAMINATED: dict[str, str] = {
    "fr": (
        "Je ne peux pas formuler cette réponse de cette manière. Je peux te décrire "
        "les conditions actuelles du marché si tu veux."
    ),
    "en": (
        "I cannot phrase that answer this way. I can describe the current market "
        "conditions for you if you like."
    ),
    "de": (
        "Ich kann diese Antwort so nicht formulieren. Gern beschreibe ich dir "
        "stattdessen die aktuellen Marktbedingungen."
    ),
    "es": (
        "No puedo formular esa respuesta de esta manera. Si quieres, puedo "
        "describirte las condiciones actuales del mercado."
    ),
    "it": (
        "Non posso formulare questa risposta in questo modo. Se vuoi, posso "
        "descriverti le condizioni attuali del mercato."
    ),
    "nl": (
        "Ik kan dat antwoord zo niet formuleren. Ik kan je wel de huidige "
        "marktomstandigheden beschrijven als je wilt."
    ),
    "pl": (
        "Nie mogę sformułować tej odpowiedzi w ten sposób. Mogę natomiast opisać "
        "obecne warunki rynkowe, jeśli chcesz."
    ),
    "pt": (
        "Não posso formular essa resposta desta forma. Se quiseres, posso "
        "descrever-te as condições atuais do mercado."
    ),
    "ar": (
        "لا يمكنني صياغة هذه الإجابة بهذه الطريقة. يمكنني أن أصف لك أحوال السوق "
        "الحالية إن أردت."
    ),
}

# --------------------------------------------------------------------------- #
# Couche 2 — fail-safe when the model call errors (timeout / rate limit / network).
# --------------------------------------------------------------------------- #
LLM_ERROR: dict[str, str] = {
    "fr": (
        "Je ne peux pas répondre pour le moment, le service de description est "
        "temporairement indisponible. Tu peux consulter directement les conditions "
        "du marché sur le dashboard."
    ),
    "en": (
        "I cannot answer right now — the description service is temporarily "
        "unavailable. You can check the market conditions directly on the dashboard."
    ),
    "de": (
        "Ich kann im Moment nicht antworten, der Beschreibungsdienst ist "
        "vorübergehend nicht verfügbar. Die Marktbedingungen findest du direkt im "
        "Dashboard."
    ),
    "es": (
        "No puedo responder en este momento: el servicio de descripción no está "
        "disponible temporalmente. Puedes consultar las condiciones del mercado "
        "directamente en el panel."
    ),
    "it": (
        "Non posso rispondere in questo momento: il servizio di descrizione è "
        "temporaneamente non disponibile. Puoi consultare le condizioni di mercato "
        "direttamente nella dashboard."
    ),
    "nl": (
        "Ik kan nu niet antwoorden, de beschrijvingsdienst is tijdelijk niet "
        "beschikbaar. Je kunt de marktomstandigheden rechtstreeks op het dashboard "
        "bekijken."
    ),
    "pl": (
        "Nie mogę teraz odpowiedzieć — usługa opisu jest chwilowo niedostępna. "
        "Warunki rynkowe możesz sprawdzić bezpośrednio w panelu."
    ),
    "pt": (
        "Não consigo responder neste momento: o serviço de descrição está "
        "temporariamente indisponível. Podes consultar as condições do mercado "
        "diretamente no painel."
    ),
    "ar": (
        "لا أستطيع الإجابة في الوقت الحالي، فخدمة الوصف غير متاحة مؤقتاً. يمكنك "
        "الاطّلاع على أحوال السوق مباشرة من لوحة المعلومات."
    ),
}

# --------------------------------------------------------------------------- #
# Couche 4 — a chart-view action outside the display-only whitelist.
# --------------------------------------------------------------------------- #
VIEW_ACTION_REFUSAL: dict[str, str] = {
    "fr": (
        "Je n'invente pas de structure — je n'affiche que ce que le marché montre. "
        "Je peux masquer, filtrer, ou me centrer sur les zones détectées."
    ),
    "en": (
        "I do not invent structures — I only display what the market shows. I can "
        "hide, filter, or centre on the detected zones."
    ),
    "de": (
        "Ich erfinde keine Strukturen — ich zeige nur, was der Markt hergibt. Ich "
        "kann erkannte Zonen ausblenden, filtern oder zentrieren."
    ),
    "es": (
        "No invento estructuras: solo muestro lo que el mercado enseña. Puedo "
        "ocultar, filtrar o centrarme en las zonas detectadas."
    ),
    "it": (
        "Non invento strutture: mostro solo ciò che il mercato presenta. Posso "
        "nascondere, filtrare o centrare le zone rilevate."
    ),
    "nl": (
        "Ik verzin geen structuren — ik toon alleen wat de markt laat zien. Ik kan "
        "gedetecteerde zones verbergen, filteren of centreren."
    ),
    "pl": (
        "Nie wymyślam struktur — pokazuję tylko to, co widać na rynku. Mogę ukryć, "
        "przefiltrować lub wyśrodkować wykryte strefy."
    ),
    "pt": (
        "Não invento estruturas — mostro apenas o que o mercado apresenta. Posso "
        "ocultar, filtrar ou centrar nas zonas detetadas."
    ),
    "ar": (
        "أنا لا أختلق البُنى — أعرض فقط ما يُظهره السوق. يمكنني إخفاء المناطق "
        "المرصودة أو تصفيتها أو التمركز عليها."
    ),
}

# --------------------------------------------------------------------------- #
# Couche 4 — a category mask that resolves to zero engine-emitted structures.
# --------------------------------------------------------------------------- #
VIEW_ACTION_EMPTY_CATEGORY: dict[str, str] = {
    "fr": (
        "Le moteur n'émet aucune structure de cette catégorie sur la lecture "
        "actuelle — il n'y a rien à masquer ni à ré-afficher. Je n'affiche que ce "
        "que le marché montre."
    ),
    "en": (
        "The engine emits no structure of that category on the current reading — "
        "there is nothing to hide or restore. I only display what the market shows."
    ),
    "de": (
        "Die Engine liefert in der aktuellen Lesung keine Struktur dieser Kategorie "
        "— es gibt nichts auszublenden oder wieder einzublenden. Ich zeige nur, was "
        "der Markt hergibt."
    ),
    "es": (
        "El motor no emite ninguna estructura de esa categoría en la lectura "
        "actual: no hay nada que ocultar ni que volver a mostrar. Solo muestro lo "
        "que el mercado enseña."
    ),
    "it": (
        "Il motore non emette alcuna struttura di questa categoria nella lettura "
        "attuale: non c'è nulla da nascondere né da ripristinare. Mostro solo ciò "
        "che il mercato presenta."
    ),
    "nl": (
        "De engine levert in de huidige lezing geen enkele structuur van die "
        "categorie — er is niets te verbergen of te herstellen. Ik toon alleen wat "
        "de markt laat zien."
    ),
    "pl": (
        "Silnik nie zwraca żadnej struktury tej kategorii w bieżącym odczycie — nie "
        "ma czego ukrywać ani przywracać. Pokazuję tylko to, co widać na rynku."
    ),
    "pt": (
        "O motor não emite qualquer estrutura dessa categoria na leitura atual — "
        "não há nada para ocultar nem para repor. Mostro apenas o que o mercado "
        "apresenta."
    ),
    "ar": (
        "لا يُصدر المحرّك أي بنية من هذه الفئة في القراءة الحالية — لا شيء لإخفائه "
        "أو لإعادة إظهاره. أنا أعرض فقط ما يُظهره السوق."
    ),
}

# --------------------------------------------------------------------------- #
# MIA-4S — the landing demo's quota messages. Not a defence layer, but returned
# verbatim on the same path, so they get the same treatment: a visitor who hits
# the cap must be told why in their own language.
# --------------------------------------------------------------------------- #
QUOTA_SESSION_LIMIT: dict[str, str] = {
    "fr": (
        "On s'arrête ici pour la démonstration : elle est limitée à quelques "
        "questions par visiteur. Ce que tu viens de voir tourne sur un scénario "
        "figé — dans le produit, M.I.A lit les marchés réels et la conversation "
        "n'est pas limitée."
    ),
    "en": (
        "That is where the demo stops — it is limited to a few questions per "
        "visitor. What you just saw runs on a frozen scenario; in the product, "
        "M.I.A reads real markets and the conversation is not capped."
    ),
    "de": (
        "Hier endet die Demo — sie ist auf einige Fragen pro Besucher begrenzt. "
        "Was du gerade gesehen hast, läuft auf einem festen Szenario; im Produkt "
        "liest M.I.A echte Märkte, und das Gespräch ist nicht begrenzt."
    ),
    "es": (
        "Aquí termina la demostración: está limitada a unas pocas preguntas por "
        "visitante. Lo que acabas de ver funciona sobre un escenario fijo; en el "
        "producto, M.I.A lee mercados reales y la conversación no tiene límite."
    ),
    "it": (
        "La dimostrazione si ferma qui: è limitata a poche domande per visitatore. "
        "Quello che hai visto gira su uno scenario fisso; nel prodotto M.I.A legge "
        "mercati reali e la conversazione non ha limiti."
    ),
    "nl": (
        "Hier stopt de demo — hij is beperkt tot enkele vragen per bezoeker. Wat "
        "je net zag draait op een vast scenario; in het product leest M.I.A echte "
        "markten en is het gesprek niet begrensd."
    ),
    "pl": (
        "Tu kończy się pokaz — jest ograniczony do kilku pytań na odwiedzającego. "
        "To, co widzisz, działa na zamrożonym scenariuszu; w produkcie M.I.A czyta "
        "prawdziwe rynki, a rozmowa nie ma limitu."
    ),
    "pt": (
        "A demonstração termina aqui: está limitada a algumas perguntas por "
        "visitante. O que acabaste de ver corre sobre um cenário fixo; no produto, "
        "a M.I.A lê mercados reais e a conversa não tem limite."
    ),
    "ar": (
        "تتوقّف التجربة هنا: فهي محدودة ببضعة أسئلة لكل زائر. ما رأيته يعمل على "
        "سيناريو ثابت؛ أما في المنتج فتقرأ M.I.A أسواقاً حقيقية والمحادثة غير محدودة."
    ),
}

QUOTA_IP_LIMIT: dict[str, str] = {
    "fr": (
        "Beaucoup de questions sont arrivées depuis cette connexion en peu de "
        "temps. La démonstration se met en pause un moment."
    ),
    "en": (
        "A lot of questions have come from this connection in a short time. The "
        "demo is pausing for a while."
    ),
    "de": (
        "Von dieser Verbindung kamen in kurzer Zeit sehr viele Fragen. Die Demo "
        "pausiert eine Weile."
    ),
    "es": (
        "Han llegado muchas preguntas desde esta conexión en poco tiempo. La "
        "demostración se pausa un rato."
    ),
    "it": (
        "Sono arrivate molte domande da questa connessione in poco tempo. La "
        "dimostrazione si mette in pausa per un po'."
    ),
    "nl": (
        "Er kwamen in korte tijd veel vragen vanaf deze verbinding. De demo pauzeert "
        "even."
    ),
    "pl": (
        "Z tego połączenia napłynęło w krótkim czasie wiele pytań. Pokaz robi "
        "chwilową przerwę."
    ),
    "pt": (
        "Chegaram muitas perguntas desta ligação em pouco tempo. A demonstração "
        "faz uma pausa."
    ),
    "ar": (
        "وردت أسئلة كثيرة من هذا الاتصال في وقت قصير. تتوقّف التجربة مؤقتاً."
    ),
}

QUOTA_DAILY_BUDGET: dict[str, str] = {
    "fr": (
        "La démonstration a atteint son quota du jour. Les onglets de la page "
        "restent utilisables, et le produit, lui, n'est pas concerné."
    ),
    "en": (
        "The demo has reached its quota for today. The tabs on this page still "
        "work, and the product itself is unaffected."
    ),
    "de": (
        "Die Demo hat ihr Tageskontingent erreicht. Die Reiter dieser Seite "
        "funktionieren weiterhin, das Produkt selbst ist nicht betroffen."
    ),
    "es": (
        "La demostración ha alcanzado su cuota del día. Las pestañas de la página "
        "siguen funcionando y el producto no se ve afectado."
    ),
    "it": (
        "La dimostrazione ha raggiunto la quota giornaliera. Le schede della pagina "
        "restano utilizzabili e il prodotto non è coinvolto."
    ),
    "nl": (
        "De demo heeft zijn quotum voor vandaag bereikt. De tabbladen op deze "
        "pagina werken gewoon, en het product zelf staat hier los van."
    ),
    "pl": (
        "Pokaz wyczerpał dzienny limit. Zakładki na tej stronie nadal działają, a "
        "sam produkt nie jest tym objęty."
    ),
    "pt": (
        "A demonstração atingiu a quota do dia. Os separadores da página continuam "
        "utilizáveis e o produto não é afetado."
    ),
    "ar": (
        "بلغت التجربة حصّتها لهذا اليوم. تبويبات الصفحة ما زالت تعمل، والمنتج نفسه "
        "غير متأثّر."
    ),
}

#: Every localised family, by the constant name it backs.
TEMPLATES_BY_NAME: dict[str, dict[str, str]] = {
    "QUOTA_SESSION_LIMIT": QUOTA_SESSION_LIMIT,
    "QUOTA_IP_LIMIT": QUOTA_IP_LIMIT,
    "QUOTA_DAILY_BUDGET": QUOTA_DAILY_BUDGET,
    "REFUSAL_TEMPLATE": REFUSAL,
    "PREDICTION_REFUSAL_TEMPLATE": PREDICTION_REFUSAL,
    "OUTPUT_CONTAMINATED_TEMPLATE": OUTPUT_CONTAMINATED,
    "LLM_ERROR_TEMPLATE": LLM_ERROR,
    "VIEW_ACTION_REFUSAL_TEMPLATE": VIEW_ACTION_REFUSAL,
    "VIEW_ACTION_EMPTY_CATEGORY_TEMPLATE": VIEW_ACTION_EMPTY_CATEGORY,
}


def template_for(name: str, locale: Optional[str] = None) -> str:
    """The template ``name`` in ``locale``, falling back to French.

    A locale we do not ship, or a gap in a family, yields the French text: the
    caller always gets a real refusal, never an empty string or a KeyError. A
    silent fallback is right HERE — a defence layer must answer something.
    """
    family = TEMPLATES_BY_NAME[name]
    if locale and locale in family:
        return family[locale]
    return family[DEFAULT_LOCALE]


__all__ = [
    "DEFAULT_LOCALE",
    "LLM_ERROR",
    "OUTPUT_CONTAMINATED",
    "PREDICTION_REFUSAL",
    "REFUSAL",
    "SUPPORTED_LOCALES",
    "TEMPLATES_BY_NAME",
    "VIEW_ACTION_EMPTY_CATEGORY",
    "VIEW_ACTION_REFUSAL",
    "template_for",
]
