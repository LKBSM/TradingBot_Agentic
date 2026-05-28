# DG-114-REDUCED — 3 questions suggérées contextuelles

**Effort** : ~6-8h · **Sprint** : S4 · **Owner** : code

---

## Objectif

Afficher **3 questions suggérées contextuelles** dans le chatbot UI, choisies dynamiquement selon le contexte du signal en cours. Démontre la valeur du chatbot en <5s.

**Note** : version réduite (3 au lieu de 6 questions du brief original). Les 3 questions supplémentaires sont reportées V2 (DG-114-FULL).

## Contexte

Sans questions suggérées, un prospect ouvre le chatbot, voit un champ vide, et ne sait pas quoi écrire. Avec 3 questions contextuelles, il clique en 1 seconde et expérimente immédiatement la valeur.

## Périmètre

**IN — 3 questions par défaut** :
1. "Pourquoi la conviction n'est que de **{X}** ?" (dynamique selon conviction)
2. "C'est quoi **{terme_dominant}**, en simple ?" (dynamique : terme le plus contributeur)
3. Une 3e contextuelle selon priorité :
   - Si event ≤ 4h : "Le **{event}** dans **{N minutes}**, ça change quoi ?"
   - Sinon si conviction ≥ 70 : "Ça ressemble à quoi historiquement, ce setup ?"
   - Sinon : "Quelle est ta marge d'erreur sur cette lecture ?"

**OUT (V2 — DG-114-FULL)** :
- 6 questions au lieu de 3
- Personalisation selon tier user
- Apprentissage des questions populaires per-instrument

## Dépendances

- DG-110 chatbot wiring opérationnel
- DG-101 renderer (pour intégrer dans le UI chatbot panel)

## Fichiers à toucher

```
backend/src/intelligence/chatbot/
└── suggested_questions.py            (à créer)

backend/src/api/routes/
└── chat.py                           (extension — endpoint /chat/suggestions)

frontend/components/
├── ChatbotPanel.tsx                  (extension — affichage chips suggérés)
└── SuggestedQuestionChip.tsx         (à créer)
```

## Implémentation

### 1. Backend — derivation des 3 questions

```python
# backend/src/intelligence/chatbot/suggested_questions.py
from src.api.insight_signal_v2 import InsightSignalV2

def derive_suggested_questions(signal: InsightSignalV2, lang: str = "fr") -> list[str]:
    """3 questions contextuelles dynamiques."""
    questions = []

    # Q1: toujours — décomposition conviction
    if lang == "fr":
        questions.append(f"Pourquoi la conviction n'est que de {signal.conviction_0_100} ?")
    else:
        questions.append(f"Why is conviction only {signal.conviction_0_100}?")

    # Q2: jargon dominant
    top_component = max(signal.breakdown_components, key=lambda c: c.contribution)
    glossary_term = component_to_layman_term(top_component.name, lang)
    if lang == "fr":
        questions.append(f"C'est quoi {glossary_term}, en simple ?")
    else:
        questions.append(f"What is {glossary_term}, in plain terms?")

    # Q3: priorité contextuelle
    if signal.event_readout.next_event_in_minutes and signal.event_readout.next_event_in_minutes < 240:
        time_str = format_time(signal.event_readout.next_event_in_minutes, lang)
        evt = signal.event_readout.next_event_label
        if lang == "fr":
            questions.append(f"Le {evt} dans {time_str}, ça change quoi ?")
        else:
            questions.append(f"The {evt} in {time_str}, what does it change?")
    elif signal.conviction_0_100 >= 70:
        if lang == "fr":
            questions.append("Ça ressemble à quoi historiquement, ce setup ?")
        else:
            questions.append("What does this setup look like historically?")
    else:
        if lang == "fr":
            questions.append("Quelle est ta marge d'erreur sur cette lecture ?")
        else:
            questions.append("What's your margin of error on this reading?")

    return questions


COMPONENT_LAYMAN_FR = {
    "bos": "une cassure de structure (BOS)",
    "fvg": "une zone de déséquilibre (FVG)",
    "ob": "une zone d'absorption (OB)",
    "regime": "le régime de marché",
    "volatility": "la volatilité attendue",
    "news": "le contexte news",
    "momentum": "le momentum",
    "rsi_divergence": "une divergence RSI",
    "smart_money": "le Smart Money",
    "liquidity": "la liquidité",
    "sessions": "la session de trading",
    "multi_timeframe": "le multi-timeframe",
}

COMPONENT_LAYMAN_EN = {
    # ... en équivalent
}

def component_to_layman_term(name: str, lang: str) -> str:
    table = COMPONENT_LAYMAN_FR if lang == "fr" else COMPONENT_LAYMAN_EN
    return table.get(name.lower(), name)


def format_time(minutes: int, lang: str) -> str:
    if minutes < 60:
        return f"{minutes} min" if lang == "fr" else f"{minutes}min"
    h = minutes // 60
    m = minutes % 60
    if lang == "fr":
        return f"{h}h{m:02d}" if m else f"{h}h"
    return f"{h}h{m:02d}m" if m else f"{h}h"
```

### 2. Endpoint backend

```python
# backend/src/api/routes/chat.py (extension)
from src.intelligence.chatbot.suggested_questions import derive_suggested_questions

@router.get("/suggestions/{signal_id}")
async def get_suggestions(signal_id: str, lang: str = "fr", user=Depends(require_api_key)):
    signal = await SignalStore.get(signal_id)
    if not signal:
        raise HTTPException(404)
    return {"questions": derive_suggested_questions(signal, lang)}
```

### 3. Frontend — chips

```tsx
// frontend/components/SuggestedQuestionChip.tsx
interface Props {
  question: string;
  onClick: (q: string) => void;
  disabled?: boolean;
}

export default function SuggestedQuestionChip({ question, onClick, disabled }: Props) {
  return (
    <button
      className="chat-question text-left px-3.5 py-3 border border-border-light rounded-md text-text-secondary text-sm hover:bg-bg-hover hover:border-gold-soft hover:text-text-primary disabled:opacity-50 disabled:cursor-not-allowed"
      onClick={() => onClick(question)}
      disabled={disabled}
    >
      {question}
    </button>
  );
}
```

```tsx
// frontend/components/ChatbotPanel.tsx (extension)
'use client';

import { useState, useEffect } from 'react';
import { fetchSuggestions, askChatbot } from '@/lib/chat-api';
import SuggestedQuestionChip from './SuggestedQuestionChip';

export default function ChatbotPanel({ signalId, lang = 'fr' }: Props) {
  const [suggested, setSuggested] = useState<string[]>([]);
  const [messages, setMessages] = useState<Msg[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchSuggestions(signalId, lang).then(setSuggested);
  }, [signalId, lang]);

  return (
    <div className="chatbot-panel">
      <div className="chat-thread">
        {messages.length === 0 ? (
          <div className="suggestions-empty">
            <div className="chat-side-title">Suggestions contextuelles</div>
            <div className="chat-questions">
              {suggested.map((q, i) => (
                <SuggestedQuestionChip key={i} question={q} onClick={ask} disabled={loading} />
              ))}
            </div>
          </div>
        ) : (
          messages.map((m, i) => <ChatMessage key={i} msg={m} />)
        )}
      </div>
      <div className="chat-input">...</div>
    </div>
  );
}
```

## Acceptance criteria

- [ ] À l'ouverture du chatbot panel (sans message envoyé), 3 chips suggérées affichées
- [ ] Q1 est dynamique : contient le score de conviction du signal en cours
- [ ] Q2 est dynamique : mentionne le terme jargon du composant principal contributeur
- [ ] Q3 est contextuelle : event si <4h, sinon historique si conviction ≥70, sinon marge erreur
- [ ] Clic sur chip → question envoyée immédiatement au chatbot (DG-110 ask)
- [ ] Event `chatbot_question` émis avec `question_category="suggested"` pour différencier des questions libres
- [ ] Multilingue : FR/EN supportés (default FR)
- [ ] Chips masquées dès le premier message envoyé (UI propre)
- [ ] Sur erreur fetch suggestions, fallback 3 questions par défaut hardcodées

## Tests requis

```python
# backend/tests/test_suggested_questions.py
def test_q3_event_priority():
    """Si FOMC dans 2h, Q3 doit être contextualisation event."""
    signal = build_test_signal(event_label="FOMC", event_minutes=120)
    qs = derive_suggested_questions(signal, "fr")
    assert "FOMC" in qs[2]
    assert "2h" in qs[2]

def test_q3_historical_when_high_conviction():
    signal = build_test_signal(event_label=None, event_minutes=None, conviction=80)
    qs = derive_suggested_questions(signal, "fr")
    assert "historiquement" in qs[2].lower()

def test_q3_uncertainty_fallback():
    signal = build_test_signal(event_label=None, conviction=55)
    qs = derive_suggested_questions(signal, "fr")
    assert "marge" in qs[2].lower() or "erreur" in qs[2].lower()

def test_q1_dynamic_conviction():
    for c in [42, 72, 89]:
        signal = build_test_signal(conviction=c)
        qs = derive_suggested_questions(signal, "fr")
        assert str(c) in qs[0]
```

## Risques / pièges

- ❌ **Questions hardcodées non dynamiques** : si c'est toujours "Pourquoi 72 ?" même quand conviction = 45, c'est faux et ridicule.
- ❌ **Trop de questions** (6+) : noyé visuellement. 3 = sweet spot.
- ❌ **Questions en jargon dans le chip** : "C'est quoi BOS ?" est moins bon que "C'est quoi une cassure de structure (BOS) ?". On définit le jargon avant d'en parler.
- ❌ **Fetch suggestions bloquant** : si l'API tarde, le chatbot panel reste vide. Fallback hardcodé obligatoire.
- ❌ **Pas de différenciation analytique** : sans `question_category="suggested"` vs "free", on ne peut pas mesurer le taux d'engagement chip.
