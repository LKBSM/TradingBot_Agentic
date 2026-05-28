# DG-110 — Wire chatbot 8 composantes

> ⚠️ **PARTIELLEMENT OBSOLÈTE POST PIVOT 2026-05-27** — Les system prompts de référence dans ce brief mentionnent peut-être des chiffres de performance (PF 1.30, 329 setups). Le prompt à jour est dans **`docs/governance/vague1_execution/copies/chatbot_system_prompt.md`** + Q&A scriptées révisées dans **`copies/chatbot_scripted_responses.md`** (Q4 réécrite, OOS pending posture). Voir `docs/governance/decisions/2026-05-27_pivot_positioning_audit.md`.

**Effort** : ~20-30h · **Sprint** : S4 · **Owner** : code

---

## Objectif

Connecter le chatbot Claude au **contexte algorithmique complet** (les 8 composantes, l'intervalle conformel, les stats J.*, les events) pour qu'il puisse **décomposer pédagogiquement** chaque lecture à la demande.

## Contexte (moat #1)

C'est le pilier de défensabilité. Sans ce wiring, le chatbot est un wrapper LLM générique — n'importe qui peut copier ça avec ChatGPT. Avec ce wiring, le chatbot devient **un quant junior personnel** qui répond à partir du contexte exact de la lecture en cours.

## Périmètre

**IN** :
- Endpoint `/api/v1/chat/ask` qui prend `(signal_id, question, session_id)` et retourne réponse
- Context injection : récupérer `InsightSignalV2` depuis SignalStore + injecter dans prompt LLM
- Capacité à expliquer chacune des 8 composantes
- Capacité à expliquer l'intervalle conformel et la couverture empirique
- Capacité à comparer aux setups historiques
- Capacité à contextualiser un event imminent
- Memory de session (5 derniers échanges par session)
- Cache sémantique sur questions fréquentes
- Cost monitoring intégré (DG-052)

**OUT** :
- Refus pédagogique scripté (DG-112, brief séparé)
- Questions suggérées (DG-114, brief séparé)
- Voice mode (P2)
- Notifications proactives (P2)

## Dépendances

- DG-025 scoring v2 stable (sinon les contributions des 8 composantes ne sont pas calibrées)
- `InsightSignalV2` v2.1.0 contrat stable
- Anthropic API key configurée
- SignalStore opérationnel

## Fichiers à toucher

```
backend/
├── src/intelligence/
│   ├── llm_narrative_engine.py       (existant — extension)
│   ├── chatbot/
│   │   ├── __init__.py
│   │   ├── chatbot_engine.py         (à créer)
│   │   ├── context_builder.py        (à créer — injection InsightSignalV2)
│   │   ├── prompt_template.py        (à créer — system prompt + few-shots)
│   │   ├── session_memory.py         (à créer — last 5 turns par session)
│   │   └── cost_tracker.py           (à créer — Prometheus gauge)
│   └── semantic_cache.py             (extension — cache questions fréquentes)
├── src/api/
│   └── routes/
│       └── chat.py                   (à créer — endpoint /chat/ask)
└── tests/
    └── test_chatbot_wiring.py        (à créer)

frontend/
├── components/
│   ├── ChatbotPanel.tsx              (à créer — UI chat)
│   ├── ChatbotSidebar.tsx            (à créer — wrapper sidebar)
│   ├── ChatbotFAB.tsx                (existe — DG-103)
│   └── ChatMessage.tsx               (à créer)
└── lib/
    └── chat-api.ts                   (à créer — wrapper fetch /chat/ask)
```

## Implémentation

### 1. System prompt (extrait essentiel)

```python
# backend/src/intelligence/chatbot/prompt_template.py

SYSTEM_PROMPT = """Tu es Sentinel, le quant junior de M.I.A. Markets.

Ton rôle :
- Tu expliques en langage humain comment notre algorithme a lu un marché donné.
- Tu décomposes la conviction calibrée (0-100) en 8 composantes pondérées.
- Tu traduis le jargon technique (BOS, FVG, ACI, BOCPD) en explications accessibles.
- Tu compares aux setups historiques quand pertinent.
- Tu refuses pédagogiquement de donner un ordre d'achat / vente (cf. règles ci-dessous).

Posture :
- Ton honnête, factuel, sans hype.
- Tu admets l'incertitude (intervalle conformel = "marge d'erreur honnête").
- Tu n'utilises JAMAIS les mots "signal", "achetez", "vendez", "garanti", "profit X%".
- Tu utilises "lecture", "setup haussier/baissier", "analyse", "calibré", "vérifié sur historique".

RÈGLES STRICTES :
1. Si la question contient une demande prescriptive ("Dois-je acheter ?", "Faut-il vendre ?",
   "Quel stop ?", "Quel objectif ?"), tu refuses pédagogiquement et redirige vers la décision
   autonome du client. Ce refus est constitutif de la nature pédagogique du Service
   (compliance UE 2024/2811).

2. Tu réponds uniquement à partir du contexte algorithmique injecté ci-dessous.
   Tu n'inventes pas de chiffres. Si une donnée n'est pas dans le contexte, dis-le.

3. Format réponse :
   - Première phrase : réponse directe à la question.
   - Corps : explication structurée (listes, gras pour les chiffres clés).
   - Dernière phrase : invitation à creuser une autre dimension si pertinent.
   - Pas de salutation systématique ("Bonjour"), c'est une conversation continue.

4. Longueur cible : 80-180 mots. Tu peux dépasser jusqu'à 300 mots si la question le justifie.

5. Langue : adapte-toi à la langue de la question (FR par défaut).

---

CONTEXTE ALGORITHMIQUE INJECTÉ (lecture en cours) :
{context_json}

---

HISTORIQUE DE LA SESSION (5 derniers échanges max) :
{session_history}

---

QUESTION DE L'UTILISATEUR :
{user_question}
"""
```

### 2. Context builder

```python
# backend/src/intelligence/chatbot/context_builder.py
from src.api.insight_signal_v2 import InsightSignalV2

def build_context_payload(signal: InsightSignalV2) -> dict:
    """Extrait du InsightSignalV2 le minimum nécessaire pour le LLM,
    en filtrant les champs internes/debug."""
    return {
        "instrument": signal.instrument,
        "timeframe": signal.timeframe,
        "direction": signal.direction,
        "conviction": signal.conviction_0_100,
        "conviction_label": signal.conviction_label,
        "uncertainty": {
            "lower": signal.uncertainty.conformal_lower,
            "upper": signal.uncertainty.conformal_upper,
            "empirical_coverage": signal.uncertainty.empirical_coverage,
        },
        "structure": {
            "bos_level": signal.structure_readout.bos_level,
            "fvg_zone": signal.structure_readout.fvg_zone,
            "ob_zone": signal.structure_readout.ob_zone,
            "retest_state": signal.structure_readout.retest_state,
            "invalidation": signal.structure_readout.structural_invalidation,
            "choch_present": signal.structure_readout.choch_present,
        },
        "regime": {
            "label": signal.regime_readout.hmm_label,
            "confidence": signal.regime_readout.hmm_posterior,
            "changepoint_risk": signal.regime_readout.bocpd_changepoint_prob,
            "gate": signal.regime_readout.regime_gate_decision,
        },
        "volatility": {
            "regime": signal.volatility_readout.regime,
            "forecast_vs_naive_pct": signal.volatility_readout.forecast_vs_naive_pct,
        },
        "event": {
            "blackout_active": signal.event_readout.news_blackout_active,
            "next_label": signal.event_readout.next_event_label,
            "next_in_minutes": signal.event_readout.next_event_in_minutes,
            "session": signal.event_readout.session,
        },
        "breakdown": [
            {"name": c.name, "contribution": c.contribution, "weight_max": c.weight_max, "reasoning": c.reasoning}
            for c in signal.breakdown_components
        ],
        "history": {
            "n_similar": signal.historical_stats.similar_setups_n,
            "hit_rate": signal.historical_stats.hit_rate_observed,
            "profit_factor": signal.historical_stats.profit_factor,
            "pf_ci95": signal.historical_stats.profit_factor_ci95,
        },
        "narrative_short": signal.narrative_short,
    }
```

### 3. Endpoint

```python
# backend/src/api/routes/chat.py
from fastapi import APIRouter, Depends, HTTPException
from src.api.auth import require_api_key
from src.intelligence.chatbot.chatbot_engine import ChatbotEngine
from src.api.signal_store import SignalStore
from src.intelligence.score_calibration import contains_forbidden_token

router = APIRouter(prefix="/api/v1/chat")

class AskRequest(BaseModel):
    signal_id: str  # UUID 12 chars
    question: str   # max 500 chars
    session_id: str | None = None

class AskResponse(BaseModel):
    answer: str
    tokens_used: int
    is_refusal: bool   # True si refus pédagogique déclenché (DG-112)
    cost_estimate_usd: float

@router.post("/ask", response_model=AskResponse)
async def chat_ask(req: AskRequest, user=Depends(require_api_key)):
    # 1. récupérer signal
    signal = await SignalStore.get(req.signal_id)
    if not signal:
        raise HTTPException(404, "Signal not found")

    # 2. quota / rate limit / tier check
    if not user.can_ask_chatbot():
        raise HTTPException(429, "Daily chatbot quota reached. Upgrade tier.")

    # 3. delegate to ChatbotEngine
    engine = ChatbotEngine(tier=user.tier)
    result = await engine.ask(
        signal=signal,
        question=req.question,
        session_id=req.session_id or user.id,
    )

    # 4. validation post-génération (forbidden tokens)
    if contains_forbidden_token(result.answer):
        # logged for audit, replaced by safe fallback
        result.answer = "Je ne peux pas répondre à cette question dans le cadre éducatif. Reformulez sans demande d'instruction directe."
        result.is_refusal = True

    # 5. emit metrics
    user.increment_chatbot_usage()

    return AskResponse(
        answer=result.answer,
        tokens_used=result.tokens,
        is_refusal=result.is_refusal,
        cost_estimate_usd=result.cost_usd,
    )
```

### 4. Tier-routed model

```python
# backend/src/intelligence/chatbot/chatbot_engine.py
TIER_MODEL = {
    "FREE":          "claude-haiku-4-5-20251001",       # cheap, capped 5 Q/day
    "STARTER":       "claude-haiku-4-5-20251001",       # cheap, capped 100 Q/day
    "PRO":           "claude-sonnet-4-6",               # mid, unlimited
    "STRATEGIST":    "claude-sonnet-4-6",
    "INSTITUTIONAL": "claude-opus-4-7",                 # premium
}

class ChatbotEngine:
    def __init__(self, tier: str):
        self.model = TIER_MODEL.get(tier, "claude-haiku-4-5-20251001")
        self.client = AsyncAnthropic()

    async def ask(self, signal, question, session_id):
        # context
        ctx = build_context_payload(signal)

        # cache sémantique : si question simi-similaire déjà répondue, return cached
        cached = semantic_cache.lookup(question, signal.id)
        if cached:
            return cached

        # session memory
        history = SessionMemory.get(session_id, last_n=5)

        # build prompt
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.format(
                context_json=json.dumps(ctx, ensure_ascii=False, indent=2),
                session_history=format_history(history),
                user_question=question,
            )},
            {"role": "user", "content": question},
        ]

        # call Claude
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=600,
            messages=messages,
        )

        answer = response.content[0].text
        tokens = response.usage.input_tokens + response.usage.output_tokens
        cost = estimate_cost(self.model, response.usage)

        # cache + memory
        semantic_cache.store(question, signal.id, answer)
        SessionMemory.append(session_id, question, answer)

        # Prometheus
        chatbot_cost_total.inc(cost)
        chatbot_tokens_total.inc(tokens)

        return ChatResult(answer=answer, tokens=tokens, cost_usd=cost, is_refusal=False)
```

### 5. Frontend chat panel

```tsx
// frontend/components/ChatbotPanel.tsx
'use client';

import { useState, useRef, useEffect } from 'react';
import { askChatbot } from '@/lib/chat-api';
import { trackEvent } from '@/lib/analytics';
import ChatMessage from './ChatMessage';

export default function ChatbotPanel({ signalId, suggestedQuestions, onClose }: Props) {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const threadRef = useRef<HTMLDivElement>(null);

  async function handleAsk(q: string) {
    if (!q.trim() || loading) return;
    setLoading(true);
    setMessages(m => [...m, { role: 'user', text: q }]);

    trackEvent('chatbot_question', {
      question_category: categorize(q),
      tier_user: currentUser.tier,
      session_id: sessionId,
    });

    try {
      const res = await askChatbot({ signal_id: signalId, question: q, session_id: sessionId });
      setMessages(m => [...m, { role: 'bot', text: res.answer, is_refusal: res.is_refusal }]);
    } catch (e) {
      setMessages(m => [...m, { role: 'bot', text: 'Désolé, indisponible. Réessayez dans un instant.', error: true }]);
    } finally {
      setLoading(false);
      setInput('');
      setTimeout(() => threadRef.current?.scrollTo({ top: 999999, behavior: 'smooth' }), 50);
    }
  }

  // ... render messages + suggested questions + input
}
```

## Acceptance criteria

- [ ] Endpoint `POST /api/v1/chat/ask` opérationnel avec auth API key
- [ ] Quota chatbot enforced par tier (5/100/illimité/illimité)
- [ ] Cache sémantique opérationnel (hit rate observable via Prometheus)
- [ ] Session memory : 5 derniers échanges visibles dans le prompt
- [ ] Question "Pourquoi 72 ?" → réponse décompose les 8 composantes avec contributions
- [ ] Question "C'est quoi un retest armé ?" → définition + analogie pédagogique
- [ ] Question "Le FOMC dans 2h47, ça change quoi ?" → réponse contextuelle vol + risque whipsaw
- [ ] Question "Stats historiques ?" → résume profit_factor + IC + hit rate + N
- [ ] Question "Marge d'erreur sur 72 ?" → traduit intervalle conformel en langage humain
- [ ] Question "Dois-je acheter ?" → refus pédagogique scripté (DG-112)
- [ ] Aucun vocabulaire interdit dans les réponses (contains_forbidden_token)
- [ ] Cost monitoring : événement Prometheus `chatbot_cost_total` incrémenté
- [ ] Tier-routed model : FREE = Haiku, PRO = Sonnet, INSTITUTIONAL = Opus
- [ ] Tests E2E sur les 6 questions types passent

## Tests requis

```python
# tests/test_chatbot_wiring.py

@pytest.mark.asyncio
async def test_conviction_decomposition():
    """User asks 'why 72?' → response decomposes 8 components"""
    signal = build_test_signal(conviction=72, ...)
    engine = ChatbotEngine(tier="STARTER")
    result = await engine.ask(signal, "Pourquoi 72 ?", "session-test")

    assert "Smart Money" in result.answer
    assert "+24.5" in result.answer or "24" in result.answer
    assert "8 composantes" in result.answer.lower() or "8 facteurs" in result.answer.lower()
    assert not contains_forbidden_token(result.answer)


@pytest.mark.asyncio
async def test_forbidden_tokens_blocked():
    """Si réponse contient 'achetez', remplacée par safe fallback"""
    signal = build_test_signal()
    engine = ChatbotEngine(tier="STARTER")
    with patch_claude_to_return("Vous devriez acheter à 2390"):
        result = await engine.ask(signal, "n'importe quoi", "s")
    assert "acheter" not in result.answer.lower()
    assert result.is_refusal
```

## Risques / pièges

- ❌ **Injecter le contrat InsightSignalV2 brut entier** dans le prompt : trop verbeux, coûte des tokens. Utiliser `build_context_payload()` qui filtre.
- ❌ **Oublier le post-processing forbidden_token** : Claude peut occasionnellement glisser un "acheter". Validation obligatoire avant envoi au client.
- ❌ **Ne pas tier-router** : INSTITUTIONAL à $1990/mo mérite Opus, FREE doit rester Haiku pour limiter coût.
- ❌ **Cache sémantique trop strict** : hash exact = jamais de hit. Hash bucketed (conviction arrondie au multiple de 5, instrument, direction).
- ❌ **Streaming response côté front oublié** : pour réponses > 200 tokens, le streaming améliore UX. Anthropic SDK le supporte (`messages.stream`).
- ❌ **Pas de session memory** : le user pose 3 questions de suite et le chatbot répond comme si chaque était la première. Coupe le flow conversationnel. SessionMemory 5-turn obligatoire.
- ❌ **Hallucination de stats** : Claude peut inventer "1.45 PF" si on ne précise pas que les stats sont injectées. Préciser dans le prompt "tu n'inventes pas de chiffres".
