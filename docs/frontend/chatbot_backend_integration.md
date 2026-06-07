# Chatbot — intégration webapp ↔ backend FastAPI (Chantier 5.A)

Statut : livré au Sous-Chantier 5.A (branche `feat/chantier-5a-chatbot-webapp-backend-wiring`).

Ce document décrit comment le chatbot de la webapp Next.js est câblé sur le
backend FastAPI du Chantier 4, après le décommissionnement du chatbot Next.js
standalone qui appelait Anthropic en direct.

---

## 1. Architecture

```
┌──────────────────────────┐     POST /api/chatbot/message      ┌────────────────────────────┐
│  Webapp Next.js (3000)    │  ───────────────────────────────▶ │  FastAPI backend (8000)     │
│                           │   {user_message,                   │                             │
│  ChatInput / ChatPanel    │    conversation_history}           │  routes/chatbot.py          │
│        │                  │                                    │        │                    │
│  ChatProvider.askFreeForm │   rewrite /api/* (next.config.js)  │  Chatbot.chat()             │
│        │                  │   → ${NEXT_PUBLIC_API_BASE}/api/*  │   1. Couche 1 — adversarial │
│  lib/chat/api-client.ts   │                                    │      input filter           │
│  askSentinel()            │  ◀───────────────────────────────  │   2. Couche 2 — Haiku +     │
│                           │   {content, blocked_reason,        │      tool use               │
│                           │    tool_calls_made}  (JSON)        │   3. Couche 3 — output      │
└──────────────────────────┘                                    │      forbidden tokens       │
                                                                 └────────────────────────────┘
                                                                   Anthropic SDK + clé : ICI UNIQUEMENT
```

Points clés :

- **Une seule voie** pour la saisie libre : `POST /api/chatbot/message`. Le
  navigateur ne parle jamais à `api.anthropic.com` (CSP `connect-src 'self'`).
- Le rewrite `/api/*` de `next.config.js` rend l'appel **same-origin** côté
  navigateur (`/api/chatbot/message`) puis le proxifie vers FastAPI.
- Les **3 couches de défense niveau 1.5 strict** s'exécutent toutes côté
  backend. Aucun message utilisateur ne peut les contourner depuis la webapp.
- L'**Anthropic SDK et la clé API ne vivent plus que côté backend**. Le SDK a
  été désinstallé du frontend (`@anthropic-ai/sdk` retiré de `package.json`).

### Contrats de données

Requête (`ChatbotMessageRequest`, `src/api/routes/chatbot.py`) :

```json
{
  "user_message": "string (1..2000)",
  "conversation_history": [{ "role": "user|assistant", "content": "string (1..2000)" }]
}
```

Réponse (`ChatbotMessageResponse`) :

```json
{
  "content": "string",                // texte affiché (réponse, refus, ou template)
  "blocked_reason": "string | null",  // ex. trade_request, llm_error, output_contaminated_*
  "tool_calls_made": [{ "name": "...", "input": { } }]
}
```

Codes HTTP gérés par `api-client.ts` :

| Code | Signification | Comportement frontend |
|------|---------------|-----------------------|
| 200  | réponse OK | affichée ; badge discret si `blocked_reason !== null` |
| 422  | validation Pydantic (format/longueur) | `ChatApiError(422)`, message clair |
| 503  | chatbot non bootstrappé | `ChatApiUnavailableError` → bascule scripted (`apiAvailable=false`) |
| 500  | erreur interne | `ChatApiError(500)`, message générique (jamais de leak serveur) |
| réseau | pas de réponse HTTP | `ChatApiError(0)` |

### Convention T1 — contexte signal

Le panneau est ouvert *pour* une lecture précise (`openFor(signal)`), mais
l'endpoint backend est agnostique au signal. `api-client.ts` préfixe donc le
`user_message` d'une ligne de contexte **stable et reconnaissable** quand un
signal est actif :

```
[Lecture en cours : XAUUSD H1]
{question de l'utilisateur}
```

Ce format fixe (crochets + structure stable) permet à la couche Haiku d'appeler
`get_market_reading(instrument, timeframe)` sur la bonne combinaison, et reste
détectable/supprimable si l'architecture évolue. Pas de préfixe si aucun signal
n'est actif.

---

## 2. Variables d'environnement

| Variable | Où | Rôle | Défaut |
|----------|-----|------|--------|
| `NEXT_PUBLIC_API_BASE` | webapp | base du backend pour le rewrite `/api/*` | `http://localhost:8000` |

Voir `webapp/.env.example`. En production (Vercel + FastAPI sur Fly.io) :
`NEXT_PUBLIC_API_BASE=https://api.mia.markets`.

> Il n'y a **plus** d'`ANTHROPIC_API_KEY` côté frontend. La clé Anthropic ne
> doit exister que sur le backend FastAPI. Ne pas la réintroduire dans la webapp.

---

## 3. Test local

Deux process :

```bash
# 1) Backend FastAPI (port 8000) — doit avoir le chatbot bootstrappé
#    (ANTHROPIC_API_KEY côté serveur + CHATBOT_ENABLED)
python -m src.intelligence.main

# 2) Webapp Next.js (port 3000)
cd webapp
# .env.local : NEXT_PUBLIC_API_BASE=http://localhost:8000
npm run dev
```

Ouvrir http://localhost:3000, ouvrir le chat sur une lecture, poser une
question libre :

- question descriptive (« Décris la structure XAUUSD H1 ») → réponse non bloquée ;
- question d'action (« Dois-je acheter ? ») → template de refus + badge discret
  « Question recadrée » (couche 1 ou 3, `blocked_reason` renseigné) ;
- backend éteint → message fallback « mode chatbot en direct indisponible » +
  bascule sur les questions suggérées scriptées.

Tests automatisés : `cd webapp && npm test` (Vitest — `lib/chat` + `components/chat`).

---

## 4. Décommissionnement de l'ancien chatbot Next.js

Supprimés au Chantier 5.A :

- `webapp/app/api/chat/route.ts` — route SSE qui appelait l'Anthropic SDK en
  direct, **sans aucune des 3 couches de défense** ;
- `webapp/lib/chat/system-prompt.ts` — le system prompt vit désormais côté
  backend (`src/intelligence/chatbot/chatbot.py`) ;
- `webapp/lib/chat/signal-summary.ts` — le `SignalSummaryProvider` est côté
  backend ;
- dépendance `@anthropic-ai/sdk` (désinstallée du frontend).

Conservés : `webapp/lib/chat/api-client.ts` (réécrit), `webapp/lib/chatbot.ts`
(helper scripted), `webapp/components/chat/*` (UI), `webapp/types/chatbot.ts`.

Le streaming SSE token-by-token a été remplacé par une réponse JSON synchrone
affichée d'un coup avec un loader (« Sentinel réfléchit… »).
