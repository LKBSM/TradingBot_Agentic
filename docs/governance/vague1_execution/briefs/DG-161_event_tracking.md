# DG-161 — Event tracking core (6 events)

**Effort** : ~10-14h · **Sprint** : S3 · **Owner** : code

---

## Objectif

Instrumenter les **6 events core** qui rendent observables les déclencheurs DEFER (MAU > 200, churn > 20 %, engagement chat) et permettent de construire le funnel d'acquisition.

## Contexte

Sans events, ni funnel mesurable, ni churn quantifiable, ni engagement chat évalué. C'est la fondation de l'analytique produit.

## Périmètre

**IN — 6 events core** :
1. `signal_view` — utilisateur consulte une lecture (front + Telegram + email)
2. `chatbot_question` — utilisateur pose une question au chatbot
3. `section_expanded` — utilisateur déplie une section collapsible (Pourquoi conviction, Régime, Structure, Détail technique)
4. `upgrade_clicked` — utilisateur clique CTA upgrade tier
5. `signup` — création de compte (FREE)
6. `paid_conversion` — premier paiement Stripe confirmé

**OUT** :
- Tracking comportemental fin (scroll, heatmap, replay session)
- Tracking marketing externe (UTM analysis avancé)
- A/B testing automatisé

## Dépendances

- DG-160 Plausible self-hosted opérationnel
- DG-101 Renderer + sections collapsibles (pour `section_expanded`)
- DG-110 Chatbot wiring (pour `chatbot_question`)

## Fichiers à toucher

- `frontend/lib/analytics.ts` (à créer) — wrapper Plausible custom events
- `frontend/app/lecture/[id]/page.tsx` — emit `signal_view` à mount
- `frontend/components/Chatbot.tsx` — emit `chatbot_question` à submit
- `frontend/components/CollapsibleSection.tsx` — emit `section_expanded` au déplier
- `frontend/app/pricing/page.tsx` — emit `upgrade_clicked` sur CTA tier
- `frontend/app/signup/page.tsx` — emit `signup` post succès
- `backend/src/api/routes/webhooks/stripe.py` — emit `paid_conversion` sur événement Stripe `invoice.paid`

## Implémentation

### 1. Wrapper analytics front

```typescript
// frontend/lib/analytics.ts
type EventName =
  | 'signal_view'
  | 'chatbot_question'
  | 'section_expanded'
  | 'upgrade_clicked'
  | 'signup'
  | 'paid_conversion';

type EventProps = {
  signal_view: { signal_id: string; instrument: 'XAU' | 'EUR'; tier_user: string };
  chatbot_question: { question_category?: string; tier_user: string; session_id: string };
  section_expanded: { section_id: string; tier_user: string };
  upgrade_clicked: { from_tier: string; to_tier: string; cta_location: string };
  signup: { source?: string };
  paid_conversion: { tier: string; price_id: string; amount_cents: number; currency: string };
};

export function trackEvent<E extends EventName>(name: E, props: EventProps[E]): void {
  if (typeof window === 'undefined') return;
  // Plausible custom events API
  (window as any).plausible?.(name, { props });
}
```

### 2. Intégration aux composants

```tsx
// Exemple — frontend/components/Chatbot.tsx
import { trackEvent } from '@/lib/analytics';

async function handleSubmit(question: string) {
  trackEvent('chatbot_question', {
    question_category: categorizeQuestion(question),  // simple regex categorization
    tier_user: currentUser.tier,
    session_id: chatSessionId,
  });
  // ... rest of submit logic
}
```

### 3. Webhook backend (paid_conversion)

```python
# backend/src/api/routes/webhooks/stripe.py
from src.intelligence.analytics_client import emit_server_event

@router.post("/webhooks/stripe")
async def stripe_webhook(request: Request):
    event = stripe_verify_webhook(request)
    if event.type == "invoice.paid":
        invoice = event.data.object
        emit_server_event("paid_conversion", {
            "tier": invoice.subscription.plan.metadata.tier,
            "price_id": invoice.subscription.plan.id,
            "amount_cents": invoice.amount_paid,
            "currency": invoice.currency,
        })
    return {"ok": True}
```

### 4. Client server-side (Plausible API)

```python
# backend/src/intelligence/analytics_client.py
import httpx
from config import PLAUSIBLE_URL, PLAUSIBLE_DOMAIN

async def emit_server_event(name: str, props: dict, user_ip: str = "127.0.0.1"):
    """Server-side event via Plausible Events API."""
    async with httpx.AsyncClient() as client:
        await client.post(
            f"{PLAUSIBLE_URL}/api/event",
            json={
                "name": name,
                "url": f"https://{PLAUSIBLE_DOMAIN}/server",
                "domain": PLAUSIBLE_DOMAIN,
                "props": props,
            },
            headers={
                "X-Forwarded-For": user_ip,
                "User-Agent": "Smart-Sentinel-Server/1.0",
            },
        )
```

## Acceptance criteria

- [ ] Visite landing → `pageview` capturé (déjà via DG-160)
- [ ] Inscription FREE → `signup` capturé avec `source` (utm_source ou referrer)
- [ ] Consultation d'une lecture → `signal_view` capturé avec `signal_id`, `instrument`, `tier_user`
- [ ] Question au chatbot → `chatbot_question` capturé avec catégorie
- [ ] Déploiement section collapsible → `section_expanded` capturé avec `section_id`
- [ ] Clic CTA upgrade pricing → `upgrade_clicked` capturé avec from/to tier
- [ ] Paiement Stripe confirmé → `paid_conversion` capturé server-side
- [ ] Dashboard Plausible : `Custom Events` tab affiche les 6 events après 1h de trafic test
- [ ] Aucun PII dans les props (juste catégorie/tier/IDs anonymes)

## Tests requis

```python
# tests/test_analytics_events.py
async def test_paid_conversion_event_emitted(stripe_webhook_payload):
    # Mock Plausible API
    with mock_plausible() as mock:
        await stripe_webhook(stripe_webhook_payload)
        mock.assert_called_with("paid_conversion", props=...)
```

```typescript
// frontend/__tests__/analytics.spec.ts
test('trackEvent calls plausible with correct args', () => {
  window.plausible = jest.fn();
  trackEvent('signal_view', { signal_id: 'abc', instrument: 'XAU', tier_user: 'FREE' });
  expect(window.plausible).toHaveBeenCalledWith('signal_view', { props: expect.objectContaining({ signal_id: 'abc' }) });
});
```

## Risques / pièges

- ❌ **Mettre des PII dans les props** (email, IP exacte, contenu de message chatbot) : viole RGPD + dilue les rapports. Solution : categorize_question() renvoie juste "conviction" / "structure" / "compliance" / "general", PAS le contenu.
- ❌ **Bloquer le UX en cas d'échec analytics** : `trackEvent` doit être fire-and-forget, jamais bloquer un submit form
- ❌ **Tracker côté front les événements monétaires** (paid_conversion) : faillible (ad-blockers). DOIT être server-side via webhook Stripe.
- ❌ **Oublier de catégoriser les questions chatbot** : sans catégorie, on ne peut pas mesurer "% de refus pédagogique"
