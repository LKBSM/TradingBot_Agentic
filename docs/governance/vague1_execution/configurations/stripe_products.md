# Stripe — Configuration products + prices + webhooks

À configurer dans le dashboard Stripe (test puis live S6).

---

## 4 Products + 8 Price IDs

### Product 1 — M.I.A. Markets Starter

- **Stripe Product ID** : `prod_starter_v1` (à créer)
- **Name** : M.I.A. Markets Starter
- **Description** : 4 actifs · 30 lectures/jour · chatbot 100 questions/jour · alertes event imminent
- **Metadata** : `tier=STARTER`, `version=v1`

**Prices** :

| Price ID | Cycle | Amount | Currency | Trial period |
|---|---|---|---|---|
| `price_starter_monthly_v1` | mensuel | 2 900 (= 29 USD) | usd | 14 jours **sans CB** |
| `price_starter_yearly_v1` | annuel | 29 000 (= 290 USD) | usd | 14 jours **sans CB** |

### Product 2 — M.I.A. Markets Pro

- **Stripe Product ID** : `prod_pro_v1`
- **Name** : M.I.A. Markets Pro
- **Description** : 6 actifs · lectures illimitées · chatbot illimité · détail technique (waterfall, conformal viz, RAG) · exports CSV
- **Metadata** : `tier=PRO`, `version=v1`

**Prices** :

| Price ID | Cycle | Amount | Currency | Trial period |
|---|---|---|---|---|
| `price_pro_monthly_v1` | mensuel | 7 900 (= 79 USD) | usd | 14 jours **avec CB** |
| `price_pro_yearly_v1` | annuel | 79 000 (= 790 USD) | usd | 14 jours **avec CB** |

### Product 3 — M.I.A. Markets Institutional

- **Stripe Product ID** : `prod_institutional_v1`
- **Name** : M.I.A. Markets Institutional
- **Description** : API B2B JSON complet · webhooks signés HMAC · SLA 99.9 % · licence redistribution white-label · engagement 12 mois
- **Metadata** : `tier=INSTITUTIONAL`, `version=v1`, `min_commitment_months=12`

**Prices** :

| Price ID | Cycle | Amount | Currency | Trial |
|---|---|---|---|---|
| `price_institutional_monthly_v1` | mensuel (engagement 12 mois) | 199 000 (= 1 990 USD) | usd | aucun |
| `price_institutional_yearly_v1` | annuel | 1 990 000 (= 19 900 USD) | usd | aucun |

### Product 4 — N/A (FREE = pas de product Stripe)

FREE est géré côté backend uniquement (signup sans paiement). Pas de product Stripe créé.

---

## Webhooks à configurer

URL : `https://api.mia.markets/api/v1/webhooks/stripe`
Signature : `STRIPE_WEBHOOK_SECRET` (env var)

### Events à écouter

| Event | Handler côté backend | Action |
|---|---|---|
| `checkout.session.completed` | `on_checkout_completed` | Activer abonnement, envoyer welcome email |
| `customer.subscription.created` | `on_subscription_created` | Persister `subscription_id` dans users.db |
| `customer.subscription.updated` | `on_subscription_updated` | MAJ tier dans users.db (upgrade/downgrade) |
| `customer.subscription.deleted` | `on_subscription_deleted` | Downgrade FREE, envoyer churn email (DG-131) |
| `customer.subscription.trial_will_end` | `on_trial_will_end` | Envoyer D-1 trial reminder (DG-131) |
| `invoice.paid` | `on_invoice_paid` | Émettre event analytics `paid_conversion` (DG-161) |
| `invoice.payment_failed` | `on_payment_failed` | Notifier user, retry Stripe natif |
| `customer.subscription.pending_update_applied` | `on_pending_update` | Sync metadata user |

### Implémentation handler exemple

```python
# backend/src/api/routes/webhooks/stripe.py
import stripe
from fastapi import APIRouter, Request, HTTPException
from config import STRIPE_WEBHOOK_SECRET
from src.intelligence.analytics_client import emit_server_event
from src.api.user_store import UserStore
from src.delivery.email_sender import send_template_email

router = APIRouter(prefix="/api/v1/webhooks")

@router.post("/stripe")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")
    try:
        event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
    except Exception as e:
        raise HTTPException(400, f"Invalid webhook: {e}")

    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        await UserStore.activate_subscription(
            user_email=session["customer_email"],
            stripe_customer_id=session["customer"],
            stripe_subscription_id=session["subscription"],
            tier=session["metadata"]["tier"],
        )
        await send_template_email("welcome", session["customer_email"], context={...})

    elif event["type"] == "invoice.paid":
        invoice = event["data"]["object"]
        emit_server_event("paid_conversion", {
            "tier": invoice["lines"]["data"][0]["price"]["metadata"]["tier"],
            "price_id": invoice["lines"]["data"][0]["price"]["id"],
            "amount_cents": invoice["amount_paid"],
            "currency": invoice["currency"],
        })

    elif event["type"] == "customer.subscription.deleted":
        sub = event["data"]["object"]
        user = await UserStore.get_by_stripe_subscription_id(sub["id"])
        if user:
            await UserStore.downgrade_to_free(user.id)
            await send_template_email("churn", user.email, context={...})

    return {"ok": True}
```

---

## Customer Portal — Configuration

Activer dans le dashboard Stripe : **Settings → Billing → Customer Portal**.

### Allowed features
- ✓ Update billing address
- ✓ Update payment method
- ✓ View invoices and payment history
- ✓ Cancel subscription (sans rétention)
- ✓ Update subscription quantity (futur, V2)

### Cancellation policy
- **Refund window** : 30 days from first payment (DG-079)
- **Automation** : configurer pour refund automatique si demandé < 30j post first paid invoice
- **Past 30 days** : pas de refund prorata automatique (politique standard SaaS)

### Branding
- Logo : `https://mia.markets/logo.png` (à créer)
- Couleurs : palette finance premium (gold #c9a961, bg #0b0e13)

---

## Stripe Tax (DG-044)

À activer dans **Tax settings** dès activation live :

1. **Pays activés** : FR, BE, CH, LU
2. **Auto-collect** : ON (Stripe calcule automatiquement TVA selon pays user)
3. **Reverse charge B2B** : ON (B2B EU → reverse charge automatique)
4. **EU OSS** : enregistrement auprès de l'administration fiscale FR pour utiliser le guichet unique (à initier S5 par toi avec ton expert-compta)

**Important** : tant que tu es en franchise en base TVA (CA < 36 800 €), tu peux **désactiver** Stripe Tax sur tes prix → tes prix sont HT et tu ne factures pas de TVA. Tu activeras au passage du seuil.

---

## Tests obligatoires avant live (S6)

### Mode test

1. Créer compte test, faire checkout sur chaque tier (Starter monthly, Pro monthly, Pro yearly, Institutional yearly)
2. Carte test : `4242 4242 4242 4242`, expiry future, CVC `123`, ZIP `12345`
3. Vérifier que chaque webhook arrive et le handler s'exécute (logs)
4. Tester le refund Stripe Customer Portal → backend reçoit `charge.refunded`
5. Tester l'upgrade Starter → Pro depuis Customer Portal → backend MAJ tier
6. Tester l'annulation depuis Customer Portal → backend downgrade FREE
7. Tester carte refusée : `4000 0000 0000 9995` → `invoice.payment_failed` capturé

### Soumission Stripe live review

Avant l'activation live (S6), soumettre le site pour review fintech via dashboard Stripe :

- URL du site complet
- Description activité : *"Educational algorithmic market analysis tool for gold and FX, restricted to FR/BE/CH/LU residents. Not investment advice. Compliant with EU 2024/2811 financial influencer regulation."*
- Documents : page méthodologie + CGU V0 + Privacy V0
- Délai review : 2-5 jours

---

## Webhook idempotency

```python
# backend/src/api/routes/webhooks/stripe.py
from src.api.idempotency_store import IdempotencyStore

@router.post("/stripe")
async def stripe_webhook(request: Request):
    # ...
    event_id = event["id"]
    if await IdempotencyStore.exists(f"stripe_event:{event_id}"):
        return {"ok": True, "duplicate": True}
    await IdempotencyStore.set(f"stripe_event:{event_id}", ttl_hours=24)
    # ... handle event
```

Stripe peut renvoyer le même event jusqu'à 3 fois en cas de doute. Idempotency obligatoire.
