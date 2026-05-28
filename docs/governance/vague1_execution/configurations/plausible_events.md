# Plausible events — Schéma des 6 events core

Référence DG-161. Aucun PII dans les props (privacy-by-design).

---

## Schema YAML

```yaml
events:

  - name: signal_view
    description: Utilisateur consulte une lecture (page /lecture/[id])
    trigger: useEffect à mount LectureView
    surface: webapp | telegram | email
    properties:
      signal_id: string             # UUID 12 chars, anonyme
      instrument: enum [XAU, EUR]
      direction: enum [BULLISH_SETUP, BEARISH_SETUP, NEUTRAL]
      tier_user: enum [FREE, STARTER, PRO, INSTITUTIONAL]
      surface: enum [webapp, telegram, email]

  - name: chatbot_question
    description: Utilisateur pose une question au chatbot
    trigger: handleSubmit dans ChatbotPanel
    properties:
      question_category: enum [conviction, structure, regime, vol, event, history, methodology, prescriptive_refusal, general]
      question_source: enum [free, suggested]   # suggested chip vs typing libre
      tier_user: enum [FREE, STARTER, PRO, INSTITUTIONAL]
      session_id_hash: string       # hash anonyme session

  - name: section_expanded
    description: Utilisateur déplie une section collapsible
    trigger: onClick CollapsibleSection si !locked
    properties:
      section_id: enum [conviction, regime_vol, structure, expert]
      tier_user: enum [FREE, STARTER, PRO, INSTITUTIONAL]
      was_locked: boolean           # true si user a essayé une section verrouillée

  - name: upgrade_clicked
    description: Utilisateur clique sur CTA upgrade tier
    trigger: onClick TierCard cta
    properties:
      from_tier: enum [FREE, STARTER, PRO]
      to_tier: enum [STARTER, PRO, INSTITUTIONAL]
      cta_location: enum [pricing_page, locked_overlay, upgrade_banner, email_link]

  - name: signup
    description: Création de compte FREE confirmé (email validé)
    trigger: post-validation email signup
    properties:
      source: string                # utm_source ou referrer
      campaign: string              # utm_campaign si présent

  - name: paid_conversion
    description: Premier paiement Stripe confirmé
    trigger: webhook Stripe invoice.paid (server-side seulement)
    properties:
      tier: enum [STARTER, PRO, INSTITUTIONAL]
      price_id: string              # Stripe price_id
      amount_cents: int             # ex 2900 pour 29 USD
      currency: enum [usd]
      billing_cycle: enum [monthly, yearly]
      trial_converted: boolean      # true si conversion d'un trial vs paiement direct
```

---

## Implémentation côté front

```typescript
// frontend/lib/analytics.ts (extension du brief DG-161)
type EventDef = {
  signal_view: { signal_id: string; instrument: 'XAU' | 'EUR'; direction: string; tier_user: string; surface: string };
  chatbot_question: { question_category: string; question_source: string; tier_user: string; session_id_hash: string };
  section_expanded: { section_id: string; tier_user: string; was_locked: boolean };
  upgrade_clicked: { from_tier: string; to_tier: string; cta_location: string };
  signup: { source?: string; campaign?: string };
  paid_conversion: { tier: string; price_id: string; amount_cents: number; currency: string; billing_cycle: string; trial_converted: boolean };
};

export function trackEvent<E extends keyof EventDef>(name: E, props: EventDef[E]): void {
  if (typeof window === 'undefined') return;
  if (!(window as any).plausible) return;
  (window as any).plausible(name, { props });
}
```

---

## Catégorisation des questions chatbot

Pour `chatbot_question.question_category`, classification regex côté front avant emit :

```typescript
function categorizeQuestion(q: string): string {
  const lower = q.toLowerCase();
  if (/dois[-\s]je\s+(acheter|vendre|trader)|faut[-\s]il\s+(acheter|vendre)|quel\s+stop|que\s+ferais|si\s+tu\s+étais\s+moi/i.test(lower)) return 'prescriptive_refusal';
  if (/pourquoi.*72|conviction|score|composant|breakdown|waterfall/.test(lower)) return 'conviction';
  if (/bos|fvg|ob|cassure|retest|invalidation|structure|order\s+block/.test(lower)) return 'structure';
  if (/régime|regime|hmm|trend|range|stress/.test(lower)) return 'regime';
  if (/vol|volat|atr|amplitude/.test(lower)) return 'vol';
  if (/fomc|nfp|cpi|event|news|publication/.test(lower)) return 'event';
  if (/historique|hit rate|profit factor|setups\s+similaires|backtest/.test(lower)) return 'history';
  if (/méthodologie|how\s+do\s+you|comment\s+ça\s+marche|algorithme|conformel/.test(lower)) return 'methodology';
  return 'general';
}
```

---

## Dashboards Plausible à créer manuellement

Une fois events flux, créer ces vues dans Plausible :

### 1. Funnel d'acquisition
- `pageview` (landing) → `signup` → `paid_conversion`
- KPI cible : conv 0.5-2% landing → trial, 10-15% trial → paid

### 2. Engagement chatbot
- Goals : `chatbot_question` avec property breakdown `question_category`
- KPI : ratio prescriptive_refusal / total (mesure du refus pédagogique en action)
- KPI : % users avec ≥ 1 chatbot_question / total signups (mesure du moat)

### 3. Exploration profondeur
- Goals : `section_expanded` avec breakdown `section_id`
- KPI : % users qui essaient la section verrouillée → upgrade_clicked

### 4. Conversion par tier
- Funnels `upgrade_clicked` par from_tier → paid_conversion
- KPI : conversion FREE → STARTER, STARTER → PRO

### 5. Retention cohort
- Plausible n'a pas de cohort analysis native — utiliser DG-162 V2 ou dashboard custom

---

## Quotas Plausible self-hosted

Plausible self-hosted est **illimité en events** (contrairement au plan cloud $9/mo). Tu peux tracker autant que tu veux sans coût additionnel.

Limite pratique : disk usage ClickHouse. Estimation : ~1 GB par million d'events. Tier Fly.io 10 GB suffit pour 1+ an de bootstrap.

---

## Tests

```python
# backend/tests/test_analytics_emit.py
async def test_paid_conversion_emitted_server_side(stripe_invoice_paid_payload):
    with mock_plausible() as mock:
        await stripe_webhook(stripe_invoice_paid_payload)

    mock.assert_called_with(
        "paid_conversion",
        props=dict(
            tier="STARTER",
            price_id="price_starter_monthly_v1",
            amount_cents=2900,
            currency="usd",
            billing_cycle="monthly",
            trial_converted=True,
        ),
    )

async def test_no_pii_in_event_props():
    """Vérifie qu'on n'envoie jamais email ou IP exacte"""
    for evt in CAPTURED_EVENTS:
        props = evt['props']
        for v in props.values():
            assert "@" not in str(v), f"Email leaked in event {evt['name']}"
            assert not re.match(r"\d+\.\d+\.\d+\.\d+", str(v)), f"IP leaked in event {evt['name']}"
```

---

## Pièges

- ❌ **Mettre email user dans props** : violation RGPD + Plausible refuse les PII
- ❌ **Tracker `paid_conversion` côté front** : ad-blockers bloquent. DOIT être server-side via webhook Stripe.
- ❌ **Catégoriser chatbot questions côté backend** : ajoute latence + couplage. Catégorisation regex côté front suffit.
- ❌ **Plausible cloud à la place de self-hosted** : viole budget bootstrap ($9/mo économisé). Self-hosted requis V0.
- ❌ **Oublier `session_id_hash`** : sans, impossible de mesurer "engagement chatbot per session" (questions consécutives).
