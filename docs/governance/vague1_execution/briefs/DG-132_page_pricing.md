# DG-132 — Page pricing avec decoy + dual trial

> ⚠️ **OBSOLÈTE POST PIVOT 2026-05-27** — Pricing révisé en **FREE / 9 € / 19 €** (au lieu de $29/$79/$1990). INSTITUTIONAL retiré de la grille publique → "Contact us / Calendly". Pour la grille à jour, utiliser **`docs/governance/vague1_execution/copies/pricing_copy.md`** (révisé 2026-05-27). Voir `docs/governance/decisions/2026-05-27_pivot_positioning_audit.md`.

**Effort** : ~10-16h · **Sprint** : S5 · **Owner** : code

---

## Objectif

Construire la page `/pricing` complète avec :
- **4 tiers** affichés en parallèle (FREE / STARTER $29 / PRO $79 / INSTITUTIONAL $1990 — decoy)
- **Dual trial** : 14j sans CB (FREE→STARTER) + 14j avec CB (STARTER→PRO)
- **Toggle mensuel/annuel** (-16.7% annuel)
- **Refund 30j** mentionné en chaque card
- **Bouton CTA tier-appropriate** (essai gratuit / book demo)

## Contexte

Sans page pricing fonctionnelle, pas de checkout possible. Le decoy $1990 a un effet psychologique mesuré +25-40 % de conversion sur PRO (eval_27).

## Périmètre

**IN** :
- 4 cards tarifs side-by-side desktop, stacked mobile
- Toggle monthly/yearly avec discount visible
- Bouton CTA différent par tier :
  - FREE : "Commencer gratuitement" → /signup
  - STARTER : "Essayer 14 jours sans CB" → /signup?tier=starter&trial=nocard
  - PRO : "Essayer 14 jours avec CB" → /signup?tier=pro&trial=withcard (devient `/checkout?...`)
  - INSTITUTIONAL : "Réserver une démo" → Calendly embed ou link
- Liste features par tier avec checkmarks/locks
- Refund 30j codifié en footer de chaque card
- Section "Pourquoi vous payez pour la profondeur, pas pour le nombre de signaux"
- Tableau comparaison sections débloquées par tier

**OUT** :
- Stripe Checkout intégration (sprint S6, après pricing UI livré)
- Calendly embed configuration (S5 user action)
- A/B test pricing (V2)

## Dépendances

- DG-070 grille pricing actée (✅ signé 2026-05-26)
- DG-083 decoy visible (inclus dans DG-132)
- DG-084 dual trial (inclus dans DG-132)
- DG-080 INSTITUTIONAL Calendly (besoin du lien Calendly du user)
- DG-161 event tracking (pour `upgrade_clicked`)

## Fichiers à toucher

```
frontend/
├── app/
│   └── pricing/
│       └── page.tsx                  (à créer)
├── components/
│   ├── pricing/
│   │   ├── BillingCycleToggle.tsx    (mensuel/annuel)
│   │   ├── TierCard.tsx              (1 card tarif)
│   │   ├── PricingGrid.tsx           (4 cards)
│   │   ├── ComparisonTable.tsx       (table sections vs tier)
│   │   └── RefundBadge.tsx           (badge "Remboursé 30j")
└── lib/
    └── pricing-data.ts               (specs 4 tiers + features)
```

## Implémentation

### 1. Spec tiers centralisée

```typescript
// frontend/lib/pricing-data.ts
export type BillingCycle = 'monthly' | 'yearly';
export const ANNUAL_DISCOUNT = 0.167; // 16.7% off

export interface TierSpec {
  id: 'FREE' | 'STARTER' | 'PRO' | 'INSTITUTIONAL';
  name: string;
  prices: { monthly: number; yearly: number };
  currency: 'USD';
  target: string;
  ctaLabel: string;
  ctaHref: string;
  featured?: boolean;
  features: { label: string; included: boolean }[];
  sections: string[];
  trial?: { days: number; requiresCard: boolean };
  notes?: string;
}

export const TIERS: TierSpec[] = [
  {
    id: 'FREE',
    name: 'FREE',
    prices: { monthly: 0, yearly: 0 },
    currency: 'USD',
    target: 'Découverte · 1 actif XAU · 3 lectures/jour',
    ctaLabel: 'Commencer gratuitement',
    ctaHref: '/signup?tier=free',
    features: [
      { label: '1 actif (XAU)', included: true },
      { label: 'Chatbot 5 questions/jour', included: true },
      { label: 'Hero card + lecture verbale', included: true },
      { label: 'Sections structure / régime / vol', included: false },
      { label: 'Détail technique (waterfall, RAG)', included: false },
      { label: 'Alertes event imminent', included: false },
    ],
    sections: ['Hero card', 'Lecture verbale'],
  },
  {
    id: 'STARTER',
    name: 'Starter',
    prices: { monthly: 29, yearly: 29 * 12 * (1 - ANNUAL_DISCOUNT) },
    currency: 'USD',
    target: 'Trader engagé · 4 actifs · 30 lectures/jour',
    ctaLabel: '14 jours d\'essai sans CB',
    ctaHref: '/signup?tier=starter&trial=nocard',
    trial: { days: 14, requiresCard: false },
    features: [
      { label: '4 actifs (XAU, EUR + 2)', included: true },
      { label: 'Chatbot 100 questions/jour', included: true },
      { label: 'Hero card + Conviction + Régime + Vol + Structure', included: true },
      { label: 'Alertes event imminent', included: true },
      { label: 'Détail technique (waterfall, RAG)', included: false },
    ],
    sections: ['Hero', 'Conviction', 'Régime+Vol', 'Structure'],
  },
  {
    id: 'PRO',
    name: 'Pro',
    prices: { monthly: 79, yearly: 79 * 12 * (1 - ANNUAL_DISCOUNT) },
    currency: 'USD',
    target: 'Power user · 6 actifs · illimité',
    featured: true,
    ctaLabel: '14 jours d\'essai avec CB',
    ctaHref: '/signup?tier=pro&trial=withcard',
    trial: { days: 14, requiresCard: true },
    features: [
      { label: '6 actifs · lectures illimitées', included: true },
      { label: 'Chatbot illimité (cache sémantique)', included: true },
      { label: 'Waterfall 8 composantes + conformal viz', included: true },
      { label: 'Sources RAG académiques cliquables', included: true },
      { label: 'Exports CSV · email digest hebdo', included: true },
    ],
    sections: ['Tout débloqué (FOCUS + Pourquoi conviction + Régime+Vol + Structure + Détail technique)'],
  },
  {
    id: 'INSTITUTIONAL',
    name: 'Institutional',
    prices: { monthly: 1990, yearly: 1990 * 12 * 0.9 }, // 10% off engagement annuel
    currency: 'USD',
    target: 'Broker · family office · intégration API',
    ctaLabel: 'Réserver une démo',
    ctaHref: 'https://calendly.com/mia-markets/demo-institutional', // à compléter
    notes: 'Engagement 12 mois minimum',
    features: [
      { label: 'API B2B JSON complet', included: true },
      { label: 'Webhooks signés HMAC', included: true },
      { label: 'SLA 99.9% · support dédié', included: true },
      { label: 'Licence redistribution white-label', included: true },
      { label: 'Engagement 12 mois min', included: true },
    ],
    sections: ['Tout débloqué + API B2B'],
  },
];
```

### 2. Toggle billing cycle

```tsx
// frontend/components/pricing/BillingCycleToggle.tsx
'use client';
import { useState } from 'react';

export default function BillingCycleToggle({ value, onChange }: { value: BillingCycle; onChange: (c: BillingCycle) => void }) {
  return (
    <div className="inline-flex bg-bg-elevated border border-border rounded-full p-1">
      <button
        className={`px-5 py-2 text-sm rounded-full transition ${value === 'monthly' ? 'bg-gold text-bg-page font-medium' : 'text-text-secondary'}`}
        onClick={() => onChange('monthly')}
      >
        Mensuel
      </button>
      <button
        className={`px-5 py-2 text-sm rounded-full transition flex items-center gap-2 ${value === 'yearly' ? 'bg-gold text-bg-page font-medium' : 'text-text-secondary'}`}
        onClick={() => onChange('yearly')}
      >
        Annuel <span className="text-xs opacity-80">−16,7 %</span>
      </button>
    </div>
  );
}
```

### 3. Tier Card

```tsx
// frontend/components/pricing/TierCard.tsx
import { trackEvent } from '@/lib/analytics';
import { TierSpec, BillingCycle } from '@/lib/pricing-data';
import RefundBadge from './RefundBadge';

export default function TierCard({ tier, cycle, currentTier }: { tier: TierSpec; cycle: BillingCycle; currentTier: TierSpec['id'] }) {
  const price = cycle === 'monthly' ? tier.prices.monthly : Math.round(tier.prices.yearly / 12);
  const cycleSuffix = cycle === 'monthly' ? '/ mois' : '/ mois (facturé annuel)';

  const handleCta = () => {
    trackEvent('upgrade_clicked', {
      from_tier: currentTier,
      to_tier: tier.id,
      cta_location: 'pricing_page',
    });
    window.location.href = tier.ctaHref;
  };

  return (
    <article className={`tier-card relative p-6 bg-bg-card border rounded-xl flex flex-col gap-3 ${tier.featured ? 'border-gold bg-gradient-to-b from-gold/5 to-bg-card' : 'border-border'}`}>
      {tier.featured && (
        <span className="absolute -top-2.5 left-1/2 -translate-x-1/2 px-3 py-1 text-[10px] font-bold tracking-widest bg-gold text-bg-page rounded">
          RECOMMANDÉ
        </span>
      )}
      <div className="text-sm uppercase tracking-widest text-text-muted">{tier.name}</div>
      <div className="text-3xl font-semibold font-mono">
        {price === 0 ? 'Gratuit' : `${price} $`}
        {price > 0 && <small className="text-sm text-text-muted font-normal block">{cycleSuffix}</small>}
      </div>
      <p className="text-sm text-text-secondary">{tier.target}</p>

      <div className="text-xs">
        <div className="font-semibold text-gold uppercase tracking-widest text-[10px] mb-1.5">Sections débloquées</div>
        <div className="text-text-muted">{tier.sections.join(' · ')}</div>
      </div>

      <ul className="text-sm flex flex-col gap-1.5">
        {tier.features.map((f, i) => (
          <li key={i} className={`flex items-start gap-2 ${f.included ? 'text-text-secondary' : 'text-text-muted opacity-60'}`}>
            <span className={`${f.included ? 'text-bullish' : 'text-text-dim'} font-bold`}>{f.included ? '✓' : '×'}</span>
            {f.label}
          </li>
        ))}
      </ul>

      {tier.id !== 'INSTITUTIONAL' && tier.id !== 'FREE' && <RefundBadge />}

      <button
        onClick={handleCta}
        className={`mt-auto w-full py-2.5 text-sm font-medium rounded-lg border transition ${tier.featured ? 'bg-gold text-bg-page border-gold hover:opacity-90' : 'bg-bg-elevated text-text-primary border-border-light hover:bg-gold hover:text-bg-page hover:border-gold'}`}
      >
        {tier.ctaLabel}
      </button>

      {tier.notes && <p className="text-xs text-text-muted text-center">{tier.notes}</p>}
    </article>
  );
}
```

### 4. Refund badge

```tsx
// frontend/components/pricing/RefundBadge.tsx
export default function RefundBadge() {
  return (
    <div className="text-xs text-text-muted flex items-center gap-1.5 bg-bg-elevated px-2.5 py-1.5 rounded">
      <span className="text-bullish">✓</span>
      Remboursement intégral 30 jours, sans question
    </div>
  );
}
```

### 5. Page complète

```tsx
// frontend/app/pricing/page.tsx
'use client';
import { useState } from 'react';
import { TIERS, BillingCycle } from '@/lib/pricing-data';
import BillingCycleToggle from '@/components/pricing/BillingCycleToggle';
import TierCard from '@/components/pricing/TierCard';
import ComparisonTable from '@/components/pricing/ComparisonTable';

export default function PricingPage() {
  const [cycle, setCycle] = useState<BillingCycle>('monthly');
  const currentTier = 'FREE'; // ou depuis context user authentifié

  return (
    <main className="container mx-auto px-7 py-14 md:py-20">
      <div className="text-center mb-10">
        <div className="text-xs text-gold uppercase tracking-widest font-semibold mb-3">Tarifs &amp; différenciation</div>
        <h1 className="text-3xl md:text-4xl font-semibold tracking-tight mb-3 max-w-3xl mx-auto">
          Vous payez pour la profondeur, pas pour le nombre de signaux.
        </h1>
        <p className="text-base md:text-lg text-text-secondary max-w-2xl mx-auto leading-relaxed">
          Le pipeline algorithmique produit la même donnée pour tout le monde. Ce qui change : la profondeur de lecture que vous débloquez en montant de tier.
        </p>
      </div>

      <div className="flex justify-center mb-10">
        <BillingCycleToggle value={cycle} onChange={setCycle} />
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-14 max-w-7xl mx-auto">
        {TIERS.map(t => <TierCard key={t.id} tier={t} cycle={cycle} currentTier={currentTier} />)}
      </div>

      <ComparisonTable tiers={TIERS} />

      <p className="text-center text-xs text-text-muted mt-10 max-w-3xl mx-auto">
        M.I.A. Markets est en phase d'accès anticipé. Les fonctionnalités évoluent.
        Inscription limitée à 50 abonnés pendant cette phase. Service proposé en tant qu'outil pédagogique d'analyse algorithmique, sans recommandation d'investissement.
      </p>
    </main>
  );
}
```

## Acceptance criteria

- [ ] Page `/pricing` accessible, render < 2s LCP
- [ ] 4 cards visibles côte-à-côte ≥ 1024px, stacked < 1024px
- [ ] Tier PRO (featured) affiche badge "RECOMMANDÉ" doré
- [ ] Tier INSTITUTIONAL $1990 toujours visible (decoy) — n'est pas masqué sur mobile
- [ ] Toggle mensuel/annuel → prix mis à jour dans toutes les cards en <100ms
- [ ] Annuel affiche le discount −16.7 % et le suffix "facturé annuel"
- [ ] CTA FREE → `/signup?tier=free`
- [ ] CTA STARTER → `/signup?tier=starter&trial=nocard`
- [ ] CTA PRO → `/signup?tier=pro&trial=withcard`
- [ ] CTA INSTITUTIONAL → lien Calendly (à compléter avec URL fournie par toi)
- [ ] Refund 30j badge visible sur STARTER + PRO
- [ ] Event `upgrade_clicked` émis avec from/to tier + cta_location='pricing_page'
- [ ] Aucun vocabulaire interdit (audit)
- [ ] Mobile : 4 cards stacked, toggle visible, lisibilité OK 375px
- [ ] Audit Lighthouse mobile ≥ 90

## Tests requis

```typescript
// frontend/__tests__/pricing.spec.ts
test('all 4 tiers visible on pricing page', () => {
  const { getByText } = render(<PricingPage />);
  expect(getByText('FREE')).toBeInTheDocument();
  expect(getByText('Starter')).toBeInTheDocument();
  expect(getByText('Pro')).toBeInTheDocument();
  expect(getByText('Institutional')).toBeInTheDocument();
});

test('toggle billing cycle updates prices', async () => {
  const { getByText, findByText } = render(<PricingPage />);
  expect(getByText('29 $')).toBeInTheDocument();
  fireEvent.click(getByText(/Annuel/));
  await findByText('24 $'); // 29 × (1 - 0.167) = 24.16 arrondi 24
});

test('CTA emits upgrade_clicked event with correct tiers', () => {
  const spy = jest.spyOn(analytics, 'trackEvent');
  const { getByText } = render(<PricingPage />);
  fireEvent.click(getByText("14 jours d'essai avec CB"));
  expect(spy).toHaveBeenCalledWith('upgrade_clicked', expect.objectContaining({ to_tier: 'PRO' }));
});

test('PRO tier is featured with RECOMMANDÉ badge', () => {
  const { getByText } = render(<PricingPage />);
  expect(getByText('RECOMMANDÉ')).toBeInTheDocument();
});
```

## Risques / pièges

- ❌ **Cacher INSTITUTIONAL $1990 sur mobile** : c'est le decoy, sa visibilité augmente la conversion PRO. Garder TOUJOURS les 4 cards visibles.
- ❌ **Toggle annuel calcule mal** : 29 × 12 × 0.833 = 290.7 → arrondi 290 €/an = 24.17 €/mois. Display 24 €/mois "facturé 290 € annuellement".
- ❌ **CTA PRO sans trial card** : oublier de coller `trial=withcard` au paramètre URL → l'inscription enchaîne sans CB requise.
- ❌ **Bouton "S'abonner maintenant" sur INSTITUTIONAL** : devrait être "Réserver une démo" (DG-080). Standard B2B SaaS pour $1990/mo.
- ❌ **Discount yearly < 10 %** : 16.7 % est la valeur eval_27 (≈ 2 mois gratuits). Si tu mets 5 %, c'est ridicule.
