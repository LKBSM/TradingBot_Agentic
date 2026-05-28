# DG-101-MODIFIED — Renderer unique + sections collapsibles tier-gated

> ⚠️ **PARTIELLEMENT OBSOLÈTE depuis pivot 2026-05-27**
> - Hero card NE DOIT PLUS afficher "PF historique 1.30 [1.12-1.49]" (claim retiré, voir `pivot_positioning_2026_05_27` + `decisions/2026-05-27_pivot_positioning_audit.md`).
> - Remplacer par : badge "Méthodologie publique · OOS validation pending" ou jauge de conformal coverage.
> - Tiers `'weak' | 'moderate' | 'strong' | 'institutional'` (conviction_label) restent **backend uniquement** (debug). UI client n'affiche pas le label tier brut.
> - Re-saisir ces points dans le brief lors du sprint S3 effectif.

**Effort** : ~16-24h · **Sprint** : S3 · **Owner** : code

---

## Objectif

Implémenter en Next.js l'**architecture progressive uniforme** validée le 2026-05-26 (remplace le toggle 3 modes DG-100 DROP). Un seul layout responsive, hero card permanent + sections collapsibles dépliables au clic, gating tier par disponibilité de contenu (pas par layout).

## Contexte

C'est le **cœur visuel du produit**. Référence : `mockups/v3/best_concept_demo.html` (réécrit 2026-05-26 en architecture progressive uniforme — à porter en Next.js).

## Périmètre

**IN** :
- Composant `<LectureView>` qui rend une lecture complète
- Hero card permanent (visible toujours, sans déploiement)
- 4 sections collapsibles (Pourquoi conviction / Régime+Vol / Structure / Détail technique)
- Gating tier : sections verrouillées avec overlay "Strategist required" si tier insuffisant
- Bouton "Tout déplier" pour STRATEGIST+ (1 clic = tout ouvert)
- Bouton "Demander à Sentinel" qui scroll vers chatbot
- Composants typés Pydantic→TypeScript (générés à partir du contrat `InsightSignalV2`)

**OUT** :
- Mobile responsive (DG-103, sprint S4)
- Chatbot wiring (DG-110, sprint S4)
- Page pricing (DG-132, sprint S5)

## Dépendances

- DG-023 Next.js 15 + Tailwind + shadcn initialisé
- Endpoint backend `/api/v1/lectures/:id` qui retourne `InsightSignalV2` v2.1.0
- DG-161 wrapper analytics (pour event `section_expanded`)

## Fichiers à toucher

```
frontend/
├── app/
│   └── lecture/[id]/page.tsx         (à créer)
├── components/
│   ├── LectureView.tsx               (à créer — composant principal)
│   ├── HeroCard.tsx                  (à créer)
│   ├── CollapsibleSection.tsx        (à créer — réutilisable)
│   ├── TierGate.tsx                  (à créer — overlay verrouillé)
│   ├── ConvictionSection.tsx         (à créer — section "Pourquoi 72")
│   ├── RegimeVolSection.tsx          (à créer)
│   ├── StructureSection.tsx          (à créer)
│   └── ExpertSection.tsx             (à créer — détail technique tier-gated)
├── lib/
│   ├── api.ts                        (fetch wrapper)
│   ├── types.ts                      (TS types depuis InsightSignalV2)
│   └── analytics.ts                  (déjà créé DG-161)
└── styles/
    └── lecture.css                   (ou modules CSS si choix Tailwind utility-only)
```

## Implémentation

### 1. Types TS depuis Pydantic

Générer les types TS à partir de `src/api/insight_signal_v2.py` via `pydantic-to-typescript` ou écrire manuellement :

```typescript
// frontend/lib/types.ts
export type Direction = 'BULLISH_SETUP' | 'BEARISH_SETUP' | 'NEUTRAL';
export type Tier = 'FREE' | 'STARTER' | 'PRO' | 'STRATEGIST' | 'INSTITUTIONAL';
// FREE et STARTER pour le tier d'abonnement; STRATEGIST utilisé comme alias pour PRO+
// Note : ANALYST dans les docs anciens = STARTER. Aligner sur DG-070 final.

export interface InsightSignalV2 {
  id: string;
  instrument: 'XAUUSD' | 'EURUSD';
  timeframe: 'M15' | 'H1' | 'H4' | 'D1';
  direction: Direction;
  conviction_0_100: number;
  conviction_label: 'weak' | 'moderate' | 'strong' | 'institutional';
  uncertainty: {
    conformal_lower: number;
    conformal_upper: number;
    coverage_alpha: number;
    empirical_coverage: number;
  };
  structure_readout: {
    bos_level: number | null;
    fvg_zone: [number, number] | null;
    ob_zone: [number, number] | null;
    retest_state: 'idle' | 'awaiting' | 'armed' | 'consumed';
    structural_invalidation: number | null;
    choch_present: boolean;
  };
  regime_readout: {
    hmm_label: string;
    hmm_posterior: number;
    bocpd_changepoint_prob: number;
    jump_ratio: number;
    regime_gate_decision: 'TRADE' | 'REDUCE' | 'BLOCK';
  };
  volatility_readout: {
    regime: 'low' | 'normal' | 'high';
    forecast_atr_pips: number;
    naive_atr_pips: number;
    forecast_vs_naive_pct: number;
  };
  event_readout: {
    news_blackout_active: boolean;
    next_event_label: string | null;
    next_event_in_minutes: number | null;
    session: string;
  };
  breakdown_components: Array<{
    name: string;
    contribution: number;
    weight_max: number;
    reasoning: string;
  }>;
  historical_stats: {
    similar_setups_n: number;
    hit_rate_observed: number;
    profit_factor: number;
    profit_factor_ci95: [number, number];
    backtest_window: string;
  };
  narrative_short: string;
  narrative_long: string;
  narrative_language: 'fr' | 'en' | 'de' | 'es';
  compliance: {
    is_paper_demo: boolean;
    edge_claim: boolean;
    jurisdiction_blocked: string[];
  };
  created_at_utc: string;
  valid_until_utc: string;
}
```

### 2. Composant principal `<LectureView>`

```tsx
// frontend/components/LectureView.tsx
'use client';

import { useState } from 'react';
import HeroCard from './HeroCard';
import CollapsibleSection from './CollapsibleSection';
import ConvictionSection from './ConvictionSection';
import RegimeVolSection from './RegimeVolSection';
import StructureSection from './StructureSection';
import ExpertSection from './ExpertSection';
import { InsightSignalV2, Tier } from '@/lib/types';

interface Props {
  signal: InsightSignalV2;
  userTier: Tier;
}

export default function LectureView({ signal, userTier }: Props) {
  const [allOpen, setAllOpen] = useState(false);

  const isExpertUnlocked = ['PRO', 'STRATEGIST', 'INSTITUTIONAL'].includes(userTier);

  return (
    <main className="market-frame">
      <HeroCard signal={signal} onAskChatbot={() => document.getElementById('chatbot')?.scrollIntoView({ behavior: 'smooth' })} onToggleAll={() => setAllOpen(o => !o)} allOpen={allOpen} />

      <CollapsibleSection id="conviction" title="Pourquoi cette conviction ?" summary={`Conviction ${signal.conviction_0_100} · marge ${signal.uncertainty.conformal_lower}–${signal.uncertainty.conformal_upper}`} defaultOpen={allOpen}>
        <ConvictionSection signal={signal} />
      </CollapsibleSection>

      <CollapsibleSection id="regime_vol" title="Régime de marché + volatilité" summary={`${signal.regime_readout.hmm_label} · vol ${signal.volatility_readout.forecast_vs_naive_pct > 0 ? '+' : ''}${signal.volatility_readout.forecast_vs_naive_pct.toFixed(0)}% vs normale`} defaultOpen={allOpen}>
        <RegimeVolSection signal={signal} />
      </CollapsibleSection>

      <CollapsibleSection id="structure" title="Structure du marché" summary={`Cassure ${signal.structure_readout.bos_level?.toFixed(2)} · invalidation ${signal.structure_readout.structural_invalidation?.toFixed(2)}`} defaultOpen={allOpen}>
        <StructureSection signal={signal} />
      </CollapsibleSection>

      <CollapsibleSection id="expert" title="Détail technique" tierRequired="STRATEGIST" userTier={userTier} defaultOpen={allOpen && isExpertUnlocked}>
        <ExpertSection signal={signal} />
      </CollapsibleSection>
    </main>
  );
}
```

### 3. `<CollapsibleSection>` réutilisable avec gating

```tsx
// frontend/components/CollapsibleSection.tsx
'use client';

import { useState, useEffect } from 'react';
import { trackEvent } from '@/lib/analytics';
import { Tier } from '@/lib/types';

interface Props {
  id: string;
  title: string;
  summary?: string;
  tierRequired?: Tier;
  userTier?: Tier;
  defaultOpen?: boolean;
  children: React.ReactNode;
}

const TIER_ORDER: Record<Tier, number> = {
  FREE: 0,
  STARTER: 1,
  PRO: 2,
  STRATEGIST: 2,         // alias
  INSTITUTIONAL: 3,
};

export default function CollapsibleSection({ id, title, summary, tierRequired, userTier = 'FREE', defaultOpen = false, children }: Props) {
  const [open, setOpen] = useState(defaultOpen);

  useEffect(() => setOpen(defaultOpen), [defaultOpen]);

  const isLocked = tierRequired ? TIER_ORDER[userTier] < TIER_ORDER[tierRequired] : false;

  const handleToggle = () => {
    if (isLocked) return;
    const newOpen = !open;
    setOpen(newOpen);
    if (newOpen) {
      trackEvent('section_expanded', { section_id: id, tier_user: userTier });
    }
  };

  return (
    <div className={`collapsible ${open ? 'open' : ''} ${isLocked ? 'locked' : ''}`} data-section-id={id}>
      <button
        className="collapsible-head"
        onClick={handleToggle}
        aria-expanded={open}
        aria-disabled={isLocked}
      >
        <span className="ch-left">
          <span className="ch-icon" aria-hidden>▾</span>
          <span>{title}</span>
        </span>
        {isLocked ? (
          <span className="ch-lock">🔒 {tierRequired}</span>
        ) : (
          summary && <span className="ch-summary">{summary}</span>
        )}
      </button>
      {open && !isLocked && (
        <div className="collapsible-body">
          {children}
        </div>
      )}
      {isLocked && (
        <div className="locked-overlay">
          <strong>Section réservée au tier {tierRequired}.</strong>
          <a href="/pricing">Voir les tarifs →</a>
        </div>
      )}
    </div>
  );
}
```

### 4. CSS (Tailwind ou modules selon convention équipe)

Reprendre les classes CSS du mockup HTML (`mockups/v3/best_concept_demo.html`) — palette finance premium déjà définie. Adapter en CSS modules ou utility classes Tailwind.

## Acceptance criteria

- [ ] Page `/lecture/[id]` affiche hero card + 4 sections collapsibles
- [ ] Hero card toujours visible, non collapsible
- [ ] Sections collapsed par défaut, sauf si bouton "Tout déplier" cliqué
- [ ] Section "Détail technique" verrouillée avec overlay si tier < PRO (`🔒 STRATEGIST`)
- [ ] Clic sur section verrouillée → pas de déploiement, lien vers `/pricing` visible dans l'overlay
- [ ] Clic "Tout déplier" pour user STRATEGIST+ → 4 sections ouvertes simultanément
- [ ] Clic "Tout déplier" pour user FREE/STARTER → 3 sections ouvertes (Détail technique reste locked)
- [ ] Event `section_expanded` émis au déploiement avec `section_id` + `tier_user`
- [ ] Bouton "Demander à Sentinel" scroll vers `#chatbot` (sera fini en DG-110)
- [ ] Hero card affiche correctement : direction (couleur sémantique vert/rouge/gris), conviction (jauge + marge), **badge "Méthodologie publique · OOS validation pending"** (en remplacement du PF historique retiré 2026-05-27), event imminent si <4h, validité countdown ("encore 2h47")
- [ ] Test visuel : rendu identique au mockup HTML sur desktop ≥ 1024px
- [ ] Test accessibility (a11y) : `aria-expanded`, `aria-disabled`, navigation clavier

## Tests requis

```typescript
// frontend/__tests__/LectureView.test.tsx
test('FREE user sees expert section locked', () => {
  const { getByText } = render(<LectureView signal={mockSignal} userTier="FREE" />);
  expect(getByText('🔒 STRATEGIST')).toBeInTheDocument();
});

test('STRATEGIST user can expand all sections', () => {
  const { getByText, getAllByRole } = render(<LectureView signal={mockSignal} userTier="STRATEGIST" />);
  fireEvent.click(getByText(/Tout déplier/));
  const sections = getAllByRole('button', { expanded: true });
  expect(sections.length).toBe(4);
});

test('section_expanded event tracked on click', () => {
  const trackSpy = jest.spyOn(analytics, 'trackEvent');
  const { getByText } = render(<LectureView signal={mockSignal} userTier="FREE" />);
  fireEvent.click(getByText('Pourquoi cette conviction ?'));
  expect(trackSpy).toHaveBeenCalledWith('section_expanded', expect.objectContaining({ section_id: 'conviction' }));
});
```

## Risques / pièges

- ❌ **Reproduire le toggle 3 modes** par habitude. C'est DROP, l'architecture est progressive uniforme. Si tu vois un menu "FOCUS / CO-PILOT / EXPERT" en haut de la page, **c'est faux**.
- ❌ **Gating tier côté front seulement** : un utilisateur peut bypasser avec DevTools. Le backend DOIT aussi refuser de servir le détail si tier insuffisant. Le front masque, le back protège.
- ❌ **Charger toutes les données même pour sections verrouillées** : gaspille bandwidth + LLM costs. Charge seulement ce que le tier autorise (lazy load section "expert" si user PRO+).
- ❌ **Oublier le bouton "Tout déplier"** : c'est ce qui rend l'archi acceptable pour les power-users PRO+ (sinon ils cliquent 4 fois).
- ❌ **Forcer un layout fixe** : la responsivité mobile (DG-103) doit pouvoir réutiliser ce composant sans réécriture.
