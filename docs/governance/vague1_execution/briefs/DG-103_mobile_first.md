# DG-103 — Mobile-first responsive

**Effort** : ~16h · **Sprint** : S4 · **Owner** : code

---

## Objectif

Rendre toute l'application **mobile-first responsive** (< 768px). 60-70 % du retail consulte sur mobile, le plan original l'ignore totalement.

## Contexte (angle mort plan original)

Sans mobile-first, bounce massif sur landing + Telegram = seule surface mobile. Avec mobile-first, le produit est utilisable comme un GPS de voiture (ce qui correspond exactement à l'analogie de la vision produit).

## Périmètre

**IN** :
- Landing page mobile (DG-120)
- Page lecture `/lecture/[id]` mobile (DG-101)
- Page pricing mobile (DG-132)
- Chatbot FAB sur mobile (icône bas-droite) au lieu de sidebar
- Hero card adapté petit écran (1 colonne au lieu de 2)
- Sections collapsibles adaptées (font-size, padding, gap réduits)
- Footer compliance lisible

**OUT** :
- App native iOS/Android (DG-180, P2 Vague 3)
- PWA installable (DG-152, V2)
- Lock screen widget (P2)

## Dépendances

- DG-101 renderer unique implémenté
- DG-120 landing hero card en desktop d'abord
- DG-110 chatbot wiring (pour FAB mobile)

## Fichiers à toucher

```
frontend/
├── app/
│   ├── globals.css                   (media queries mobile)
│   └── layout.tsx                    (viewport meta + breakpoints)
├── components/
│   ├── LectureView.tsx               (responsive grid)
│   ├── HeroCard.tsx                  (1 col mobile / 2 cols desktop)
│   ├── CollapsibleSection.tsx        (padding adapté)
│   ├── Chatbot.tsx                   (sidebar desktop / FAB mobile)
│   └── ChatbotFAB.tsx                (à créer)
└── styles/
    └── responsive.css                (breakpoints)
```

## Implémentation

### Breakpoints standards

```css
/* mobile-first : styles par défaut = mobile */
/* tablet */    @media (min-width: 640px) { ... }
/* desktop */   @media (min-width: 1024px) { ... }
/* large */     @media (min-width: 1280px) { ... }
```

Tailwind utilities :
- `sm:` ≥ 640px
- `md:` ≥ 768px
- `lg:` ≥ 1024px (desktop)
- `xl:` ≥ 1280px

### Adaptations critiques

#### 1. Hero card

```tsx
// frontend/components/HeroCard.tsx
<div className="hero-card p-5 md:p-8 lg:p-10">
  <div className="hero-row flex flex-wrap items-center justify-between gap-3">
    {/* ... */}
  </div>

  {/* 1 colonne mobile, 2 colonnes desktop */}
  <div className="hero-grid grid grid-cols-1 md:grid-cols-[1.4fr_1fr] gap-3 md:gap-5">
    <TrackRecordCard {...} />
    <ConvictionCard {...} />
  </div>

  {/* Event banner si applicable */}
  {nextEventInMinutes && nextEventInMinutes < 240 && (
    <div className="hero-event">⚠ FOMC dans {formatTime(nextEventInMinutes)}</div>
  )}

  <p className="hero-verbal italic text-sm md:text-base">{narrative_short}</p>

  <div className="hero-actions flex flex-col sm:flex-row gap-3">
    <button className="btn-primary w-full sm:w-auto">💬 Demander à Sentinel</button>
    <button className="btn-secondary w-full sm:w-auto">📊 Tout déplier</button>
  </div>
</div>
```

#### 2. Chatbot — sidebar desktop / FAB mobile

```tsx
// frontend/components/Chatbot.tsx
'use client';

import { useEffect, useState } from 'react';
import ChatbotSidebar from './ChatbotSidebar';
import ChatbotFAB from './ChatbotFAB';

export default function Chatbot() {
  const [isDesktop, setIsDesktop] = useState(false);

  useEffect(() => {
    const mq = window.matchMedia('(min-width: 1024px)');
    setIsDesktop(mq.matches);
    const handler = (e: MediaQueryListEvent) => setIsDesktop(e.matches);
    mq.addEventListener('change', handler);
    return () => mq.removeEventListener('change', handler);
  }, []);

  return isDesktop ? <ChatbotSidebar /> : <ChatbotFAB />;
}
```

```tsx
// frontend/components/ChatbotFAB.tsx
import { useState } from 'react';
import ChatbotPanel from './ChatbotPanel';

export default function ChatbotFAB() {
  const [open, setOpen] = useState(false);
  return (
    <>
      {open && (
        <div className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm" onClick={() => setOpen(false)}>
          <div className="absolute bottom-0 inset-x-0 h-[75vh] rounded-t-xl bg-bg-card shadow-2xl p-4" onClick={e => e.stopPropagation()}>
            <ChatbotPanel onClose={() => setOpen(false)} />
          </div>
        </div>
      )}
      <button
        className="fixed bottom-6 right-6 z-40 h-14 w-14 rounded-full bg-gold text-bg-page text-2xl shadow-lg flex items-center justify-center lg:hidden"
        onClick={() => setOpen(true)}
        aria-label="Demander à Sentinel"
      >
        💬
      </button>
    </>
  );
}
```

#### 3. Sections collapsibles — padding mobile

```css
/* styles/responsive.css */
.collapsible-head {
  padding: 14px 18px;  /* mobile */
}
.collapsible-body {
  padding: 4px 18px 22px;
}
@media (min-width: 768px) {
  .collapsible-head {
    padding: 18px 28px;
  }
  .collapsible-body {
    padding: 4px 28px 28px;
  }
}
```

#### 4. Waterfall 8 composantes — grille adaptée

```tsx
// Mobile : grille condensée
<div className="wf-row grid grid-cols-[100px_1fr_60px_45px] sm:grid-cols-[160px_1fr_80px_60px] gap-2 sm:gap-4 text-xs sm:text-sm">
  <span>{name}</span>
  <div className="wf-bar"><div className="wf-bar-fill" style={{ width: `${pct}%` }} /></div>
  <span className="text-right tabular-nums">{contribution}</span>
  <span className="text-right tabular-nums text-text-muted">{weight}%</span>
</div>
```

#### 5. Viewport meta + accessibility

```tsx
// frontend/app/layout.tsx
export const metadata = {
  viewport: 'width=device-width, initial-scale=1, maximum-scale=5',
};
```

## Acceptance criteria

- [ ] Test sur iPhone SE (375px), iPhone 14 Pro (393px), iPad Mini (768px), desktop 1440px
- [ ] Hero card mobile : tout visible sans scroll horizontal
- [ ] Hero card mobile : 1 seule colonne (track record + conviction empilés)
- [ ] Buttons "Demander à Sentinel" + "Tout déplier" : largeur 100% mobile, auto desktop
- [ ] Chatbot : FAB visible mobile (lg:hidden), sidebar visible desktop (hidden lg:flex)
- [ ] Touch targets ≥ 44×44px (Apple HIG / Material) sur boutons et chevrons collapsibles
- [ ] Texte minimum 14px en body, jamais < 12px (sauf footer compliance)
- [ ] Aucun scroll horizontal forcé
- [ ] Performance : Lighthouse mobile ≥ 90 (LCP < 2.5s, CLS < 0.1)
- [ ] Waterfall composantes lisible mobile (4 colonnes condensées)
- [ ] Footer compliance lisible mobile (multiline OK)

## Tests requis

```typescript
// frontend/__tests__/responsive.test.tsx
test.each([
  ['iPhone SE', 375, 667],
  ['iPhone 14', 393, 852],
  ['iPad Mini', 768, 1024],
  ['Desktop', 1440, 900],
])('layout works at %s (%dx%d)', async (name, w, h) => {
  cy.viewport(w, h);
  cy.visit('/lecture/abc123');
  cy.get('[data-testid=hero-card]').should('be.visible');
  // ... assertions par viewport
});
```

Test manuel Playwright + screenshots :
```bash
npx playwright test --grep responsive --update-snapshots
```

## Risques / pièges

- ❌ **Designer desktop puis "adapter" mobile** : approche désuète. Mobile-first signifie écrire les styles mobile par défaut, puis ajouter `md:` / `lg:` pour les écrans plus grands.
- ❌ **Charger lourd côté mobile** : optimiser images (Next.js `<Image>`), code-split routes (Next App Router fait nativement)
- ❌ **Touch targets trop petits** : 44×44px minimum. Tester avec doigt sur vrai téléphone, pas seulement DevTools.
- ❌ **Oublier landscape mobile** : 568×320 environ. Vérifier que le hero reste lisible.
- ❌ **FAB chatbot qui masque du contenu** : ajouter `padding-bottom: 80px` au footer mobile pour éviter recouvrement
- ❌ **Texte trop dense mobile** : préférer `line-height: 1.6` mobile (vs 1.5 desktop)
