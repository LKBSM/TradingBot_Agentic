# DG-120 — Landing hero card track-record

> ⚠️ **OBSOLÈTE POST PIVOT 2026-05-27** — Les chiffres PF 1.30 / 329 setups / IC 95 % référencés ci-dessous ont été **retirés** par décision `docs/governance/decisions/2026-05-27_pivot_positioning_audit.md`. Pour les copies à jour, utiliser **`docs/governance/vague1_execution/copies/landing_copy.md`** (révisé 2026-05-27). Hero stats actuels : 8 facteurs / 12 papers / 7 ans / 2 actifs. Ne pas copier les chiffres performance de ce brief.

**Effort** : ~8-12h · **Sprint** : S3 · **Owner** : code

---

## Objectif

Implémenter le **hero de la landing page** avec track-record honnête en premier plan permanent. C'est ce qui doit répondre en 5 secondes à *« Pourquoi 30$/mois plutôt qu'un indicateur gratuit ? »*.

## Contexte

C'est l'angle mort #3 du plan original (Problème #5 du Livrable 1) : valeur invisible en moins de 10s. Le PF historique avec IC bootstrap est la pépite la plus différenciante du produit — elle doit être en hero, pas en footer.

## Périmètre

**IN** :
- Landing page `/` avec tagline + sub + hero stats (4 chiffres)
- Hero card lecture XAU exemple (showcase produit) avec architecture progressive uniforme
- USP "honest confidence" en sous-tagline
- CTA principal "Essayer gratuitement" (FREE tier signup)
- Comparatif visuel "Avant (signaux Telegram brut) vs Maintenant (hero card honnête)" — optionnel mais fort
- 3 différenciateurs en bas (Track record, Chatbot moat, Honest confidence)

**OUT** :
- Page pricing complète (DG-132, sprint S5)
- Section chatbot fonctionnel démo (réutilise mockup HTML pattern dans DG-110)
- Témoignages (DG-140, V2)
- Cas d'usage (DG-141, V2)

## Dépendances

- DG-023 Next.js 15 init
- Copies finales fournies dans `copies/landing_copy.md`

## Fichiers à toucher

```
frontend/
├── app/
│   └── page.tsx                      (landing, à créer)
├── components/
│   ├── landing/
│   │   ├── PitchHero.tsx             (tagline + sub + stats)
│   │   ├── LiveLectureExample.tsx    (hero card showcase + sections collapsibles, basé sur DG-101)
│   │   ├── ChatbotShowcase.tsx       (démo conversationnelle scriptée)
│   │   ├── PricingTeaser.tsx         (4 cards condensées, lien /pricing)
│   │   └── Differentiators.tsx       (3 cards trust)
│   ├── HeroCard.tsx                  (existant DG-101 — réutilisé)
│   └── Footer.tsx                    (compliance permanent)
└── public/
    └── og-image.png                  (image OpenGraph 1200×630)
```

## Implémentation

### 1. Structure landing

```tsx
// frontend/app/page.tsx
import PitchHero from '@/components/landing/PitchHero';
import LiveLectureExample from '@/components/landing/LiveLectureExample';
import ChatbotShowcase from '@/components/landing/ChatbotShowcase';
import PricingTeaser from '@/components/landing/PricingTeaser';
import Differentiators from '@/components/landing/Differentiators';
import Footer from '@/components/Footer';

export const metadata = {
  title: "M.I.A. Markets · Outil éducatif d'analyse Or & FX",
  description: "L'analyse de marché de niveau institutionnel, traduite. 329 setups historiques, PF 1.30 [1.12-1.49], walk-forward 7 ans. Honnête sur ce qu'il ne sait pas.",
  openGraph: {
    title: "M.I.A. Markets",
    description: "L'analyse Or & FX traduite. 329 setups · PF 1.30 · IC 95 %.",
    images: ["/og-image.png"],
  },
};

export default function LandingPage() {
  return (
    <>
      <PitchHero />
      <LiveLectureExample />
      <ChatbotShowcase />
      <PricingTeaser />
      <Differentiators />
      <Footer />
    </>
  );
}
```

### 2. PitchHero — tagline + 4 stats

```tsx
// frontend/components/landing/PitchHero.tsx
export default function PitchHero() {
  return (
    <section className="container mx-auto px-7 py-14 md:py-20">
      <div className="text-center max-w-4xl mx-auto">
        <h1 className="text-3xl md:text-5xl font-semibold tracking-tight leading-tight mb-6">
          L'analyse de marché de niveau institutionnel,{' '}
          <span className="text-gold">traduite</span>.
          <br className="hidden md:block" />
          Plus un quant qui répond à toutes vos questions.
        </h1>
        <p className="text-lg text-text-secondary max-w-2xl mx-auto mb-10 leading-relaxed">
          M.I.A. Markets lit le marché Or et FX pour vous, décompose ce qu'il sait — et ce qu'il ne sait pas. Vous gardez la décision.
        </p>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-6 md:gap-12 max-w-3xl mx-auto p-7 bg-bg-card border border-border rounded-xl">
          <Stat num="329" label="setups analysés" />
          <Stat num="1.30" label="profit factor" />
          <Stat num="7 ans" label="walk-forward" />
          <Stat num="95 %" label="IC bootstrap" />
        </div>

        <div className="mt-8 flex flex-col sm:flex-row gap-3 justify-center">
          <button className="btn-primary px-7 py-3">Essayer gratuitement</button>
          <button className="btn-secondary px-7 py-3">Voir la méthodologie</button>
        </div>
      </div>
    </section>
  );
}

function Stat({ num, label }: { num: string; label: string }) {
  return (
    <div className="text-center">
      <div className="text-2xl md:text-3xl font-semibold font-mono text-gold tracking-tight">{num}</div>
      <div className="text-xs text-text-muted uppercase tracking-widest mt-1.5">{label}</div>
    </div>
  );
}
```

### 3. LiveLectureExample — réutilise DG-101 avec signal de démo

```tsx
// frontend/components/landing/LiveLectureExample.tsx
import LectureView from '@/components/LectureView';
import { LANDING_DEMO_SIGNAL } from '@/lib/demo-signals';

export default function LiveLectureExample() {
  return (
    <section className="container mx-auto px-7 py-14 md:py-20 border-t border-border">
      <div className="mb-8">
        <div className="section-eyebrow text-gold font-semibold uppercase tracking-widest text-xs mb-3">Démo live</div>
        <h2 className="text-2xl md:text-4xl font-semibold tracking-tight mb-3">Tout est là. Vous décidez ce que vous voulez voir.</h2>
        <p className="text-text-secondary max-w-2xl">Une seule lecture, structurée en couches. Le hero card reste visible. Le détail s'ouvre quand vous le voulez.</p>
      </div>

      <LectureView signal={LANDING_DEMO_SIGNAL} userTier="FREE" />

      <p className="text-center text-sm text-text-muted mt-5">
        Vue actuelle : <strong className="text-gold">FREE</strong> · le détail technique reste verrouillé en démo.
        <a href="/pricing" className="text-accent ml-1">Débloquer →</a>
      </p>
    </section>
  );
}
```

### 4. Différenciateurs

```tsx
// frontend/components/landing/Differentiators.tsx
export default function Differentiators() {
  return (
    <section className="container mx-auto px-7 py-14 md:py-20 border-t border-border">
      <h3 className="text-center text-xs text-gold uppercase tracking-widest font-semibold mb-8">Trois différenciateurs que la concurrence ne peut pas copier court terme</h3>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5 max-w-5xl mx-auto">
        <DiffCard icon="📊" title="Track record honnête en hero" body="Profit factor 1.30 avec intervalle de confiance bootstrap (1.12 – 1.49) sur 329 setups, walk-forward 7 ans. Aucun concurrent retail ne publie ses stats avec une marge d'erreur honnête." />
        <DiffCard icon="💬" title="Chatbot comme moyen principal" body="Sentinel ne livre pas un dashboard à lire — il dialogue avec vous. Il définit le jargon, décompose la conviction, et refuse de vous donner un ordre. C'est ce qui sépare un indicateur d'un assistant." />
        <DiffCard icon="🛡️" title="Honest confidence assumée" body={<>« Nous ne vous disons pas quoi faire — nous vous donnons les meilleurs outils pour décider. »<br />Posture éducative, compliance UE 2024/2811 par construction, sources académiques visibles.</>} />
      </div>
    </section>
  );
}
```

## Acceptance criteria

- [ ] Page `/` accessible et chargée < 2s LCP (Lighthouse)
- [ ] H1 = exactement la tagline validée (cf. `copies/landing_copy.md`)
- [ ] 4 stats visibles dès l'ouverture (329, 1.30, 7 ans, 95 %)
- [ ] CTA "Essayer gratuitement" → `/signup`
- [ ] CTA "Voir la méthodologie" → `/methodologie` (créera DG-XXX V2)
- [ ] LiveLectureExample utilise LectureView (DG-101) en mode FREE → la section "Détail technique" est verrouillée (overlay)
- [ ] Footer compliance permanent visible
- [ ] OpenGraph meta présent (image, title, description)
- [ ] Mobile-first responsive (DG-103 contribution) : tagline lisible, stats 2×2 sur mobile, CTAs largeur 100%
- [ ] Lighthouse mobile ≥ 90
- [ ] Aucun vocabulaire interdit (audit `legal_templates/disclaimer_compliance.md`)
- [ ] Event analytics `pageview` capturé (via DG-160)

## Tests requis

```typescript
// frontend/__tests__/landing.spec.ts
test('landing hero shows 4 key stats', () => {
  const { getByText } = render(<LandingPage />);
  expect(getByText('329')).toBeInTheDocument();
  expect(getByText('1.30')).toBeInTheDocument();
  expect(getByText('7 ans')).toBeInTheDocument();
  expect(getByText('95 %')).toBeInTheDocument();
});

test('landing tagline contains "traduite"', () => {
  const { container } = render(<LandingPage />);
  expect(container.textContent).toMatch(/traduite/i);
});

test('landing FREE demo shows locked expert section', () => {
  const { getByText } = render(<LiveLectureExample />);
  expect(getByText(/🔒 STRATEGIST/)).toBeInTheDocument();
});

test('landing has no forbidden vocabulary', () => {
  const { container } = render(<LandingPage />);
  const text = container.textContent ?? '';
  for (const banned of ['signal de trading', 'achetez', 'vendez', 'garanti', 'recommandation']) {
    expect(text.toLowerCase()).not.toContain(banned);
  }
});
```

## Risques / pièges

- ❌ **Mettre les 4 stats en marketing creux** ("✨ Best AI for trading ✨") : c'est précisément ce qu'on évite. Les 4 stats sont factuelles, honnêtes, vérifiables.
- ❌ **Hero card en image statique** : on perd la démonstration que l'archi progressive marche. Utiliser le vrai composant LectureView avec demo signal.
- ❌ **Forbidden vocabulary qui passe** : "signal" peut glisser dans une copy révisée. Test automatisé obligatoire.
- ❌ **CTA "Acheter maintenant" / "Commencer à trader"** : double interdit (vocabulaire prescriptif + posture). Utiliser "Essayer gratuitement" / "Découvrir".
- ❌ **Tagline trop longue mobile** : doit tenir en 2-3 lignes sur 375px. Vérifier au DevTools.
- ❌ **Stat "PF 1.30"** sans IC à côté : c'est notre différenciateur, l'IC DOIT toujours accompagner.
