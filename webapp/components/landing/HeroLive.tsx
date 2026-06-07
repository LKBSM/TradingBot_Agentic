import { ShieldCheck, Sparkles } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { HeroChatPreview } from './HeroChatPreview';
import { HeroMarketReading } from './HeroMarketReading';
import { getHeroLandingSample } from '@/lib/market-reading/landing-samples';

/**
 * Hero LIVE — la lecture XAU/M15 du jour est rendue en grand, animée pour
 * donner l'impression d'une lecture en cours. À droite (desktop) ou en
 * dessous (mobile), Sentinel se présente et propose 3 questions.
 *
 * Volontairement : pas de H1 marketing classique, pas de chiffre de
 * performance, pas de promesse. Le produit parle pour lui (lock 2 +
 * `decision_gate_review_v2.md` section "honest confidence").
 *
 * H1 ici = la phrase de positionnement *« Comprenez le marché — sans
 * qu'on vous dise quoi faire »* en sr-only pour les lecteurs d'écran +
 * crawlers SEO. Visuellement c'est la card qui domine, l'eyebrow et le
 * badge donnent juste le cadre.
 */
export function HeroLive() {
  const sample = getHeroLandingSample();

  return (
    <section
      aria-labelledby="hero-title"
      className="container-wide pb-12 pt-10 sm:pt-16"
    >
      {/* Eyebrow row — small badges only, no marketing headline above the
          fold (cf. brief Section 3 "Aucun titre marketing au-dessus du hero"). */}
      <div
        className="hero-stagger mb-6 flex flex-wrap items-center gap-2"
        style={{ animationDelay: '0ms' }}
      >
        <Badge
          variant="secondary"
          className="text-[11px] uppercase tracking-wider"
        >
          <Sparkles className="mr-1 h-3 w-3" aria-hidden />
          Lecture en direct · {new Date().toLocaleDateString('fr-FR', {
            day: 'numeric',
            month: 'long',
          })}
        </Badge>
        <Badge variant="outline" className="text-[11px] uppercase tracking-wider">
          <ShieldCheck className="mr-1 h-3 w-3" aria-hidden />
          Indicateur · pas un service de signaux
        </Badge>
      </div>

      {/* H1 SEO-only : indexable, parlé par les lecteurs d'écran, invisible
          visuellement parce que c'est le produit qui doit attirer l'œil. */}
      <h1 id="hero-title" className="sr-only">
        MIA Markets — Comprenez le marché sans qu&apos;on vous dise quoi faire.
      </h1>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)] lg:items-start lg:gap-8">
        <HeroMarketReading sample={sample} />
        <HeroChatPreview sample={sample} introDelayMs={1400} />
      </div>

      {/* Discreet CTA — pas en hero dominant comme demandé. */}
      <div
        className="hero-stagger mt-8 flex justify-end"
        style={{ animationDelay: '900ms' }}
      >
        <a
          href="#tarifs"
          className="text-xs font-medium text-muted-foreground underline-offset-4 transition-colors hover:text-foreground hover:underline focus-visible:underline focus-visible:outline-none"
        >
          Essayer gratuitement →
        </a>
      </div>
    </section>
  );
}
