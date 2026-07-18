import type { Metadata } from 'next';
import Link from 'next/link';
import { useTranslations } from 'next-intl';
import { getTranslations } from 'next-intl/server';
import { ShieldCheck, ArrowLeft } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { MethodologySection } from '@/components/methodology/MethodologySection';
import { ConceptCard } from '@/components/methodology/ConceptCard';
import { ScoreFormula } from '@/components/methodology/ScoreFormula';
import {
  DATA_SOURCE,
  NEVER_DO,
  SCORE_FORMULAS,
  SMC_CONCEPTS,
} from '@/lib/methodology/content';

export async function generateMetadata(): Promise<Metadata> {
  const t = await getTranslations('methodology');
  return {
    title: t('meta.title'),
    description: t('meta.description'),
  };
}

const TOC = [
  { href: '#engagement', key: 'engagement' },
  { href: '#concepts', key: 'concepts' },
  { href: '#scores', key: 'scores' },
  { href: '#donnees', key: 'donnees' },
  { href: '#limites', key: 'limites' },
  { href: '#attributions', key: 'attributions' },
] as const;

/**
 * Page /methodology (Chantier 5.D) — documentation algorithmique transparente.
 *
 * Descriptive et technique, JAMAIS promotionnelle (niveau 1.5 strict). Elle
 * répond à la curiosité légitime « comment l'algorithme calcule ça ? » sans
 * jamais prétendre prédire. Contenu data-driven (lib/methodology/content.ts),
 * termes partagés avec les tooltips ⓘ via le glossaire central.
 */
export default function MethodologyPage() {
  const t = useTranslations('methodology');

  return (
    <div className="container-prose py-12 sm:py-16">
      <header className="space-y-4">
        <Link
          href="/"
          className="inline-flex items-center gap-1.5 text-sm text-muted-foreground underline-offset-4 hover:text-foreground hover:underline"
        >
          <ArrowLeft className="h-3.5 w-3.5" aria-hidden />
          {t('header.back')}
        </Link>
        <Badge
          variant="outline"
          className="text-[11px] uppercase tracking-wider"
        >
          <ShieldCheck className="mr-1 h-3 w-3" aria-hidden />
          {t('header.badge')}
        </Badge>
        <h1 className="text-balance text-2xl font-semibold tracking-tight sm:text-3xl lg:text-4xl">
          {t('header.title')}
        </h1>
        <p className="max-w-2xl text-pretty text-muted-foreground">
          {t.rich('header.intro', {
            em: (chunks) => <em>{chunks}</em>,
          })}
        </p>
        <nav
          aria-label={t('header.tocLabel')}
          className="flex flex-wrap gap-2 pt-2"
        >
          {TOC.map((item) => (
            <a
              key={item.href}
              href={item.href}
              className="rounded-full border border-border/60 px-3 py-1 text-xs text-muted-foreground transition-colors hover:border-foreground/40 hover:text-foreground"
            >
              {t(`toc.${item.key}`)}
            </a>
          ))}
        </nav>
      </header>

      <MethodologySection
        id="engagement"
        title={t('engagement.title')}
        intro={t('engagement.intro')}
      >
        <figure>
          <blockquote className="border-l-2 border-sentinel-warn pl-5 text-balance text-base italic text-foreground">
            {t('engagement.quote')}
          </blockquote>
          <figcaption className="mt-3 text-xs text-muted-foreground">
            {t('engagement.caption')}
          </figcaption>
        </figure>
      </MethodologySection>

      <MethodologySection
        id="concepts"
        title={t('concepts.title')}
        intro={t('concepts.intro')}
      >
        <div className="grid gap-4 sm:grid-cols-2">
          {SMC_CONCEPTS.map((concept) => (
            <ConceptCard key={concept.id} concept={concept} />
          ))}
        </div>
      </MethodologySection>

      <MethodologySection
        id="scores"
        title={t('scores.title')}
        intro={t('scores.intro')}
      >
        <div className="space-y-8">
          {SCORE_FORMULAS.map((formula) => (
            <ScoreFormula key={formula.id} formula={formula} />
          ))}
        </div>
      </MethodologySection>

      <MethodologySection
        id="donnees"
        title={t('donnees.title')}
        intro={t('donnees.detail')}
      >
        <dl className="grid gap-4 sm:grid-cols-3">
          <div>
            <dt className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground/80">
              {t('donnees.providerLabel')}
            </dt>
            <dd className="mt-1 text-sm text-foreground">
              {DATA_SOURCE.provider}
            </dd>
          </div>
          <div>
            <dt className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground/80">
              {t('donnees.coverageLabel')}
            </dt>
            <dd className="mt-1 text-sm text-foreground">
              {t('donnees.coverage')}
            </dd>
          </div>
          <div>
            <dt className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground/80">
              {t('donnees.refreshLabel')}
            </dt>
            <dd className="mt-1 text-sm text-foreground">
              {t('donnees.refresh')}
            </dd>
          </div>
        </dl>
      </MethodologySection>

      <MethodologySection
        id="limites"
        title={t('limites.title')}
        intro={t('limites.intro')}
      >
        <ul className="space-y-2 text-sm text-muted-foreground">
          {NEVER_DO.map((_, i) => (
            <li key={i} className="flex items-start gap-2">
              <span
                className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-sentinel-bear"
                aria-hidden
              />
              <span>{t(`limites.items.${i}`)}</span>
            </li>
          ))}
        </ul>
      </MethodologySection>

      <MethodologySection
        id="attributions"
        title={t('attributions.title')}
        intro={t('attributions.intro')}
      >
        <div className="space-y-2 text-sm text-muted-foreground">
          <p>
            {t.rich('attributions.license', {
              name: (chunks) => (
                <span className="text-foreground">{chunks}</span>
              ),
              link: (chunks) => (
                <a
                  href="https://www.apache.org/licenses/LICENSE-2.0"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="underline underline-offset-4 hover:text-foreground"
                >
                  {chunks}
                </a>
              ),
            })}
          </p>
          <p className="text-xs">
            {t.rich('attributions.note', {
              em: (chunks) => <em>{chunks}</em>,
            })}
          </p>
        </div>
      </MethodologySection>
    </div>
  );
}
