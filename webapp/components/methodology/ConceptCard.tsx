import { useTranslations } from 'next-intl';
import { Card, CardContent } from '@/components/ui/card';
import type { MethodologyConcept } from '@/lib/methodology/content';
import { conceptShort, conceptTerm } from '@/lib/methodology/content';

/**
 * Carte de concept SMC pour /methodology (Chantier 5.D).
 *
 * L'`id` sert d'ancre — il matche l'ancre du glossaire (#order-block, #fvg…),
 * donc les tooltips ⓘ « En savoir plus → » atterrissent directement sur la
 * bonne carte. Terme + définition courte viennent du glossaire (source unique) ;
 * `detection` explique comment le moteur repère le concept.
 */
export function ConceptCard({ concept }: { concept: MethodologyConcept }) {
  const t = useTranslations('methodology');

  return (
    <Card id={concept.id} className="scroll-mt-24 border-border/60">
      <CardContent className="space-y-2 p-5">
        <h3 className="text-base font-semibold tracking-tight">
          {conceptTerm(concept.glossaryKey)}
        </h3>
        <p className="text-sm text-muted-foreground">
          {conceptShort(concept.glossaryKey)}
        </p>
        <div className="pt-1">
          <p className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground/80">
            {t('concepts.detectionLabel')}
          </p>
          <p className="mt-1 text-sm leading-relaxed text-foreground">
            {t(`concepts.detection.${concept.id}`)}
          </p>
        </div>
      </CardContent>
    </Card>
  );
}
