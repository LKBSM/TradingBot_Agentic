import { useTranslations } from 'next-intl';
import { FileText } from 'lucide-react';
import {
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion';
import { Badge } from '@/components/ui/badge';
import { humaniseTag } from '@/lib/market-reading/tag-labels';
import type { MarketReadingConditions } from '@/types/market-reading';

/**
 * Section "Lecture narrée" — the present-tense narration composed 100 % by the
 * deterministic engine template from the FACTS (trend, multi-TF alignment,
 * near-price OB/FVG zones, recent BOS/CHOCH, volatility), and anchored so every
 * cited level is a real engine output. No LLM is involved, so the source line
 * states plainly that the reading is composed by the engine. The text is
 * descriptive only — never a forecast or advice.
 */
export function ConditionsSection({
  conditions,
}: {
  conditions: MarketReadingConditions;
}) {
  const t = useTranslations('reading');
  const tagLabel = (tag: string): string =>
    t.has(`tags.${tag}`) ? t(`tags.${tag}`) : humaniseTag(tag);
  const sourceLabel = t('conditions.source');

  // The narration is a short paragraph; render any sentence-level line breaks the
  // engine produced as separate lines for readability (it never adds markup).
  const paragraphs = conditions.description
    .split(/\n+/)
    .map((p) => p.trim())
    .filter(Boolean);

  return (
    <AccordionItem value="conditions">
      <AccordionTrigger className="text-left text-sm">
        <span className="flex items-center gap-2">
          <FileText className="h-4 w-4 text-muted-foreground" aria-hidden />
          <span>{t('conditions.title')}</span>
        </span>
      </AccordionTrigger>
      <AccordionContent>
        <div className="space-y-4">
          <div className="space-y-2 text-sm leading-relaxed text-foreground">
            {paragraphs.length > 0 ? (
              paragraphs.map((p, i) => <p key={i}>{p}</p>)
            ) : (
              <p>{conditions.description}</p>
            )}
          </div>

          {conditions.tags.length > 0 && (
            <div className="flex flex-wrap gap-1.5">
              {conditions.tags.map((tag) => (
                <Badge key={tag} variant="secondary" className="text-[10px]">
                  {tagLabel(tag)}
                </Badge>
              ))}
            </div>
          )}

          <p className="text-[11px] uppercase tracking-wide text-muted-foreground/70">
            {sourceLabel}
          </p>
        </div>
      </AccordionContent>
    </AccordionItem>
  );
}
