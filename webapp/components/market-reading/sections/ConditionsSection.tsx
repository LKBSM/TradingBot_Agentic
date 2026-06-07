import {
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion';
import { Badge } from '@/components/ui/badge';
import type { MarketReadingConditions } from '@/types/market-reading';

/**
 * Section "Conditions" — the plain-language synthesis plus the descriptive
 * tags. `description_source` is surfaced discreetly so the reader knows whether
 * the sentence was LLM-generated or a deterministic template fallback.
 */
export function ConditionsSection({
  conditions,
}: {
  conditions: MarketReadingConditions;
}) {
  const sourceLabel =
    conditions.description_source === 'haiku_generated'
      ? 'Synthèse générée'
      : 'Synthèse modèle (fallback)';

  return (
    <AccordionItem value="conditions">
      <AccordionTrigger className="text-left text-sm">
        <span className="flex items-center gap-2">
          <span aria-hidden>🧭</span>
          <span>Synthèse des conditions</span>
        </span>
      </AccordionTrigger>
      <AccordionContent>
        <div className="space-y-4">
          <p className="text-sm leading-relaxed text-foreground">
            {conditions.description}
          </p>

          {conditions.tags.length > 0 && (
            <div className="flex flex-wrap gap-1.5">
              {conditions.tags.map((tag) => (
                <Badge key={tag} variant="secondary" className="text-[10px]">
                  {tag}
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
