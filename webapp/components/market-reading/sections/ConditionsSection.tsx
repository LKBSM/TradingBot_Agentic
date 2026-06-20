import {
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion';
import { Badge } from '@/components/ui/badge';
import { formatTag } from '@/lib/market-reading/tag-labels';
import type { MarketReadingConditions } from '@/types/market-reading';

/**
 * Section "Lecture narrée" — the present-tense narration synthesised by the
 * engine FACTS (trend, multi-TF alignment, near-price OB/FVG zones, recent
 * BOS/CHOCH, volatility), validated against those facts server-side.
 * `description_source` is surfaced discreetly so the reader knows whether the
 * narration came from the LLM (anchored + level-validated) or the deterministic
 * template fallback. The text is descriptive only — never a forecast or advice.
 */
export function ConditionsSection({
  conditions,
}: {
  conditions: MarketReadingConditions;
}) {
  const sourceLabel =
    conditions.description_source === 'haiku_generated'
      ? 'Narration générée'
      : 'Lecture modèle (repli)';

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
          <span aria-hidden>🧭</span>
          <span>Lecture narrée</span>
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
                  {formatTag(tag)}
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
