import {
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion';
import { cn } from '@/lib/utils';
import { InfoTooltip } from '@/components/ui/InfoTooltip';
import type { GlossaryKey } from '@/lib/glossary';
import {
  formatBand,
  formatDirection,
  formatFvgStatus,
  formatObImportance,
  formatObStatus,
  formatPrice,
  formatRetestType,
  formatValidationStatus,
} from '@/lib/market-reading/formatters';
import type { MarketReadingStructure } from '@/types/market-reading';

/**
 * Section "Structure" — renders the Smart Money Concept block factually:
 * BOS / CHOCH, Order Blocks, Fair Value Gaps, retest in progress. Every line is
 * descriptive (a market fact), never prescriptive.
 */
export function StructureSection({
  structure,
  instrument,
}: {
  structure: MarketReadingStructure;
  instrument: string;
}) {
  const { bos, choch, order_blocks, fair_value_gaps, retest_in_progress } =
    structure;

  // Surfacing coherence (founder eval 2026-06-08): the engine emits `bos` only
  // on a FRESH break at the last close (by design — see market_reading_mappers
  // F6), while a retest is armed BARS AFTER that break, with `bos` already null.
  // Without this, the BOS row said "aucune cassure récente" while the retest row
  // said "retest de cassure (BOS)" — a logical contradiction. When the live
  // retest references a prior break, we state that instead of denying it. This
  // is a copy/surfacing fix only — no detection threshold is touched.
  const bosUnderRetest =
    !bos && retest_in_progress?.type === 'bos_retest';
  const chochUnderRetest =
    !choch && retest_in_progress?.type === 'choch_retest';

  const hasAnything =
    bos ||
    choch ||
    order_blocks.length > 0 ||
    fair_value_gaps.length > 0 ||
    retest_in_progress;

  return (
    <AccordionItem value="structure">
      <AccordionTrigger className="text-left text-sm">
        <span className="flex items-center gap-2">
          <span aria-hidden>📐</span>
          <span>Structure de marché</span>
        </span>
      </AccordionTrigger>
      <AccordionContent>
        {!hasAnything ? (
          <p className="text-sm text-muted-foreground">
            Aucun élément structurel notable sur la dernière bougie.
          </p>
        ) : (
          <dl className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            <Row
              label="Cassure de structure (BOS)"
              termKey="bos"
              value={
                bos
                  ? `${formatPrice(bos.level, instrument)} · ${formatDirection(bos.direction)} · ${formatValidationStatus(bos.validation_status)}`
                  : bosUnderRetest
                    ? `cassure antérieure en cours de retest (${formatPrice(retest_in_progress!.level, instrument)})`
                    : 'aucune cassure récente'
              }
            />
            <Row
              label="Changement de caractère (CHOCH)"
              termKey="choch"
              value={
                choch
                  ? `${formatPrice(choch.level, instrument)} · ${formatDirection(choch.direction)} · ${formatValidationStatus(choch.validation_status)}`
                  : chochUnderRetest
                    ? `changement antérieur en cours de retest (${formatPrice(retest_in_progress!.level, instrument)})`
                    : 'aucun changement récent'
              }
            />
            <Row
              label="Order Blocks"
              termKey="order_block"
              value={
                order_blocks.length > 0
                  ? order_blocks
                      .map(
                        (ob) =>
                          `${formatBand(ob.level_low, ob.level_high, instrument)} · importance ${formatObImportance(ob.importance)} · ${formatObStatus(ob.status)}`,
                      )
                      .join(' | ')
                  : 'aucun bloc significatif'
              }
              className="sm:col-span-2"
            />
            <Row
              label="Fair Value Gaps"
              termKey="fvg"
              value={
                fair_value_gaps.length > 0
                  ? fair_value_gaps
                      .map(
                        (fvg) =>
                          `${formatBand(fvg.level_low, fvg.level_high, instrument)} · ${formatFvgStatus(fvg.status)}`,
                      )
                      .join(' | ')
                  : 'aucune zone détectée'
              }
              className="sm:col-span-2"
            />
            <Row
              label="Retest en cours"
              termKey="retest"
              value={
                retest_in_progress
                  ? `${formatPrice(retest_in_progress.level, instrument)} · ${formatRetestType(retest_in_progress.type)}`
                  : 'aucun retest en cours'
              }
              className="sm:col-span-2"
            />
          </dl>
        )}
        <p className="mt-4 text-xs italic text-muted-foreground">
          Lecture descriptive — une invalidation structurelle est un fait de
          marché, pas un stop-loss imposé.
        </p>
      </AccordionContent>
    </AccordionItem>
  );
}

function Row({
  label,
  value,
  termKey,
  className,
}: {
  label: string;
  value: string;
  /** When set, the label becomes a vulgarisation tooltip (ⓘ + /methodology link). */
  termKey?: GlossaryKey;
  className?: string;
}) {
  return (
    <div className={cn(className)}>
      <dt className="text-xs uppercase tracking-wide text-muted-foreground">
        {termKey ? <InfoTooltip termKey={termKey}>{label}</InfoTooltip> : label}
      </dt>
      <dd className="mt-1 text-sm font-medium text-foreground">{value}</dd>
    </div>
  );
}
