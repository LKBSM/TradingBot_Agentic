import {
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion';
import {
  formatPrice,
  formatRetestState,
  formatZone,
} from '@/lib/insight-formatters';
import type { InsightSignalV2 } from '@/types/insight';

/**
 * Section "Structure" — translates the raw Smart Money Concept readout into
 * plain French. Every line is descriptive (a market fact), never prescriptive.
 */
export function StructureSection({ signal }: { signal: InsightSignalV2 }) {
  const s = signal.structure_readout;
  const inst = signal.instrument;
  const fvg = formatZone(s.fvg_zone, inst);
  const ob = formatZone(s.ob_zone, inst);

  return (
    <AccordionItem value="structure">
      <AccordionTrigger className="text-left text-sm">
        <span className="flex items-center gap-2">
          <span aria-hidden>📐</span>
          <span>Structure de marché</span>
        </span>
      </AccordionTrigger>
      <AccordionContent>
        <dl className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          <Row
            label="Cassure de structure"
            value={
              s.bos_level !== null
                ? `${formatPrice(s.bos_level, inst)}${s.choch_present ? ' · CHOCH précédent' : ''}${s.bos_event_age_bars !== null ? ` · il y a ${s.bos_event_age_bars} bougie${s.bos_event_age_bars > 1 ? 's' : ''}` : ''}`
                : 'aucune cassure récente'
            }
          />
          <Row
            label="Zone de déséquilibre (FVG)"
            value={
              fvg
                ? `${fvg}${s.fvg_size_atr !== null ? ` · ${s.fvg_size_atr.toFixed(2).replace('.', ',')} × ATR` : ''}`
                : 'aucune zone détectée'
            }
          />
          <Row
            label="Order Block"
            value={
              ob
                ? `${ob}${s.ob_strength !== null ? ` · intensité ${(s.ob_strength * 100).toFixed(0)} %` : ''}`
                : 'aucun bloc significatif'
            }
          />
          <Row label="État du retest" value={formatRetestState(s.retest_state)} />
          <Row
            label="Invalidation structurelle"
            value={
              s.structural_invalidation !== null
                ? `sous ${formatPrice(s.structural_invalidation, inst)}`
                : 'non définie'
            }
            className="sm:col-span-2"
          />
        </dl>
        <p className="mt-4 text-xs italic text-muted-foreground">
          Lecture descriptive — l'invalidation est un fait de marché, pas un
          stop-loss imposé.
        </p>
      </AccordionContent>
    </AccordionItem>
  );
}

function Row({
  label,
  value,
  className,
}: {
  label: string;
  value: string;
  className?: string;
}) {
  return (
    <div className={className}>
      <dt className="text-xs uppercase tracking-wide text-muted-foreground">{label}</dt>
      <dd className="mt-1 text-sm font-medium text-foreground">{value}</dd>
    </div>
  );
}
