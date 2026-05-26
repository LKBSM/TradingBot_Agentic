'use client';

import { Accordion } from '@/components/ui/accordion';
import { StructureSection } from './sections/StructureSection';
import { RegimeSection } from './sections/RegimeSection';
import { VolatilitySection } from './sections/VolatilitySection';
import { EventSection } from './sections/EventSection';
import { HistorySection } from './sections/HistorySection';
import type { InsightSignalV2 } from '@/types/insight';

interface InsightSectionsProps {
  signal: InsightSignalV2;
  /**
   * Section keys collapsed by default = all closed. Pass an array of section
   * IDs (`structure`, `regime`, `volatility`, `events`, `history`) to open
   * them on mount, or omit to keep the F3 default (everything closed).
   */
  defaultOpen?: ReadonlyArray<
    'structure' | 'regime' | 'volatility' | 'events' | 'history'
  >;
}

/**
 * Layer 2 of the architecture progressive uniforme — five collapsible
 * sections under the hero. Each section translates a portion of the
 * InsightSignalV2 payload into plain French. Default state: everything
 * collapsed (user must opt in by clicking — discoverable, not noisy).
 */
export function InsightSections({
  signal,
  defaultOpen = [],
}: InsightSectionsProps) {
  return (
    <Accordion
      type="multiple"
      defaultValue={[...defaultOpen]}
      className="w-full"
    >
      <StructureSection signal={signal} />
      <RegimeSection signal={signal} />
      <VolatilitySection signal={signal} />
      <EventSection signal={signal} />
      <HistorySection signal={signal} />
    </Accordion>
  );
}
