'use client';

import { Accordion } from '@/components/ui/accordion';
import { StructureSection } from './sections/StructureSection';
import { RegimeSection } from './sections/RegimeSection';
import { EventsSection } from './sections/EventsSection';
import { ConditionsSection } from './sections/ConditionsSection';
import type { MarketReading } from '@/types/market-reading';

export type MarketReadingSectionKey =
  | 'structure'
  | 'regime'
  | 'events'
  | 'conditions';

interface MarketReadingSectionsProps {
  reading: MarketReading;
  /**
   * Section keys to open on mount. Omit to keep everything collapsed (the
   * default — discoverable, not noisy).
   */
  defaultOpen?: ReadonlyArray<MarketReadingSectionKey>;
}

/**
 * Layer 2 — collapsible sections under the reading hero. Each section consumes
 * one block of the MarketReading directly (no mapper). The hero already covers
 * `header` and the phase summary of `regime`; these four sections expose the
 * full detail of structure / regime / events / conditions.
 */
export function MarketReadingSections({
  reading,
  defaultOpen = [],
}: MarketReadingSectionsProps) {
  return (
    <Accordion type="multiple" defaultValue={[...defaultOpen]} className="w-full">
      <StructureSection
        structure={reading.structure}
        instrument={reading.header.instrument}
        closePrice={reading.header.close_price}
      />
      <RegimeSection
        regime={reading.regime}
        structure={reading.structure}
        header={reading.header}
      />
      <EventsSection events={reading.events} />
      <ConditionsSection conditions={reading.conditions} />
    </Accordion>
  );
}
