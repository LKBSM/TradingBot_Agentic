import * as React from 'react';
import { render as rtlRender } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { NextIntlClientProvider } from 'next-intl';
import { TooltipProvider } from '@/components/ui/tooltip';
import messages from '@/messages/fr.json';
import { MarketReadingCard } from '@/components/market-reading/MarketReadingCard';
import { SAMPLE_READING_XAU_M15_SPARSE } from '@/lib/ds-samples';

/**
 * DS-1 §3 — absence must render as ABSENCE. A reading with no current break, no
 * retest and no events must simply not render those elements — never a dash,
 * never an "N/A" / "non disponible" generic placeholder.
 */
function render(ui: React.ReactElement) {
  return rtlRender(
    <NextIntlClientProvider locale="fr" messages={messages}>
      <TooltipProvider>{ui}</TooltipProvider>
    </NextIntlClientProvider>,
  );
}

describe('DS-1 — absence renders as absence', () => {
  it('the sparse reading card shows no generic-absence placeholder', () => {
    const { container } = render(<MarketReadingCard reading={SAMPLE_READING_XAU_M15_SPARSE} />);
    const text = container.textContent ?? '';

    // real descriptive content is present (not an empty shell)
    expect(text.length).toBeGreaterThan(20);

    // no generic "field missing" fallbacks
    expect(text).not.toMatch(/non disponible|indisponible|N\/A/i);
    // no lone dash used as a value ("label : —" / "label : -")
    expect(text).not.toMatch(/:\s*[—–-]\s*(?:$|\n)/);
    expect(text).not.toContain('undefined');
    expect(text).not.toContain('null');
  });
});
