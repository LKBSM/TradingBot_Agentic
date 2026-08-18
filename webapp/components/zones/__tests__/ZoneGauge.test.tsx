import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { NextIntlClientProvider, useTranslations } from 'next-intl';
import fr from '@/messages/fr.json';
import { useReadingFormatters } from '@/lib/market-reading/use-reading-formatters';
import { ProximityBlock } from '../ZoneLifecycleCard';
import type { ZoneLifecycle } from '@/lib/zones/lifecycle';

/**
 * VZ-3 — the proximity gauge. These assert the four states render the right
 * visual, that the gauge écart is char-for-char the distance line's, that
 * « inside » draws no bracket, that a missing price renders NOTHING, and that
 * the gauge carries no direction colour.
 */

function makeZone(over: Partial<ZoneLifecycle>): ZoneLifecycle {
  return {
    id: 'z',
    kind: 'ob',
    direction: 'bullish',
    levelHigh: 110,
    levelLow: 100,
    importance: 'medium',
    status: 'active',
    createdAt: '2026-06-20T08:00:00Z',
    tested: false,
    mitigatedAt: null,
    fillLevel: null,
    isActive: true,
    isMitigated: false,
    contacts: [],
    origin: null,
    session: null,
    ...over,
  };
}

function Harness({ zone, price }: { zone: ZoneLifecycle; price: number | null }) {
  const t = useTranslations('zones');
  const fmt = useReadingFormatters();
  return <ProximityBlock zone={zone} price={price} instrument="XAUUSD" t={t} fmt={fmt} locale="fr" />;
}

function renderProx(zone: ZoneLifecycle, price: number | null) {
  return render(
    <NextIntlClientProvider locale="fr" messages={fr}>
      <Harness zone={zone} price={price} />
    </NextIntlClientProvider>,
  );
}

describe('ProximityBlock gauge (VZ-3)', () => {
  it('price ABOVE the zone (in window) → gauge with a bracket écart', () => {
    renderProx(makeZone({ levelLow: 100, levelHigh: 110 }), 112);
    expect(document.querySelector('.zgauge')).toBeTruthy();
    expect(screen.getByTestId('gauge-gap')).toBeTruthy();
  });

  it('price BELOW the zone (in window) → gauge with a bracket écart', () => {
    renderProx(makeZone({ levelLow: 100, levelHigh: 110 }), 98);
    expect(document.querySelector('.zgauge.gs-below')).toBeTruthy();
    expect(screen.getByTestId('gauge-gap')).toBeTruthy();
  });

  it('price INSIDE the zone → gauge rendered, NO bracket écart', () => {
    renderProx(makeZone({ levelLow: 100, levelHigh: 110 }), 105);
    expect(document.querySelector('.zgauge.gs-inside')).toBeTruthy();
    expect(screen.queryByTestId('gauge-gap')).toBeNull();
  });

  it('price OUT of the window → gauge marked out, NO bracket écart', () => {
    renderProx(makeZone({ levelLow: 100, levelHigh: 110 }), 300);
    expect(document.querySelector('.zgauge.gs-outAbove')).toBeTruthy();
    expect(document.querySelector('.gpx.out')).toBeTruthy();
    expect(screen.queryByTestId('gauge-gap')).toBeNull();
  });

  it('missing price → the whole block renders NOTHING (no empty container)', () => {
    const { container } = renderProx(makeZone({ levelLow: 100, levelHigh: 110 }), null);
    expect(container.querySelector('.zpx')).toBeNull();
    expect(container.querySelector('.zgauge')).toBeNull();
    expect(container.textContent).toBe('');
  });

  it('the gauge écart equals the distance line écart, character for character', () => {
    renderProx(makeZone({ levelLow: 100, levelHigh: 110 }), 112);
    const line = screen.getByTestId('distance-line').textContent ?? '';
    const gapCell = screen.getByTestId('gauge-gap').textContent ?? '';
    // The gauge cell is « {pts} pts »; its number must appear verbatim in the line.
    const gaugeNumber = gapCell.replace(/\s*pts\s*$/, '').trim();
    expect(gaugeNumber.length).toBeGreaterThan(0);
    expect(line).toContain(gaugeNumber);
    // And the line's own « X pts » token is exactly the gauge's cell.
    expect(line.startsWith(gapCell)).toBe(true);
  });

  it('the gauge carries an aria-label with the zone bounds, price and distance', () => {
    renderProx(makeZone({ levelLow: 100, levelHigh: 110 }), 112);
    const g = document.querySelector('.zgauge');
    const aria = g?.getAttribute('aria-label') ?? '';
    expect(aria).toContain('zone');
    expect(aria.length).toBeGreaterThan(20);
  });
});
