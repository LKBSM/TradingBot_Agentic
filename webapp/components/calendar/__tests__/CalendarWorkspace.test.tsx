import { render, fireEvent } from '@/components/test-utils';
import { describe, expect, it } from 'vitest';
import fr from '@/messages/fr.json';
import { CalendarWorkspace } from '../CalendarWorkspace';
import type { CalendarEvent, CalendarResponse } from '@/types/calendar';

const NOW = new Date('2026-07-28T06:00:00Z');

function ev(p: Partial<CalendarEvent> & Pick<CalendarEvent, 'event_id' | 'event' | 'scheduled_at'>): CalendarEvent {
  return {
    source: 'forexfactory',
    series_code: null,
    license_label: null,
    currency: 'USD',
    impact: 'high',
    organism: null,
    source_timezone: 'America/New_York',
    markets: ['XAUUSD', 'EURUSD'],
    value_unit: null,
    actual: null,
    forecast: null,
    previous: null,
    revised: false,
    previous_before_revision: null,
    ...p,
  };
}

const DATA: CalendarResponse = {
  window_start: '2026-07-25T06:00:00Z',
  window_end: '2026-08-04T06:00:00Z',
  generated_at: NOW.toISOString(),
  coverage: { source: 'forexfactory', feed_start: null, feed_end: null, partial: false },
  events: [
    ev({ event_id: 'forexfactory:e1', event: 'CPI y/y', scheduled_at: '2026-07-28T12:30:00Z', impact: 'high' }),
    ev({ event_id: 'forexfactory:e2', event: 'ECB rate', scheduled_at: '2026-07-28T14:00:00Z', impact: 'medium', currency: 'EUR', markets: ['EURUSD'], organism: 'Eurostat' }),
    ev({ event_id: 'forexfactory:e3', event: 'ADP jobs', scheduled_at: '2026-07-29T12:00:00Z', impact: 'low', revised: true }),
    ev({ event_id: 'forexfactory:e4', event: 'Retail sales', scheduled_at: '2026-07-27T12:00:00Z', impact: 'high' }),
  ],
};

function renderCal() {
  return render(<CalendarWorkspace locale="fr" data={DATA} now={NOW} />);
}

const rows = (c: HTMLElement) => Array.from(c.querySelectorAll('.cal-row'));
const evNames = (c: HTMLElement) =>
  Array.from(c.querySelectorAll('.cal-ev')).map((e) => e.textContent);
function chip(c: HTMLElement, label: string): HTMLElement {
  const el = Array.from(c.querySelectorAll('.fchip')).find(
    (b) => (b.textContent ?? '').trim() === label,
  );
  if (!el) throw new Error(`chip not found: ${label}`);
  return el as HTMLElement;
}

describe('NW-1 CalendarWorkspace list', () => {
  it('shows only upcoming events by default, grouped chronologically', () => {
    const { container } = renderCal();
    // e1, e2 (today) + e3 (tomorrow); the past e4 is hidden.
    expect(rows(container)).toHaveLength(3);
    expect(evNames(container)).toEqual(['CPI y/y', 'ECB rate', 'ADP jobs']);
  });

  it('the amplitude slot shows the « mesures à venir » placeholder (NW-2 fills it)', () => {
    const { container } = renderCal();
    const vals = Array.from(container.querySelectorAll('.cal-ampl .v')).map(
      (e) => e.textContent,
    );
    expect(vals.length).toBe(3);
    for (const v of vals) expect(v).toBe(fr.calendar.amplitude.pending);
  });

  it('exposes NO amplitude/magnitude sort control — chronological order only', () => {
    const { container } = renderCal();
    expect(
      container.querySelectorAll('select, [role="combobox"], [role="listbox"]'),
    ).toHaveLength(0);
    // Order is by time; toggling nothing, the first row is the earliest today.
    expect(evNames(container)[0]).toBe('CPI y/y');
  });

  it('zero impact chip selected → empty list + explicit message, never a fallback', () => {
    const { container } = renderCal();
    fireEvent.click(chip(container, fr.calendar.impact.high));
    fireEvent.click(chip(container, fr.calendar.impact.medium));
    fireEvent.click(chip(container, fr.calendar.impact.low));
    expect(rows(container)).toHaveLength(0);
    expect(container.querySelector('.cal-empty')?.textContent).toContain(
      fr.calendar.empty.noSelection,
    );
  });

  it('a missing organism renders as visibly absent; a present one is shown', () => {
    const { container } = renderCal();
    const text = container.textContent ?? '';
    expect(text).toContain(fr.calendar.provenance.organismMissing); // e1/e3 (null)
    expect(text).toContain('Eurostat'); // e2
  });

  it('a revised value is surfaced as revised', () => {
    const { container } = renderCal();
    const adp = rows(container).find((r) => (r.textContent ?? '').includes('ADP jobs'));
    expect(adp?.textContent).toContain(fr.calendar.revisedBadge);
  });

  it('« Publications passées » switches to past releases', () => {
    const { container } = renderCal();
    fireEvent.click(container.querySelector('.cal-past')!);
    const shown = rows(container);
    expect(shown).toHaveLength(1);
    expect(shown[0]!.textContent).toContain('Retail sales');
  });

  it('the attached markets are named (affecte Or, EUR/USD)', () => {
    const { container } = renderCal();
    const cpi = rows(container).find((r) => (r.textContent ?? '').includes('CPI y/y'));
    expect(cpi?.textContent).toContain('affecte Or, EUR/USD');
  });

  it('each row exposes an « En savoir plus » link to that event detail page', () => {
    const { container } = renderCal();
    const links = Array.from(container.querySelectorAll('a.cal-more'));
    expect(links).toHaveLength(3);
    for (const a of links) {
      expect(a.getAttribute('href')).toMatch(/\/actualites\/forexfactory%3A/);
      expect((a.textContent ?? '')).toContain(fr.calendar.seeMore);
    }
  });

  it('renders no raw i18n keys', () => {
    const { container } = renderCal();
    const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT);
    let n: Node | null;
    while ((n = walker.nextNode())) {
      const t = (n.textContent ?? '').trim();
      if (!t || t.includes('/') || t.includes(':')) continue;
      expect(/^[a-z][a-zA-Z0-9]*(\.[a-zA-Z][a-zA-Z0-9]*)+$/.test(t), `raw key: ${t}`).toBe(false);
    }
  });
});
