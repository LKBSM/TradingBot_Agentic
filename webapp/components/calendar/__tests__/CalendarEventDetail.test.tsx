import { render } from '@/components/test-utils';
import { describe, expect, it } from 'vitest';
import fr from '@/messages/fr.json';
import { CalendarEventDetail } from '../CalendarEventDetail';
import type { CalendarEvent, CalendarResponse } from '@/types/calendar';

const NOW = new Date('2026-07-28T06:00:00Z');

function ev(p: Partial<CalendarEvent> & Pick<CalendarEvent, 'event_id' | 'event'>): CalendarEvent {
  return {
    source: 'forexfactory',
    series_code: null,
    license_label: null,
    currency: 'USD',
    impact: 'high',
    organism: null,
    scheduled_at: '2026-07-28T12:30:00Z',
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

const EVENT = ev({
  event_id: 'forexfactory:cpi',
  event: 'CPI y/y',
  forecast: 3.1,
  previous: 3.3,
});

const DATA: CalendarResponse = {
  window_start: '2026-06-28T06:00:00Z',
  window_end: '2026-08-27T06:00:00Z',
  generated_at: NOW.toISOString(),
  coverage: { source: 'forexfactory', feed_start: null, feed_end: null, partial: false },
  events: [EVENT],
};

function renderDetail(eventId: string, data: CalendarResponse | null = DATA) {
  return render(
    <CalendarEventDetail eventId={eventId} locale="fr" data={data} now={NOW} />,
  );
}

describe('NW-1b CalendarEventDetail', () => {
  it('shows the event title, attributed impact and attached markets', () => {
    const { container } = renderDetail('forexfactory:cpi');
    expect(container.querySelector('h1')?.textContent).toBe('CPI y/y');
    const text = container.textContent ?? '';
    expect(text).toContain(fr.calendar.impact.high);
    expect(text).toContain('affecte Or, EUR/USD');
  });

  it('shows published figures with the honesty note when the feed carries them', () => {
    const { container } = renderDetail('forexfactory:cpi');
    const text = container.textContent ?? '';
    expect(text).toContain(fr.calendar.detail.forecastLabel);
    expect(text).toContain(fr.calendar.detail.previousLabel);
    expect(text).toContain(fr.calendar.detail.publishedFiguresNote);
  });

  it('renders absent organism and unit as visibly absent, never fabricated', () => {
    const { container } = renderDetail('forexfactory:cpi');
    const text = container.textContent ?? '';
    expect(text).toContain(fr.calendar.provenance.organismMissing);
    expect(text).toContain(fr.calendar.detail.unitMissing);
  });

  it('marks the engine-measured history as « mesures à venir » (NW-2)', () => {
    const { container } = renderDetail('forexfactory:cpi');
    expect(container.textContent ?? '').toContain(fr.calendar.detail.measuresPending);
  });

  it('renders the « ce que cette page ne dit pas » refusal list (4 items)', () => {
    const { container } = renderDetail('forexfactory:cpi');
    const items = Array.from(container.querySelectorAll('.cal-nono li'));
    expect(items).toHaveLength(4);
    expect(container.textContent ?? '').toContain('haussier ou baissier');
  });

  it('resolves a bare pipeline hash (App news deep-link, no source prefix)', () => {
    // The App news module links with the raw hash « cpi » (no « forexfactory: »).
    const { container } = renderDetail('cpi');
    expect(container.querySelector('h1')?.textContent).toBe('CPI y/y');
  });

  it('shows an honest « introuvable » state for an unknown id', () => {
    const { container } = renderDetail('forexfactory:does-not-exist');
    expect(container.querySelector('.cal-empty')?.textContent).toContain(
      fr.calendar.detail.notFound,
    );
  });

  it('renders no raw i18n keys', () => {
    const { container } = renderDetail('forexfactory:cpi');
    const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT);
    let n: Node | null;
    while ((n = walker.nextNode())) {
      const t = (n.textContent ?? '').trim();
      if (!t || t.includes('/') || t.includes(':')) continue;
      expect(/^[a-z][a-zA-Z0-9]*(\.[a-zA-Z][a-zA-Z0-9]*)+$/.test(t), `raw key: ${t}`).toBe(false);
    }
  });
});
