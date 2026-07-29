import { render } from '@/components/test-utils';
import { describe, expect, it } from 'vitest';
import fr from '@/messages/fr.json';
import { CalendarEventDetail } from '../CalendarEventDetail';
import type {
  CalendarAttribution,
  CalendarEvent,
  CalendarResponse,
} from '@/types/calendar';

const NOW = new Date('2026-07-28T06:00:00Z');

function ev(
  p: Partial<CalendarEvent> & Pick<CalendarEvent, 'event_id' | 'event' | 'source'>,
): CalendarEvent {
  return {
    series_code: null,
    license_label: null,
    currency: 'USD',
    organism: null,
    periodicity: 'monthly',
    scheduled_at: '2026-07-28T12:30:00Z',
    source_timezone: 'America/New_York',
    time_confirmed: true,
    markets: ['XAUUSD', 'EURUSD'],
    value_unit: null,
    actual: null,
    actual_initial: null,
    previous: null,
    revised: false,
    revised_at: null,
    actual_state: 'pending',
    refreshed_at: null,
    ...p,
  };
}

const ATTRIBUTION: CalendarAttribution[] = [
  { source: 'bls', organism: 'Bureau of Labor Statistics', license_label: 'Domaine public (17 U.S.C. §105)', policy_url: 'https://www.bls.gov/opub/copyright-information.htm' },
];

function makeData(event: CalendarEvent): CalendarResponse {
  return {
    window_start: '2026-06-28T06:00:00Z',
    window_end: '2026-08-27T06:00:00Z',
    generated_at: NOW.toISOString(),
    coverage: { source: 'official', feed_start: null, feed_end: null, partial: false, last_success: {}, stale_sources: [] },
    attribution: ATTRIBUTION,
    events: [event],
  };
}

const OFFICIAL = ev({
  event_id: 'bls:us_cpi:2026-07-28',
  source: 'bls',
  event: 'IPC',
  organism: 'Bureau of Labor Statistics',
  value_unit: 'indice (1982-84 = 100)',
  actual: 322.9,
  actual_initial: 321.5,
  previous: 320.0,
  revised: true,
  revised_at: '2026-07-27T12:30:00Z',
  actual_state: 'published',
});

function renderDetail(eventId: string, event: CalendarEvent = OFFICIAL) {
  return render(
    <CalendarEventDetail eventId={eventId} locale="fr" data={makeData(event)} now={NOW} />,
  );
}

describe('NW-1b CalendarEventDetail', () => {
  it('shows the event title and attached markets, with NO impact ranking', () => {
    const { container } = renderDetail('bls:us_cpi:2026-07-28');
    expect(container.querySelector('h1')?.textContent).toBe('IPC');
    const text = container.textContent ?? '';
    expect(text).toContain('rattaché à Or, EUR/USD');
    expect(text).not.toContain('affecte');
    expect(container.querySelectorAll('.cal-impact')).toHaveLength(0);
  });

  it('shows the published value and previous, but NO consensus/forecast', () => {
    const { container } = renderDetail('bls:us_cpi:2026-07-28');
    const text = container.textContent ?? '';
    expect(text).toContain(fr.calendar.detail.actualLabel);
    expect(text).toContain(fr.calendar.detail.previousLabel);
    expect(text).toContain(fr.calendar.detail.publishedFiguresNote);
    // no consensus surface at all
    expect(text.toLowerCase()).not.toContain('consensus');
  });

  it('shows values AS PUBLISHED — no re-rounding', () => {
    const raw = ev({
      event_id: 'bls:x:1', source: 'bls', event: 'X',
      organism: 'Bureau of Labor Statistics', actual: 3.14159, previous: 2.71828,
      actual_state: 'published',
    });
    const { container } = renderDetail('bls:x:1', raw);
    const vals = Array.from(container.querySelectorAll('.cald-fig .v')).map((e) => e.textContent);
    expect(vals).toContain('3.14159');
    expect(vals).toContain('2.71828');
  });

  it('the three value absences are distinct on screen (pending / unfetched / unavailable)', () => {
    // 1) PENDING — future date, value does not exist yet
    const pend = ev({ event_id: 'bls:p:1', source: 'bls', event: 'P', organism: 'Bureau of Labor Statistics', actual: null, actual_state: 'pending' });
    let c = renderDetail('bls:p:1', pend).container;
    expect(c.textContent).toContain(fr.calendar.detail.actualPending);

    // 2) UNFETCHED — published but not obtained; a PRODUCT state, with last attempt
    const unf = ev({ event_id: 'bls:u:1', source: 'bls', event: 'U', organism: 'Bureau of Labor Statistics', actual: null, actual_state: 'unfetched', refreshed_at: '2026-07-20T12:30:00Z' });
    c = renderDetail('bls:u:1', unf).container;
    const utxt = c.textContent ?? '';
    expect(utxt).toContain('non récupérée');
    expect(utxt).not.toContain(fr.calendar.detail.actualPending); // never confused with "not yet published"

    // 3) UNAVAILABLE — source exposes no single figure, organism named
    const una = ev({ event_id: 'fed:f:1', source: 'federal_reserve', event: 'FOMC', organism: 'Federal Reserve Board', actual: null, actual_state: 'unavailable', series_code: null });
    c = renderDetail('fed:f:1', una).container;
    expect(c.textContent).toContain('non disponible');
    expect(c.textContent).toContain('Federal Reserve Board');
  });

  it('shows initial and current value coexisting, with the revision date', () => {
    const { container } = renderDetail('bls:us_cpi:2026-07-28');
    const revLine = container.querySelector('.cald-rev-line')?.textContent ?? '';
    expect(revLine).toContain('321.5'); // initial, never overwritten
    expect(revLine).toContain('322.9'); // current
    // a bare date is present (no qualification like "major"/"up"/"surprise")
    expect(revLine.toLowerCase()).not.toContain('surprise');
  });

  it('a never-revised value says so explicitly', () => {
    const notRevised = ev({
      event_id: 'bls:y:1', source: 'bls', event: 'Y',
      organism: 'Bureau of Labor Statistics', actual: 100.0, revised: false,
      actual_state: 'published',
    });
    const { container } = renderDetail('bls:y:1', notRevised);
    expect(container.querySelector('.cald-rev-line')?.textContent).toBe(
      fr.calendar.detail.notRevised,
    );
  });

  it('renders the attribution for the event source (licence condition)', () => {
    const { container } = renderDetail('bls:us_cpi:2026-07-28');
    const block = container.querySelector('.cal-attrib');
    expect(block?.textContent).toContain('Bureau of Labor Statistics');
    expect(container.querySelector('.cal-attrib a')?.getAttribute('href')).toBe(
      'https://www.bls.gov/opub/copyright-information.htm',
    );
  });

  it('renders absent organism and unit as visibly absent, never fabricated', () => {
    const bare = ev({ event_id: 'forexfactory:z:1', source: 'forexfactory', event: 'ADP', organism: null, value_unit: null });
    const { container } = render(
      <CalendarEventDetail eventId="forexfactory:z:1" locale="fr" data={{ ...makeData(bare), attribution: [] }} now={NOW} />,
    );
    const text = container.textContent ?? '';
    expect(text).toContain(fr.calendar.provenance.organismMissing);
    expect(text).toContain(fr.calendar.detail.unitMissing);
  });

  it('does NOT render the engine-measures card while no measure exists', () => {
    const { container } = renderDetail('bls:us_cpi:2026-07-28');
    // no data, no element: no card title, no placeholder, no future promise
    const text = (container.textContent ?? '').toLowerCase();
    expect(text).not.toContain('à venir');
    expect(container.querySelector('.cald-pending')).toBeNull();
    expect(text).not.toContain(fr.calendar.detail.measuresTitle.toLowerCase());
  });

  it('renders the « ce que cette page ne dit pas » refusal list (3 items, no amplitude line)', () => {
    const { container } = renderDetail('bls:us_cpi:2026-07-28');
    const items = Array.from(container.querySelectorAll('.cal-nono li'));
    expect(items).toHaveLength(3);
    expect(container.textContent ?? '').toContain('haussier ou baissier');
    // the past-amplitude refusal is gone while no amplitude is shown
    expect((container.textContent ?? '').toLowerCase()).not.toContain('amplitudes passées');
  });

  it('resolves a bare pipeline ref (App news deep-link, no source prefix)', () => {
    const { container } = renderDetail('2026-07-28'); // last segment of event_id
    expect(container.querySelector('h1')?.textContent).toBe('IPC');
  });

  it('shows an honest « introuvable » state for an unknown id', () => {
    const { container } = renderDetail('bls:does-not-exist');
    expect(container.querySelector('.cal-empty')?.textContent).toContain(
      fr.calendar.detail.notFound,
    );
  });

  it('renders no raw i18n keys', () => {
    const { container } = renderDetail('bls:us_cpi:2026-07-28');
    const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT);
    let n: Node | null;
    while ((n = walker.nextNode())) {
      const t = (n.textContent ?? '').trim();
      if (!t || t.includes('/') || t.includes(':')) continue;
      expect(/^[a-z][a-zA-Z0-9]*(\.[a-zA-Z][a-zA-Z0-9]*)+$/.test(t), `raw key: ${t}`).toBe(false);
    }
  });
});
