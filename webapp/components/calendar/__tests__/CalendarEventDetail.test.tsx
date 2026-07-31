import { render } from '@/components/test-utils';
import { describe, expect, it } from 'vitest';
import fr from '@/messages/fr.json';
import { CalendarEventDetail } from '../CalendarEventDetail';
import type {
  CalendarAttribution,
  CalendarEvent,
  CalendarResponse,
} from '@/types/calendar';
import type {
  PublicationMeasures,
  MeasureProvenance,
} from '@/types/measures';

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
    value_series: [],
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

const SERIES: CalendarEvent['value_series'] = [
  { period: '2026-03', value: 319.1 },
  { period: '2026-04', value: 320.0 },
  { period: '2026-05', value: 321.5 },
  { period: '2026-06', value: 322.9 },
];

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
  value_series: SERIES,
});

function prov(p: Partial<MeasureProvenance> = {}): MeasureProvenance {
  return {
    method_key: 'm',
    sample_size: 24,
    market: 'XAUUSD',
    period_start: '2024-07-01T00:00:00Z',
    period_end: '2026-06-30T00:00:00Z',
    reference_days: 40,
    quote_unit: 'USD',
    ...p,
  };
}

const MEASURES: PublicationMeasures = {
  event_key: 'us_cpi',
  market: 'XAUUSD',
  calm_before: {
    provenance: prov(),
    reference_amount: 1.8,
    calmer_count: 15,
    busier_count: 9,
    calmest: { observed_at: '2025-02-12T13:30:00Z', minutes: null, amount: 0.7 },
    busiest: { observed_at: '2025-09-11T12:30:00Z', minutes: null, amount: 6.4 },
  },
  structure_state: {
    provenance: prov(),
    inside_zone_count: 11,
    intact_pocket_count: 7,
    range_lower_count: 8,
    range_middle_count: 9,
    range_upper_count: 7,
    now_inside_zone: true,
    now_intact_pocket_within: false,
    now_range_position: 'upper',
  },
  return_to_calm: {
    provenance: prov(),
    tranches: [
      { lower_minutes: 0, upper_minutes: 60, count: 10 },
      { lower_minutes: 60, upper_minutes: 180, count: 8 },
      { lower_minutes: 180, upper_minutes: null, count: 4 },
    ],
    fastest: { observed_at: '2025-04-10T12:30:00Z', minutes: 22, amount: null },
    slowest: { observed_at: '2025-11-13T13:30:00Z', minutes: 195, amount: null },
    never_settled_count: 2,
  },
};

function renderDetail(
  eventId: string,
  event: CalendarEvent = OFFICIAL,
  measures: PublicationMeasures | null = null,
) {
  return render(
    <CalendarEventDetail
      eventId={eventId}
      locale="fr"
      data={makeData(event)}
      now={NOW}
      measures={measures}
    />,
  );
}

describe('NW-3 CalendarEventDetail', () => {
  it('shows the event title and attached markets, with NO impact ranking', () => {
    const { container } = renderDetail('bls:us_cpi:2026-07-28');
    expect(container.querySelector('h1')?.textContent).toBe('IPC');
    const text = container.textContent ?? '';
    expect(text).toContain('rattaché à Or, EUR/USD');
    expect(text).not.toContain('affecte');
    expect(container.querySelectorAll('.cal-impact')).toHaveLength(0);
  });

  it('(a) renders the value curve when value_series has points, and the upcoming point shows NO number', () => {
    const { container } = renderDetail('bls:us_cpi:2026-07-28');
    const svg = container.querySelector('.pub-curve-svg');
    expect(svg).not.toBeNull();
    // Every real value is plotted with its number...
    const dots = svg!.querySelectorAll('circle.dot');
    expect(dots).toHaveLength(SERIES.length);
    const valTexts = Array.from(svg!.querySelectorAll('text.pt-val')).map((e) => e.textContent);
    expect(valTexts).toContain('322.9');
    // ...but the single trailing UPCOMING point is hollow and carries no figure.
    const upcoming = svg!.querySelector('circle[data-upcoming="1"]');
    expect(upcoming).not.toBeNull();
    expect(upcoming?.classList.contains('dot-upcoming')).toBe(true);
    const upLabel = svg!.querySelector('.pt-upcoming-label')?.textContent ?? '';
    expect(upLabel).toBe(fr.calendar.pub.curve.upcoming);
    // the upcoming label must not be a number
    expect(/\d/.test(upLabel)).toBe(false);
  });

  it('omits the curve card entirely when value_series is empty (no placeholder)', () => {
    const bare = ev({ event_id: 'bls:e:1', source: 'bls', event: 'E', organism: 'Bureau of Labor Statistics', value_series: [], actual_state: 'pending' });
    const { container } = renderDetail('bls:e:1', bare);
    expect(container.querySelector('.pub-curve-svg')).toBeNull();
    expect(container.textContent ?? '').not.toContain(fr.calendar.pub.curve.empty);
  });

  it('keeps the current-release absence state visible on the curve card (pending)', () => {
    const { container } = renderDetail('bls:p:1', ev({
      event_id: 'bls:p:1', source: 'bls', event: 'P', organism: 'Bureau of Labor Statistics',
      value_series: SERIES, actual: null, actual_state: 'pending',
    }));
    expect(container.textContent).toContain(fr.calendar.detail.actualPending);
  });

  it('(f) shows both the initial AND the current value on the revision line', () => {
    const { container } = renderDetail('bls:us_cpi:2026-07-28');
    const revLine = Array.from(container.querySelectorAll('.cald-rev-line'))
      .map((e) => e.textContent ?? '')
      .join(' ');
    expect(revLine).toContain('321.5'); // initial, never overwritten
    expect(revLine).toContain('322.9'); // current
    expect(revLine.toLowerCase()).not.toContain('surprise');
  });

  it('a never-revised value says so explicitly', () => {
    const notRevised = ev({
      event_id: 'bls:y:1', source: 'bls', event: 'Y',
      organism: 'Bureau of Labor Statistics', actual: 100.0, revised: false,
      actual_state: 'published', value_series: SERIES,
    });
    const { container } = renderDetail('bls:y:1', notRevised);
    const lines = Array.from(container.querySelectorAll('.cald-rev-line')).map((e) => e.textContent);
    expect(lines).toContain(fr.calendar.pub.curve.notRevised);
  });

  it('(b) renders the four-questions section with source lines carrying a denominator, durations in h/min', () => {
    const { container } = renderDetail('bls:us_cpi:2026-07-28', OFFICIAL, MEASURES);
    const text = container.textContent ?? '';
    expect(text).toContain(fr.calendar.pub.questions.sectionTitle);
    // three measured questions present
    expect(container.querySelectorAll('.pub-qcard')).toHaveLength(3);
    // every source line shows the sample-size denominator (24)
    const sources = Array.from(container.querySelectorAll('.pub-qsource')).map((e) => e.textContent ?? '');
    expect(sources).toHaveLength(3);
    for (const s of sources) expect(s).toContain('24');
    // durations render as hours/minutes, NEVER candles
    const details = Array.from(container.querySelectorAll('.pub-detail')).map((e) => e.textContent ?? '').join(' ');
    expect(details).toContain('22 min'); // fastest = 22 minutes
    expect(details).toContain('3 h 15'); // slowest = 195 minutes → 3 h 15
    expect(text.toLowerCase()).not.toContain('bougie');
  });

  it('(b) the structure "now" line and return-to-calm never-settled render from the measures', () => {
    const { container } = renderDetail('bls:us_cpi:2026-07-28', OFFICIAL, MEASURES);
    expect(container.querySelector('.pub-now')).not.toBeNull();
    const never = container.querySelector('.pub-never')?.textContent ?? '';
    expect(never).toContain('2'); // never_settled_count
  });

  it('never-settled line is absent when the count is 0', () => {
    const m: PublicationMeasures = {
      ...MEASURES,
      return_to_calm: { ...MEASURES.return_to_calm!, never_settled_count: 0 },
    };
    const { container } = renderDetail('bls:us_cpi:2026-07-28', OFFICIAL, m);
    expect(container.querySelector('.pub-never')).toBeNull();
  });

  it('(c) the questions section is ABSENT when measures is null', () => {
    const { container } = renderDetail('bls:us_cpi:2026-07-28', OFFICIAL, null);
    expect(container.querySelector('.pub-qsection')).toBeNull();
    expect((container.textContent ?? '')).not.toContain(fr.calendar.pub.questions.sectionTitle);
  });

  it('(d) pedagogy body switches by event key', () => {
    const cpi = renderDetail('bls:us_cpi:2026-07-28').container;
    expect(cpi.textContent).toContain(fr.calendar.pub.pedagogy.us_cpi.body);

    const hicp = renderDetail('eurostat:ea_hicp_flash:2026-07-31', ev({
      event_id: 'eurostat:ea_hicp_flash:2026-07-31', source: 'bls', event: 'HICP',
      organism: 'Eurostat', actual_state: 'published', value_series: SERIES,
    })).container;
    expect(hicp.textContent).toContain(fr.calendar.pub.pedagogy.ea_hicp_flash.body);

    const other = renderDetail('bls:us_ppi:2026-08-01', ev({
      event_id: 'bls:us_ppi:2026-08-01', source: 'bls', event: 'PPI',
      organism: 'Bureau of Labor Statistics', actual_state: 'published', value_series: SERIES,
    })).container;
    expect(other.textContent).toContain(fr.calendar.pub.pedagogy.default.body);
  });

  it('(e) the go-to-source link points to the issuing organism domain', () => {
    const { container } = renderDetail('bls:us_cpi:2026-07-28');
    const link = container.querySelector('.pub-src-link');
    expect(link?.getAttribute('href')).toBe(
      'https://www.bls.gov/opub/copyright-information.htm',
    );
    expect(link?.getAttribute('target')).toBe('_blank');
    expect(link?.getAttribute('rel')).toBe('noreferrer noopener');
    expect(link?.textContent).toBe(fr.calendar.pub.source.link);
  });

  it('omits the go-to-source link when there is no attribution', () => {
    const bare = ev({ event_id: 'forexfactory:z:1', source: 'forexfactory', event: 'ADP', organism: null, value_unit: null, value_series: SERIES });
    const { container } = render(
      <CalendarEventDetail eventId="forexfactory:z:1" locale="fr" data={{ ...makeData(bare), attribution: [] }} now={NOW} measures={null} />,
    );
    // the section header is still present, but no link
    expect(container.textContent ?? '').toContain(fr.calendar.pub.source.title);
    expect(container.querySelector('.pub-src-link')).toBeNull();
  });

  it('the MIA block offers three clickable suggestions and a send action', () => {
    const { container } = renderDetail('bls:us_cpi:2026-07-28');
    expect(container.querySelectorAll('.pub-mia-chip')).toHaveLength(3);
    expect(container.querySelector('.pub-mia-send')?.textContent).toBe(fr.calendar.pub.mia.send);
    expect(container.querySelector('.pub-mia-input')?.getAttribute('placeholder')).toBe(
      fr.calendar.pub.mia.placeholder,
    );
  });

  it('the three value absences are distinct on the curve card (pending / unfetched / unavailable)', () => {
    const pend = ev({ event_id: 'bls:p:1', source: 'bls', event: 'P', organism: 'Bureau of Labor Statistics', actual: null, actual_state: 'pending', value_series: SERIES });
    let c = renderDetail('bls:p:1', pend).container;
    expect(c.textContent).toContain(fr.calendar.detail.actualPending);

    const unf = ev({ event_id: 'bls:u:1', source: 'bls', event: 'U', organism: 'Bureau of Labor Statistics', actual: null, actual_state: 'unfetched', refreshed_at: '2026-07-20T12:30:00Z', value_series: SERIES });
    c = renderDetail('bls:u:1', unf).container;
    const utxt = c.textContent ?? '';
    expect(utxt).toContain('non récupérée');
    expect(utxt).not.toContain(fr.calendar.detail.actualPending);

    const una = ev({ event_id: 'fed:f:1', source: 'federal_reserve', event: 'FOMC', organism: 'Federal Reserve Board', actual: null, actual_state: 'unavailable', series_code: null, value_series: SERIES });
    c = renderDetail('fed:f:1', una).container;
    const atxt = c.textContent ?? '';
    expect(atxt).toContain('sans valeur chiffrée unique');
    expect(atxt).toContain('Federal Reserve Board');
    expect(atxt).not.toContain('non récupérée');
  });

  it('renders absent organism and unit as visibly absent, never fabricated', () => {
    const bare = ev({ event_id: 'forexfactory:z:1', source: 'forexfactory', event: 'ADP', organism: null, value_unit: null });
    const { container } = render(
      <CalendarEventDetail eventId="forexfactory:z:1" locale="fr" data={{ ...makeData(bare), attribution: [] }} now={NOW} measures={null} />,
    );
    const text = container.textContent ?? '';
    expect(text).toContain(fr.calendar.provenance.organismMissing);
    expect(text).toContain(fr.calendar.detail.unitMissing);
  });

  it('renders the page-level « ce que cette page ne dit pas » refusal list (3 items)', () => {
    const { container } = renderDetail('bls:us_cpi:2026-07-28');
    const items = Array.from(container.querySelectorAll('.cal-nono li'));
    expect(items).toHaveLength(3);
    expect(container.textContent ?? '').toContain('haussier ou baissier');
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
    const { container } = renderDetail('bls:us_cpi:2026-07-28', OFFICIAL, MEASURES);
    const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT);
    let n: Node | null;
    while ((n = walker.nextNode())) {
      const t = (n.textContent ?? '').trim();
      if (!t || t.includes('/') || t.includes(':')) continue;
      expect(/^[a-z][a-zA-Z0-9]*(\.[a-zA-Z][a-zA-Z0-9]*)+$/.test(t), `raw key: ${t}`).toBe(false);
    }
  });
});
