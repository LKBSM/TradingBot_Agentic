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
  zone_lifecycle: {
    provenance: prov(),
    zones_created_count: 34,
    tranches: [
      { lower_minutes: 0, upper_minutes: 60, count: 18 },
      { lower_minutes: 60, upper_minutes: 120, count: 8 },
      { lower_minutes: 120, upper_minutes: 1440, count: 5 },
    ],
    fastest: { observed_at: '2025-03-12T13:30:00Z', minutes: 12, amount: null },
    slowest: { observed_at: '2025-05-13T13:30:00Z', minutes: 600, amount: null },
    never_mitigated_count: 3,
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
    // four measured questions present (NW-7 added the zone-lifecycle card)
    expect(container.querySelectorAll('.pub-qcard')).toHaveLength(4);
    // every source line shows the sample-size denominator (24)
    const sources = Array.from(container.querySelectorAll('.pub-qsource')).map((e) => e.textContent ?? '');
    expect(sources).toHaveLength(4);
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

  it('(d) pedagogy renders a REAL fiche per publication — and NOTHING for a key without one', () => {
    const cpi = renderDetail('bls:us_cpi:2026-07-28').container;
    expect(cpi.textContent).toContain(fr.calendar.pub.pedagogy.us_cpi.body);

    const hicp = renderDetail('eurostat:ea_hicp_flash:2026-07-31', ev({
      event_id: 'eurostat:ea_hicp_flash:2026-07-31', source: 'bls', event: 'HICP',
      organism: 'Eurostat', actual_state: 'published', value_series: SERIES,
    })).container;
    expect(hicp.textContent).toContain(fr.calendar.pub.pedagogy.ea_hicp_flash.body);

    // us_ppi now has its OWN hand-written fiche (no generic filler).
    const ppi = renderDetail('bls:us_ppi:2026-08-01', ev({
      event_id: 'bls:us_ppi:2026-08-01', source: 'bls', event: 'PPI',
      organism: 'Bureau of Labor Statistics', actual_state: 'published', value_series: SERIES,
    })).container;
    expect(ppi.textContent).toContain(fr.calendar.pub.pedagogy.us_ppi.body);

    // An unknown publication key ships NO fiche → the pedagogy card is not rendered
    // at all (Défaut B: never a generic placeholder occupying the slot).
    const unknown = render(
      <CalendarEventDetail
        eventId="forexfactory:z:1"
        locale="fr"
        data={{ ...makeData(ev({ event_id: 'forexfactory:z:1', source: 'forexfactory', event: 'ADP', organism: null, value_series: SERIES })), attribution: [] }}
        now={NOW}
        measures={null}
      />,
    ).container;
    expect(unknown.querySelector('.pub-ped-body')).toBeNull();
    expect(unknown.textContent ?? '').not.toContain(fr.calendar.pub.pedagogy.title);
  });

  it('(Défaut A) the countdown label matches tense: « Publication dans » ahead, « Publiée » once past', () => {
    // OFFICIAL is scheduled 2026-07-28T12:30Z; NOW is 06:00Z the same day → still ahead.
    const ahead = renderDetail('bls:us_cpi:2026-07-28').container;
    expect(ahead.querySelector('.cald-cd .k')?.textContent).toBe(
      fr.calendar.detail.countdownLabel,
    );

    // A release moved a day into the past reads « Publiée », never « Publication dans ».
    const past = renderDetail('bls:us_cpi:2026-07-26', ev({
      event_id: 'bls:us_cpi:2026-07-26', source: 'bls', event: 'IPC',
      organism: 'Bureau of Labor Statistics', scheduled_at: '2026-07-27T12:30:00Z',
      actual_state: 'published', value_series: SERIES,
    })).container;
    expect(past.querySelector('.cald-cd .k')?.textContent).toBe(
      fr.calendar.detail.countdownLabelPast,
    );
  });

  it('(e) the named go-to-source links point only to the issuing organism domain', () => {
    const { container } = renderDetail('bls:us_cpi:2026-07-28');
    const docs = Array.from(container.querySelectorAll('.pub-src-doc'));
    // us_cpi ships four named documents, all on bls.gov
    expect(docs.length).toBe(4);
    for (const a of docs) {
      const href = a.getAttribute('href') ?? '';
      const host = new URL(href).hostname.replace(/^www\./, '');
      expect(host === 'bls.gov' || host.endsWith('.bls.gov'), href).toBe(true);
      expect(a.getAttribute('target')).toBe('_blank');
      expect(a.getAttribute('rel')).toBe('noreferrer noopener');
    }
    // the license line is preserved
    expect(container.querySelector('.pub-src-license')?.textContent).toContain(
      'Bureau of Labor Statistics',
    );
  });

  it('(e) shows no named links and no license for an unknown organism/event, never a generic link', () => {
    const bare = ev({ event_id: 'forexfactory:z:1', source: 'forexfactory', event: 'ADP', organism: null, value_unit: null, value_series: SERIES });
    const { container } = render(
      <CalendarEventDetail eventId="forexfactory:z:1" locale="fr" data={{ ...makeData(bare), attribution: [] }} now={NOW} measures={null} />,
    );
    // the section header is still present, but no named link and no license
    expect(container.textContent ?? '').toContain(fr.calendar.pub.source.title);
    expect(container.querySelector('.pub-src-doc')).toBeNull();
    expect(container.querySelector('.pub-src-license')).toBeNull();
    expect(container.textContent ?? '').toContain(fr.calendar.pub.source.noneYet);
  });

  it('(d) the MIA block reuses the shared avatar (presence dot) and offers publication-specific suggestions', () => {
    const { container } = renderDetail('bls:us_cpi:2026-07-28');
    // shared AgentAvatar: the candlestick logo SVG + the presence pastille
    expect(container.querySelector('.pub-mia-head svg')).not.toBeNull();
    expect(container.querySelector('.pub-mia-head [data-presence="1"]')).not.toBeNull();
    // us_cpi has four bespoke suggestions
    expect(container.querySelectorAll('.pub-mia-chip')).toHaveLength(4);
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

describe('NW-8 variation-first curve', () => {
  const idxEvent = ev({
    event_id: 'bls:us_cpi:2026-07-28', source: 'bls', event: 'IPC',
    organism: 'Bureau of Labor Statistics', series_code: 'CUUR0000SA0',
    value_unit: 'indice (1982-84 = 100)', variation_kind: 'index_change',
    variation_published: true, actual: 322.9, actual_state: 'published',
    value_series: [
      { period: '2026-05', value: 3.4, level: 321.5, change_mom: 0.5 },
      { period: '2026-06', value: 3.1, level: 322.9, change_mom: 0.3 },
    ],
  });

  it('index_change: mo + yr % in evidence, level small, published attribution, blank upcoming', () => {
    const { container } = renderDetail('bls:us_cpi:2026-07-28', idxEvent);
    const head = container.querySelector('.pub-var-headline')?.textContent ?? '';
    expect(head).toContain('+0,3'); // month-over-month %, signed (fr)
    expect(head).toContain('+3,1'); // year-over-year %
    expect(container.querySelector('.pub-var-level')?.textContent ?? '').toContain('322,9');
    const attrib = container.querySelector('.pub-curve-attrib')?.textContent ?? '';
    expect(attrib).toContain('Bureau of Labor Statistics');
    expect(attrib).toContain('CUUR0000SA0');
    // the upcoming point still carries NO number (level OR variation)
    expect(container.querySelector('circle[data-upcoming="1"]')).not.toBeNull();
    expect(/\d/.test(container.querySelector('.pt-upcoming-label')?.textContent ?? '')).toBe(false);
    // a real explanation sentence is rendered (us_cpi is whitelisted)
    expect(container.querySelector('.pub-curve-explain')).not.toBeNull();
  });

  it('count_change: monthly absolute change in evidence, total kept as the level', () => {
    const nfp = ev({
      event_id: 'bls:us_employment_situation:2026-07-28', source: 'bls', event: 'NFP',
      organism: 'Bureau of Labor Statistics', series_code: 'CES0000000001',
      value_unit: "milliers d'emplois", variation_kind: 'count_change',
      variation_published: true, actual: 159000, actual_state: 'published',
      value_series: [
        { period: '2026-05', value: -30, level: 158850 },
        { period: '2026-06', value: 150, level: 159000 },
      ],
    });
    const { container } = renderDetail('bls:us_employment_situation:2026-07-28', nfp);
    const head = container.querySelector('.pub-var-headline')?.textContent ?? '';
    expect(head).toContain('+150');
    expect(head).toContain("milliers d'emplois");
    expect(container.querySelector('.pub-var-level')?.textContent ?? '').toContain('159');
  });

  it('published_change: the value IS the variation, no separate level line', () => {
    const gdp = ev({
      event_id: 'bea:us_gdp:2026-07-28', source: 'bea', event: 'PIB',
      organism: 'Bureau of Economic Analysis', series_code: 'NIPA-T10101',
      value_unit: '% (variation du PIB réel)', variation_kind: 'published_change',
      variation_published: true, actual: 2.8, actual_state: 'published',
      value_series: [
        { period: '2026-Q1', value: 1.4 },
        { period: '2026-Q2', value: 2.8 },
      ],
    });
    const { container } = renderDetail('bea:us_gdp:2026-07-28', gdp);
    expect(container.querySelector('.pub-var-headline')?.textContent ?? '').toContain('+2,8');
    expect(container.querySelector('.pub-var-level')).toBeNull();
  });
});

describe('NW-8 Batch 2 computed variation', () => {
  it('amount_change (computed): monthly % in evidence, amount as level, CALCULATED attribution', () => {
    const retail = ev({
      event_id: 'census:us_retail_sales:2026-07-28', source: 'census', event: 'Ventes de détail',
      organism: 'U.S. Census Bureau', series_code: 'MARTS-RSAFS',
      value_unit: 'millions de dollars', variation_kind: 'amount_change',
      variation_published: false, actual: 720000, actual_state: 'published',
      value_series: [
        { period: '2026-05', value: 0.4, level: 716000 },
        { period: '2026-06', value: 0.6, level: 720000 },
      ],
    });
    const { container } = renderDetail('census:us_retail_sales:2026-07-28', retail);
    const head = container.querySelector('.pub-var-headline')?.textContent ?? '';
    expect(head).toContain('+0,6'); // month-over-month %, signed
    expect(container.querySelector('.pub-var-level')?.textContent ?? '').toContain('720'); // amount level
    // attribution is the COMPUTED one — names MIA + organism, marked "calculée"
    const attrib = container.querySelector('.pub-curve-attrib')?.textContent ?? '';
    expect(attrib.toLowerCase()).toContain('calculée');
    expect(attrib).toContain('U.S. Census Bureau');
    // explanation sentence rendered (us_retail_sales now whitelisted)
    expect(container.querySelector('.pub-curve-explain')).not.toBeNull();
  });
});
