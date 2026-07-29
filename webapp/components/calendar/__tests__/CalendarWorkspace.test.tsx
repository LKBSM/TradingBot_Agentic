import { render, fireEvent } from '@/components/test-utils';
import { describe, expect, it } from 'vitest';
import fr from '@/messages/fr.json';
import { CalendarWorkspace } from '../CalendarWorkspace';
import type {
  CalendarAttribution,
  CalendarEvent,
  CalendarResponse,
} from '@/types/calendar';

const NOW = new Date('2026-07-28T06:00:00Z');

function ev(
  p: Partial<CalendarEvent> &
    Pick<CalendarEvent, 'event_id' | 'event' | 'scheduled_at' | 'source'>,
): CalendarEvent {
  return {
    series_code: null,
    license_label: null,
    currency: 'USD',
    organism: null,
    periodicity: 'monthly',
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
  { source: 'ecb', organism: 'Banque centrale européenne', license_label: 'Réutilisation si source citée et non modifiée', policy_url: 'https://www.ecb.europa.eu/x' },
  { source: 'census', organism: 'U.S. Census Bureau', license_label: 'Domaine public (17 U.S.C. §105)', policy_url: 'https://www.census.gov/x' },
];

const DATA: CalendarResponse = {
  window_start: '2026-07-25T06:00:00Z',
  window_end: '2026-08-04T06:00:00Z',
  generated_at: NOW.toISOString(),
  coverage: {
    source: 'official',
    feed_start: null,
    feed_end: null,
    partial: false,
    last_success: {},
    stale_sources: [],
  },
  attribution: ATTRIBUTION,
  events: [
    ev({ event_id: 'bls:us_cpi:2026-07-28', source: 'bls', event: 'IPC', currency: 'USD', scheduled_at: '2026-07-28T12:30:00Z', organism: 'Bureau of Labor Statistics', periodicity: 'monthly' }),
    ev({ event_id: 'ecb:ea_ecb_rate:2026-07-28', source: 'ecb', event: 'Décision BCE', currency: 'EUR', scheduled_at: '2026-07-28T12:15:00Z', markets: ['EURUSD'], organism: 'Banque centrale européenne', periodicity: 'eight_per_year' }),
    ev({ event_id: 'census:us_retail:2026-07-29', source: 'census', event: 'Ventes au détail', scheduled_at: '2026-07-29T12:30:00Z', organism: 'U.S. Census Bureau', periodicity: 'monthly', revised: true, actual: 322.9, actual_initial: 321.5, revised_at: '2026-07-27T12:30:00Z', actual_state: 'published' }),
    ev({ event_id: 'bls:us_retail:2026-07-27', source: 'bls', event: 'IPP', scheduled_at: '2026-07-27T12:30:00Z', organism: 'Bureau of Labor Statistics' }),
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

describe('NW-1b CalendarWorkspace list', () => {
  it('shows only upcoming events by default, grouped chronologically', () => {
    const { container } = renderCal();
    // today: ECB 12:15 then IPC 12:30 ; tomorrow: retail. Past IPP hidden.
    expect(rows(container)).toHaveLength(3);
    expect(evNames(container)).toEqual(['Décision BCE', 'IPC', 'Ventes au détail']);
  });

  it('renders NO amplitude slot while no measure exists (no future-content promise)', () => {
    const { container } = renderCal();
    // no data, no element: the amplitude column is absent, and no « mesures à
    // venir »-style promise appears anywhere in the list.
    expect(container.querySelectorAll('.cal-ampl')).toHaveLength(0);
    expect((container.textContent ?? '').toLowerCase()).not.toContain('à venir');
  });

  it('exposes NO impact ranking on any surface — no colour-graded badge', () => {
    const { container } = renderCal();
    expect(container.querySelectorAll('.cal-impact')).toHaveLength(0);
    const text = (container.textContent ?? '').toLowerCase();
    // the impact vocabulary (élevé/moyen/faible as a ranking) must be gone
    expect(text).not.toContain('impact');
  });

  it('the filters are factual: organism, market, periodicity (no impact filter)', () => {
    const { container } = renderCal();
    const secs = Array.from(container.querySelectorAll('.fsec')).map((e) => e.textContent);
    expect(secs).toContain(fr.calendar.filters.organismLabel);
    expect(secs).toContain(fr.calendar.filters.marketLabel);
    expect(secs).toContain(fr.calendar.filters.periodicityLabel);
  });

  it('exposes NO amplitude/magnitude sort control — chronological order only', () => {
    const { container } = renderCal();
    expect(
      container.querySelectorAll('select, [role="combobox"], [role="listbox"]'),
    ).toHaveLength(0);
    expect(evNames(container)[0]).toBe('Décision BCE');
  });

  it('zero periodicity chip selected → empty list + explicit message, never a fallback', () => {
    const { container } = renderCal();
    fireEvent.click(chip(container, fr.calendar.periodicity.monthly));
    fireEvent.click(chip(container, fr.calendar.periodicity.quarterly));
    fireEvent.click(chip(container, fr.calendar.periodicity.eight_per_year));
    expect(rows(container)).toHaveLength(0);
    expect(container.querySelector('.cal-empty')?.textContent).toContain(
      fr.calendar.empty.noSelection,
    );
  });

  it('an organism filter keeps only that source', () => {
    const { container } = renderCal();
    // deselect everything but BCE by toggling the others off.
    for (const s of ['bls', 'bea', 'census', 'federal_reserve', 'eurostat', 'forexfactory']) {
      fireEvent.click(chip(container, fr.calendar.organism[s as 'bls']));
    }
    expect(evNames(container)).toEqual(['Décision BCE']);
  });

  it('a missing organism renders as visibly absent; a present one is shown', () => {
    const withNull: CalendarResponse = {
      ...DATA,
      attribution: [...ATTRIBUTION, { source: 'forexfactory', organism: 'ForexFactory (prototype)', license_label: 'dev', policy_url: 'https://x' }],
      events: [
        ...DATA.events,
        ev({ event_id: 'forexfactory:x:1', source: 'forexfactory', event: 'ADP', scheduled_at: '2026-07-28T13:00:00Z', organism: null, periodicity: null }),
      ],
    };
    const { container } = render(<CalendarWorkspace locale="fr" data={withNull} now={NOW} />);
    const text = container.textContent ?? '';
    expect(text).toContain(fr.calendar.provenance.organismMissing);
    expect(text).toContain('Bureau of Labor Statistics');
  });

  it('a revised value is surfaced as revised', () => {
    const { container } = renderCal();
    const r = rows(container).find((x) => (x.textContent ?? '').includes('Ventes au détail'));
    expect(r?.textContent).toContain(fr.calendar.revisedBadge);
  });

  it('« Publications passées » switches to past releases', () => {
    const { container } = renderCal();
    fireEvent.click(container.querySelector('.cal-past')!);
    const shown = rows(container);
    expect(shown).toHaveLength(1);
    expect(shown[0]!.textContent).toContain('IPP');
  });

  it('every rendered event has an attribution for its source (licence condition)', () => {
    const { container } = renderCal();
    const attribSources = new Set(DATA.attribution.map((a) => a.source));
    // the block itself is present, naming each source used…
    const block = container.querySelector('.cal-attrib');
    expect(block).not.toBeNull();
    expect(block?.textContent).toContain('Bureau of Labor Statistics');
    expect(block?.textContent).toContain('Banque centrale européenne');
    // …and no served event lacks an attribution entry.
    for (const e of DATA.events) expect(attribSources.has(e.source)).toBe(true);
  });

  it('the attribution block links each source reuse policy', () => {
    const { container } = renderCal();
    const links = Array.from(container.querySelectorAll('.cal-attrib a')).map((a) =>
      a.getAttribute('href'),
    );
    expect(links).toContain('https://www.bls.gov/opub/copyright-information.htm');
  });

  it('the attached markets are named as a relation, not a cause (rattaché à Or, EUR/USD)', () => {
    const { container } = renderCal();
    const cpi = rows(container).find((r) => (r.textContent ?? '').includes('IPC'));
    expect(cpi?.textContent).toContain('rattaché à Or, EUR/USD');
    expect(cpi?.textContent).not.toContain('affecte');
  });

  it('each row exposes an « En savoir plus » link to that event detail page', () => {
    const { container } = renderCal();
    const links = Array.from(container.querySelectorAll('a.cal-more'));
    expect(links).toHaveLength(3);
    for (const a of links) {
      expect(a.getAttribute('href')).toMatch(/\/actualites\/[a-z]+%3A/);
      expect(a.textContent ?? '').toContain(fr.calendar.seeMore);
    }
  });

  it('a publication whose time is not confirmed is marked', () => {
    const withUnconf: CalendarResponse = {
      ...DATA,
      events: [ev({ event_id: 'census:x:1', source: 'census', event: 'Housing', scheduled_at: '2026-07-28T13:00:00Z', organism: 'U.S. Census Bureau', time_confirmed: false })],
    };
    const { container } = render(<CalendarWorkspace locale="fr" data={withUnconf} now={NOW} />);
    expect(container.querySelector('.cal-unconf')?.textContent).toBe(
      fr.calendar.timeUnconfirmed,
    );
  });

  it('the list shows the published value and distinguishes unfetched vs unavailable', () => {
    const past = '2026-07-27T12:30:00Z'; // before NOW
    const data: CalendarResponse = {
      ...DATA,
      attribution: [
        ...ATTRIBUTION,
        { source: 'federal_reserve', organism: 'Federal Reserve Board', license_label: 'Domaine public', policy_url: 'https://x' },
      ],
      events: [
        ev({ event_id: 'bls:pub:1', source: 'bls', event: 'Pub', scheduled_at: past, organism: 'Bureau of Labor Statistics', actual: 3.2, actual_state: 'published' }),
        ev({ event_id: 'bls:unf:1', source: 'bls', event: 'Unf', scheduled_at: past, organism: 'Bureau of Labor Statistics', actual_state: 'unfetched', refreshed_at: past }),
        ev({ event_id: 'fed:una:1', source: 'federal_reserve', event: 'Fomc', scheduled_at: past, organism: 'Federal Reserve Board', actual_state: 'unavailable', series_code: null }),
      ],
    };
    const { container } = render(<CalendarWorkspace locale="fr" data={data} now={NOW} />);
    fireEvent.click(container.querySelector('.cal-past')!); // show past releases
    const text = container.textContent ?? '';
    expect(Array.from(container.querySelectorAll('.cal-val .v')).map((e) => e.textContent)).toContain('3.2');
    expect(text).toContain('non récupérée');   // unfetched
    expect(text).toContain('non disponible');  // unavailable
  });

  it('shows per-organism freshness and flags a stale source', () => {
    const data: CalendarResponse = {
      ...DATA,
      coverage: {
        ...DATA.coverage,
        last_success: {
          bls: '2026-07-28T00:00:00Z',
          census: '2026-07-20T00:00:00Z',
          ecb: '2026-07-28T00:00:00Z',
        },
        stale_sources: ['census'],
      },
    };
    const { container } = render(<CalendarWorkspace locale="fr" data={data} now={NOW} />);
    const attrib = container.querySelector('.cal-attrib');
    expect(attrib?.textContent).toContain('rafraîchi le'); // fresh sources dated
    const stale = container.querySelector('.cal-attrib .fresh.stale');
    expect(stale?.textContent).toContain('non rafraîchi depuis'); // retrieval delay is not silent
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
