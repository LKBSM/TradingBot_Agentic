import { render, fireEvent, within } from '@/components/test-utils';
import { describe, expect, it } from 'vitest';
import fr from '@/messages/fr.json';
import { CalendarMonthView } from '../CalendarMonthView';
import type { CalendarEvent, CalendarResponse } from '@/types/calendar';

// A fixed "now" inside July 2026 (31 days; the 1st is a Wednesday). All event
// times are chosen at midday UTC so they land on the intended calendar day in
// the test runner's local timezone regardless of a modest offset.
const NOW = new Date('2026-07-15T12:00:00Z');

function ev(
  p: Partial<CalendarEvent> &
    Pick<CalendarEvent, 'event_id' | 'event' | 'scheduled_at' | 'source'>,
): CalendarEvent {
  return {
    series_code: null,
    license_label: null,
    currency: 'USD',
    organism: 'Bureau of Labor Statistics',
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
    value_series: [],
    ...p,
  };
}

// The 10th of July carries THREE publications (→ 2 shown + a "+1" counter).
// The 20th carries one, attached to EUR/USD only. The 5th carries one, quarterly.
const EVENTS: CalendarEvent[] = [
  ev({ event_id: 'bls:a:2026-07-10', source: 'bls', event: 'IPC', scheduled_at: '2026-07-10T12:30:00Z' }),
  ev({ event_id: 'bls:b:2026-07-10', source: 'bls', event: 'IPP', scheduled_at: '2026-07-10T13:00:00Z' }),
  ev({ event_id: 'census:c:2026-07-10', source: 'census', event: 'Ventes', scheduled_at: '2026-07-10T14:00:00Z', organism: 'U.S. Census Bureau' }),
  ev({ event_id: 'ecb:d:2026-07-20', source: 'ecb', event: 'Decision BCE', scheduled_at: '2026-07-20T12:15:00Z', markets: ['EURUSD'], organism: 'Banque centrale europeenne', periodicity: 'eight_per_year' }),
  ev({ event_id: 'bea:e:2026-07-05', source: 'bea', event: 'PIB', scheduled_at: '2026-07-05T12:30:00Z', organism: 'Bureau of Economic Analysis', periodicity: 'quarterly' }),
];

function makeData(events: CalendarEvent[]): CalendarResponse {
  return {
    window_start: '2026-07-01T00:00:00Z',
    window_end: '2026-07-31T23:59:59Z',
    generated_at: NOW.toISOString(),
    coverage: {
      source: 'official',
      feed_start: null,
      feed_end: null,
      partial: false,
      last_success: {},
      stale_sources: [],
    },
    attribution: [],
    events,
  };
}

const DATA = makeData(EVENTS);

function renderCal(data: CalendarResponse | null = DATA) {
  return render(<CalendarMonthView locale="fr" data={data} now={NOW} />);
}

const cells = (c: HTMLElement) => Array.from(c.querySelectorAll('.calm-cell'));
const dayCells = (c: HTMLElement) =>
  Array.from(c.querySelectorAll('.calm-cell:not(.blank)'));

/** The visible day cell whose day-number matches `n`. */
function cellForDay(c: HTMLElement, n: number): HTMLElement {
  const el = dayCells(c).find(
    (x) => x.querySelector('.calm-daynum')?.textContent === String(n),
  );
  if (!el) throw new Error(`day cell not found: ${n}`);
  return el as HTMLElement;
}

function chip(c: HTMLElement, label: string): HTMLElement {
  const el = Array.from(c.querySelectorAll('.fchip')).find(
    (b) => (b.textContent ?? '').trim() === label,
  );
  if (!el) throw new Error(`chip not found: ${label}`);
  return el as HTMLElement;
}

describe('NW-3 CalendarMonthView', () => {
  it('renders a full month: 7 weekday headers and every day of July as a cell', () => {
    const { container } = renderCal();
    expect(container.querySelectorAll('.calm-wd')).toHaveLength(7);
    // July 2026 has 31 days → 31 non-blank cells.
    expect(dayCells(container)).toHaveLength(31);
  });

  it('a grid is padded to whole weeks with leading/trailing blanks (July 1 is a Wednesday)', () => {
    const { container } = renderCal();
    const all = cells(container);
    // total cells is a multiple of 7.
    expect(all.length % 7).toBe(0);
    const blanks = container.querySelectorAll('.calm-cell.blank');
    // July 1 2026 is Wednesday → Monday-first grid has 2 leading blanks.
    expect(blanks.length).toBeGreaterThanOrEqual(2);
  });

  it('empty days remain visible (a day with no publication is still a cell)', () => {
    const { container } = renderCal();
    const empties = container.querySelectorAll('.calm-cell.empty');
    // Most of July has no publication in the fixture.
    expect(empties.length).toBeGreaterThan(20);
    // The 2nd has no event and is present.
    expect(cellForDay(container, 2)).toBeTruthy();
  });

  it('a day with more than 2 publications shows 2 chips then a "+{count}" counter', () => {
    const { container } = renderCal();
    const c10 = cellForDay(container, 10);
    expect(c10.querySelectorAll('.calm-chip')).toHaveLength(2);
    const more = c10.querySelector('.calm-more');
    expect(more?.textContent).toBe(fr.calendar.month.moreCount.replace('{count}', '1'));
  });

  it('clicking a day opens the side panel listing that day\'s publications', () => {
    const { container } = renderCal();
    fireEvent.click(cellForDay(container, 10));
    const panel = container.querySelector('.calm-panel') as HTMLElement;
    const rows = panel.querySelectorAll('.calm-panel-row');
    expect(rows).toHaveLength(3); // all three of the 10th, not just the 2 shown in-cell
    const text = panel.textContent ?? '';
    expect(text).toContain('IPC');
    expect(text).toContain('IPP');
    expect(text).toContain('Ventes');
    // each row deep-links to the event detail page.
    const links = Array.from(panel.querySelectorAll('a.calm-panel-link'));
    expect(links).toHaveLength(3);
    for (const a of links) {
      expect(a.getAttribute('href')).toMatch(/\/actualites\//);
    }
  });

  it('clicking an empty day opens the panel with the explicit empty message', () => {
    const { container } = renderCal();
    fireEvent.click(cellForDay(container, 2));
    const panel = container.querySelector('.calm-panel') as HTMLElement;
    expect(panel.textContent).toContain(fr.calendar.month.panel.empty);
  });

  it('the "this month" box counts total publications and empty days', () => {
    const { container } = renderCal();
    const box = container.querySelector('.calm-thismonth') as HTMLElement;
    // 5 events total this month → the mono count line reads "5 publications".
    expect(box.querySelector('.calm-tm-count')?.textContent).toBe('5 publications');
    // 31 days − 3 with publications (5th, 10th, 20th) = 28 empty days.
    expect(box.textContent).toContain('28');
    // per-market split names both markets.
    const withBox = within(box);
    expect(withBox.getByText(/Or/)).toBeTruthy();
  });

  it('a filter set emptied → the whole grid area shows noSelection', () => {
    const { container } = renderCal();
    for (const p of ['monthly', 'quarterly', 'eight_per_year'] as const) {
      fireEvent.click(chip(container, fr.calendar.periodicity[p]));
    }
    expect(container.querySelector('.calm-grid')).toBeNull();
    expect(container.querySelector('.cal-empty')?.textContent).toContain(
      fr.calendar.empty.noSelection,
    );
  });

  it('filters non-empty but matching nothing this month → filterEmpty (never silent)', () => {
    // Only the ECB source stays selected, but every ECB event is on a market the
    // ECB event carries; instead exclude every source that produced an event so
    // the filter matches nothing while the sets are non-empty.
    const { container } = renderCal();
    for (const s of ['bls', 'bea', 'census', 'ecb', 'forexfactory'] as const) {
      fireEvent.click(chip(container, fr.calendar.organism[s]));
    }
    // eurostat + federal_reserve remain selected but produced no event.
    expect(container.querySelector('.calm-grid')).not.toBeNull(); // grid still drawn
    expect(container.querySelector('.cal-empty')?.textContent).toContain(
      fr.calendar.month.filterEmpty,
    );
  });

  it('a month with no events at all shows month.empty', () => {
    const { container } = renderCal(makeData([]));
    expect(container.querySelector('.cal-empty')?.textContent).toContain(
      fr.calendar.month.empty,
    );
  });

  it('month navigation moves to an adjacent month (August has 31 days too)', () => {
    const { container } = renderCal();
    fireEvent.click(container.querySelector('.calm-navbtn')!); // prev → June
    // June 2026 has 30 days.
    expect(dayCells(container)).toHaveLength(30);
  });

  it('a time-not-confirmed publication is visibly distinct from a verified one (grid + panel)', () => {
    // Two events on July 8: a confirmed BLS one and an unconfirmed Eurostat one.
    const data = makeData([
      ev({
        event_id: 'bls:cpi:2026-07-08', source: 'bls', event: 'IPC',
        scheduled_at: '2026-07-08T12:30:00Z', time_confirmed: true,
      }),
      ev({
        event_id: 'eurostat:hicp:2026-07-08', source: 'eurostat', event: 'IPCH',
        scheduled_at: '2026-07-08T13:00:00Z', currency: 'EUR', markets: ['EURUSD'],
        organism: 'Eurostat', time_confirmed: false,
      }),
    ]);
    const { container } = render(
      <CalendarMonthView locale="fr" data={data} now={NOW} />,
    );
    // In the grid: exactly one chip time carries the "unconfirmed" marker class,
    // and the approximate glyph is present — the verified time has neither.
    const c8 = cellForDay(container, 8);
    expect(c8.querySelectorAll('.calm-chip-time.unconf')).toHaveLength(1);
    expect(c8.querySelector('.calm-chip-approx')).not.toBeNull();
    // In the day panel: the explicit label appears once, for the unconfirmed row only.
    fireEvent.click(c8);
    const panel = container.querySelector('.calm-panel') as HTMLElement;
    const marks = panel.querySelectorAll('.calm-panel-unconf');
    expect(marks).toHaveLength(1);
    expect(marks[0]?.textContent).toBe(fr.calendar.timeUnconfirmed);
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
