import { render } from '@/components/test-utils';
import { describe, expect, it } from 'vitest';
import fr from '@/messages/fr.json';
import { CalendarPreview } from '../CalendarPreview';
import type { CalendarEvent, CalendarResponse } from '@/types/calendar';

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

function makeData(events: CalendarEvent[]): CalendarResponse {
  return {
    window_start: '2026-06-28T06:00:00Z',
    window_end: '2026-08-27T06:00:00Z',
    generated_at: NOW.toISOString(),
    coverage: { source: 'official', feed_start: null, feed_end: null, partial: false, last_success: {}, stale_sources: [] },
    attribution: [],
    events,
  };
}

// A spread of upcoming releases across the 30-day horizon (well past 7 days),
// plus one past event that must be filtered out of the upcoming list.
function manyUpcoming(): CalendarEvent[] {
  const out: CalendarEvent[] = [];
  for (let i = 0; i < 8; i++) {
    const day = 29 + i; // 2026-07-29 .. beyond, spanning weeks
    const iso = new Date(NOW.getTime() + (i + 1) * 3 * 86_400_000).toISOString();
    out.push(
      ev({
        event_id: `bls:evt-${i}:${day}`,
        source: 'bls',
        event: `Publication ${i}`,
        organism: 'Bureau of Labor Statistics',
        scheduled_at: iso,
      }),
    );
  }
  return out;
}

describe('NW-3 §2C CalendarPreview', () => {
  it('shows the header title and a "see all" link to /actualites', () => {
    const { container } = render(
      <CalendarPreview data={makeData(manyUpcoming())} now={NOW} />,
    );
    expect(container.textContent).toContain(fr.calendar.preview.title);
    const link = container.querySelector('.calprev-link');
    expect(link?.textContent).toContain(fr.calendar.preview.seeAll);
    expect(link?.getAttribute('href')).toBe('/actualites');
  });

  it('lists ALL upcoming publications (no slice cap) in a scrollable list', () => {
    const events = manyUpcoming();
    const { container } = render(
      <CalendarPreview data={makeData(events)} now={NOW} />,
    );
    const rows = container.querySelectorAll('.calprev-row');
    expect(rows).toHaveLength(events.length); // all 8, not capped at 3/4
  });

  it('each row carries countdown, local time, name, organism, periodicity, markets and an Open link', () => {
    const one = ev({
      event_id: 'bls:cpi:2026-07-31',
      source: 'bls',
      event: 'IPC',
      organism: 'Bureau of Labor Statistics',
      periodicity: 'monthly',
      scheduled_at: '2026-07-31T12:30:00Z',
      markets: ['XAUUSD', 'EURUSD'],
    });
    const { container } = render(
      <CalendarPreview data={makeData([one])} now={NOW} />,
    );
    const row = container.querySelector('.calprev-row');
    const text = row?.textContent ?? '';
    expect(container.querySelector('.calprev-cd')).not.toBeNull(); // countdown
    expect(container.querySelector('.calprev-time')).not.toBeNull(); // local time
    expect(text).toContain('IPC'); // name
    expect(text).toContain(
      fr.calendar.provenance.organism.replace('{organism}', 'Bureau of Labor Statistics'),
    );
    expect(text).toContain(fr.calendar.periodicity.monthly); // periodicity
    expect(text).toContain(fr.calendar.market.XAUUSD); // attached markets
    expect(text).toContain(fr.calendar.market.EURUSD);
    const open = row?.querySelector('.calprev-more');
    expect(open?.textContent).toContain(fr.calendar.seeMore);
    expect(open?.getAttribute('href')).toBe(
      `/actualites/${encodeURIComponent('bls:cpi:2026-07-31')}`,
    );
  });

  it('an organism the source does not provide renders as visibly absent', () => {
    // An official source that simply did not supply organism metadata (the source
    // is whitelisted; the organism field is null) — CAL-1 keeps only official
    // sources, so the fixture uses one rather than a private aggregator.
    const bare = ev({
      event_id: 'census:adv:2026-07-30',
      source: 'census',
      event: 'ADP',
      organism: null,
      scheduled_at: '2026-07-30T12:15:00Z',
    });
    const { container } = render(
      <CalendarPreview data={makeData([bare])} now={NOW} />,
    );
    expect(container.textContent).toContain(fr.calendar.provenance.organismMissing);
  });

  it('shows the honest empty state (never an error) when nothing is upcoming', () => {
    const pastOnly = ev({
      event_id: 'bls:old:2026-07-01',
      source: 'bls',
      event: 'Ancienne publication',
      scheduled_at: '2026-07-01T12:30:00Z', // before NOW
    });
    const { container } = render(
      <CalendarPreview data={makeData([pastOnly])} now={NOW} />,
    );
    expect(container.querySelector('.calprev-empty')?.textContent).toBe(
      fr.calendar.preview.empty,
    );
    expect(container.querySelectorAll('.calprev-row')).toHaveLength(0);
  });

  it('renders no raw i18n keys', () => {
    const { container } = render(
      <CalendarPreview data={makeData(manyUpcoming())} now={NOW} />,
    );
    const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT);
    let n: Node | null;
    while ((n = walker.nextNode())) {
      const t = (n.textContent ?? '').trim();
      if (!t || t.includes('/') || t.includes(':')) continue;
      expect(/^[a-z][a-zA-Z0-9]*(\.[a-zA-Z][a-zA-Z0-9]*)+$/.test(t), `raw key: ${t}`).toBe(false);
    }
  });
});
