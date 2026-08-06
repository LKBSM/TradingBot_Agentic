import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { render } from '@/components/test-utils';
import { describe, expect, it } from 'vitest';
import fr from '@/messages/fr.json';
import en from '@/messages/en.json';
import {
  SOURCE_LINKS,
  SOURCE_DOMAINS,
  sourceLinksFor,
} from '@/lib/calendar/sourceLinks';
import { CalendarEventDetail } from '../CalendarEventDetail';
import type { CalendarEvent, CalendarResponse } from '@/types/calendar';
import type { PublicationMeasures, MeasureProvenance } from '@/types/measures';

/**
 * NW-5 — page d'une publication : garde-fous propres à la mission.
 *  · liens « aller à la source » = organisme émetteur UNIQUEMENT ;
 *  · composant M.I.A partagé avec /app — une seule implémentation ;
 *  · la mention de refus est conservée dans les deux langues ;
 *  · le point à venir de la courbe n'affiche jamais de valeur ;
 *  · l'avertissement commun « comment lire » est rendu sous les quatre cartes.
 */

const NOW = new Date('2026-07-28T06:00:00Z');
const REPO = join(process.cwd(), '..');
const WEBAPP = process.cwd();

function ev(
  p: Partial<CalendarEvent> & Pick<CalendarEvent, 'event_id' | 'event' | 'source'>,
): CalendarEvent {
  return {
    series_code: null, license_label: null, currency: 'USD', organism: null,
    periodicity: 'monthly', scheduled_at: '2026-07-28T12:30:00Z',
    source_timezone: 'America/New_York', time_confirmed: true,
    markets: ['XAUUSD', 'EURUSD'], value_unit: null, actual: null,
    actual_initial: null, previous: null, revised: false, revised_at: null,
    actual_state: 'pending', refreshed_at: null, value_series: [], ...p,
  };
}

const SERIES: CalendarEvent['value_series'] = [
  { period: '2026-03', value: 319.1 },
  { period: '2026-04', value: 320.0 },
  { period: '2026-05', value: 321.5 },
  { period: '2026-06', value: 322.9 },
];

function makeData(event: CalendarEvent): CalendarResponse {
  return {
    window_start: '2026-06-28T06:00:00Z', window_end: '2026-08-27T06:00:00Z',
    generated_at: NOW.toISOString(),
    coverage: { source: 'official', feed_start: null, feed_end: null, partial: false, last_success: {}, stale_sources: [] },
    attribution: [{ source: 'bls', organism: 'Bureau of Labor Statistics', license_label: 'Domaine public (17 U.S.C. §105)', policy_url: 'https://www.bls.gov/opub/copyright-information.htm' }],
    events: [event],
  };
}

function prov(p: Partial<MeasureProvenance> = {}): MeasureProvenance {
  return { method_key: 'm', sample_size: 12, market: 'XAUUSD', period_start: '2025-07-01T00:00:00Z', period_end: '2026-06-30T00:00:00Z', reference_days: 60, quote_unit: 'USD', ...p };
}

const MEASURES: PublicationMeasures = {
  event_key: 'us_cpi', market: 'XAUUSD',
  calm_before: { provenance: prov(), reference_amount: 9, calmer_count: 10, busier_count: 2, calmest: { observed_at: '2026-02-12T13:30:00Z', minutes: null, amount: 3.4 }, busiest: { observed_at: '2026-04-10T12:30:00Z', minutes: null, amount: 12 } },
  structure_state: { provenance: prov(), inside_zone_count: 5, intact_pocket_count: 9, range_lower_count: 2, range_middle_count: 2, range_upper_count: 8, now_inside_zone: true, now_intact_pocket_within: false, now_range_position: 'upper' },
  zone_lifecycle: null,
  return_to_calm: { provenance: prov(), tranches: [ { lower_minutes: 0, upper_minutes: 60, count: 4 }, { lower_minutes: 60, upper_minutes: 180, count: 5 }, { lower_minutes: 180, upper_minutes: null, count: 3 } ], fastest: { observed_at: '2026-03-12T12:30:00Z', minutes: 15, amount: null }, slowest: { observed_at: '2026-05-13T13:30:00Z', minutes: 285, amount: null }, never_settled_count: 0 },
};

const OFFICIAL = ev({
  event_id: 'bls:us_cpi:2026-08-12', source: 'bls', event: 'IPC',
  organism: 'Bureau of Labor Statistics', value_unit: 'indice (1982-84 = 100)',
  actual: 322.9, actual_initial: 321.5, previous: 320.0, revised: true,
  revised_at: '2026-07-27T12:30:00Z', actual_state: 'published', value_series: SERIES,
});

function renderDetail(measures: PublicationMeasures | null = MEASURES) {
  return render(<CalendarEventDetail eventId="bls:us_cpi:2026-08-12" locale="fr" data={makeData(OFFICIAL)} now={NOW} measures={measures} />);
}

describe('NW-5 publication page guards', () => {
  it('every source link points to the issuing organism domain, and to it alone', () => {
    // Map event key → source from the catalog (single source of truth).
    const catalog = JSON.parse(readFileSync(join(REPO, 'config', 'calendar_catalog.json'), 'utf-8')) as { events: Array<{ key: string; source: string }> };
    const sourceByKey = new Map(catalog.events.map((e) => [e.key, e.source]));
    for (const [key, links] of Object.entries(SOURCE_LINKS)) {
      const source = sourceByKey.get(key);
      expect(source, `event ${key} missing from catalog`).toBeTypeOf('string');
      const domain = SOURCE_DOMAINS[source as string];
      expect(domain, `no domain for source ${source}`).toBeTypeOf('string');
      for (const url of Object.values(links)) {
        const host = new URL(url).hostname.replace(/^www\./, '');
        expect(host === domain || host.endsWith(`.${domain}`), `${key}: ${url} not on ${domain}`).toBe(true);
      }
    }
  });

  it('never emits more than four named links per publication', () => {
    for (const key of Object.keys(SOURCE_LINKS)) {
      expect(sourceLinksFor(key).length).toBeLessThanOrEqual(4);
    }
    expect(sourceLinksFor(null)).toHaveLength(0);
    expect(sourceLinksFor('does-not-exist')).toHaveLength(0);
  });

  it('the M.I.A component is shared with /app — a single logo implementation', () => {
    // The publication page reuses the shared AgentAvatar, never a second glyph.
    const detail = readFileSync(join(WEBAPP, 'components', 'calendar', 'CalendarEventDetail.tsx'), 'utf-8');
    expect(detail).toMatch(/import \{ AgentAvatar \} from '@\/components\/chat\/AgentAvatar'/);
    expect(detail.includes('MiaAgentLogo')).toBe(false); // no direct re-use of the raw glyph
    // no duplicated avatar glyph (candlestick rects / sparkle path) in the page
    expect(detail.includes('M12 3l1.9')).toBe(false); // the maquette sparkle path

    // Exactly ONE definition of the candlestick logo across all components.
    const files: string[] = [];
    const walk = (dir: string) => {
      for (const e of readdirSync(dir, { withFileTypes: true })) {
        const p = join(dir, e.name);
        if (e.isDirectory()) walk(p);
        else if ((e.name.endsWith('.tsx') || e.name.endsWith('.ts')) && !/\.(test|spec)\./.test(e.name))
          files.push(p);
      }
    };
    walk(join(WEBAPP, 'components'));
    const defs = files.filter((f) => /export function MiaAgentLogo\b/.test(readFileSync(f, 'utf-8')));
    expect(defs).toHaveLength(1);
  });

  it('the refusal mention is preserved in both languages', () => {
    const frCap = (fr as { calendar: { pub: { mia: { capability: string } } } }).calendar.pub.mia.capability;
    const enCap = (en as { calendar: { pub: { mia: { capability: string } } } }).calendar.pub.mia.capability;
    // refuses: what price will do / what value will be published / whether to act
    expect(frCap).toContain('ce que le prix fera');
    expect(frCap).toContain('quelle valeur sera publiée');
    expect(frCap).toContain('intervenir');
    expect(enCap.toLowerCase()).toContain('what price will do');
    expect(enCap.toLowerCase()).toContain('what value will be published');
    expect(enCap.toLowerCase()).toContain('act');
  });

  it('the upcoming curve point never carries a value; the stats row shows last value + range', () => {
    const { container } = renderDetail(null);
    const up = container.querySelector('circle[data-upcoming="1"]');
    expect(up).not.toBeNull();
    // the upcoming label is not a number
    const upLabel = container.querySelector('.pt-upcoming-label')?.textContent ?? '';
    expect(/\d/.test(upLabel)).toBe(false);
    // stats row shows the last real value (322.9) and the range bounds (319.1 → 322.9)
    const stats = container.querySelector('.pub-curve-stats')?.textContent ?? '';
    expect(stats).toContain('322.9');
    expect(stats).toContain('319.1');
  });

  it('renders the common « comment lire » guide once, under the question cards', () => {
    const { container } = renderDetail(MEASURES);
    const warn = container.querySelectorAll('.pub-qwarn');
    expect(warn).toHaveLength(1);
    expect(container.querySelector('.pub-qwarn')?.textContent ?? '').toContain(
      fr.calendar.pub.questions.readGuide.title,
    );
    // three measured cards, numbered 1..3 (deferred zone measure leaves no gap)
    const nums = Array.from(container.querySelectorAll('.pub-qn')).map((e) => e.textContent);
    expect(nums).toHaveLength(3);
  });

  it('a deferred/absent measure is never rendered as an empty card', () => {
    // Only calm_before present → exactly one card, numbered « Question 1 ».
    const only: PublicationMeasures = { ...MEASURES, structure_state: null, return_to_calm: null };
    const { container } = renderDetail(only);
    expect(container.querySelectorAll('.pub-qcard')).toHaveLength(1);
    expect(container.querySelector('.pub-qn')?.textContent).toBe(
      fr.calendar.pub.questions.qLabel.replace('{n}', '1'),
    );
  });
});
