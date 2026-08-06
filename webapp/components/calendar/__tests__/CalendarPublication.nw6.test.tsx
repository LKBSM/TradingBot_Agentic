import { render } from '@/components/test-utils';
import { describe, expect, it } from 'vitest';
import fr from '@/messages/fr.json';
import en from '@/messages/en.json';
import { CalendarEventDetail } from '../CalendarEventDetail';
import type { CalendarEvent, CalendarResponse } from '@/types/calendar';

/**
 * NW-6 — page de publication : garde-fous propres à la mission.
 *  · Défaut A — le libellé du compte à rebours suit le temps (passé/futur) ;
 *  · Défaut B — aucun texte pédagogique générique ; un bloc sans fiche n'est pas rendu ;
 *  · Défaut C — un SEUL bloc d'avertissement (hors mention M.I.A) ;
 *  · vocabulaire interdit sur la surface (médiane / moyenne / bougie) dans les deux langues ;
 *  · le point à venir de la courbe n'affiche jamais de valeur.
 */

const NOW = new Date('2026-07-28T06:00:00Z');
const PEDAGOGY_KEYS = [
  'us_employment_situation', 'us_cpi', 'us_cpi_core', 'us_ppi', 'us_jolts',
  'us_gdp', 'us_pce', 'us_retail_sales', 'us_housing_starts', 'us_durable_goods',
  'us_fomc_rate', 'us_fomc_minutes', 'us_fomc_dotplot',
  'ea_hicp_flash', 'ea_gdp_flash', 'ea_unemployment', 'ea_ecb_rate',
] as const;

function ev(
  p: Partial<CalendarEvent> & Pick<CalendarEvent, 'event_id' | 'event' | 'source'>,
): CalendarEvent {
  return {
    series_code: null, license_label: null, currency: 'USD', organism: null,
    periodicity: 'monthly', scheduled_at: '2026-07-28T12:30:00Z',
    source_timezone: 'America/New_York', time_confirmed: true,
    markets: ['XAUUSD'], value_unit: null, actual: null, actual_initial: null,
    previous: null, revised: false, revised_at: null, actual_state: 'pending',
    refreshed_at: null, value_series: [], ...p,
  };
}

const SERIES: CalendarEvent['value_series'] = [
  { period: '2026-04', value: 320.0 },
  { period: '2026-05', value: 321.5 },
  { period: '2026-06', value: 322.9 },
];

function makeData(event: CalendarEvent): CalendarResponse {
  return {
    window_start: '2026-06-28T06:00:00Z', window_end: '2026-08-27T06:00:00Z',
    generated_at: NOW.toISOString(),
    coverage: { source: 'official', feed_start: null, feed_end: null, partial: false, last_success: {}, stale_sources: [] },
    attribution: [{ source: 'bls', organism: 'Bureau of Labor Statistics', license_label: 'Domaine public', policy_url: 'https://www.bls.gov/opub/copyright-information.htm' }],
    events: [event],
  };
}

function renderPub(eventId: string, event: CalendarEvent, locale = 'fr') {
  return render(
    <CalendarEventDetail eventId={eventId} locale={locale} data={makeData(event)} now={NOW} measures={null} />,
  );
}

const JOLTS = ev({
  event_id: 'bls:us_jolts:2026-08-04', source: 'bls', event: 'JOLTS',
  organism: 'Bureau of Labor Statistics', value_unit: 'milliers de postes',
  actual: 7400, actual_state: 'published', value_series: SERIES,
});

describe('NW-6 publication page guards', () => {
  it('(Défaut C) renders exactly ONE page-level warning block (M.I.A caption aside)', () => {
    const { container } = renderPub('bls:us_jolts:2026-08-04', JOLTS);
    // The single consolidated « ce que cette page ne dit pas » block.
    expect(container.querySelectorAll('.cal-nono')).toHaveLength(1);
    // The former pedagogy-card disclaimer no longer exists.
    expect(container.querySelector('.pub-ped-nono')).toBeNull();
    // The M.I.A capability mention stays (it is about M.I.A, not the page).
    expect(container.querySelector('.pub-mia-cap')).not.toBeNull();
  });

  it('(Défaut B) every catalog publication has a REAL fiche, and none renders the old filler', () => {
    const FILLER = 'indicateur économique officiel, paru selon un calendrier connu';
    for (const key of PEDAGOGY_KEYS) {
      const body = (fr.calendar.pub.pedagogy as unknown as Record<string, { body: string }>)[key]?.body ?? '';
      expect(body.length, `fr fiche ${key} missing`).toBeGreaterThan(80);
      expect(body).not.toContain(FILLER);
      const enBody = (en.calendar.pub.pedagogy as unknown as Record<string, { body: string }>)[key]?.body ?? '';
      expect(enBody.length, `en fiche ${key} missing`).toBeGreaterThan(80);
    }
  });

  it('(Défaut B) a publication with a real fiche renders the pedagogy card', () => {
    const { container } = renderPub('bls:us_jolts:2026-08-04', JOLTS);
    expect(container.querySelector('.pub-ped-body')?.textContent).toContain(
      fr.calendar.pub.pedagogy.us_jolts.body,
    );
  });

  it('the upcoming curve point never carries a value', () => {
    const { container } = renderPub('bls:us_jolts:2026-08-04', JOLTS);
    const up = container.querySelector('circle[data-upcoming="1"]');
    expect(up).not.toBeNull();
    const upLabel = container.querySelector('.pt-upcoming-label')?.textContent ?? '';
    expect(/\d/.test(upLabel)).toBe(false);
  });

  it('no statistic jargon (médiane / moyenne / bougie) is visible on this surface, both languages', () => {
    // Scan the whole calendar.pub namespace text — what the page can render.
    const forbidden = [/m[ée]diane/i, /moyenne/i, /\bmean\b/i, /\bmedian\b/i, /bougies?\b/i, /\bcandles?\b/i];
    const collect = (o: unknown, acc: string[] = []): string[] => {
      if (typeof o === 'string') acc.push(o);
      else if (o && typeof o === 'object') for (const v of Object.values(o)) collect(v, acc);
      return acc;
    };
    for (const [loc, msg] of [['fr', fr], ['en', en]] as const) {
      const strings = collect((msg as { calendar: { pub: unknown } }).calendar.pub);
      for (const s of strings) {
        for (const re of forbidden) {
          expect(re.test(s), `${loc}: forbidden term in "${s}"`).toBe(false);
        }
      }
    }
  });
});
