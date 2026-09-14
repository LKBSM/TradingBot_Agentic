import { describe, expect, it } from 'vitest';
import fr from '@/messages/fr.json';
import en from '@/messages/en.json';
import de from '@/messages/de.json';
import es from '@/messages/es.json';
import itMsg from '@/messages/it.json';
import pt from '@/messages/pt.json';
import nl from '@/messages/nl.json';
import pl from '@/messages/pl.json';
import ar from '@/messages/ar.json';
import { LANDING_STATS } from '@/lib/landing/stats';
import { DISPLAY_TIMEFRAMES, SUPPORTED_TIMEFRAMES } from '@/lib/market-reading/perimeter';

/**
 * LP-3 §D — the copy may not advertise a wider perimeter than the product serves.
 *
 * This guard exists because the drift it catches actually happened. Decision
 * DATA-1 (docs/governance/decisions/2026-08-16_data-1_m1_retrait_perimetre.md)
 * cut M1 from the perimeter — the selector shows five units, the scanner sweeps
 * 2 × 5 = 10 combinations — and the landing kept saying "6 unités de temps" and
 * "12 combinaisons" in nine languages for weeks. Nothing failed, because nothing
 * tied the sentence to the number.
 *
 * So: the figures are DERIVED (lib/landing/stats.ts), and every string that
 * states one is checked against them here. If M1 is ever re-enabled, this file
 * fails first and tells you which sentences to rewrite.
 */

type Dict = Record<string, unknown>;
const LOCALES: Record<string, Dict> = { fr, en, de, es, it: itMsg, pt, nl, pl, ar };

function at(root: Dict, path: string): string {
  const v = path.split('.').reduce<unknown>((acc, k) => (acc as Dict)?.[k], root);
  if (typeof v !== 'string') throw new Error(`missing string at home.${path}`);
  return v;
}

const home = (root: Dict) => root.home as Dict;

/** Strings that state how many timeframes the product serves. */
const TIMEFRAME_CLAIMS = ['tools.app.b5', 'how.step1.p', 'pricing.paid.f1', 'faq.a4'];
/** Strings that state how many market × timeframe combinations are swept. */
const COMBINATION_CLAIMS = [
  'how.step1.p',
  'pricing.paid.f2',
  'who.w3.p',
  'demo.scanner.side',
  'tools.scanner.p',
];

describe('LP-3 — the perimeter figures are derived, not written down', () => {
  it('mirrors the DATA-1 decision: M1 is out, five units are shown', () => {
    expect(SUPPORTED_TIMEFRAMES).toContain('M1'); // still a valid deep-link target
    expect(DISPLAY_TIMEFRAMES).not.toContain('M1'); // but never offered
    expect([...DISPLAY_TIMEFRAMES]).toEqual(['M5', 'M15', 'H1', 'H4', 'D1']);
    expect(LANDING_STATS.timeframes).toBe(5);
    expect(LANDING_STATS.combinations).toBe(LANDING_STATS.markets * LANDING_STATS.timeframes);
    expect(LANDING_STATS.combinations).toBe(10);
  });
});

describe('LP-3 — every perimeter claim matches the derived figures (9 locales)', () => {
  it('states the real number of timeframes, never the stale one', () => {
    for (const [loc, root] of Object.entries(LOCALES)) {
      for (const path of TIMEFRAME_CLAIMS) {
        const v = at(home(root), path);
        expect(v, `${loc} ${path}`).toContain(String(LANDING_STATS.timeframes));
        expect(v, `${loc} ${path} still claims 6 units`).not.toMatch(/(^|\D)6(\D|$)/);
      }
    }
  });

  it('states the real number of combinations, never the stale one', () => {
    for (const [loc, root] of Object.entries(LOCALES)) {
      for (const path of COMBINATION_CLAIMS) {
        const v = at(home(root), path);
        expect(v, `${loc} ${path}`).toContain(String(LANDING_STATS.combinations));
        expect(v, `${loc} ${path} still claims 12 combinations`).not.toMatch(/(^|\D)12(\D|$)/);
      }
    }
  });

  it('names the units it serves — the minute is no longer one of them', () => {
    for (const [loc, root] of Object.entries(LOCALES)) {
      for (const path of ['tools.app.b5', 'faq.a4']) {
        const v = at(home(root), path);
        expect(v, `${loc} ${path} must name M5`).toContain('M5');
        expect(v, `${loc} ${path} must name D1`).toContain('D1');
      }
    }
  });
});

/**
 * The two wording decisions of LP-3. Both are honesty rules, so they are pinned
 * the way honesty rules have to be: by what the sentence may NOT say.
 */
describe('LP-3 — the news headline does not predict', () => {
  // "the market will move" is an amplitude prediction — the very thing the page
  // promises two sections later it will never do.
  const PREDICTIVE: Record<string, string[]> = {
    fr: ['va bouger', 'va monter', 'va baisser'],
    en: ['will move', 'will rise', 'will drop'],
    de: ['bewegen wird'],
    es: ['se va a mover'],
    it: ['si muoverà'],
    pt: ['vai mexer'],
    nl: ['gaat bewegen'],
    pl: ['się poruszy'],
  };
  it('no locale announces a move (8 checked languages)', () => {
    for (const [loc, needles] of Object.entries(PREDICTIVE)) {
      const v = at(home(LOCALES[loc]!), 'tools.news.h3').toLowerCase();
      for (const n of needles) {
        expect(v, `${loc} news.h3 must not say "${n}": ${v}`).not.toContain(n);
      }
    }
  });

  it('it announces the publications instead — the fact the calendar carries', () => {
    expect(at(home(fr as Dict), 'tools.news.h3')).toContain('publications');
    expect(at(home(en as Dict), 'tools.news.h3')).toContain('releases');
    expect(at(home(es as Dict), 'tools.news.h3')).toContain('publicaciones');
  });
});

describe('LP-3 — a zone is described, never valued', () => {
  const VALUING: Record<string, string[]> = {
    fr: ['valent pas la même chose', "savoir ce qu'il vaut"],
    en: ['not worth the same', "knowing what it's worth"],
    de: ['nicht dasselbe wert', 'was es wert ist'],
    es: ['no valen lo mismo', 'saber lo que vale'],
    it: ['non valgono la stessa cosa', 'sapere quanto vale'],
    pt: ['não valem o mesmo', 'saber o que ele vale'],
    nl: ['niet hetzelfde waard', 'weten wat hij waard is'],
    pl: ['to nie to samo', 'ile jest wart'],
  };
  it('neither the zones headline nor its paragraph judges a zone (8 languages)', () => {
    for (const [loc, needles] of Object.entries(VALUING)) {
      const both = `${at(home(LOCALES[loc]!), 'tools.zones.h3')} ${at(home(LOCALES[loc]!), 'tools.zones.p')}`;
      for (const n of needles) {
        expect(both, `${loc} zones copy must not say "${n}"`).not.toContain(n);
      }
    }
  });

  it('the Tools section now says what the Demo section already said', () => {
    // The page used to contradict itself: Tools valued the zone, Demo read it.
    expect(at(home(fr as Dict), 'tools.zones.h3')).toContain('ne se lisent pas pareil');
    expect(at(home(fr as Dict), 'demo.zones.side.desc')).toContain('ne se lisent pas pareil');
    expect(at(home(fr as Dict), 'tools.zones.p')).toContain('connaître son histoire');
  });
});
