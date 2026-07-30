import { describe, expect, it } from 'vitest';
import fr from '@/messages/fr.json';
import en from '@/messages/en.json';
import de from '@/messages/de.json';
import es from '@/messages/es.json';
import itLocale from '@/messages/it.json';
import pt from '@/messages/pt.json';
import nl from '@/messages/nl.json';
import pl from '@/messages/pl.json';
import ar from '@/messages/ar.json';

/**
 * NW-1 copy-honesty guard (mission §0). The calendar announces MOMENTS, never
 * DIRECTIONS. This scans EVERY string in the `calendar` namespace (fr + en) and
 * fails if any predicts a value/reaction, calls a deviation bullish/bearish, or
 * qualifies an event as major / to-watch / an opportunity / a bias / a target.
 */

// Paths whose strings deliberately QUOTE the forbidden terms in order to REFUSE
// them ("n'interprète pas … comme haussier ou baissier"). Whitelisted from the
// phrase scan, then separately asserted to remain a refusal (cf. scanner.note).
const REFUSAL_PATHS = ['detail.nono'];

function collectStrings(node: unknown, out: string[], path = ''): void {
  if (REFUSAL_PATHS.some((p) => path === p || path.startsWith(`${p}.`))) return;
  if (typeof node === 'string') out.push(node);
  else if (Array.isArray(node))
    node.forEach((n, i) => collectStrings(n, out, `${path}.${i}`));
  else if (node && typeof node === 'object')
    Object.entries(node).forEach(([k, n]) =>
      collectStrings(n, out, path ? `${path}.${k}` : k),
    );
}

// Predictive / directional tokens that must never appear (fr + en). Chosen so a
// genuine violation trips them while the honest copy (which negates or omits
// them) stays clean.
const FORBIDDEN = [
  // FR — direction / prediction / trade call
  'haussier', 'baissier', 'va rebondir', 'va monter', 'va baisser',
  'devrait', 'risque de', 'attends-toi', 'prépare-toi', 'opportunité',
  'à surveiller', 'biais', 'cible', "signal d'achat", 'signal de vente',
  'majeur', 'important',
  // EN — direction / prediction / trade call
  'bullish', 'bearish', 'will rise', 'will fall', 'you should',
  'buy signal', 'sell signal', 'price target', 'opportunity', 'to watch for',
  'major', 'important',
];

describe('NW-1 calendar copy honesty', () => {
  const strings: string[] = [];
  collectStrings((fr as Record<string, unknown>).calendar, strings);
  collectStrings((en as Record<string, unknown>).calendar, strings);

  it('the calendar namespace exists in fr and en with matching keys', () => {
    const frCal = (fr as Record<string, unknown>).calendar;
    const enCal = (en as Record<string, unknown>).calendar;
    expect(frCal).toBeTypeOf('object');
    expect(enCal).toBeTypeOf('object');
    const flat = (o: unknown, p = ''): string[] =>
      o && typeof o === 'object' && !Array.isArray(o)
        ? Object.entries(o).flatMap(([k, v]) => flat(v, p ? `${p}.${k}` : k))
        : [p];
    expect(flat(enCal).sort()).toEqual(flat(frCal).sort());
  });

  it('no calendar string predicts a value, direction or reaction', () => {
    for (const s of strings) {
      const low = s.toLowerCase();
      for (const tok of FORBIDDEN) {
        expect(low.includes(tok), `« ${tok} » leaked in: "${s}"`).toBe(false);
      }
    }
  });

  it('the nav entry exists', () => {
    const frNav = (fr as Record<string, unknown>).nav as Record<string, unknown>;
    const enNav = (en as Record<string, unknown>).nav as Record<string, unknown>;
    expect(frNav.calendar).toBeTypeOf('string');
    expect(enNav.calendar).toBeTypeOf('string');
  });

  it('intro + « ce que ce calendrier ne dit pas » match the mockup verbatim', () => {
    const cal = (
      fr as unknown as {
        calendar: {
          intro: { lead: string; rest: string };
          nono: { title: string; body: string };
        };
      }
    ).calendar;
    expect(cal.intro.lead).toBe(
      'Ce calendrier annonce des moments, pas des directions.',
    );
    expect(cal.intro.rest).toContain(
      "MIA ne dit pas ce que le prix fera — elle mesure ce qui s'est produit les fois précédentes.",
    );
    expect(cal.nono.title).toBe('Ce que ce calendrier ne dit pas');
    expect(cal.nono.body).toContain(
      'Une publication peut ne rien provoquer, et un marché calme peut bouger sans aucune publication.',
    );
  });

  it('NW-1b: no impact ranking and no consensus keys survive anywhere', () => {
    const frCal = (fr as Record<string, unknown>).calendar as Record<string, unknown>;
    const enCal = (en as Record<string, unknown>).calendar as Record<string, unknown>;
    for (const cal of [frCal, enCal]) {
      // impact ranking removed (no organism grades its releases)
      expect(cal.impact).toBeUndefined();
      expect((cal.filters as Record<string, unknown>).impactLabel).toBeUndefined();
      const detail = cal.detail as Record<string, unknown>;
      // consensus removed (no organism publishes an analyst forecast)
      expect(detail.forecastLabel).toBeUndefined();
      expect(detail.forecastNote).toBeUndefined();
      // factual filters + attribution + revision surfaces present
      expect((cal.filters as Record<string, unknown>).organismLabel).toBeTypeOf('string');
      expect((cal.filters as Record<string, unknown>).periodicityLabel).toBeTypeOf('string');
      expect(cal.organism).toBeTypeOf('object');
      expect(cal.periodicity).toBeTypeOf('object');
      expect(cal.attribution).toBeTypeOf('object');
      expect(detail.revisedFromTo).toBeTypeOf('string');
      expect(detail.notRevised).toBeTypeOf('string');
    }
  });

  it('NW-1b: the « ne dit pas » block states no forecast + no ranking, as choices', () => {
    const nono = (fr as unknown as { calendar: { nono: { noForecast: string; noRanking: string } } }).calendar.nono;
    expect(nono.noForecast.toLowerCase()).toContain('prévision');
    expect(nono.noRanking.toLowerCase()).toContain('hiérarchie');
  });

  it('NW-1c: the calendar namespace is natively translated in all 9 locales (no EN fallback)', () => {
    const locales: Record<string, unknown> = { de, es, it: itLocale, pt, nl, pl, ar };
    const enCal = (en as { calendar: { title: string; intro: { lead: string }; detail: { actualPending: string } } }).calendar;
    for (const [name, msgs] of Object.entries(locales)) {
      const cal = (msgs as { calendar: { title: string; intro: { lead: string }; detail: { actualPending: string } } }).calendar;
      // Representative strings must NOT equal the English source (would be a fallback).
      expect(cal.title, `${name}.title untranslated`).not.toBe(enCal.title);
      expect(cal.intro.lead, `${name}.intro.lead untranslated`).not.toBe(enCal.intro.lead);
      expect(cal.detail.actualPending, `${name}.detail.actualPending untranslated`).not.toBe(enCal.detail.actualPending);
    }
  });

  it('the detail « ne dit pas » block quotes haussier/baissier only to REFUSE them', () => {
    const items = (
      fr as unknown as { calendar: { detail: { nono: { items: Record<string, string> } } } }
    ).calendar.detail.nono.items;
    const joined = Object.values(items).join(' ').toLowerCase();
    // It is present…
    expect(joined).toContain('haussier');
    expect(joined).toContain('baissier');
    // …strictly inside a refusal ("n'interprète pas … comme haussier ou baissier").
    expect(joined).toContain("n'interprète pas");
  });

  it('NW-1c: no causality verb links an event to a market', () => {
    // « affecte/impacte/influence… » would assert the publication ACTS on the
    // market — exactly what the « ne dit pas » block denies. The attachment is a
    // display convention (« rattaché à »), never a cause/effect.
    const CAUSALITY = [
      'affecte', 'affects', 'impacte', 'impacts', 'influence',
      'agit sur', 'joue sur', 'pèse sur', 'pese sur',
    ];
    const all: string[] = [];
    const collectAll = (node: unknown): void => {
      if (typeof node === 'string') all.push(node);
      else if (Array.isArray(node)) node.forEach(collectAll);
      else if (node && typeof node === 'object') Object.values(node).forEach(collectAll);
    };
    collectAll((fr as Record<string, unknown>).calendar);
    collectAll((en as Record<string, unknown>).calendar);
    for (const s of all) {
      const low = s.toLowerCase();
      for (const v of CAUSALITY) {
        expect(low.includes(v), `causal verb « ${v} » links an event to a market: "${s}"`).toBe(false);
      }
    }
    // the attachment label is a relation, not a cause/effect
    expect((fr as unknown as { calendar: { affects: string } }).calendar.affects).toMatch(/rattaché/);
    expect((en as unknown as { calendar: { affects: string } }).calendar.affects).toMatch(/attached/);
  });

  it('NW-1c: no i18n string contains an internal mission/ticket code', () => {
    // No mission/branch/ticket code (NW-2, RG-1, …) may appear in a user-visible
    // string. `_comment` keys are dev-only metadata (never rendered) → skipped.
    const CODE = /\b(?:NW|RG|LB|TF|VZ|UI|DG|MC)-\d/;
    const offenders: string[] = [];
    const walk = (node: unknown, path: string): void => {
      if (typeof node === 'string') {
        if (CODE.test(node)) offenders.push(`${path}: ${node}`);
      } else if (Array.isArray(node)) {
        node.forEach((n, i) => walk(n, `${path}.${i}`));
      } else if (node && typeof node === 'object') {
        for (const [k, v] of Object.entries(node)) {
          if (k === '_comment') continue; // dev-only metadata, not rendered
          walk(v, path ? `${path}.${k}` : k);
        }
      }
    };
    walk(fr, 'fr');
    walk(en, 'en');
    expect(offenders, offenders.join(' | ')).toEqual([]);
  });
});
