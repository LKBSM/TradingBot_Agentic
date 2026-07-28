import { describe, expect, it } from 'vitest';
import fr from '@/messages/fr.json';
import en from '@/messages/en.json';

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
    expect(joined).toContain('aucune valeur prédictive');
  });
});
