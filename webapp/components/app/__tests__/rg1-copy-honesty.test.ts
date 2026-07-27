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
 * RG-1 copy-honesty guard for the enriched Régime panel (`regimePanel.*`).
 *
 * The inviolable line: descriptive at all times, never a favourable ranking or a
 * prediction. On the ASSERTIVE surface (tile labels, source sub-lines, values,
 * Donnée row labels, state labels, Concept titles) the full forbidden vocabulary
 * is banned outright. The Concept BODIES and « ce que ça ne dit pas » blocks are
 * pedagogy that must be free to *negate* those very notions (« MIA n'affirmera
 * jamais qu'un alignement rend une entrée plus sûre ») — the same refusal-context
 * nuance the UI-2 guard already recognises — so they are checked only against
 * unambiguously promotional PHRASES, and are additionally required to carry a
 * non-empty « ne dit pas » block.
 */

const BUNDLES: [string, Record<string, unknown>][] = [
  ['fr', fr], ['en', en], ['de', de], ['es', es], ['it', itLocale],
  ['pt', pt], ['nl', nl], ['pl', pl], ['ar', ar],
];

const CONCEPT_KEYS = [
  'regime', 'phase', 'trend', 'vol', 'pos', 'align', 'mat', 'last', 'dens', 'sess', 'lvl',
] as const;

// Full ban on the assertive surface — a bare occurrence there is a promise.
const ASSERTIVE_FORBIDDEN: RegExp[] = [
  /\bva\b/i, /\bcible/i, /\bbiais/i, /\bsetup\b/i, /\bsignal/i, /probabilit/i,
  /plus s[ûu]re?\b/i, /plus safe/i, /\bmeilleur/i, /recommand/i, /recommend/i,
];

// Purely promotional phrasings — never a legitimate negation, so banned even in
// the Concept prose.
const PROMO_FORBIDDEN: RegExp[] = [
  /recommand/i, /recommend/i, /plus safe/i, /\bsetup\b/i, /va rebondir/i,
  /signal d['’]achat/i, /signal de vente/i, /meilleur choix/i, /setup gagnant/i,
];

function rp(bundle: Record<string, unknown>): Record<string, any> {
  const v = (bundle as any).regimePanel;
  if (!v) throw new Error('regimePanel block missing');
  return v;
}

/** Collect every string under an object, excluding the concept BODY/notSay. */
function assertiveStrings(panel: Record<string, any>): string[] {
  const out: string[] = [];
  const walk = (node: unknown, path: string) => {
    if (typeof node === 'string') {
      // concept.<k>.body / .notSay are pedagogy — handled separately.
      if (/^concept\.[^.]+\.(body|notSay)$/.test(path)) return;
      out.push(node);
      return;
    }
    if (node && typeof node === 'object') {
      for (const [k, v] of Object.entries(node)) walk(v, path ? `${path}.${k}` : k);
    }
  };
  walk(panel, '');
  return out;
}

describe('RG-1 — Régime panel copy honesty', () => {
  it('the regimePanel block exists in all 9 locales', () => {
    for (const [name, b] of BUNDLES) expect(rp(b), name).toBeTruthy();
  });

  it('the assertive surface never contains a favourable/predictive term', () => {
    for (const [name, b] of BUNDLES) {
      for (const s of assertiveStrings(rp(b))) {
        for (const re of ASSERTIVE_FORBIDDEN) {
          expect(re.test(s), `${name}: « ${s} » matches ${re}`).toBe(false);
        }
      }
    }
  });

  it('the Concept prose never uses a purely promotional phrasing', () => {
    for (const [name, b] of BUNDLES) {
      const concept = rp(b).concept as Record<string, { body: string; notSay: string }>;
      for (const k of CONCEPT_KEYS) {
        const c = concept[k]!;
        const text = `${c.body}\n${c.notSay}`;
        for (const re of PROMO_FORBIDDEN) {
          expect(re.test(text), `${name}.concept.${k} matches ${re}`).toBe(false);
        }
      }
    }
  });

  it('every Concept carries a non-empty « ce que ça ne dit pas » block', () => {
    for (const [name, b] of BUNDLES) {
      const concept = rp(b).concept as Record<string, { title: string; body: string; notSay: string }>;
      for (const k of CONCEPT_KEYS) {
        const c = concept[k];
        expect(c, `${name}.concept.${k}`).toBeTruthy();
        expect((c?.title ?? '').length, `${name}.concept.${k}.title`).toBeGreaterThan(0);
        expect((c?.body ?? '').length, `${name}.concept.${k}.body`).toBeGreaterThan(0);
        expect((c?.notSay ?? '').trim().length, `${name}.concept.${k}.notSay`).toBeGreaterThan(0);
      }
    }
  });

  it('there is no combined/aggregate regime score key', () => {
    for (const [name, b] of BUNDLES) {
      const panel = rp(b);
      const keys = JSON.stringify(panel).toLowerCase();
      // No « note globale » / « score » VALUE is offered as a measure.
      expect(panel.tiles.score, `${name}: no score tile`).toBeUndefined();
      expect(keys.includes('"scoreglobal"'), name).toBe(false);
    }
  });
});
