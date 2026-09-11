/**
 * @vitest-environment node
 *
 * Pure data guard over the message catalogues — no DOM is needed, so this
 * file skips the jsdom setup entirely (which dominates its runtime and, on a
 * loaded machine, was starving the worker past its startup timeout).
 */
import { describe, expect, it } from 'vitest';
import ar from '@/messages/ar.json';
import de from '@/messages/de.json';
import en from '@/messages/en.json';
import es from '@/messages/es.json';
import fr from '@/messages/fr.json';
import itMessages from '@/messages/it.json';
import nl from '@/messages/nl.json';
import pl from '@/messages/pl.json';
import pt from '@/messages/pt.json';

/**
 * VZ-4 §3 — the zone sheet's copy carries NO value judgement on a zone.
 *
 * The rule (mission §0, non-negotiable): a zone is never « stable / instable /
 * solide / fiable / robuste / respectée / validée / de qualité », in any locale,
 * on the page or in a M.I.A starter question. SMC has no consensus on those
 * notions, so the product reports dated, numbered facts — contacts, comblement,
 * dates — and lets the reader conclude.
 *
 * This guard walks EVERY string the new `zones.detail` block adds, in all nine
 * locales, and fails on the banned vocabulary. It also pins the two engine words
 * that are NOT judgements and must keep working: a zone's own factual status
 * (« comblée », « mitigé ») is engine output, not an opinion — the ban targets
 * the copy we write, so it is scoped to `zones.detail`.
 */

type Dict = Record<string, unknown>;

const LOCALES: Record<string, Dict> = { ar, de, en, es, fr, it: itMessages, nl, pl, pt } as unknown as Record<
  string,
  Dict
>;

/** Locales the mission names explicitly; the rest are covered by the sweep too. */
const REQUIRED = ['fr', 'en', 'es'];

/**
 * Banned stems. Deliberately stems, not whole words, so « fiabilité », « solidité »
 * or « respectée » are caught as well. Accent-insensitive via `normalize`.
 */
const BANNED_FR = [
  'stable',
  'instable',
  'solide',
  'solidit',
  'fiable',
  'fiabilit',
  'robuste',
  'respectee',
  'respecte par',
  'validee',
  'de qualite',
  'zone forte',
  'zone faible',
];
const BANNED_EN = [
  'stable',
  'unstable',
  'solid',
  'reliable',
  'reliability',
  'robust',
  'respected',
  'validated',
  'quality zone',
  'strong zone',
  'weak zone',
];

function normalize(s: string): string {
  return s
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .toLowerCase();
}

/** Flatten every string under a subtree, keyed by dotted path. */
function strings(node: unknown, path = ''): Array<[string, string]> {
  if (typeof node === 'string') return [[path, node]];
  if (node && typeof node === 'object') {
    return Object.entries(node as Dict).flatMap(([k, v]) =>
      strings(v, path ? `${path}.${k}` : k),
    );
  }
  return [];
}

function detailStrings(locale: string): Array<[string, string]> {
  const zones = (LOCALES[locale] as Dict).zones as Dict;
  expect(zones, `zones missing in ${locale}`).toBeTruthy();
  const detail = zones.detail;
  expect(detail, `zones.detail missing in ${locale}`).toBeTruthy();
  return strings(detail, 'zones.detail');
}

describe('VZ-4 — the zone sheet never passes judgement on a zone', () => {
  it('ships the whole `zones.detail` block in the nine locales', () => {
    for (const locale of Object.keys(LOCALES)) {
      const entries = detailStrings(locale);
      expect(entries.length, `${locale} has too few detail strings`).toBeGreaterThan(15);
      for (const [path, value] of entries) {
        expect(value.trim(), `${locale} · ${path} is empty`).not.toBe('');
      }
    }
  });

  it('carries none of the banned judgement vocabulary (fr / en / es required)', () => {
    for (const locale of REQUIRED) {
      expect(Object.keys(LOCALES)).toContain(locale);
    }
    const banned = [...new Set([...BANNED_FR, ...BANNED_EN])];
    for (const locale of Object.keys(LOCALES)) {
      for (const [path, value] of detailStrings(locale)) {
        const hay = normalize(value);
        for (const word of banned) {
          expect(
            hay.includes(normalize(word)),
            `${locale} · ${path} contains the banned word « ${word} »: ${value}`,
          ).toBe(false);
        }
      }
    }
  });

  it('ships three factual chips plus ONE deliberate refusal probe', () => {
    // The fourth chip asks for a forecast on purpose: Couche 1's `prediction`
    // bucket intercepts it deterministically and answers with the dedicated
    // forecast refusal, so the chip demonstrates the product's honesty instead
    // of inviting a forecast. tests/test_vz4_zone_refusal.py locks that.
    const predictive = [
      'rebondir',
      'bounce',
      'va monter',
      'va descendre',
      'will go up',
      'will rise',
      'will drop',
      'tu penses',
      'do you think',
      'forecast',
    ];
    const FACTUAL = ['formed', 'nested', 'tested'];
    for (const locale of Object.keys(LOCALES)) {
      const zones = (LOCALES[locale] as Dict).zones as Dict;
      const detail = zones.detail as Dict;
      const starters = detail.starters as Record<string, string>;
      expect(Object.keys(starters).sort(), `${locale} starters`).toEqual([
        'formed',
        'nested',
        'probe',
        'tested',
      ]);

      // The three factual chips never ask for a forecast.
      for (const key of FACTUAL) {
        const hay = normalize(starters[key]!);
        for (const word of predictive) {
          expect(
            hay.includes(normalize(word)),
            `${locale} · starters.${key} asks for a prediction («${word}»): ${starters[key]}`,
          ).toBe(false);
        }
      }

      // The probe DOES ask for one — that is its whole job. If this ever stops
      // being true, the refusal it exercises is no longer being exercised.
      const probe = normalize(starters.probe!);
      expect(
        predictive.some((w) => probe.includes(normalize(w))),
        `${locale} · starters.probe no longer asks for a forecast: ${starters.probe}`,
      ).toBe(true);
    }
  });

  it('fr keeps real French while the other locales fall back to the en wording', () => {
    const frDetail = Object.fromEntries(detailStrings('fr'));
    const enDetail = Object.fromEntries(detailStrings('en'));
    expect(frDetail['zones.detail.state.heading']).toBe('État actuel');
    expect(enDetail['zones.detail.state.heading']).toBe('Current state');
    expect(frDetail['zones.detail.nested.heading']).toBe("Zones à l'intérieur");
    // Same key set everywhere — no locale silently missing a sentence.
    for (const locale of Object.keys(LOCALES)) {
      expect(Object.keys(Object.fromEntries(detailStrings(locale))).sort()).toEqual(
        Object.keys(frDetail).sort(),
      );
    }
  });
});
