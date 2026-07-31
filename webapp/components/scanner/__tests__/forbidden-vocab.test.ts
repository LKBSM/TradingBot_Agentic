import { describe, expect, it } from 'vitest';
import fr from '@/messages/fr.json';
import en from '@/messages/en.json';

/**
 * SC-1 section 0: NO scanner-surface string, in EITHER language, may use the
 * forbidden score/quality/prediction vocabulary. Whole-word checks so legitimate
 * terms are not tripped (« range » must not match « rang », a « London » session
 * must not match anything). This scans the entire `scanner` namespace of both
 * locale bundles.
 */

function collectStrings(node: unknown, out: string[]): void {
  if (typeof node === 'string') out.push(node);
  else if (Array.isArray(node)) node.forEach((n) => collectStrings(n, out));
  else if (node && typeof node === 'object') Object.values(node).forEach((n) => collectStrings(n, out));
}

const FORBIDDEN_FR = [
  'setup', 'signal', 'opportunité', 'meilleur', 'plus sûr', 'recommandé',
  'probabilité', 'fort', 'idéal', 'qualité', 'score', 'rang', 'top',
];
const FORBIDDEN_EN = [
  'setup', 'signal', 'opportunity', 'best', 'safer', 'recommended',
  'probability', 'strong', 'ideal', 'quality', 'score', 'rank', 'top',
];

// A denial of the forbidden thing is not a use of it: « sans score », « no
// ranking », « pas de classement » are exactly the promises the product makes.
const NEG = '(?:sans|aucun|aucune|no|without|never|pas de|ni)';

function assertClean(bundle: { scanner?: unknown }, forbidden: string[], locale: string) {
  const strings: string[] = [];
  collectStrings(bundle.scanner, strings);
  expect(strings.length).toBeGreaterThan(0);
  for (const s of strings) {
    const hay = s.toLowerCase();
    for (const word of forbidden) {
      const esc = word.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      if (!new RegExp(`\\b${esc}\\b`).test(hay)) continue;
      // Present — but allowed if it appears only as a negation (an anti-promise).
      const asDenial = new RegExp(`${NEG}\\s+(?:\\w+\\s+)?${esc}`).test(hay);
      expect(asDenial, `[${locale}] forbidden « ${word} » used positively in: ${s}`).toBe(true);
    }
  }
}

describe('scanner i18n — no forbidden vocabulary (both languages)', () => {
  it('French scanner strings are clean', () => {
    assertClean(fr as { scanner?: unknown }, FORBIDDEN_FR, 'fr');
  });
  it('English scanner strings are clean', () => {
    assertClean(en as { scanner?: unknown }, FORBIDDEN_EN, 'en');
  });
});
