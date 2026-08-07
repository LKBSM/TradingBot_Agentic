import { describe, expect, it } from 'vitest';
import fr from '@/messages/fr.json';
import en from '@/messages/en.json';

/**
 * SC-2 §5 — the conversational surface (`scannerChat` namespace) must carry NO
 * score/quality/ranking/prediction vocabulary in EITHER language. M.I.A is a
 * translator, never a judge: her chrome never speaks of "best", "top", "score"…
 * The refusal states deny ranking/prediction explicitly, but they do it WITHOUT
 * ever using the forbidden words positively (they were worded around them). This
 * mirrors the SC-1 guard on the `scanner` namespace, extended to `scannerChat`.
 */

function collectStrings(node: unknown, out: string[]): void {
  if (typeof node === 'string') out.push(node);
  else if (Array.isArray(node)) node.forEach((n) => collectStrings(n, out));
  else if (node && typeof node === 'object') Object.values(node).forEach((n) => collectStrings(n, out));
}

const FORBIDDEN_FR = [
  'setup', 'signal', 'opportunité', 'meilleur', 'plus sûr', 'recommandé',
  'probabilité', 'idéal', 'qualité', 'score', 'rang', 'top',
];
const FORBIDDEN_EN = [
  'setup', 'signal', 'opportunity', 'best', 'safer', 'recommended',
  'probability', 'ideal', 'quality', 'score', 'rank', 'top',
];

const NEG = '(?:sans|aucun|aucune|no|without|never|pas de|ni)';

function assertClean(bundle: { scannerChat?: unknown }, forbidden: string[], locale: string) {
  const strings: string[] = [];
  collectStrings(bundle.scannerChat, strings);
  expect(strings.length).toBeGreaterThan(0);
  for (const s of strings) {
    const hay = s.toLowerCase();
    for (const word of forbidden) {
      const esc = word.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      if (!new RegExp(`\\b${esc}\\b`).test(hay)) continue;
      const asDenial = new RegExp(`${NEG}\\s+(?:\\w+\\s+)?${esc}`).test(hay);
      expect(asDenial, `[${locale}] forbidden « ${word} » used positively in: ${s}`).toBe(true);
    }
  }
}

describe('scannerChat i18n — no forbidden vocabulary (both languages)', () => {
  it('French conversational strings are clean', () => {
    assertClean(fr as { scannerChat?: unknown }, FORBIDDEN_FR, 'fr');
  });
  it('English conversational strings are clean', () => {
    assertClean(en as { scannerChat?: unknown }, FORBIDDEN_EN, 'en');
  });
});
