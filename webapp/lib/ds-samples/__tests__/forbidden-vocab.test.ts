import { describe, expect, it } from 'vitest';
import { SAMPLE_READINGS } from '../readings';
import { SAMPLE_SCAN_RESPONSE, SAMPLE_SCAN_NO_MATCH } from '../scanner';
import { SAMPLE_CHAT_TURNS } from '../chat';

/**
 * DS-1 §3 — the example DATA must respect the descriptive line: no word from the
 * forbidden list (score / prediction / advice / causation), in FR or EN. Whole-
 * word checks so legitimate terms are not tripped. A denial ("sans biais", "no
 * ranking") is allowed — it is the very promise the product makes.
 */

function collectStrings(node: unknown, out: string[]): void {
  if (typeof node === 'string') out.push(node);
  else if (Array.isArray(node)) node.forEach((n) => collectStrings(n, out));
  else if (node && typeof node === 'object') Object.values(node).forEach((n) => collectStrings(n, out));
}

// FR list (mission) + causal verbs; EN mirror for completeness.
const FORBIDDEN_FR = [
  'va', 'cible', 'biais', 'setup', 'signal', 'opportunité', 'meilleur', 'recommandé',
  'probabilité', 'classement', 'provoque', 'entraîne', 'entraine', 'pousse', 'déclenche',
];
const FORBIDDEN_EN = [
  'target', 'bias', 'setup', 'signal', 'opportunity', 'best', 'recommended', 'probability',
  'ranking', 'causes', 'triggers',
];
const NEG = '(?:sans|aucun|aucune|no|without|never|pas de|ni)';

function assertClean(strings: string[], forbidden: string[], label: string) {
  expect(strings.length).toBeGreaterThan(0);
  for (const s of strings) {
    const hay = s.toLowerCase();
    for (const word of forbidden) {
      const esc = word.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      if (!new RegExp(`\\b${esc}\\b`).test(hay)) continue;
      const asDenial = new RegExp(`${NEG}\\s+(?:\\w+\\s+)?${esc}`).test(hay);
      expect(asDenial, `[${label}] forbidden « ${word} » used in: ${s}`).toBe(true);
    }
  }
}

describe('DS-1 sample data — no forbidden vocabulary', () => {
  it('collects every human-readable string from the samples', () => {
    const strings: string[] = [];
    collectStrings(SAMPLE_READINGS, strings);
    collectStrings(SAMPLE_SCAN_RESPONSE, strings);
    collectStrings(SAMPLE_SCAN_NO_MATCH, strings);
    collectStrings(SAMPLE_CHAT_TURNS, strings);
    assertClean(strings, FORBIDDEN_FR, 'fr');
    assertClean(strings, FORBIDDEN_EN, 'en');
  });
});
