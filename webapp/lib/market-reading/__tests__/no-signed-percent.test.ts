import { describe, expect, it } from 'vitest';
import fr from '@/messages/fr.json';
import en from '@/messages/en.json';

// TF-1 point E1 — the reading surfaces state distances in WORDS (au-dessus /
// en dessous), NEVER with a +/- sign (the word already carries the direction).
// This guard fails if any reading/regime string re-introduces a signed
// percentage. Scoped to the reading namespaces so a legitimate marketing
// discount ("−20%") elsewhere is not caught.
const NAMESPACES = ['reading', 'regimePanel'] as const;

// A sign (+, - or the Unicode minus −) immediately before a number or an ICU
// placeholder, anywhere in a string that also mentions a percentage.
const SIGNED_BEFORE_TOKEN = /[+−-]\s*(\{[a-zA-Z]+\}|\d)/;

function walk(node: unknown, path: string, hits: string[]): void {
  if (typeof node === 'string') {
    if (node.includes('%') && SIGNED_BEFORE_TOKEN.test(node)) {
      hits.push(`${path} → ${node}`);
    }
    return;
  }
  if (node && typeof node === 'object') {
    for (const [k, v] of Object.entries(node)) walk(v, `${path}.${k}`, hits);
  }
}

describe('reading surfaces never render a signed percentage (E1)', () => {
  for (const [name, messages] of [['fr', fr], ['en', en]] as const) {
    it(`${name}: no signed percentage in the reading namespaces`, () => {
      const hits: string[] = [];
      for (const ns of NAMESPACES) {
        walk((messages as Record<string, unknown>)[ns], ns, hits);
      }
      expect(hits, `signed percentages found:\n${hits.join('\n')}`).toEqual([]);
    });
  }
});
