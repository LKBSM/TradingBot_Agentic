import { describe, expect, it } from 'vitest';
import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';

/**
 * MIA-3 invariant — ONE M.I.A conversation panel in the product.
 *
 * Every product surface (/app, /zones, /actualites, the shell) must render the
 * conversation through the single shared `MiaPanel`; none may render the chat
 * transcript (`<ChatMessage`) or the chat input (`<ChatInput`) itself. This test
 * FAILS the moment a second product panel appears — e.g. if the old /zones stub
 * comes back, or a page re-implements its own thread.
 *
 * Scope: the product component surfaces. The public landing slide-over
 * (`components/chat/ChatPanel.tsx`, used only by the marketing site) is a
 * separate surface and intentionally out of scope here.
 */

const PRODUCT_DIRS = [
  'components/app',
  'components/zones',
  'components/calendar',
  'components/shell',
];

function walk(dir: string): string[] {
  if (!existsSync(dir)) return [];
  const out: string[] = [];
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (name === '__tests__' || name.endsWith('.test.tsx') || name.endsWith('.test.ts')) {
      continue;
    }
    if (statSync(p).isDirectory()) out.push(...walk(p));
    else if (name.endsWith('.tsx')) out.push(p);
  }
  return out;
}

describe('MIA-3 — single conversation panel', () => {
  it('no product surface renders the transcript or input directly (only MiaPanel does)', () => {
    const offenders: string[] = [];
    for (const dir of PRODUCT_DIRS) {
      for (const file of walk(dir)) {
        const src = readFileSync(file, 'utf8');
        if (src.includes('<ChatInput') || src.includes('<ChatMessage')) {
          offenders.push(file);
        }
      }
    }
    expect(offenders, `these product files render their own chat panel: ${offenders.join(', ')}`).toEqual([]);
  });

  it('the old /zones local stub is gone', () => {
    expect(existsSync('components/zones/ZoneMiaPanel.tsx')).toBe(false);
  });

  it('MiaPanel is the one panel that renders the shared conversation', () => {
    const src = readFileSync('components/chat/MiaPanel.tsx', 'utf8');
    expect(src.includes('<ChatInput')).toBe(true);
    expect(src.includes('<ChatMessage')).toBe(true);
  });
});
