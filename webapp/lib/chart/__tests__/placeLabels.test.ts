import { describe, it, expect } from 'vitest';
import {
  placeLabels,
  rectsOverlap,
  type LabelReq,
  type LRect,
} from '../zoneOverlayPrimitive';

/**
 * CHART-1 — the chart label de-collision. These pin the two reported defects:
 * labels piling on top of each other at the top-left, and the "EN DIRECT" badge
 * covering them. `placeLabels` is the pure placement core (no canvas).
 */
describe('placeLabels — chart label de-collision', () => {
  const pile = (n: number): LabelReq[] =>
    Array.from({ length: n }, () => ({ x: 2, y: 4, w: 40, h: 12 }));

  it('never returns two overlapping label rects (the reported left-edge pile)', () => {
    const { rects } = placeLabels(pile(8), [], 600);
    const placed = rects.filter((r): r is LRect => r !== null);
    expect(placed.length).toBe(8);
    for (let i = 0; i < placed.length; i += 1) {
      for (let j = i + 1; j < placed.length; j += 1) {
        expect(rectsOverlap(placed[i]!, placed[j]!), `label #${i} overlaps #${j}`).toBe(
          false,
        );
      }
    }
  });

  it('keeps every label clear of the reserved live-badge rect', () => {
    const badge: LRect = { l: 0, t: 0, r: 154, b: 26 };
    const { rects } = placeLabels(pile(4), [badge], 600);
    for (const r of rects) {
      if (r) expect(rectsOverlap(r, badge), 'a label sits under the EN DIRECT badge').toBe(false);
    }
  });

  it('groups overflow past the density cap instead of stacking illegibly', () => {
    const { rects, overflow } = placeLabels(pile(30), [], 300, 14);
    const drawn = rects.filter(Boolean).length;
    expect(drawn).toBeLessThanOrEqual(14);
    expect(overflow).toBeGreaterThan(0);
    // every input is accounted for — drawn or grouped, never silently lost.
    expect(drawn + overflow).toBe(30);
  });

  it('drops (groups) labels that would fall off the plot bottom', () => {
    const { rects } = placeLabels(pile(40), [], 80); // tiny plot
    for (const r of rects) {
      if (r) expect(r.b).toBeLessThanOrEqual(80 - 2 + 1e-6);
    }
  });

  it('never nudges a label above its anchor (stays at/below its zone)', () => {
    const items: LabelReq[] = [
      { x: 2, y: 100, w: 40, h: 12 },
      { x: 2, y: 100, w: 40, h: 12 },
    ];
    const { rects } = placeLabels(items, [], 600);
    for (const r of rects) if (r) expect(r.t).toBeGreaterThanOrEqual(100);
  });
});
