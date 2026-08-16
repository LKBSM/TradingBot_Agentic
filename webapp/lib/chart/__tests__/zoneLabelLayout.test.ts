import { describe, expect, it } from 'vitest';
import {
  layoutZoneLabels,
  type LabelCandidate,
  type LabelLayout,
} from '../zoneLabelLayout';

// Deterministic, canvas-free text measurer (≈6px per char) so the placement is
// fully reproducible in the test environment.
const measure = (t: string) => t.length * 6;
const clusterText = (n: number) => `${n} zones`;
const opts = { measure, clusterText };

interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}
function overlaps(a: Rect, b: Rect): boolean {
  return a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y;
}
/** Every drawn rectangle — single labels AND cluster pills. */
function drawnRects(layout: LabelLayout): Rect[] {
  return [
    ...layout.placed.map((p) => ({ x: p.x, y: p.y, w: p.w, h: p.h })),
    ...layout.clusters.map((c) => ({ x: c.x, y: c.y, w: c.w, h: c.h })),
  ];
}
function assertNoOverlaps(layout: LabelLayout): void {
  const rects = drawnRects(layout);
  for (let i = 0; i < rects.length; i += 1) {
    for (let j = i + 1; j < rects.length; j += 1) {
      expect(
        overlaps(rects[i]!, rects[j]!),
        `rect ${i} overlaps rect ${j}`,
      ).toBe(false);
    }
  }
}

const cand = (
  id: string,
  x: number,
  y: number,
  over: Partial<LabelCandidate> = {},
): LabelCandidate => ({
  id,
  x,
  y,
  w: 24,
  h: 13,
  priority: 1,
  tooltip: 'OB',
  ...over,
});

const PLOT = { width: 1000, height: 500 };

describe('layoutZoneLabels', () => {
  it('places well-separated labels as-is, with no clusters', () => {
    const layout = layoutZoneLabels(
      [cand('a', 20, 40), cand('b', 400, 200), cand('c', 800, 350)],
      PLOT,
      opts,
    );
    expect(layout.placed).toHaveLength(3);
    expect(layout.clusters).toHaveLength(0);
    assertNoOverlaps(layout);
  });

  it('collapses labels sharing a cell into ONE « N zones » cluster', () => {
    const layout = layoutZoneLabels(
      [
        cand('a', 300, 100, { tooltip: 'FVG' }),
        cand('b', 305, 102, { tooltip: 'FVG' }),
        cand('c', 302, 104, { tooltip: 'OB · touché' }),
      ],
      PLOT,
      opts,
    );
    expect(layout.clusters).toHaveLength(1);
    expect(layout.clusters[0]!.count).toBe(3);
    // Detail preserved for the hover tooltip (unique lines).
    expect(layout.clusters[0]!.tooltip).toContain('FVG');
    expect(layout.clusters[0]!.tooltip).toContain('OB · touché');
    expect(layout.placed).toHaveLength(0);
    assertNoOverlaps(layout);
  });

  it('never overlaps two drawings across three zoom densities', () => {
    // Model three zoom levels by the horizontal pixel spacing between formation
    // anchors: zoomed-in (wide), mid, and dezoomed (tight → heavy overlap).
    for (const spacing of [60, 18, 4]) {
      const candidates: LabelCandidate[] = Array.from({ length: 40 }, (_, i) =>
        cand(`z${i}`, 10 + i * spacing, 60 + (i % 5) * 8, {
          priority: i % 3,
          tooltip: i % 2 ? 'OB' : 'FVG',
        }),
      );
      const layout = layoutZoneLabels(candidates, PLOT, opts);
      assertNoOverlaps(layout);
      // Something is always drawn (labels and/or clusters), never a silent blank.
      expect(layout.placed.length + layout.clusters.length).toBeGreaterThan(0);
    }
  });

  it('keeps every drawing clear of a reserved rectangle (the status badge)', () => {
    const reserved = [{ x: 0, y: 0, w: 172, h: 26 }];
    const candidates: LabelCandidate[] = Array.from({ length: 12 }, (_, i) =>
      cand(`b${i}`, 4 + i * 3, 2 + (i % 3) * 4),
    );
    const layout = layoutZoneLabels(candidates, PLOT, { ...opts, reserved });
    for (const r of drawnRects(layout)) {
      expect(overlaps(r, reserved[0]!)).toBe(false);
    }
    assertNoOverlaps(layout);
  });

  it('caps the number of drawn labels (density reduction, not a wall of text)', () => {
    const candidates: LabelCandidate[] = Array.from({ length: 200 }, (_, i) =>
      // Spread across the plot so most are singles rather than clusters.
      cand(`m${i}`, (i * 47) % 990, (i * 31) % 490),
    );
    const layout = layoutZoneLabels(candidates, PLOT, { ...opts, maxLabels: 15 });
    expect(layout.placed.length + layout.clusters.length).toBeLessThanOrEqual(15);
    assertNoOverlaps(layout);
  });

  it('drops candidates whose anchor is off-plot', () => {
    const layout = layoutZoneLabels(
      [cand('on', 100, 100), cand('off', 1200, 100), cand('left', -50, 100)],
      PLOT,
      opts,
    );
    const ids = layout.placed.map((p) => p.id);
    expect(ids).toContain('on');
    expect(ids).not.toContain('off');
    expect(ids).not.toContain('left');
  });
});
