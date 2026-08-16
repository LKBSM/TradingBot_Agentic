/**
 * CHART-2 — zone-label placement: collision avoidance + clustering + density cap.
 *
 * The overlay used to draw every zone's label at its own box corner, blind to the
 * others — so stacked FVGs / an OB-touché over a BOS piled illegible labels at the
 * same spot, and dezooming (more history, more zones at one x) made it worse. This
 * pure function takes the per-frame PIXEL anchors the primitive already computes
 * and resolves them into a legible set:
 *
 *   1. CLUSTER  — labels whose anchors fall in the same small pixel cell collapse
 *      into ONE « N zones » pill (detail on hover), instead of overprinting.
 *   2. DE-COLLIDE — the remaining single labels are nudged vertically so no two
 *      rectangles overlap.
 *   3. CAP      — at most `maxLabels` are drawn; the lowest-priority overflow is
 *      dropped (its box still renders, just unlabelled) so a very dense window
 *      stays readable rather than a wall of text.
 *
 * It is deliberately free of any canvas / lightweight-charts dependency (text
 * widths are injected via `measure`) so the non-overlap guarantee is unit-testable
 * at several zoom densities. It NEVER changes detection — labels are pure chrome
 * over already-detected zones.
 */

export interface LabelCandidate {
  id: string;
  /** Desired top-left corner, in pixels (already clamped to the box's first visible px). */
  x: number;
  y: number;
  /** Measured label width + height, in pixels. */
  w: number;
  h: number;
  /**
   * Placement priority — higher stays. The caller sets it (active zones over
   * tested, recent over old) so that when space runs out the most relevant labels
   * survive and the rest fold into clusters / drop.
   */
  priority: number;
  /** One line for the cluster hover tooltip (e.g. « OB · touché »). */
  tooltip: string;
}

export interface PlacedLabel {
  id: string;
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface PlacedCluster {
  /** Stable key for the frame (tooltip registration + hit id). */
  key: string;
  x: number;
  y: number;
  w: number;
  h: number;
  count: number;
  ids: string[];
  /** Multi-line detail for the hover tooltip. */
  tooltip: string;
}

export interface LabelLayout {
  placed: PlacedLabel[];
  clusters: PlacedCluster[];
}

export interface LabelLayoutOptions {
  /** Pixel cell width for spatial clustering (default 84). */
  clusterX?: number;
  /** Pixel cell height for spatial clustering (default 22). */
  clusterY?: number;
  /** Group size (labels sharing a cell) at/above which they collapse to a cluster (default 2). */
  clusterMin?: number;
  /** Max labels + cluster pills drawn in one frame (default 22). */
  maxLabels?: number;
  /** Vertical gap kept between stacked labels, px (default 2). */
  gap?: number;
  /** Uniform label height when the caller doesn't set per-candidate `h`. */
  labelHeight?: number;
  /** Text-width measurer (px). Injected so this module needs no canvas. */
  measure: (text: string) => number;
  /** Cluster pill text from a member count (e.g. `n => \`${n} zones\``). */
  clusterText: (count: number) => string;
  /**
   * Rectangles labels must avoid (px) — e.g. the top-left status badge plane, so a
   * label is never drawn under it. Seeded into the occupied set before placement.
   */
  reserved?: readonly Rect[];
}

interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

function overlaps(a: Rect, b: Rect): boolean {
  return a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y;
}

/** Max vertical nudges (each way) attempted before a label is dropped. */
const MAX_NUDGES = 6;
/** Horizontal padding used to size a cluster pill around its measured text. */
const CLUSTER_PAD_X = 6;

/**
 * Resolve label candidates into a non-overlapping set of single labels + cluster
 * pills. Input order does not matter (candidates are sorted by priority here).
 */
export function layoutZoneLabels(
  candidates: readonly LabelCandidate[],
  plot: { width: number; height: number },
  options: LabelLayoutOptions,
): LabelLayout {
  const clusterX = options.clusterX ?? 84;
  const clusterY = options.clusterY ?? 22;
  const clusterMin = options.clusterMin ?? 2;
  const maxLabels = options.maxLabels ?? 22;
  const gap = options.gap ?? 2;
  const labelH = options.labelHeight ?? 13;
  const { measure, clusterText } = options;

  // Keep only candidates whose anchor is on-plot horizontally (an off-screen box
  // must not sprout a label pinned to the edge).
  const onPlot = candidates.filter((c) => c.x < plot.width && c.x + c.w > 0);

  // 1) Spatial clustering by pixel cell.
  const cells = new Map<string, LabelCandidate[]>();
  for (const c of onPlot) {
    const key = `${Math.floor(c.x / clusterX)}:${Math.floor(c.y / clusterY)}`;
    const bucket = cells.get(key);
    if (bucket) bucket.push(c);
    else cells.set(key, [c]);
  }

  interface Item {
    kind: 'single' | 'cluster';
    priority: number;
    rect: Rect;
    // single
    id?: string;
    // cluster
    key?: string;
    ids?: string[];
    tooltip?: string;
    count?: number;
  }

  const items: Item[] = [];
  for (const [key, group] of cells) {
    if (group.length >= clusterMin) {
      const x = Math.min(...group.map((g) => g.x));
      const y = Math.min(...group.map((g) => g.y));
      const text = clusterText(group.length);
      const w = measure(text) + CLUSTER_PAD_X;
      const priority = Math.max(...group.map((g) => g.priority));
      // Unique, order-stable tooltip lines.
      const seen = new Set<string>();
      const lines: string[] = [];
      for (const g of group) {
        if (!seen.has(g.tooltip)) {
          seen.add(g.tooltip);
          lines.push(g.tooltip);
        }
      }
      items.push({
        kind: 'cluster',
        priority,
        rect: { x, y, w, h: labelH },
        key: `cluster:${key}`,
        ids: group.map((g) => g.id),
        tooltip: lines.join(' · '),
        count: group.length,
      });
    } else {
      const c = group[0]!;
      items.push({
        kind: 'single',
        priority: c.priority,
        rect: { x: c.x, y: c.y, w: c.w, h: c.h || labelH },
        id: c.id,
      });
    }
  }

  // 2) De-collide, highest priority first (ties: higher on the plot first).
  items.sort((a, b) => b.priority - a.priority || a.rect.y - b.rect.y);

  const occupied: Rect[] = options.reserved ? [...options.reserved] : [];
  const placed: PlacedLabel[] = [];
  const clusters: PlacedCluster[] = [];

  const fits = (r: Rect): boolean =>
    r.y >= 0 &&
    r.y + r.h <= plot.height &&
    !occupied.some((o) => overlaps(r, o));

  for (const item of items) {
    if (placed.length + clusters.length >= maxLabels) break;
    const step = item.rect.h + gap;
    let chosen: Rect | null = null;
    // Try the desired spot, then nudge down, then up.
    const base = item.rect;
    if (fits(base)) chosen = base;
    for (let n = 1; !chosen && n <= MAX_NUDGES; n += 1) {
      const down = { ...base, y: base.y + step * n };
      if (fits(down)) {
        chosen = down;
        break;
      }
      const up = { ...base, y: base.y - step * n };
      if (fits(up)) {
        chosen = up;
        break;
      }
    }
    if (!chosen) continue; // no room → drop this label (box still drawn)
    occupied.push(chosen);
    if (item.kind === 'single') {
      placed.push({ id: item.id!, x: chosen.x, y: chosen.y, w: chosen.w, h: chosen.h });
    } else {
      clusters.push({
        key: item.key!,
        x: chosen.x,
        y: chosen.y,
        w: chosen.w,
        h: chosen.h,
        count: item.count!,
        ids: item.ids!,
        tooltip: item.tooltip!,
      });
    }
  }

  return { placed, clusters };
}
