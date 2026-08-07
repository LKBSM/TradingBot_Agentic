/**
 * VZ-1 — « Ce qu'il y a d'autre au même endroit ». Pure interval geometry over
 * data the page ALREADY holds (the current reading's own zones + liquidity pools,
 * and the sibling-timeframe zones fetched once per combo). NO per-card request,
 * NO recompute of detection. Every fact is phrased as containment ("à l'intérieur
 * de" / "englobe" / "au même niveau"), NEVER as confluence/reinforcement/quality.
 *
 * The word "chevauche"/"overlap" is deliberately absent from this module and its
 * i18n (mission §0 vocabulary).
 */

import type { Direction } from '@/types/market-reading';
import type { LiquidityPool } from '@/types/market-reading';
import type { SiblingZone } from './lifecycle';
import type { ZoneKind, ZoneLifecycle } from './lifecycle';

/**
 * A containment fact:
 *   · `inner`      — a zone sits INSIDE this one (« à l'intérieur de celle-ci »).
 *   · `outer`      — a zone ENGLOBES this one (« l'englobe »).
 *   · `same_level` — bands meet without containment (« au même niveau »).
 *   · `liquidity`  — a resting-liquidity pocket nearby.
 * `timeframe` is null for a same-unit zone, else the other unit's name.
 */
export type ConfluenceRelation = 'inner' | 'outer' | 'same_level' | 'liquidity';

export interface ConfluenceFact {
  relation: ConfluenceRelation;
  // Zone facts (inner / outer / same_level):
  kind?: ZoneKind;
  direction?: Direction | null;
  timeframe?: string | null;
  levelLow?: number;
  levelHigh?: number;
  status?: string;
  // Liquidity facts:
  liquiditySide?: 'bsl' | 'ssl';
  liquidityStatus?: string;
  level?: number;
  distance?: number;
  distanceSide?: 'above' | 'below' | 'inside';
}

interface Candidate {
  kind: ZoneKind;
  direction: Direction | null;
  levelLow: number;
  levelHigh: number;
  status: string;
  timeframe: string | null; // null = same unit as the subject zone
  id: string;
}

function classifyZone(zone: ZoneLifecycle, c: Candidate): ConfluenceFact | null {
  const contains = zone.levelLow <= c.levelLow && zone.levelHigh >= c.levelHigh;
  const contained = c.levelLow <= zone.levelLow && c.levelHigh >= zone.levelHigh;
  const strictlyBigger = contained && (c.levelLow < zone.levelLow || c.levelHigh > zone.levelHigh);
  const strictlySmaller = contains && (c.levelLow > zone.levelLow || c.levelHigh < zone.levelHigh);
  let relation: ConfluenceRelation;
  if (strictlyBigger) relation = 'outer';
  else if (strictlySmaller) relation = 'inner';
  else if (c.levelLow < zone.levelHigh && zone.levelLow < c.levelHigh) relation = 'same_level';
  else return null; // no vertical overlap at all
  return {
    relation,
    kind: c.kind,
    direction: c.direction,
    timeframe: c.timeframe,
    levelLow: c.levelLow,
    levelHigh: c.levelHigh,
    status: c.status,
  };
}

/**
 * Build the containment + liquidity facts for one zone.
 *
 * @param sameTf   the OTHER live zones of the SAME reading (nesting, timeframe=null)
 * @param siblings zones of the other timeframes (each carries its timeframe)
 * @param pools    the reading's own liquidity pockets
 */
export function buildConfluence(
  zone: ZoneLifecycle,
  sameTf: readonly ZoneLifecycle[],
  siblings: readonly SiblingZone[],
  pools: readonly LiquidityPool[],
): ConfluenceFact[] {
  const candidates: Candidate[] = [
    ...sameTf
      .filter((z) => z.id !== zone.id)
      .map((z) => ({
        kind: z.kind,
        direction: z.direction,
        levelLow: z.levelLow,
        levelHigh: z.levelHigh,
        status: z.status,
        timeframe: null,
        id: z.id,
      })),
    ...siblings.map((s) => ({
      kind: s.kind,
      direction: s.direction,
      levelLow: s.levelLow,
      levelHigh: s.levelHigh,
      status: 'active',
      timeframe: s.timeframe,
      id: `${s.timeframe}-${s.id}`,
    })),
  ];

  const zoneFacts = candidates
    .map((c) => classifyZone(zone, c))
    .filter((f): f is ConfluenceFact => f !== null);

  // Order: inner, outer, same_level (containment before mere adjacency).
  const rank: Record<ConfluenceRelation, number> = {
    inner: 0,
    outer: 1,
    same_level: 2,
    liquidity: 3,
  };
  zoneFacts.sort((a, b) => rank[a.relation] - rank[b.relation]);

  // Liquidity: pockets near an edge (within twice the zone height), nearest first,
  // capped — a resting-liquidity FACT, never a target.
  const height = Math.max(zone.levelHigh - zone.levelLow, 0);
  const reach = Math.max(height * 2, 0);
  const liq: (ConfluenceFact & { _d: number })[] = [];
  for (const p of pools) {
    let side: 'above' | 'below' | 'inside';
    let distance: number;
    if (p.level > zone.levelHigh) {
      side = 'above';
      distance = p.level - zone.levelHigh;
    } else if (p.level < zone.levelLow) {
      side = 'below';
      distance = zone.levelLow - p.level;
    } else {
      side = 'inside';
      distance = 0;
    }
    if (distance > reach) continue;
    liq.push({
      relation: 'liquidity',
      liquiditySide: p.side,
      liquidityStatus: p.status,
      level: p.level,
      distance,
      distanceSide: side,
      _d: distance,
    });
  }
  liq.sort((a, b) => a._d - b._d);
  const liqFacts = liq.slice(0, 2).map(({ _d, ...f }) => f);

  return [...zoneFacts, ...liqFacts];
}
