import { fireEvent, render as rtlRender } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { NextIntlClientProvider } from 'next-intl';
import type { LiquidityPool } from '@/types/market-reading';
import type { SiblingZone, ZoneLifecycle } from '@/lib/zones/lifecycle';
import { ZoneLifecycleCard } from '../ZoneLifecycleCard';
import messages from '@/messages/fr.json';

function zl(over: Partial<ZoneLifecycle> = {}): ZoneLifecycle {
  return {
    id: 'z',
    kind: 'ob',
    direction: 'bullish',
    levelHigh: 2380,
    levelLow: 2370,
    importance: 'high',
    status: 'active',
    createdAt: '2026-05-26T08:00:00+00:00',
    tested: false,
    mitigatedAt: null,
    fillLevel: null,
    isActive: true,
    isMitigated: false,
    contacts: [],
    origin: null,
    session: null,
    ...over,
  };
}

const noop = () => {};

function renderCard(
  onNavigateToZone: (id: string, tf: string | null) => void,
  opts: {
    sameTf?: ZoneLifecycle[];
    siblings?: SiblingZone[];
    pools?: LiquidityPool[];
  } = {},
) {
  const subject = zl({ id: 'subject', levelLow: 2370, levelHigh: 2380 });
  const result = rtlRender(
    <NextIntlClientProvider locale="fr" messages={messages}>
      <ZoneLifecycleCard
        zone={subject}
        instrument="XAUUSD"
        referencePrice={2400}
        candles={null}
        sameTfZones={opts.sameTf ?? [subject]}
        siblingZones={opts.siblings ?? []}
        liquidityPools={opts.pools ?? []}
        isHidden={false}
        onToggleHide={noop}
        onShowOnChart={noop}
        onSelect={noop}
        onNavigateToZone={onNavigateToZone}
      />
    </NextIntlClientProvider>,
  );
  // The confluence block is deferred behind the « Détails » toggle.
  fireEvent.click(result.container.querySelector('button.zdeth')!);
  return result;
}

describe('ZoneLifecycleCard — « même endroit » navigation (id lock)', () => {
  it('each item opens ITS OWN zone by real engine id — never a price neighbour', () => {
    const nav = vi.fn();
    // Two same-TF nested zones with ALMOST IDENTICAL bands but distinct ids, plus
    // a sibling on H1. If navigation matched on price, the two near-twins would be
    // indistinguishable — the id lock forbids that.
    const twinA = zl({ id: 'zone-A', levelLow: 2373.0, levelHigh: 2376.0 });
    const twinB = zl({ id: 'zone-B', levelLow: 2373.1, levelHigh: 2375.9 });
    const subject = zl({ id: 'subject', levelLow: 2370, levelHigh: 2380 });
    const sib: SiblingZone = {
      id: 'sib-h1', kind: 'ob', direction: 'bearish', levelLow: 2365, levelHigh: 2385, timeframe: 'H1',
    };
    const { container } = renderCard(nav, { sameTf: [subject, twinA, twinB], siblings: [sib] });

    const buttons = Array.from(container.querySelectorAll<HTMLButtonElement>('.zconf button.clnav'));
    // Three ZONE facts → three navigable items (inner twinA, inner twinB, outer sib).
    expect(buttons).toHaveLength(3);

    const fired: Array<[string, string | null]> = [];
    for (const b of buttons) {
      nav.mockClear();
      fireEvent.click(b);
      expect(nav).toHaveBeenCalledTimes(1);
      fired.push([nav.mock.calls[0]![0], nav.mock.calls[0]![1]]);
    }
    // Every real id is reached exactly once, each with its correct timeframe
    // (null = same unit, 'H1' = sibling). No composite, no fabricated id.
    expect(new Set(fired.map(([id]) => id))).toEqual(new Set(['zone-A', 'zone-B', 'sib-h1']));
    expect(fired).toContainEqual(['sib-h1', 'H1']);
    expect(fired).toContainEqual(['zone-A', null]);
    expect(fired).toContainEqual(['zone-B', null]);
  });

  it('a liquidity pocket is NOT clickable (a pool is not a navigable zone)', () => {
    const nav = vi.fn();
    const pool: LiquidityPool = {
      id: 'liq', side: 'bsl', kind: 'equal_highs', level: 2385, touches: 2,
      is_external: true, status: 'intact', created_at: 'x', user_flagged: false,
    };
    const { container } = renderCard(nav, { pools: [pool] });
    // The liquidity fact renders as plain text, never a button.
    expect(container.querySelectorAll('.zconf button.clnav')).toHaveLength(0);
    const lines = container.querySelectorAll('.zconf .cl');
    expect(lines.length).toBeGreaterThan(0);
  });
});
