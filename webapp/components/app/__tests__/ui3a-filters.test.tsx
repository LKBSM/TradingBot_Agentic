import { render, fireEvent } from '@/components/test-utils';
import { afterEach, describe, expect, it, vi } from 'vitest';
import fr from '@/messages/fr.json';
import { LiquidityCard } from '../LiquidityCard';
import { StructureCard } from '../StructureCard';
import type { MarketReadingStructure } from '@/types/market-reading';

afterEach(() => vi.restoreAllMocks());

const noop = () => {};

// Liquidity pools: 2 BSL / 2 SSL, states intact×2 · swept×1 · broken×1.
const LIQ = {
  order_blocks: [],
  fair_value_gaps: [],
  liquidity_pools: [
    { id: 'p1', side: 'bsl', kind: 'range_high', level: 2420, touches: 1, is_external: true, status: 'intact', created_at: '2026-07-26T08:00:00Z', user_flagged: false },
    { id: 'p2', side: 'ssl', kind: 'range_low', level: 2360, touches: 1, is_external: true, status: 'intact', created_at: '2026-07-26T08:00:00Z', user_flagged: false },
    { id: 'p3', side: 'bsl', kind: 'equal_highs', level: 2410, touches: 2, is_external: false, status: 'swept', created_at: '2026-07-26T09:00:00Z', swept_at: '2026-07-26T12:00:00Z', user_flagged: false },
    { id: 'p4', side: 'ssl', kind: 'equal_lows', level: 2370, touches: 2, is_external: false, status: 'broken', created_at: '2026-07-26T09:00:00Z', broken_at: '2026-07-26T13:00:00Z', user_flagged: false },
  ],
} as unknown as MarketReadingStructure;

// Zones: OB active · OB mitig · FVG tested · FVG active.
const STRUCT = {
  order_blocks: [
    { id: 'ob1', direction: 'bearish', level_high: 2401, level_low: 2398, importance: 'high', status: 'active', created_at: '2026-07-26T11:00:00Z', tested: false, user_flagged: false },
    { id: 'ob2', direction: 'bullish', level_high: 2385, level_low: 2383, importance: 'medium', status: 'mitigated', created_at: '2026-07-26T09:00:00Z', tested: true, mitigated_at: '2026-07-26T12:45:00Z', user_flagged: false },
  ],
  fair_value_gaps: [
    { id: 'fvg1', direction: 'bullish', level_high: 2394, level_low: 2389, status: 'partially_filled', created_at: '2026-07-26T10:15:00Z', tested: true, fill_level: 2391.9, user_flagged: false },
    { id: 'fvg2', direction: 'bearish', level_high: 2415, level_low: 2412, status: 'active', created_at: '2026-07-26T10:00:00Z', tested: false, user_flagged: false },
  ],
} as unknown as MarketReadingStructure;

function renderLiq() {
  return render(
    <div className="app-shell">
      <LiquidityCard structure={LIQ} instrument="XAUUSD" price={2395} selectedId={null} onSelect={noop} openHelp={null} onToggleHelp={noop} />
    </div>,
  );
}
function renderStruct() {
  return render(
    <div className="app-shell">
      <StructureCard structure={STRUCT} instrument="XAUUSD" price={2395} selectedId={null} onSelect={noop} onOpenZone={noop} openHelp={null} onToggleHelp={noop} />
    </div>,
  );
}

/** Click a filter chip (or the Réinitialiser action) by its exact label. */
function chip(c: HTMLElement, label: string): HTMLElement {
  const el = Array.from(c.querySelectorAll('.fchip')).find(
    (b) => (b.textContent ?? '').trim() === label,
  );
  if (!el) throw new Error(`chip not found: ${label}`);
  return el as HTMLElement;
}
const rows = (c: HTMLElement) => Array.from(c.querySelectorAll('.zrow'));
const badge = (c: HTMLElement) => c.querySelector('.badge2')?.textContent ?? '';

describe('UI-3a — Liquidity multi-select filters', () => {
  it('default: every pocket shown, counter is « N sur M » with N=M', () => {
    const { container } = renderLiq();
    expect(rows(container)).toHaveLength(4);
    expect(badge(container)).toBe('4 sur 4 poches');
  });

  it('two states kept (intact + swept) → those appear, broken does not', () => {
    const { container } = renderLiq();
    fireEvent.click(chip(container, fr.app.liq2.st.broken)); // toggle broken OFF
    const shown = rows(container);
    expect(shown).toHaveLength(3); // p1, p2 (intact) + p3 (swept)
    expect(badge(container)).toBe('3 sur 4 poches');
    // the broken pocket (p4) is gone — no visible ROW carries the broken badge.
    expect(shown.some((r) => (r.textContent ?? '').includes(fr.app.liq2.badge.broken))).toBe(false);
  });

  it('zero state selected → empty list + honest message, NEVER a fallback to all', () => {
    const { container } = renderLiq();
    fireEvent.click(chip(container, fr.app.liq2.st.intact));
    fireEvent.click(chip(container, fr.app.liq2.st.swept));
    fireEvent.click(chip(container, fr.app.liq2.st.broken));
    expect(rows(container)).toHaveLength(0); // NOT 4 — no silent "show all"
    expect(container.querySelector('.zempty')?.textContent).toContain(fr.app.liq2.noneState);
    expect(badge(container)).toBe('0 sur 4 poches');
  });

  it('« Réinitialiser » re-selects the whole group', () => {
    const { container } = renderLiq();
    fireEvent.click(chip(container, fr.app.liq2.st.intact));
    fireEvent.click(chip(container, fr.app.liq2.st.swept));
    expect(rows(container).length).toBeLessThan(4);
    // Two reset buttons (side, then state group); the state reset is the second.
    const resets = Array.from(container.querySelectorAll('.fchip.freset'));
    fireEvent.click(resets[1] as HTMLElement);
    expect(rows(container)).toHaveLength(4);
  });

  it('cross-group AND: BSL only + intact only → only the BSL intact pocket', () => {
    const { container } = renderLiq();
    fireEvent.click(chip(container, fr.app.liq2.side.ssl)); // side → {BSL}
    fireEvent.click(chip(container, fr.app.liq2.st.swept)); // state → {intact, broken}
    fireEvent.click(chip(container, fr.app.liq2.st.broken)); // state → {intact}
    const shown = rows(container);
    expect(shown).toHaveLength(1); // p1 = BSL intact
    expect(shown[0]!.textContent).toContain('BSL');
    expect(badge(container)).toBe('1 sur 4 poches');
  });
});

describe('UI-3a — Structure multi-select filters', () => {
  it('default shows all zones with « N sur M »', () => {
    const { container } = renderStruct();
    expect(rows(container)).toHaveLength(4);
    expect(badge(container)).toBe('4 sur 4 zones');
  });

  it('cross-group AND: type OB + state active → only the active OB', () => {
    const { container } = renderStruct();
    fireEvent.click(chip(container, fr.app.struct.type.fvg)); // type → {ob}
    fireEvent.click(chip(container, fr.app.struct.st.tested)); // state → {active, mitig}
    fireEvent.click(chip(container, fr.app.struct.st.mitig)); // state → {active}
    const shown = rows(container);
    expect(shown).toHaveLength(1); // ob1 (active)
    expect(badge(container)).toBe('1 sur 4 zones');
  });

  it('zero type selected → empty + noneType message, no fallback', () => {
    const { container } = renderStruct();
    fireEvent.click(chip(container, fr.app.struct.type.ob));
    fireEvent.click(chip(container, fr.app.struct.type.fvg));
    expect(rows(container)).toHaveLength(0);
    expect(container.querySelector('.zempty')?.textContent).toContain(fr.app.struct.noneType);
  });

  it('no raw i18n key is rendered in either card', () => {
    for (const { container } of [renderLiq(), renderStruct()]) {
      const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT);
      let n: Node | null;
      while ((n = walker.nextNode())) {
        const t = (n.textContent ?? '').trim();
        if (!t || t.includes('/') || t.includes(':')) continue;
        expect(/^[a-z][a-zA-Z0-9]*(\.[a-zA-Z][a-zA-Z0-9]*)+$/.test(t), `raw key: ${t}`).toBe(false);
      }
    }
  });
});
