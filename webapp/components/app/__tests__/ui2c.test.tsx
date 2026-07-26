import { render, screen, fireEvent } from '@/components/test-utils';
import { afterEach, describe, expect, it, vi } from 'vitest';
import fr from '@/messages/fr.json';
import { StructureCard } from '../StructureCard';
import { RegimeCard } from '../RegimeCard';
import { coerceViewActions } from '@/lib/chart/viewActions';
import type { MarketReadingStructure } from '@/types/market-reading';

// RegimeCard fetches sibling-TF trends via useMtfTrends — stub it so the render
// is deterministic (no network) and the alignment cell is populated.
vi.mock('@/lib/market-reading/hooks', async (orig) => {
  const actual = await orig<typeof import('@/lib/market-reading/hooks')>();
  return {
    ...actual,
    useMtfTrends: () => ({
      trends: { h4: 'bearish', h1: 'bearish', m15: 'bearish' },
      isLoading: false,
    }),
  };
});

afterEach(() => vi.restoreAllMocks());

/* ── Fixtures (minimal, real-shaped engine output) ───────────────────────── */
const STRUCT: MarketReadingStructure = {
  order_blocks: [
    { id: 'ob1', direction: 'bearish', level_high: 2401.1, level_low: 2398.4, importance: 'high', status: 'active', created_at: '2026-07-26T11:00:00Z', tested: false, user_flagged: false },
    { id: 'ob2', direction: 'bullish', level_high: 2385.5, level_low: 2383.0, importance: 'medium', status: 'mitigated', created_at: '2026-07-26T09:00:00Z', tested: true, mitigated_at: '2026-07-26T12:45:00Z', user_flagged: false },
  ],
  fair_value_gaps: [
    { id: 'fvg1', direction: 'bullish', level_high: 2394.0, level_low: 2389.0, status: 'partially_filled', created_at: '2026-07-26T10:15:00Z', tested: true, mitigated_at: '2026-07-26T13:00:00Z', fill_level: 2391.9, user_flagged: false },
  ],
  choch_events: [{ level: 2400.8, direction: 'bearish', broken_at: '2026-07-26T14:00:00Z', validation_status: 'confirmed' }],
  bos_events: [{ level: 2399.0, direction: 'bearish', broken_at: '2026-07-26T11:00:00Z', validation_status: 'confirmed' }],
} as unknown as MarketReadingStructure;

const HEADER = { instrument: 'XAUUSD', timeframe: 'H1', close_price: 2392.35, candle_close_ts: null } as never;
const REGIME = { trend: 'bearish', volatility_observed: 'normal', market_phase: 'trend', mtf_confluence: {} } as never;

const noop = () => {};

function renderStructure(price = 2392.35) {
  return render(
    <StructureCard
      structure={STRUCT}
      instrument="XAUUSD"
      price={price}
      selectedId={null}
      onSelect={noop}
      onOpenZone={noop}
      openHelp={null}
      onToggleHelp={noop}
    />,
  );
}

/* ── Honesty of the copy (short UI strings) ──────────────────────────────── */
describe('UI-2c — copy honesty', () => {
  const FORBIDDEN = [
    'va rebondir', 'cible', 'biais', 'setup gagnant', "signal d'achat",
    'signal de vente', 'plus sûr', 'plus safe', 'probabilité', 'recommandé',
  ];
  // Short UI namespaces only — the pedagogical help texts deliberately QUOTE some
  // of these words to REFUSE them (covered by the negative-block test below).
  function leaves(o: unknown, out: string[] = []): string[] {
    if (typeof o === 'string') out.push(o);
    else if (o && typeof o === 'object') for (const v of Object.values(o)) leaves(v, out);
    return out;
  }
  it('no forbidden claim word in the Structure / Liquidity / Régime labels', () => {
    const strings = [
      ...leaves((fr as never as { app: { struct: unknown } }).app.struct),
      ...leaves((fr as never as { app: { liq2: unknown } }).app.liq2),
      ...leaves((fr as never as { app: { reg2: unknown } }).app.reg2),
    ];
    for (const s of strings) {
      const low = s.toLowerCase();
      for (const bad of FORBIDDEN) {
        expect(low.includes(bad.toLowerCase()), `« ${bad} » in: ${s}`).toBe(false);
      }
      expect(s).not.toMatch(/\bTrader\b/);
    }
  });
});

/* ── Pedagogical help: the "ce que ça ne dit pas" block ──────────────────── */
describe('UI-2c — help texts carry a negative-disclaimer block', () => {
  const NEG = [
    'ne dit pas', "n'affirmera jamais", 'ne dit jamais', "n'écrira jamais",
    'ne veut pas dire', 'pas une projection', 'contradiction', 'aucune probabilité',
    'aucune direction',
  ];
  const perMeasure = ['trend', 'vol', 'mat', 'align', 'dens', 'struct', 'liq'];
  it.each(perMeasure)('help "%s" refuses to over-claim', (key) => {
    const body = (fr as never as { reading: { help: Record<string, { body: string }> } })
      .reading.help[key]!.body.toLowerCase();
    expect(NEG.some((m) => body.includes(m.toLowerCase())), `no negative block in help.${key}`).toBe(true);
  });
});

/* ── id-lock: an invented id is rejected by the highlight ────────────────── */
describe('UI-2c — highlight id-lock', () => {
  it('rejects a zone id the engine never emitted', () => {
    const valid = new Set(['ob1', 'fvg1']);
    const good = coerceViewActions([{ action: 'highlight_zone', params: { zone_id: 'ob1' } }], valid);
    const bad = coerceViewActions([{ action: 'highlight_zone', params: { zone_id: 'invented-42' } }], valid);
    expect(good).toHaveLength(1);
    expect(bad).toEqual([]);
  });
});

/* ── Structure card: filter → empty, and tick → no reorder ───────────────── */
describe('UI-2c — Structure card list', () => {
  const rows = (c: HTMLElement) => Array.from(c.querySelectorAll('.zrow'));

  it('renders the full zone list (no 4-item truncation)', () => {
    const { container } = renderStructure();
    expect(rows(container)).toHaveLength(3);
  });

  it('an empty filter shows the honest message and no rows', () => {
    const { container } = renderStructure();
    // Open the sort/filter panel, then keep only FVG + Actives (fvg1 is
    // partially_filled → "tested", so this filter matches nothing).
    fireEvent.click(screen.getByRole('button', { name: fr.app.struct.sortAria }));
    fireEvent.click(screen.getByRole('button', { name: fr.app.struct.type.fvg }));
    fireEvent.click(screen.getByRole('button', { name: fr.app.struct.st.active }));
    expect(rows(container)).toHaveLength(0);
    const empty = container.querySelector('.zempty')?.textContent ?? '';
    expect(empty).toContain(fr.app.struct.empty1);
    expect(empty).toContain(fr.app.struct.empty2);
  });

  it('a price tick updates the distances WITHOUT reordering the list', () => {
    const { container, rerender } = renderStructure(2392.35);
    const bands = () => rows(container).map((r) => r.querySelector('.zr')?.textContent);
    const dists = () => rows(container).map((r) => r.querySelector('.zd')?.textContent);
    const bandsBefore = bands();
    const distBefore = dists();
    rerender(
      <StructureCard
        structure={STRUCT}
        instrument="XAUUSD"
        price={2405.0}
        selectedId={null}
        onSelect={noop}
        onOpenZone={noop}
        openHelp={null}
        onToggleHelp={noop}
      />,
    );
    expect(bands()).toEqual(bandsBefore); // order frozen
    expect(dists()).not.toEqual(distBefore); // distances recomputed
  });
});

/* ── Régime card: every displayed measure names its source ───────────────── */
describe('UI-2c — Régime measures each show a source sub-line', () => {
  it('renders a non-empty .sub2 under the sourced measures', () => {
    const { container } = render(
      <RegimeCard regime={REGIME} structure={STRUCT} header={HEADER} openHelp={null} onToggleHelp={noop} />,
    );
    const subs = Array.from(container.querySelectorAll('.sub2'));
    // Trend · Volatilité · Maturité · Alignement · Dernier évén. · Densité all
    // carry a source line here (maturity resolves from the CHOCH history).
    expect(subs.length).toBeGreaterThanOrEqual(5);
    for (const s of subs) expect((s.textContent ?? '').trim().length).toBeGreaterThan(0);
  });
});
