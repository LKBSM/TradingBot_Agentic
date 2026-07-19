import { fireEvent, render as rtlRender, screen, waitFor, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { NextIntlClientProvider } from 'next-intl';
import { ZonesWorkspace } from '../ZonesWorkspace';
import { ChartViewProvider, useChartViewOptional } from '@/lib/chart/viewState';
import { coerceViewActions } from '@/lib/chart/viewActions';
import { collectZones } from '@/lib/zones/lifecycle';
import { FIXTURE_XAU_M15 } from '@/lib/market-reading/fixtures';
import messages from '@/messages/fr.json';

// The zones surface consumes the `zones` namespace (+ `reading` enums via
// useReadingFormatters); both live in the fr bundle so the asserted FR strings
// resolve directly.

const fetchMock = vi.fn();
vi.mock('@/lib/market-reading/api-client', async (importActual) => {
  const actual =
    await importActual<typeof import('@/lib/market-reading/api-client')>();
  return {
    ...actual,
    fetchMarketReading: (...args: unknown[]) => fetchMock(...args),
    fetchCandles: () => Promise.resolve([]),
  };
});

// ZonesWorkspace now drives the combo from the URL (NAV-04); stub the app-router
// hooks. Empty search params → it falls back to the default combo as before.
vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
  usePathname: () => '/zones',
  useSearchParams: () => new URLSearchParams(),
}));

/** Probe exposing the SHARED chart view state so we can assert hide reflects. */
function HiddenProbe() {
  const { view } = useChartViewOptional();
  return <div data-testid="hidden-ids">{view.hiddenZoneIds.join(',')}</div>;
}

function renderZones() {
  return rtlRender(
    <NextIntlClientProvider locale="fr" messages={messages}>
      <ChartViewProvider>
        <ZonesWorkspace locale="fr" />
        <HiddenProbe />
      </ChartViewProvider>
    </NextIntlClientProvider>,
  );
}

/** The <article> card whose "Analyser" link targets a given zone id. */
function cardForZone(zoneId: string): HTMLElement {
  const articles = screen.getAllByRole('article');
  const target = articles.find((a) =>
    within(a)
      .getByRole('link', { name: /analyser/i })
      .getAttribute('href')
      ?.includes(`focus=${zoneId}`),
  );
  if (!target) throw new Error(`no card for zone ${zoneId}`);
  return target;
}

beforeEach(() => {
  fetchMock.mockReset();
  fetchMock.mockResolvedValue(FIXTURE_XAU_M15);
});

afterEach(() => {
  vi.restoreAllMocks();
});

/** Expand a card's "Détails" section (timeline + overlaps live there now). */
function expandCard(card: HTMLElement): void {
  fireEvent.click(within(card).getByRole('button', { name: 'Détails' }));
}

describe('ZonesWorkspace', () => {
  it('(a) renders zone cards with their real lifecycle data (expanded)', async () => {
    renderZones();
    // Cards for every emitted zone (2 OB + 2 FVG in the fixture).
    await waitFor(() => expect(screen.getAllByRole('article')).toHaveLength(4));
    // Real lifecycle facts surfaced once the details are unfolded.
    expect(screen.getAllByText('Order Block').length).toBeGreaterThan(0);
    for (const card of screen.getAllByRole('article')) expandCard(card);
    expect(screen.getAllByText('Formé').length).toBeGreaterThan(0);
    expect(screen.getByText('Mitigé')).toBeInTheDocument(); // ob-xau-2-mitigated
    expect(screen.getByText('Partiellement comblé')).toBeInTheDocument(); // fvg-xau-2-partial
  });

  it('(b) degrades gracefully when lifecycle data is absent (no invented event)', async () => {
    renderZones();
    await waitFor(() => expect(screen.getAllByRole('article')).toHaveLength(4));
    // The partially-filled FVG has NO fill_level in the fixture → no fill bar is
    // fabricated (no progressbar), and no "%" is invented for it.
    expect(screen.queryByRole('progressbar')).not.toBeInTheDocument();
    // The untested active OB (ob-xau-1) shows no "Testé" step.
    const card = cardForZone('ob-xau-1');
    expandCard(card);
    expect(within(card).queryByText('Testé')).not.toBeInTheDocument();
    expect(within(card).getByText('Suivi en cours')).toBeInTheDocument();
  });

  it('(c) "Analyser" links to /app focusing the right zone id', async () => {
    renderZones();
    await waitFor(() => expect(screen.getAllByRole('article')).toHaveLength(4));
    const link = within(cardForZone('ob-xau-1')).getByRole('link', { name: /analyser/i });
    const href = link.getAttribute('href')!;
    expect(href).toContain('instrument=XAUUSD');
    expect(href).toContain('timeframe=M15');
    expect(href).toContain('focus=ob-xau-1');
  });

  it('(d) "Masquer" hides the right zone in the shared view state, reversibly', async () => {
    renderZones();
    await waitFor(() => expect(screen.getAllByRole('article')).toHaveLength(4));
    const card = cardForZone('fvg-xau-1');

    fireEvent.click(within(card).getByRole('button', { name: /masquer du graphique/i }));
    // Reflected in the SHARED chart view state (what /app reads).
    expect(screen.getByTestId('hidden-ids')).toHaveTextContent('fvg-xau-1');

    // Reversible — the button now offers to show it again.
    fireEvent.click(within(card).getByRole('button', { name: /afficher sur le graphique/i }));
    expect(screen.getByTestId('hidden-ids')).not.toHaveTextContent('fvg-xau-1');
  });

  it('(e) an inexistent id is rejected by the same id-lock — nothing is masked', () => {
    // The workspace routes every hide through coerceViewActions against the
    // on-screen id set; an invented id drops the whole action.
    const validZoneIds = new Set(collectZones(FIXTURE_XAU_M15.structure).map((z) => z.id));
    const coerced = coerceViewActions(
      [{ action: 'hide_zones', params: { zone_ids: ['does-not-exist'] } }],
      validZoneIds,
    );
    expect(coerced).toEqual([]);
  });

  it('filters to Mitigées (mitigated OB + filled/partial FVG)', async () => {
    renderZones();
    await waitFor(() => expect(screen.getAllByRole('article')).toHaveLength(4));
    fireEvent.click(screen.getByRole('button', { name: 'Mitigées' }));
    // ob-xau-2-mitigated + fvg-xau-2-partial remain (2 cards).
    await waitFor(() => expect(screen.getAllByRole('article')).toHaveLength(2));
  });

  it('(f) sorts by proximity to the price by DEFAULT; no importance/quality sort exists', async () => {
    renderZones();
    await waitFor(() => expect(screen.getAllByRole('article')).toHaveLength(4));

    // close_price = 2392.35 → nearest band is ob-xau-2-mitigated (2384–2386).
    const first = screen.getAllByRole('article')[0]!;
    expect(
      within(first).getByRole('link', { name: /analyser/i }).getAttribute('href'),
    ).toContain('focus=ob-xau-2-mitigated');

    // The three FACTUAL orders — and no "Importance" (quality) option at all.
    const sortGroup = screen.getByRole('group', { name: 'Trier les zones' });
    expect(within(sortGroup).getByRole('button', { name: 'Proximité' })).toBeInTheDocument();
    expect(within(sortGroup).getByRole('button', { name: 'Fraîcheur' })).toBeInTheDocument();
    expect(within(sortGroup).getByRole('button', { name: 'État' })).toBeInTheDocument();
    expect(within(sortGroup).queryByRole('button', { name: 'Importance' })).not.toBeInTheDocument();
  });

  it('(g) shows the relation to the price as a present-tense fact (distance to the band)', async () => {
    renderZones();
    await waitFor(() => expect(screen.getAllByRole('article')).toHaveLength(4));
    // ob-xau-2-mitigated: band 2384–2386, price 2392.35 → 6.35 pts below the price.
    const card = cardForZone('ob-xau-2-mitigated');
    expect(within(card).getByText('à 6,35 pts en dessous du prix')).toBeInTheDocument();
    // And the age fact is present (duration-only here: no candle window is served).
    expect(within(card).getByText(/formée il y a/)).toBeInTheDocument();
  });

  it('(h) surfaces geometric overlaps with the other timeframes, phrased as geometry', async () => {
    renderZones();
    await waitFor(() => expect(screen.getAllByRole('article')).toHaveLength(4));
    const card = cardForZone('ob-xau-1');
    expandCard(card);
    // Sibling readings (H1/H4 fetches) carry the same fixture → the same band
    // overlaps; the wording is pure geometry ("chevauche … (bornes)").
    await waitFor(() =>
      expect(within(card).getAllByText(/chevauche un OB H1 haussier/).length).toBeGreaterThan(0),
    );
  });

  it('(i) never renders a test count, a score or any confluence/quality wording', async () => {
    const { container } = renderZones();
    await waitFor(() => expect(screen.getAllByRole('article')).toHaveLength(4));
    for (const card of screen.getAllByRole('article')) expandCard(card);
    const text = container.textContent ?? '';
    // No "Testé ×N" (the engine tracks no count) and no ranking vocabulary.
    expect(text).not.toMatch(/×\s*\d/);
    expect(text).not.toMatch(/score|confluence|renforc|fiab|étoile|classement/i);
  });
});
