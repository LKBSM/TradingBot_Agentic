import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ZonesWorkspace } from '../ZonesWorkspace';
import { ChartViewProvider, useChartViewOptional } from '@/lib/chart/viewState';
import { coerceViewActions } from '@/lib/chart/viewActions';
import { collectZones } from '@/lib/zones/lifecycle';
import { FIXTURE_XAU_M15 } from '@/lib/market-reading/fixtures';

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

/** Probe exposing the SHARED chart view state so we can assert hide reflects. */
function HiddenProbe() {
  const { view } = useChartViewOptional();
  return <div data-testid="hidden-ids">{view.hiddenZoneIds.join(',')}</div>;
}

function renderZones() {
  return render(
    <ChartViewProvider>
      <ZonesWorkspace locale="fr" />
      <HiddenProbe />
    </ChartViewProvider>,
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

describe('ZonesWorkspace', () => {
  it('(a) renders zone cards with their real lifecycle data', async () => {
    renderZones();
    // Cards for every emitted zone (2 OB + 2 FVG in the fixture).
    await waitFor(() => expect(screen.getAllByRole('article')).toHaveLength(4));
    // Real lifecycle facts surfaced.
    expect(screen.getAllByText('Order Block').length).toBeGreaterThan(0);
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
});
