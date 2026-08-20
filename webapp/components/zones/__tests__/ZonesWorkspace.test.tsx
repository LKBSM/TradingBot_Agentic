import { fireEvent, render as rtlRender, screen, waitFor, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { NextIntlClientProvider } from 'next-intl';
import { ZonesWorkspace } from '../ZonesWorkspace';
import { ChartViewProvider, useChartViewOptional } from '@/lib/chart/viewState';
import { coerceViewActions } from '@/lib/chart/viewActions';
import { collectZones } from '@/lib/zones/lifecycle';
import { FIXTURE_XAU_M15 } from '@/lib/market-reading/fixtures';
import messages from '@/messages/fr.json';

const fetchMock = vi.fn();
vi.mock('@/lib/market-reading/api-client', async (importActual) => {
  const actual = await importActual<typeof import('@/lib/market-reading/api-client')>();
  return {
    ...actual,
    fetchMarketReading: (...args: unknown[]) => fetchMock(...args),
    fetchCandles: () => Promise.resolve([]),
  };
});

const pushMock = vi.fn();
const replaceMock = vi.fn();
let mockSearchParams = new URLSearchParams();
vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: pushMock, replace: replaceMock }),
  usePathname: () => '/zones',
  useSearchParams: () => mockSearchParams,
}));

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

function card(zoneId: string): HTMLElement {
  const el = document.querySelector<HTMLElement>(`[data-zone-id="${zoneId}"]`);
  if (!el) throw new Error(`no card for ${zoneId}`);
  return el;
}

beforeEach(() => {
  fetchMock.mockReset();
  fetchMock.mockResolvedValue(FIXTURE_XAU_M15);
  pushMock.mockReset();
  replaceMock.mockReset();
  mockSearchParams = new URLSearchParams();
});
afterEach(() => vi.restoreAllMocks());

describe('ZonesWorkspace (VZ-1)', () => {
  it('renders every emitted zone, grouped by position (all below the price here)', async () => {
    renderZones();
    await waitFor(() => expect(screen.getAllByRole('article')).toHaveLength(4));
    // price 2392.35 → all four bands sit below → the « Sous le prix » group only.
    expect(screen.getByText('Sous le prix')).toBeInTheDocument();
    expect(screen.queryByText('Le prix est dedans')).not.toBeInTheDocument();
  });

  it('exposes the factual filters and sorts — NO importance/quality control', async () => {
    renderZones();
    await waitFor(() => expect(screen.getAllByRole('article')).toHaveLength(4));
    const sortGroup = screen.getByRole('group', { name: 'Trier les zones' });
    expect(within(sortGroup).getByRole('button', { name: 'Proximité' })).toBeInTheDocument();
    expect(within(sortGroup).getByRole('button', { name: 'Formation' })).toBeInTheDocument();
    expect(within(sortGroup).getByRole('button', { name: 'Contacts' })).toBeInTheDocument();
    expect(within(sortGroup).queryByRole('button', { name: /importance|qualité|score/i })).not.toBeInTheDocument();
    const filterGroup = screen.getByRole('group', { name: 'Filtrer les zones' });
    for (const label of ['Toutes', 'Actives', 'Jamais touchées', 'Comblées']) {
      expect(within(filterGroup).getByRole('button', { name: label })).toBeInTheDocument();
    }
  });

  it('an empty filter shows an EXPLICIT message and never suggests relaxing it', async () => {
    renderZones();
    await waitFor(() => expect(screen.getAllByRole('article')).toHaveLength(4));
    // No consumed zone in the fixture → « Comblées » is empty.
    fireEvent.click(screen.getByRole('button', { name: 'Comblées' }));
    const empty = await screen.findByTestId('zones-empty');
    expect(empty.textContent ?? '').toMatch(/aucune zone comblée/i);
    expect(empty.textContent ?? '').not.toMatch(/assoupl|élargir|relax|moins strict/i);
  });

  it('« Masquer » hides the right zone in the shared view state, reversibly', async () => {
    renderZones();
    await waitFor(() => expect(screen.getAllByRole('article')).toHaveLength(4));
    const c = card('fvg-xau-1');
    fireEvent.click(within(c).getByRole('button', { name: 'Masquer' }));
    expect(screen.getByTestId('hidden-ids')).toHaveTextContent('fvg-xau-1');
    fireEvent.click(within(c).getByRole('button', { name: 'Afficher' }));
    expect(screen.getByTestId('hidden-ids')).not.toHaveTextContent('fvg-xau-1');
  });

  it('an invented id is rejected by the id-lock (nothing masked)', () => {
    const validZoneIds = new Set(collectZones(FIXTURE_XAU_M15.structure).map((z) => z.id));
    expect(
      coerceViewActions([{ action: 'hide_zones', params: { zone_ids: ['nope'] } }], validZoneIds),
    ).toEqual([]);
  });

  it('the M.I.A panel shows the selected zone and SWITCHES subject on a card click (no reload)', async () => {
    renderZones();
    await waitFor(() => expect(screen.getAllByRole('article')).toHaveLength(4));
    const subject = screen.getByTestId('mia-subject');
    // Default subject = the nearest zone (ob-xau-2-mitigated, 2384–2386).
    expect(subject.textContent).toContain('384,00');
    // Click a different card → the subject changes, without any navigation.
    fireEvent.click(card('ob-xau-1'));
    await waitFor(() => expect(screen.getByTestId('mia-subject').textContent).toContain('375,00'));
    expect(pushMock).not.toHaveBeenCalled();
  });

  it('the M.I.A panel answers with facts drawn from the SAME data as the card', async () => {
    renderZones();
    await waitFor(() => expect(screen.getAllByRole('article')).toHaveLength(4));
    const body = document.querySelector('.zmia-body')!;
    // Only the intro bubble so far.
    expect(body.querySelectorAll('.bub').length).toBe(1);
    // Ask « qu'est-ce qu'il y a d'autre » — the answer is built by buildConfluence
    // over the SAME data as the card's confluence block (a same-level sibling zone,
    // or the honest absence state — never a parallel source).
    fireEvent.click(screen.getByRole('button', { name: /qu’est-ce qu’il y a d’autre/i }));
    await waitFor(() => expect(body.querySelectorAll('.bub').length).toBe(3)); // intro + Q + A
    const answer = body.querySelectorAll('.bub.a')[1]!;
    expect(answer.textContent ?? '').toMatch(
      /au même niveau|rien d’autre n’est détecté|à l’intérieur|englobe|poche de liquidité/i,
    );
  }, 20000);

  it('the M.I.A free-text input routes LOCALLY to a factual answer (no LLM)', async () => {
    renderZones();
    await waitFor(() => expect(screen.getAllByRole('article')).toHaveLength(4));
    const body = document.querySelector('.zmia-body')!;
    const input = screen.getByPlaceholderText('Pose ta question sur cette zone…');
    fireEvent.change(input, { target: { value: 'explique moi cette zone' } });
    fireEvent.click(screen.getByRole('button', { name: 'Envoyer' }));
    await waitFor(() => expect(body.querySelectorAll('.bub').length).toBe(3));
    // Default subject is an OB → the concept explanation, drawn from the card.
    expect(body.querySelectorAll('.bub.a')[1]!.textContent ?? '').toMatch(/Order Block|Fair Value Gap/);
  });

  it('an unrecognised question gets the honest fallback, never a fabrication', async () => {
    renderZones();
    await waitFor(() => expect(screen.getAllByRole('article')).toHaveLength(4));
    const body = document.querySelector('.zmia-body')!;
    const input = screen.getByPlaceholderText('Pose ta question sur cette zone…');
    fireEvent.change(input, { target: { value: 'zzzzqwerty' } });
    fireEvent.click(screen.getByRole('button', { name: 'Envoyer' }));
    await waitFor(() => expect(body.querySelectorAll('.bub').length).toBe(3));
    expect(body.querySelectorAll('.bub.a')[1]!.textContent ?? '').toMatch(/à partir de ses faits/i);
  });

  it('never renders « chevauche » nor any judgement wording', async () => {
    const { container } = renderZones();
    await waitFor(() => expect(screen.getAllByRole('article')).toHaveLength(4));
    const text = container.textContent ?? '';
    expect(text).not.toMatch(/chevauche/i);
    expect(text).not.toMatch(/respect|valid|solide|fiable|qualité|meilleur|score|classement/i);
    expect(text).not.toMatch(/×\s*\d/);
  });

  it('`?zone=<id>` highlights + selects the referenced card and scrolls it into view', async () => {
    const scrollSpy = vi.fn();
    (HTMLElement.prototype as unknown as { scrollIntoView: unknown }).scrollIntoView = scrollSpy;
    mockSearchParams = new URLSearchParams('zone=fvg-xau-1');
    renderZones();
    await waitFor(() => expect(screen.getAllByRole('article')).toHaveLength(4));
    await waitFor(() => expect(card('fvg-xau-1')).toHaveClass('zsel'));
    expect(screen.getByTestId('mia-subject').textContent).toContain('381,00');
    await waitFor(() =>
      expect(scrollSpy).toHaveBeenCalledWith(expect.objectContaining({ block: 'center' })),
    );
  });

  it('clicking a « même endroit » item navigates by REAL zone id (never a fabricated/price id)', async () => {
    renderZones();
    await waitFor(() => expect(screen.getAllByRole('article')).toHaveLength(4));
    // Expand a card's « Détails » to reveal its confluence block.
    const c = card('ob-xau-1');
    fireEvent.click(c.querySelector('button.zdeth')!);
    const navBtn = await waitFor(() => {
      const b = c.querySelector<HTMLButtonElement>('.zconf button.clnav');
      if (!b) throw new Error('no navigable confluence item');
      return b;
    });
    fireEvent.click(navBtn);
    // The handler writes a `?zone=<id>` deep-link with a REAL engine id — one of
    // the ids the engine actually emitted (same-TF live zones or a sibling id),
    // never a value derived from the displayed price/label.
    const realIds = new Set(collectZones(FIXTURE_XAU_M15.structure).map((z) => z.id));
    expect(replaceMock).toHaveBeenCalledTimes(1);
    const url = String(replaceMock.mock.calls[0]![0]);
    const zoneId = new URLSearchParams(url.split('?')[1]).get('zone');
    expect(zoneId).toBeTruthy();
    expect(realIds.has(zoneId!)).toBe(true);
    expect(url).toMatch(/instrument=/);
    expect(url).toMatch(/timeframe=/);
  });

  it('a stale `?zone=<id>` shows the honest notice, never a fabricated card', async () => {
    mockSearchParams = new URLSearchParams('zone=does-not-exist');
    renderZones();
    await waitFor(() => expect(screen.getAllByRole('article')).toHaveLength(4));
    expect(
      screen.getByText('Cette zone n\'est plus détectée dans la lecture courante.'),
    ).toBeInTheDocument();
    expect(screen.getAllByRole('article')).toHaveLength(4);
  });
});
