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
    // CLN-1 §2 — the four prefabricated question blocks are gone; the same
    // factual answers are reached via the shared composer (free text, routed
    // LOCALLY). Ask « qu'est-ce qu'il y a d'autre à ce niveau » — the answer is
    // built by buildConfluence over the SAME data as the card's confluence block.
    const field = screen.getByPlaceholderText('Pose ta question sur cette zone…');
    fireEvent.change(field, { target: { value: 'qu’est-ce qu’il y a d’autre à ce niveau' } });
    fireEvent.click(screen.getByRole('button', { name: 'Envoyer' }));
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

  // ── CLN-1 ────────────────────────────────────────────────────────────────
  it('CLN-1 §1 — the product pitch line under the title is no longer rendered', async () => {
    const { container } = renderZones();
    await waitFor(() => expect(screen.getAllByRole('article')).toHaveLength(4));
    expect(container.textContent ?? '').not.toContain('cycle de vie de chaque zone');
    expect(document.querySelector('.pghead .sub')).toBeNull();
    // The factual context line on the right stays.
    expect(document.querySelector('.pghead .livebadge')).not.toBeNull();
  });

  it('CLN-1 §2 — the four prefabricated question blocks are gone', async () => {
    renderZones();
    await waitFor(() => expect(screen.getAllByRole('article')).toHaveLength(4));
    expect(document.querySelector('.zmia-sugg')).toBeNull();
    for (const q of [
      /Explique-moi ce type de zone/i,
      /Compare-la à l’unité supérieure/i,
      /Que s’est-il passé au dernier contact/i,
      /Qu’est-ce qu’il y a d’autre à ce niveau/i,
    ]) {
      expect(screen.queryByRole('button', { name: q })).toBeNull();
    }
  });

  it('CLN-1 §3 — re-clicking the selected zone deselects it; the conversation is kept', async () => {
    renderZones();
    await waitFor(() => expect(screen.getAllByRole('article')).toHaveLength(4));
    // A zone is selected by default → subject shown, exactly one card highlighted.
    expect(screen.getByTestId('mia-subject')).toBeInTheDocument();
    const selected = document.querySelector<HTMLElement>('.zone.zsel');
    expect(selected).toBeTruthy();
    // Start a conversation so we can prove « deselect ≠ reset ».
    const field = screen.getByPlaceholderText('Pose ta question sur cette zone…');
    fireEvent.change(field, { target: { value: 'explique moi cette zone' } });
    fireEvent.click(screen.getByRole('button', { name: 'Envoyer' }));
    const body = document.querySelector('.zmia-body')!;
    await waitFor(() => expect(body.querySelectorAll('.bub.u').length).toBe(1));
    // Re-click the SAME (selected) card → deselect.
    fireEvent.click(selected!);
    await waitFor(() => expect(screen.queryByTestId('mia-subject')).not.toBeInTheDocument());
    // No card highlighted, no filler subject block.
    expect(document.querySelector('.zone.zsel')).toBeNull();
    // The running conversation is NOT wiped.
    expect(body.querySelectorAll('.bub.u').length).toBe(1);
    // The field stays usable, and its idle hint does not claim a zone is chosen.
    expect(screen.getByPlaceholderText('Pose ta question…')).toBeInTheDocument();
    expect(screen.queryByPlaceholderText('Pose ta question sur cette zone…')).toBeNull();
  });
});
