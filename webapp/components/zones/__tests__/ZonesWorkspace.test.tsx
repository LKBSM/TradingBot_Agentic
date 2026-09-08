import { fireEvent, render as rtlRender, screen, waitFor, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { NextIntlClientProvider } from 'next-intl';
import { ZonesWorkspace } from '../ZonesWorkspace';
import { ChartViewProvider, useChartViewOptional } from '@/lib/chart/viewState';
import { ChatProvider } from '@/components/chat/ChatProvider';
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

// The /zones panel is now the shared backend-agent MiaPanel — stub the streaming
// chat call so a sent question resolves deterministically (no network in jsdom).
const askStreamMock = vi.fn();
vi.mock('@/lib/chat/api-client', async (importActual) => {
  const actual = await importActual<typeof import('@/lib/chat/api-client')>();
  return {
    ...actual,
    askSentinelStream: (...args: unknown[]) => askStreamMock(...args),
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
      <ChatProvider>
        <ChartViewProvider>
          <ZonesWorkspace locale="fr" />
          <HiddenProbe />
        </ChartViewProvider>
      </ChatProvider>
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
  askStreamMock.mockReset();
  askStreamMock.mockResolvedValue({
    text: 'Réponse de M.I.A.',
    blockedReason: null,
    toolCallsMade: [],
    viewActions: [],
  });
  // The single product conversation persists in localStorage — clear it so each
  // test starts empty.
  window.localStorage.clear();
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

  it('MIA-3 — a zone question is routed to the SHARED backend agent with the selected-zone orientation', async () => {
    renderZones();
    await waitFor(() => expect(screen.getAllByRole('article')).toHaveLength(4));
    // Ask through the shared panel (no local answering engine anymore).
    const field = screen.getAllByPlaceholderText(/Pose une question à M\.I\.A/i)[0]!;
    fireEvent.change(field, { target: { value: 'qu’est-ce qu’il y a d’autre à ce niveau' } });
    fireEvent.click(screen.getAllByRole('button', { name: 'Envoyer la question' })[0]!);
    await waitFor(() => expect(askStreamMock).toHaveBeenCalledTimes(1));
    const opts = askStreamMock.mock.calls[0]![0] as {
      focus?: string | null;
      signal?: { instrument: string; timeframe: string } | null;
    };
    // Orientation = the selected zone; combo = the page market. The answer comes
    // from the agent (tool-grounded), never a locally fabricated string.
    expect(opts.focus).toMatch(/^\[Zone sélectionnée : /);
    expect(opts.signal?.instrument).toBe('XAUUSD');
    expect(await screen.findByText('Réponse de M.I.A.')).toBeInTheDocument();
  });

  it('MIA-3 — a question OUTSIDE the zone is still answered (orientation, not prison)', async () => {
    renderZones();
    await waitFor(() => expect(screen.getAllByRole('article')).toHaveLength(4));
    const field = screen.getAllByPlaceholderText(/Pose une question à M\.I\.A/i)[0]!;
    fireEvent.change(field, {
      target: { value: 'et sur l’EURUSD, quelles publications macro sont à venir ?' },
    });
    fireEvent.click(screen.getAllByRole('button', { name: 'Envoyer la question' })[0]!);
    // Sent to the agent — never refused with « je ne peux parler que de cette zone ».
    await waitFor(() => expect(askStreamMock).toHaveBeenCalledTimes(1));
    expect(await screen.findByText('Réponse de M.I.A.')).toBeInTheDocument();
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

  it('MIA-3/CLN-1 §3 — re-clicking the selected zone deselects it; the shared conversation is kept', async () => {
    renderZones();
    await waitFor(() => expect(screen.getAllByRole('article')).toHaveLength(4));
    // A zone is selected by default → the orientation subject block is shown, and
    // exactly one card is highlighted.
    expect(screen.getByTestId('mia-subject')).toBeInTheDocument();
    const selected = document.querySelector<HTMLElement>('.zone.zsel');
    expect(selected).toBeTruthy();

    // Start a conversation through the SHARED panel so we can prove deselect ≠ reset.
    // The desktop column and the mobile sheet both mount the panel; target the
    // first input (the visible desktop column in jsdom).
    const fields = screen.getAllByPlaceholderText(/Pose une question à M\.I\.A/i);
    fireEvent.change(fields[0]!, { target: { value: 'explique moi cette zone' } });
    const sendButtons = screen.getAllByRole('button', { name: 'Envoyer la question' });
    fireEvent.click(sendButtons[0]!);
    // The user turn appears in the shared transcript, and the backend was asked
    // with the selected-zone orientation preamble.
    expect(await screen.findByText('explique moi cette zone')).toBeInTheDocument();
    await waitFor(() => expect(askStreamMock).toHaveBeenCalledTimes(1));
    const focus = (askStreamMock.mock.calls[0]![0] as { focus?: string | null }).focus;
    expect(focus).toMatch(/^\[Zone sélectionnée : /);

    // Re-click the SAME (selected) card → deselect.
    fireEvent.click(selected!);
    await waitFor(() =>
      expect(screen.queryByTestId('mia-subject')).not.toBeInTheDocument(),
    );
    // No card highlighted, no filler subject block.
    expect(document.querySelector('.zone.zsel')).toBeNull();
    // The running conversation is NOT wiped — the earlier question is still shown.
    expect(screen.getByText('explique moi cette zone')).toBeInTheDocument();
    // The field stays usable; its placeholder never claimed a zone was chosen.
    expect(screen.getAllByPlaceholderText(/Pose une question à M\.I\.A/i)[0]).toBeInTheDocument();
  });
});
