import { fireEvent, render as rtlRender, screen, waitFor, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { NextIntlClientProvider } from 'next-intl';
import { ZonesWorkspace } from '../ZonesWorkspace';
import { ChartViewProvider, useChartViewOptional } from '@/lib/chart/viewState';
import { ChatProvider, useChat } from '@/components/chat/ChatProvider';
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

// The M.I.A panel is now the shared shell column (not rendered inside the page).
// This probe surfaces the shared orientation (`focus`) so the /zones unit tests
// can assert the selected-zone subject WITHOUT mounting the whole shell.
function FocusProbe() {
  const { focus } = useChat();
  return <div data-testid="mia-focus">{focus ? focus.label : ''}</div>;
}

function renderZones() {
  return rtlRender(
    <NextIntlClientProvider locale="fr" messages={messages}>
      <ChatProvider>
        <ChartViewProvider>
          <ZonesWorkspace locale="fr" />
          <HiddenProbe />
          <FocusProbe />
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
    // VZ-5 turned the sort into a dropdown; what this guard owns is unchanged —
    // the criteria are FACTUAL only, never importance / quality / score.
    const sortSelect = screen.getByLabelText('Trié par') as HTMLSelectElement;
    const criteria = Array.from(sortSelect.options).map((o) => o.textContent ?? '');
    expect(criteria).toEqual(['Proximité', 'Formation', 'Contacts']);
    expect(criteria.some((c) => /importance|qualité|score/i.test(c))).toBe(false);
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

  it('MIA-3 — selecting a zone sets the shared orientation and SWITCHES it on a card click (no reload)', async () => {
    renderZones();
    await waitFor(() => expect(screen.getAllByRole('article')).toHaveLength(4));
    // The shared panel (shell column) reads this orientation; the probe surfaces
    // its label. Default subject = the nearest zone (ob-xau-2-mitigated, 2384–2386).
    await waitFor(() => expect(screen.getByTestId('mia-focus').textContent).toContain('384,00'));
    // Click a different card → the orientation changes, without any navigation.
    fireEvent.click(card('ob-xau-1'));
    await waitFor(() => expect(screen.getByTestId('mia-focus').textContent).toContain('375,00'));
    expect(pushMock).not.toHaveBeenCalled();
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
    await waitFor(() => expect(screen.getByTestId('mia-focus').textContent).toContain('381,00'));
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
    // The factual context line on the right stays. VZ-5 unframed it and dropped
    // the market/timeframe it repeated from the selector — what this guard owns
    // is that the FACTS survive, not the pill they used to sit in.
    expect(document.querySelector('.pghead .zstatus')).not.toBeNull();
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

  it('MIA-3/CLN-1 §3 — re-clicking the selected zone clears the orientation (deselect), no navigation', async () => {
    renderZones();
    await waitFor(() => expect(screen.getAllByRole('article')).toHaveLength(4));
    // A zone is selected by default → the shared orientation carries its label,
    // and exactly one card is highlighted.
    await waitFor(() =>
      expect(screen.getByTestId('mia-focus').textContent!.length).toBeGreaterThan(0),
    );
    const selected = document.querySelector<HTMLElement>('.zone.zsel');
    expect(selected).toBeTruthy();

    // Re-click the SAME (selected) card → deselect: the orientation is cleared
    // (the shared panel's subject block then disappears — tested at the MiaPanel
    // level), no card highlighted, and NO navigation. The conversation itself is
    // untouched by clearing focus (proven in ChatProvider.test).
    fireEvent.click(selected!);
    await waitFor(() => expect(screen.getByTestId('mia-focus').textContent).toBe(''));
    expect(document.querySelector('.zone.zsel')).toBeNull();
    expect(pushMock).not.toHaveBeenCalled();
  });

  // ── VZ-5 — filter bar ─────────────────────────────────────────────────────
  it('VZ-5 — the status line keeps ONLY the zone count, unframed', async () => {
    renderZones();
    await waitFor(() => expect(screen.getAllByRole('article')).toHaveLength(4));

    const status = screen.getByTestId('zones-status');
    // The count is the one thing the left side does not already say.
    expect(status.textContent).toMatch(/\d+\s+zones? suivies?/);
    // The market and the timeframe left with the pill — the selector states them.
    expect(status.textContent).not.toContain('XAU');
    expect(status.textContent).not.toContain('M15');
    // No framed pill any more.
    expect(document.querySelector('.pghead .livebadge')).toBeNull();
  });

  it('VZ-5 — no uppercase étiquette above the filters, and the sort is a dropdown', async () => {
    renderZones();
    await waitFor(() => expect(screen.getAllByRole('article')).toHaveLength(4));

    const bar = document.querySelector('.zbar');
    expect(bar).not.toBeNull();
    // The two 9px uppercase labels that sat above the pill rows are gone.
    expect(bar!.querySelector('.uppercase')).toBeNull();
    // The filters stay a segmented control, named for assistive tech.
    expect(within(bar as HTMLElement).getByRole('group', { name: 'Filtrer les zones' })).toBeTruthy();
    // The sort is a discreet dropdown, not a third row of equal-weight pills.
    const sort = screen.getByLabelText('Trié par') as HTMLSelectElement;
    expect(sort.tagName).toBe('SELECT');
    expect(Array.from(sort.options).map((o) => o.textContent)).toEqual([
      'Proximité',
      'Formation',
      'Contacts',
    ]);
  });

  it('VZ-5 — the sort dropdown still drives the sort (logic untouched)', async () => {
    renderZones();
    await waitFor(() => expect(screen.getAllByRole('article')).toHaveLength(4));

    const sort = screen.getByLabelText('Trié par') as HTMLSelectElement;
    expect(sort.value).toBe('proximity');
    fireEvent.change(sort, { target: { value: 'formation' } });
    await waitFor(() => expect(sort.value).toBe('formation'));
    // Re-sorting reorders, it never filters anything out.
    expect(screen.getAllByRole('article')).toHaveLength(4);
  });

  it('VZ-5 — the price-freshness line sits under the bar, not in the header', async () => {
    renderZones();
    await waitFor(() => expect(screen.getAllByRole('article')).toHaveLength(4));

    const fresh = document.querySelector('[data-testid="price-freshness"]');
    if (fresh) {
      expect(fresh.closest('.pghead')).toBeNull();
      expect(fresh.classList.contains('zfresh')).toBe(true);
    }
  });

  it('VZ-5 — the group heading is plain sentence case, not an uppercased label', async () => {
    renderZones();
    await waitFor(() => expect(screen.getAllByRole('article')).toHaveLength(4));

    const seps = Array.from(document.querySelectorAll('.zsep'));
    expect(seps.length).toBeGreaterThan(0);
    for (const s of seps) {
      // The string itself was already sentence case — the shouting came from the
      // CSS. Guard the copy so nobody re-uppercases it in the markup instead.
      expect(s.textContent).not.toBe(s.textContent!.toUpperCase());
    }
  });
});
