import { render as rtlRender, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { NextIntlClientProvider } from 'next-intl';
import { ZoneDetail } from '../ZoneDetail';
import { ChatProvider } from '@/components/chat/ChatProvider';
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

vi.mock('@/lib/chat/api-client', async (importActual) => {
  const actual = await importActual<typeof import('@/lib/chat/api-client')>();
  return { ...actual, askSentinelStream: () => Promise.resolve({ text: '', blockedReason: null, toolCallsMade: [], viewActions: [] }) };
});

// lightweight-charts does not paint in jsdom (and the sheet is not what we are
// testing here) — the chart is stubbed to a marker.
vi.mock('@/components/app/ReadingChart', () => ({
  ReadingChart: () => <div data-testid="reading-chart-stub" />,
}));

vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
  usePathname: () => '/zones/ob-xau-1',
  useSearchParams: () => new URLSearchParams('instrument=XAUUSD&timeframe=M15'),
}));

type Reading = typeof FIXTURE_XAU_M15;

function renderDetail(zoneId: string) {
  return rtlRender(
    <NextIntlClientProvider locale="fr" messages={messages}>
      <ChatProvider>
        <ZoneDetail zoneId={zoneId} />
      </ChatProvider>
    </NextIntlClientProvider>,
  );
}

/** Deep-clone the fixture so a test can bend one field without leaking. */
function reading(mutate?: (r: Reading) => void): Reading {
  const clone = JSON.parse(JSON.stringify(FIXTURE_XAU_M15)) as Reading;
  mutate?.(clone);
  return clone;
}

/** FR copy uses the typographic apostrophe; assertions stay agnostic. */
function flat(s: string | null | undefined): string {
  // Prices use a NARROW NO-BREAK SPACE as thousands separator (U+202F) and
  // the FR copy a typographic apostrophe — fold both so assertions read plainly.
  return (s ?? '')
    .replace(/[’]/g, "'")
    .replace(/[\u202f\u00a0]/g, ' ')
    .replace(/\s+/g, ' ');
}

beforeEach(() => {
  fetchMock.mockReset();
  fetchMock.mockResolvedValue(FIXTURE_XAU_M15);
  window.localStorage.clear();
});

describe('VZ-4 — zone sheet', () => {
  it('renders the sheet for a REAL engine id, in the mission order', async () => {
    renderDetail('ob-xau-1');
    const sheet = await screen.findByTestId('zone-detail');
    expect(sheet.getAttribute('data-zone-id')).toBe('ob-xau-1');

    // Header: type + direction + the band, as facts.
    expect(screen.getByRole('heading', { level: 1 }).textContent).toContain('Order Block');
    expect(flat(sheet.textContent)).toContain('2 375,00');
    expect(flat(sheet.textContent)).toContain('2 378,00');

    // Sections that DO have data, in reading order. A section the engine has
    // no data for is simply ABSENT — never a placeholder (mission §0): this OB
    // carries no `origin` and no contacts, so only « État actuel » renders.
    const headings = Array.from(sheet.querySelectorAll('h2')).map((h) => flat(h.textContent));
    expect(headings).toEqual(['État actuel']);

    // The honesty line closes the document, on every zone.
    expect(flat(sheet.textContent)).toContain('M.I.A rapporte des faits');
  });

  it('renders the remaining sections when the engine DOES carry their data', async () => {
    fetchMock.mockResolvedValue(
      reading((r) => {
        const ob = r.structure.order_blocks.find((z) => z.id === 'ob-xau-1')!;
        (ob as { contacts?: unknown[] }).contacts = [
          { at: '2026-05-26T08:00:00+00:00', level: 2377.2, outcome: 'entry_exit' },
        ];
        (ob as { origin?: unknown }).origin = {
          kind: 'bos',
          direction: 'bullish',
          at: '2026-05-26T07:30:00+00:00',
          level: 2379.5,
        };
      }),
    );
    renderDetail('ob-xau-1');
    const sheet = await screen.findByTestId('zone-detail');
    await waitFor(() =>
      expect(sheet.querySelectorAll('h2').length).toBeGreaterThan(1),
    );
    const headings = Array.from(sheet.querySelectorAll('h2')).map((h) => flat(h.textContent));
    // Mission §0 order: État actuel → Historique des contacts → Ce qui l'a créée.
    expect(headings).toEqual([
      'État actuel',
      'Historique des contacts',
      "Ce qui l'a créée",
    ]);
  });

  it('an id the reading no longer carries renders an honest state, never a look-alike', async () => {
    renderDetail('ob-does-not-exist');
    const missing = await screen.findByTestId('zone-detail-missing');
    expect(flat(missing.textContent)).toContain('Zone introuvable');
    // No zone was substituted by resemblance.
    expect(screen.queryByTestId('zone-detail')).toBeNull();
  });

  it('« Zones à l’intérieur » is ABSENT when no containment fact exists', async () => {
    // The fixture zones are adjacent/disjoint — no zone sits inside another.
    renderDetail('ob-xau-1');
    await screen.findByTestId('zone-detail');
    expect(screen.queryByTestId('zone-detail-nested')).toBeNull();
    // And no placeholder / generic filler took its place (mission §0).
    expect(screen.queryByText(/Zones à l['’]intérieur/)).toBeNull();
  });

  it('shows « Zones à l’intérieur » ONLY from a real containment fact', async () => {
    fetchMock.mockResolvedValue(
      reading((r) => {
        // A gap strictly inside the OB's own band — real geometry, real bounds.
        r.structure.fair_value_gaps.push({
          id: 'fvg-inside-ob',
          direction: 'bullish',
          level_high: 2377.0,
          level_low: 2376.0,
          status: 'partially_filled',
          created_at: '2026-05-26T08:00:00+00:00',
          tested: true,
          user_flagged: false,
        } as Reading['structure']['fair_value_gaps'][number]);
      }),
    );
    renderDetail('ob-xau-1');
    const nested = await screen.findByTestId('zone-detail-nested');
    expect(flat(nested.textContent)).toContain("à l'intérieur de cette zone");
    // The neighbour's REAL engine status — never a fabricated percentage.
    expect(flat(nested.textContent)).toContain('partiellement comblée');
    expect(nested.textContent).not.toMatch(/\d+([.,]\d+)?\s*%/);
  });

  it('the comblement appears EXACTLY ONCE, as the current state (VZ-4 §1)', async () => {
    fetchMock.mockResolvedValue(
      reading((r) => {
        const fvg = r.structure.fair_value_gaps.find((z) => z.id === 'fvg-xau-2-partial')!;
        // Deepest penetration published by the engine → 50 % of the 1.0 band.
        (fvg as { fill_level?: number }).fill_level = 2382.0;
        // Four contacts whose depths PLATEAU — exactly the shape that made the
        // old per-row rendering print the same figure on line after line.
        (fvg as { contacts?: unknown[] }).contacts = [
          { at: '2026-05-26T07:00:00+00:00', level: 2382.4, outcome: 'edge_touch' },
          { at: '2026-05-26T08:00:00+00:00', level: 2382.0, outcome: 'entry_exit' },
          { at: '2026-05-26T09:00:00+00:00', level: 2382.3, outcome: 'entry_exit' },
          { at: '2026-05-26T10:00:00+00:00', level: 2382.2, outcome: 'edge_touch' },
        ];
      }),
    );
    renderDetail('fvg-xau-2-partial');
    const sheet = await screen.findByTestId('zone-detail');
    await waitFor(() => expect(screen.getByTestId('zone-detail-fill')).toBeTruthy());

    // ONE fill bar, ONE printed comblement value.
    expect(sheet.querySelectorAll('[role="progressbar"]').length).toBe(1);
    const fillValues = sheet.querySelectorAll('.zdt-fillval');
    expect(fillValues.length).toBe(1);
    const printed = flat(fillValues[0]!.textContent);

    // That exact figure appears NOWHERE else on the sheet — the old bug
    // repeated it on every contact row, including edge touches that filled
    // nothing (VZ-4 §1). The height-in-% of the header is a DIFFERENT figure
    // and is allowed to coexist, so we count the comblement value itself.
    const whole = flat(sheet.textContent);
    expect(whole.split(printed).length - 1).toBe(1);

    // And no contact row carries a percentage at all.
    const rows = Array.from(sheet.querySelectorAll('.zdt-row'));
    expect(rows.length).toBeGreaterThan(0);
    for (const row of rows) {
      expect(row.textContent ?? '').not.toMatch(/%/);
    }
  });

  it('keeps only three contacts before the fold, the rest behind an explicit count', async () => {
    fetchMock.mockResolvedValue(
      reading((r) => {
        const ob = r.structure.order_blocks.find((z) => z.id === 'ob-xau-1')!;
        (ob as { contacts?: unknown[] }).contacts = Array.from({ length: 7 }, (_, i) => ({
          at: `2026-05-26T0${i}:00:00+00:00`,
          level: 2376 + i * 0.1,
          outcome: i % 2 === 0 ? 'edge_touch' : 'entry_exit',
        }));
      }),
    );
    renderDetail('ob-xau-1');
    const sheet = await screen.findByTestId('zone-detail');
    await waitFor(() => expect(sheet.querySelectorAll('.zdt-row').length).toBe(3));
    // 7 contacts − 3 visible = 4 behind the fold, named by an explicit count.
    expect(flat(sheet.textContent)).toContain('4 contacts précédents');
  });
});
