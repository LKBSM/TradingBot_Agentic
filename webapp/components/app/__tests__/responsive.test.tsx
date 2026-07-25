import { fireEvent, render, screen, waitFor, within } from '@/components/test-utils';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { AppWorkspace } from '../AppWorkspace';
import { ChatProvider } from '@/components/chat/ChatProvider';
import { ChartViewProvider } from '@/lib/chart/viewState';
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

// Stub the candlestick chart (lightweight-charts needs a real canvas).
vi.mock('@/components/app/ReadingChart', () => ({
  ReadingChart: () => <div data-testid="reading-chart" />,
}));

// AppWorkspace syncs the active combo with the URL (NAV-01/02); stub the
// app-router hooks so the workspace renders under jsdom.
vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
  usePathname: () => '/app',
  useSearchParams: () => new URLSearchParams(),
}));

/** Drive useMediaQuery / useStackedLayout by stubbing window.matchMedia. */
function stubMatchMedia(matches: boolean) {
  vi.stubGlobal(
    'matchMedia',
    (query: string) =>
      ({
        matches,
        media: query,
        onchange: null,
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        addListener: vi.fn(),
        removeListener: vi.fn(),
        dispatchEvent: vi.fn(),
      }) as unknown as MediaQueryList,
  );
}

function renderApp() {
  return render(
    <ChatProvider>
      <ChartViewProvider>
        <AppWorkspace dataSource="live" />
      </ChartViewProvider>
    </ChatProvider>,
  );
}

beforeEach(() => {
  fetchMock.mockReset();
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('responsive layout — desktop (≥1280px / xl)', () => {
  it('renders only the reading; the rail + chat are owned by the product shell', async () => {
    // matchMedia(false) → the stacked-layout query does NOT match → desktop path.
    fetchMock.mockResolvedValue(FIXTURE_XAU_M15);
    stubMatchMedia(false);
    render(
      <ChatProvider>
        <ChartViewProvider>
          <AppWorkspace
            dataSource="live"
            initialCombo={{ instrument: 'XAUUSD', timeframe: 'M15' }}
          />
        </ChartViewProvider>
      </ChatProvider>,
    );

    // The reading renders in the shell's centre column.
    await waitFor(() =>
      expect(screen.getByText('Tendance haussière')).toBeInTheDocument(),
    );
    // No mobile tab bar on desktop.
    expect(screen.queryAllByRole('tab')).toHaveLength(0);
    // The instrument selector and chat are NO LONGER inside the workspace on
    // desktop — the product shell (components/shell) owns them now.
    expect(
      screen.queryByRole('navigation', { name: /combinaisons disponibles/i }),
    ).toBeNull();
    expect(
      screen.queryByRole('complementary', { name: /assistant m.i.a agent/i }),
    ).toBeNull();
  });
});

describe('responsive layout — phone + tablet (<1280px)', () => {
  it('renders a three-tab bar (Marchés · Lecture · Chat)', async () => {
    stubMatchMedia(true);
    renderApp();

    await waitFor(() =>
      expect(screen.getByRole('tab', { name: /Marchés/ })).toBeInTheDocument(),
    );
    expect(screen.getByRole('tab', { name: /Lecture/ })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: /Chat/ })).toBeInTheDocument();
  });

  it('shows the Marchés tab first with the idle header', async () => {
    stubMatchMedia(true);
    renderApp();

    await waitFor(() =>
      expect(screen.getByText('Espace de lecture')).toBeInTheDocument(),
    );
    // The instruments list is the active tabpanel.
    expect(screen.getByText('Or (XAU/USD)')).toBeInTheDocument();
  });

  it('jumps to the Lecture tab when a combo is selected', async () => {
    fetchMock.mockResolvedValue(FIXTURE_XAU_M15);
    stubMatchMedia(true);
    renderApp();

    const marketsPanel = await screen.findByRole('tabpanel');
    fireEvent.click(within(marketsPanel).getAllByRole('button')[0]!);

    // Switched to Lecture → the reading renders.
    await waitFor(() =>
      expect(screen.getByText('Tendance haussière')).toBeInTheDocument(),
    );
    // Header now reflects the active combo.
    expect(screen.getByText(/Or \(XAU\/USD\) · 15 minutes/)).toBeInTheDocument();
  });

  it('shows the chat sidebar when the Chat tab is opened', async () => {
    stubMatchMedia(true);
    renderApp();

    const chatTab = await screen.findByRole('tab', { name: /Chat/ });
    // Radix Tabs use automatic activation (on focus); jsdom click doesn't move
    // focus, so drive the focus event directly.
    fireEvent.focus(chatTab);
    fireEvent.click(chatTab);

    await waitFor(() =>
      expect(
        screen.getByRole('complementary', { name: /assistant m.i.a agent/i }),
      ).toBeInTheDocument(),
    );
  });
});
