import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { ChatProvider, useChat } from '../ChatProvider';
import type { InsightSignalV2 } from '@/types/insight';

// Keep the real error classes; override only askSentinel.
const askSentinelMock = vi.fn();
vi.mock('@/lib/chat/api-client', async (importActual) => {
  const actual = await importActual<typeof import('@/lib/chat/api-client')>();
  return { ...actual, askSentinel: (...args: unknown[]) => askSentinelMock(...args) };
});

const SIGNAL = { id: 'sig-1', instrument: 'XAUUSD', timeframe: 'H1' } as InsightSignalV2;

/** Minimal harness exercising the provider through its public hook. */
function Harness() {
  const { turns, isLoading, openFor, askFreeForm } = useChat();
  return (
    <div>
      <button type="button" onClick={() => openFor(SIGNAL)}>
        open
      </button>
      <button type="button" onClick={() => void askFreeForm('Quelle conviction ?')}>
        ask
      </button>
      <span data-testid="loading">{String(isLoading)}</span>
      <ul>
        {turns.map((t) => (
          <li key={t.id} data-role={t.role} data-blocked={t.blockedReason ?? ''}>
            {t.text}
          </li>
        ))}
      </ul>
    </div>
  );
}

function renderHarness() {
  render(
    <ChatProvider>
      <Harness />
    </ChatProvider>,
  );
  // Open the panel for SIGNAL first — askFreeForm requires an activeSignal.
  fireEvent.click(screen.getByText('open'));
}

afterEach(() => {
  askSentinelMock.mockReset();
});

describe('ChatProvider.askFreeForm', () => {
  it('pushes the user turn and the assistant answer on success', async () => {
    askSentinelMock.mockResolvedValue({
      text: 'Conviction 62/100, phase de consolidation.',
      blockedReason: null,
      toolCallsMade: [],
    });
    renderHarness();

    fireEvent.click(screen.getByText('ask'));

    expect(await screen.findByText('Quelle conviction ?')).toBeInTheDocument();
    expect(
      await screen.findByText('Conviction 62/100, phase de consolidation.'),
    ).toBeInTheDocument();
  });

  it('carries blockedReason through to the assistant turn', async () => {
    askSentinelMock.mockResolvedValue({
      text: 'Je décris les conditions du marché. La décision t’appartient.',
      blockedReason: 'trade_request',
      toolCallsMade: [],
    });
    renderHarness();

    fireEvent.click(screen.getByText('ask'));

    const assistantTurn = await screen.findByText(/La décision t/);
    expect(assistantTurn.closest('li')?.dataset.blocked).toBe('trade_request');
  });

  it('renders a friendly fallback when the backend is unavailable (503)', async () => {
    const { ChatApiUnavailableError } = await import('@/lib/chat/api-client');
    askSentinelMock.mockRejectedValue(
      new ChatApiUnavailableError('chatbot_unavailable', '503'),
    );
    renderHarness();

    fireEvent.click(screen.getByText('ask'));

    expect(
      await screen.findByText(/mode chatbot en direct n'est pas disponible/i),
    ).toBeInTheDocument();
  });

  it('toggles isLoading false after the call resolves', async () => {
    askSentinelMock.mockResolvedValue({ text: 'ok', blockedReason: null, toolCallsMade: [] });
    renderHarness();

    fireEvent.click(screen.getByText('ask'));

    await waitFor(() =>
      expect(screen.getByTestId('loading').textContent).toBe('false'),
    );
  });
});
