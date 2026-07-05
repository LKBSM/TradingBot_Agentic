import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { ChatProvider, useChat } from '../ChatProvider';
import { STORAGE_KEY } from '@/lib/chat/thread-store';
import type { ChatSignalContext } from '@/lib/chat/types';

// Keep the real error classes; override only askSentinel.
const askSentinelMock = vi.fn();
vi.mock('@/lib/chat/api-client', async (importActual) => {
  const actual = await importActual<typeof import('@/lib/chat/api-client')>();
  return { ...actual, askSentinel: (...args: unknown[]) => askSentinelMock(...args) };
});

const SIGNAL: ChatSignalContext = { id: 'sig-1', instrument: 'XAUUSD', timeframe: 'H1' };

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
  window.localStorage.clear();
});

describe('ChatProvider.askFreeForm', () => {
  it('pushes the user turn and the assistant answer on success', async () => {
    askSentinelMock.mockResolvedValue({
      text: 'Le marché est en phase de consolidation.',
      blockedReason: null,
      toolCallsMade: [],
    });
    renderHarness();

    fireEvent.click(screen.getByText('ask'));

    expect(await screen.findByText('Quelle conviction ?')).toBeInTheDocument();
    expect(
      await screen.findByText('Le marché est en phase de consolidation.'),
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

  it('does not replay an empty-content turn in history (avoids the 422)', async () => {
    // 1st answer is display-only: the model toggled a layer with no prose, so the
    // assistant turn renders empty. Replaying content:"" would 422 the backend
    // ("format ou longueur") — it must be dropped from the next request's history.
    askSentinelMock.mockResolvedValueOnce({
      text: '',
      blockedReason: null,
      toolCallsMade: [],
      viewActions: [{ action: 'set_layer_visibility', params: { layers: ['fvg', 'ob'], visible: false } }],
    });
    askSentinelMock.mockResolvedValueOnce({
      text: 'Deuxième réponse.',
      blockedReason: null,
      toolCallsMade: [],
      viewActions: [],
    });
    renderHarness();

    fireEvent.click(screen.getByText('ask'));
    await waitFor(() => expect(askSentinelMock).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(screen.getByTestId('loading').textContent).toBe('false'));

    fireEvent.click(screen.getByText('ask'));
    await waitFor(() => expect(askSentinelMock).toHaveBeenCalledTimes(2));

    const secondCall = askSentinelMock.mock.calls[1];
    expect(secondCall).toBeDefined();
    const secondArgs = secondCall![0] as {
      history: Array<{ role: string; content: string }>;
    };
    // No empty-content message survives; the prior user question still does.
    for (const m of secondArgs.history) {
      expect(m.content.length).toBeGreaterThan(0);
    }
    expect(secondArgs.history).toEqual([{ role: 'user', content: 'Quelle conviction ?' }]);
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

/** Harness for combo-scoped threads (the /app sidebar path via openForCombo). */
function ComboHarness() {
  const { turns, openForCombo, askFreeForm, resetTurns, recentThreads } =
    useChat();
  return (
    <div>
      <button
        type="button"
        onClick={() => openForCombo({ instrument: 'XAUUSD', timeframe: 'H1' })}
      >
        combo-h1
      </button>
      <button
        type="button"
        onClick={() => openForCombo({ instrument: 'XAUUSD', timeframe: 'H4' })}
      >
        combo-h4
      </button>
      <button type="button" onClick={() => void askFreeForm('Ma question ?')}>
        ask-combo
      </button>
      <button type="button" onClick={resetTurns}>
        reset-combo
      </button>
      <span data-testid="turn-count">{turns.length}</span>
      <span data-testid="recents">{recentThreads.map((t) => t.id).join(',')}</span>
      <ul>
        {turns.map((t) => (
          <li key={t.id}>{t.text}</li>
        ))}
      </ul>
    </div>
  );
}

describe('ChatProvider thread scoping & persistence (client-only)', () => {
  it("keeps each combo's conversation and restores it when coming back", async () => {
    askSentinelMock.mockResolvedValue({
      text: 'Réponse H1.',
      blockedReason: null,
      toolCallsMade: [],
    });
    render(
      <ChatProvider>
        <ComboHarness />
      </ChatProvider>,
    );

    fireEvent.click(screen.getByText('combo-h1'));
    fireEvent.click(screen.getByText('ask-combo'));
    expect(await screen.findByText('Réponse H1.')).toBeInTheDocument();

    // Switch to H4 → its own fresh thread, H1's turns are NOT shown (no mixing).
    fireEvent.click(screen.getByText('combo-h4'));
    await waitFor(() =>
      expect(screen.getByTestId('turn-count').textContent).toBe('0'),
    );
    expect(screen.queryByText('Réponse H1.')).not.toBeInTheDocument();

    // Back to H1 → the conversation is restored intact.
    fireEvent.click(screen.getByText('combo-h1'));
    expect(await screen.findByText('Réponse H1.')).toBeInTheDocument();
    expect(screen.getByTestId('turn-count').textContent).toBe('2');
    expect(screen.getByTestId('recents').textContent).toBe('app:XAUUSD:H1');
  });

  it('persists combo threads to localStorage and rehydrates a fresh provider', async () => {
    askSentinelMock.mockResolvedValue({
      text: 'Réponse persistée.',
      blockedReason: null,
      toolCallsMade: [],
    });
    const first = render(
      <ChatProvider>
        <ComboHarness />
      </ChatProvider>,
    );
    fireEvent.click(screen.getByText('combo-h1'));
    fireEvent.click(screen.getByText('ask-combo'));
    expect(await screen.findByText('Réponse persistée.')).toBeInTheDocument();
    await waitFor(() =>
      expect(window.localStorage.getItem(STORAGE_KEY)).toContain(
        'Réponse persistée.',
      ),
    );
    first.unmount();

    // Fresh provider (simulates a page refresh): the thread comes back from
    // localStorage — no server involved.
    render(
      <ChatProvider>
        <ComboHarness />
      </ChatProvider>,
    );
    fireEvent.click(screen.getByText('combo-h1'));
    expect(await screen.findByText('Réponse persistée.')).toBeInTheDocument();
    expect(await screen.findByText('Ma question ?')).toBeInTheDocument();
  });

  it('resetTurns clears ONLY the active thread, in memory and in storage', async () => {
    askSentinelMock
      .mockResolvedValueOnce({ text: 'Réponse H1.', blockedReason: null, toolCallsMade: [] })
      .mockResolvedValueOnce({ text: 'Réponse H4.', blockedReason: null, toolCallsMade: [] });
    render(
      <ChatProvider>
        <ComboHarness />
      </ChatProvider>,
    );

    fireEvent.click(screen.getByText('combo-h1'));
    fireEvent.click(screen.getByText('ask-combo'));
    expect(await screen.findByText('Réponse H1.')).toBeInTheDocument();
    fireEvent.click(screen.getByText('combo-h4'));
    fireEvent.click(screen.getByText('ask-combo'));
    expect(await screen.findByText('Réponse H4.')).toBeInTheDocument();

    // Reset while H4 is active: H4 gone everywhere, H1 untouched.
    fireEvent.click(screen.getByText('reset-combo'));
    await waitFor(() =>
      expect(screen.getByTestId('turn-count').textContent).toBe('0'),
    );
    await waitFor(() => {
      const raw = window.localStorage.getItem(STORAGE_KEY) ?? '';
      expect(raw).not.toContain('Réponse H4.');
      expect(raw).toContain('Réponse H1.');
    });
    fireEvent.click(screen.getByText('combo-h1'));
    expect(await screen.findByText('Réponse H1.')).toBeInTheDocument();
  });

  it('never persists landing signal threads (only app:* combo threads)', async () => {
    askSentinelMock.mockResolvedValue({
      text: 'Réponse signal.',
      blockedReason: null,
      toolCallsMade: [],
    });
    renderHarness(); // opens for SIGNAL (id 'sig-1')
    fireEvent.click(screen.getByText('ask'));
    expect(await screen.findByText('Réponse signal.')).toBeInTheDocument();

    await waitFor(() => {
      const raw = window.localStorage.getItem(STORAGE_KEY);
      expect(raw === null || !raw.includes('sig-1')).toBe(true);
    });
  });
});
