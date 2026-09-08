import { fireEvent, render, screen, waitFor } from '@/components/test-utils';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { ChatProvider, useChat } from '../ChatProvider';
import { STORAGE_KEY } from '@/lib/chat/thread-store';
import type { ChatSignalContext } from '@/lib/chat/types';

// Keep the real error classes; override only the streaming call the provider
// now uses (MIA-1). The mock's first arg is still the opts object (with
// `.history`), so the existing call-inspection assertions hold unchanged.
const askSentinelMock = vi.fn();
vi.mock('@/lib/chat/api-client', async (importActual) => {
  const actual = await importActual<typeof import('@/lib/chat/api-client')>();
  return {
    ...actual,
    askSentinelStream: (...args: unknown[]) => askSentinelMock(...args),
  };
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

  it('renders the user message before the network round-trip resolves (MIA-2)', async () => {
    // A stream that never resolves: the provider must surface the user's message
    // and the loading state WITHOUT waiting on the network. This pins the
    // perceived-latency budget — the message is echoed on send, not on reply.
    let release: (() => void) | undefined;
    askSentinelMock.mockImplementation(
      () => new Promise<void>((resolve) => (release = () => resolve())),
    );
    renderHarness();

    fireEvent.click(screen.getByText('ask'));

    // User turn is visible while the request is still in flight…
    expect(await screen.findByText('Quelle conviction ?')).toBeInTheDocument();
    // …in the loading state, and the network was actually invoked (after the echo).
    await waitFor(() => expect(screen.getByTestId('loading').textContent).toBe('true'));
    expect(askSentinelMock).toHaveBeenCalledTimes(1);

    release?.();
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

describe('ChatProvider single conversation (MIA-3 — follows the user)', () => {
  it('keeps ONE conversation across combo switches (orientation ≠ thread)', async () => {
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

    // Switch combo → the SAME conversation stays (decision E): the earlier turn
    // is still shown, nothing is reset, no per-combo split.
    fireEvent.click(screen.getByText('combo-h4'));
    expect(screen.getByText('Réponse H1.')).toBeInTheDocument();
    expect(screen.getByTestId('turn-count').textContent).toBe('2');

    // Ask again after switching → appended to the one conversation (4 turns).
    fireEvent.click(screen.getByText('ask-combo'));
    expect(await screen.findByText('Réponse H4.')).toBeInTheDocument();
    expect(screen.getByText('Réponse H1.')).toBeInTheDocument();
    expect(screen.getByTestId('turn-count').textContent).toBe('4');
    // No per-combo "recents" entries anymore — one continuous conversation.
    expect(screen.getByTestId('recents').textContent).toBe('');
  });

  it('persists the single conversation and rehydrates a fresh provider', async () => {
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

    // Fresh provider (simulates a page refresh): the one conversation comes back
    // from localStorage — no server involved, and no combo needed to restore it.
    render(
      <ChatProvider>
        <ComboHarness />
      </ChatProvider>,
    );
    expect(await screen.findByText('Réponse persistée.')).toBeInTheDocument();
    expect(await screen.findByText('Ma question ?')).toBeInTheDocument();
  });

  it('resetTurns clears the whole conversation, in memory and in storage', async () => {
    askSentinelMock.mockResolvedValue({
      text: 'Réponse à effacer.',
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
    expect(await screen.findByText('Réponse à effacer.')).toBeInTheDocument();

    fireEvent.click(screen.getByText('reset-combo'));
    await waitFor(() =>
      expect(screen.getByTestId('turn-count').textContent).toBe('0'),
    );
    await waitFor(() => {
      const raw = window.localStorage.getItem(STORAGE_KEY) ?? '';
      expect(raw).not.toContain('Réponse à effacer.');
    });
  });

  it('sends the selected-zone orientation as a preamble focus, cleared without touching the conversation', async () => {
    askSentinelMock.mockResolvedValue({
      text: 'Réponse orientée.',
      blockedReason: null,
      toolCallsMade: [],
    });

    function FocusHarness() {
      const { openForCombo, setFocus, askFreeForm, turns } = useChat();
      return (
        <div>
          <button type="button" onClick={() => openForCombo({ instrument: 'XAUUSD', timeframe: 'M15' })}>
            combo
          </button>
          <button
            type="button"
            onClick={() => setFocus({ kind: 'zone', zoneId: 'OB_xau_m15_7', label: 'OB ↑ · 4100–4110' })}
          >
            focus-zone
          </button>
          <button type="button" onClick={() => setFocus(null)}>
            deselect
          </button>
          <button type="button" onClick={() => void askFreeForm('Décris cette zone ?')}>
            ask
          </button>
          <span data-testid="count">{turns.length}</span>
          <ul>
            {turns.map((t) => (
              <li key={t.id}>{t.text}</li>
            ))}
          </ul>
        </div>
      );
    }

    render(
      <ChatProvider>
        <FocusHarness />
      </ChatProvider>,
    );
    fireEvent.click(screen.getByText('combo'));
    fireEvent.click(screen.getByText('focus-zone'));
    fireEvent.click(screen.getByText('ask'));
    await waitFor(() => expect(askSentinelMock).toHaveBeenCalledTimes(1));
    const firstArgs = askSentinelMock.mock.calls[0]![0] as { focus?: string | null };
    expect(firstArgs.focus).toBe('[Zone sélectionnée : OB_xau_m15_7]');
    await screen.findByText('Réponse orientée.');

    // Deselect → the conversation is NOT cleared, and the next question carries
    // no zone focus (an off-zone question is still answered).
    fireEvent.click(screen.getByText('deselect'));
    expect(screen.getByTestId('count').textContent).toBe('2');
    fireEvent.click(screen.getByText('ask'));
    await waitFor(() => expect(askSentinelMock).toHaveBeenCalledTimes(2));
    const secondArgs = askSentinelMock.mock.calls[1]![0] as { focus?: string | null };
    expect(secondArgs.focus).toBeNull();
  });

  it('persists the conversation even when opened for a non-combo signal', async () => {
    // Under the single-conversation model every turn lives in the one product
    // thread, so a chat opened for a landing signal now persists too (it is the
    // same conversation the user continues on /app or /zones).
    askSentinelMock.mockResolvedValue({
      text: 'Réponse signal.',
      blockedReason: null,
      toolCallsMade: [],
    });
    renderHarness(); // opens for SIGNAL
    fireEvent.click(screen.getByText('ask'));
    expect(await screen.findByText('Réponse signal.')).toBeInTheDocument();

    await waitFor(() =>
      expect(window.localStorage.getItem(STORAGE_KEY)).toContain('Réponse signal.'),
    );
  });
});
