import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ChatProvider, useChat } from '../ChatProvider';
import { ChatMessage } from '../ChatMessage';
import type { InsightSignalV2 } from '@/types/insight';

/**
 * Chantier 5.A — smoke "e2e équivalent" de l'intégration chatbot.
 *
 * Contrairement à ChatProvider.test (qui mocke askSentinel), ce smoke n'mocke
 * QUE `fetch` : il exerce donc la pile RÉELLE api-client → ChatProvider →
 * ChatMessage, au plus près d'un vrai bout-en-bout frontend. Le pendant
 * Playwright (tests/e2e/chatbot-backend-integration.spec.ts) couvre la même
 * matrice dans un navigateur avec page.route().
 */

const SIGNAL = { id: 'sig-1', instrument: 'XAUUSD', timeframe: 'H1' } as InsightSignalV2;

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

/** Renders each turn through the real ChatMessage so the badge path is real. */
function Harness({ question }: { question: string }) {
  const { turns, openFor, askFreeForm } = useChat();
  return (
    <div>
      <button type="button" onClick={() => openFor(SIGNAL)}>
        open
      </button>
      <button type="button" onClick={() => void askFreeForm(question)}>
        ask
      </button>
      {turns.map((t) => (
        <ChatMessage key={t.id} role={t.role} text={t.text} blockedReason={t.blockedReason} />
      ))}
    </div>
  );
}

function drive(question = 'Décris la structure XAUUSD H1') {
  render(
    <ChatProvider>
      <Harness question={question} />
    </ChatProvider>,
  );
  fireEvent.click(screen.getByText('open'));
  fireEvent.click(screen.getByText('ask'));
}

let fetchMock: ReturnType<typeof vi.fn>;

beforeEach(() => {
  fetchMock = vi.fn();
  vi.stubGlobal('fetch', fetchMock);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('chatbot backend integration — smoke', () => {
  it('Scénario 1: question descriptive → réponse non bloquée + bon endpoint/préfixe', async () => {
    fetchMock.mockResolvedValue(
      jsonResponse(200, {
        content: 'XAUUSD H1 est en phase de consolidation sous résistance.',
        blocked_reason: null,
        tool_calls_made: [],
      }),
    );

    drive('Décris la structure XAUUSD H1');

    expect(
      await screen.findByText('XAUUSD H1 est en phase de consolidation sous résistance.'),
    ).toBeInTheDocument();
    // No "recadrée" badge on a normal answer.
    expect(screen.queryByText('Question recadrée')).not.toBeInTheDocument();

    // Real api-client hit the backend endpoint with the T1 context preamble.
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe('/api/chatbot/message');
    const body = JSON.parse(init.body as string);
    expect(body.user_message).toBe('[Lecture en cours : XAUUSD H1]\nDécris la structure XAUUSD H1');
  });

  it('Scénario 2: demande d’action → template de refus + indicateur blocked_reason', async () => {
    fetchMock.mockResolvedValue(
      jsonResponse(200, {
        content: 'Je décris les conditions du marché. La décision d’agir t’appartient.',
        blocked_reason: 'trade_request',
        tool_calls_made: [],
      }),
    );

    drive('Dois-je acheter EURUSD ?');

    expect(await screen.findByText(/La décision d’agir/)).toBeInTheDocument();
    // The discreet redirect indicator is rendered.
    expect(await screen.findByText('Question recadrée')).toBeInTheDocument();
  });

  it('Scénario 3: backend non bootstrappé (503) → message fallback user-friendly', async () => {
    fetchMock.mockResolvedValue(jsonResponse(503, { detail: 'Chatbot service not configured' }));

    drive('Bonjour');

    await waitFor(() =>
      expect(
        screen.getByText(/mode chatbot en direct n'est pas disponible/i),
      ).toBeInTheDocument(),
    );
  });
});
