import Anthropic from '@anthropic-ai/sdk';
import { NextRequest } from 'next/server';
import {
  DEFAULT_MODEL,
  SENTINEL_SYSTEM_PROMPT,
} from '@/lib/chat/system-prompt';
import { buildSignalSummary } from '@/lib/chat/signal-summary';
import type { InsightSignalV2 } from '@/types/insight';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

interface ChatRequest {
  signal: InsightSignalV2;
  question: string;
  /** Optional history for multi-turn (last N user/assistant exchanges). */
  history?: ReadonlyArray<{ role: 'user' | 'assistant'; content: string }>;
}

// Hard limit on user-supplied question length — protects against abuse + caps
// LLM cost. 2 KB is more than enough for a free-form trading question.
const MAX_QUESTION_CHARS = 2000;
const MAX_HISTORY_TURNS = 6;

function jsonError(status: number, code: string, message: string): Response {
  return new Response(JSON.stringify({ error: { code, message } }), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

export async function POST(req: NextRequest) {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    return jsonError(
      503,
      'no_api_key',
      "Le chatbot LLM n'est pas configuré sur ce déploiement. Le frontend utilise les réponses scriptées V1.",
    );
  }

  let payload: ChatRequest;
  try {
    payload = (await req.json()) as ChatRequest;
  } catch {
    return jsonError(400, 'bad_json', 'Corps de requête JSON invalide.');
  }

  if (!payload.signal || typeof payload.signal !== 'object') {
    return jsonError(400, 'missing_signal', 'Le champ `signal` est requis.');
  }
  if (typeof payload.question !== 'string' || payload.question.trim() === '') {
    return jsonError(400, 'missing_question', 'Le champ `question` est requis.');
  }
  if (payload.question.length > MAX_QUESTION_CHARS) {
    return jsonError(
      400,
      'question_too_long',
      `La question ne doit pas dépasser ${MAX_QUESTION_CHARS} caractères.`,
    );
  }

  const signalSummary = buildSignalSummary(payload.signal);
  const trimmedHistory = (payload.history ?? []).slice(-MAX_HISTORY_TURNS);

  const client = new Anthropic({ apiKey });

  // Build the messages array. The signal context is the FIRST user turn
  // marked with cache_control so Anthropic caches it across requests for
  // the same signal (cache key = signal serialisation + system prompt).
  const messages: Anthropic.MessageParam[] = [
    {
      role: 'user',
      content: [
        {
          type: 'text',
          text: `Voici la lecture de marché en cours. Tu réponds à mes questions à partir de ce contexte.\n\n${signalSummary}`,
          cache_control: { type: 'ephemeral' },
        },
      ],
    },
    {
      role: 'assistant',
      content:
        "Très bien, j'ai la lecture en tête. Pose-moi ta question — je décris la structure, j'explique le jargon, je contextualise les événements, mais je ne donne aucune instruction d'achat ou de vente.",
    },
    ...trimmedHistory.map<Anthropic.MessageParam>((h) => ({
      role: h.role,
      content: h.content,
    })),
    { role: 'user', content: payload.question },
  ];

  const encoder = new TextEncoder();

  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      const send = (event: string, data: unknown) => {
        controller.enqueue(
          encoder.encode(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`),
        );
      };

      try {
        const streamResponse = await client.messages.stream({
          model: DEFAULT_MODEL,
          max_tokens: 1024,
          system: [
            {
              type: 'text',
              text: SENTINEL_SYSTEM_PROMPT,
              cache_control: { type: 'ephemeral' },
            },
          ],
          messages,
        });

        for await (const event of streamResponse) {
          if (
            event.type === 'content_block_delta' &&
            event.delta.type === 'text_delta'
          ) {
            send('delta', { text: event.delta.text });
          }
        }

        const finalMessage = await streamResponse.finalMessage();
        send('done', {
          stop_reason: finalMessage.stop_reason,
          usage: finalMessage.usage,
          model: finalMessage.model,
        });
      } catch (err) {
        const message =
          err instanceof Error ? err.message : 'Unknown LLM error';
        send('error', { message });
      } finally {
        controller.close();
      }
    },
  });

  return new Response(stream, {
    headers: {
      'content-type': 'text/event-stream; charset=utf-8',
      'cache-control': 'no-cache, no-store, no-transform',
      connection: 'keep-alive',
      'x-accel-buffering': 'no',
    },
  });
}
