import Anthropic from '@anthropic-ai/sdk';
import { NextRequest } from 'next/server';
import {
  DEFAULT_MODEL,
  SENTINEL_SYSTEM_PROMPT,
  modelForTier,
} from '@/lib/chat/system-prompt';
import { buildSignalSummary } from '@/lib/chat/signal-summary';
import {
  SAFE_FALLBACK_EN,
  SAFE_FALLBACK_FR,
  containsForbiddenToken,
} from '@/lib/chat/forbidden-tokens';
import { classifyUserInput } from '@/lib/chat/adversarial-patterns';
import type { InsightSignalV2 } from '@/types/insight';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

interface ChatRequest {
  signal: InsightSignalV2;
  question: string;
  /** Optional history for multi-turn (last N user/assistant exchanges). */
  history?: ReadonlyArray<{ role: 'user' | 'assistant'; content: string }>;
  /**
   * Subscriber tier for DG-042 model routing. When omitted, defaults to
   * cost-safe Haiku. Resolved from auth/Stripe in V3; for now the client
   * surfaces whatever is known.
   */
  tier?: 'FREE' | 'STARTER' | 'PRO' | 'INSTITUTIONAL';
}

// Hard limit on user-supplied question length — protects against abuse + caps
// LLM cost. 2 KB is more than enough for a free-form trading question.
const MAX_QUESTION_CHARS = 2000;
// DG-114-REDUCED session memory — keep up to 5 prior user/assistant turns
// so the chatbot stays coherent across 3 consecutive questions on the same
// insight without bloating context. Brief asks for 5-turn memory; we cap
// here to enforce it server-side regardless of what the client sends.
const MAX_HISTORY_TURNS = 5;

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

  // DG-112 — pre-LLM adversarial gate. If the user is asking for a
  // buy/sell/timing/sizing/guarantee/jailbreak, short-circuit with the
  // pedagogical refusal *before* paying for an LLM round-trip.
  const adversarial = classifyUserInput(payload.question);
  if (adversarial !== null) {
    const fallback =
      adversarial.language === 'en' ? SAFE_FALLBACK_EN : SAFE_FALLBACK_FR;
    const refusalStream = new ReadableStream<Uint8Array>({
      start(controller) {
        const enc = new TextEncoder();
        const send = (event: string, data: unknown) =>
          controller.enqueue(
            enc.encode(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`),
          );
        send('delta', { text: fallback });
        send('refusal', {
          category: adversarial.category,
          language: adversarial.language,
          pattern_source: adversarial.pattern_source,
        });
        send('done', {
          stop_reason: 'refused_by_gate',
          usage: { input_tokens: 0, output_tokens: 0 },
          model: 'gate-refusal',
          compliance_filtered: true,
        });
        controller.close();
      },
    });
    return new Response(refusalStream, {
      headers: {
        'content-type': 'text/event-stream; charset=utf-8',
        'cache-control': 'no-cache, no-store, no-transform',
        connection: 'keep-alive',
        'x-accel-buffering': 'no',
      },
    });
  }

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
        const selectedModel = modelForTier(payload.tier);
        const streamResponse = await client.messages.stream({
          model: selectedModel,
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

        // Buffer the full text alongside the stream so the post-process
        // compliance filter (DG-112 / forbidden tokens) can inspect the
        // complete response, not just one chunk in isolation.
        let buffered = '';
        for await (const event of streamResponse) {
          if (
            event.type === 'content_block_delta' &&
            event.delta.type === 'text_delta'
          ) {
            buffered += event.delta.text;
            send('delta', { text: event.delta.text });
          }
        }

        const finalMessage = await streamResponse.finalMessage();

        // Forbidden-token gate. Run both FR and EN filters because the
        // assistant follows the user's language (system-prompt rule).
        const offenderFr = containsForbiddenToken(buffered, 'fr');
        const offenderEn = containsForbiddenToken(buffered, 'en');
        const offender = offenderFr ?? offenderEn;
        if (offender) {
          const fallback = offenderEn && !offenderFr ? SAFE_FALLBACK_EN : SAFE_FALLBACK_FR;
          send('forbidden_token', {
            token: offender,
            language: offenderEn && !offenderFr ? 'en' : 'fr',
            safe_fallback: fallback,
          });
        }

        send('done', {
          stop_reason: finalMessage.stop_reason,
          usage: finalMessage.usage,
          model: finalMessage.model,
          compliance_filtered: offender !== null,
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
