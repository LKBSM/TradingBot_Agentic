import { useTranslations } from 'next-intl';
import chatbotResponsesJson from '@/mocks/chatbot_responses.json';
import type { ChatbotResponses, ChatbotScript } from '@/types/chatbot';

// Same unsafe-cast pattern as lib/mocks.ts — JSON tuples widen, hand-curated
// fixture is the source of truth. zod validation will replace this at the
// backend integration sprint.
//
// NOTE: this JSON is the FRENCH source of truth for the scripted chatbot prose.
// The human-readable strings (question.text / reply / instrument_label) are now
// ALSO mirrored into the `chatbot` i18n namespace so that translated locales
// render localized prose. Consumers that render on the page resolve the prose
// via `useChatbotScript()` (below); the raw JSON stays as the French fallback
// and keeps the stable id/question-id topology.
const RESPONSES = chatbotResponsesJson as unknown as ChatbotResponses;

export function getChatbotScript(signalId: string): ChatbotScript | null {
  return RESPONSES[signalId] ?? null;
}

/**
 * Maps a stable landing-sample / signal id to its readable key in the
 * `chatbot.scripts.*` i18n namespace. Ids are opaque hashes shared with
 * `landing-samples.ts` and `mocks/chatbot_responses.json`; this table is the
 * single place that translates them into human-readable message keys.
 */
const SCRIPT_KEY_BY_ID: Readonly<Record<string, string>> = {
  '0193c7a42f1b': 'xauM15',
  '0193c7a4ab51': 'eurH1',
  '0193c7a5c8e2': 'quietXauH4',
};

/**
 * Fallback script used when a signal id is not present in the JSON (defensive
 * — should not happen with the mocked demo). Provides a single educational
 * refusal so the UI never renders an empty panel.
 *
 * Kept as the FRENCH source of truth; `useChatbotScript` localizes it via the
 * `chatbot.fallback` namespace.
 */
export const FALLBACK_SCRIPT: ChatbotScript = {
  instrument_label: 'cet instrument',
  questions: [
    {
      id: 'no-context',
      text: 'Pourquoi je ne vois pas de questions ?',
      reply:
        "Je n'ai pas trouvé de scénario scripté pour cette lecture spécifique. C'est temporaire — l'intégration du moteur de réponse en temps réel arrive au prochain sprint d'intégration backend.",
    },
  ],
};

/** Translator bound to the `chatbot` namespace. */
type ChatbotTranslator = ReturnType<typeof useTranslations>;

/**
 * Localizes a raw (French) ChatbotScript against the `chatbot` namespace,
 * preserving the exact id/question-id topology. Falls back to the raw French
 * string whenever a key is missing so nothing ever renders empty and the
 * compliance refusal wording stays intact even before locale files are merged.
 *
 * Pure (no React hooks) so it is safe to call from both Server and Client
 * Components — the caller supplies `t = useTranslations('chatbot')`.
 */
function localizeScript(
  raw: ChatbotScript,
  scriptKey: string,
  t: ChatbotTranslator,
): ChatbotScript {
  const base = `scripts.${scriptKey}`;
  const labelKey = `${base}.instrumentLabel`;
  return {
    instrument_label: t.has(labelKey) ? t(labelKey) : raw.instrument_label,
    questions: raw.questions.map((q) => {
      const textKey = `${base}.questions.${q.id}.text`;
      const replyKey = `${base}.questions.${q.id}.reply`;
      return {
        id: q.id,
        text: t.has(textKey) ? t(textKey) : q.text,
        reply: t.has(replyKey) ? t(replyKey) : q.reply,
      };
    }),
  };
}

function localizeFallback(t: ChatbotTranslator): ChatbotScript {
  const q0 = FALLBACK_SCRIPT.questions[0]!;
  return {
    instrument_label: t.has('fallback.instrumentLabel')
      ? t('fallback.instrumentLabel')
      : FALLBACK_SCRIPT.instrument_label,
    questions: [
      {
        id: q0.id,
        text: t.has('fallback.text') ? t('fallback.text') : q0.text,
        reply: t.has('fallback.reply') ? t('fallback.reply') : q0.reply,
      },
    ],
  };
}

/**
 * Pure resolver: localized ChatbotScript for a signal id, or `null` when the id
 * is unknown (the raw French JSON has no entry). The caller supplies the
 * `chatbot`-bound translator so this works in Server Components too.
 */
export function resolveLocalizedScript(
  signalId: string,
  t: ChatbotTranslator,
): ChatbotScript | null {
  const raw = getChatbotScript(signalId);
  if (!raw) return null;
  const scriptKey = SCRIPT_KEY_BY_ID[signalId] ?? signalId;
  return localizeScript(raw, scriptKey, t);
}

/**
 * Pure resolver: localized ChatbotScript for a signal id, or the localized
 * FALLBACK script when the id is null/unknown. Used by the chat panel, which
 * must always render a non-null script (suggested questions + intro bubble).
 */
export function resolveLocalizedScriptOrFallback(
  signalId: string | null,
  t: ChatbotTranslator,
): ChatbotScript {
  if (signalId) {
    const localized = resolveLocalizedScript(signalId, t);
    if (localized) return localized;
  }
  return localizeFallback(t);
}

/**
 * Returns a getter that resolves a localized ChatbotScript by signal id.
 * Reads the `chatbot` namespace so the rendered prose (question chips + scripted
 * replies) follows the active locale. Safe in Server and Client Components.
 * Returns `null` for an unknown id (consumers that hide themselves in that case,
 * e.g. the landing replay section, rely on this).
 */
export function useChatbotScriptGetter(): (
  signalId: string,
) => ChatbotScript | null {
  const t = useTranslations('chatbot');
  return (signalId: string) => resolveLocalizedScript(signalId, t);
}

/**
 * Hook returning the localized script for an id, or the localized FALLBACK
 * script when the id is null/unknown. Used by the chat panel, which must always
 * render a non-null script (suggested questions + intro bubble).
 */
export function useChatbotScriptOrFallback(
  signalId: string | null,
): ChatbotScript {
  const t = useTranslations('chatbot');
  return resolveLocalizedScriptOrFallback(signalId, t);
}
