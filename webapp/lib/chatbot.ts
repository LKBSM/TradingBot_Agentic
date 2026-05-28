import chatbotResponsesJson from '@/mocks/chatbot_responses.json';
import type { ChatbotResponses, ChatbotScript } from '@/types/chatbot';

// Same unsafe-cast pattern as lib/mocks.ts — JSON tuples widen, hand-curated
// fixture is the source of truth. zod validation will replace this at the
// backend integration sprint.
const RESPONSES = chatbotResponsesJson as unknown as ChatbotResponses;

export function getChatbotScript(signalId: string): ChatbotScript | null {
  return RESPONSES[signalId] ?? null;
}

/**
 * Fallback script used when a signal id is not present in the JSON (defensive
 * — should not happen with the mocked demo). Provides a single educational
 * refusal so the UI never renders an empty panel.
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
