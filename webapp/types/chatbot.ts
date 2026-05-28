/**
 * Scripted chatbot dialogue shape. Every signal exposes a fixed list of
 * pre-written question/reply pairs — no free-text LLM call in V1. When the
 * real backend lands the same shape is what the API will return after a
 * retrieval pass, so consumers stay stable.
 */

export interface ChatbotQuestion {
  id: string;
  /** Text shown on the suggestion chip. */
  text: string;
  /** Pre-written assistant reply, plain markdown-free string. */
  reply: string;
}

export interface ChatbotScript {
  /** Short instrument label used in the panel header. */
  instrument_label: string;
  questions: ChatbotQuestion[];
}

/** Top-level JSON shape: signal_id → scripted dialogue. */
export type ChatbotResponses = Record<string, ChatbotScript>;
