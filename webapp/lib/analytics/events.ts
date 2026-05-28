/**
 * DG-160 + DG-161 — analytics event catalogue + typed dispatcher.
 *
 * V1 ships Plausible (self-hosted on Fly.io per docs/governance/.../
 * configurations/fly_secrets.md, planned post-deploy). The dispatcher
 * is intentionally analytics-vendor-agnostic — it calls
 * ``window.plausible`` if defined and is a no-op otherwise, so the
 * webapp keeps working offline and there's no console error in dev.
 *
 * The six **core events** below are the ones the product cares about
 * for conversion + UX iteration. Anything else (button hovers, scroll
 * depth, etc.) is out of scope V1 — add via a new entry here, never
 * via free-form strings at the call site.
 */

export type CoreEventName =
  | 'hero_view'
  | 'chatbot_open'
  | 'chatbot_question_sent'
  | 'chatbot_refusal_received'
  | 'track_record_view'
  | 'suggested_question_click';

/**
 * Per-event prop schema. Each event keeps its own typed payload — the
 * dispatcher won't accept arbitrary keys. If you need to add a prop,
 * grow this discriminated union *and* run `npx tsc` before deploy.
 *
 * Props must be primitive (string | number | boolean) — Plausible
 * rejects nested objects.
 */
export type CoreEventProps = {
  hero_view: {
    /** Whether the user landed straight from a referrer (vs internal nav). */
    is_landing: boolean;
    /** ISO-639-1 user locale (fr / en / de / es). */
    locale: 'fr' | 'en' | 'de' | 'es';
  };
  chatbot_open: {
    /** "hero" | "track_record" | "navbar" — the trigger surface. */
    source: 'hero' | 'track_record' | 'navbar';
  };
  chatbot_question_sent: {
    /** Length bucket — avoids leaking the raw question text. */
    length_bucket: 'tiny' | 'short' | 'medium' | 'long';
    /** Whether the user chose a suggested question vs typed free-form. */
    via_suggestion: boolean;
  };
  chatbot_refusal_received: {
    /** Category from the DG-112 classifier (see adversarial-patterns.ts). */
    category:
      | 'prescriptive'
      | 'guarantee'
      | 'jailbreak'
      | 'personal_advice'
      | 'signal_request';
    /** Whether the refusal came from the pre-LLM gate or the post-stream filter. */
    layer: 'pre_llm_gate' | 'post_stream_filter';
  };
  track_record_view: {
    /** Whether the visitor has opened the page on this session before. */
    is_first_view: boolean;
  };
  suggested_question_click: {
    /** Q1 / Q2 / Q3 slot from suggested-questions.ts. */
    slot: 'q1_conviction' | 'q2_top_component' | 'q3_event' | 'q3_ci' | 'q3_regime';
  };
};

declare global {
  interface Window {
    /**
     * Plausible dispatch shape — set by the Plausible script.
     * When missing (script not loaded, dev mode without analytics),
     * ``trackEvent`` returns silently.
     */
    plausible?: (
      event: string,
      options?: { props?: Record<string, string | number | boolean> },
    ) => void;
  }
}

/**
 * Type-safe analytics dispatcher.
 *
 * Compile-time guarantees:
 *  - ``name`` must be one of CoreEventName.
 *  - ``props`` must match the schema for the chosen name (TS error if missing).
 *
 * Runtime guarantees:
 *  - Never throws — analytics outage must NEVER break the UX.
 *  - Server-side calls (during SSR) are no-ops.
 */
export function trackEvent<E extends CoreEventName>(
  name: E,
  props: CoreEventProps[E],
): void {
  if (typeof window === 'undefined') return;
  const dispatcher = window.plausible;
  if (typeof dispatcher !== 'function') return;
  try {
    dispatcher(name, { props: props as Record<string, string | number | boolean> });
  } catch {
    // Swallow — analytics MUST NOT crash the host page.
  }
}

/**
 * Helper: bucket a question length for the chatbot_question_sent event.
 * Keeps the raw length out of the analytics payload (privacy + cardinality).
 */
export function lengthBucket(text: string): CoreEventProps['chatbot_question_sent']['length_bucket'] {
  const n = text.trim().length;
  if (n <= 20) return 'tiny';
  if (n <= 80) return 'short';
  if (n <= 200) return 'medium';
  return 'long';
}
