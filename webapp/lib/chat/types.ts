/**
 * Minimal context the chat layer needs to bind a conversation to a market
 * reading. The Sentinel chatbot only ever reads three fields — an id (to reset
 * turns when the active reading changes + resolve the scripted demo) and the
 * instrument / timeframe combo (for the Tension-T1 context preamble and panel
 * labels).
 *
 * Replaces the former dependency on the heavy `InsightSignalV2` type (removed in
 * Chantier 5.C): the chatbot is signal-agnostic, so it should not be coupled to
 * the full insight contract. Both the landing (`openFor`) and the /app sidebar
 * (`openForCombo`) produce this shape natively — no unsafe cast.
 */
export interface ChatSignalContext {
  /** Stable id — drives turn-reset on change + scripted demo lookup. */
  id: string;
  instrument: string;
  timeframe: string;
}
