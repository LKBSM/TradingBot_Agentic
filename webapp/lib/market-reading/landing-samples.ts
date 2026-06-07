import {
  FIXTURE_XAU_M15,
  FIXTURE_EUR_H1,
  FIXTURE_QUIET_XAU_H4,
} from './fixtures';
import type { MarketReading } from '@/types/market-reading';

/**
 * Static demo readings used by the marketing landing (hero, multi-market
 * gallery, conversation replay). NOT live data — they exercise the landing
 * surfaces before the backend integration, mirroring the role the former
 * `mocks/sample_signals.json` played for the deleted `insight/` components
 * (Chantier 5.C migration to the native MarketReading contract).
 *
 * Each sample pairs a MarketReading fixture with a stable `id` whose value
 * matches a key in `mocks/chatbot_responses.json`, so the scripted chatbot
 * demo (`getChatbotScript(id)`) keeps resolving the right conversation.
 */
export interface LandingSample {
  /** Stable id — resolves the scripted chatbot demo + chat turn-reset. */
  id: string;
  reading: MarketReading;
}

export const LANDING_SAMPLES: readonly LandingSample[] = [
  { id: '0193c7a42f1b', reading: FIXTURE_XAU_M15 }, // hero — XAU M15 bullish
  { id: '0193c7a4ab51', reading: FIXTURE_EUR_H1 }, //  EUR H1 ranging
  { id: '0193c7a5c8e2', reading: FIXTURE_QUIET_XAU_H4 }, // quiet XAU H4
];

export function getHeroLandingSample(): LandingSample {
  const first = LANDING_SAMPLES[0];
  if (!first) {
    throw new Error('LANDING_SAMPLES must contain at least one entry');
  }
  return first;
}
