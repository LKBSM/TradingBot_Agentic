/**
 * DS-1 — the scanner palette response, built from the REAL product palette
 * (`CONDITION_PALETTE` in lib/conditions/palette.ts — the actual condition set
 * the builder offers). Served for GET /api/conditions-scan/palette so /scanner
 * renders its builder with no backend. No network.
 */
import { CONDITION_PALETTE, FAMILIES } from '@/lib/conditions/palette';
import type { PaletteResponse } from '@/lib/conditions/types';

export const SAMPLE_PALETTE_RESPONSE: PaletteResponse = {
  families: [...FAMILIES],
  palette: [...CONDITION_PALETTE],
  blocked: [],
};
