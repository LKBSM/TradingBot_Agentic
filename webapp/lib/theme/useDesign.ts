'use client';

import { useTheme } from 'next-themes';
import { DEFAULT_THEME, THEMES, themeById, type ThemeId, type ThemeMeta } from './themes';

export interface UseDesign {
  /** The active design id, resolved to a known theme (falls back to the default
   *  before hydration / for an unknown stored value). */
  design: ThemeId;
  /** Metadata for the active design (name, base, preview swatch). */
  active: ThemeMeta;
  /** The full catalogue, in display order. */
  designs: readonly ThemeMeta[];
  /** Apply a design — persists (localStorage) and repaints `data-design` on
   *  <html> immediately, via next-themes. */
  setDesign(id: ThemeId): void;
}

/**
 * `setDesign` utility (UI-1). A thin, intention-revealing wrapper over
 * next-themes' `useTheme`, named in the product's vocabulary ("design" =
 * `data-design`) rather than the library's generic "theme". UI-2 wires this into
 * the Réglages / Apparence surface; the existing ThemeMenu / AppearancePicker can
 * migrate to it too. No new state — next-themes remains the single source of
 * truth (localStorage + pre-paint inline script), so this stays flash-free.
 */
export function useDesign(): UseDesign {
  const { theme, setTheme } = useTheme();
  const active = themeById(theme ?? DEFAULT_THEME);
  return {
    design: active.id,
    active,
    designs: THEMES,
    setDesign: (id: ThemeId) => setTheme(id),
  };
}
