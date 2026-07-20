/**
 * Single source of truth for the four user-selectable themes. A theme is only a
 * set of token VALUES (see app/globals.css); this module carries the human-facing
 * metadata + a small STATIC preview swatch (literal hex, describing the theme —
 * these are not runtime tokens, only vignette chips). Consumed by the nav theme
 * menu and the account "Apparence" picker so both stay in lockstep.
 */
export const THEME_IDS = ['terminal', 'atelier', 'schema', 'ardoise'] as const;
export type ThemeId = (typeof THEME_IDS)[number];

/** The default theme (also the pre-hydration `:root` fallback in globals.css). */
export const DEFAULT_THEME: ThemeId = 'terminal';

export interface ThemeMeta {
  id: ThemeId;
  /** Whether the theme reads as light or dark (drives the quick sun/moon hint). */
  base: 'light' | 'dark';
  /** Fixed identity name, kept across locales (the localised DESCRIPTION lives in
   *  the `appearance` i18n namespace). */
  name: string;
  /** Static preview chips — bg / panel / accent / bull / bear. */
  swatch: {
    bg: string;
    panel: string;
    accent: string;
    bull: string;
    bear: string;
  };
}

export const THEMES: readonly ThemeMeta[] = [
  {
    id: 'terminal',
    base: 'dark',
    name: 'Terminal',
    swatch: { bg: '#0a0f1c', panel: '#111a2c', accent: '#4d9de0', bull: '#37b98c', bear: '#dd6b7a' },
  },
  {
    id: 'atelier',
    base: 'light',
    name: 'Atelier',
    swatch: { bg: '#f6f4ee', panel: '#efece3', accent: '#1d6f6a', bull: '#2f8f6b', bear: '#b4564f' },
  },
  {
    id: 'schema',
    base: 'dark',
    name: 'Schéma',
    swatch: { bg: '#14161a', panel: '#1b1e24', accent: '#5fb3c4', bull: '#c3c8cf', bear: '#7c828b' },
  },
  {
    id: 'ardoise',
    base: 'dark',
    name: 'Ardoise',
    swatch: { bg: '#1a1613', panel: '#241f1a', accent: '#d8b878', bull: '#6bbf9a', bear: '#d98b7a' },
  },
] as const;

/** Lookup helper; falls back to the default theme for an unknown id. */
export function themeById(id: string | undefined): ThemeMeta {
  return THEMES.find((t) => t.id === id) ?? THEMES[0]!;
}
