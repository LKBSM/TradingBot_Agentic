/**
 * Single source of truth for the four user-selectable themes. A theme is only a
 * set of token VALUES (see app/globals.css); this module carries the STABLE
 * metadata (internal id + light/dark base). The human-facing NAME and DESCRIPTION
 * are localised in the `appearance` i18n namespace (`names.<id>` / `descriptions.<id>`),
 * and the menu VIGNETTES are DERIVED from the live design tokens (a `data-design`
 * wrapper, THM-1) — so neither a name nor a swatch is ever hand-maintained here
 * and drift from the real palette is impossible.
 */
export const THEME_IDS = ['terminal', 'atelier', 'schema', 'ardoise'] as const;
export type ThemeId = (typeof THEME_IDS)[number];

/** The default theme (also the pre-hydration `:root` fallback in globals.css). */
export const DEFAULT_THEME: ThemeId = 'terminal';

export interface ThemeMeta {
  id: ThemeId;
  /** Whether the theme reads as light or dark (drives the quick sun/moon hint). */
  base: 'light' | 'dark';
}

export const THEMES: readonly ThemeMeta[] = [
  { id: 'terminal', base: 'dark' },
  { id: 'atelier', base: 'light' },
  { id: 'schema', base: 'dark' },
  { id: 'ardoise', base: 'dark' },
] as const;

/** Lookup helper; falls back to the default theme for an unknown id. */
export function themeById(id: string | undefined): ThemeMeta {
  return THEMES.find((t) => t.id === id) ?? THEMES[0]!;
}
