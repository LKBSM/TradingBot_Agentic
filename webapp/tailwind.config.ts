import type { Config } from 'tailwindcss';

const config: Config = {
  // A "theme" is `data-theme="<id>"` on <html>. The three dark themes drive
  // Tailwind's `dark:` variant; Atelier (light) is intentionally absent so
  // `dark:` utilities do not apply there.
  darkMode: [
    'variant',
    [
      '[data-theme=terminal] &',
      '[data-theme=schema] &',
      '[data-theme=ardoise] &',
    ],
  ],
  content: ['./app/**/*.{ts,tsx}', './components/**/*.{ts,tsx}', './lib/**/*.{ts,tsx}'],
  theme: {
    container: {
      center: true,
      padding: '2rem',
      screens: { '2xl': '1400px' },
    },
    extend: {
      colors: {
        border: 'hsl(var(--border))',
        input: 'hsl(var(--input))',
        ring: 'hsl(var(--ring))',
        background: 'hsl(var(--background))',
        foreground: 'hsl(var(--foreground))',
        primary: {
          DEFAULT: 'hsl(var(--primary))',
          foreground: 'hsl(var(--primary-foreground))',
        },
        secondary: {
          DEFAULT: 'hsl(var(--secondary))',
          foreground: 'hsl(var(--secondary-foreground))',
        },
        destructive: {
          DEFAULT: 'hsl(var(--destructive))',
          foreground: 'hsl(var(--destructive-foreground))',
        },
        muted: {
          DEFAULT: 'hsl(var(--muted))',
          foreground: 'hsl(var(--muted-foreground))',
        },
        accent: {
          DEFAULT: 'hsl(var(--accent))',
          foreground: 'hsl(var(--accent-foreground))',
        },
        popover: {
          DEFAULT: 'hsl(var(--popover))',
          foreground: 'hsl(var(--popover-foreground))',
        },
        card: {
          DEFAULT: 'hsl(var(--card))',
          foreground: 'hsl(var(--card-foreground))',
        },
        // Domain accents (kept from Phase 2B; used for verdict / risk semantics)
        sentinel: {
          gold: '#C9A14A',
          bull: 'hsl(var(--sentinel-bull))',
          bear: 'hsl(var(--sentinel-bear))',
          neutral: 'hsl(var(--sentinel-neutral))',
          warn: 'hsl(var(--sentinel-warn))',
          liq: 'hsl(var(--sentinel-liq))',
        },
      },
      borderRadius: {
        lg: 'var(--radius)',
        md: 'calc(var(--radius) - 2px)',
        sm: 'calc(var(--radius) - 4px)',
      },
      fontFamily: {
        sans: [
          'var(--font-sans)',
          'ui-sans-serif',
          'system-ui',
          '-apple-system',
          'sans-serif',
        ],
        mono: ['ui-monospace', 'SFMono-Regular', 'monospace'],
        // Narrative voice — resolves to `--font-narrative` which a theme may
        // swap (e.g. Atelier → serif). Defaults to the sans stack elsewhere.
        narrative: [
          'var(--font-narrative)',
          'var(--font-sans)',
          'ui-serif',
          'Georgia',
          'serif',
        ],
      },
      keyframes: {
        'accordion-down': {
          from: { height: '0' },
          to: { height: 'var(--radix-accordion-content-height)' },
        },
        'accordion-up': {
          from: { height: 'var(--radix-accordion-content-height)' },
          to: { height: '0' },
        },
      },
      animation: {
        'accordion-down': 'accordion-down 0.2s ease-out',
        'accordion-up': 'accordion-up 0.2s ease-out',
      },
    },
  },
  plugins: [require('tailwindcss-animate')],
};
export default config;
