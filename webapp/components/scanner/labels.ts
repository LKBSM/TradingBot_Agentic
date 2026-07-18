/**
 * Display helpers for the Scanner — neutral, descriptive vocabulary only.
 * No predictive/prescriptive wording lives here.
 */

export function instrumentLabel(instrument: string): string {
  switch (instrument) {
    case 'XAUUSD':
      return 'XAU/USD';
    case 'EURUSD':
      return 'EUR/USD';
    default:
      return instrument;
  }
}

export type Tone = 'bull' | 'bear' | 'neutral';

export function biasTone(value: string | null | undefined): Tone {
  if (value === 'bullish') return 'bull';
  if (value === 'bearish') return 'bear';
  return 'neutral';
}

/** Static (JIT-safe) text colour class for a bias tone. */
export function toneTextClass(value: string | null | undefined): string {
  if (value === 'bullish') return 'text-sentinel-bull';
  if (value === 'bearish') return 'text-sentinel-bear';
  return 'text-sentinel-neutral';
}

export function biasGlyph(value: string | null | undefined): string {
  if (value === 'bullish') return '↗';
  if (value === 'bearish') return '↘';
  if (value === 'neutral' || value === 'ranging') return '→';
  return '·';
}

