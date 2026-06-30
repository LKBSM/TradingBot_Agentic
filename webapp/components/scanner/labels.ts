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

export function biasLabel(value: string | null | undefined): string {
  switch (value) {
    case 'bullish':
      return 'Haussier';
    case 'bearish':
      return 'Baissier';
    case 'ranging':
      return 'Range';
    case 'neutral':
      return 'Neutre';
    default:
      return '—';
  }
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

export function phaseLabel(value: string | null | undefined): string {
  switch (value) {
    case 'accumulation':
      return 'Accumulation';
    case 'distribution':
      return 'Distribution';
    case 'trend':
      return 'Tendance';
    case 'ranging':
      return 'Range';
    case 'expansion':
      return 'Expansion';
    default:
      return '—';
  }
}

/**
 * "il y a ~2 h" style relative age from an ISO close timestamp.
 * `nowMs` is injectable so the label can tick (and be tested) deterministically.
 */
export function relativeAge(
  iso: string | null | undefined,
  nowMs: number = Date.now(),
): string | null {
  if (!iso) return null;
  const then = Date.parse(iso);
  if (Number.isNaN(then)) return null;
  const minutes = Math.max(0, Math.round((nowMs - then) / 60000));
  if (minutes < 60) return `il y a ${minutes} min`;
  const hours = Math.round(minutes / 60);
  if (hours < 48) return `il y a ${hours} h`;
  const days = Math.round(hours / 24);
  return `il y a ${days} j`;
}
