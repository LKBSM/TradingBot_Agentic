import type {
  ConvictionLabel,
  Direction,
  InsightSignalV2,
} from '@/types/insight';

/**
 * UI-facing strings derived from the raw InsightSignalV2 contract. Kept in a
 * single module so wording is auditable in one place (compliance reviews, FR
 * adjustments, future i18n). Anything user-visible should land here, not be
 * inlined in JSX.
 */

const INSTRUMENT_LABEL: Record<InsightSignalV2['instrument'], string> = {
  XAUUSD: 'Or (XAU/USD)',
  EURUSD: 'Euro / Dollar (EUR/USD)',
  BTCUSD: 'Bitcoin (BTC/USD)',
  US500: 'S&P 500 (US500)',
  GBPUSD: 'Livre / Dollar (GBP/USD)',
  USDJPY: 'Dollar / Yen (USD/JPY)',
};

const TIMEFRAME_LABEL: Record<InsightSignalV2['timeframe'], string> = {
  M1: '1 minute',
  M5: '5 minutes',
  M15: '15 minutes',
  M30: '30 minutes',
  H1: '1 heure',
  H4: '4 heures',
  D1: '1 jour',
  W1: '1 semaine',
};

export function formatInstrument(s: Pick<InsightSignalV2, 'instrument'>): string {
  return INSTRUMENT_LABEL[s.instrument] ?? s.instrument;
}

export function formatTimeframe(s: Pick<InsightSignalV2, 'timeframe'>): string {
  return TIMEFRAME_LABEL[s.timeframe] ?? s.timeframe;
}

/**
 * Short verdict line — the single sentence shown in the hero card. Built from
 * direction + instrument + timeframe + conviction label. Intentionally
 * compliance-safe (no "buy"/"sell"/"signal" wording).
 *
 * LEGAL-PENDING: the conviction adjectives below ("modérée", "marquée"...) are
 * placeholders pending the legal review of finfluencer-safe vocabulary
 * (terminal légal en cours).
 */
export function formatVerdict(signal: InsightSignalV2): string {
  const subject =
    signal.instrument === 'XAUUSD'
      ? "l'or"
      : signal.instrument === 'BTCUSD'
        ? 'le bitcoin'
        : signal.instrument === 'US500'
          ? 'le S&P 500'
          : signal.instrument === 'EURUSD'
            ? "l'euro"
            : signal.instrument === 'GBPUSD'
              ? 'la livre'
              : 'le yen';

  const tone = convictionTone(signal.conviction_label);

  switch (signal.direction) {
    case 'BULLISH_SETUP':
      return `Lecture haussière sur ${subject}, conviction ${tone}.`;
    case 'BEARISH_SETUP':
      return `Lecture baissière sur ${subject}, conviction ${tone}.`;
    case 'NEUTRAL':
      return `Lecture neutre sur ${subject}, conviction ${tone}.`;
  }
}

export function convictionTone(label: ConvictionLabel): string {
  switch (label) {
    case 'weak':
      return 'faible';
    case 'moderate':
      return 'modérée';
    case 'strong':
      return 'marquée';
    case 'institutional':
      return 'institutionnelle';
  }
}

export function convictionLabelLong(label: ConvictionLabel): string {
  switch (label) {
    case 'weak':
      return 'Conviction faible';
    case 'moderate':
      return 'Conviction modérée';
    case 'strong':
      return 'Conviction marquée';
    case 'institutional':
      return 'Conviction institutionnelle';
  }
}

export function directionBadgeVariant(
  direction: Direction,
): 'bull' | 'bear' | 'neutral' {
  switch (direction) {
    case 'BULLISH_SETUP':
      return 'bull';
    case 'BEARISH_SETUP':
      return 'bear';
    case 'NEUTRAL':
      return 'neutral';
  }
}

// ─── Time helpers ───────────────────────────────────────────────────────────

const SECOND = 1_000;
const MINUTE = 60 * SECOND;
const HOUR = 60 * MINUTE;
const DAY = 24 * HOUR;

/** "il y a 12 minutes", "il y a 3 heures", "il y a 2 jours". */
export function formatRelativePast(
  iso: string,
  now: Date = new Date(),
): string {
  const diff = now.getTime() - new Date(iso).getTime();
  if (diff < MINUTE) return "à l'instant";
  if (diff < HOUR) {
    const m = Math.floor(diff / MINUTE);
    return `il y a ${m} minute${m > 1 ? 's' : ''}`;
  }
  if (diff < DAY) {
    const h = Math.floor(diff / HOUR);
    return `il y a ${h} heure${h > 1 ? 's' : ''}`;
  }
  const d = Math.floor(diff / DAY);
  return `il y a ${d} jour${d > 1 ? 's' : ''}`;
}

/** "expire dans 2h47", "expirée il y a 10 min". */
export function formatValidityCountdown(
  validUntilIso: string,
  now: Date = new Date(),
): { label: string; expired: boolean } {
  const diff = new Date(validUntilIso).getTime() - now.getTime();
  if (diff <= 0) {
    return {
      label: `expirée ${formatRelativePast(validUntilIso, now)}`,
      expired: true,
    };
  }
  if (diff < HOUR) {
    const m = Math.max(1, Math.floor(diff / MINUTE));
    return { label: `expire dans ${m} min`, expired: false };
  }
  const h = Math.floor(diff / HOUR);
  const m = Math.floor((diff % HOUR) / MINUTE);
  const hm = m === 0 ? `${h}h` : `${h}h${m.toString().padStart(2, '0')}`;
  return { label: `expire dans ${hm}`, expired: false };
}

/** "FOMC Minutes dans 18h05" / "ECB Rate Decision dans 3h" / "Non-Farm Payrolls dans 2j". */
export function formatNextEventCountdown(minutes: number | null): string | null {
  if (minutes === null) return null;
  if (minutes < 60) return `dans ${Math.max(1, Math.round(minutes))} min`;
  if (minutes < 60 * 24) {
    const h = Math.floor(minutes / 60);
    const m = Math.round(minutes % 60);
    return m === 0 ? `dans ${h}h` : `dans ${h}h${m.toString().padStart(2, '0')}`;
  }
  const d = Math.floor(minutes / (60 * 24));
  const remH = Math.floor((minutes % (60 * 24)) / 60);
  return remH === 0 ? `dans ${d}j` : `dans ${d}j${remH}h`;
}
