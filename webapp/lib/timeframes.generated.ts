// AUTO-GENERATED from config/timeframes.json by scripts/gen_timeframes.mjs.
// DO NOT EDIT BY HAND. Run `node scripts/gen_timeframes.mjs` after editing the JSON.
export interface TimeframeSpec {
  id: string;
  minutes: number;
  seconds: number;
  provider: string;
  labelLong: string;
  dateFormat: string;
  perimeter: boolean;
  reference: boolean;
  sessionRelevant: boolean;
  prevLevelsRelevant: boolean;
  index: number;
}

export const TIMEFRAME_SPECS: readonly TimeframeSpec[] = [
  { id: "M1", minutes: 1, seconds: 60, provider: "1min", labelLong: "1 minute", dateFormat: "HH:mm", perimeter: true, reference: false, sessionRelevant: true, prevLevelsRelevant: true, index: 0 },
  { id: "M5", minutes: 5, seconds: 300, provider: "5min", labelLong: "5 minutes", dateFormat: "HH:mm", perimeter: true, reference: false, sessionRelevant: true, prevLevelsRelevant: true, index: 1 },
  { id: "M15", minutes: 15, seconds: 900, provider: "15min", labelLong: "15 minutes", dateFormat: "HH:mm", perimeter: true, reference: false, sessionRelevant: true, prevLevelsRelevant: true, index: 2 },
  { id: "M30", minutes: 30, seconds: 1800, provider: "30min", labelLong: "30 minutes", dateFormat: "HH:mm", perimeter: false, reference: false, sessionRelevant: true, prevLevelsRelevant: true, index: 3 },
  { id: "H1", minutes: 60, seconds: 3600, provider: "1h", labelLong: "1 heure", dateFormat: "HH:mm", perimeter: true, reference: false, sessionRelevant: true, prevLevelsRelevant: true, index: 4 },
  { id: "H4", minutes: 240, seconds: 14400, provider: "4h", labelLong: "4 heures", dateFormat: "dd MMM HH:mm", perimeter: true, reference: false, sessionRelevant: true, prevLevelsRelevant: true, index: 5 },
  { id: "D1", minutes: 1440, seconds: 86400, provider: "1day", labelLong: "1 jour", dateFormat: "dd MMM", perimeter: true, reference: true, sessionRelevant: false, prevLevelsRelevant: false, index: 6 },
  { id: "W1", minutes: 10080, seconds: 604800, provider: "1week", labelLong: "1 semaine", dateFormat: "dd MMM yyyy", perimeter: false, reference: true, sessionRelevant: false, prevLevelsRelevant: false, index: 7 },
];
