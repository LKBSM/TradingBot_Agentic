import { describe, expect, it } from 'vitest';
import {
  changeTone,
  formatBand,
  formatChangePercent,
  formatImpact,
  formatInstrument,
  formatMarketPhase,
  formatMinutesAgo,
  formatMtfBias,
  formatPrice,
  formatRelativePast,
  formatSurprise,
  formatTimeframe,
  formatTimeToEvent,
  formatTrend,
  formatTriggerType,
  formatValidationStatus,
  formatVolatility,
} from './formatters';

describe('market-reading formatters', () => {
  it('labels known instruments and falls back to the raw code', () => {
    expect(formatInstrument('XAUUSD')).toBe('Or (XAU/USD)');
    expect(formatInstrument('EURUSD')).toContain('EUR/USD');
    expect(formatInstrument('ZZZ')).toBe('ZZZ');
  });

  it('labels known timeframes and falls back to the raw code', () => {
    expect(formatTimeframe('M15')).toBe('15 minutes');
    expect(formatTimeframe('H4')).toBe('4 heures');
    expect(formatTimeframe('Q1')).toBe('Q1');
  });

  it('formats prices with instrument-specific precision', () => {
    // XAU → 2 decimals, EUR → 5 decimals (FR locale uses a comma).
    expect(formatPrice(2392.35, 'XAUUSD')).toMatch(/2[\s ]?392,35/);
    expect(formatPrice(1.08423, 'EURUSD')).toBe('1,08423');
    // Unknown instrument defaults to 2 decimals.
    expect(formatPrice(10, 'ZZZ')).toBe('10,00');
  });

  it('formats a price band with an en-dash', () => {
    expect(formatBand(2375, 2378, 'XAUUSD')).toMatch(/2[\s ]?375,00 – 2[\s ]?378,00/);
  });

  it('formats a signed daily change percentage (fr-FR, descriptive)', () => {
    expect(formatChangePercent(-0.0322)).toBe('−3,22 %');
    expect(formatChangePercent(0.011)).toBe('+1,10 %');
    expect(formatChangePercent(0)).toBe('0,00 %');
  });

  it('maps a change to a colour tone (green up / red down / muted flat)', () => {
    expect(changeTone(0.01)).toBe('bull');
    expect(changeTone(-0.01)).toBe('bear');
    expect(changeTone(0)).toBe('neutral');
    expect(changeTone(null)).toBe('neutral');
  });

  it('maps trend to a descriptive label + tone (never a score)', () => {
    expect(formatTrend('bullish')).toEqual({
      label: 'Tendance haussière',
      tone: 'bull',
    });
    expect(formatTrend('bearish').tone).toBe('bear');
    expect(formatTrend('ranging')).toEqual({
      label: 'Marché en range',
      tone: 'neutral',
    });
  });

  it('flags elevated volatility as a warn tone', () => {
    expect(formatVolatility('elevated').tone).toBe('warn');
    expect(formatVolatility('normal').tone).toBe('neutral');
  });

  it('labels every market phase', () => {
    expect(formatMarketPhase('accumulation').label).toContain('accumulation');
    expect(formatMarketPhase('expansion').label).toContain('expansion');
  });

  it('maps MTF bias to label + tone', () => {
    expect(formatMtfBias('bullish')).toEqual({ label: 'haussier', tone: 'bull' });
    expect(formatMtfBias('ranging').tone).toBe('neutral');
  });

  it('maps high impact to a warn tone', () => {
    expect(formatImpact('high').tone).toBe('warn');
    expect(formatImpact('low').tone).toBe('neutral');
  });

  it('describes a news surprise without directive wording', () => {
    expect(formatSurprise('beat')).toBe('au-dessus du consensus');
    expect(formatSurprise('miss')).toBe('en-dessous du consensus');
  });

  it('validates statuses in French', () => {
    expect(formatValidationStatus('confirmed')).toBe('confirmée');
    expect(formatValidationStatus('pending')).toContain('attente');
  });

  it('humanises composite technical-trigger codes', () => {
    expect(formatTriggerType('bos_m15_bullish')).toBe(
      'Cassure de structure haussier (M15)',
    );
    expect(formatTriggerType('choch_h1_bearish')).toBe(
      'Changement de caractère baissier (H1)',
    );
    expect(formatTriggerType('fvg_h4')).toBe('Comblement de Fair Value Gap (H4)');
    expect(formatTriggerType('retest_m15')).toBe('Retest (M15)');
  });

  it('formats event timing relative to now', () => {
    expect(formatTimeToEvent(0)).toBe('imminent');
    expect(formatTimeToEvent(18)).toBe('dans 18 min');
    expect(formatTimeToEvent(125)).toBe('dans 2h05');
    expect(formatMinutesAgo(0)).toBe("à l'instant");
    expect(formatMinutesAgo(30)).toBe('il y a 30 min');
  });

  it('formats relative past against a fixed clock', () => {
    const now = new Date('2026-05-26T12:00:00Z');
    expect(formatRelativePast('2026-05-26T11:59:30Z', now)).toBe("à l'instant");
    expect(formatRelativePast('2026-05-26T11:45:00Z', now)).toBe(
      'il y a 15 minutes',
    );
    expect(formatRelativePast('2026-05-26T09:00:00Z', now)).toBe('il y a 3 heures');
  });
});
