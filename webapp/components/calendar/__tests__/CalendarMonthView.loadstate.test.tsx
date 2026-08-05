import * as React from 'react';
import { render as rtlRender } from '@testing-library/react';
import { NextIntlClientProvider } from 'next-intl';
import { describe, expect, it, vi, beforeEach } from 'vitest';
import fr from '@/messages/fr.json';
import en from '@/messages/en.json';
import type { CalendarResponse } from '@/types/calendar';

/**
 * CAL-1 — a count is NEVER asserted before the data has loaded, and the waiting
 * state is visually distinct from a settled empty result, in BOTH languages.
 *
 * The month view reads its data through `useCalendarMonth`; we mock that hook so
 * the three states (loading / loaded-with-events / loaded-empty / error) can be
 * driven directly, without injecting `data` (which would force isLoading=false).
 */

const mockHook = vi.fn();
vi.mock('@/lib/calendar/useCalendar', () => ({
  useCalendarMonth: (month: string) => mockHook(month),
  useCalendar: () => mockHook(),
}));

// Imported AFTER the mock is registered.
import { CalendarMonthView } from '../CalendarMonthView';

const NOW = new Date('2026-08-05T12:00:00Z');

const MESSAGES = { fr, en };

function renderIn(locale: 'fr' | 'en') {
  return rtlRender(<CalendarMonthView locale={locale} now={NOW} />, {
    wrapper: ({ children }) => (
      <NextIntlClientProvider locale={locale} messages={locale === 'fr' ? fr : en}>
        {children}
      </NextIntlClientProvider>
    ),
  });
}

const AUGUST: CalendarResponse = {
  window_start: '2026-08-01T00:00:00Z',
  window_end: '2026-08-31T23:59:59Z',
  generated_at: NOW.toISOString(),
  coverage: {
    source: 'official',
    feed_start: null,
    feed_end: null,
    partial: false,
    last_success: {},
    stale_sources: [],
  },
  attribution: [],
  events: [
    {
      event_id: 'bls:cpi:2026-08-12',
      source: 'bls',
      series_code: null,
      license_label: null,
      event: 'IPC',
      currency: 'USD',
      organism: 'Bureau of Labor Statistics',
      periodicity: 'monthly',
      scheduled_at: '2026-08-12T12:30:00Z',
      source_timezone: 'America/New_York',
      time_confirmed: true,
      markets: ['XAUUSD', 'EURUSD'],
      value_unit: null,
      actual: null,
      actual_initial: null,
      previous: null,
      revised: false,
      revised_at: null,
      actual_state: 'pending',
      refreshed_at: null,
      value_series: [],
    },
  ],
};

beforeEach(() => mockHook.mockReset());

describe('CAL-1 month view — no count before load', () => {
  for (const locale of ['fr', 'en'] as const) {
    const M: typeof fr = MESSAGES[locale];

    it(`[${locale}] while loading: NO count, a distinct waiting status, no empty claim`, () => {
      mockHook.mockReturnValue({ data: null, isLoading: true, error: null, refresh: () => {} });
      const { container } = renderIn(locale);

      // No count line, no "N empty days" is asserted.
      expect(container.querySelector('.calm-tm-count')).toBeNull();
      expect(container.textContent).not.toContain(M.calendar.month.panel.empty);

      // A distinct waiting status is shown in the side box AND the grid.
      const status = container.querySelector('.calm-tm-status[role="status"]');
      expect(status?.textContent).toContain(M.calendar.loading);
      expect(container.querySelector('.calm-grid')).toBeNull();
      expect(container.querySelector('.cal-status')?.textContent).toContain(
        M.calendar.loading,
      );
    });

    it(`[${locale}] loaded WITH publications: the count is asserted`, () => {
      mockHook.mockReturnValue({ data: AUGUST, isLoading: false, error: null, refresh: () => {} });
      const { container } = renderIn(locale);
      expect(container.querySelector('.calm-grid')).not.toBeNull();
      expect(container.querySelector('.calm-tm-count')?.textContent).toBe(
        `1 ${locale === 'fr' ? 'publication' : 'publication'}`,
      );
    });

    it(`[${locale}] loaded but genuinely EMPTY: explicit empty, no bare zero count`, () => {
      mockHook.mockReturnValue({
        data: { ...AUGUST, events: [] },
        isLoading: false,
        error: null,
        refresh: () => {},
      });
      const { container } = renderIn(locale);
      // No fabricated "0 publications / N empty days" — an explicit legitimate note.
      expect(container.querySelector('.calm-tm-count')).toBeNull();
      const status = container.querySelector('.calm-tm-status');
      expect(status?.textContent).toContain(M.calendar.month.empty);
    });

    it(`[${locale}] server unreachable: distinct error, no count`, () => {
      mockHook.mockReturnValue({
        data: null,
        isLoading: false,
        error: new Error('unreachable'),
        refresh: () => {},
      });
      const { container } = renderIn(locale);
      expect(container.querySelector('.calm-tm-count')).toBeNull();
      expect(container.querySelector('.calm-tm-status')?.textContent).toContain(
        M.calendar.error,
      );
      // The day panel does not claim "no publication this day" on error.
      expect(container.textContent).not.toContain(M.calendar.month.panel.empty);
    });
  }
});
