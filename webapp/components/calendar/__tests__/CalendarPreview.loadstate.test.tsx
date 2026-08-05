import * as React from 'react';
import { render as rtlRender } from '@testing-library/react';
import { NextIntlClientProvider } from 'next-intl';
import { describe, expect, it, vi, beforeEach } from 'vitest';
import fr from '@/messages/fr.json';
import en from '@/messages/en.json';

/**
 * CAL-1 — the dashboard preview must not claim "nothing upcoming" before the
 * data has loaded: the waiting state is distinct from an empty result, in both
 * languages.
 */

const mockHook = vi.fn();
vi.mock('@/lib/calendar/useCalendar', () => ({
  useCalendar: () => mockHook(),
}));

import { CalendarPreview } from '../CalendarPreview';

const NOW = new Date('2026-08-05T12:00:00Z');
const MESSAGES = { fr, en };

function renderIn(locale: 'fr' | 'en') {
  return rtlRender(<CalendarPreview now={NOW} />, {
    wrapper: ({ children }) => (
      <NextIntlClientProvider locale={locale} messages={locale === 'fr' ? fr : en}>
        {children}
      </NextIntlClientProvider>
    ),
  });
}

beforeEach(() => mockHook.mockReset());

describe('CAL-1 preview — waiting is distinct from empty', () => {
  for (const locale of ['fr', 'en'] as const) {
    const M: typeof fr = MESSAGES[locale];

    it(`[${locale}] while loading: a distinct waiting status, never "nothing upcoming"`, () => {
      mockHook.mockReturnValue({ data: null, isLoading: true, error: null, refresh: () => {} });
      const { container } = renderIn(locale);
      const status = container.querySelector('.calprev-status[role="status"]');
      expect(status?.textContent).toContain(M.calendar.loading);
      expect(container.textContent).not.toContain(M.calendar.preview.empty);
      expect(container.querySelectorAll('.calprev-row')).toHaveLength(0);
    });

    it(`[${locale}] on error: a distinct error status, never the empty state`, () => {
      mockHook.mockReturnValue({
        data: null,
        isLoading: false,
        error: new Error('unreachable'),
        refresh: () => {},
      });
      const { container } = renderIn(locale);
      expect(container.querySelector('.calprev-status')?.textContent).toContain(
        M.calendar.error,
      );
      expect(container.textContent).not.toContain(M.calendar.preview.empty);
    });

    it(`[${locale}] loaded and truly empty: the honest empty state is shown`, () => {
      mockHook.mockReturnValue({
        data: {
          window_start: '2026-08-01T00:00:00Z',
          window_end: '2026-08-31T23:59:59Z',
          generated_at: NOW.toISOString(),
          coverage: { source: 'official', feed_start: null, feed_end: null, partial: false, last_success: {}, stale_sources: [] },
          attribution: [],
          events: [],
        },
        isLoading: false,
        error: null,
        refresh: () => {},
      });
      const { container } = renderIn(locale);
      expect(container.querySelector('.calprev-empty')?.textContent).toBe(
        M.calendar.preview.empty,
      );
    });
  }
});
