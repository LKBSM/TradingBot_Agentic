import { render } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

/**
 * The error boundaries render OUTSIDE the (site)/(product) route groups, so they
 * inherit NEITHER the ProductShell rail disclaimer NOR the site Footer disclaimer
 * that cover every content page. They are the only surfaces that would otherwise
 * show ZERO regulatory notice — this guard fails if the notice ever disappears.
 * (global-error.tsx renders its own <html>; verified by inspection, not mounted.)
 *
 * I18N-1: both boundaries are now localized. NotFound is a Server Component
 * reading the fr/en/es bundle via `getTranslations` (mocked here to resolve the
 * real fr strings); LocaleError is a Client Component with an inline per-locale
 * map keyed off the URL `locale` param (mocked to fr). We assert on the fr notice.
 */

vi.mock('next-intl/server', () => ({
  getTranslations: async (ns: string) => {
    const fr = (await import('@/messages/fr.json')).default as Record<string, unknown>;
    const dict = ns.split('.').reduce<unknown>((o, k) => (o as Record<string, unknown>)?.[k], fr);
    return (key: string) =>
      key.split('.').reduce<unknown>((o, k) => (o as Record<string, unknown>)?.[k], dict) ?? key;
  },
}));

vi.mock('next/navigation', () => ({ useParams: () => ({ locale: 'fr' }) }));

// eslint-disable-next-line import/first
import NotFound from './not-found';
// eslint-disable-next-line import/first
import LocaleError from './error';

const NOTICE = /Ni signal de trading, ni conseil en investissement\.\s*18\+\./;

describe('error boundaries carry the regulatory notice — never zero', () => {
  it('the 404 page shows the notice (fr default)', async () => {
    const ui = await NotFound();
    const { container } = render(ui);
    expect(container.textContent).toMatch(NOTICE);
  });

  it('the route error boundary shows the notice (fr default)', () => {
    const { container } = render(
      <LocaleError error={Object.assign(new Error('x'), { digest: undefined })} reset={() => {}} />,
    );
    expect(container.textContent).toMatch(NOTICE);
  });
});
