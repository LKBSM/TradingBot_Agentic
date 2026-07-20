import { describe, expect, it } from 'vitest';
import { localizeHref } from '../href';
import { DEFAULT_LOCALE } from '@/i18n';

/**
 * NAV-06 — internal paths must carry the active locale (except the default,
 * served prefixless). Hash-only and external hrefs pass through untouched.
 */
describe('localizeHref', () => {
  it('leaves the default locale prefixless', () => {
    expect(localizeHref('/app', DEFAULT_LOCALE)).toBe('/app');
    expect(localizeHref('/', DEFAULT_LOCALE)).toBe('/');
  });

  it('prefixes a non-default locale', () => {
    expect(localizeHref('/app', 'en')).toBe('/en/app');
    expect(localizeHref('/connexion', 'de')).toBe('/de/connexion');
    expect(localizeHref('/', 'es')).toBe('/es/');
  });

  it('prefixes paths that carry a hash', () => {
    expect(localizeHref('/#faq', 'en')).toBe('/en/#faq');
    expect(localizeHref('/methodology#attributions', 'it')).toBe(
      '/it/methodology#attributions',
    );
  });

  it('passes through hash-only and external hrefs untouched', () => {
    expect(localizeHref('#faq', 'en')).toBe('#faq');
    expect(localizeHref('mailto:contact@mia.markets', 'en')).toBe(
      'mailto:contact@mia.markets',
    );
    expect(localizeHref('https://example.com', 'de')).toBe('https://example.com');
  });
});
