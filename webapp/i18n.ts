import { getRequestConfig } from 'next-intl/server';
import { notFound } from 'next/navigation';

export const SUPPORTED_LOCALES = ['fr', 'en', 'de', 'es'] as const;
export type Locale = (typeof SUPPORTED_LOCALES)[number];
export const DEFAULT_LOCALE: Locale = 'fr';

export default getRequestConfig(async ({ locale }) => {
  if (!SUPPORTED_LOCALES.includes(locale as Locale)) notFound();
  return {
    messages: (await import(`./messages/${locale}.json`)).default,
  };
});
