/**
 * Server component that injects structured data (JSON-LD) into the page
 * head. Used for SoftwareApplication + FAQPage so Google can surface the
 * landing in rich results (sitelinks, FAQ snippets). Keep the wording
 * compliance-safe — what's in here is publicly indexable.
 */

import { PRICING } from '@/lib/pricing.generated';

interface JsonLdProps {
  data: Record<string, unknown>;
}

export function JsonLd({ data }: JsonLdProps) {
  return (
    <script
      type="application/ld+json"
      // Pre-stringified to avoid React escaping quotes inside the payload.
      // `<` is escaped to its < form so a value containing `</script>` (or
      // `<!--`) can never break out of the script tag (UI-18: XSS-safe even
      // though today's payloads are all static).
      dangerouslySetInnerHTML={{
        __html: JSON.stringify(data).replace(/</g, '\\u003c'),
      }}
    />
  );
}

const SITE_URL =
  process.env.NEXT_PUBLIC_SITE_URL ?? 'https://mia.markets';

// BCP-47 (language_TERRITORY) per shipped locale, for the JSON-LD `inLanguage`.
// Spanish ships as es-ES (I18N-1). Mirrors the OG_LOCALES map in the root layout.
const LD_LANG: Record<string, string> = {
  fr: 'fr-FR',
  en: 'en-US',
  es: 'es-ES',
};

/**
 * SoftwareApplication entity for the landing, built PER LOCALE so `inLanguage`
 * and the description match the page the crawler is reading (I18N-1: the former
 * static object always claimed `fr-FR` + a French description on /en and /es).
 * The description is the localized `seo.description` string, passed in by the
 * (locale-aware) site layout. Description stays compliance-safe (no perf promise).
 */
export function buildSoftwareApplicationLd(
  locale: string,
  description: string,
): Record<string, unknown> {
  return {
    ...SOFTWARE_APPLICATION_BASE,
    description,
    inLanguage: LD_LANG[locale] ?? LD_LANG.fr,
  };
}

const SOFTWARE_APPLICATION_BASE = {
  '@context': 'https://schema.org',
  '@type': 'SoftwareApplication',
  name: 'M.I.A Markets',
  alternateName: 'Multi-asset Intelligence Assistant for Markets',
  url: SITE_URL,
  applicationCategory: 'FinanceApplication',
  operatingSystem: 'Web · iOS · Android (PWA)',
  isAccessibleForFree: false,
  // Amounts come from the single pricing source (config/pricing.json) — never
  // hard-coded. Currency is USD everywhere. The annual offer is the yearly
  // total ($348), consistent with what the pricing section headlines.
  offers: [
    {
      '@type': 'Offer',
      name: 'Accès intégral MIA · mensuel',
      price: String(PRICING.monthly),
      priceCurrency: PRICING.currency,
      category: 'Subscription',
    },
    {
      '@type': 'Offer',
      name: 'Accès intégral MIA · annuel',
      price: String(PRICING.annualPerYear),
      priceCurrency: PRICING.currency,
      category: 'Subscription',
    },
  ],
  publisher: {
    '@type': 'Organization',
    name: 'M.I.A Markets',
    url: SITE_URL,
    logo: `${SITE_URL}/icon.svg`,
  },
};
