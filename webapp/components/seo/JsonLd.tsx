/**
 * Server component that injects structured data (JSON-LD) into the page
 * head. Used for SoftwareApplication + FAQPage so Google can surface the
 * landing in rich results (sitelinks, FAQ snippets). Keep the wording
 * compliance-safe — what's in here is publicly indexable.
 */

interface JsonLdProps {
  data: Record<string, unknown>;
}

export function JsonLd({ data }: JsonLdProps) {
  return (
    <script
      type="application/ld+json"
      // Pre-stringified to avoid React escaping quotes inside the payload.
      dangerouslySetInnerHTML={{ __html: JSON.stringify(data) }}
    />
  );
}

const SITE_URL =
  process.env.NEXT_PUBLIC_SITE_URL ?? 'https://mia.markets';

/**
 * SoftwareApplication entity for the landing. Goes inside <head> via the
 * RootLayout. Description stays compliance-safe (no perf promise).
 */
export const softwareApplicationLd = {
  '@context': 'https://schema.org',
  '@type': 'SoftwareApplication',
  name: 'MIA Markets',
  alternateName: 'Multi-asset Intelligence Assistant for Markets',
  url: SITE_URL,
  applicationCategory: 'FinanceApplication',
  operatingSystem: 'Web · iOS · Android (PWA)',
  description:
    'MIA Markets est un indicateur de marché conversationnel pour XAU/USD et le forex. Lectures algorithmiques contextuelles, posture éducative, chatbot Sentinel.',
  inLanguage: 'fr-FR',
  isAccessibleForFree: true,
  offers: [
    {
      '@type': 'Offer',
      name: 'Découverte',
      price: '0',
      priceCurrency: 'EUR',
      category: 'Free',
    },
    {
      '@type': 'Offer',
      name: 'Analyste',
      price: '29',
      priceCurrency: 'EUR',
      category: 'Subscription',
    },
    {
      '@type': 'Offer',
      name: 'Stratège',
      price: '79',
      priceCurrency: 'EUR',
      category: 'Subscription',
    },
  ],
  publisher: {
    '@type': 'Organization',
    name: 'MIA Markets',
    url: SITE_URL,
  },
};
