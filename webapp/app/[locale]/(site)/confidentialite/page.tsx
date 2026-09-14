import type { Metadata } from 'next';
import Link from 'next/link';
import { useTranslations } from 'next-intl';
import { getTranslations } from 'next-intl/server';
import { ArrowLeft } from 'lucide-react';
import { LegalDocument } from '@/components/legal/LegalDocument';

export async function generateMetadata({
  params,
}: {
  params: Promise<{ locale: string }>;
}): Promise<Metadata> {
  const { locale } = await params;
  const t = await getTranslations({ locale, namespace: 'legal' });
  return {
    title: t('privacy.meta.title'),
    description: t('privacy.meta.description'),
  };
}

/**
 * /confidentialite — renders the Privacy Policy document
 * (docs/legal/politique-confidentialite.{fr,en,es}.md) TEL QUEL via the backend
 * endpoint.
 *
 * LEG-1 replaced the structured PLACEHOLDER that used to live here (a set of
 * `legal.privacy.sections.*` i18n keys under a "preliminary version" notice)
 * with the real document. Same architecture as /conditions — one source, served
 * verbatim, version stamped by the backend — so the two pages can never drift
 * apart, and the version a customer consents to is the version they read.
 */
export default function ConfidentialitePage() {
  const tPage = useTranslations('pages');
  return (
    <div className="container-prose py-12 sm:py-16">
      <Link
        href="/"
        className="mb-6 inline-flex items-center gap-1.5 text-sm text-muted-foreground underline-offset-4 hover:text-foreground hover:underline"
      >
        <ArrowLeft className="h-3.5 w-3.5" aria-hidden />
        {tPage('confidentialite.backHome')}
      </Link>
      <LegalDocument doc="privacy" />
    </div>
  );
}
