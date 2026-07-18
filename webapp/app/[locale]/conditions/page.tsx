import type { Metadata } from 'next';
import Link from 'next/link';
import { useTranslations } from 'next-intl';
import { getTranslations } from 'next-intl/server';
import { ArrowLeft } from 'lucide-react';
import { ConditionsDocument } from '@/components/legal/ConditionsDocument';

export async function generateMetadata({
  params,
}: {
  params: Promise<{ locale: string }>;
}): Promise<Metadata> {
  const { locale } = await params;
  const t = await getTranslations({ locale, namespace: 'legal' });
  return {
    title: t('conditions.meta.title'),
    description: t('conditions.meta.description'),
  };
}

/**
 * /conditions — renders the canonical CGU document
 * (docs/legal/conditions-utilisation.md) TEL QUEL via the backend endpoint. The
 * text is never rewritten here; only the markdown is formatted for the web.
 */
export default function ConditionsPage() {
  const t = useTranslations('pages');
  return (
    <div className="container-prose py-12 sm:py-16">
      <Link
        href="/"
        className="mb-6 inline-flex items-center gap-1.5 text-sm text-muted-foreground underline-offset-4 hover:text-foreground hover:underline"
      >
        <ArrowLeft className="h-3.5 w-3.5" aria-hidden />
        {t('conditions.backHome')}
      </Link>
      <ConditionsDocument />
    </div>
  );
}
