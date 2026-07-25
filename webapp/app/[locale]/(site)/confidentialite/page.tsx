import type { Metadata } from 'next';
import Link from 'next/link';
import { useTranslations } from 'next-intl';
import { getTranslations } from 'next-intl/server';
import { ArrowLeft, ShieldCheck } from 'lucide-react';

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
 * /confidentialite — placeholder STRUCTURÉ. The full, lawyer-reviewed privacy
 * policy is mission ④ (terminal légal). This page already exposes the spine of
 * the document (controller, data, basis, rights, contact) so the consent
 * checkbox at registration links to a real, honest page — but the canonical
 * legally-binding text is explicitly pending.
 *
 * Section keys map 1:1 to `legal.privacy.sections.*` — order is fixed and the
 * numbering lives in the translated titles (verbatim legal text).
 */
const SECTION_KEYS = [
  'responsable',
  'donnees',
  'bases',
  'droits',
  'conservation',
  'contact',
] as const;

export default function ConfidentialitePage() {
  const t = useTranslations('legal');
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

      <header className="space-y-3">
        <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">
          {t('privacy.title')}
        </h1>
        <div
          className="flex items-start gap-2 rounded-md border border-dashed border-border/70 bg-card/50 p-3 text-sm text-muted-foreground"
          data-legal-pending="privacy-placeholder"
        >
          <ShieldCheck className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
          <span>{t('privacy.pendingNotice')}</span>
        </div>
      </header>

      <div className="mt-8 space-y-6">
        {SECTION_KEYS.map((key) => (
          <section key={key} className="space-y-1.5">
            <h2 className="text-lg font-semibold text-foreground">
              {t(`privacy.sections.${key}.title`)}
            </h2>
            <p className="leading-relaxed text-muted-foreground">
              {t(`privacy.sections.${key}.body`)}
            </p>
          </section>
        ))}
      </div>
    </div>
  );
}
