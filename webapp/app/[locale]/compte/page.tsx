import type { Metadata } from 'next';
import { getTranslations } from 'next-intl/server';
import { AccountPanel } from '@/components/auth/AccountPanel';

export async function generateMetadata({
  params,
}: {
  params: Promise<{ locale: string }>;
}): Promise<Metadata> {
  const { locale } = await params;
  const t = await getTranslations({ locale, namespace: 'pages' });
  return {
    title: t('compte.meta.title'),
    description: t('compte.meta.description'),
    robots: { index: false, follow: false },
  };
}

export default function AccountPage() {
  return (
    <div className="container-prose py-12 sm:py-16">
      <AccountPanel />
    </div>
  );
}
