import type { Metadata } from 'next';
import { Suspense } from 'react';
import { getTranslations } from 'next-intl/server';
import { GoogleFinalizeForm } from '@/components/auth/GoogleFinalizeForm';

export async function generateMetadata({
  params,
}: {
  params: Promise<{ locale: string }>;
}): Promise<Metadata> {
  const { locale } = await params;
  const t = await getTranslations({ locale, namespace: 'auth' });
  return {
    title: t('google.finalizeTitle'),
    robots: { index: false, follow: false },
  };
}

export default function GoogleFinalizePage() {
  return (
    <div className="mx-auto max-w-md px-4 py-12">
      <Suspense fallback={null}>
        <GoogleFinalizeForm />
      </Suspense>
    </div>
  );
}
