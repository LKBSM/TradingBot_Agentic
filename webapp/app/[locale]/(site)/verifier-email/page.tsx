import type { Metadata } from 'next';
import { Suspense } from 'react';
import { getTranslations } from 'next-intl/server';
import { EmailVerifier } from '@/components/auth/EmailVerifier';

export async function generateMetadata({
  params,
}: {
  params: Promise<{ locale: string }>;
}): Promise<Metadata> {
  const { locale } = await params;
  const t = await getTranslations({ locale, namespace: 'auth' });
  return {
    title: t('verifyEmail.title'),
    robots: { index: false, follow: false },
  };
}

export default function VerifyEmailPage() {
  return (
    <div className="container-prose py-12 sm:py-16">
      <Suspense fallback={null}>
        <EmailVerifier />
      </Suspense>
    </div>
  );
}
