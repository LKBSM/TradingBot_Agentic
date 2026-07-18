import type { Metadata } from 'next';
import { useTranslations } from 'next-intl';
import { getTranslations } from 'next-intl/server';
import { ForgotPasswordForm } from '@/components/auth/ForgotPasswordForm';

export async function generateMetadata({
  params,
}: {
  params: Promise<{ locale: string }>;
}): Promise<Metadata> {
  const { locale } = await params;
  const t = await getTranslations({ locale, namespace: 'pages' });
  return {
    title: t('motDePasseOublie.meta.title'),
    description: t('motDePasseOublie.meta.description'),
    robots: { index: false, follow: false },
  };
}

export default function ForgotPasswordPage() {
  const t = useTranslations('pages');
  return (
    <div className="container-prose flex justify-center py-12 sm:py-16">
      <div className="w-full max-w-md space-y-6">
        <header className="space-y-1.5 text-center">
          <h1 className="text-2xl font-semibold tracking-tight">
            {t('motDePasseOublie.title')}
          </h1>
        </header>
        <ForgotPasswordForm />
      </div>
    </div>
  );
}
