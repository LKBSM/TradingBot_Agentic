import type { Metadata } from 'next';
import { useTranslations } from 'next-intl';
import { getTranslations } from 'next-intl/server';
import { ResetPasswordForm } from '@/components/auth/ResetPasswordForm';

export async function generateMetadata({
  params,
}: {
  params: Promise<{ locale: string }>;
}): Promise<Metadata> {
  const { locale } = await params;
  const t = await getTranslations({ locale, namespace: 'pages' });
  return {
    title: t('motDePasseOublieConfirmer.meta.title'),
    description: t('motDePasseOublieConfirmer.meta.description'),
    robots: { index: false, follow: false },
  };
}

export default function ResetPasswordPage() {
  const t = useTranslations('pages');
  return (
    <div className="container-prose flex justify-center py-12 sm:py-16">
      <div className="w-full max-w-md space-y-6">
        <header className="space-y-1.5 text-center">
          <h1 className="text-2xl font-semibold tracking-tight">
            {t('motDePasseOublieConfirmer.title')}
          </h1>
        </header>
        <ResetPasswordForm />
      </div>
    </div>
  );
}
