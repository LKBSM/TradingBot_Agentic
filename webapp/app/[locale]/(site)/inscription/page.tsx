import type { Metadata } from 'next';
import { useTranslations } from 'next-intl';
import { getTranslations } from 'next-intl/server';
import { RegisterForm } from '@/components/auth/RegisterForm';

export async function generateMetadata({
  params,
}: {
  params: Promise<{ locale: string }>;
}): Promise<Metadata> {
  const { locale } = await params;
  const t = await getTranslations({ locale, namespace: 'pages' });
  return {
    title: t('inscription.meta.title'),
    description: t('inscription.meta.description'),
    robots: { index: false, follow: false },
  };
}

export default function RegisterPage() {
  const t = useTranslations('pages');
  return (
    <div className="container-prose flex justify-center py-12 sm:py-16">
      <div className="w-full max-w-md space-y-6">
        <header className="space-y-1.5 text-center">
          <h1 className="text-2xl font-semibold tracking-tight">
            {t('inscription.title')}
          </h1>
          <p className="text-sm text-muted-foreground">
            {t('inscription.intro')}
          </p>
        </header>
        <RegisterForm />
      </div>
    </div>
  );
}
