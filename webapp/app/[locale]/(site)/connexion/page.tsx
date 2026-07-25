import type { Metadata } from 'next';
import { useTranslations } from 'next-intl';
import { getTranslations } from 'next-intl/server';
import { LoginForm } from '@/components/auth/LoginForm';

export async function generateMetadata({
  params,
}: {
  params: Promise<{ locale: string }>;
}): Promise<Metadata> {
  const { locale } = await params;
  const t = await getTranslations({ locale, namespace: 'pages' });
  return {
    title: t('connexion.meta.title'),
    description: t('connexion.meta.description'),
    robots: { index: false, follow: false },
  };
}

export default function LoginPage() {
  const t = useTranslations('pages');
  return (
    <div className="container-prose flex justify-center py-12 sm:py-16">
      <div className="w-full max-w-md space-y-6">
        <header className="space-y-1.5 text-center">
          <h1 className="text-2xl font-semibold tracking-tight">
            {t('connexion.title')}
          </h1>
          <p className="text-sm text-muted-foreground">
            {t('connexion.intro')}
          </p>
        </header>
        <LoginForm />
      </div>
    </div>
  );
}
