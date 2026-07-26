import type { Metadata } from 'next';
import { useLocale, useTranslations } from 'next-intl';
import { getTranslations } from 'next-intl/server';
import { CandleDriftCanvas } from '@/components/auth/CandleDriftCanvas';
import { LoginForm } from '@/components/auth/LoginForm';
import { localizeHref } from '@/lib/i18n/href';
import '@/components/auth/login-hero.css';

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
  const tf = useTranslations('footer');
  const locale = useLocale();
  return (
    <div className="login-hero">
      <CandleDriftCanvas />
      <div className="login-stack">
        <LoginForm />
        <div className="under-links">
          <a href={localizeHref('/conditions', locale)}>
            {tf('legal.terms')}
          </a>
          ·
          <a href={localizeHref('/confidentialite', locale)}>
            {tf('legal.privacy')}
          </a>
        </div>
      </div>
    </div>
  );
}
