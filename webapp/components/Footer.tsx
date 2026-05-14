import { useTranslations } from 'next-intl';
import Link from 'next/link';

export function Footer() {
  const t = useTranslations('footer');
  return (
    <footer className="border-t border-slate-200 bg-slate-50 py-8 text-sm text-slate-600">
      <div className="container-prose flex flex-col items-center justify-between gap-4 sm:flex-row">
        <span>{t('copyright')}</span>
        <nav className="flex gap-4">
          <Link href="/terms">{t('terms')}</Link>
          <Link href="/privacy">{t('privacy')}</Link>
          <Link href="/contact">{t('contact')}</Link>
        </nav>
      </div>
    </footer>
  );
}
