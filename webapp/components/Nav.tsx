'use client';

import { useTranslations, useLocale } from 'next-intl';
import { usePathname, useRouter } from 'next/navigation';
import Link from 'next/link';
import { SUPPORTED_LOCALES } from '../i18n';

const LINKS: { key: 'dashboard' | 'narratives' | 'transparency' | 'glossary' | 'pricing' | 'chat'; href: string }[] = [
  { key: 'dashboard', href: '/dashboard' },
  { key: 'transparency', href: '/transparency' },
  { key: 'glossary', href: '/glossary' },
  { key: 'pricing', href: '/pricing' },
  { key: 'chat', href: '/chat' },
];

export function Nav() {
  const t = useTranslations('nav');
  const locale = useLocale();
  const router = useRouter();
  const pathname = usePathname();

  function switchLocale(newLocale: string) {
    // pathname starts with /<locale>/... — replace prefix.
    const stripped = pathname.replace(new RegExp(`^/${locale}`), '');
    router.push(`/${newLocale}${stripped}`);
  }

  return (
    <header className="border-b border-slate-200 bg-white">
      <div className="container-prose flex items-center justify-between py-4">
        <Link href="/" className="font-bold tracking-tight">
          Smart Sentinel AI
        </Link>
        <nav className="hidden gap-6 text-sm sm:flex">
          {LINKS.map((l) => (
            <Link key={l.key} href={l.href} className="text-slate-700 hover:text-sentinel-ink">
              {t(l.key)}
            </Link>
          ))}
        </nav>
        <select
          aria-label={t('language')}
          className="rounded-md border border-slate-300 px-2 py-1 text-sm"
          value={locale}
          onChange={(e) => switchLocale(e.target.value)}
        >
          {SUPPORTED_LOCALES.map((l) => (
            <option key={l} value={l}>
              {l.toUpperCase()}
            </option>
          ))}
        </select>
      </div>
    </header>
  );
}
