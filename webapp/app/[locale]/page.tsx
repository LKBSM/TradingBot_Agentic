import { useTranslations } from 'next-intl';
import Link from 'next/link';

export default function LandingPage() {
  const t = useTranslations('landing');
  const td = useTranslations('disclaimer');
  return (
    <div className="container-prose py-16">
      <h1 className="text-4xl font-bold tracking-tight sm:text-5xl">
        {t('hero_title')}
      </h1>
      <p className="mt-6 text-lg text-slate-600">{t('hero_subtitle')}</p>
      <div className="mt-8 flex gap-4">
        <Link
          href="/dashboard"
          className="rounded-full bg-sentinel-ink px-6 py-3 text-white"
        >
          {t('hero_cta_primary')}
        </Link>
        <Link
          href="/transparency"
          className="rounded-full border border-sentinel-ink px-6 py-3"
        >
          {t('hero_cta_secondary')}
        </Link>
      </div>

      <section className="mt-20 grid gap-6 sm:grid-cols-3">
        {[1, 2, 3].map((n) => (
          <div key={n} className="card">
            <h3 className="font-semibold">{t(`value_${n}_title`)}</h3>
            <p className="mt-2 text-sm text-slate-600">
              {t(`value_${n}_body`)}
            </p>
          </div>
        ))}
      </section>

      <p className="mt-16 text-xs text-slate-500 italic">{td('long')}</p>
    </div>
  );
}
