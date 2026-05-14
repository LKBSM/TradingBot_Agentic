import { useTranslations } from 'next-intl';
import Link from 'next/link';

const TIERS = [
  { key: 'free', price: 0, features_fr: ['Analyses publiques', 'Transparence en direct', '1 chat/jour'] },
  { key: 'lite', price: 19, features_fr: ['10 chats/jour', 'Notification Telegram', 'Glossaire interactif'] },
  { key: 'pro', price: 39, features_fr: ['Chat illimité', 'Multi-asset', 'API access read-only'] },
  { key: 'pro_plus', price: 99, features_fr: ['Tout PRO', 'Régime + corrélations', 'API access full', 'Support prioritaire'] },
];

export default function PricingPage() {
  const t = useTranslations('pricing');
  return (
    <div className="container-prose py-12">
      <h1 className="text-3xl font-bold">{t('title')}</h1>
      <p className="mt-2 text-slate-600">{t('subtitle')}</p>

      <div className="mt-10 grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
        {TIERS.map((tier) => (
          <div key={tier.key} className="card flex flex-col">
            <h3 className="text-lg font-semibold">
              {t(`${tier.key}_title` as any)}
            </h3>
            <p className="mt-2 text-3xl font-bold">
              {tier.price === 0 ? '—' : `€${tier.price}`}
              <span className="text-sm font-normal">
                {tier.price !== 0 ? t('billed_monthly') : ''}
              </span>
            </p>
            <ul className="mt-4 flex-1 space-y-2 text-sm">
              {tier.features_fr.map((f) => (
                <li key={f}>• {f}</li>
              ))}
            </ul>
            <Link
              href="/signup"
              className="mt-6 rounded-full bg-sentinel-ink py-2 text-center text-white"
            >
              {tier.price === 0 ? t('cta_signup') : t('cta_signup')}
            </Link>
          </div>
        ))}
      </div>
    </div>
  );
}
