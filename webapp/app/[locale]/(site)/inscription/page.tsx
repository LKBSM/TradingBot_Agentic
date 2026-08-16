import type { Metadata } from 'next';
import { useTranslations } from 'next-intl';
import { getTranslations } from 'next-intl/server';
import { Check } from 'lucide-react';
import { RegisterForm } from '@/components/auth/RegisterForm';
import { PRICING } from '@/lib/pricing.generated';

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
  const currency = useTranslations('billing')('currency');

  const values = [
    t('inscription.sell.value1'),
    t('inscription.sell.value2'),
    t('inscription.sell.value3'),
    t('inscription.sell.value4'),
    t('inscription.sell.value5'),
  ];
  const steps = [
    t('inscription.sell.step1'),
    t('inscription.sell.step2'),
    t('inscription.sell.step3'),
  ];
  const mentions = [
    t('inscription.sell.mentionRenew'),
    t('inscription.sell.mentionTool'),
    t('inscription.sell.mentionRisk'),
    t('inscription.sell.mentionAge'),
  ];

  return (
    <div className="container-prose py-12 sm:py-16">
      <div className="grid gap-10 lg:grid-cols-[1.1fr_1fr] lg:gap-14">
        {/* Left — what you're buying, how much, and what happens next. */}
        <section className="space-y-8">
          <header className="space-y-1.5">
            <h1 className="text-2xl font-semibold tracking-tight">
              {t('inscription.title')}
            </h1>
            <p className="text-sm text-muted-foreground">{t('inscription.intro')}</p>
          </header>

          <div className="space-y-3">
            <h2 className="text-sm font-medium uppercase tracking-wider text-muted-foreground">
              {t('inscription.sell.valueTitle')}
            </h2>
            <ul className="space-y-2">
              {values.map((v) => (
                <li key={v} className="flex items-start gap-2 text-sm text-foreground">
                  <Check className="mt-0.5 h-4 w-4 shrink-0 text-primary" aria-hidden />
                  <span>{v}</span>
                </li>
              ))}
            </ul>
          </div>

          <div className="space-y-2 rounded-lg border border-border/60 bg-muted/20 p-4">
            <h2 className="text-sm font-medium uppercase tracking-wider text-muted-foreground">
              {t('inscription.sell.priceTitle')}
            </h2>
            <p className="text-base font-semibold text-foreground">
              {t('inscription.sell.priceMonthly', {
                amount: PRICING.monthly,
                currency,
              })}
            </p>
            <p className="text-sm text-muted-foreground">
              {t('inscription.sell.priceAnnual', {
                total: PRICING.annualPerYear,
                perMonth: PRICING.annualPerMonth,
                currency,
              })}
            </p>
            <p className="text-xs text-muted-foreground">
              {t('inscription.sell.priceNote', { currency })}
            </p>
          </div>

          <div className="space-y-3">
            <h2 className="text-sm font-medium uppercase tracking-wider text-muted-foreground">
              {t('inscription.sell.stepsTitle')}
            </h2>
            <ol className="space-y-2">
              {steps.map((s, i) => (
                <li key={s} className="flex items-center gap-3 text-sm text-foreground">
                  <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full border border-border/60 text-xs font-medium">
                    {i + 1}
                  </span>
                  <span>{s}</span>
                </li>
              ))}
            </ol>
          </div>

          <ul className="space-y-1.5 border-t border-border/60 pt-4 text-xs text-muted-foreground">
            {mentions.map((m) => (
              <li key={m}>{m}</li>
            ))}
          </ul>
        </section>

        {/* Right — the form itself (email + Google, at parity). */}
        <section className="lg:pt-1">
          <div className="rounded-lg border border-border/60 p-5 sm:p-6">
            <h2 className="mb-4 text-base font-semibold text-foreground">
              {t('inscription.sell.formTitle')}
            </h2>
            <RegisterForm />
          </div>
        </section>
      </div>
    </div>
  );
}
