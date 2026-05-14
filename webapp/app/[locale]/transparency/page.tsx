'use client';

import { useTranslations } from 'next-intl';
import useSWR from 'swr';
import { EquityCurveChart } from '@/components/EquityCurveChart';
import { fetcher } from '@/lib/api';

export default function TransparencyPage() {
  const t = useTranslations('transparency');
  const { data: payload } = useSWR('/api/v1/forward-test/snapshot', fetcher);
  const stats = payload?.stats ?? {};

  return (
    <div className="container-prose py-12">
      <h1 className="text-3xl font-bold">{t('title')}</h1>
      <p className="mt-2 text-slate-600">{t('subtitle')}</p>

      <div className="mt-6 rounded-xl border border-amber-300 bg-amber-50 p-4 text-sm text-amber-900">
        {t('disclaimer')}
      </div>

      <section className="mt-10">
        <h2 className="mb-3 text-lg font-semibold">{t('stats_title')}</h2>
        <dl className="grid grid-cols-2 gap-4 sm:grid-cols-5">
          {[
            ['n_trades', stats.n_trades],
            ['win_rate', stats.win_rate],
            ['total_r', stats.total_R],
            ['max_dd', stats.max_drawdown_R],
            ['sharpe', stats.sharpe_per_trade],
          ].map(([k, v]) => (
            <div key={k as string} className="card text-center">
              <dt className="text-xs uppercase text-slate-500">{t(k as any)}</dt>
              <dd className="mt-1 font-mono text-xl">{v ?? '—'}</dd>
            </div>
          ))}
        </dl>
      </section>

      <section className="mt-12">
        <EquityCurveChart payload={payload} />
      </section>
    </div>
  );
}
