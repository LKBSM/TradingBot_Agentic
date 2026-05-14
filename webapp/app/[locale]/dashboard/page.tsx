'use client';

import { useTranslations } from 'next-intl';
import useSWR from 'swr';
import { NarrativeCard } from '@/components/NarrativeCard';
import { EquityCurveChart } from '@/components/EquityCurveChart';
import { RegimeTimeline } from '@/components/RegimeTimeline';
import { fetcher } from '@/lib/api';

export default function DashboardPage() {
  const t = useTranslations('nav');
  const td = useTranslations('disclaimer');

  const { data: insights } = useSWR('/api/v1/insights/history?limit=5', fetcher);
  const { data: paper } = useSWR('/api/v1/forward-test/snapshot', fetcher);
  const { data: regime } = useSWR('/api/v1/regime/timeline', fetcher);

  return (
    <div className="container-prose py-12">
      <h1 className="text-2xl font-bold">{t('dashboard')}</h1>
      <p className="mt-2 text-xs italic text-slate-500">{td('short')}</p>

      <section className="mt-8">
        <h2 className="mb-3 text-lg font-semibold">{t('narratives')}</h2>
        <div className="space-y-4">
          {(insights?.entries ?? []).slice(0, 3).map((row: any) => (
            <NarrativeCard key={row.seq} entry={row} />
          ))}
        </div>
      </section>

      <section className="mt-12">
        <h2 className="mb-3 text-lg font-semibold">{t('transparency')}</h2>
        <EquityCurveChart payload={paper} />
      </section>

      <section className="mt-12">
        <h2 className="mb-3 text-lg font-semibold">Regime timeline</h2>
        <RegimeTimeline payload={regime} />
      </section>
    </div>
  );
}
