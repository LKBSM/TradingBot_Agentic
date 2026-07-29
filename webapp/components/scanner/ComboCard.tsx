'use client';

import Link from 'next/link';
import { useTranslations } from 'next-intl';
import { buildAppHref } from '@/lib/conditions/app-link';
import { mtfOrderFor } from '@/lib/market-reading/mtf-trend';
import type { ComboMatch } from '@/lib/conditions/types';
import { biasGlyph, instrumentLabel } from './labels';
import { useScannerLabels } from './use-scanner-labels';

/**
 * One scan result, rendered as a terminal-reference `.combo` card (UI-2):
 *   [ market · TF ] · met/unmet conditions (.cl.yes/.cl.no) · context (.ctx2) · Analyser
 *
 * Shows which conditions are met AND which are unmet (transparency — "2 de tes
 * 3"), and the FULL context (including what goes against). No quality score,
 * no ranking. "Analyser" sends the user to look at the chart themselves.
 */
export function ComboCard({
  match,
  locale,
  now,
}: {
  match: ComboMatch;
  locale: string;
  /** Injected ticking clock (epoch ms) so the age stays honest; defaults to now. */
  now?: number;
}) {
  const t = useTranslations('scanner.combo');
  const { bias, phase, age } = useScannerLabels();
  const ctx = match.context;
  const readingAge = age(match.candle_close_ts, now ?? Date.now());

  // Heads-up factuel : on ne garde que les actus à IMPACT HAUT (les
  // moyennes/faibles sont du bruit). Aucune direction, aucune prédiction —
  // juste un signalement de volatilité programmée. Détection inchangée : on
  // filtre uniquement à l'affichage.
  const importantNewsCount = ctx.news_upcoming.filter(
    (n) => n.impact === 'high',
  ).length;

  // Full context sentence: trend · phase · MTF · OB/FVG (+ BOS/news heads-up).
  // Descriptive only — includes what goes against, never a recommendation. The
  // MTF units are RELATIVE to this combo's timeframe (TF-1 decision C), not a
  // fixed H4·H1·M15 triplet.
  const mtf = mtfOrderFor(match.timeframe)
    .map(({ key, label }) => {
      const mtfTrend = ctx.mtf_trends?.[key] ?? ctx.mtf_confluence?.[key];
      return `${label} ${biasGlyph(mtfTrend)}`;
    })
    .join(' · ');

  return (
    <div className="combo">
      <div className="t1">
        <span className="nm">{instrumentLabel(match.instrument)}</span>
        <span className="tf2">· {match.timeframe}</span>
        <span className="mono ml-auto text-[10px] text-[color:var(--faint)]">
          {t('conditionsPresent', { met: match.met_count, total: match.total })}
        </span>
      </div>

      {/* Conditions met (transparent) */}
      {match.conditions_met.map((c) => (
        <div key={`met-${c.type}`} className="cl yes">
          <span className="mk2" aria-hidden>
            ✓
          </span>
          <span>
            <span className="font-medium">{c.label}</span>
            <span className="text-[color:var(--faint)]"> — {c.detail}</span>
          </span>
        </div>
      ))}
      {/* Conditions unmet (transparent both ways) */}
      {match.conditions_unmet.map((c) => {
        // Data gap (a sibling timeframe not read yet) reads differently from a
        // genuine "not met" — never let the client mistake one for the other.
        const unavailable = c.available === false;
        return (
          <div key={`unmet-${c.type}`} className="cl no">
            <span className="mk2" aria-hidden />
            <span>
              <span className="font-medium">{c.label}</span>
              {unavailable && (
                <span className="mx-1 align-middle text-[10px] uppercase text-[color:var(--faint)]">
                  ({t('unavailable')})
                </span>
              )}{' '}
              — {c.detail}
            </span>
          </div>
        );
      })}

      {/* Full context — descriptive, includes what goes against */}
      <div className="ctx2">
        <b>{t('fullContext')} :</b>{' '}
        {t('trend', { label: bias(ctx.trend) })} · {phase(ctx.market_phase)} · MTF {mtf} ·{' '}
        {t('obFvg', { ob: ctx.active_order_blocks, fvg: ctx.active_fair_value_gaps })}
        {ctx.bos ? t('bosSuffix', { dir: bias(ctx.bos.direction) }) : ''}
        {match.close_price != null ? ` · ${t('price', { price: match.close_price })}` : ''}
        {readingAge ? ` · ${t('reading', { age: readingAge })}` : ''}
        {importantNewsCount > 0
          ? ` · ${t('importantNews', { count: importantNewsCount })}`
          : ''}
      </div>

      <Link className="btn acc" href={buildAppHref(locale, match)}>
        {t('analyse')} →
      </Link>

      <p className="mt-2 text-[10px] leading-snug text-[color:var(--faint)]">
        {t('disclaimer')}
      </p>
    </div>
  );
}
