'use client';

import Link from 'next/link';
import { useTranslations } from 'next-intl';
import { buildAppHref } from '@/lib/conditions/app-link';
import { mtfOrderFor } from '@/lib/market-reading/mtf-trend';
import type { ComboMatch, ConditionOutcome } from '@/lib/conditions/types';
import { biasGlyph, instrumentLabel } from './labels';
import { useScannerLabels } from './use-scanner-labels';

/**
 * One scan result, rendered as a `.combo` card with THREE blocks ALWAYS in the
 * same order (SC-1):
 *   1. « Ce qui correspond »        — each met condition with its OBSERVED value.
 *   2. « Ce qui va à l'encontre »   — unmet conditions + context against. This
 *      block is NEVER hidden nor collapsible (a reading you only half-read is a
 *      reading you misread).
 *   3. « Contexte que tu n'as pas demandé » — nearby pockets, upcoming release,
 *      OB/FVG, BOS — context the user did not ask for but should see.
 * A NON-EVALUABLE condition is shown apart (it adjusts the denominator, counted
 * neither as met nor as against). No score, no ranking. "Analyser" / "Ouvrir
 * dans le graphique" send the user to look for themselves — never "Trader".
 */
export function ComboCard({
  match,
  locale,
  now,
}: {
  match: ComboMatch;
  locale: string;
  now?: number;
}) {
  const t = useTranslations('scanner.combo');
  const { bias, phase, age } = useScannerLabels();
  const ctx = match.context;
  const readingAge = age(match.candle_close_ts, now ?? Date.now());

  const nonEvaluable: ConditionOutcome[] = match.conditions_non_evaluable ?? [];
  const nonEvaluableCount = match.non_evaluable_count ?? nonEvaluable.length;
  const againstItems = match.context_against ?? [];

  const importantNewsCount = ctx.news_upcoming.filter((n) => n.impact === 'high').length;

  // MTF units RELATIVE to this combo's timeframe (TF-1), for the context block.
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
          {nonEvaluableCount > 0 ? ` · ${t('nonEvaluableCount', { count: nonEvaluableCount })}` : ''}
        </span>
      </div>

      {/* Block 1 — Ce qui correspond */}
      <div className="blk-lbl">{t('matchBlock')}</div>
      {match.conditions_met.length === 0 ? (
        <p className="blk-empty">{t('matchNone')}</p>
      ) : (
        match.conditions_met.map((c) => (
          <div key={`met-${c.type}`} className="cl yes">
            <span className="mk2" aria-hidden>✓</span>
            <span>
              <span className="font-medium">{c.label}</span>
              <span className="text-[color:var(--faint)]"> — {c.detail}</span>
            </span>
          </div>
        ))
      )}

      {/* Block 2 — Ce qui va à l'encontre (NEVER hidden nor collapsible).
          Holds the UNMET selected conditions AND the factual against-signals
          (multi-unit disagreement, contracted volatility, tested zone) that the
          engine surfaces even on a full match. */}
      <div className="blk-lbl" data-testid="against-block">{t('againstBlock')}</div>
      {match.conditions_unmet.length === 0 && againstItems.length === 0 ? (
        <p className="blk-empty">{t('againstNone')}</p>
      ) : (
        <>
          {match.conditions_unmet.map((c) => (
            <div key={`unmet-${c.type}`} className="cl no">
              <span className="mk2" aria-hidden />
              <span>
                <span className="font-medium">{c.label}</span>
                <span className="text-[color:var(--faint)]"> — {c.detail}</span>
              </span>
            </div>
          ))}
          {againstItems.map((it, i) => (
            <div key={`against-${i}`} className="cl no">
              <span className="mk2" aria-hidden>✗</span>
              <span>
                <span className="font-medium">{it.label}</span>
                <span className="text-[color:var(--faint)]"> — {it.detail}</span>
              </span>
            </div>
          ))}
        </>
      )}

      {/* Non-evaluable — a third state, adjusts the denominator (not met, not against) */}
      {nonEvaluable.length > 0 && (
        <>
          <div className="blk-lbl">{t('nonEvaluableBlock')}</div>
          {nonEvaluable.map((c) => (
            <div key={`na-${c.type}`} className="cl na">
              <span className="mk2" aria-hidden>–</span>
              <span>
                <span className="font-medium">{c.label}</span>
                <span className="text-[color:var(--faint)]"> — {c.detail}</span>
              </span>
            </div>
          ))}
        </>
      )}

      {/* Block 3 — Contexte que tu n'as pas demandé */}
      <div className="blk-lbl">{t('contextBlock')}</div>
      <div className="ctx2">
        {t('trend', { label: bias(ctx.trend) })} · {phase(ctx.market_phase)} · MTF {mtf} ·{' '}
        {t('obFvg', { ob: ctx.active_order_blocks, fvg: ctx.active_fair_value_gaps })}
        {ctx.bos ? t('bosSuffix', { dir: bias(ctx.bos.direction) }) : ''}
        {ctx.structural_range
          ? ` · ${t('range', { low: ctx.structural_range.low, high: ctx.structural_range.high })}`
          : ''}
        {importantNewsCount > 0 ? ` · ${t('importantNews', { count: importantNewsCount })}` : ''}
      </div>

      {match.close_price != null && (
        <div className="mono mt-1 text-[10px] text-[color:var(--faint)]">
          {t('price', { price: match.close_price })}
          {readingAge ? ` · ${t('freshCandle', { age: readingAge })}` : ''}
        </div>
      )}

      <div className="mt-2 flex flex-wrap gap-2">
        <Link className="btn acc" href={buildAppHref(locale, match)}>
          {t('analyse')} →
        </Link>
        <Link className="btn" href={buildAppHref(locale, match)}>
          {t('openChart')}
        </Link>
      </div>

      <p className="mt-2 text-[10px] leading-snug text-[color:var(--faint)]">{t('disclaimer')}</p>
    </div>
  );
}
