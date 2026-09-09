'use client';

import Link from 'next/link';
import { useTranslations } from 'next-intl';
import { buildAppHref } from '@/lib/conditions/app-link';
import { mtfOrderFor } from '@/lib/market-reading/mtf-trend';
import type { ComboMatch, ConditionOutcome, ConditionType } from '@/lib/conditions/types';
import { biasGlyph, biasTone, instrumentLabel } from './labels';
import { useScannerLabels } from './use-scanner-labels';

/**
 * SC-3 — the condition types whose backend `detail` is a bare VALUE completing
 * the label, rather than a standalone explanatory sentence. For those, the card
 * renders « label + value » with NO em-dash between them:
 *
 *   « La tendance structurelle est » + « haussier. »
 *       → « La tendance structurelle est haussier. »
 *
 * Every OTHER condition keeps the « label — detail » form, because its detail IS
 * a sentence (« Le prix est dans un Order Block » — « Prix dans un Order Block
 * actif (4382.1–4401.7). ») and would read as gibberish glued to its label.
 *
 * These seven are the entries reworked when the forbidden word « cible » was
 * removed from `src/intelligence/conditions_scanner.py`. The eighth reworked
 * entry, `price_in_range_third`, is deliberately ABSENT here: its label carries
 * an inline placeholder (« Le prix est dans le tiers … du range »), so a value
 * appended after it would not read as French — it keeps the em-dash form.
 *
 * `context_against` items are never value-only: they are « label — explanation »
 * pairs, not « label + value », even though they live in the same block as the
 * unmet conditions.
 */
const VALUE_ONLY_DETAIL: ReadonlySet<ConditionType> = new Set<ConditionType>([
  'trend_is',
  'higher_tf_agrees',
  'last_event_is',
  'last_event_age',
  'market_phase_is',
  'volatility_is',
  'session_is',
]);

/** True when `type` composes as « label value », false for « label — detail ». */
export function isValueOnlyDetail(type: string): boolean {
  return VALUE_ONLY_DETAIL.has(type as ConditionType);
}

/**
 * One condition line: its label, then either its value (no separator) or its
 * explanatory detail (em-dash separator). The label always keeps the readable
 * weight; the trailing part is receded.
 */
function ConditionText({ label, detail, type }: { label: string; detail: string; type?: string }) {
  const valueOnly = type != null && isValueOnlyDetail(type);
  return (
    <span>
      <span className="font-medium">{label}</span>
      <span className="text-[color:var(--faint)]">{valueOnly ? ' ' : ' — '}{detail}</span>
    </span>
  );
}

/** One structure chip: a coloured dot and a short factual label. */
type Chip = { key: string; tone: string; label: React.ReactNode };

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
 *
 * SC-3 reworked the DENSITY of that same material, never its content:
 *   · Block 1 leads with ONE full-weight fact; the other met conditions follow
 *     in the same family at reduced weight. Nothing is dropped or truncated —
 *     a card matching four conditions still shows all four.
 *   · A row of coloured chips summarises the structure. Every chip is backed by
 *     a real field of `match.context`; a field that is null (or a count of zero)
 *     produces NO chip — never a dash, never a default zero. Nothing is invented
 *     and nothing extra is fetched: the whole row reads the scan response the
 *     card already received.
 *   · Block 3 folds into a <details>, closed by default, with the high-impact
 *     news count kept in the <summary> so that warning stays readable folded.
 *   · Block 2 does NOT fold. It is never wrapped in a <details>, not even one
 *     open by default (see the block comment on it below).
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

  /*
   * SC-3 — the structure chips. Each entry is pushed ONLY when its source field
   * is really there. A null `structural_range` / `bos` / `choch`, an absent
   * trend or phase, a zone count of zero: no chip at all. A count of zero is a
   * true reading, not a missing one — but a « 0 OB » chip would say nothing, so
   * the row keeps only what IS present and the exact counts stay in block 3
   * below, which still lists them whatever their value.
   */
  const chips: Chip[] = [];
  if (ctx.trend) {
    chips.push({ key: 'trend', tone: biasTone(ctx.trend), label: bias(ctx.trend) });
  }
  if (ctx.market_phase) {
    chips.push({ key: 'phase', tone: 'neutral', label: phase(ctx.market_phase) });
  }
  if (ctx.active_order_blocks > 0) {
    chips.push({
      key: 'ob',
      tone: 'ob',
      label: t.rich('chipOb', {
        count: ctx.active_order_blocks,
        n: (c) => <span className="mono">{c}</span>,
      }),
    });
  }
  if (ctx.active_fair_value_gaps > 0) {
    chips.push({
      key: 'fvg',
      tone: 'fvg',
      label: t.rich('chipFvg', {
        count: ctx.active_fair_value_gaps,
        n: (c) => <span className="mono">{c}</span>,
      }),
    });
  }
  if (ctx.bos?.direction) {
    chips.push({
      key: 'bos',
      tone: biasTone(ctx.bos.direction),
      label: t('chipBos', { dir: bias(ctx.bos.direction) }),
    });
  }
  if (ctx.structural_range) {
    chips.push({
      key: 'range',
      tone: 'acc',
      label: (
        <span className="mono">
          {ctx.structural_range.low}–{ctx.structural_range.high}
        </span>
      ),
    });
  }

  return (
    <div className="combo">
      <div className="t1">
        <span className="nm">{instrumentLabel(match.instrument)}</span>
        <span className="tf2">· {match.timeframe}</span>
        <span className="mono ml-auto fs-legal text-[color:var(--faint)]">
          {t('conditionsPresent', { met: match.met_count, total: match.total })}
          {nonEvaluableCount > 0 ? ` · ${t('nonEvaluableCount', { count: nonEvaluableCount })}` : ''}
        </span>
      </div>

      {/* Block 1 — Ce qui correspond. The FIRST met condition carries the full
          weight (its own marker, readable text); the following ones stay in the
          same family, receded. All of them are rendered: a card matching four
          conditions shows four lines, none truncated. */}
      <div className="blk-lbl">{t('matchBlock')}</div>
      {match.conditions_met.length === 0 ? (
        <p className="blk-empty">{t('matchNone')}</p>
      ) : (
        match.conditions_met.map((c, i) => (
          <div key={`met-${c.type}`} className={i === 0 ? 'cl yes lead' : 'cl yes sub'}>
            {i === 0 ? (
              <span className="mk2" aria-hidden>✓</span>
            ) : (
              <span className="mk2 mk2-void" aria-hidden />
            )}
            <ConditionText label={c.label} detail={c.detail} type={c.type} />
          </div>
        ))
      )}

      {/* Structure chips — a coloured summary of what this reading holds. Each
          one is backed by a real field of the scan response (see `chips` above);
          an absent field yields no chip. Purely additive: everything summarised
          here is also written out in the context block further down. */}
      {chips.length > 0 && (
        <div className="chips" data-testid="combo-chips" aria-label={t('chipsAria')}>
          {chips.map((c) => (
            <span className="chip" key={c.key} data-chip={c.key}>
              <span className={`dot d-${c.tone}`} aria-hidden />
              {c.label}
            </span>
          ))}
        </div>
      )}

      {/* Block 2 — Ce qui va à l'encontre (NEVER hidden nor collapsible).
          Holds the UNMET selected conditions AND the factual against-signals
          (multi-unit disagreement, contracted volatility, tested zone) that the
          engine surfaces even on a full match.
          SC-3: block 3 below folds into a <details>; THIS ONE MUST NOT, not even
          inside a <details open>. « A reading you only half-read is a reading you
          misread » is a promise made on the landing page and in two audits
          (AUDIT-lp-1-accueil, AUDIT-lp-2); the SC-3 design mock did fold it, and
          we deliberately departed from the mock on this single point. Guarded by
          `sc1-results.test.tsx`, `sc3-against-block.test.tsx` and
          `sc1-scanner.spec.ts`. */}
      <div className="blk-lbl" data-testid="against-block">{t('againstBlock')}</div>
      {match.conditions_unmet.length === 0 && againstItems.length === 0 ? (
        <p className="blk-empty">{t('againstNone')}</p>
      ) : (
        <>
          {match.conditions_unmet.map((c) => (
            <div key={`unmet-${c.type}`} className="cl no">
              <span className="mk2" aria-hidden />
              <ConditionText label={c.label} detail={c.detail} type={c.type} />
            </div>
          ))}
          {againstItems.map((it, i) => (
            <div key={`against-${i}`} className="cl no">
              <span className="mk2" aria-hidden>✗</span>
              <ConditionText label={it.label} detail={it.detail} />
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
              <ConditionText label={c.label} detail={c.detail} type={c.type} />
            </div>
          ))}
        </>
      )}

      {/* Block 3 — Contexte que tu n'as pas demandé. Folded by default (SC-3):
          it is, by definition, what the user did NOT ask for. Its body is
          unchanged — every fact stays one click away, none is dropped. The
          high-impact news count rides in the SUMMARY, so the one warning in
          there stays readable while the block is closed. */}
      <details className="ctxd" data-testid="context-block">
        <summary className="blk-lbl">
          {importantNewsCount > 0
            ? t('contextBlockWithNews', { count: importantNewsCount })
            : t('contextBlock')}
        </summary>
        <div className="ctx2">
          {t('trend', { label: bias(ctx.trend) })} · {phase(ctx.market_phase)} · MTF {mtf} ·{' '}
          {t('obFvg', { ob: ctx.active_order_blocks, fvg: ctx.active_fair_value_gaps })}
          {ctx.bos ? t('bosSuffix', { dir: bias(ctx.bos.direction) }) : ''}
          {ctx.structural_range
            ? ` · ${t('range', { low: ctx.structural_range.low, high: ctx.structural_range.high })}`
            : ''}
          {importantNewsCount > 0 ? ` · ${t('importantNews', { count: importantNewsCount })}` : ''}
        </div>
      </details>

      {match.close_price != null && (
        <div className="mono mt-1 fs-legal text-[color:var(--faint)]">
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
      {/* CLN-1 §5 — the per-combo educational note was removed: the single page
          disclaimer (rail footer on desktop, mobile footer < 768px) already
          carries the « ne prédit rien, ne recommande rien » posture, and one
          notice per result card stacked it many times over on one screen. */}
    </div>
  );
}
