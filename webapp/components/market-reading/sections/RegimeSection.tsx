'use client';

import { useTranslations } from 'next-intl';
import { Activity, AlertTriangle } from 'lucide-react';
import {
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion';
import { Badge } from '@/components/ui/badge';
import { InfoTooltip } from '@/components/ui/InfoTooltip';
import { cn } from '@/lib/utils';
import { type Tone } from '@/lib/market-reading/formatters';
import { useReadingFormatters } from '@/lib/market-reading/use-reading-formatters';
import { useMtfTrends } from '@/lib/market-reading/hooks';
import {
  MTF_TREND_ORDER,
  classifyMtfAlignment,
  mtfTrendGlyph,
} from '@/lib/market-reading/mtf-trend';
import type {
  MarketReadingHeader,
  MarketReadingRegime,
  MarketReadingStructure,
} from '@/types/market-reading';

const TONE_TO_VARIANT: Record<Tone, 'bull' | 'bear' | 'neutral' | 'warn'> = {
  bull: 'bull',
  bear: 'bear',
  neutral: 'neutral',
  warn: 'warn',
};

/**
 * Section "Régime de marché" — read-only, present-tense, descriptive.
 *
 * On top of the existing volatility + multi-timeframe trend alignment, it now
 * surfaces five additional facts the engine already produces:
 *   (a) market phase            → regime.market_phase
 *   (b) trend maturity          → most recent structure.choch_events (CHOCH only)
 *   (c) last structural event   → most recent of structure.bos / structure.choch
 *   (d) active zone density     → structure.order_blocks / fair_value_gaps (status active)
 *   (e) multi-TF disagreement   → classifyMtfAlignment(...).disagreement
 *
 * Nothing here detects, recomputes, scores or predicts. A fact that is absent for
 * the combo renders « non disponible » — never invented. The « contre » case
 * (e) is shown in a warn callout, with the same visibility as the « accord ».
 */
export function RegimeSection({
  regime,
  structure,
  header,
}: {
  regime: MarketReadingRegime;
  structure: MarketReadingStructure;
  header: MarketReadingHeader;
}) {
  const t = useTranslations('reading.regime');
  return (
    <AccordionItem value="regime">
      <AccordionTrigger className="text-left text-sm">
        <span className="flex items-center gap-2">
          <Activity className="h-4 w-4 text-muted-foreground" aria-hidden />
          <span>{t('title')}</span>
        </span>
      </AccordionTrigger>
      <AccordionContent>
        {/* Body lives in a child so the read-only MTF fetch fires lazily — only
            when the panel is expanded (Radix unmounts collapsed content). */}
        <RegimeBody regime={regime} structure={structure} header={header} />
      </AccordionContent>
    </AccordionItem>
  );
}

function RegimeBody({
  regime,
  structure,
  header,
}: {
  regime: MarketReadingRegime;
  structure: MarketReadingStructure;
  header: MarketReadingHeader;
}) {
  const t = useTranslations('reading.regime');
  const fmt = useReadingFormatters();
  const instrument = header.instrument;
  const { trends, isLoading } = useMtfTrends(instrument);
  const volatility = fmt.volatility(regime.volatility_observed);
  const phaseLabel = fmt.marketPhaseShort(regime.market_phase);
  const relation = classifyMtfAlignment(trends);
  const mtfText = fmt.mtfAlignmentText(trends, relation.kind);
  const hasAnyTrend = MTF_TREND_ORDER.some(({ key }) => trends[key] !== null);

  // (b)(c)(d) — present-tense facts read straight from the engine's structure.
  const maturity = fmt.regimeMaturity(structure, header);
  const lastEvent = fmt.regimeLastEvent(structure, header);
  const density = fmt.regimeZoneDensity(structure);

  return (
    <div className="space-y-4">
      {/* Volatilité + Phase de marché (a) */}
      <div className="flex flex-wrap items-center gap-2">
        <span className="inline-flex items-center gap-1">
          <Badge variant={TONE_TO_VARIANT[volatility.tone]}>
            {volatility.label}
          </Badge>
          <InfoTooltip termKey="volatility" iconOnly />
        </span>
        <span className="inline-flex items-center gap-1">
          <Badge variant="neutral">{t('phaseLabel', { label: phaseLabel })}</Badge>
          <InfoTooltip termKey="market_phase" iconOnly />
        </span>
      </div>

      {/* État structurel courant : maturité (b) · dernier événement (c) · zones (d) */}
      <div>
        <p className="mb-2 text-xs uppercase tracking-wide text-muted-foreground">
          {t('structuralState')}
        </p>
        <dl className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          <Fact
            label={t('factMaturity')}
            termKey="choch"
            value={maturity}
            className="sm:col-span-2"
          />
          <Fact label={t('factLastEvent')} termKey="choch" value={lastEvent} />
          <Fact label={t('factZones')} termKey="order_block" value={density} />
        </dl>
      </div>

      {/* Alignement multi-timeframe + désaccord (e) */}
      <div>
        <p className="mb-2 text-xs uppercase tracking-wide text-muted-foreground">
          <InfoTooltip termKey="mtf">{t('mtfTitle')}</InfoTooltip>
        </p>

        {isLoading && !hasAnyTrend ? (
          <p className="text-xs text-muted-foreground">{t('mtfLoading')}</p>
        ) : hasAnyTrend ? (
          <div className="flex flex-wrap items-center gap-2">
            {MTF_TREND_ORDER.map(({ key, label }) => {
              const g = mtfTrendGlyph(trends[key]);
              return (
                <Badge
                  key={key}
                  variant={TONE_TO_VARIANT[g.tone]}
                  className="font-mono tabular-nums"
                >
                  {label} {g.arrow}
                </Badge>
              );
            })}
          </div>
        ) : (
          <p className="text-xs text-muted-foreground">{t('mtfUnavailable')}</p>
        )}

        {/* The « contre » is given the same visibility as the « accord » — a warn
            callout, not a discreet grey line. */}
        {relation.disagreement ? (
          <div
            role="status"
            className="mt-2 flex items-start gap-2 rounded-md border border-sentinel-warn/30 bg-sentinel-warn/10 p-3"
          >
            <AlertTriangle
              className="mt-0.5 size-4 shrink-0 text-sentinel-warn"
              aria-hidden
            />
            <div className="space-y-0.5">
              <p className="text-sm font-semibold text-sentinel-warn">
                {t('disagreementTitle')}
              </p>
              <p className="text-sm text-foreground">{mtfText}</p>
            </div>
          </div>
        ) : (
          mtfText && <p className="mt-2 text-sm">{mtfText}</p>
        )}
      </div>

      <p className="text-xs italic text-muted-foreground">{t('disclaimer')}</p>
    </div>
  );
}

/**
 * One descriptive fact row. A null/empty `value` renders « non disponible » —
 * the engine had no datum for this combo and we never invent one.
 */
function Fact({
  label,
  value,
  termKey,
  className,
}: {
  label: string;
  value: string | null;
  termKey?: 'choch' | 'order_block';
  className?: string;
}) {
  const t = useTranslations('reading.regime');
  const available = value != null && value !== '';
  return (
    <div className={cn(className)}>
      <dt className="text-xs uppercase tracking-wide text-muted-foreground">
        {termKey ? <InfoTooltip termKey={termKey}>{label}</InfoTooltip> : label}
      </dt>
      <dd
        className={cn(
          'mt-1 text-sm font-medium',
          available ? 'text-foreground' : 'italic text-muted-foreground',
        )}
      >
        {available ? value : t('unavailable')}
      </dd>
    </div>
  );
}
