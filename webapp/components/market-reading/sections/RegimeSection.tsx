'use client';

import { AlertTriangle } from 'lucide-react';
import {
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion';
import { Badge } from '@/components/ui/badge';
import { InfoTooltip } from '@/components/ui/InfoTooltip';
import { cn } from '@/lib/utils';
import {
  formatMarketPhaseShort,
  formatVolatility,
  type Tone,
} from '@/lib/market-reading/formatters';
import {
  formatLastStructuralEvent,
  formatTrendMaturity,
  formatZoneDensity,
} from '@/lib/market-reading/regime-facts';
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

const UNAVAILABLE = 'non disponible';

/**
 * Section "Régime de marché" — read-only, present-tense, descriptive.
 *
 * On top of the existing volatility + multi-timeframe trend alignment, it now
 * surfaces five additional facts the engine already produces:
 *   (a) market phase            → regime.market_phase
 *   (b) trend maturity          → structure.choch, else structure.bos (fallback)
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
  return (
    <AccordionItem value="regime">
      <AccordionTrigger className="text-left text-sm">
        <span className="flex items-center gap-2">
          <span aria-hidden>🌊</span>
          <span>Régime de marché</span>
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
  const instrument = header.instrument;
  const { trends, isLoading } = useMtfTrends(instrument);
  const volatility = formatVolatility(regime.volatility_observed);
  const phaseLabel = formatMarketPhaseShort(regime.market_phase);
  const relation = classifyMtfAlignment(trends);
  const hasAnyTrend = MTF_TREND_ORDER.some(({ key }) => trends[key] !== null);

  // (b)(c)(d) — present-tense facts read straight from the engine's structure.
  const maturity = formatTrendMaturity(structure, header);
  const lastEvent = formatLastStructuralEvent(structure, header);
  const density = formatZoneDensity(structure);

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
          <Badge variant="neutral">Phase : {phaseLabel}</Badge>
          <InfoTooltip termKey="market_phase" iconOnly />
        </span>
      </div>

      {/* État structurel courant : maturité (b) · dernier événement (c) · zones (d) */}
      <div>
        <p className="mb-2 text-xs uppercase tracking-wide text-muted-foreground">
          État structurel courant
        </p>
        <dl className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          <Fact
            label="Maturité de tendance"
            termKey="choch"
            value={maturity}
            className="sm:col-span-2"
          />
          <Fact label="Dernier événement" termKey="choch" value={lastEvent} />
          <Fact label="Zones actives" termKey="order_block" value={density} />
        </dl>
      </div>

      {/* Alignement multi-timeframe + désaccord (e) */}
      <div>
        <p className="mb-2 text-xs uppercase tracking-wide text-muted-foreground">
          <InfoTooltip termKey="mtf">Alignement multi-timeframe</InfoTooltip>
        </p>

        {isLoading && !hasAnyTrend ? (
          <p className="text-xs text-muted-foreground">
            Lecture des timeframes…
          </p>
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
          <p className="text-xs text-muted-foreground">
            Alignement multi-timeframe indisponible.
          </p>
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
                Désaccord multi-timeframe
              </p>
              <p className="text-sm text-foreground">{relation.text}</p>
            </div>
          </div>
        ) : (
          relation.text && <p className="mt-2 text-sm">{relation.text}</p>
        )}
      </div>

      <p className="text-xs italic text-muted-foreground">
        Ces faits décrivent l’état observé du marché en ce moment. Ils ne
        constituent pas une instruction adressée au trader.
      </p>
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
        {available ? value : UNAVAILABLE}
      </dd>
    </div>
  );
}
