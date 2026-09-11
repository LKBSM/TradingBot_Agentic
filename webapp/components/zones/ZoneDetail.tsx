'use client';

import * as React from 'react';
import dynamic from 'next/dynamic';
import Link from 'next/link';
import { useSearchParams } from 'next/navigation';
import { useLocale, useTranslations } from 'next-intl';
import { ChevronRight } from 'lucide-react';
import { cn } from '@/lib/utils';
import type { Candle, LiquidityPool, MarketReadingStructure } from '@/types/market-reading';
import { useLocalizedHref } from '@/lib/i18n/href';
import { useReadingFormatters } from '@/lib/market-reading/use-reading-formatters';
import { resolveComboFromQuery } from '@/lib/conditions/app-link';
import {
  useCandles,
  useLatestPrice,
  useMarketReading,
} from '@/lib/market-reading/hooks';
import {
  DEFAULT_INSTRUMENT,
  DEFAULT_TIMEFRAME,
} from '@/lib/market-reading/perimeter';
import { useSiblingZones } from '@/lib/zones/use-sibling-zones';
import { useChat } from '@/components/chat/ChatProvider';
import {
  collectConsumedZones,
  collectZones,
  fillFraction,
  formatDurationShort,
  formatZoneDateTime,
  lastEntryContact,
  zoneHeaderState,
  type DurationLabels,
  type SiblingZone,
  type ZoneLifecycle,
} from '@/lib/zones/lifecycle';
import { buildConfluence, type ConfluenceFact } from '@/lib/zones/confluence';
import { formationSession } from '@/lib/zones/formation-session';

const ReadingChart = dynamic(
  () => import('@/components/app/ReadingChart').then((m) => ({ default: m.ReadingChart })),
  { ssr: false, loading: () => <div className="h-[300px] sm:h-[360px]" /> },
);

type ZonesT = ReturnType<typeof useTranslations>;
type ReadingFmt = ReturnType<typeof useReadingFormatters>;

/** How many contacts stay visible before the fold (mission §0). */
const VISIBLE_CONTACTS = 3;

/**
 * VZ-4 — the dedicated page of ONE zone (`/zones/<engine id>`), reached from the
 * compact /zones card and from the /app structure list. Same shell (rail +
 * the single M.I.A column), but the content reads as a document: headings and
 * prose, one chart, and deliberately almost no boxes — the sweep note and the
 * chart are the only framed things.
 *
 * Strictly descriptive, like every zone surface. Vocabulary discipline (mission
 * §0): a zone is never « stable / solide / fiable / respectée / validée ». It is
 * entered, exited, traversed, filled — with dates and levels — and the reader
 * concludes. Containment reuses the SAME wording as the card
 * (`zones.confluence.*`), never « chevauche ».
 *
 * Every fact is read off the reading the engine already produced (the same
 * payload /app and /zones use); nothing here recomputes detection and nothing is
 * derived that the data does not carry. In particular the comblement is shown
 * ONCE, as the zone's CURRENT state — never re-attributed to a past contact
 * (VZ-4 §1: `fvgContactFills()` is monotone, so printing it per row repeated the
 * same figure on every line, including edge touches that filled nothing).
 */
export function ZoneDetail({ zoneId }: { zoneId: string }) {
  const t = useTranslations('zones');
  const fmt = useReadingFormatters();
  const locale = useLocale();
  const lh = useLocalizedHref();
  const searchParams = useSearchParams();
  const { openForCombo, setFocus } = useChat();

  const urlCombo = React.useMemo(
    () =>
      resolveComboFromQuery(
        searchParams.get('instrument') ?? undefined,
        searchParams.get('timeframe') ?? undefined,
      ),
    [searchParams],
  );
  const instrument = urlCombo?.instrument ?? DEFAULT_INSTRUMENT;
  const timeframe = urlCombo?.timeframe ?? DEFAULT_TIMEFRAME;

  const { data, isLoading, error } = useMarketReading(instrument, timeframe);
  const { change } = useLatestPrice(instrument, {
    candleCloseTs: data?.header.candle_close_ts ?? null,
  });
  const { candles } = useCandles(instrument, timeframe, {
    candleCloseTs: data?.header.candle_close_ts ?? null,
  });
  const { siblings } = useSiblingZones(instrument, timeframe);

  const liveZones = React.useMemo(() => collectZones(data?.structure), [data]);
  const consumedZones = React.useMemo(() => collectConsumedZones(data?.structure), [data]);
  const zone = React.useMemo(
    () => [...liveZones, ...consumedZones].find((z) => z.id === zoneId) ?? null,
    [liveZones, consumedZones, zoneId],
  );
  const liquidityPools = React.useMemo(
    () => data?.structure.liquidity_pools ?? [],
    [data],
  );
  const referencePrice = change?.price ?? data?.header.close_price ?? null;

  // Bind the page combo so M.I.A reads the right market even before any click.
  React.useEffect(() => {
    openForCombo({ instrument, timeframe });
  }, [openForCombo, instrument, timeframe]);

  // Orient the SINGLE shared M.I.A conversation on this zone (MIA-3): the panel
  // is docked by the shell, we only set its subject. Leaving the page clears the
  // subject — never the conversation (deselect ≠ reset, CLN-1 §3 / mission §7).
  const lastFocusKeyRef = React.useRef<string | null>(null);
  React.useEffect(() => {
    if (!zone) return;
    const label = `${zoneTag(zone)} · ${fmt.band(zone.levelLow, zone.levelHigh, instrument)}`;
    const key = `${zone.id}|${label}`;
    if (lastFocusKeyRef.current === key) return;
    lastFocusKeyRef.current = key;
    setFocus({ kind: 'zone', zoneId: zone.id, label });
    // `fmt` is recreated on every render (unstable) and only read for the label;
    // the key guard makes re-runs no-ops (same pattern as ZonesWorkspace).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [setFocus, zone, instrument]);
  React.useEffect(() => () => setFocus(null), [setFocus]);

  const backHref = lh(
    `/zones?instrument=${encodeURIComponent(instrument)}&timeframe=${encodeURIComponent(timeframe)}`,
  );

  if (isLoading && !data) {
    return (
      <div className="zdt" data-testid="zone-detail-loading">
        <p className="zdt-muted">{t('detail.loading')}</p>
      </div>
    );
  }

  // Unknown / no longer detected id — say so plainly instead of rendering an
  // empty document or, worse, a zone picked by resemblance (mission §4 id lock).
  if (!zone) {
    return (
      <div className="zdt" data-testid="zone-detail-missing">
        <Crumb backHref={backHref} instrument={instrument} timeframe={timeframe} t={t} fmt={fmt} />
        <h1 className="zdt-h1">{t('detail.missing.title')}</h1>
        <p className="zdt-lead">
          {error ? t('detail.missing.error') : t('detail.missing.body')}
        </p>
        <p>
          <Link className="zdt-back" href={backHref}>
            {t('detail.missing.back')}
          </Link>
        </p>
      </div>
    );
  }

  return (
    <ZoneDetailBody
      zone={zone}
      instrument={instrument}
      timeframe={timeframe}
      referencePrice={referencePrice}
      candles={candles}
      structure={data?.structure ?? null}
      liveZones={liveZones}
      siblings={siblings}
      liquidityPools={liquidityPools}
      backHref={backHref}
      locale={locale}
      t={t}
      fmt={fmt}
    />
  );
}

function zoneTag(zone: ZoneLifecycle): string {
  const kind = zone.kind === 'ob' ? 'OB' : 'FVG';
  const arrow = zone.direction === 'bullish' ? ' ↑' : zone.direction === 'bearish' ? ' ↓' : '';
  return `${kind}${arrow}`;
}

function Crumb({
  backHref,
  instrument,
  timeframe,
  t,
  fmt,
}: {
  backHref: string;
  instrument: string;
  timeframe: string;
  t: ZonesT;
  fmt: ReadingFmt;
}) {
  return (
    <nav className="zdt-crumb" aria-label={t('detail.crumbAria')}>
      <Link href={backHref}>{t('detail.crumbRoot')}</Link>
      <ChevronRight className="h-3 w-3 shrink-0 opacity-60" aria-hidden />
      <span>
        {fmt.instrument(instrument)} · {fmt.timeframe(timeframe)}
      </span>
    </nav>
  );
}

function ZoneDetailBody({
  zone,
  instrument,
  timeframe,
  referencePrice,
  candles,
  structure,
  liveZones,
  siblings,
  liquidityPools,
  backHref,
  locale,
  t,
  fmt,
}: {
  zone: ZoneLifecycle;
  instrument: string;
  timeframe: string;
  referencePrice: number | null;
  candles: Candle[] | null;
  structure: MarketReadingStructure | null;
  liveZones: readonly ZoneLifecycle[];
  siblings: readonly SiblingZone[];
  liquidityPools: readonly LiquidityPool[];
  backHref: string;
  locale: string;
  t: ZonesT;
  fmt: ReadingFmt;
}) {
  const [showAll, setShowAll] = React.useState(false);

  const facts = React.useMemo(
    () => buildConfluence(zone, liveZones, siblings, liquidityPools),
    [zone, liveZones, siblings, liquidityPools],
  );
  // « Zones à l'intérieur » — ONLY real containment facts. The engine exposes no
  // containment field; these come from `buildConfluence`, pure interval geometry
  // over the engine's own bounds (VZ-1, already shipped on the card). When there
  // is none, the section is simply absent — no placeholder, no filler sentence.
  const innerFacts = React.useMemo(() => facts.filter((f) => f.relation === 'inner'), [facts]);
  // The sweep note: a resting-liquidity pocket sitting INSIDE the band. Same
  // source, same wording as the card.
  const insidePools = React.useMemo(
    () => facts.filter((f) => f.relation === 'liquidity' && f.distanceSide === 'inside'),
    [facts],
  );

  const badge = zoneHeaderState(zone);
  const badgeLabel =
    badge.key === 'contacts'
      ? t('header.contacts', { count: badge.count })
      : t(`header.${badge.key}`);

  const height = zone.levelHigh - zone.levelLow;
  const heightLabel =
    referencePrice != null && referencePrice > 0
      ? t('card.height', {
          pts: fmt.points(height, instrument),
          pct: fmt.pctShort(height / referencePrice),
        })
      : t('card.heightPtsOnly', { pts: fmt.points(height, instrument) });

  // État actuel — the comblement appears HERE and only here (VZ-4 §1).
  const fill = fillFraction(zone);
  const durationLabels: DurationLabels = {
    underMinute: t('duration.underMinute'),
    min: t('duration.min'),
    hour: t('duration.hour'),
    day: t('duration.day'),
  };
  const last = lastEntryContact(zone);
  const lastMs = last ? Date.parse(last.at) : NaN;
  const lastAgo = Number.isNaN(lastMs)
    ? null
    : formatDurationShort(Math.max(Date.now() - lastMs, 0), durationLabels);

  // Ledger, most recent first — 3 visible, the rest behind an explicit count.
  const ledger = React.useMemo(
    () => zone.contacts.filter((c) => c.outcome !== 'inside').slice().reverse(),
    [zone.contacts],
  );
  const visible = showAll ? ledger : ledger.slice(0, VISIBLE_CONTACTS);
  const hiddenCount = Math.max(ledger.length - VISIBLE_CONTACTS, 0);

  const session = zone.session ?? formationSession(zone.createdAt, instrument);
  let originLine: string | null = null;
  if (zone.kind === 'ob' && zone.origin) {
    originLine = t('origin.ob', {
      kind: zone.origin.kind === 'choch' ? 'CHOCH' : 'BOS',
      dir: fmt.direction(zone.origin.direction),
      time: formatZoneDateTime(zone.origin.at, locale),
      level: fmt.price(zone.origin.level, instrument),
    });
  } else if (zone.kind === 'fvg') {
    originLine = t('origin.fvg');
  }

  return (
    <div className="zdt" data-testid="zone-detail" data-zone-id={zone.id}>
      <Crumb backHref={backHref} instrument={instrument} timeframe={timeframe} t={t} fmt={fmt} />

      {/* En-tête — type, direction, bornes. */}
      <div className="zdt-head">
        <h1 className="zdt-h1">{t(`kind.${zone.kind}`)}</h1>
        {zone.direction && (
          <span className={cn('zdt-dir', zone.direction === 'bullish' ? 'bull' : 'bear')}>
            {fmt.direction(zone.direction)}
          </span>
        )}
        <span className="zdt-state">{badgeLabel}</span>
      </div>
      <p className="zdt-range">
        {t('detail.range', {
          high: fmt.price(zone.levelHigh, instrument),
          low: fmt.price(zone.levelLow, instrument),
          mid: fmt.price((zone.levelHigh + zone.levelLow) / 2, instrument),
        })}
        <span className="zdt-faint"> · {heightLabel}</span>
      </p>

      {/* Aperçu — the SAME chart component as « Lire une structure » (/app), with
          this zone isolated so the page shows one visual, not a second product. */}
      {candles && candles.length > 0 && structure && (
        <figure className="zdt-chart" data-testid="zone-detail-chart">
          <ReadingChart
            candles={candles}
            structure={structure}
            instrument={instrument}
            timeframe={timeframe}
            livePrice={null}
            isolatedZoneIds={[zone.id]}
            highlightZoneId={zone.id}
            selection={{ family: 'zone', id: zone.id }}
            heightClassName="h-[300px] sm:h-[360px]"
          />
          <figcaption className="zdt-cap">
            {t('detail.chartCaption', {
              instrument: fmt.instrument(instrument),
              tf: fmt.timeframe(timeframe),
            })}
          </figcaption>
        </figure>
      )}

      {/* Note de sweep — the one real box on the page, because it is a real
          notice: liquidity resting INSIDE the band. Absent when there is none. */}
      {insidePools.length > 0 && (
        <aside className="zdt-sweep" data-testid="zone-detail-sweep">
          {insidePools.map((f, i) => (
            <p key={i}>{liquidityInsideLine(f, t, fmt, instrument)}</p>
          ))}
        </aside>
      )}

      <h2 className="zdt-h2">{t('detail.state.heading')}</h2>
      {fill != null && (
        <div className="zdt-fill" data-testid="zone-detail-fill">
          <div
            className="zdt-fillbar"
            role="progressbar"
            aria-valuenow={Math.round(fill * 100)}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-label={t('fill.aria')}
          >
            <span style={{ width: `${Math.round(fill * 100)}%` }} />
          </div>
          <span className="zdt-fillval">{fmt.pctShort(fill)}</span>
        </div>
      )}
      <p className="zdt-muted">
        {lastAgo ? t('detail.state.lastContact', { ago: lastAgo }) : t('detail.state.noContact')}
      </p>

      {/* Zones à l'intérieur — prose, and ONLY when the containment facts exist. */}
      {innerFacts.length > 0 && (
        <section data-testid="zone-detail-nested">
          <h2 className="zdt-h2">{t('detail.nested.heading')}</h2>
          {innerFacts.map((f, i) => (
            <p className="zdt-nested" key={i}>
              {innerLine(f, t, fmt, instrument)}
            </p>
          ))}
        </section>
      )}

      {ledger.length > 0 && (
        <section>
          <h2 className="zdt-h2">{t('detail.contacts.heading')}</h2>
          {visible.map((c, i) => (
            <p className="zdt-row" key={i}>
              <time className="zdt-when">{formatZoneDateTime(c.at, locale)}</time>
              <span>{contactText(c, t, fmt, instrument)}</span>
            </p>
          ))}
          {hiddenCount > 0 && !showAll && (
            <button type="button" className="zdt-more" onClick={() => setShowAll(true)}>
              {t('detail.contacts.more', { count: hiddenCount })}
            </button>
          )}
          {showAll && hiddenCount > 0 && (
            <button type="button" className="zdt-more" onClick={() => setShowAll(false)}>
              {t('detail.contacts.less')}
            </button>
          )}
        </section>
      )}

      {originLine && (
        <section>
          <h2 className="zdt-h2">{t('origin.heading')}</h2>
          <p>
            {originLine}
            {session && ` ${t('origin.session', { session: t(`session.${session}`) })}`}
          </p>
        </section>
      )}

      <hr className="zdt-rule" />
      <p className="zdt-legal">{t('contacts.honesty')}</p>
    </div>
  );
}

function contactText(
  c: ZoneLifecycle['contacts'][number],
  t: ZonesT,
  fmt: ReadingFmt,
  instrument: string,
): string {
  const level = fmt.price(c.level, instrument);
  if (c.outcome === 'edge_touch') return t('contacts.edgeTouch');
  if (c.outcome === 'traversal') return t('contacts.traversal', { level });
  return t('contacts.entryExit', { level });
}

/** Reuses the card's containment sentence verbatim — one wording, one meaning. */
function innerLine(f: ConfluenceFact, t: ZonesT, fmt: ReadingFmt, instrument: string): string {
  const kd = `${t(`kind.${f.kind}`)}${
    f.direction ? ` ${fmt.direction(f.direction).toLowerCase()}` : ''
  }`;
  const band = fmt.band(f.levelLow as number, f.levelHigh as number, instrument);
  const line =
    f.timeframe == null
      ? t('confluence.innerSame', { kd, band })
      : t('confluence.innerTf', { kd, tf: f.timeframe, band });
  // The neighbour's REAL engine status — never a fabricated percentage. The
  // containment fact carries no fill figure, so we state the status the engine
  // published and stop there.
  const statusLabel =
    f.kind === 'fvg' ? fmt.fvgStatus(f.status as never) : fmt.obStatus(f.status as never);
  return statusLabel
    ? `${line} ${t('detail.nested.status', { status: statusLabel.toLowerCase() })}`
    : line;
}

function liquidityInsideLine(
  f: ConfluenceFact,
  t: ZonesT,
  fmt: ReadingFmt,
  instrument: string,
): string {
  const side = t(`liquiditySide.${f.liquiditySide}`);
  const status = fmt.liquidityStatus(f.liquidityStatus as never).label.toLowerCase();
  const level = fmt.price(f.level as number, instrument);
  return t('confluence.liquidityInside', { side, status, level });
}
