'use client';

import * as React from 'react';
import Link from 'next/link';
import { useTranslations } from 'next-intl';
import {
  ChevronLeft,
  ArrowUpRight,
  BookText,
  FileText,
  LineChart,
  CalendarDays,
} from 'lucide-react';
import { AgentAvatar } from '@/components/chat/AgentAvatar';
import { useLocalizedHref } from '@/lib/i18n/href';
import { sourceLinksFor, type SourceDocKind } from '@/lib/calendar/sourceLinks';
import { useCalendarEvent, usePublicationMeasures } from '@/lib/calendar/useCalendar';
import { countdown, hmInZone } from '@/lib/calendar/grouping';
import { parseUtc, utcOffsetLabel, formatLocalDayLong } from '@/lib/time/localTime';
import type { CalendarEvent, CalendarResponse } from '@/types/calendar';
import {
  hasAnyMeasure,
  type PublicationMeasures,
  type CalmBeforeMeasure,
  type StructureStateMeasure,
  type ReturnToCalmMeasure,
  type MeasureExtreme,
  type MeasureProvenance,
} from '@/types/measures';
import './calendar.css';
import './calendar-pub.css';

/** Render a published value AS PUBLISHED — no conversion, no re-rounding. */
function asPublished(n: number | null): string {
  return n == null ? '—' : String(n);
}

/**
 * Format a WHOLE-MINUTES duration as hours/minutes — NEVER candles/"bougies".
 * m<60 → "42 min" ; else "1 h" / "2 h 05".
 */
function formatMinutes(m: number): string {
  if (m < 60) return `${m} min`;
  const h = Math.floor(m / 60);
  const rem = m % 60;
  return rem ? `${h} h ${String(rem).padStart(2, '0')}` : `${h} h`;
}

function tzCity(iana: string | null): string | null {
  if (!iana) return null;
  const seg = iana.split('/').pop() ?? iana;
  return seg.replace(/_/g, ' ');
}

/** "2026-06" → localized short month ("juin", "Jun") for the curve x-axis. */
function periodShort(period: string, locale: string): string {
  const m = /^(\d{4})-(\d{2})$/.exec(period);
  if (!m) return period;
  const d = new Date(Date.UTC(Number(m[1]), Number(m[2]) - 1, 1));
  return new Intl.DateTimeFormat(locale, { month: 'short', timeZone: 'UTC' }).format(d);
}

/** "2026-06" → localized "juin 2026" for the "last value" line under the curve. */
function periodLong(period: string, locale: string): string {
  const m = /^(\d{4})-(\d{2})$/.exec(period);
  if (!m) return period;
  const d = new Date(Date.UTC(Number(m[1]), Number(m[2]) - 1, 1));
  return new Intl.DateTimeFormat(locale, {
    month: 'long',
    year: 'numeric',
    timeZone: 'UTC',
  }).format(d);
}

/** Bold-highlight callback shared by every t.rich() call. */
const RICH = { b: (chunks: React.ReactNode) => <b>{chunks}</b> } as const;

/** MIDDLE segment of `source:event_key:date` (e.g. "bls:us_cpi:…" → "us_cpi"). */
function deriveEventKey(eventId: string): string | null {
  const parts = eventId.split(':');
  return parts.length >= 2 ? parts[1] ?? null : null;
}

function cap(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

/**
 * Publications with a REAL, hand-written pedagogy fiche. A key absent here — a bad
 * URL, or a future catalog entry not yet written — renders NO pedagogy block at
 * all, never a generic filler. Kept in sync with the `calendar.pub.pedagogy.<key>`
 * messages (a guard test asserts every key here has a fr+en body, and vice-versa).
 */
const PEDAGOGY_FICHES: ReadonlySet<string> = new Set([
  'us_employment_situation', 'us_cpi', 'us_cpi_core', 'us_ppi', 'us_jolts',
  'us_gdp', 'us_pce', 'us_retail_sales', 'us_housing_starts', 'us_durable_goods',
  'us_fomc_rate', 'us_fomc_minutes', 'us_fomc_dotplot',
  'ea_hicp_flash', 'ea_gdp_flash', 'ea_unemployment', 'ea_ecb_rate',
]);

/**
 * Per-publication detail page (NW-3). Honest and layered: the published value
 * series (curve, upcoming point left blank), the four things the engine measured
 * at past releases (never a forecast), a MIA prompt, a link to the issuing
 * organism, and a pedagogy card. Every visible sentence is composed from static
 * i18n templates — nothing is generated. It announces a MOMENT, never a direction.
 */
export function CalendarEventDetail({
  eventId,
  locale,
  data: injectedData,
  now: injectedNow,
  measures: injectedMeasures,
}: {
  eventId: string;
  locale: string;
  data?: CalendarResponse | null;
  now?: Date;
  measures?: PublicationMeasures | null;
}) {
  const t = useTranslations('calendar');
  const lh = useLocalizedHref();
  // REC point 1: load the event by its STABLE ID from storage, independent of any
  // window. A deep-linked event exists by definition; "introuvable" now means the
  // id genuinely does not exist (bad manual URL / deleted), never "out of window".
  const hook = useCalendarEvent(eventId);
  const data = injectedData !== undefined ? injectedData : hook.data;
  const isLoading = injectedData !== undefined ? false : hook.isLoading;
  const error = injectedData !== undefined ? null : hook.error;
  const now = React.useMemo(() => injectedNow ?? new Date(), [injectedNow]);

  const eventKey = deriveEventKey(eventId);
  // When measures are injected (tests), use them; otherwise fetch by event key.
  const measuresHook = usePublicationMeasures(
    injectedMeasures !== undefined ? null : eventKey,
  );
  const measures =
    injectedMeasures !== undefined ? injectedMeasures : measuresHook.data;

  // The by-id endpoint returns exactly the matching event (or none). Match the
  // full id, or the bare ref — the App news module deep-links with a raw ref.
  const ev =
    data?.events.find(
      (e) => e.event_id === eventId || e.event_id.split(':').pop() === eventId,
    ) ?? null;

  const attribution = ev
    ? data?.attribution.find((a) => a.source === ev.source) ?? null
    : null;

  return (
    <div className="cal-page cald">
      <Link className="cald-back" href={lh('/actualites')}>
        <ChevronLeft width={14} height={14} aria-hidden />
        {t('detail.back')}
      </Link>

      {isLoading ? (
        <div className="cal-status">{t('loading')}</div>
      ) : error ? (
        <div className="cal-status">{t('error')}</div>
      ) : !ev ? (
        <div className="cal-empty">
          <strong>{t('detail.notFoundTitle')}</strong>
          <br />
          {t('detail.notFound')}
        </div>
      ) : (
        <Detail
          ev={ev}
          eventKey={deriveEventKey(ev.event_id) ?? eventKey}
          attribution={attribution}
          measures={measures}
          now={now}
          locale={locale}
          t={t}
        />
      )}
    </div>
  );
}

function fmtCountdown(
  cd: ReturnType<typeof countdown>,
  t: ReturnType<typeof useTranslations>,
): string {
  const { past, days, hours, minutes } = cd;
  const mm = String(minutes).padStart(2, '0');
  if (past) {
    if (days >= 1) return t('countdown.agoDays', { days });
    if (hours >= 1) return t('countdown.agoHours', { hours, minutes: mm });
    return t('countdown.agoMinutes', { minutes });
  }
  if (days >= 1) return t('countdown.days', { days });
  if (hours >= 1) return t('countdown.hours', { hours, minutes: mm });
  if (minutes >= 1) return t('countdown.minutes', { minutes });
  return t('countdown.now');
}

/* --------------------------------------------------------------------------
 * Section 2 — CURVE (inline SVG line chart of the published value series)
 * ------------------------------------------------------------------------ */

function CurveCard({
  ev,
  locale,
  t,
}: {
  ev: CalendarEvent;
  locale: string;
  t: ReturnType<typeof useTranslations>;
}) {
  const pts = ev.value_series ?? []; // oldest → newest
  const values = pts.map((p) => p.value);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1; // avoid /0 on a flat series

  // Geometry — evenly-spaced x for the real points plus ONE trailing upcoming.
  const W = 760;
  const H = 156;
  const padX = 34;
  const padTop = 20;
  const padBottom = 32;
  const plotW = W - padX * 2;
  const plotH = H - padTop - padBottom;
  const total = pts.length + 1; // + upcoming slot
  const stepX = total > 1 ? plotW / (total - 1) : 0;
  const xAt = (i: number) => padX + stepX * i;
  const yAt = (v: number) => padTop + plotH * (1 - (v - min) / span);

  const linePath = pts
    .map((p, i) => `${i === 0 ? 'M' : 'L'}${xAt(i).toFixed(1)},${yAt(p.value).toFixed(1)}`)
    .join(' ');

  const lastReal = pts[pts.length - 1];
  const upcomingIdx = pts.length; // slot after the newest real point
  const upcomingX = xAt(upcomingIdx);
  const upcomingY = lastReal ? yAt(lastReal.value) : yAt((min + max) / 2);

  // Three horizontal guides (low / mid / high) with the value AS PUBLISHED.
  const guides = [min, (min + max) / 2, max];

  const revisedDate = ev.revised_at ? parseUtc(ev.revised_at) : null;
  const revisedDateLabel = revisedDate ? revisedDate.toLocaleDateString(locale) : '—';
  const lastAttempt = ev.refreshed_at ? parseUtc(ev.refreshed_at) : null;
  const lastAttemptLabel = lastAttempt ? lastAttempt.toLocaleDateString(locale) : '—';

  return (
    <div className="cald-card">
      <div className="cald-card-h">
        <h3>{t('pub.curve.title')}</h3>
        <span className="cald-badge">{t('pub.curve.badge')}</span>
      </div>

      <div className="pub-curve">
        <svg
          className="pub-curve-svg"
          viewBox={`0 0 ${W} ${H}`}
          role="img"
          aria-label={t('pub.curve.title')}
        >
          {/* Horizontal guides + value labels (as published, no re-rounding) */}
          {guides.map((v, i) => (
            <g key={`g${i}`}>
              <line
                className="grid"
                x1={padX}
                y1={yAt(v).toFixed(1)}
                x2={W - padX}
                y2={yAt(v).toFixed(1)}
              />
              <text className="grid-val" x={padX - 6} y={(yAt(v) + 3).toFixed(1)} textAnchor="end">
                {Number(v.toFixed(2))}
              </text>
            </g>
          ))}
          {/* Dashed connector from the last real point to the hollow upcoming one */}
          {lastReal && (
            <path
              className="line-upcoming"
              d={`M${xAt(pts.length - 1).toFixed(1)},${yAt(lastReal.value).toFixed(1)} L${upcomingX.toFixed(1)},${upcomingY.toFixed(1)}`}
            />
          )}
          <path className="line" d={linePath} />
          {pts.map((p, i) => (
            <g key={`${p.period}-${i}`}>
              <circle
                className={i === pts.length - 1 ? 'dot dot-last' : 'dot'}
                cx={xAt(i)}
                cy={yAt(p.value)}
                r={i === pts.length - 1 ? 4 : 3}
              />
              {i === pts.length - 1 && (
                <text className="pt-val" x={xAt(i)} y={yAt(p.value) - 9} textAnchor="middle">
                  {p.value}
                </text>
              )}
              <text className="pt-label" x={xAt(i)} y={H - 12} textAnchor="middle">
                {periodShort(p.period, locale)}
              </text>
            </g>
          ))}
          {/* The upcoming point: hollow, dashed, NEVER shows a number. */}
          <circle
            className="dot-upcoming"
            cx={upcomingX}
            cy={upcomingY}
            r={4}
            data-upcoming="1"
          />
          <text
            className="pt-upcoming-label"
            x={upcomingX}
            y={upcomingY - 10}
            textAnchor="middle"
          >
            {t('pub.curve.upcoming')}
          </text>
          <text className="pt-label" x={upcomingX} y={H - 12} textAnchor="middle">
            ?
          </text>
        </svg>
      </div>

      {/* Stats row — last value + its period, and the range observed over the series. */}
      {lastReal && (
        <div className="pub-curve-stats">
          <div>
            {t.rich('pub.curve.lastValue', {
              ...RICH,
              value: lastReal.value,
              period: periodLong(lastReal.period, locale),
            })}
          </div>
          <div>
            {t.rich('pub.curve.range', {
              ...RICH,
              min: Number(min.toFixed(2)),
              max: Number(max.toFixed(2)),
            })}
          </div>
        </div>
      )}

      <p className="cald-note">{t('pub.curve.note')}</p>

      {/* Current-release state — keep the three DISTINCT absence states visible. */}
      <div className="cald-rev">
        {ev.actual_state === 'pending' && (
          <p className="cald-rev-line">{t('detail.actualPending')}</p>
        )}
        {ev.actual_state === 'unfetched' && (
          <p className="cald-rev-line">
            {t('detail.actualUnfetched', { date: lastAttemptLabel })}
          </p>
        )}
        {ev.actual_state === 'unavailable' && (
          <p className="cald-rev-line">
            {t('detail.actualUnavailable', { organism: ev.organism ?? '—' })}
          </p>
        )}

        {/* Revision line — initial + current + date, or an explicit "not revised". */}
        {ev.revised ? (
          <p className="cald-rev-line">
            {t('pub.curve.revisedFromTo', {
              initial: asPublished(ev.actual_initial),
              current: asPublished(ev.actual),
              date: revisedDateLabel,
            })}
          </p>
        ) : ev.actual != null ? (
          <p className="cald-rev-line">{t('pub.curve.notRevised')}</p>
        ) : null}
      </div>

      <p className="pub-source-line">
        {t('pub.curve.source', { organism: ev.organism ?? '—' })}
      </p>
    </div>
  );
}

/* --------------------------------------------------------------------------
 * Section 3 — FOUR QUESTIONS (only the measures actually computed)
 * ------------------------------------------------------------------------ */

function fmtProvDate(iso: string, locale: string): string {
  return parseUtc(iso)?.toLocaleDateString(locale) ?? '—';
}

function fmtExtremeDate(x: MeasureExtreme, locale: string): string {
  return parseUtc(x.observed_at)?.toLocaleDateString(locale) ?? '—';
}

function SourceLine({
  prov,
  msgKey,
  locale,
  t,
}: {
  prov: MeasureProvenance;
  msgKey: string;
  locale: string;
  t: ReturnType<typeof useTranslations>;
}) {
  return (
    <p className="pub-qsource">
      {t(msgKey, {
        sample: prov.sample_size,
        market: t(`market.${prov.market}` as 'market.XAUUSD'),
        periodStart: fmtProvDate(prov.period_start, locale),
        periodEnd: fmtProvDate(prov.period_end, locale),
        referenceDays: prov.reference_days ?? '—',
      })}
    </p>
  );
}

function CalmBeforeCard({
  n,
  m,
  locale,
  t,
}: {
  n: number;
  m: CalmBeforeMeasure;
  locale: string;
  t: ReturnType<typeof useTranslations>;
}) {
  const market = t(`market.${m.provenance.market}` as 'market.XAUUSD');
  return (
    <div className="pub-qcard">
      <div className="pub-qn">{t('pub.questions.qLabel', { n })}</div>
      <p className="pub-q">{t('pub.questions.calmBefore.q')}</p>
      <p className="pub-a">
        {t.rich('pub.questions.calmBefore.a', {
          ...RICH,
          sample: m.provenance.sample_size,
          market,
          calmer: m.calmer_count,
          busier: m.busier_count,
        })}
      </p>
      <p className="pub-detail">
        {t.rich('pub.questions.calmBefore.detail', {
          ...RICH,
          reference: m.reference_amount,
          unit: m.provenance.quote_unit ?? '—',
          calmest: asPublished(m.calmest.amount),
          calmestDate: fmtExtremeDate(m.calmest, locale),
          busiest: asPublished(m.busiest.amount),
          busiestDate: fmtExtremeDate(m.busiest, locale),
        })}
      </p>
      <SourceLine
        prov={m.provenance}
        msgKey="pub.questions.calmBefore.source"
        locale={locale}
        t={t}
      />
    </div>
  );
}

function StructureCard({
  n,
  m,
  locale,
  t,
}: {
  n: number;
  m: StructureStateMeasure;
  locale: string;
  t: ReturnType<typeof useTranslations>;
}) {
  const market = t(`market.${m.provenance.market}` as 'market.XAUUSD');
  return (
    <div className="pub-qcard">
      <div className="pub-qn">{t('pub.questions.qLabel', { n })}</div>
      <p className="pub-q">{t('pub.questions.structure.q')}</p>
      <p className="pub-a">
        {t.rich('pub.questions.structure.a', {
          ...RICH,
          sample: m.provenance.sample_size,
          market,
          insideZone: m.inside_zone_count,
          intactPocket: m.intact_pocket_count,
        })}
      </p>
      <p className="pub-detail">
        {t.rich('pub.questions.structure.detail', {
          ...RICH,
          lower: m.range_lower_count,
          middle: m.range_middle_count,
          upper: m.range_upper_count,
        })}
      </p>
      {m.now_range_position != null && (
        <p className="pub-now">
          {t('pub.questions.structure.now', {
            nowInside: t(
              m.now_inside_zone
                ? 'pub.questions.structure.nowInsideYes'
                : 'pub.questions.structure.nowInsideNo',
            ),
            nowPocket: t(
              m.now_intact_pocket_within
                ? 'pub.questions.structure.nowPocketYes'
                : 'pub.questions.structure.nowPocketNo',
            ),
            nowRange: t(
              `pub.questions.structure.range${cap(m.now_range_position)}` as
                'pub.questions.structure.rangeLower',
            ),
          })}
        </p>
      )}
      <SourceLine
        prov={m.provenance}
        msgKey="pub.questions.structure.source"
        locale={locale}
        t={t}
      />
    </div>
  );
}

function ReturnToCalmCard({
  n,
  m,
  locale,
  t,
}: {
  n: number;
  m: ReturnToCalmMeasure;
  locale: string;
  t: ReturnType<typeof useTranslations>;
}) {
  const market = t(`market.${m.provenance.market}` as 'market.XAUUSD');
  const t0 = m.tranches[0]?.count ?? 0;
  const t1 = m.tranches[1]?.count ?? 0;
  const t2 = m.tranches[2]?.count ?? 0;
  return (
    <div className="pub-qcard">
      <div className="pub-qn">{t('pub.questions.qLabel', { n })}</div>
      <p className="pub-q">{t('pub.questions.returnToCalm.q')}</p>
      <p className="pub-a">
        {t.rich('pub.questions.returnToCalm.a', {
          ...RICH,
          sample: m.provenance.sample_size,
          market,
          t0,
          t1,
          t2,
        })}
      </p>
      <p className="pub-detail">
        {t.rich('pub.questions.returnToCalm.detail', {
          ...RICH,
          fastest: m.fastest.minutes != null ? formatMinutes(m.fastest.minutes) : '—',
          fastestDate: fmtExtremeDate(m.fastest, locale),
          slowest: m.slowest.minutes != null ? formatMinutes(m.slowest.minutes) : '—',
          slowestDate: fmtExtremeDate(m.slowest, locale),
        })}
      </p>
      {m.never_settled_count > 0 && (
        <p className="pub-never">
          {t('pub.questions.returnToCalm.neverSettled', {
            count: m.never_settled_count,
          })}
        </p>
      )}
      <SourceLine
        prov={m.provenance}
        msgKey="pub.questions.returnToCalm.source"
        locale={locale}
        t={t}
      />
    </div>
  );
}

function QuestionsSection({
  measures,
  locale,
  t,
}: {
  measures: PublicationMeasures;
  locale: string;
  t: ReturnType<typeof useTranslations>;
}) {
  const market = t(`market.${measures.market}` as 'market.XAUUSD');
  // Number the cards in render order — only the measures that exist are shown,
  // so a deferred measure (zone lifecycle, #3) leaves no gap and no empty card.
  let n = 0;
  return (
    <div className="cald-card pub-qsection">
      <div className="cald-card-h">
        <h3>{t('pub.questions.sectionTitle')}</h3>
        <span className="cald-badge">
          {t('pub.questions.sectionBadge', { market })}
        </span>
      </div>
      <div className="pub-qgrid">
        {measures.calm_before && (
          <CalmBeforeCard n={++n} m={measures.calm_before} locale={locale} t={t} />
        )}
        {measures.structure_state && (
          <StructureCard n={++n} m={measures.structure_state} locale={locale} t={t} />
        )}
        {measures.return_to_calm && (
          <ReturnToCalmCard n={++n} m={measures.return_to_calm} locale={locale} t={t} />
        )}
      </div>
      {/* Common reading guide under the cards — decounts, not probabilities. */}
      <div className="pub-qwarn" role="note">
        <div className="t">{t('pub.questions.readGuide.title')}</div>
        <div className="b">{t('pub.questions.readGuide.body')}</div>
      </div>
    </div>
  );
}

/* --------------------------------------------------------------------------
 * Section 4 — MIA (presentational only)
 * ------------------------------------------------------------------------ */

function MiaBlock({
  pedKey,
  t,
}: {
  pedKey: string;
  t: ReturnType<typeof useTranslations>;
}) {
  // Suggested questions adapted to the publication (bespoke for us_cpi /
  // ea_hicp_flash, a concept-only default otherwise). next-intl rejects arrays,
  // so the set is an object; we keep only the keys present for this publication.
  const suggests = t.raw(`pub.mia.suggests.${pedKey}`) as Record<string, string>;
  const chips = Object.values(suggests);
  return (
    <div className="cald-card pub-mia">
      <div className="pub-mia-head">
        {/* SAME shared avatar and candlestick icon as the /app M.I.A Agent. */}
        <AgentAvatar size="md" presence />
        <div className="pub-mia-id">
          <div className="pub-mia-name">{t('pub.mia.title')}</div>
          <div className="pub-mia-sub">{t('pub.mia.subtitle')}</div>
        </div>
      </div>
      <div className="pub-mia-suggests-label">{t('pub.mia.suggestsLabel')}</div>
      <div className="pub-mia-suggests">
        {chips.map((label, i) => (
          <button key={i} type="button" className="pub-mia-chip">
            <span>{label}</span>
            <ArrowUpRight className="pub-mia-chip-ar" width={13} height={13} aria-hidden />
          </button>
        ))}
      </div>
      <p className="pub-mia-cap">{t('pub.mia.capability')}</p>
      <div className="pub-mia-form">
        <input
          className="pub-mia-input"
          type="text"
          placeholder={t('pub.mia.placeholder')}
          aria-label={t('pub.mia.placeholder')}
        />
        <button type="button" className="pub-mia-send">
          {t('pub.mia.send')}
        </button>
      </div>
    </div>
  );
}

/* --------------------------------------------------------------------------
 * Section 5 — GO TO SOURCE (up to four NAMED links, issuing organism only)
 * ------------------------------------------------------------------------ */

const DOC_ICON: Record<SourceDocKind, React.ComponentType<{ className?: string }>> = {
  methodology: BookText,
  release: FileText,
  series: LineChart,
  schedule: CalendarDays,
};

function SourceSection({
  eventKey,
  attribution,
  t,
}: {
  eventKey: string | null;
  attribution: { source: string; organism: string; license_label: string; policy_url: string } | null;
  t: ReturnType<typeof useTranslations>;
}) {
  const links = sourceLinksFor(eventKey);
  return (
    <div className="cald-card">
      <div className="cald-card-h">
        <h3>{t('pub.source.title')}</h3>
      </div>
      <p className="pub-src-intro">{t('pub.source.intro')}</p>

      {links.length > 0 ? (
        <div className="pub-src-grid">
          {links.map(({ kind, url }) => {
            const Icon = DOC_ICON[kind];
            return (
              <a
                key={kind}
                className="pub-src-doc"
                href={url}
                target="_blank"
                rel="noreferrer noopener"
              >
                <span className="pub-src-ic">
                  <Icon className="pub-src-ic-svg" />
                </span>
                <span className="pub-src-txt">
                  <span className="pub-src-name">{t(`pub.source.docs.${kind}.label`)}</span>
                  <span className="pub-src-desc">{t(`pub.source.docs.${kind}.desc`)}</span>
                </span>
                <ArrowUpRight className="pub-src-ext" width={14} height={14} aria-hidden />
              </a>
            );
          })}
        </div>
      ) : (
        // No precise per-document URLs known for this organism yet: the generic
        // policy link is NOT a substitute — we show the honest fallback line.
        <p className="pub-src-none">{t('pub.source.noneYet')}</p>
      )}

      {attribution && (
        <div className="pub-src-foot">
          <span className="pub-src-only">{t('pub.source.organismOnly')}</span>
          <span className="pub-src-note">{t('pub.source.onlyNote')}</span>
        </div>
      )}
      {attribution && (
        <p className="pub-src-license">
          {attribution.organism} · {attribution.license_label}
        </p>
      )}
    </div>
  );
}

/* --------------------------------------------------------------------------
 * Detail — page composition (mission §2B order)
 * ------------------------------------------------------------------------ */

function Detail({
  ev,
  eventKey,
  attribution,
  measures,
  now,
  locale,
  t,
}: {
  ev: CalendarEvent;
  eventKey: string | null;
  attribution: { source: string; organism: string; license_label: string; policy_url: string } | null;
  measures: PublicationMeasures | null;
  now: Date;
  locale: string;
  t: ReturnType<typeof useTranslations>;
}) {
  const when = parseUtc(ev.scheduled_at);
  const cd = when ? countdown(now.getTime(), when.getTime()) : null;
  const city = tzCity(ev.source_timezone);
  const marketTime = when ? hmInZone(when, ev.source_timezone ?? undefined) : '—';
  const localTime = when ? hmInZone(when) : '—';
  const dayLabel = when ? formatLocalDayLong(when, locale) : '—';
  const affects = ev.markets.map((m) => t(`market.${m}` as 'market.XAUUSD')).join(', ');

  const nonoItems = Object.values(
    t.raw('detail.nono.items') as Record<string, string>,
  );

  // A pedagogy fiche is shown ONLY when a real, hand-written one exists for this
  // publication — no generic filler (Défaut B). The M.I.A suggested questions use
  // their OWN key: bespoke for CPI/HICP, a concept-only default otherwise.
  const hasFiche = eventKey != null && PEDAGOGY_FICHES.has(eventKey);
  const miaKey =
    eventKey === 'us_cpi' || eventKey === 'ea_hicp_flash' ? eventKey : 'default';

  const showQuestions = hasAnyMeasure(measures);

  return (
    <>
      {/* 1 — HEADER */}
      <div className="cald-head">
        <div className="cald-flag mono">{ev.currency}</div>
        <div className="cald-headmain">
          <h1>{ev.event}</h1>
          <div className="cald-sub">
            <span>{t('affects', { markets: affects })}</span>
            {ev.periodicity && (
              <>
                <span className="sep">·</span>
                <span>{t(`periodicity.${ev.periodicity}`)}</span>
              </>
            )}
            {!ev.time_confirmed && (
              <>
                <span className="sep">·</span>
                <span>{t('timeUnconfirmed')}</span>
              </>
            )}
            {ev.revised && (
              <>
                <span className="sep">·</span>
                <span>{t('revisedBadge')}</span>
              </>
            )}
          </div>
          <div className="cald-times mono">
            {dayLabel} · {marketTime}
            {city ? ` ${city}` : ''} ·{' '}
            {t('localTime', { time: localTime, offset: utcOffsetLabel() })}
          </div>
          <div className="cald-prov">
            {ev.organism ? (
              t('provenance.organism', { organism: ev.organism })
            ) : (
              <span className="missing">{t('provenance.organismMissing')}</span>
            )}
            {' · '}
            {ev.value_unit ? (
              ev.value_unit
            ) : (
              <span className="missing">{t('detail.unitMissing')}</span>
            )}
          </div>
        </div>
        {cd && (
          <div className="cald-cd">
            {/* Défaut A — the label matches the tense: "Publiée" once the release
                moment has passed, "Publication dans" while it is still ahead. The
                bascule is scheduled_at vs now (cd.past), so a release due later
                today still reads as upcoming. */}
            <div className="k">
              {t(cd.past ? 'detail.countdownLabelPast' : 'detail.countdownLabel')}
            </div>
            <div className="v mono">{fmtCountdown(cd, t)}</div>
          </div>
        )}
      </div>

      {/* 2 — CURVE (only when a published series exists) */}
      {(ev.value_series?.length ?? 0) > 0 && (
        <CurveCard ev={ev} locale={locale} t={t} />
      )}

      {/* 3 — FOUR QUESTIONS (only when at least one measure was computed) */}
      {showQuestions && measures && (
        <QuestionsSection measures={measures} locale={locale} t={t} />
      )}

      {/* 4 — MIA */}
      <MiaBlock pedKey={miaKey} t={t} />

      {/* 5 — GO TO SOURCE (issuing organism only) */}
      <SourceSection eventKey={eventKey} attribution={attribution} t={t} />

      {/* 6 — PEDAGOGY (only when a REAL fiche exists for this publication — no
          generic filler; Défaut B). The per-fiche disclaimer was folded into the
          single page-level warning below (Défaut C). */}
      {hasFiche && eventKey && (
        <div className="cald-card">
          <div className="cald-card-h">
            <h3>{t('pub.pedagogy.title')}</h3>
            <span className="cald-badge">{t('pub.pedagogy.badge')}</span>
          </div>
          <p className="pub-ped-body">{t(`pub.pedagogy.${eventKey}.body`)}</p>
        </div>
      )}

      {/* SINGLE page-level warning — "what this page does not say" — placed after
          the last factual content (Défaut C). The three former disclaimers are
          consolidated here; the mention under M.I.A stays, as it is about her. */}
      <div className="cal-nono" role="note">
        <div>
          <div className="t">{t('detail.nono.title')}</div>
          <div className="b">
            <ul>
              {nonoItems.map((item, i) => (
                <li key={i}>{item}</li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </>
  );
}
