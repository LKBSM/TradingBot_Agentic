'use client';

import * as React from 'react';
import Link from 'next/link';
import { useTranslations } from 'next-intl';
import { ChevronLeft } from 'lucide-react';
import { useLocalizedHref } from '@/lib/i18n/href';
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
  const W = 640;
  const H = 150;
  const padX = 34;
  const padTop = 22;
  const padBottom = 26;
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
  const upcomingY = yAt((min + max) / 2); // midline — it carries NO value

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
              <circle className="dot" cx={xAt(i)} cy={yAt(p.value)} r={3} />
              <text
                className="pt-val"
                x={xAt(i)}
                y={yAt(p.value) - 7}
                textAnchor="middle"
              >
                {p.value}
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
            y={upcomingY - 9}
            textAnchor="middle"
          >
            {t('pub.curve.upcoming')}
          </text>
        </svg>
      </div>

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
  m,
  locale,
  t,
}: {
  m: CalmBeforeMeasure;
  locale: string;
  t: ReturnType<typeof useTranslations>;
}) {
  const market = t(`market.${m.provenance.market}` as 'market.XAUUSD');
  return (
    <div className="pub-qcard">
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
  m,
  locale,
  t,
}: {
  m: StructureStateMeasure;
  locale: string;
  t: ReturnType<typeof useTranslations>;
}) {
  const market = t(`market.${m.provenance.market}` as 'market.XAUUSD');
  return (
    <div className="pub-qcard">
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
  m,
  locale,
  t,
}: {
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
  return (
    <div className="cald-card pub-qsection">
      <div className="cald-card-h">
        <h3>{t('pub.questions.sectionTitle')}</h3>
        <span className="cald-badge">
          {t('pub.questions.sectionBadge', { market })}
        </span>
      </div>
      {measures.calm_before && (
        <CalmBeforeCard m={measures.calm_before} locale={locale} t={t} />
      )}
      {measures.structure_state && (
        <StructureCard m={measures.structure_state} locale={locale} t={t} />
      )}
      {measures.return_to_calm && (
        <ReturnToCalmCard m={measures.return_to_calm} locale={locale} t={t} />
      )}
    </div>
  );
}

/* --------------------------------------------------------------------------
 * Section 4 — MIA (presentational only)
 * ------------------------------------------------------------------------ */

function MiaBlock({ t }: { t: ReturnType<typeof useTranslations> }) {
  return (
    <div className="cald-card pub-mia">
      <div className="cald-card-h">
        <h3>{t('pub.mia.title')}</h3>
      </div>
      <p className="pub-mia-sub">{t('pub.mia.subtitle')}</p>
      <div className="pub-mia-suggests">
        {(['s1', 's2', 's3'] as const).map((s) => (
          <button key={s} type="button" className="pub-mia-chip">
            {t(`pub.mia.${s}`)}
          </button>
        ))}
      </div>
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
      <p className="pub-mia-cap">{t('pub.mia.capability')}</p>
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

  // Pedagogy body/nono keyed by the event; only these two have a bespoke fiche.
  const pedKey =
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
            <div className="k">{t('detail.countdownLabel')}</div>
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
      <MiaBlock t={t} />

      {/* 5 — GO TO SOURCE (issuing organism only) */}
      <div className="cald-card">
        <div className="cald-card-h">
          <h3>{t('pub.source.title')}</h3>
        </div>
        <p className="pub-src-intro">{t('pub.source.intro')}</p>
        {attribution && (
          <a
            className="pub-src-link"
            href={attribution.policy_url}
            target="_blank"
            rel="noreferrer noopener"
          >
            {t('pub.source.link')}
          </a>
        )}
      </div>

      {/* 6 — PEDAGOGY */}
      <div className="cald-card">
        <div className="cald-card-h">
          <h3>{t('pub.pedagogy.title')}</h3>
          <span className="cald-badge">{t('pub.pedagogy.badge')}</span>
        </div>
        <p className="pub-ped-body">{t(`pub.pedagogy.${pedKey}.body`)}</p>
        <div className="pub-ped-nono">
          <div className="t">{t('pub.pedagogy.nonoTitle')}</div>
          <div className="b">{t(`pub.pedagogy.${pedKey}.nono`)}</div>
        </div>
      </div>

      {/* Page-level "what this page does not say" — kept at the very bottom. */}
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
