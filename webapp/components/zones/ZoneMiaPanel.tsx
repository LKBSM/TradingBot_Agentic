'use client';

import * as React from 'react';
import { useLocale, useTranslations } from 'next-intl';
import { cn } from '@/lib/utils';
import { AgentAvatar } from '@/components/chat/AgentAvatar';
import { ChatComposer } from '@/components/chat/ChatComposer';
import { useDictationCopy } from '@/lib/scanner-chat/use-dictation-copy';
import { useReadingFormatters } from '@/lib/market-reading/use-reading-formatters';
import type { LiquidityPool } from '@/types/market-reading';
import {
  contactCount,
  formatZoneShortTime,
  isConsumed,
  type SiblingZone,
  type ZoneLifecycle,
  zoneProximity,
} from '@/lib/zones/lifecycle';
import { buildConfluence } from '@/lib/zones/confluence';

type ZonesT = ReturnType<typeof useTranslations>;
type ReadingFmt = ReturnType<typeof useReadingFormatters>;

/**
 * M.I.A panel for /zones. It shows the selected zone as its subject and switches
 * subject on a card click WITHOUT reloading. Every fact it states is read from
 * the SAME data as the card (`buildConfluence`, the contact ledger, the geometry)
 * — no parallel source, no recompute, no prediction. It describes and explains;
 * it never says where the price will go nor whether a zone is « good ».
 *
 * The user runs at zero credits by choice, so answers are generated LOCALLY from
 * that shared data — deterministic, honest, and identical to what the card shows.
 *
 * CLN-1 §2 — the panel now mirrors the /app chat format: the shared
 * `ChatComposer` carries the input + dictation + privacy note, the four
 * prefabricated question blocks are gone, and the conversation takes the majority
 * of the height. The subject block compacts once a conversation has started, and
 * disappears entirely when no zone is selected (CLN-1 §3) — the field stays
 * usable and the running conversation is preserved.
 */

type Topic = 'whatElse' | 'explainKind' | 'compareUpper' | 'lastContact';

interface PanelCtx {
  instrument: string;
  price: number | null;
  sameTf: readonly ZoneLifecycle[];
  siblings: readonly SiblingZone[];
  pools: readonly LiquidityPool[];
}

function kindName(zone: ZoneLifecycle, t: ZonesT): string {
  return zone.kind === 'ob' ? t('kind.ob') : t('kind.fvg');
}

function answerFor(
  topic: Topic,
  zone: ZoneLifecycle,
  ctx: PanelCtx,
  t: ZonesT,
  fmt: ReadingFmt,
  locale: string,
): string {
  const kn = kindName(zone, t);
  if (topic === 'explainKind') {
    return zone.kind === 'ob' ? t('mia.answer.explainOb') : t('mia.answer.explainFvg');
  }
  if (topic === 'whatElse') {
    const facts = buildConfluence(zone, ctx.sameTf, ctx.siblings, ctx.pools);
    if (facts.length === 0) return t('mia.answer.whatElseNone');
    const parts = facts.map((f) => {
      if (f.relation === 'liquidity') {
        return t('mia.frag.liquidity', {
          side: t(`liquiditySide.${f.liquiditySide}`),
          level: fmt.price(f.level as number, ctx.instrument),
        });
      }
      const kd =
        (f.kind === 'ob' ? t('kind.ob') : t('kind.fvg')) +
        (f.direction ? ` ${fmt.direction(f.direction)}` : '');
      const where = f.timeframe ? t('mia.frag.unit', { tf: f.timeframe }) : t('mia.frag.sameUnit');
      const rel = t(`mia.frag.rel_${f.relation}`);
      return t('mia.frag.zone', { kd, where, rel });
    });
    return t('mia.answer.whatElse', { list: parts.join(' ') });
  }
  if (topic === 'compareUpper') {
    const upper = ctx.siblings.filter(
      (s) => s.levelLow < zone.levelHigh && zone.levelLow < s.levelHigh,
    );
    if (upper.length === 0) return t('mia.answer.compareNone');
    const s = upper[0]!;
    const kd = (s.kind === 'ob' ? t('kind.ob') : t('kind.fvg')) + (s.direction ? ` ${fmt.direction(s.direction)}` : '');
    return t('mia.answer.compareUpper', {
      tf: s.timeframe,
      kd,
      band: fmt.band(s.levelLow, s.levelHigh, ctx.instrument),
    });
  }
  // lastContact
  const last = zone.contacts[zone.contacts.length - 1] ?? null;
  if (!last) return t('mia.answer.lastNone');
  const when = fmt.relativePast(last.at);
  const time = formatZoneShortTime(last.at, locale);
  const level = fmt.price(last.level, ctx.instrument);
  if (last.outcome === 'inside') return t('mia.answer.lastInside', { when, time });
  if (last.outcome === 'traversal') return t('mia.answer.lastTraversal', { when, time, level });
  if (last.outcome === 'edge_touch') return t('mia.answer.lastEdge', { when, time });
  return t('mia.answer.lastEntry', { when, time, level });
}

/**
 * Route a free-text question to one of the closed topics by keyword — LOCAL, no
 * network, no LLM (the user runs at zero credits by choice). Bilingual stems
 * (fr + en). Returns null when nothing matches → the panel answers with an honest
 * "here is what I can describe" fallback rather than fabricating an interpretation.
 * M.I.A stays a describer bound to the card's data; it never judges or predicts.
 */
function matchTopic(raw: string): Topic | null {
  const t = raw
    .toLowerCase()
    .normalize('NFD')
    .replace(/[̀-ͯ]/g, ''); // strip accents for robust matching
  const has = (re: RegExp) => re.test(t);
  // Order matters: an explicit "explain/what is" wins over a generic "level".
  if (has(/explique|explain|c ?est quoi|qu ?est-?ce|what ?is|definition|type de zone|kind of/))
    return 'explainKind';
  if (has(/superieur|au-?dessus.*unite|upper|higher|compare|comparer|h1|h4|daily|plus grande? unite/))
    return 'compareUpper';
  if (has(/dernier|last|contact|touche|entre|entree|passe|happened|penetr/)) return 'lastContact';
  if (has(/autre|meme (niveau|endroit)|else|around|level|nearby|proximite|liquidit|confluen/))
    return 'whatElse';
  return null;
}

interface Turn {
  q: string;
  a: string;
}

export function ZoneMiaPanel({
  zone,
  ctx,
  className,
}: {
  zone: ZoneLifecycle | null;
  ctx: PanelCtx;
  className?: string;
}) {
  const t = useTranslations('zones');
  const fmt = useReadingFormatters();
  const locale = useLocale();
  const [turns, setTurns] = React.useState<Turn[]>([]);
  const dictationCopy = useDictationCopy();

  // Clear the (local) conversation only when arriving at a DIFFERENT non-null
  // zone — a new zone is a new topic. Deselecting (zone → null) or re-selecting
  // the same zone keeps the running conversation (CLN-1 §3: deselect ≠ reset).
  const zoneId = zone?.id ?? null;
  const prevZoneIdRef = React.useRef<string | null>(null);
  React.useEffect(() => {
    if (zoneId && zoneId !== prevZoneIdRef.current) setTurns([]);
    if (zoneId) prevZoneIdRef.current = zoneId;
  }, [zoneId]);

  // Free-text, routed LOCALLY to a topic (no network, no LLM — the user runs at
  // zero credits by choice). An unrecognised question gets an honest « here is
  // what I can describe » answer. With no zone selected, the field stays usable
  // but M.I.A honestly says it needs a zone rather than inventing a subject.
  const handleSubmit = React.useCallback(
    (q: string) => {
      if (!zone) {
        setTurns((prev) => [...prev, { q, a: t('mia.answer.noZone') }]);
        return;
      }
      const topic = matchTopic(q);
      const a = topic ? answerFor(topic, zone, ctx, t, fmt, locale) : t('mia.answer.fallback');
      setTurns((prev) => [...prev, { q, a }]);
    },
    [zone, ctx, t, fmt, locale],
  );

  const hasConversation = turns.length > 0;

  const prox = zone ? zoneProximity(zone, ctx.price) : null;
  const position = prox
    ? prox.inside
      ? t('mia.pos.inside')
      : t(`mia.pos.${prox.side}`)
    : '';
  const tag = zone
    ? `${zone.kind === 'ob' ? 'OB' : 'FVG'}${zone.direction === 'bullish' ? ' ↑' : zone.direction === 'bearish' ? ' ↓' : ''}`
    : '';

  return (
    <aside className={cn('zmia', className)} aria-label={t('mia.title')}>
      <div className="zmiah">
        <AgentAvatar size="sm" presence />
        <span>
          <span className="nm">{t('mia.name')}</span>
        </span>
      </div>

      {/* Subject — only when a zone is selected (CLN-1 §3: no zone → no element,
          never an empty or filler block). Compacts once a conversation started. */}
      {zone && (
        <div
          className={cn('zmia-subj', hasConversation && 'compact')}
          data-testid="mia-subject"
        >
          <div className="k">{t('mia.subjectLabel')}</div>
          <div className="v">
            {tag} · {fmt.band(zone.levelLow, zone.levelHigh, ctx.instrument)}
          </div>
          {position && !hasConversation && <div className="m">{position}</div>}
        </div>
      )}

      {/* Conversation (local, factual) — takes the majority of the height. The
          intro is M.I.A's opening message for the selected zone. */}
      <div className="zmia-body">
        {zone && (
          <div className="bub a">
            {t('mia.intro', { kind: kindName(zone, t), count: contactCount(zone) })}
          </div>
        )}
        {!zone && !hasConversation && <div className="zmia-empty">{t('mia.empty')}</div>}
        {turns.map((turn, i) => (
          <React.Fragment key={i}>
            <div className="bub u">{turn.q}</div>
            <div className="bub a">{turn.a}</div>
          </React.Fragment>
        ))}
      </div>

      {/* Shared composer (CLN-1 §2) — same format/dictation/privacy note as /app.
          The placeholder never claims a zone is chosen. */}
      <div className="zmia-foot">
        <ChatComposer
          onSubmit={handleSubmit}
          placeholder={zone ? t('mia.input.placeholder') : t('mia.input.placeholderIdle')}
          ariaLabel={zone ? t('mia.input.placeholder') : t('mia.input.placeholderIdle')}
          sendAria={t('mia.input.send')}
          privacyNote={dictationCopy.privacy}
        />
      </div>
      {/* CLN-1 §5 — the panel's own educational note was removed: the single
          page disclaimer (rail footer on desktop, mobile footer < 768px) carries
          it, and two stacked notices on one view neutralise each other. */}
    </aside>
  );
}
