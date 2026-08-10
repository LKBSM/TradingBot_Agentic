'use client';

import * as React from 'react';
import { useLocale, useTranslations } from 'next-intl';
import { cn } from '@/lib/utils';
import { AgentAvatar } from '@/components/chat/AgentAvatar';
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
 * M.I.A panel for /zones (mission §3). It ALWAYS shows the selected zone as its
 * subject and switches subject on a card click WITHOUT reloading. Its suggested
 * questions are contextual to the zone, and every fact it states is read from the
 * SAME data as the card (`buildConfluence`, the contact ledger, the geometry) —
 * no parallel source, no recompute, no prediction. It describes and explains; it
 * never says where the price will go nor whether a zone is « good ».
 *
 * The user runs at zero credits by choice, so answers are generated LOCALLY from
 * that shared data — deterministic, honest, and identical to what the card shows.
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

const TOPICS: Topic[] = ['whatElse', 'explainKind', 'compareUpper', 'lastContact'];

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
  const [draft, setDraft] = React.useState('');

  // Switching subject clears the (local) conversation — a new zone, a new topic.
  const zoneId = zone?.id ?? null;
  React.useEffect(() => {
    setTurns([]);
    setDraft('');
  }, [zoneId]);

  if (!zone) {
    return (
      <aside className={cn('zmia', className)} aria-label={t('mia.title')}>
        <div className="zmiah">
          <AgentAvatar size="sm" presence />
          <span>
            <span className="nm">{t('mia.name')}</span>
            <span className="sb">{t('mia.tagline')}</span>
          </span>
        </div>
        <div className="zmia-empty">{t('mia.empty')}</div>
      </aside>
    );
  }

  const ask = (topic: Topic) => {
    const q = t(`mia.suggest.${topic}`);
    const a = answerFor(topic, zone, ctx, t, fmt, locale);
    setTurns((prev) => [...prev, { q, a }]);
  };

  // Free-text: route LOCALLY to a topic (no network, no LLM). An unrecognised
  // question gets an honest « here is what I can describe » answer — never an
  // invented interpretation.
  const submitDraft = (e: React.FormEvent) => {
    e.preventDefault();
    const q = draft.trim();
    if (!q) return;
    const topic = matchTopic(q);
    const a = topic ? answerFor(topic, zone, ctx, t, fmt, locale) : t('mia.answer.fallback');
    setTurns((prev) => [...prev, { q, a }]);
    setDraft('');
  };

  const prox = zoneProximity(zone, ctx.price);
  const position = prox
    ? prox.inside
      ? t('mia.pos.inside')
      : t(`mia.pos.${prox.side}`)
    : '';
  const tag = `${zone.kind === 'ob' ? 'OB' : 'FVG'}${zone.direction === 'bullish' ? ' ↑' : zone.direction === 'bearish' ? ' ↓' : ''}`;

  return (
    <aside className={cn('zmia', className)} aria-label={t('mia.title')}>
      <div className="zmiah">
        <AgentAvatar size="sm" presence />
        <span>
          <span className="nm">{t('mia.name')}</span>
          <span className="sb">{t('mia.tagline')}</span>
        </span>
      </div>

      {/* Subject — always the selected zone. */}
      <div className="zmia-subj" data-testid="mia-subject">
        <div className="k">{t('mia.subjectLabel')}</div>
        <div className="v">
          {tag} · {fmt.band(zone.levelLow, zone.levelHigh, ctx.instrument)}
        </div>
        {position && <div className="m">{position}</div>}
      </div>

      {/* Conversation (local, factual). */}
      <div className="zmia-body">
        <div className="bub a">{t('mia.intro', { kind: kindName(zone, t), count: contactCount(zone) })}</div>
        {turns.map((turn, i) => (
          <React.Fragment key={i}>
            <div className="bub u">{turn.q}</div>
            <div className="bub a">{turn.a}</div>
          </React.Fragment>
        ))}
      </div>

      {/* Contextual suggestions. */}
      <div className="zmia-sugg">
        {TOPICS.map((topic) => (
          <button key={topic} type="button" className="sg" onClick={() => ask(topic)}>
            {t(`mia.suggest.${topic}`)}
          </button>
        ))}
      </div>

      {/* Free-text — routed locally to the same factual answers (0 credit). */}
      <form className="zmia-input" onSubmit={submitDraft}>
        <input
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          placeholder={t('mia.input.placeholder')}
          aria-label={t('mia.input.placeholder')}
        />
        <button type="submit" aria-label={t('mia.input.send')} disabled={!draft.trim()}>
          →
        </button>
      </form>

      <div className="zmia-disc">{t('mia.disclaimer')}</div>
    </aside>
  );
}
