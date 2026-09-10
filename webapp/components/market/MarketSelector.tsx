'use client';

import * as React from 'react';
import { useTranslations } from 'next-intl';
import { ChevronDown, ChevronRight, Pin, PinOff, Search } from 'lucide-react';
import { cn } from '@/lib/utils';
import {
  MARKET_SPECS,
  marketGlyph,
  marketLabel,
  marketTimeframes,
} from '@/lib/markets';
import {
  CATALOG_GROUPS,
  CATALOG_ONLY_COUNT,
  CATALOG_UX_TEST_ENABLED,
  catalogEntriesByGroup,
  catalogGlyph,
  catalogLabel,
  type CatalogEntry,
  type CatalogGroup,
} from '@/lib/market-catalog';
import { M1_ENABLED } from '@/lib/market-reading/perimeter';
import { formatInstrument, formatTimeframe } from '@/lib/market-reading/formatters';
import { usePinnedMarkets } from '@/lib/market-reading/market-pins';
import { SearchField } from '@/components/shell/primitives';
import type { Combo } from '@/lib/market-reading/store';

/**
 * MKT-1 — the single reusable market selector (search + pinned markets +
 * timeframe), shared by /app (rail + mobile sidebar) and /zones (header bar). It
 * reads the market registry, so it never enumerates markets itself: adding a
 * market to config/markets.json surfaces it here everywhere at once.
 *
 * Three variants render the SAME logic in the three shells that need it:
 *   • 'rail'  — always-open column, the product-shell rail CSS (.mkt/.tf/.rail-lbl).
 *   • 'panel' — always-open column, Tailwind (the /app mobile sidebar).
 *   • 'bar'   — compact header form: a market dropdown + inline timeframe pills.
 *
 * Honesty (mission §3): the search + pinned list ONLY ever show markets present
 * in the registry (no phantom market); an empty search shows an explicit
 * "aucun marché ne correspond" message — never a silent fallback.
 *
 * ── MKT-1 TEST UX (flag, OFF by default) ────────────────────────────────────
 * When NEXT_PUBLIC_SHOW_MARKET_CATALOG_UX_TEST is on, the selector ALSO lists a
 * display-only catalogue of ~100 markets, to test how this component holds at
 * that scale before the data exists. Those markets are grouped by category in
 * collapsed sections, visually separated from the real ones ("Suivis par le
 * moteur"), and carry a permanent banner so the mode is never forgotten on.
 * Picking one renders the honest empty state — no request is even sent
 * (cf. lib/market-reading/hooks.ts). With the flag off, CATALOG_UX_TEST_ENABLED
 * is a build-time false and every branch below is dead code: the component
 * behaves EXACTLY as it did before.
 */

type Variant = 'rail' | 'panel' | 'bar';

export interface MarketSelectorProps {
  variant?: Variant;
  /** Currently active market + timeframe (null = nothing selected yet). */
  active: Combo | null;
  /** Called with the full combo when the user picks a market or a timeframe. */
  onSelect(combo: Combo): void;
  /**
   * Whether to reflect the active highlight. The rail is shared chrome shown on
   * every product route but only /app owns the combo, so it passes false off /app.
   */
  reflectActive?: boolean;
  className?: string;
}

/** Accent/case-insensitive normalisation for the search filter. */
function normalize(value: string): string {
  return value
    .normalize('NFD')
    .replace(/[̀-ͯ]/g, '')
    .toLowerCase()
    .trim();
}

/** A market matches the query if it appears in its id or its human label. */
function marketMatches(id: string, query: string): boolean {
  const q = normalize(query);
  if (!q) return true;
  return normalize(`${id} ${marketLabel(id)}`).includes(q);
}

/**
 * Same rule for a catalogue entry, widened to the provider symbol and to the
 * category name: at 100 entries "crypto" or "indice" is how one actually looks
 * for a market, and an id is not always what the user has in mind.
 */
function catalogMatches(entry: CatalogEntry, query: string, groupName: string): boolean {
  const q = normalize(query);
  if (!q) return true;
  return normalize(`${entry.id} ${entry.label} ${entry.symbol} ${groupName}`).includes(q);
}

/** The timeframes to display for a market: its perimeter minus M1 when gated off. */
function displayTimeframesFor(marketId: string): string[] {
  return marketTimeframes(marketId).filter((tf) => tf !== 'M1' || M1_ENABLED);
}

/**
 * Build the combo when a market is picked: keep the current timeframe if the
 * market serves it, else fall back to that market's first displayed timeframe.
 *
 * A catalogue-only market serves no timeframe at all (it has none, by design),
 * so the current one is carried over unchanged — the empty state is about the
 * market, and switching the user's timeframe under them would be gratuitous.
 */
function comboForMarket(marketId: string, currentTf: string | undefined): Combo {
  const tfs = displayTimeframesFor(marketId);
  if (tfs.length === 0) return { instrument: marketId, timeframe: currentTf ?? '' };
  const timeframe = currentTf && tfs.includes(currentTf) ? currentTf : tfs[0] ?? currentTf ?? '';
  return { instrument: marketId, timeframe };
}

/** Label for any id, real or catalogue-only. */
function displayLabel(id: string): string {
  const real = formatInstrument(id);
  return real === id ? catalogLabel(id) : real;
}

/** Glyph for any id, real or catalogue-only. */
function displayGlyph(id: string): string {
  return MARKET_SPECS.some((s) => s.id === id) ? marketGlyph(id) : catalogGlyph(id);
}

/** i18n key of a catalogue category header. */
const GROUP_KEY: Record<CatalogGroup, string> = {
  'fx-major': 'sidebar.groupFxMajor',
  'fx-minor': 'sidebar.groupFxMinor',
  'fx-exotic': 'sidebar.groupFxExotic',
  metal: 'sidebar.groupMetal',
  index: 'sidebar.groupIndex',
  crypto: 'sidebar.groupCrypto',
};

// ─────────────────────────────────────────────────────────────────────────────
// Root
// ─────────────────────────────────────────────────────────────────────────────
export function MarketSelector({
  variant = 'panel',
  active,
  onSelect,
  reflectActive = true,
  className,
}: MarketSelectorProps) {
  if (variant === 'bar') {
    return <BarSelector active={active} onSelect={onSelect} className={className} />;
  }
  return (
    <ColumnSelector
      variant={variant}
      active={active}
      onSelect={onSelect}
      reflectActive={reflectActive}
      className={className}
    />
  );
}

/** Category names, resolved once, for the search-by-category filter. */
function useGroupNames(): Record<CatalogGroup, string> {
  const t = useTranslations('app');
  return React.useMemo(() => {
    const out = {} as Record<CatalogGroup, string>;
    for (const g of CATALOG_GROUPS) out[g] = t(GROUP_KEY[g]);
    return out;
  }, [t]);
}

/** Catalogue entries per category, filtered by the live query. */
function useCatalogGroups(query: string): { group: CatalogGroup; entries: CatalogEntry[] }[] {
  const groupNames = useGroupNames();
  return React.useMemo(() => {
    if (!CATALOG_UX_TEST_ENABLED) return [];
    return CATALOG_GROUPS.map((group) => ({
      group,
      entries: catalogEntriesByGroup(group).filter((e) =>
        catalogMatches(e, query, groupNames[group]),
      ),
    }));
  }, [query, groupNames]);
}

// ─────────────────────────────────────────────────────────────────────────────
// "Mode test UX" banner — permanent while the flag is on, deliberately not
// dismissible: the whole point is that it cannot be left on by accident. The
// counts are COMPUTED, so they stay true if the catalogue ever changes.
// ─────────────────────────────────────────────────────────────────────────────
function UxTestBanner({ variant }: { variant: Variant }) {
  const t = useTranslations('app');
  const rail = variant === 'rail';
  return (
    <div
      className={cn(
        rail
          ? 'marketsel-uxtest'
          : 'rounded-md border border-sentinel-warn/40 bg-sentinel-warn/10 px-2.5 py-2',
      )}
      data-testid="mkt-uxtest-banner"
      role="note"
    >
      <span
        className={cn(
          rail ? 'marketsel-uxtest-t' : 'block text-xs font-semibold text-foreground',
        )}
      >
        {t('sidebar.uxTestBadge')}
      </span>
      <span
        className={cn(
          rail
            ? 'marketsel-uxtest-b'
            : 'mt-0.5 block text-[11px] leading-snug text-muted-foreground',
        )}
      >
        {t('sidebar.uxTestNote', { count: CATALOG_ONLY_COUNT, real: MARKET_SPECS.length })}
      </span>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// One collapsible catalogue category
// ─────────────────────────────────────────────────────────────────────────────
function CatalogGroupSection({
  variant,
  group,
  entries,
  activeMarket,
  forceOpen,
  onPick,
}: {
  variant: Variant;
  group: CatalogGroup;
  entries: readonly CatalogEntry[];
  activeMarket: string | null;
  /** A live search opens every group that has a hit, so results are never hidden. */
  forceOpen: boolean;
  onPick(id: string): void;
}) {
  const t = useTranslations('app');
  // Collapsed by default — 98 markets expanded at once is a wall, and the repo
  // already settled this same question for the scanner condition families
  // (commit c6266e5, « les 4 familles de conditions se chargent fermées »).
  const [open, setOpen] = React.useState(false);
  const isOpen = open || forceOpen;

  if (entries.length === 0) return null;

  const rail = variant === 'rail';
  const rowVariant: 'rail' | 'panel' = rail ? 'rail' : 'panel';

  return (
    <div data-testid={`mkt-group-${group}`}>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={isOpen}
        data-testid={`mkt-group-head-${group}`}
        className={cn(
          rail
            ? 'marketsel-grouphead'
            : 'flex w-full items-center gap-1.5 px-1 py-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground transition-colors hover:text-foreground',
        )}
      >
        {isOpen ? (
          <ChevronDown className={rail ? undefined : 'h-3.5 w-3.5 shrink-0'} aria-hidden />
        ) : (
          <ChevronRight className={rail ? undefined : 'h-3.5 w-3.5 shrink-0'} aria-hidden />
        )}
        <span>{t(GROUP_KEY[group])}</span>
        <span className={cn(rail ? 'marketsel-groupcount' : 'ml-auto font-normal tabular-nums')}>
          {entries.length}
        </span>
      </button>
      {isOpen && (
        <ul className={rail ? undefined : 'space-y-1'}>
          {entries.map((e) => (
            <li key={e.id}>
              <MarketRow
                variant={rowVariant}
                marketId={e.id}
                activeMarket={activeMarket}
                pinned={false}
                catalogOnly
                onPick={() => onPick(e.id)}
                onTogglePin={() => undefined}
              />
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Column form (rail + panel): always-open search / pinned / list / timeframe
// ─────────────────────────────────────────────────────────────────────────────
function ColumnSelector({
  variant,
  active,
  onSelect,
  reflectActive,
  className,
}: {
  variant: 'rail' | 'panel';
  active: Combo | null;
  onSelect(combo: Combo): void;
  reflectActive: boolean;
  className?: string;
}) {
  const t = useTranslations('app');
  const [query, setQuery] = React.useState('');
  const { pinned, isPinned, toggle } = usePinnedMarkets();

  const activeMarket = reflectActive ? active?.instrument ?? null : null;

  // Memoised: at 100 entries the filter would otherwise re-run on every
  // keystroke in every mounted variant. Cheap to memoise, and it keeps typing
  // free of work the render does not need.
  const allMarkets = React.useMemo(
    () => MARKET_SPECS.map((s) => s.id).filter((id) => marketMatches(id, query)),
    [query],
  );
  const pinnedMarkets = React.useMemo(
    () => pinned.filter((id) => marketMatches(id, query)),
    [pinned, query],
  );

  const catalogByGroup = useCatalogGroups(query);
  const catalogHits = catalogByGroup.reduce((n, g) => n + g.entries.length, 0);
  const hasResults = allMarkets.length > 0 || catalogHits > 0;
  const searching = normalize(query).length > 0;

  // APP-1 — the « Marchés » section lists only what « Épinglés » does NOT already
  // show. With the 2-market product both pinned, it would just repeat the pinned
  // list, so it is hidden entirely (a section with no NEW content does not show).
  // The empty-search message still appears when nothing matches at all.
  const unpinnedMarkets = allMarkets.filter((id) => !isPinned(id));
  const showAllSection = !hasResults || unpinnedMarkets.length > 0;

  const rail = variant === 'rail';
  // Only the UX-test mode renames the section: with the flag off the rail reads
  // exactly as it did before.
  const realSectionLabel = CATALOG_UX_TEST_ENABLED
    ? t('sidebar.followedByEngine')
    : t('sidebar.markets');

  return (
    <div
      className={cn(rail ? 'marketsel-rail' : 'space-y-4', className)}
      data-testid={`mkt-selector-${variant}`}
    >
      {/* Search — sticky, so it stays reachable however long the list grows. */}
      <div className={rail ? 'marketsel-stick' : 'sticky top-0 z-10 bg-background pb-1'}>
        {rail ? (
          <SearchField
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={t('sidebar.searchPlaceholder')}
            aria-label={t('sidebar.searchAria')}
          />
        ) : (
          <div className="relative">
            <Search
              className="pointer-events-none absolute left-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground"
              aria-hidden
            />
            <input
              type="search"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder={t('sidebar.searchPlaceholder')}
              aria-label={t('sidebar.searchAria')}
              className="w-full rounded-md border border-border/60 bg-background py-2 pl-8 pr-3 text-sm text-foreground placeholder:text-muted-foreground focus:border-ring focus:outline-none focus:ring-1 focus:ring-ring"
            />
          </div>
        )}
      </div>

      {CATALOG_UX_TEST_ENABLED && <UxTestBanner variant={variant} />}

      {/* Pinned markets */}
      {pinnedMarkets.length > 0 && (
        <div className={rail ? undefined : 'space-y-1.5'}>
          <div
            className={cn(
              rail ? 'rail-lbl' : 'px-1',
              !rail && 'flex items-baseline justify-between',
            )}
          >
            <span className={rail ? undefined : 'text-xs font-semibold uppercase tracking-wide text-primary'}>
              {t('sidebar.pinned')}
            </span>
            <span
              className={cn(
                rail ? 'marketsel-notsynced' : 'text-[10px] font-normal normal-case text-muted-foreground',
              )}
              title={t('sidebar.notSynced')}
            >
              {t('sidebar.notSynced')}
            </span>
          </div>
          <ul className={rail ? undefined : 'space-y-1'}>
            {pinnedMarkets.map((id) => (
              <li key={`pin:${id}`}>
                <MarketRow
                  variant={variant}
                  marketId={id}
                  activeMarket={activeMarket}
                  pinned
                  onPick={() => onSelect(comboForMarket(id, active?.timeframe))}
                  onTogglePin={() => toggle(id)}
                />
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Markets not already pinned (APP-1) — hidden when it would only repeat
          « Épinglés »; still shows the empty-search message when nothing matches. */}
      {showAllSection && (
        <div className={rail ? undefined : 'space-y-1.5'}>
          {rail ? (
            <div className="rail-lbl">{realSectionLabel}</div>
          ) : (
            <p className="px-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              {realSectionLabel}
            </p>
          )}
          {hasResults ? (
            <ul className={rail ? undefined : 'space-y-1'}>
              {unpinnedMarkets.map((id) => (
                <li key={id}>
                  <MarketRow
                    variant={variant}
                    marketId={id}
                    activeMarket={activeMarket}
                    pinned={isPinned(id)}
                    onPick={() => onSelect(comboForMarket(id, active?.timeframe))}
                    onTogglePin={() => toggle(id)}
                  />
                </li>
              ))}
            </ul>
          ) : (
            <p
              className={cn(
                rail ? 'marketsel-empty' : 'px-1 py-6 text-center text-sm text-muted-foreground',
              )}
            >
              {t('sidebar.noResults', { query })}
            </p>
          )}
        </div>
      )}

      {/* Display-only catalogue, grouped by category (UX-test flag ON only). */}
      {CATALOG_UX_TEST_ENABLED &&
        catalogByGroup.map(({ group, entries }) => (
          <CatalogGroupSection
            key={group}
            variant={variant}
            group={group}
            entries={entries}
            activeMarket={activeMarket}
            forceOpen={searching}
            onPick={(id) => onSelect(comboForMarket(id, active?.timeframe))}
          />
        ))}

      {/* Timeframe — for the active market (falls back to the first market). */}
      <TimeframeControl
        variant={variant}
        marketId={active?.instrument ?? MARKET_SPECS[0]?.id ?? ''}
        activeTimeframe={reflectActive ? active?.timeframe ?? null : null}
        onPick={(tf) =>
          onSelect({ instrument: active?.instrument ?? MARKET_SPECS[0]?.id ?? '', timeframe: tf })
        }
      />
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// One market row (rail or panel)
// ─────────────────────────────────────────────────────────────────────────────
function MarketRow({
  variant,
  marketId,
  activeMarket,
  pinned,
  catalogOnly = false,
  onPick,
  onTogglePin,
}: {
  variant: 'rail' | 'panel';
  marketId: string;
  activeMarket: string | null;
  pinned: boolean;
  /**
   * Display-only market: no pin affordance. Pinning a market with no data would
   * be a promise the product cannot keep — the row stays pickable (that IS the
   * test), it simply offers nothing that implies coverage.
   */
  catalogOnly?: boolean;
  onPick(): void;
  onTogglePin(): void;
}) {
  const t = useTranslations('app');
  const isActive = activeMarket === marketId;
  const label = displayLabel(marketId);
  const pinLabel = pinned
    ? t('sidebar.unpinAria', { combo: label })
    : t('sidebar.pinAria', { combo: label });

  if (variant === 'rail') {
    return (
      <div className="marketsel-railrow">
        <button
          type="button"
          className={cn('mkt', isActive && 'on', catalogOnly && 'mkt-nodata')}
          aria-current={isActive ? 'true' : undefined}
          data-catalog-only={catalogOnly ? 'true' : undefined}
          onClick={onPick}
        >
          <span className="ic mono" aria-hidden>
            {displayGlyph(marketId)}
          </span>
          <span className="nm">{label}</span>
        </button>
        {!catalogOnly && (
          <button
            type="button"
            className={cn('mkt-pin', pinned && 'on')}
            aria-pressed={pinned}
            aria-label={pinLabel}
            title={pinned ? t('sidebar.unpin') : t('sidebar.pin')}
            onClick={onTogglePin}
          >
            {pinned ? <PinOff aria-hidden /> : <Pin aria-hidden />}
          </button>
        )}
      </div>
    );
  }

  return (
    <div
      className={cn(
        'flex items-stretch rounded-md border-l-2 transition-colors',
        isActive ? 'border-l-primary bg-primary/10' : 'border-l-transparent hover:bg-muted',
      )}
    >
      <button
        type="button"
        onClick={onPick}
        aria-current={isActive ? 'true' : undefined}
        data-catalog-only={catalogOnly ? 'true' : undefined}
        className={cn(
          'flex min-w-0 flex-1 items-center gap-2 px-3 py-2 text-left text-sm',
          isActive ? 'font-medium text-foreground' : 'text-muted-foreground hover:text-foreground',
        )}
      >
        <span
          className="grid h-5 w-5 shrink-0 place-items-center rounded bg-muted font-mono text-[10px] font-semibold text-muted-foreground"
          aria-hidden
        >
          {displayGlyph(marketId)}
        </span>
        <span className="truncate">{label}</span>
      </button>
      {!catalogOnly && (
        <button
          type="button"
          onClick={onTogglePin}
          aria-pressed={pinned}
          aria-label={pinLabel}
          title={pinned ? t('sidebar.unpin') : t('sidebar.pin')}
          className={cn(
            'flex min-h-[44px] min-w-[44px] shrink-0 items-center justify-center px-2 text-muted-foreground/60 transition-colors hover:text-primary xl:min-h-0 xl:min-w-0',
            pinned && 'text-primary',
          )}
        >
          {pinned ? <PinOff className="h-3.5 w-3.5" aria-hidden /> : <Pin className="h-3.5 w-3.5" aria-hidden />}
        </button>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Timeframe control (rail pills / panel pills / bar pills all share this)
// ─────────────────────────────────────────────────────────────────────────────
function TimeframeControl({
  variant,
  marketId,
  activeTimeframe,
  onPick,
}: {
  variant: Variant;
  marketId: string;
  activeTimeframe: string | null;
  onPick(tf: string): void;
}) {
  const t = useTranslations('app');
  const tfs = displayTimeframesFor(marketId);
  if (tfs.length === 0) return null;

  if (variant === 'rail') {
    return (
      <div className="marketsel-tfstick">
        <div className="rail-lbl">{t('rail.timeframe')}</div>
        <div style={{ display: 'flex', gap: 5, padding: '0 4px' }}>
          {tfs.map((tf) => (
            <button
              key={tf}
              type="button"
              className={cn('tf', activeTimeframe === tf && 'on')}
              aria-pressed={activeTimeframe === tf}
              aria-label={formatTimeframe(tf)}
              onClick={() => onPick(tf)}
            >
              {tf}
            </button>
          ))}
        </div>
      </div>
    );
  }

  // panel + bar: bordered chip group (matches the shell's .tf intent in Tailwind).
  return (
    <div className={variant === 'panel' ? 'space-y-1.5' : undefined}>
      {variant === 'panel' && (
        <p className="px-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          {t('rail.timeframe')}
        </p>
      )}
      <div role="group" aria-label={t('rail.timeframe')} className="inline-flex flex-wrap gap-1 rounded-md border border-border/70 p-1">
        {tfs.map((tf) => (
          <button
            key={tf}
            type="button"
            onClick={() => onPick(tf)}
            aria-pressed={activeTimeframe === tf}
            aria-label={formatTimeframe(tf)}
            className={cn(
              'rounded px-2.5 py-1 font-mono text-xs font-medium transition-colors',
              activeTimeframe === tf
                ? 'bg-foreground text-background'
                : 'text-muted-foreground hover:bg-accent hover:text-accent-foreground',
            )}
          >
            {tf}
          </button>
        ))}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Bar form (/zones header): a market dropdown + inline timeframe pills
// ─────────────────────────────────────────────────────────────────────────────
function BarSelector({
  active,
  onSelect,
  className,
}: {
  active: Combo | null;
  onSelect(combo: Combo): void;
  className?: string;
}) {
  const t = useTranslations('app');
  const [open, setOpen] = React.useState(false);
  const [query, setQuery] = React.useState('');
  const { pinned, isPinned, toggle } = usePinnedMarkets();
  const rootRef = React.useRef<HTMLDivElement>(null);

  const activeMarket = active?.instrument ?? MARKET_SPECS[0]?.id ?? '';

  // Close on outside click / Escape.
  React.useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false);
    };
    document.addEventListener('mousedown', onDown);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onDown);
      document.removeEventListener('keydown', onKey);
    };
  }, [open]);

  const allMarkets = React.useMemo(
    () => MARKET_SPECS.map((s) => s.id).filter((id) => marketMatches(id, query)),
    [query],
  );
  const pinnedMarkets = React.useMemo(
    () => pinned.filter((id) => marketMatches(id, query)),
    [pinned, query],
  );

  const catalogByGroup = useCatalogGroups(query);
  const catalogHits = catalogByGroup.reduce((n, g) => n + g.entries.length, 0);
  const searching = normalize(query).length > 0;

  const pick = (id: string) => {
    onSelect(comboForMarket(id, active?.timeframe));
    setOpen(false);
    setQuery('');
  };

  return (
    <div className="inline-flex flex-wrap items-center gap-2" data-testid="mkt-selector-bar">
      <div ref={rootRef} className={cn('relative', className)}>
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          aria-haspopup="listbox"
          aria-expanded={open}
          aria-label={t('sidebar.markets')}
          className="inline-flex items-center gap-2 rounded-md border border-border/70 bg-background px-2.5 py-1.5 text-sm font-medium text-foreground transition-colors hover:bg-accent"
        >
          <span className="grid h-5 w-5 place-items-center rounded bg-muted font-mono text-[10px] font-semibold text-muted-foreground" aria-hidden>
            {displayGlyph(activeMarket)}
          </span>
          <span>{displayLabel(activeMarket)}</span>
          <ChevronDown className="h-4 w-4 text-muted-foreground" aria-hidden />
        </button>

        {open && (
          <div className="absolute left-0 top-full z-50 mt-1 w-72 rounded-md border border-border bg-popover p-2 shadow-lg">
            <div className="relative mb-2">
              <Search className="pointer-events-none absolute left-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" aria-hidden />
              <input
                type="search"
                autoFocus
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder={t('sidebar.searchPlaceholder')}
                aria-label={t('sidebar.searchAria')}
                className="w-full rounded-md border border-border/60 bg-background py-1.5 pl-8 pr-3 text-sm text-foreground placeholder:text-muted-foreground focus:border-ring focus:outline-none focus:ring-1 focus:ring-ring"
              />
            </div>

            {CATALOG_UX_TEST_ENABLED && (
              <div className="mb-2">
                <UxTestBanner variant="panel" />
              </div>
            )}

            {pinnedMarkets.length > 0 && (
              <div className="mb-2">
                <div className="flex items-baseline justify-between px-1 pb-1">
                  <span className="text-xs font-semibold uppercase tracking-wide text-primary">{t('sidebar.pinned')}</span>
                  <span className="text-[10px] text-muted-foreground" title={t('sidebar.notSynced')}>{t('sidebar.notSynced')}</span>
                </div>
                <ul className="space-y-0.5">
                  {pinnedMarkets.map((id) => (
                    <li key={`pin:${id}`}>
                      <BarRow id={id} active={activeMarket === id} pinned onPick={() => pick(id)} onTogglePin={() => toggle(id)} />
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {allMarkets.length > 0 || catalogHits > 0 ? (
              <div className="max-h-64 space-y-1 overflow-y-auto">
                <ul className="space-y-0.5">
                  {allMarkets.map((id) => (
                    <li key={id}>
                      <BarRow id={id} active={activeMarket === id} pinned={isPinned(id)} onPick={() => pick(id)} onTogglePin={() => toggle(id)} />
                    </li>
                  ))}
                </ul>
                {catalogByGroup.map(({ group, entries }) => (
                  <CatalogGroupSection
                    key={group}
                    variant="bar"
                    group={group}
                    entries={entries}
                    activeMarket={activeMarket}
                    forceOpen={searching}
                    onPick={pick}
                  />
                ))}
              </div>
            ) : (
              <p className="px-1 py-4 text-center text-sm text-muted-foreground">{t('sidebar.noResults', { query })}</p>
            )}
          </div>
        )}
      </div>

      <TimeframeControl
        variant="bar"
        marketId={activeMarket}
        activeTimeframe={active?.timeframe ?? null}
        onPick={(tf) => onSelect({ instrument: activeMarket, timeframe: tf })}
      />
    </div>
  );
}

function BarRow({
  id,
  active,
  pinned,
  onPick,
  onTogglePin,
}: {
  id: string;
  active: boolean;
  pinned: boolean;
  onPick(): void;
  onTogglePin(): void;
}) {
  const t = useTranslations('app');
  const label = displayLabel(id);
  return (
    <div
      className={cn(
        'flex items-stretch rounded transition-colors',
        active ? 'bg-primary/10' : 'hover:bg-muted',
      )}
    >
      <button
        type="button"
        onClick={onPick}
        aria-current={active ? 'true' : undefined}
        className={cn(
          'flex min-w-0 flex-1 items-center gap-2 px-2 py-1.5 text-left text-sm',
          active ? 'font-medium text-foreground' : 'text-muted-foreground hover:text-foreground',
        )}
      >
        <span className="grid h-5 w-5 shrink-0 place-items-center rounded bg-muted font-mono text-[10px] font-semibold text-muted-foreground" aria-hidden>
          {displayGlyph(id)}
        </span>
        <span className="truncate">{label}</span>
      </button>
      <button
        type="button"
        onClick={onTogglePin}
        aria-pressed={pinned}
        aria-label={pinned ? t('sidebar.unpinAria', { combo: label }) : t('sidebar.pinAria', { combo: label })}
        title={pinned ? t('sidebar.unpin') : t('sidebar.pin')}
        className={cn(
          'flex shrink-0 items-center justify-center px-2 text-muted-foreground/60 transition-colors hover:text-primary',
          pinned && 'text-primary',
        )}
      >
        {pinned ? <PinOff className="h-3.5 w-3.5" aria-hidden /> : <Pin className="h-3.5 w-3.5" aria-hidden />}
      </button>
    </div>
  );
}
