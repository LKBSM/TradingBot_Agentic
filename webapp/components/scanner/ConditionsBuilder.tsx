'use client';

import * as React from 'react';
import { useTranslations } from 'next-intl';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { cn } from '@/lib/utils';
import {
  CONDITION_PALETTE,
  DEFAULT_BOS_MAX_BARS,
  DEFAULT_PROXIMITY_PCT,
  LIQUIDITY_SIDE_OPTIONS,
  PHASE_OPTIONS,
  TREND_OPTIONS,
  VOLATILITY_OPTIONS,
  paletteEntry,
} from '@/lib/conditions/palette';
import type {
  ConditionType,
  ConditionsConfig,
  DirectionFilter,
  LiquiditySideFilter,
  PhaseChoice,
  ScanCondition,
  ScanLogic,
  TrendChoice,
  VolatilityChoice,
} from '@/lib/conditions/types';
import {
  MAX_NAME_CHARS,
  type StrategyMutationResult,
} from '@/lib/conditions/strategy-store';
import { mutationErrorMessage } from './StrategyPanel';

/** Capitalise a locale-agnostic enum value to build an ICU key suffix. */
function cap(v: string): string {
  return v.charAt(0).toUpperCase() + v.slice(1);
}

/**
 * Builder where the user COMPOSES their conditions from the present-tense
 * palette + an AND/OR logic. Used both for first-visit onboarding and for later
 * editing. Descriptive only — no condition here speaks of a future outcome.
 */

interface RowState {
  selected: boolean;
  direction: DirectionFilter;
  maxBars: number;
  trend: TrendChoice;
  phase: PhaseChoice;
  volatility: VolatilityChoice;
  proximityPct: number;
  side: LiquiditySideFilter;
}

type BuilderState = Record<ConditionType, RowState>;

const DEFAULT_ROW: RowState = {
  selected: false,
  direction: 'any',
  maxBars: DEFAULT_BOS_MAX_BARS,
  trend: 'bullish',
  phase: 'trend',
  volatility: 'elevated',
  proximityPct: DEFAULT_PROXIMITY_PCT,
  side: 'any',
};

function initialState(config: ConditionsConfig | null): BuilderState {
  const base = {} as BuilderState;
  for (const entry of CONDITION_PALETTE) {
    base[entry.type] = { ...DEFAULT_ROW };
  }
  if (config) {
    for (const cond of config.conditions) {
      const row = base[cond.type];
      if (row) {
        base[cond.type] = {
          ...row,
          selected: true,
          direction: cond.direction ?? row.direction,
          maxBars: cond.max_bars ?? row.maxBars,
          trend: cond.trend ?? row.trend,
          phase: cond.phase ?? row.phase,
          volatility: cond.volatility ?? row.volatility,
          proximityPct: cond.proximity_pct ?? row.proximityPct,
          side: cond.side ?? row.side,
        };
      }
    }
  }
  return base;
}

const SELECT_CLASS =
  'rounded-md border border-input bg-background px-2 py-1 text-xs text-foreground';

export function ConditionsBuilder({
  config,
  onSubmit,
  onCancel,
  mode,
  onSaveStrategy,
  initialStrategyName,
}: {
  config: ConditionsConfig | null;
  onSubmit(config: ConditionsConfig): void;
  onCancel?(): void;
  mode: 'onboarding' | 'edit';
  /** When provided, the builder offers to save the composition as a named strategy (client-only). */
  onSaveStrategy?(name: string, config: ConditionsConfig): StrategyMutationResult;
  /** Prefilled strategy name when the palette was repopulated from a saved strategy. */
  initialStrategyName?: string;
}) {
  const t = useTranslations('scanner');
  // Palette label/description: translated when a key exists, else the palette's
  // own FR label (covers conditions added after the i18n pass — deferred).
  const plabel = (type: ConditionType): string =>
    t.has(`palette.${type}_label`) ? t(`palette.${type}_label`) : paletteEntry(type)?.label ?? type;
  const pdesc = (type: ConditionType): string =>
    t.has(`palette.${type}_desc`) ? t(`palette.${type}_desc`) : paletteEntry(type)?.description ?? '';
  const [rows, setRows] = React.useState<BuilderState>(() => initialState(config));
  const [logic, setLogic] = React.useState<ScanLogic>(config?.logic ?? 'AND');
  const [strategyName, setStrategyName] = React.useState(initialStrategyName ?? '');
  const [strategyFeedback, setStrategyFeedback] = React.useState<{
    kind: 'ok' | 'error';
    text: string;
  } | null>(null);

  const selectedCount = CONDITION_PALETTE.filter((e) => rows[e.type].selected).length;

  function patch(type: ConditionType, partial: Partial<RowState>) {
    setRows((prev) => ({ ...prev, [type]: { ...prev[type], ...partial } }));
  }

  function composeConfig(): ConditionsConfig {
    const conditions: ScanCondition[] = CONDITION_PALETTE.filter(
      (e) => rows[e.type].selected,
    ).map((e) => {
      const row = rows[e.type];
      const cond: ScanCondition = { type: e.type };
      if (e.controls.includes('direction')) cond.direction = row.direction;
      if (e.controls.includes('bars')) cond.max_bars = row.maxBars;
      if (e.controls.includes('trend')) cond.trend = row.trend;
      if (e.controls.includes('phase')) cond.phase = row.phase;
      if (e.controls.includes('volatility')) cond.volatility = row.volatility;
      if (e.controls.includes('proximity')) cond.proximity_pct = row.proximityPct;
      if (e.controls.includes('side')) cond.side = row.side;
      return cond;
    });
    return { logic, conditions };
  }

  function submit() {
    onSubmit(composeConfig());
  }

  function saveAsStrategy() {
    if (!onSaveStrategy) return;
    const result = onSaveStrategy(strategyName, composeConfig());
    if (result.ok) {
      setStrategyFeedback({
        kind: 'ok',
        text: t('saveStrategy.saved', { name: result.strategy.name }),
      });
    } else {
      setStrategyFeedback({
        kind: 'error',
        text: mutationErrorMessage(result, t) ?? t('saveStrategy.saveFailed'),
      });
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">
          {mode === 'onboarding' ? t('builder.composeTitle') : t('builder.editTitle')}
        </CardTitle>
        <p className="mt-1 text-sm text-muted-foreground">
          {t.rich('builder.intro', { b: (chunks) => <strong>{chunks}</strong> })}
        </p>
      </CardHeader>
      <CardContent className="space-y-4">
        <ul className="space-y-2">
          {CONDITION_PALETTE.map((entry) => {
            const row = rows[entry.type];
            const hasControls = entry.controls.length > 0;
            return (
              <li
                key={entry.type}
                className={cn(
                  'rounded-lg border p-3 transition-colors',
                  row.selected ? 'border-primary/60 bg-primary/5' : 'border-border/60',
                )}
              >
                <label className="flex cursor-pointer items-start gap-3">
                  <input
                    type="checkbox"
                    checked={row.selected}
                    onChange={() => patch(entry.type, { selected: !row.selected })}
                    className="mt-0.5 h-4 w-4 shrink-0 rounded border-input accent-primary"
                    aria-label={plabel(entry.type)}
                  />
                  <span className="flex flex-col">
                    <span className="text-sm font-medium text-foreground">
                      {plabel(entry.type)}
                    </span>
                    <span className="text-xs leading-snug text-muted-foreground">
                      {pdesc(entry.type)}
                    </span>
                  </span>
                </label>

                {row.selected && hasControls && (
                  <div className="mt-3 flex flex-wrap items-center gap-3 pl-7">
                    {entry.controls.includes('direction') && (
                      <label className="flex items-center gap-2 text-xs text-muted-foreground">
                        {t('builder.direction')}
                        <select
                          value={row.direction}
                          onChange={(e) =>
                            patch(entry.type, { direction: e.target.value as DirectionFilter })
                          }
                          className={SELECT_CLASS}
                        >
                          {(['any', 'bullish', 'bearish'] as DirectionFilter[]).map((d) => (
                            <option key={d} value={d}>
                              {t(`options.direction${cap(d)}`)}
                            </option>
                          ))}
                        </select>
                      </label>
                    )}
                    {entry.controls.includes('trend') && (
                      <label className="flex items-center gap-2 text-xs text-muted-foreground">
                        {t('builder.trend')}
                        <select
                          value={row.trend}
                          onChange={(e) =>
                            patch(entry.type, { trend: e.target.value as TrendChoice })
                          }
                          className={SELECT_CLASS}
                        >
                          {TREND_OPTIONS.map((o) => (
                            <option key={o.value} value={o.value}>
                              {t(`options.trend${cap(o.value)}`)}
                            </option>
                          ))}
                        </select>
                      </label>
                    )}
                    {entry.controls.includes('phase') && (
                      <label className="flex items-center gap-2 text-xs text-muted-foreground">
                        {t('builder.phase')}
                        <select
                          value={row.phase}
                          onChange={(e) =>
                            patch(entry.type, { phase: e.target.value as PhaseChoice })
                          }
                          className={SELECT_CLASS}
                        >
                          {PHASE_OPTIONS.map((o) => (
                            <option key={o.value} value={o.value}>
                              {t(`options.phase${cap(o.value)}`)}
                            </option>
                          ))}
                        </select>
                      </label>
                    )}
                    {entry.controls.includes('volatility') && (
                      <label className="flex items-center gap-2 text-xs text-muted-foreground">
                        {t('builder.volatility')}
                        <select
                          value={row.volatility}
                          onChange={(e) =>
                            patch(entry.type, { volatility: e.target.value as VolatilityChoice })
                          }
                          className={SELECT_CLASS}
                        >
                          {VOLATILITY_OPTIONS.map((o) => (
                            <option key={o.value} value={o.value}>
                              {t(`options.volatility${cap(o.value)}`)}
                            </option>
                          ))}
                        </select>
                      </label>
                    )}
                    {entry.controls.includes('bars') && (
                      <label className="flex items-center gap-2 text-xs text-muted-foreground">
                        {t('builder.inLast')}
                        <input
                          type="number"
                          min={1}
                          max={50}
                          value={row.maxBars}
                          onChange={(e) =>
                            patch(entry.type, {
                              maxBars: Math.min(50, Math.max(1, Number(e.target.value) || 1)),
                            })
                          }
                          className="w-16 rounded-md border border-input bg-background px-2 py-1 text-xs text-foreground"
                        />
                        {t('builder.lastCandles')}
                      </label>
                    )}
                    {entry.controls.includes('proximity') && (
                      <label className="flex items-center gap-2 text-xs text-muted-foreground">
                        {t('builder.proximityWithin')}
                        <input
                          type="number"
                          min={0.05}
                          max={10}
                          step={0.05}
                          value={row.proximityPct}
                          onChange={(e) =>
                            patch(entry.type, {
                              proximityPct: Math.min(
                                10,
                                Math.max(0.05, Number(e.target.value) || DEFAULT_PROXIMITY_PCT),
                              ),
                            })
                          }
                          className="w-16 rounded-md border border-input bg-background px-2 py-1 text-xs text-foreground"
                        />
                        {t('builder.proximityOfPrice')}
                      </label>
                    )}
                    {entry.controls.includes('side') && (
                      <label className="flex items-center gap-2 text-xs text-muted-foreground">
                        {t('builder.side')}
                        <select
                          value={row.side}
                          onChange={(e) =>
                            patch(entry.type, { side: e.target.value as LiquiditySideFilter })
                          }
                          className={SELECT_CLASS}
                        >
                          {LIQUIDITY_SIDE_OPTIONS.map((o) => (
                            <option key={o.value} value={o.value}>
                              {t(`options.side${cap(o.value)}`)}
                            </option>
                          ))}
                        </select>
                      </label>
                    )}
                  </div>
                )}
              </li>
            );
          })}
        </ul>

        <div className="flex flex-wrap items-center gap-3 rounded-lg border border-border/60 p-3">
          <span className="text-sm text-muted-foreground">{t('builder.combination')}</span>
          <div className="flex gap-1">
            {(['AND', 'OR'] as ScanLogic[]).map((l) => (
              <Button
                key={l}
                type="button"
                size="sm"
                variant={logic === l ? 'default' : 'outline'}
                onClick={() => setLogic(l)}
              >
                {l === 'AND' ? t('builder.logicAnd') : t('builder.logicOr')}
              </Button>
            ))}
          </div>
        </div>

        {onSaveStrategy && (
          <div className="space-y-2 rounded-lg border border-border/60 p-3">
            <p className="text-sm text-muted-foreground">
              {t('saveStrategy.prompt')}
            </p>
            <div className="flex flex-wrap items-center gap-2">
              <input
                value={strategyName}
                onChange={(e) => {
                  setStrategyName(e.target.value);
                  setStrategyFeedback(null);
                }}
                maxLength={MAX_NAME_CHARS}
                placeholder={t('saveStrategy.placeholder')}
                aria-label={t('saveStrategy.nameAria')}
                className="min-w-0 flex-1 rounded-md border border-input bg-background px-2 py-1 text-sm text-foreground"
              />
              <Button
                type="button"
                size="sm"
                variant="outline"
                onClick={saveAsStrategy}
                disabled={selectedCount === 0 || strategyName.trim().length === 0}
              >
                {t('saveStrategy.save')}
              </Button>
            </div>
            {strategyFeedback && (
              <p
                role={strategyFeedback.kind === 'error' ? 'alert' : 'status'}
                className={cn(
                  'text-xs',
                  strategyFeedback.kind === 'error'
                    ? 'text-destructive'
                    : 'text-muted-foreground',
                )}
              >
                {strategyFeedback.text}
              </p>
            )}
          </div>
        )}

        <div className="flex flex-wrap gap-2">
          <Button type="button" onClick={submit} disabled={selectedCount === 0}>
            {mode === 'onboarding' ? t('builder.submitOnboarding') : t('builder.submitEdit')}
          </Button>
          {onCancel && (
            <Button type="button" variant="ghost" onClick={onCancel}>
              {t('builder.cancel')}
            </Button>
          )}
          {selectedCount === 0 && (
            <span className="self-center text-xs text-muted-foreground">
              {t('builder.selectAtLeastOne')}
            </span>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
