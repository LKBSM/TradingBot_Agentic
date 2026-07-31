'use client';

import * as React from 'react';
import { useTranslations } from 'next-intl';
import { ChevronDown, Copy, Check, Info } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { cn } from '@/lib/utils';
import {
  CONDITION_PALETTE,
  DEFAULT_OPEN_FAMILIES,
  groupByFamily,
  optionLabelFallback,
  paletteEntry,
} from '@/lib/conditions/palette';
import type {
  ConditionType,
  ConditionsConfig,
  ControlDescriptor,
  ControlName,
  Family,
  PaletteEntry,
  ScanCondition,
  ScanLogic,
} from '@/lib/conditions/types';
import { MAX_NAME_CHARS, type StrategyMutationResult } from '@/lib/conditions/strategy-store';
import { mutationErrorMessage } from './StrategyPanel';
import { Segmented } from './Segmented';

interface RowState {
  selected: boolean;
  values: Record<string, string | number>;
}

type BuilderState = Record<ConditionType, RowState>;

function defaultRow(entry: PaletteEntry): RowState {
  const values: Record<string, string | number> = {};
  for (const c of entry.controls) values[c.name] = c.default;
  return { selected: false, values };
}

function initialState(config: ConditionsConfig | null): BuilderState {
  const base = {} as BuilderState;
  for (const entry of CONDITION_PALETTE) base[entry.type] = defaultRow(entry);
  if (config) {
    for (const cond of config.conditions) {
      const entry = paletteEntry(cond.type);
      const row = base[cond.type];
      if (!entry || !row) continue;
      const values = { ...row.values };
      for (const c of entry.controls) {
        const v = (cond as unknown as Record<string, unknown>)[c.name];
        if (v !== undefined && (typeof v === 'string' || typeof v === 'number')) values[c.name] = v;
      }
      base[cond.type] = { selected: true, values };
    }
  }
  return base;
}

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
  onSaveStrategy?(name: string, config: ConditionsConfig): StrategyMutationResult;
  initialStrategyName?: string;
}) {
  const t = useTranslations('scanner');

  const plabel = (type: ConditionType): string =>
    t.has(`palette.${type}_label`) ? t(`palette.${type}_label`) : paletteEntry(type)?.label ?? type;
  const pdesc = (type: ConditionType): string =>
    t.has(`palette.${type}_desc`) ? t(`palette.${type}_desc`) : paletteEntry(type)?.description ?? '';
  const optLabel = (control: ControlName, value: string | number): string => {
    const key = `opt.${control}.${value}`;
    return t.has(key) ? t(key) : optionLabelFallback(control, value);
  };

  const [rows, setRows] = React.useState<BuilderState>(() => initialState(config));
  const [logic, setLogic] = React.useState<ScanLogic>(config?.logic ?? 'AND');
  const [open, setOpen] = React.useState<Set<Family>>(() => new Set(DEFAULT_OPEN_FAMILIES));
  const [concept, setConcept] = React.useState<ConditionType | null>(null);
  const [strategyName, setStrategyName] = React.useState(initialStrategyName ?? '');
  const [strategyFeedback, setStrategyFeedback] = React.useState<{ kind: 'ok' | 'error'; text: string } | null>(null);
  const [copied, setCopied] = React.useState(false);

  const groups = React.useMemo(() => groupByFamily(CONDITION_PALETTE), []);
  const selectedCount = CONDITION_PALETTE.filter((e) => rows[e.type].selected).length;
  const activeInFamily = (family: Family): number =>
    CONDITION_PALETTE.filter((e) => e.family === family && rows[e.type].selected).length;

  function toggleFamily(family: Family) {
    setOpen((prev) => {
      const next = new Set(prev);
      if (next.has(family)) next.delete(family);
      else next.add(family);
      return next;
    });
  }

  function patch(type: ConditionType, partial: Partial<RowState>) {
    setRows((prev) => ({ ...prev, [type]: { ...prev[type], ...partial } }));
  }
  function setValue(type: ConditionType, name: string, value: string | number) {
    setRows((prev) => ({
      ...prev,
      [type]: { ...prev[type], values: { ...prev[type].values, [name]: value } },
    }));
  }

  function composeConfig(): ConditionsConfig {
    const conditions: ScanCondition[] = CONDITION_PALETTE.filter((e) => rows[e.type].selected).map((e) => {
      const cond = { type: e.type } as ScanCondition;
      const target = cond as unknown as Record<string, unknown>;
      for (const c of e.controls) target[c.name] = rows[e.type].values[c.name];
      return cond;
    });
    return { logic, conditions };
  }

  // Read-only recap sentence, present/passé composé, composed from the ticked
  // conditions. Each condition's own label already reads at the present or past
  // — we only append the chosen values and join with the logic connector.
  const recap = React.useMemo(() => {
    const parts = CONDITION_PALETTE.filter((e) => rows[e.type].selected).map((e) => {
      const vals = e.controls
        .map((c) => optLabel(c.name, rows[e.type].values[c.name] ?? c.default))
        .join(', ');
      return vals ? `${plabel(e.type)} (${vals})` : plabel(e.type);
    });
    if (parts.length === 0) return '';
    const joiner = logic === 'AND' ? ` ${t('recap.and')} ` : ` ${t('recap.or')} `;
    return parts.join(joiner) + '.';
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rows, logic]);

  async function copyRecap() {
    try {
      await navigator.clipboard.writeText(recap);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard unavailable — no-op */
    }
  }

  function saveAsStrategy() {
    if (!onSaveStrategy) return;
    const result = onSaveStrategy(strategyName, composeConfig());
    setStrategyFeedback(
      result.ok
        ? { kind: 'ok', text: t('saveStrategy.saved', { name: result.strategy.name }) }
        : { kind: 'error', text: mutationErrorMessage(result, t) ?? t('saveStrategy.saveFailed') },
    );
  }

  function renderControl(entry: PaletteEntry, control: ControlDescriptor, disabled: boolean) {
    const options = control.values.map((v) => ({ value: v, label: optLabel(control.name, v) }));
    const current = rows[entry.type].values[control.name] ?? control.default;
    return (
      <div key={control.name} className="flex flex-col gap-1">
        <span className="text-[11px] uppercase tracking-wide text-muted-foreground">
          {t.has(`control.${control.name}`) ? t(`control.${control.name}`) : control.name}
        </span>
        <Segmented
          options={options}
          value={current}
          onChange={(v) => setValue(entry.type, control.name, v)}
          disabled={disabled}
          ariaLabel={`${plabel(entry.type)} — ${control.name}`}
        />
      </div>
    );
  }

  return (
    <div className="space-y-4 pb-24">
      <Card>
        <CardHeader>
          <CardTitle className="text-base">
            {mode === 'onboarding' ? t('builder.composeTitle') : t('builder.editTitle')}
          </CardTitle>
          <p className="mt-1 text-sm text-muted-foreground">
            {t.rich('builder.intro', { b: (chunks) => <strong>{chunks}</strong> })}
          </p>
        </CardHeader>
        <CardContent className="space-y-3">
          {groups.map(({ family, entries }, familyIndex) => {
            const isOpen = open.has(family);
            const count = activeInFamily(family);
            return (
              <section key={family} className="rounded-lg border border-border/60">
                <button
                  type="button"
                  onClick={() => toggleFamily(family)}
                  aria-expanded={isOpen}
                  className="flex w-full items-center justify-between gap-2 px-3 py-2 text-left"
                >
                  <span className="flex items-baseline gap-2">
                    <span className="text-xs text-muted-foreground">{familyIndex + 1}.</span>
                    <span className="text-sm font-semibold text-foreground">{t(`family.${family}.title`)}</span>
                    <span className="text-xs text-muted-foreground">{t(`family.${family}.desc`)}</span>
                  </span>
                  <span className="flex items-center gap-2">
                    <span
                      className={cn(
                        'rounded-full px-2 py-0.5 text-[11px]',
                        count > 0 ? 'bg-foreground/10 text-foreground' : 'text-muted-foreground',
                      )}
                    >
                      {t('builder.activeInFamily', { count })}
                    </span>
                    <ChevronDown className={cn('h-4 w-4 transition-transform', isOpen && 'rotate-180')} />
                  </span>
                </button>

                {isOpen && (
                  <ul className="space-y-2 p-2">
                    {entries.map((entry) => {
                      const row = rows[entry.type];
                      const showConcept = concept === entry.type;
                      return (
                        <li
                          key={entry.type}
                          className={cn(
                            'rounded-lg border p-3 transition-colors',
                            row.selected ? 'border-foreground/40 bg-foreground/5' : 'border-border/50',
                          )}
                        >
                          <div className="flex items-start justify-between gap-2">
                            <label className="flex cursor-pointer items-start gap-3">
                              <input
                                type="checkbox"
                                checked={row.selected}
                                onChange={() => patch(entry.type, { selected: !row.selected })}
                                className="mt-0.5 h-4 w-4 shrink-0 rounded border-input accent-foreground"
                                aria-label={plabel(entry.type)}
                              />
                              <span className="text-sm font-medium text-foreground">{plabel(entry.type)}</span>
                            </label>
                            <button
                              type="button"
                              onClick={() => setConcept(showConcept ? null : entry.type)}
                              aria-expanded={showConcept}
                              aria-label={t('builder.conceptAria', { label: plabel(entry.type) })}
                              className="shrink-0 rounded p-1 text-muted-foreground hover:text-foreground"
                            >
                              <Info className="h-4 w-4" />
                            </button>
                          </div>

                          {showConcept && (
                            <dl className="mt-2 space-y-1 rounded-md border border-border/50 bg-background/40 p-2 text-xs">
                              <div>
                                <dt className="font-semibold text-foreground">{t('concept.whatLabel')}</dt>
                                <dd className="text-muted-foreground">
                                  {t.has(`concept.${entry.type}.what`) ? t(`concept.${entry.type}.what`) : pdesc(entry.type)}
                                </dd>
                              </div>
                              {t.has(`concept.${entry.type}.smc`) && (
                                <div>
                                  <dt className="font-semibold text-foreground">{t('concept.smcLabel')}</dt>
                                  <dd className="text-muted-foreground">{t(`concept.${entry.type}.smc`)}</dd>
                                </div>
                              )}
                              <div>
                                <dt className="font-semibold text-foreground">{t('concept.notLabel')}</dt>
                                <dd className="text-muted-foreground">
                                  {t.has(`concept.${entry.type}.not`) ? t(`concept.${entry.type}.not`) : t('concept.notDefault')}
                                </dd>
                              </div>
                            </dl>
                          )}

                          {entry.controls.length > 0 && (
                            <div className="mt-3 flex flex-wrap gap-4 pl-7">
                              {entry.controls.map((c) => renderControl(entry, c, !row.selected))}
                            </div>
                          )}
                        </li>
                      );
                    })}
                  </ul>
                )}
              </section>
            );
          })}

          <div className="flex flex-wrap items-center gap-3 rounded-lg border border-border/60 p-3">
            <span className="text-sm text-muted-foreground">{t('builder.combination')}</span>
            <Segmented
              options={[
                { value: 'AND' as ScanLogic, label: t('builder.logicAnd') },
                { value: 'OR' as ScanLogic, label: t('builder.logicOr') },
              ]}
              value={logic}
              onChange={(v) => setLogic(v as ScanLogic)}
              ariaLabel={t('builder.combination')}
            />
          </div>

          {recap && (
            <div className="space-y-2 rounded-lg border border-border/60 bg-background/40 p-3">
              <div className="flex items-center justify-between gap-2">
                <span className="text-xs uppercase tracking-wide text-muted-foreground">{t('recap.title')}</span>
                <Button type="button" size="sm" variant="ghost" onClick={copyRecap} className="h-7 gap-1">
                  {copied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
                  {copied ? t('recap.copied') : t('recap.copy')}
                </Button>
              </div>
              <p className="text-sm text-foreground">{recap}</p>
            </div>
          )}

          {onSaveStrategy && (
            <div className="space-y-2 rounded-lg border border-border/60 p-3">
              <p className="text-sm text-muted-foreground">{t('saveStrategy.prompt')}</p>
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
                  className={cn('text-xs', strategyFeedback.kind === 'error' ? 'text-destructive' : 'text-muted-foreground')}
                >
                  {strategyFeedback.text}
                </p>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Sticky action bar — selected count + go to results. */}
      <div className="fixed inset-x-0 bottom-0 z-10 border-t border-border/60 bg-background/95 backdrop-blur">
        <div className="mx-auto flex max-w-5xl flex-wrap items-center justify-between gap-3 px-4 py-3">
          <span className="text-sm text-muted-foreground">
            {t('builder.selectedCount', { count: selectedCount })}
          </span>
          <div className="flex flex-wrap gap-2">
            {onCancel && (
              <Button type="button" variant="ghost" onClick={onCancel}>
                {t('builder.cancel')}
              </Button>
            )}
            <Button type="button" onClick={() => onSubmit(composeConfig())} disabled={selectedCount === 0}>
              {mode === 'onboarding' ? t('builder.submitOnboarding') : t('builder.submitEdit')}
            </Button>
          </div>
        </div>
        {selectedCount === 0 && (
          <p className="px-4 pb-2 text-center text-xs text-muted-foreground">
            {t('builder.selectAtLeastOne')} {t('builder.zeroNote')}
          </p>
        )}
      </div>
    </div>
  );
}
