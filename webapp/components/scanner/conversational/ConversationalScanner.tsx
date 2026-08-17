'use client';

import * as React from 'react';
import { useTranslations } from 'next-intl';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import type {
  ConditionsConfig,
  ConditionsScanResponse,
  ConditionType,
  ControlName,
  ScanCondition,
} from '@/lib/conditions/types';
import { fetchConditionsScan, ScanNotAvailableError } from '@/lib/conditions/api-client';
import { useLiveComboCount } from '@/lib/conditions/use-live-combo-count';
import { useSavedStrategies, type SavedStrategy } from '@/lib/conditions/strategy-store';
import { useAutoRefreshPref } from '@/lib/conditions/auto-refresh-store';
import {
  translateStrategy,
  TranslateUnavailableError,
  type TranslateAssumption,
  type TranslateResult,
  type TranslateUntranslatable,
} from '@/lib/scanner-chat/translate-client';
import { ScanResults } from '../ScanResults';
import { StrategyPanel } from '../StrategyPanel';
import { DescribePanel } from './DescribePanel';
import { EditableConditionCard } from './EditableConditionCard';
import { AddConditionPicker } from './AddConditionPicker';
import { useConditionLabels } from './labels';

type Mode = 'describe' | 'translation' | 'refusal' | 'results' | 'strategies';

interface Working {
  sourceText: string;
  conditions: ScanCondition[];
  assumptions: TranslateAssumption[];
  untranslatable: TranslateUntranslatable[];
}

/**
 * SC-2 — the conversational scanner. An ADDITIONAL entry to the scanner, never a
 * replacement: it translates a sentence into conditions of the CLOSED palette,
 * always shows what it understood / assumed / could not translate, and lets the
 * user edit everything before anything runs. Results reuse the SC-1 ScanResults
 * (three non-maskable blocks, no sort) and strategies reuse the SC-1 StrategyPanel
 * (device-local, counts re-evaluated on open, stale counts flagged).
 */
export function ConversationalScanner({ locale }: { locale: string }) {
  const t = useTranslations('scannerChat');
  const { conditionLabel, optionLabel } = useConditionLabels();
  const saved = useSavedStrategies();
  const { enabled: autoRefresh, setEnabled: setAutoRefresh } = useAutoRefreshPref();

  const [mode, setMode] = React.useState<Mode>('describe');
  const [text, setText] = React.useState('');
  const [translating, setTranslating] = React.useState(false);
  const [inlineError, setInlineError] = React.useState<string | null>(null);
  const [working, setWorking] = React.useState<Working | null>(null);
  const [refusalKind, setRefusalKind] = React.useState<string | null>(null);

  const [response, setResponse] = React.useState<ConditionsScanResponse | null>(null);
  const [scanning, setScanning] = React.useState(false);
  const [saveName, setSaveName] = React.useState('');
  const [saveFeedback, setSaveFeedback] = React.useState<string | null>(null);

  const workingConfig: ConditionsConfig | null = working
    ? { logic: 'AND', conditions: working.conditions }
    : null;
  const { live } = useLiveComboCount(mode === 'translation' ? workingConfig : null);

  // ── Translate ──────────────────────────────────────────────────────────────
  const runTranslate = React.useCallback(
    async (source: string) => {
      const trimmed = source.trim();
      if (!trimmed) return;
      setTranslating(true);
      setInlineError(null);
      let result: TranslateResult;
      try {
        result = await translateStrategy(trimmed, locale);
      } catch (err) {
        setTranslating(false);
        if (err instanceof TranslateUnavailableError) {
          setInlineError(t('errors.unavailable'));
        } else {
          setInlineError(t('errors.failed'));
        }
        return;
      }
      setTranslating(false);

      if (result.outcome === 'refused' && result.refusal) {
        setRefusalKind(result.refusal.kind);
        setMode('refusal');
        return;
      }
      if (result.outcome === 'error') {
        setInlineError(t('errors.failed'));
        return;
      }
      if (result.outcome === 'empty') {
        setInlineError(t('errors.empty'));
        return;
      }
      setWorking({
        sourceText: trimmed,
        conditions: result.conditions,
        assumptions: result.assumptions,
        untranslatable: result.untranslatable,
      });
      setSaveName('');
      setSaveFeedback(null);
      setMode('translation');
    },
    [locale, t],
  );

  // ── Edit working conditions ─────────────────────────────────────────────────
  const updateCondition = React.useCallback((index: number, next: ScanCondition) => {
    setWorking((prev) => {
      if (!prev) return prev;
      const conditions = prev.conditions.slice();
      conditions[index] = next;
      return { ...prev, conditions };
    });
  }, []);

  const removeCondition = React.useCallback((index: number) => {
    setWorking((prev) => {
      if (!prev) return prev;
      const conditions = prev.conditions.filter((_, i) => i !== index);
      return { ...prev, conditions };
    });
  }, []);

  const addCondition = React.useCallback((condition: ScanCondition) => {
    setWorking((prev) => (prev ? { ...prev, conditions: [...prev.conditions, condition] } : prev));
  }, []);

  // ── Run the scan (state 5) ──────────────────────────────────────────────────
  const runScan = React.useCallback(async () => {
    if (!workingConfig || workingConfig.conditions.length === 0) return;
    setScanning(true);
    setInlineError(null);
    try {
      const res = await fetchConditionsScan(workingConfig);
      setResponse(res);
      setMode('results');
    } catch (err) {
      const message =
        err instanceof ScanNotAvailableError ? t('errors.scanUnavailable') : t('errors.scanFailed');
      setInlineError(message);
    } finally {
      setScanning(false);
    }
  }, [workingConfig, t]);

  const refreshScan = React.useCallback(async () => {
    if (!workingConfig) return;
    setScanning(true);
    try {
      const res = await fetchConditionsScan(workingConfig);
      setResponse(res);
    } catch {
      /* keep the last good results; the meta line signals staleness */
    } finally {
      setScanning(false);
    }
  }, [workingConfig]);

  const saveStrategy = React.useCallback(() => {
    if (!workingConfig) return;
    const result = saved.saveStrategy(saveName, workingConfig);
    if (result.ok) {
      setSaveFeedback(t('save.done', { name: result.strategy.name }));
      setSaveName('');
    } else {
      setSaveFeedback(t(`save.errors.${result.error}`));
    }
  }, [workingConfig, saveName, saved, t]);

  const loadStrategy = React.useCallback(
    (strategy: SavedStrategy) => {
      saved.markUsed(strategy.id);
      setWorking({
        sourceText: strategy.name,
        conditions: strategy.config.conditions,
        assumptions: [],
        untranslatable: [],
      });
      setText('');
      setSaveName(strategy.name);
      setMode('translation');
    },
    [saved],
  );

  // Map condition type → set of controls M.I.A assumed (inline card flag).
  const assumedByType = React.useMemo(() => {
    const map = new Map<ConditionType, Set<ControlName>>();
    for (const a of working?.assumptions ?? []) {
      const set = map.get(a.condition_type) ?? new Set<ControlName>();
      set.add(a.control);
      map.set(a.condition_type, set);
    }
    return map;
  }, [working]);

  // ── Render per mode ──────────────────────────────────────────────────────────
  if (mode === 'describe') {
    return (
      <DescribePanel
        text={text}
        onTextChange={setText}
        onTranslate={() => runTranslate(text)}
        onOpenStrategies={() => setMode('strategies')}
        isTranslating={translating}
        inlineError={inlineError}
        locale={locale}
      />
    );
  }

  if (mode === 'refusal') {
    return (
      <RefusalPanel
        kind={refusalKind}
        sourceText={working?.sourceText ?? text}
        onReformulate={() => {
          setMode('describe');
        }}
        onExample={(example) => {
          setText(example);
          setMode('describe');
        }}
      />
    );
  }

  if (mode === 'results' && response && workingConfig) {
    return (
      <div className="space-y-4">
        <BackBar label={t('results.back')} onBack={() => setMode('translation')} />
        <ScanResults
          response={response}
          config={workingConfig}
          locale={locale}
          onEdit={() => setMode('translation')}
          onRefresh={refreshScan}
          isRefreshing={scanning}
          autoRefreshEnabled={autoRefresh}
          onToggleAutoRefresh={setAutoRefresh}
        />
      </div>
    );
  }

  if (mode === 'strategies') {
    return (
      <div className="space-y-4">
        <BackBar label={t('strategies.back')} onBack={() => setMode('describe')} />
        <div>
          <h1 className="fs-title font-semibold tracking-tight text-foreground">{t('strategies.title')}</h1>
          <p className="mt-1 fs-secondary text-muted-foreground">{t('strategies.subtitle')}</p>
        </div>
        {saved.strategies.length === 0 ? (
          <p className="rounded-xl border border-border bg-card p-4 fs-secondary text-muted-foreground">
            {t('strategies.empty')}
          </p>
        ) : (
          <StrategyPanel
            strategies={saved.strategies}
            locale={locale}
            onLoad={loadStrategy}
            onRename={saved.renameStrategy}
            onDuplicate={saved.duplicateStrategy}
            onDelete={saved.deleteStrategy}
            onExport={saved.exportText}
            onImport={saved.importText}
          />
        )}
      </div>
    );
  }

  // mode === 'translation'
  if (!working) return null;
  const hasConditions = working.conditions.length > 0;
  const isPartial = working.untranslatable.length > 0;
  const presentTypes = new Set<ConditionType>(working.conditions.map((c) => c.type));

  return (
    <div className="space-y-5">
      <div>
        <h1 className="fs-title font-semibold tracking-tight text-foreground">
          {hasConditions ? (isPartial ? t('translation.partialTitle') : t('translation.title')) : t('translation.noneTitle')}
        </h1>
        <p className="mt-1 fs-secondary text-muted-foreground">
          {hasConditions ? t('translation.subtitle') : t('translation.noneSubtitle')}
        </p>
      </div>

      {/* The original request — recalled and editable. */}
      <div className="rounded-2xl border border-border bg-card p-4">
        <div className="flex items-start gap-3">
          <div className="flex-1">
            <div className="mb-1 fs-legal text-muted-foreground">{t('translation.yourRequest')}</div>
            <div className="fs-body leading-relaxed text-foreground">“{working.sourceText}”</div>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={() => {
              setText(working.sourceText);
              setMode('describe');
            }}
          >
            {t('translation.modify')}
          </Button>
        </div>
      </div>

      {hasConditions && (
        <div>
          <div className="mb-2 fs-label uppercase tracking-widest text-muted-foreground">
            {t('translation.conditionsLabel', { count: working.conditions.length })}
          </div>
          <div className="grid gap-2 sm:grid-cols-2">
            {working.conditions.map((condition, index) => (
              <EditableConditionCard
                key={`${condition.type}-${index}`}
                condition={condition}
                index={index}
                assumedControls={assumedByType.get(condition.type)}
                onChange={updateCondition}
                onRemove={removeCondition}
              />
            ))}
            <AddConditionPicker present={presentTypes} onAdd={addCondition} />
          </div>
        </div>
      )}

      {/* « Ce que j'ai supposé » — mandatory, never invisible. */}
      {working.assumptions.length > 0 && (
        <div
          data-testid="assumptions-block"
          className="rounded-r-xl border border-amber-500/40 border-l-2 border-l-amber-500 bg-amber-500/5 p-3.5"
        >
          <h4 className="mb-1.5 fs-secondary font-semibold text-amber-600">{t('assumptions.title')}</h4>
          <ul className="space-y-1">
            {working.assumptions.map((a, i) => (
              <li key={i} className="fs-secondary leading-relaxed text-amber-700/90 dark:text-amber-200/80">
                {t('assumptions.item', {
                  phrase: a.source_phrase ?? t('assumptions.aVagueWord'),
                  condition: conditionLabel(a.condition_type),
                  value: a.value ? optionLabel(a.control, a.value) : '—',
                })}
              </li>
            ))}
          </ul>
          <p className="mt-1.5 fs-label text-amber-700/80 dark:text-amber-200/70">{t('assumptions.note')}</p>
        </div>
      )}

      {/* « Ce que je n'ai pas pu traduire » — named precisely, never substituted. */}
      {isPartial && (
        <div
          data-testid="untranslatable-block"
          className="rounded-r-xl border border-amber-500/40 border-l-2 border-l-amber-500 bg-amber-500/5 p-3.5"
        >
          <h4 className="mb-1.5 fs-secondary font-semibold text-amber-600">
            {hasConditions ? t('untranslatable.titlePartial') : t('untranslatable.titleNone')}
          </h4>
          <ul className="space-y-1.5">
            {working.untranslatable.map((u, i) => (
              <li key={i} className="fs-secondary leading-relaxed text-amber-700/90 dark:text-amber-200/80">
                {u.fragment ? <b className="font-medium text-amber-600">“{u.fragment}”</b> : null}{' '}
                {categoryMessage(u.category, t)}
              </li>
            ))}
          </ul>
          <p className="mt-1.5 fs-label text-amber-700/80 dark:text-amber-200/70">{t('untranslatable.honesty')}</p>
        </div>
      )}

      {/* Live count + actions, OR a reformulate path when nothing was understood. */}
      {hasConditions ? (
        <div className="sticky bottom-0 flex flex-wrap items-center gap-4 rounded-2xl border border-border bg-card/95 p-4 backdrop-blur">
          <span data-testid="live-count" className="fs-title font-semibold tracking-tight text-foreground">
            {live.status === 'ready' ? live.count : '—'}
          </span>
          <span className="fs-secondary leading-tight text-muted-foreground">
            {t('translation.comboCaption', { conditions: working.conditions.length })}
            <br />
            {live.status === 'ready'
              ? t('translation.comboScanned', { scanned: live.scanned })
              : t('translation.comboComputing')}
          </span>
          <div className="ml-auto flex flex-wrap items-center gap-2">
            <div className="flex items-center gap-1">
              <input
                data-testid="save-name"
                value={saveName}
                onChange={(e) => setSaveName(e.target.value)}
                placeholder={t('save.namePlaceholder')}
                aria-label={t('save.namePlaceholder')}
                className="w-40 rounded-md border border-input bg-background px-2 py-1.5 text-sm text-foreground"
              />
              <Button variant="outline" onClick={saveStrategy}>
                {t('save.cta')}
              </Button>
            </div>
            <Button data-testid="see-results" onClick={runScan} disabled={scanning}>
              {scanning ? t('translation.scanning') : t('translation.seeResults')}
            </Button>
          </div>
          {saveFeedback && (
            <p data-testid="save-feedback" className="w-full fs-label text-muted-foreground">
              {saveFeedback}
            </p>
          )}
          {inlineError && (
            <p role="alert" className="w-full fs-label text-destructive">
              {inlineError}
            </p>
          )}
        </div>
      ) : (
        <div className="flex flex-wrap gap-2">
          <Button onClick={() => setMode('describe')}>{t('translation.reformulate')}</Button>
        </div>
      )}
    </div>
  );
}

function categoryMessage(category: string, t: ReturnType<typeof useTranslations>): string {
  const key = `untranslatable.categories.${category}`;
  return t.has(key) ? t(key) : t('untranslatable.categories.other');
}

function BackBar({ label, onBack }: { label: string; onBack(): void }) {
  return (
    <button
      type="button"
      onClick={onBack}
      className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground"
    >
      ← {label}
    </button>
  );
}

/**
 * State 4 — refusal. Firm and non-culpabilising: it is a design constraint, not
 * a gap. Explains that the product ranks nothing and offers to reformulate with
 * two clickable examples.
 */
function RefusalPanel({
  kind,
  sourceText,
  onReformulate,
  onExample,
}: {
  kind: string | null;
  sourceText: string;
  onReformulate(): void;
  onExample(example: string): void;
}) {
  const t = useTranslations('scannerChat');
  const safeKind = kind && ['ranking', 'prediction', 'recommendation'].includes(kind) ? kind : 'ranking';
  const examples = [t('refusal.examples.0'), t('refusal.examples.1')];

  return (
    <div className="space-y-5">
      <div>
        <h1 className="fs-title font-semibold tracking-tight text-foreground">{t('refusal.pageTitle')}</h1>
        <p className="mt-1 fs-secondary text-muted-foreground">{t('refusal.pageSubtitle')}</p>
      </div>

      <div className="rounded-2xl border border-border bg-card p-4">
        <div className="mb-1 fs-legal text-muted-foreground">{t('translation.yourRequest')}</div>
        <div className="fs-body leading-relaxed text-foreground">“{sourceText}”</div>
      </div>

      <div
        data-testid="refusal-block"
        className="rounded-r-xl border border-amber-500/40 border-l-2 border-l-amber-500 bg-amber-500/5 p-4"
      >
        <h4 className="mb-2 fs-body font-semibold text-amber-600">{t(`refusal.${safeKind}.title`)}</h4>
        <p className="mb-2 text-[13.5px] leading-relaxed text-amber-700/90 dark:text-amber-200/80">
          {t(`refusal.${safeKind}.body`)}
        </p>
        <p className="fs-secondary leading-relaxed text-amber-700/90 dark:text-amber-200/80">{t('refusal.invite')}</p>
        <div className="mt-3 flex flex-wrap gap-2">
          {examples.map((example, i) => (
            <button
              key={i}
              type="button"
              data-testid="refusal-example"
              onClick={() => onExample(example)}
              className="rounded-lg border border-border bg-muted/50 px-3 py-2 text-left fs-secondary text-muted-foreground transition hover:border-primary/50 hover:text-foreground"
            >
              {example}
            </button>
          ))}
        </div>
      </div>

      <div className="flex gap-2">
        <Button variant="outline" onClick={onReformulate}>
          {t('refusal.reformulate')}
        </Button>
      </div>
      <p className="fs-legal leading-relaxed text-muted-foreground">{t('refusal.footer')}</p>
    </div>
  );
}
