'use client';

import * as React from 'react';
import { useTranslations } from 'next-intl';
import { Button } from '@/components/ui/button';
import type {
  ConditionsConfig,
  ConditionType,
  ControlName,
  ScanCondition,
} from '@/lib/conditions/types';
import { useLiveScan } from '@/lib/conditions/use-live-scan';
import { useAutoRefreshPref } from '@/lib/conditions/auto-refresh-store';
import { useCandleCloseRefresh } from '@/lib/conditions/use-candle-close-refresh';
import { useSavedStrategies, type SavedStrategy } from '@/lib/conditions/strategy-store';
import { useLiveTranslation } from '@/lib/scanner-chat/use-live-translation';
import {
  EMPTY_MANUAL_STATE,
  conditionKey,
  purgeRemovals,
  reconcile,
  type ManualState,
} from '@/lib/scanner-chat/reconciliation';
import { ScanResults } from '../ScanResults';
import { StrategyPanel } from '../StrategyPanel';
import { DescribePanel } from './DescribePanel';
import { EditableConditionCard } from './EditableConditionCard';
import { AddConditionPicker } from './AddConditionPicker';
import { useConditionLabels } from './labels';

type Mode = 'compose' | 'strategies';

/**
 * SC-2/SC-4 — the conversational scanner. An ADDITIONAL entry to the scanner,
 * never a replacement: it translates a sentence into conditions of the CLOSED
 * palette, always shows what it understood / assumed / could not translate, and
 * lets the user edit everything. Results reuse the SC-1 `ScanResults` (three
 * non-maskable blocks, no sort) and strategies reuse the SC-1 `StrategyPanel`.
 *
 * SC-4 replaced the click-through (describe → verify → results, three screens
 * that each REPLACED the previous one) with a single composition surface: the
 * field, the reading and the results are on screen together, and the reading
 * advances as the sentence is written. Two consequences drive everything below.
 *
 *   · The field is never unmounted. That is why a refusal now renders INLINE
 *     (`RefusalNotice`) instead of taking over the page: refusing mid-sentence
 *     by unmounting the textarea would destroy text the user was still writing.
 *     The refusal is no softer for being smaller — it still translates nothing.
 *
 *   · The automatic reading is a PROPOSAL, never an authority. Removals, edits
 *     and additions made by hand survive every re-translation, and a removal is
 *     released only when the words behind it leave the text. That logic lives in
 *     `reconciliation.ts`, deliberately pure and tested on its own.
 *
 * The results are unchanged and un-special: the same `ScanResults` the manual
 * palette renders, so « ce qui va à l'encontre » is exactly as visible and
 * exactly as non-collapsible here as it is there. No second results pipeline,
 * and no click between a reading and its results.
 */
export function ConversationalScanner({ locale }: { locale: string }) {
  const t = useTranslations('scannerChat');
  const { conditionLabel, optionLabel } = useConditionLabels();
  const saved = useSavedStrategies();
  const { enabled: autoRefresh, setEnabled: setAutoRefresh } = useAutoRefreshPref();

  const [mode, setMode] = React.useState<Mode>('compose');
  const [text, setText] = React.useState('');
  const [manual, setManual] = React.useState<ManualState>(EMPTY_MANUAL_STATE);
  const [saveName, setSaveName] = React.useState('');
  const [saveFeedback, setSaveFeedback] = React.useState<string | null>(null);
  const inputRef = React.useRef<HTMLTextAreaElement | null>(null);

  const { live, translateNow, reset: resetLive } = useLiveTranslation(text, locale);

  // A removal lives only as long as the words that produced it. Derived at
  // render from the CURRENT text (not the translated one) so the release is
  // immediate: the instant the fragment is edited away, the condition is free
  // to come back on the next reading.
  const activeRemovals = React.useMemo(
    () => purgeRemovals(manual.removed, text),
    [manual.removed, text],
  );

  // Emptying the field is a fresh start for everything DERIVED from text:
  // removals and edits are filed under proposals that no longer exist. Additions
  // are NOT cleared — they belong to the user and to no sentence, which is also
  // what makes `loadStrategy` (empty text + `added`) survive this effect.
  const textIsEmpty = text.trim().length === 0;
  React.useEffect(() => {
    if (!textIsEmpty) return;
    setManual((prev) =>
      prev.removed.length === 0 && Object.keys(prev.edited).length === 0
        ? prev
        : { ...prev, removed: [], edited: {} },
    );
  }, [textIsEmpty]);

  const reconciled = React.useMemo(
    () => reconcile(live.conditions, live.sources, { ...manual, removed: activeRemovals }),
    [live.conditions, live.sources, manual, activeRemovals],
  );

  const conditions = React.useMemo(() => reconciled.map((r) => r.condition), [reconciled]);
  const workingConfig: ConditionsConfig | null =
    conditions.length > 0 ? { logic: 'AND', conditions } : null;

  const { scan, refresh: refreshScan } = useLiveScan(workingConfig);

  // Auto-refresh on candle close, aligned with the manual palette. Before SC-4
  // this page passed the preference to `ScanResults` (which renders the toggle)
  // but never armed the timer, so the control was decorative here. The live flow
  // re-scans on every change to the reading, but NOT when a candle closes under
  // an unchanged reading — which is exactly what this covers.
  const scannedTimeframes = React.useMemo(
    () => Array.from(new Set((scan.response?.matches ?? []).map((m) => m.timeframe))),
    [scan.response],
  );
  useCandleCloseRefresh({
    timeframes: scannedTimeframes,
    enabled: autoRefresh && !!scan.response,
    isScanning: scan.status === 'loading',
    onRefresh: refreshScan,
  });

  // ── Manual interventions ────────────────────────────────────────────────────
  const removeCondition = React.useCallback(
    (index: number) => {
      const target = reconciled[index];
      if (!target) return;
      setManual((prev) => {
        if (target.userAdded) {
          return {
            ...prev,
            added: prev.added.filter((c) => conditionKey(c) !== target.proposalKey),
          };
        }
        // File the removal under M.I.A's own proposal, with the fragment that
        // produced it, and drop any edit of it — an edited-then-removed chip
        // must not resurface wearing its old edit.
        const { [target.proposalKey]: _dropped, ...edited } = prev.edited;
        return {
          ...prev,
          edited,
          removed: [...prev.removed, { key: target.proposalKey, phrase: target.phrase }],
        };
      });
    },
    [reconciled],
  );

  const updateCondition = React.useCallback(
    (index: number, next: ScanCondition) => {
      const target = reconciled[index];
      if (!target) return;
      setManual((prev) => {
        if (target.userAdded) {
          return {
            ...prev,
            added: prev.added.map((c) => (conditionKey(c) === target.proposalKey ? next : c)),
          };
        }
        // Keyed on the PROPOSAL, so the next reading re-applies the same edit
        // instead of silently resetting the value the user just corrected.
        return { ...prev, edited: { ...prev.edited, [target.proposalKey]: next } };
      });
    },
    [reconciled],
  );

  const addCondition = React.useCallback((condition: ScanCondition) => {
    setManual((prev) => {
      const key = conditionKey(condition);
      if (prev.added.some((c) => conditionKey(c) === key)) return prev;
      // Adding back something previously removed must actually add it back.
      return {
        ...prev,
        removed: prev.removed.filter((r) => r.key !== key),
        added: [...prev.added, condition],
      };
    });
  }, []);

  const clearAll = React.useCallback(() => {
    setText('');
    setManual(EMPTY_MANUAL_STATE);
    setSaveFeedback(null);
    resetLive();
    inputRef.current?.focus();
  }, [resetLive]);

  // ── Strategies ──────────────────────────────────────────────────────────────
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
      // A loaded strategy belongs to the user, not to any sentence — it lands in
      // `added`, where no re-translation can touch it.
      setText('');
      resetLive();
      setManual({ removed: [], edited: {}, added: strategy.config.conditions });
      setSaveName(strategy.name);
      setMode('compose');
    },
    [saved, resetLive],
  );

  // Map condition type → controls M.I.A assumed a value for (inline card flag).
  const assumedByType = React.useMemo(() => {
    const map = new Map<ConditionType, Set<ControlName>>();
    for (const a of live.assumptions ?? []) {
      const set = map.get(a.condition_type) ?? new Set<ControlName>();
      set.add(a.control);
      map.set(a.condition_type, set);
    }
    return map;
  }, [live.assumptions]);

  if (mode === 'strategies') {
    return (
      <div className="space-y-4">
        <BackBar label={t('strategies.back')} onBack={() => setMode('compose')} />
        <div>
          <h1 className="fs-title font-semibold tracking-tight text-foreground">
            {t('strategies.title')}
          </h1>
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

  const hasConditions = conditions.length > 0;
  const presentTypes = new Set<ConditionType>(conditions.map((c) => c.type));
  const untranslatable = live.untranslatable ?? [];
  const assumptions = live.assumptions ?? [];
  // Withheld, not absent: the sentence is still moving, so the honesty blocks
  // have not been read through yet. Saying so beats showing them about a
  // half-written clause, and beats showing a stale copy of the previous read.
  const honestyPending = live.status !== 'idle' && live.assumptions === null && !live.refusal;

  return (
    <div className="lg:grid lg:grid-cols-[minmax(0,380px)_minmax(0,1fr)] lg:items-start lg:gap-5">
      {/* Compose — never unmounted, so nothing being written can be lost. */}
      <div className="lg:sticky lg:top-4">
        <DescribePanel
          text={text}
          onTextChange={setText}
          onTranslate={translateNow}
          onOpenStrategies={() => setMode('strategies')}
          isTranslating={live.status === 'reading'}
          inlineError={live.error ? t(`errors.${live.error}`) : null}
          locale={locale}
          textareaRef={inputRef}
          statusSlot={<LiveStatusLine status={live.status} />}
        />
      </div>

      {/* Live reading + results. */}
      <div className="mt-4 space-y-4 lg:mt-0">
        {live.refusal && (
          <RefusalNotice
            kind={live.refusal.kind}
            onClear={clearAll}
            onExample={(example) => {
              setManual(EMPTY_MANUAL_STATE);
              setText(example);
              inputRef.current?.focus();
            }}
          />
        )}

        {hasConditions && (
          <div>
            <div className="mb-2 flex flex-wrap items-baseline gap-x-3 gap-y-1">
              <span className="fs-label uppercase tracking-widest text-muted-foreground">
                {t('translation.conditionsLabel', { count: conditions.length })}
              </span>
              <span className="fs-legal text-muted-foreground">{t('live.proposalNote')}</span>
            </div>
            <div className="grid gap-2 xl:grid-cols-2">
              {reconciled.map((item, index) => (
                <EditableConditionCard
                  key={`${item.proposalKey}-${item.userAdded ? 'user' : 'mia'}`}
                  condition={item.condition}
                  index={index}
                  assumedControls={
                    item.userEdited || item.userAdded
                      ? undefined
                      : assumedByType.get(item.condition.type)
                  }
                  onChange={updateCondition}
                  onRemove={removeCondition}
                />
              ))}
              <AddConditionPicker present={presentTypes} onAdd={addCondition} />
            </div>
          </div>
        )}

        {/* « Ce que j'ai supposé » — mandatory, never invisible. */}
        {assumptions.length > 0 && (
          <div
            data-testid="assumptions-block"
            className="rounded-r-xl border border-amber-500/40 border-l-2 border-l-amber-500 bg-amber-500/5 p-3.5"
          >
            <h4 className="mb-1.5 fs-secondary font-semibold text-amber-600">
              {t('assumptions.title')}
            </h4>
            <ul className="space-y-1">
              {assumptions.map((a, i) => (
                <li
                  key={i}
                  className="fs-secondary leading-relaxed text-amber-700/90 dark:text-amber-200/80"
                >
                  {t('assumptions.item', {
                    phrase: a.source_phrase ?? t('assumptions.aVagueWord'),
                    condition: conditionLabel(a.condition_type),
                    value: a.value ? optionLabel(a.control, a.value) : '—',
                  })}
                </li>
              ))}
            </ul>
            <p className="mt-1.5 fs-label text-amber-700/80 dark:text-amber-200/70">
              {t('assumptions.note')}
            </p>
          </div>
        )}

        {/* « Ce que je n'ai pas pu traduire » — named precisely, never substituted. */}
        {untranslatable.length > 0 && (
          <div
            data-testid="untranslatable-block"
            className="rounded-r-xl border border-amber-500/40 border-l-2 border-l-amber-500 bg-amber-500/5 p-3.5"
          >
            <h4 className="mb-1.5 fs-secondary font-semibold text-amber-600">
              {hasConditions ? t('untranslatable.titlePartial') : t('untranslatable.titleNone')}
            </h4>
            <ul className="space-y-1.5">
              {untranslatable.map((u, i) => (
                <li
                  key={i}
                  className="fs-secondary leading-relaxed text-amber-700/90 dark:text-amber-200/80"
                >
                  {u.fragment ? (
                    <b className="font-medium text-amber-600">“{u.fragment}”</b>
                  ) : null}{' '}
                  {categoryMessage(u.category, t)}
                </li>
              ))}
            </ul>
            <p className="mt-1.5 fs-label text-amber-700/80 dark:text-amber-200/70">
              {t('untranslatable.honesty')}
            </p>
          </div>
        )}

        {honestyPending && (
          <p
            data-testid="honesty-pending"
            className="fs-legal leading-relaxed text-muted-foreground"
          >
            {t('live.honestyPending')}
          </p>
        )}

        {/* Save + the live count. No « Voir les résultats »: they are below. */}
        {hasConditions && (
          <div className="flex flex-wrap items-center gap-3 rounded-2xl border border-border bg-card p-4">
            <span
              data-testid="live-count"
              className="fs-title font-semibold tracking-tight text-foreground"
            >
              {scan.status === 'ready' ? scan.count : '—'}
            </span>
            <span className="fs-secondary leading-tight text-muted-foreground">
              {t('translation.comboCaption', { conditions: conditions.length })}
              <br />
              {scan.status === 'ready'
                ? t('translation.comboScanned', { scanned: scan.scanned })
                : t('translation.comboComputing')}
            </span>
            <div className="ml-auto flex flex-wrap items-center gap-1">
              <input
                data-testid="save-name"
                value={saveName}
                onChange={(e) => setSaveName(e.target.value)}
                placeholder={t('save.namePlaceholder')}
                aria-label={t('save.namePlaceholder')}
                className="w-40 rounded-md border border-input bg-background px-2 py-1.5 fs-secondary text-foreground"
              />
              <Button variant="outline" onClick={saveStrategy}>
                {t('save.cta')}
              </Button>
            </div>
            {saveFeedback && (
              <p data-testid="save-feedback" className="w-full fs-label text-muted-foreground">
                {saveFeedback}
              </p>
            )}
          </div>
        )}

        {/* The SAME results component the manual palette renders — three blocks,
            fixed order, « ce qui va à l'encontre » never hidden nor collapsible. */}
        {hasConditions && scan.response && workingConfig && (
          <ScanResults
            response={scan.response}
            config={workingConfig}
            locale={locale}
            onEdit={() => inputRef.current?.focus()}
            onRefresh={refreshScan}
            isRefreshing={scan.status === 'loading'}
            autoRefreshEnabled={autoRefresh}
            onToggleAutoRefresh={setAutoRefresh}
          />
        )}

        {hasConditions && !scan.response && (
          <p data-testid="results-pending" className="fs-secondary text-muted-foreground">
            {scan.unavailable ? t('errors.scanUnavailable') : t('translation.comboComputing')}
          </p>
        )}
      </div>
    </div>
  );
}

/**
 * The reading indicator. Its height is RESERVED at every status — a line that
 * grows when it appears would shift the console under the pointer, and that is
 * exactly how the dictation button lost its clicks in MIA-1 (the press landed on
 * a moved target between mousedown and mouseup). Content changes, size never.
 */
function LiveStatusLine({ status }: { status: 'idle' | 'reading' | 'read' | 'capped' }) {
  const t = useTranslations('scannerChat');
  const label =
    status === 'reading'
      ? t('live.reading')
      : status === 'read'
        ? t('live.read')
        : status === 'capped'
          ? t('live.capped')
          : ' ';
  return (
    <p
      data-testid="live-status"
      data-status={status}
      aria-live="polite"
      className="mt-2 min-h-[1.25rem] fs-legal leading-relaxed text-muted-foreground"
    >
      {label}
    </p>
  );
}

/**
 * A refusal, rendered in place. Firm and non-culpabilising: the product ranks
 * nothing, predicts nothing and advises nothing — a design constraint, not a
 * gap. It occupies the reading column and leaves the field alone, because in a
 * live flow the refusal fires mid-sentence and must not take the user's own text
 * down with it.
 */
function RefusalNotice({
  kind,
  onClear,
  onExample,
}: {
  kind: string;
  onClear(): void;
  onExample(example: string): void;
}) {
  const t = useTranslations('scannerChat');
  const safeKind = ['ranking', 'prediction', 'recommendation'].includes(kind) ? kind : 'ranking';
  const examples = [t('refusal.examples.0'), t('refusal.examples.1')];
  return (
    <div
      data-testid="refusal-block"
      className="rounded-r-xl border border-amber-500/40 border-l-2 border-l-amber-500 bg-amber-500/5 p-4"
    >
      <h4 className="mb-2 fs-body font-semibold text-amber-600">{t(`refusal.${safeKind}.title`)}</h4>
      <p className="mb-2 fs-secondary leading-relaxed text-amber-700/90 dark:text-amber-200/80">
        {t(`refusal.${safeKind}.body`)}
      </p>
      <p className="fs-secondary leading-relaxed text-amber-700/90 dark:text-amber-200/80">
        {t('refusal.invite')}
      </p>
      {/* Two things the scanner CAN do, one click away — a refusal that only
          says no teaches nothing. Clicking replaces the field verbatim. */}
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
      <div className="mt-3">
        <Button variant="outline" onClick={onClear}>
          {t('refusal.reformulate')}
        </Button>
      </div>
      <p className="mt-3 fs-legal leading-relaxed text-muted-foreground">{t('refusal.footer')}</p>
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
      className="inline-flex items-center gap-1 fs-secondary text-muted-foreground hover:text-foreground"
    >
      ← {label}
    </button>
  );
}
