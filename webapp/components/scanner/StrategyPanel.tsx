'use client';

import * as React from 'react';
import { useTranslations } from 'next-intl';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { cn } from '@/lib/utils';
import {
  validateStrategy,
  type SavedStrategy,
  type StrategyMutationResult,
} from '@/lib/conditions/strategy-store';
import { useSavedReadingCounts } from '@/lib/conditions/use-saved-reading-counts';
import { useNow } from '@/lib/conditions/use-now';
import { useScannerLabels } from './use-scanner-labels';

/**
 * "Mes stratégies" — the saved-strategy list (client-only, localStorage).
 *
 * Each row shows the name (free text, display only), the condition count and
 * the validity against the CURRENT schema. An invalid strategy (out-of-schema
 * condition, unsupported version) shows the precise reasons and cannot be
 * loaded — no silent partial execution. Loading a valid one repopulates the
 * builder palette; the scan itself still goes through the existing
 * "Enregistrer & relancer" path.
 */

/** Translator bound to the `scanner` namespace (next-intl's useTranslations). */
type ScannerT = ReturnType<typeof useTranslations<'scanner'>>;

/** Map a mutation error code to a localized message via the scanner translator. */
export function mutationErrorMessage(
  result: StrategyMutationResult,
  t: ScannerT,
): string | null {
  if (result.ok) return null;
  const key = `strategyPanel.errors.${result.error}`;
  return t(key);
}

function formatDate(ts: number, locale: string): string {
  if (!Number.isFinite(ts) || ts <= 0) return '—';
  try {
    return new Intl.DateTimeFormat(locale, {
      day: '2-digit',
      month: 'short',
      hour: '2-digit',
      minute: '2-digit',
    }).format(new Date(ts));
  } catch {
    return '—';
  }
}

export function StrategyPanel({
  strategies,
  locale,
  onLoad,
  onRename,
  onDuplicate,
  onDelete,
  onExport = () => '',
  onImport = () => ({ ok: false, error: 'malformed' }),
}: {
  strategies: SavedStrategy[];
  locale: string;
  onLoad(strategy: SavedStrategy): void;
  onRename(id: string, name: string): StrategyMutationResult;
  onDuplicate(id: string): StrategyMutationResult;
  onDelete(id: string): void;
  /** Serialise all readings to portable text (device portability). */
  onExport?(): string;
  /** Merge readings from a text envelope. */
  onImport?(text: string): { ok: true; imported: number; skipped: number } | { ok: false; error: string };
}) {
  const t = useTranslations('scanner');
  // C4: counts re-evaluated ON OPEN, per-combo staleness on its own timeframe.
  const { counts, rescan } = useSavedReadingCounts(strategies, true);
  const { age } = useScannerLabels();
  const now = useNow(30_000);
  const [renamingId, setRenamingId] = React.useState<string | null>(null);
  const [renameDraft, setRenameDraft] = React.useState('');
  const [confirmDeleteId, setConfirmDeleteId] = React.useState<string | null>(null);
  const [feedback, setFeedback] = React.useState<string | null>(null);
  const [transfer, setTransfer] = React.useState(false);
  const [exportText, setExportText] = React.useState('');
  const [importDraft, setImportDraft] = React.useState('');
  const [transferMsg, setTransferMsg] = React.useState<{ kind: 'ok' | 'error'; text: string } | null>(null);

  if (strategies.length === 0) return null;

  function doExport() {
    setExportText(onExport());
    setImportDraft('');
    setTransferMsg(null);
    setTransfer(true);
  }
  function doImport() {
    const res = onImport(importDraft);
    if (res.ok) {
      setTransferMsg({ kind: 'ok', text: t('strategyPanel.importResult', { imported: res.imported, skipped: res.skipped }) });
      setImportDraft('');
    } else {
      setTransferMsg({
        kind: 'error',
        text: t.has(`strategyPanel.importErrors.${res.error}`)
          ? t(`strategyPanel.importErrors.${res.error}`)
          : t('strategyPanel.importErrors.malformed'),
      });
    }
  }

  function submitRename(id: string) {
    const result = onRename(id, renameDraft);
    const message = mutationErrorMessage(result, t);
    setFeedback(message);
    if (result.ok) {
      setRenamingId(null);
      setRenameDraft('');
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">{t('strategyPanel.title')}</CardTitle>
        {/* Mandatory, non-maskable: on-device only, and the count is a finding
            not a ranking (a reading returning seven combos is not "better" than
            one returning none — it is wider). */}
        <p className="mt-1 text-xs text-muted-foreground">{t('strategyPanel.subtitle')}</p>
        <p className="mt-1 rounded-md border border-border/50 bg-background/40 p-2 text-[11px] text-muted-foreground">
          {t('strategyPanel.notRankingNote')}
        </p>
      </CardHeader>
      <CardContent className="space-y-2">
        {feedback && (
          <p role="alert" className="text-xs text-destructive">
            {feedback}
          </p>
        )}

        {/* Transfer — export/import readings as text (device portability). */}
        <div className="rounded-lg border border-border/60 p-2">
          <div className="flex flex-wrap items-center gap-2">
            <button type="button" className="btn" onClick={doExport}>{t('strategyPanel.export')}</button>
            <button
              type="button"
              className="btn"
              onClick={() => {
                setTransfer(true);
                setExportText('');
                setTransferMsg(null);
              }}
            >
              {t('strategyPanel.import')}
            </button>
            <span className="text-[11px] text-muted-foreground">{t('strategyPanel.transferHint')}</span>
          </div>
          {transfer && (
            <div className="mt-2 space-y-2">
              {exportText && (
                <textarea
                  readOnly
                  value={exportText}
                  aria-label={t('strategyPanel.exportLabel')}
                  onFocus={(e) => e.currentTarget.select()}
                  className="h-24 w-full rounded-md border border-input bg-background p-2 font-mono text-[11px] text-foreground"
                />
              )}
              <textarea
                value={importDraft}
                onChange={(e) => setImportDraft(e.target.value)}
                placeholder={t('strategyPanel.importPlaceholder')}
                aria-label={t('strategyPanel.importPlaceholder')}
                className="h-20 w-full rounded-md border border-input bg-background p-2 font-mono text-[11px] text-foreground"
              />
              <div className="flex items-center gap-2">
                <button type="button" className="btn" onClick={doImport} disabled={importDraft.trim().length === 0}>
                  {t('strategyPanel.importDo')}
                </button>
                <button type="button" className="btn" onClick={() => setTransfer(false)}>{t('strategyPanel.cancel')}</button>
                {transferMsg && (
                  <span className={cn('text-[11px]', transferMsg.kind === 'error' ? 'text-destructive' : 'text-muted-foreground')}>
                    {transferMsg.text}
                  </span>
                )}
              </div>
            </div>
          )}
        </div>
        <ul className="space-y-2">
          {strategies.map((strategy) => {
            const problems = validateStrategy(strategy);
            const invalid = problems.length > 0;
            const conditionCount = Array.isArray(strategy.config?.conditions)
              ? strategy.config.conditions.length
              : 0;
            return (
              <li
                key={strategy.id}
                className={cn(
                  'rounded-lg border p-3',
                  invalid ? 'border-destructive/40 bg-destructive/5' : 'border-border/60',
                )}
              >
                <div className="flex flex-wrap items-center gap-2">
                  {renamingId === strategy.id ? (
                    <>
                      <input
                        value={renameDraft}
                        onChange={(e) => setRenameDraft(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter') submitRename(strategy.id);
                          if (e.key === 'Escape') setRenamingId(null);
                        }}
                        aria-label={t('strategyPanel.newNameAria')}
                        autoFocus
                        className="min-w-0 flex-1 rounded-md border border-input bg-background px-2 py-1 text-sm text-foreground"
                      />
                      <Button size="sm" onClick={() => submitRename(strategy.id)}>
                        {t('strategyPanel.ok')}
                      </Button>
                      <Button size="sm" variant="ghost" onClick={() => setRenamingId(null)}>
                        {t('strategyPanel.cancel')}
                      </Button>
                    </>
                  ) : (
                    <>
                      <span className="text-sm font-medium text-foreground">
                        {strategy.name}
                      </span>
                      {invalid && (
                        <span className="rounded-full border border-destructive/50 px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide text-destructive">
                          {t('strategyPanel.invalid')}
                        </span>
                      )}
                      <span className="text-xs text-muted-foreground">
                        {t('strategyPanel.meta', {
                          count: conditionCount,
                          logic:
                            strategy.config?.logic === 'OR'
                              ? t('strategyPanel.logicOr')
                              : t('strategyPanel.logicAnd'),
                          when: formatDate(strategy.lastUsedAt, locale),
                        })}
                      </span>
                    </>
                  )}
                </div>

                {/* C4 — combo count re-evaluated on open; per-combo staleness on
                    its own timeframe; a partial count is NEVER shown as complete. */}
                {!invalid && (() => {
                  const c = counts[strategy.id];
                  if (!c || c.status === 'loading') {
                    return <p className="mt-1 text-xs text-muted-foreground">{t('strategyPanel.counting')}</p>;
                  }
                  if (c.status === 'error') {
                    return (
                      <div className="mt-1 flex items-center gap-2 text-xs text-muted-foreground">
                        {t('strategyPanel.countError')}
                        <Button size="sm" variant="ghost" className="h-6" onClick={() => rescan(strategy.id)}>
                          {t('strategyPanel.rescan')}
                        </Button>
                      </div>
                    );
                  }
                  const evaluated = age(c.evaluatedAt, now);
                  return (
                    <div className="mt-1 flex flex-wrap items-center gap-2">
                      <span className="text-sm font-semibold text-foreground">
                        {t('strategyPanel.comboCount', { count: c.count })}
                      </span>
                      {c.staleCount > 0 ? (
                        <>
                          <span className="rounded-full border border-amber-500/50 px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide text-amber-600">
                            {t('strategyPanel.incompleteBadge')}
                          </span>
                          <span className="text-xs text-muted-foreground">
                            {t('strategyPanel.incompleteDetail', { count: c.staleCount })}
                          </span>
                          <Button size="sm" variant="ghost" className="h-6" onClick={() => rescan(strategy.id)}>
                            {t('strategyPanel.rescan')}
                          </Button>
                        </>
                      ) : (
                        evaluated && (
                          <span className="text-xs text-muted-foreground">
                            {t('strategyPanel.evaluatedAgo', { age: evaluated })}
                          </span>
                        )
                      )}
                    </div>
                  );
                })()}

                {invalid && (
                  <ul className="mt-2 list-disc space-y-0.5 pl-5 text-xs text-destructive">
                    {problems.map((p, i) => (
                      <li key={i}>{p}</li>
                    ))}
                  </ul>
                )}

                {renamingId !== strategy.id && (
                  <div className="mt-2 flex flex-wrap gap-2">
                    <Button
                      size="sm"
                      variant="outline"
                      disabled={invalid}
                      title={invalid ? t('strategyPanel.loadDisabledTitle') : undefined}
                      onClick={() => onLoad(strategy)}
                    >
                      {t('strategyPanel.load')}
                    </Button>
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => {
                        setRenamingId(strategy.id);
                        setRenameDraft(strategy.name);
                        setConfirmDeleteId(null);
                        setFeedback(null);
                      }}
                    >
                      {t('strategyPanel.rename')}
                    </Button>
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => setFeedback(mutationErrorMessage(onDuplicate(strategy.id), t))}
                    >
                      {t('strategyPanel.duplicate')}
                    </Button>
                    {confirmDeleteId === strategy.id ? (
                      <>
                        <Button
                          size="sm"
                          variant="destructive"
                          onClick={() => {
                            onDelete(strategy.id);
                            setConfirmDeleteId(null);
                          }}
                        >
                          {t('strategyPanel.confirmDelete')}
                        </Button>
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() => setConfirmDeleteId(null)}
                        >
                          {t('strategyPanel.keep')}
                        </Button>
                      </>
                    ) : (
                      <Button
                        size="sm"
                        variant="ghost"
                        className="text-destructive hover:text-destructive"
                        onClick={() => setConfirmDeleteId(strategy.id)}
                      >
                        {t('strategyPanel.delete')}
                      </Button>
                    )}
                  </div>
                )}
              </li>
            );
          })}
        </ul>
      </CardContent>
    </Card>
  );
}
