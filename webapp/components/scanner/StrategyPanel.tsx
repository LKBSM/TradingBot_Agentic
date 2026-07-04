'use client';

import * as React from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { cn } from '@/lib/utils';
import {
  validateStrategy,
  type SavedStrategy,
  type StrategyMutationResult,
} from '@/lib/conditions/strategy-store';

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

const ERROR_MESSAGES: Record<string, string> = {
  name_required: 'Le nom ne peut pas être vide.',
  limit_reached:
    'Limite de stratégies atteinte — supprime une stratégie existante pour en créer une nouvelle.',
  not_found: 'Stratégie introuvable.',
  storage_failed:
    'Impossible d’écrire dans le stockage local (quota ou navigation privée). Rien n’a été sauvegardé.',
};

export function mutationErrorMessage(result: StrategyMutationResult): string | null {
  if (result.ok) return null;
  return ERROR_MESSAGES[result.error] ?? 'Opération impossible.';
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
}: {
  strategies: SavedStrategy[];
  locale: string;
  onLoad(strategy: SavedStrategy): void;
  onRename(id: string, name: string): StrategyMutationResult;
  onDuplicate(id: string): StrategyMutationResult;
  onDelete(id: string): void;
}) {
  const [renamingId, setRenamingId] = React.useState<string | null>(null);
  const [renameDraft, setRenameDraft] = React.useState('');
  const [confirmDeleteId, setConfirmDeleteId] = React.useState<string | null>(null);
  const [feedback, setFeedback] = React.useState<string | null>(null);

  if (strategies.length === 0) return null;

  function submitRename(id: string) {
    const result = onRename(id, renameDraft);
    const message = mutationErrorMessage(result);
    setFeedback(message);
    if (result.ok) {
      setRenamingId(null);
      setRenameDraft('');
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Mes stratégies</CardTitle>
        <p className="mt-1 text-xs text-muted-foreground">
          Sauvegardées sur cet appareil uniquement — rien n’est envoyé au serveur.
        </p>
      </CardHeader>
      <CardContent className="space-y-2">
        {feedback && (
          <p role="alert" className="text-xs text-destructive">
            {feedback}
          </p>
        )}
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
                        aria-label="Nouveau nom"
                        autoFocus
                        className="min-w-0 flex-1 rounded-md border border-input bg-background px-2 py-1 text-sm text-foreground"
                      />
                      <Button size="sm" onClick={() => submitRename(strategy.id)}>
                        OK
                      </Button>
                      <Button size="sm" variant="ghost" onClick={() => setRenamingId(null)}>
                        Annuler
                      </Button>
                    </>
                  ) : (
                    <>
                      <span className="text-sm font-medium text-foreground">
                        {strategy.name}
                      </span>
                      {invalid && (
                        <span className="rounded-full border border-destructive/50 px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide text-destructive">
                          Invalide
                        </span>
                      )}
                      <span className="text-xs text-muted-foreground">
                        {conditionCount} condition{conditionCount > 1 ? 's' : ''} ·{' '}
                        {strategy.config?.logic === 'OR' ? 'OU' : 'ET'} · utilisée le{' '}
                        {formatDate(strategy.lastUsedAt, locale)}
                      </span>
                    </>
                  )}
                </div>

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
                      title={
                        invalid
                          ? 'Cette stratégie contient des éléments hors schéma actuel — elle ne peut pas être relancée.'
                          : undefined
                      }
                      onClick={() => onLoad(strategy)}
                    >
                      Charger
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
                      Renommer
                    </Button>
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => setFeedback(mutationErrorMessage(onDuplicate(strategy.id)))}
                    >
                      Dupliquer
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
                          Confirmer la suppression
                        </Button>
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() => setConfirmDeleteId(null)}
                        >
                          Garder
                        </Button>
                      </>
                    ) : (
                      <Button
                        size="sm"
                        variant="ghost"
                        className="text-destructive hover:text-destructive"
                        onClick={() => setConfirmDeleteId(strategy.id)}
                      >
                        Supprimer
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
