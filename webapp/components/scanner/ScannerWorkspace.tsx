'use client';

import * as React from 'react';
import { useConditionsConfig } from '@/lib/conditions/config-store';
import { useAutoRefreshPref } from '@/lib/conditions/auto-refresh-store';
import { useCandleCloseRefresh } from '@/lib/conditions/use-candle-close-refresh';
import {
  fetchConditionsScan,
  ScanNotAvailableError,
} from '@/lib/conditions/api-client';
import type { ConditionsConfig, ConditionsScanResponse } from '@/lib/conditions/types';
import { Button } from '@/components/ui/button';
import { ConditionsBuilder } from './ConditionsBuilder';
import { ScanResults } from './ScanResults';

/**
 * Orchestrates the Scanner page:
 *  · First visit (no saved config) → onboarding builder.
 *  · With a saved config → run the read-only scan and show results.
 *  · "Modifier mes conditions" → edit the config, then re-scan.
 */
export function ScannerWorkspace({ locale }: { locale: string }) {
  const { config, ready, save } = useConditionsConfig();
  const { enabled: autoRefresh, setEnabled: setAutoRefresh } = useAutoRefreshPref();
  const [editing, setEditing] = React.useState(false);

  const [response, setResponse] = React.useState<ConditionsScanResponse | null>(null);
  const [isScanning, setIsScanning] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);

  const runScan = React.useCallback(async (cfg: ConditionsConfig) => {
    setIsScanning(true);
    setError(null);
    try {
      const res = await fetchConditionsScan(cfg);
      setResponse(res);
    } catch (err) {
      setResponse(null);
      setError(
        err instanceof ScanNotAvailableError
          ? "Le service de scan n’est pas disponible sur cet environnement."
          : err instanceof Error
            ? err.message
            : 'Le scan a échoué.',
      );
    } finally {
      setIsScanning(false);
    }
  }, []);

  // Run a scan whenever we have a config and are not in the builder.
  const showBuilder = editing || (ready && !config);
  React.useEffect(() => {
    if (ready && config && !showBuilder) {
      void runScan(config);
    }
    // re-run when the saved config identity changes
  }, [ready, config, showBuilder, runScan]);

  // Timeframes actually scanned (from the latest response) drive the auto-refresh
  // cadence. The scan covers fixed combos (M15/H1/H4) — we read them off the
  // response so the cadence stays correct if that set ever changes.
  const timeframes = React.useMemo(
    () => Array.from(new Set((response?.matches ?? []).map((m) => m.timeframe))),
    [response],
  );

  // Auto-refresh aligned on candle closes (not a per-second poll). Only active
  // once we have a config, results, and are not editing.
  const canAutoRefresh = ready && !!config && !showBuilder && !!response;
  useCandleCloseRefresh({
    timeframes,
    enabled: autoRefresh && canAutoRefresh,
    isScanning,
    onRefresh: React.useCallback(() => {
      if (config) void runScan(config);
    }, [config, runScan]),
  });

  if (!ready) {
    return <p className="text-sm text-muted-foreground">Chargement…</p>;
  }

  if (showBuilder) {
    return (
      <ConditionsBuilder
        config={config}
        mode={config ? 'edit' : 'onboarding'}
        onCancel={config ? () => setEditing(false) : undefined}
        onSubmit={(cfg) => {
          save(cfg);
          setEditing(false);
          void runScan(cfg);
        }}
      />
    );
  }

  if (error) {
    return (
      <div className="space-y-3 rounded-lg border border-destructive/40 bg-destructive/5 p-4">
        <p className="text-sm text-foreground">{error}</p>
        <div className="flex gap-2">
          <Button size="sm" variant="outline" onClick={() => config && runScan(config)}>
            Réessayer
          </Button>
          <Button size="sm" variant="ghost" onClick={() => setEditing(true)}>
            Modifier mes conditions
          </Button>
        </div>
      </div>
    );
  }

  if (!response || !config) {
    return <p className="text-sm text-muted-foreground">Scan en cours…</p>;
  }

  return (
    <ScanResults
      response={response}
      config={config}
      locale={locale}
      isRefreshing={isScanning}
      onRefresh={() => runScan(config)}
      onEdit={() => setEditing(true)}
      autoRefreshEnabled={autoRefresh}
      onToggleAutoRefresh={setAutoRefresh}
    />
  );
}
