'use client';

import * as React from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { useLocalizedHref } from '@/lib/i18n/href';
import { resolveComboFromQuery } from '@/lib/conditions/app-link';
import { AppChatSidebar } from '@/components/app/AppChatSidebar';
import type { Combo } from '@/lib/market-reading/store';

const DEFAULT_COMBO: Combo = { instrument: 'XAUUSD', timeframe: 'M15' };

/**
 * The docked chat column of the product shell (right column, /app only). It hosts
 * the existing AppChatSidebar unchanged — the shared ChatProvider (locale layout)
 * keeps this in lockstep with the /app workspace, whose effects already align the
 * chat context to the active combo. Here we only feed it the URL-derived combo
 * and route combo switches back through the URL (the single source of truth).
 */
export function ShellChat() {
  const router = useRouter();
  const lh = useLocalizedHref();
  const searchParams = useSearchParams();

  const active =
    resolveComboFromQuery(
      searchParams.get('instrument') ?? undefined,
      searchParams.get('timeframe') ?? undefined,
    ) ?? DEFAULT_COMBO;

  const onSelectCombo = React.useCallback(
    (combo: Combo) => {
      router.replace(
        lh(`/app?instrument=${combo.instrument}&timeframe=${combo.timeframe}`),
        { scroll: false },
      );
    },
    [lh, router],
  );

  return (
    <aside className="chatcol">
      <AppChatSidebar active={active} onSelectCombo={onSelectCombo} />
    </aside>
  );
}
