'use client';

import { Globe } from 'lucide-react';
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import { cn } from '@/lib/utils';

/**
 * Locale switcher V1 — FR is locked active, EN is shown disabled with a
 * tooltip. The international ambition is signalled, but EN routes still
 * 302-redirect to FR via the middleware. Will be replaced by a real
 * dropdown when EN content lands (V3 per `dev_focus_pivot_2026_05_27`).
 */
export function LocaleToggle() {
  return (
    <div className="flex items-center gap-1 rounded-md border border-border/70 px-1 py-1 text-xs">
      <span
        aria-current="page"
        className="rounded-sm bg-secondary px-2 py-0.5 font-medium text-secondary-foreground"
      >
        <Globe className="mr-1 inline h-3 w-3" aria-hidden />
        FR
      </span>
      <Tooltip>
        <TooltipTrigger asChild>
          <button
            type="button"
            disabled
            aria-disabled="true"
            aria-label="Version anglaise — bientôt disponible"
            className={cn(
              'cursor-not-allowed rounded-sm px-2 py-0.5 font-medium text-muted-foreground/60 transition-colors',
            )}
          >
            EN
          </button>
        </TooltipTrigger>
        <TooltipContent side="bottom" className="max-w-xs text-xs">
          English version — coming soon.
        </TooltipContent>
      </Tooltip>
    </div>
  );
}
