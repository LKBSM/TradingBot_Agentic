'use client';

import Link from 'next/link';
import { Info } from 'lucide-react';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import { cn } from '@/lib/utils';
import { GLOSSARY, type GlossaryKey } from '@/lib/glossary';

/**
 * Vulgarisation tooltip (Chantier 5.D).
 *
 * Wraps a technical term with a discreet dotted underline + an ⓘ icon. On
 * hover OR keyboard focus it surfaces the short, plain-language definition from
 * the central glossary, plus an "En savoir plus →" link to the matching
 * /methodology anchor.
 *
 * Single source of truth: the copy lives in `lib/glossary.ts`, shared with the
 * /methodology page — no duplicated wording to drift out of sync.
 *
 * Accessibility: the trigger is a real <button> so it is reachable by keyboard
 * and announced by screen readers; Radix keeps the content open while it is
 * hovered, so the inner link stays clickable.
 */
export function InfoTooltip({
  termKey,
  children,
  iconOnly = false,
  className,
}: {
  /** Glossary key resolving the term, short definition and /methodology anchor. */
  termKey: GlossaryKey;
  /** Visible label. Defaults to the glossary `term` when omitted. */
  children?: React.ReactNode;
  /** Render only the ⓘ icon (no label) — for placing next to an existing badge. */
  iconOnly?: boolean;
  className?: string;
}) {
  const entry = GLOSSARY[termKey];

  return (
    <TooltipProvider delayDuration={150}>
      <Tooltip>
        <TooltipTrigger asChild>
          <button
            type="button"
            className={cn(
              iconOnly
                ? 'inline-flex items-center text-muted-foreground hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring focus-visible:ring-offset-1 rounded-sm'
                : 'inline-flex items-center gap-0.5 underline decoration-dotted decoration-muted-foreground/50 underline-offset-2 hover:decoration-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring focus-visible:ring-offset-1 rounded-sm',
              className,
            )}
            aria-label={`${entry.term} — définition`}
          >
            {!iconOnly && (children ?? entry.term)}
            <Info className="h-3 w-3 shrink-0 text-muted-foreground" aria-hidden />
          </button>
        </TooltipTrigger>
        <TooltipContent className="max-w-xs text-pretty">
          <p className="text-xs font-semibold">{entry.term}</p>
          <p className="mt-1 text-xs font-normal leading-relaxed text-popover-foreground/90">
            {entry.short}
          </p>
          <Link
            href={`/methodology${entry.anchor}`}
            className="mt-1.5 inline-block text-[11px] font-medium underline underline-offset-2 hover:text-primary"
          >
            En savoir plus →
          </Link>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}
