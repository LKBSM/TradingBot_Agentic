import { cn } from '@/lib/utils';

/**
 * Compact disclaimer line shown under a market-reading hero. Single source of
 * truth (Chantier 5.C) — previously duplicated across `insight/` and
 * `market-reading/`. The wording is a placeholder pending the legal terminal.
 *
 * LEGAL-PENDING: do not ship this wording to production. Reviewers grep for
 * "LEGAL-PENDING" before each release tag.
 */
export function DisclaimerStub({ className }: { className?: string }) {
  return (
    <p
      className={cn(
        'text-[11px] italic leading-relaxed text-muted-foreground/80',
        className,
      )}
      data-legal-pending="hero-disclaimer"
    >
      {/* LEGAL-PENDING: hero disclaimer — to be replaced with the wording
          finalised by the legal terminal. */}
      Démonstration en accès anticipé. Lecture algorithmique éducative — ne
      constitue ni un signal de trading, ni un conseil en investissement.
    </p>
  );
}
