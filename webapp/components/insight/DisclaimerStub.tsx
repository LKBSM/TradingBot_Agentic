import { cn } from '@/lib/utils';

/**
 * Compact disclaimer line shown under the hero. Pure visual placeholder for
 * F1-F4 — the actual wording must come from the legal terminal (CGU + MiFID
 * II finfluencer-safe formulation, UE 2024/2811). Replaced during the legal
 * integration pass.
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
      data-legal-pending="card-hero-disclaimer"
    >
      {/* LEGAL-PENDING: hero card disclaimer — to be replaced with the
          wording finalised by the legal terminal (UE 2024/2811 + MiFID II
          finfluencer 03/2026). */}
      Démonstration paper-trading. Lecture algorithmique éducative — ne
      constitue ni un signal de trading, ni un conseil en investissement.
    </p>
  );
}
