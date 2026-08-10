import { cn } from '@/lib/utils';
import { BrandMark } from '@/components/BrandMark';
import { BRAND_NAME, BRAND_BASELINE } from '@/lib/brand';

/**
 * BrandLockup — the stacked brand block used across the connected surfaces
 * (product rail, mobile /app header, loading screen). It pairs the glyph
 * (`BrandMark`) with the wordmark on top and the developed acronym below:
 *
 *     ▪  MIA Markets
 *        Multi-asset Intelligence Assistant
 *
 * The baseline carries the meaning ("MIA" = Multi-asset Intelligence Assistant)
 * and stays legible — it is real text, never an image, and stays in English in
 * every locale on purpose (see lib/brand.ts: a localized expansion would spell a
 * different acronym). This is the SINGLE source of brand markup for those
 * surfaces; nothing hand-stacks the wordmark and baseline elsewhere.
 *
 * Presentational only — no state, no theming logic. Colours come from the
 * shared `foreground` / `muted-foreground` tokens so all four themes are
 * covered, on both the marketing chrome and inside the product shell.
 */
export function BrandLockup({
  size = 'md',
  baseline = true,
  className,
}: {
  /** `md` for headers, `sm` for the narrow product rail / mobile header. */
  size?: 'sm' | 'md';
  /** Show the developed acronym under the wordmark (default). */
  baseline?: boolean;
  className?: string;
}) {
  const glyph = size === 'sm' ? 22 : 28;
  return (
    <span className={cn('inline-flex items-center gap-2', className)}>
      <BrandMark size={glyph} />
      <span className="flex min-w-0 flex-col leading-none">
        <span
          className={cn(
            'font-semibold tracking-tight text-foreground',
            size === 'sm' ? 'text-[13px]' : 'text-sm',
          )}
        >
          {BRAND_NAME}
        </span>
        {baseline && (
          <span
            className={cn(
              'mt-0.5 font-normal tracking-tight text-muted-foreground',
              size === 'sm' ? 'text-[10px] leading-tight' : 'text-[11px] leading-tight',
            )}
          >
            {BRAND_BASELINE}
          </span>
        )}
      </span>
    </span>
  );
}
