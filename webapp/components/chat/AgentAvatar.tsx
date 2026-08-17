import { cn } from '@/lib/utils';
import { MiaLogo } from '@/components/brand/MiaLogo';

/**
 * M.I.A Agent avatar — the compact prism logo inside a soft brand-tinted disc.
 * One shared component so the header, the message rows and the empty-state hero
 * all read identically (brand consistency). Purely presentational.
 *
 * The tint is derived from the brand mark colour (`--brand-mark`) so the disc
 * follows the active light/dark theme instead of a hard-coded accent.
 */
const RING = 'color-mix(in srgb, var(--brand-mark) 35%, transparent)';
const FILL =
  'radial-gradient(circle at 30% 30%, color-mix(in srgb, var(--brand-mark) 16%, transparent), hsl(var(--card)))';

interface AgentAvatarProps {
  size?: 'sm' | 'md' | 'lg';
  className?: string;
  /**
   * Render the small "online" presence pastille at the bottom-right (the green
   * dot shown on the publication M.I.A header). Off by default so /app keeps its
   * own inline dot untouched — same shared avatar, opt-in decoration.
   */
  presence?: boolean;
}

export function AgentAvatar({ size = 'md', className, presence = false }: AgentAvatarProps) {
  const box =
    size === 'lg'
      ? 'h-14 w-14 rounded-2xl'
      : size === 'sm'
        ? 'h-7 w-7 rounded-full'
        : 'h-9 w-9 rounded-full';
  const glyph = size === 'lg' ? 'h-8 w-8' : size === 'sm' ? 'h-4 w-4' : 'h-5 w-5';
  return (
    <div className="relative shrink-0">
      <div
        aria-hidden
        className={cn('flex items-center justify-center border', box, className)}
        style={{ background: FILL, borderColor: RING }}
      >
        <MiaLogo variant="compact" decorative className={glyph} />
      </div>
      {presence && (
        <span
          aria-hidden
          data-presence="1"
          className="absolute -bottom-0.5 -right-0.5 h-2.5 w-2.5 rounded-full border-2 border-card bg-[hsl(var(--sentinel-bull))]"
        />
      )}
    </div>
  );
}
