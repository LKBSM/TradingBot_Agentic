import { cn } from '@/lib/utils';
import { MiaAgentLogo } from './MiaAgentLogo';

/**
 * M.I.A Agent avatar — the candlestick logo inside a soft gold-ringed disc.
 * One shared component so the header, the message rows and the empty-state hero
 * all read identically (brand consistency). Purely presentational.
 *
 * The gold tint comes from the `--sentinel-warn` family (the MIA accent); kept
 * as an inline radial gradient because a multi-stop gradient is more legible
 * here than a long arbitrary Tailwind value.
 */
const RING = 'hsl(35 92% 55% / 0.35)';
const FILL =
  'radial-gradient(circle at 30% 30%, hsl(35 92% 60% / 0.18), hsl(var(--card)))';

interface AgentAvatarProps {
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

export function AgentAvatar({ size = 'md', className }: AgentAvatarProps) {
  const box =
    size === 'lg'
      ? 'h-14 w-14 rounded-2xl'
      : size === 'sm'
        ? 'h-7 w-7 rounded-full'
        : 'h-9 w-9 rounded-full';
  const glyph = size === 'lg' ? 'h-8 w-8' : size === 'sm' ? 'h-4 w-4' : 'h-5 w-5';
  return (
    <div
      aria-hidden
      className={cn('flex shrink-0 items-center justify-center border', box, className)}
      style={{ background: FILL, borderColor: RING }}
    >
      <MiaAgentLogo className={glyph} />
    </div>
  );
}
