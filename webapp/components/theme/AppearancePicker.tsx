'use client';

import { Check } from 'lucide-react';
import { useTranslations } from 'next-intl';
import { useTheme } from 'next-themes';
import * as React from 'react';
import { cn } from '@/lib/utils';
import { THEMES, DEFAULT_THEME, type ThemeMeta } from '@/lib/theme/themes';

/**
 * "Apparence" picker — a clickable vignette per theme (preview + name + one-line
 * description). Selecting one applies it immediately (next-themes) and persists
 * across sessions (localStorage). Mount-guarded so the "active" ring doesn't
 * cause a hydration mismatch. Purely presentational.
 */
export function AppearancePicker() {
  const t = useTranslations('appearance');
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = React.useState(false);
  React.useEffect(() => setMounted(true), []);
  const active = mounted ? (theme ?? DEFAULT_THEME) : null;

  return (
    <div
      role="radiogroup"
      aria-label={t('groupLabel')}
      className="grid grid-cols-1 gap-3 sm:grid-cols-2"
    >
      {THEMES.map((theme) => (
        <ThemeCard
          key={theme.id}
          theme={theme}
          name={t(`names.${theme.id}`)}
          description={t(`descriptions.${theme.id}`)}
          activeLabel={t('active')}
          selected={active === theme.id}
          onSelect={() => setTheme(theme.id)}
        />
      ))}
    </div>
  );
}

function ThemeCard({
  theme,
  name,
  description,
  activeLabel,
  selected,
  onSelect,
}: {
  theme: ThemeMeta;
  name: string;
  description: string;
  activeLabel: string;
  selected: boolean;
  onSelect: () => void;
}) {
  return (
    <button
      type="button"
      role="radio"
      aria-checked={selected}
      onClick={onSelect}
      className={cn(
        'group flex flex-col gap-3 rounded-lg border p-3 text-left transition-colors',
        'focus:outline-none focus-visible:ring-2 focus-visible:ring-ring',
        selected
          ? 'border-primary ring-1 ring-primary'
          : 'border-border/60 hover:border-border',
      )}
    >
      <ThemeSwatch theme={theme} />
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="flex items-center gap-1.5">
            <span className="text-sm font-medium text-foreground">{name}</span>
            {selected && (
              <span className="inline-flex items-center gap-0.5 text-xs font-medium text-primary">
                <Check className="h-3.5 w-3.5" aria-hidden />
                {activeLabel}
              </span>
            )}
          </div>
          <p className="mt-0.5 text-xs text-muted-foreground">{description}</p>
        </div>
      </div>
    </button>
  );
}

/**
 * A miniature mock of the app in the theme's colours, DERIVED from the live design
 * tokens (THM-1): the wrapper carries `data-design={id}` so every `var(--…)` below
 * resolves to THAT theme's palette regardless of the active one — the vignette can
 * never drift from the real tokens. Accent bar (--acc), a muted text line (--txt),
 * and the reserved bull/bear data chips (--bull / --bear).
 */
function ThemeSwatch({ theme }: { theme: ThemeMeta }) {
  return (
    <div
      data-design={theme.id}
      className="flex h-16 w-full items-center gap-2 rounded-md border p-2"
      style={{ background: 'var(--bg)', borderColor: 'var(--line-2)' }}
      aria-hidden
    >
      <div
        className="flex h-full flex-1 flex-col justify-between rounded p-1.5"
        style={{ background: 'var(--panel)' }}
      >
        <span className="h-1.5 w-2/3 rounded-full" style={{ background: 'var(--acc)' }} />
        <span
          className="h-1 w-1/2 rounded-full"
          style={{ background: 'var(--txt)', opacity: 0.4 }}
        />
        <div className="flex gap-1">
          <span className="h-1.5 w-4 rounded-full" style={{ background: 'var(--bull)' }} />
          <span className="h-1.5 w-4 rounded-full" style={{ background: 'var(--bear)' }} />
        </div>
      </div>
    </div>
  );
}
