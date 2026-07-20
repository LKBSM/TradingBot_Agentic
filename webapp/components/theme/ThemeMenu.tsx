'use client';

import { Check, Palette } from 'lucide-react';
import { useTheme } from 'next-themes';
import * as React from 'react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import { THEMES, DEFAULT_THEME } from '@/lib/theme/themes';

/**
 * Compact theme selector for the nav — a button that opens a small menu listing
 * the four themes (swatch + name, check on the active one). Selecting applies
 * immediately (next-themes) and persists. Closes on outside-click / Escape.
 * Mount-guarded to avoid a hydration mismatch on the active marker. The richer
 * preview picker lives in the account "Apparence" settings.
 */
export function ThemeMenu() {
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = React.useState(false);
  const [open, setOpen] = React.useState(false);
  const rootRef = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => setMounted(true), []);

  React.useEffect(() => {
    if (!open) return;
    function onPointer(e: MouseEvent) {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) setOpen(false);
    }
    function onKey(e: KeyboardEvent) {
      if (e.key === 'Escape') setOpen(false);
    }
    document.addEventListener('mousedown', onPointer);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onPointer);
      document.removeEventListener('keydown', onKey);
    };
  }, [open]);

  const active = mounted ? (theme ?? DEFAULT_THEME) : null;

  return (
    <div ref={rootRef} className="relative">
      <Button
        variant="ghost"
        size="icon"
        aria-label="Choisir le thème"
        aria-haspopup="menu"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
      >
        <Palette className="h-4 w-4" />
      </Button>

      {open && (
        <div
          role="menu"
          aria-label="Thèmes"
          className="absolute right-0 z-50 mt-2 w-60 overflow-hidden rounded-lg border border-border bg-popover p-1 text-popover-foreground shadow-md"
        >
          {THEMES.map((t) => {
            const selected = active === t.id;
            return (
              <button
                key={t.id}
                type="button"
                role="menuitemradio"
                aria-checked={selected}
                onClick={() => {
                  setTheme(t.id);
                  setOpen(false);
                }}
                className={cn(
                  'flex w-full items-center gap-2.5 rounded-md px-2 py-1.5 text-left text-sm transition-colors',
                  'focus:outline-none focus-visible:bg-accent hover:bg-accent',
                )}
              >
                <span
                  className="flex h-6 w-6 shrink-0 items-center justify-center rounded border border-black/10"
                  style={{ background: t.swatch.bg }}
                  aria-hidden
                >
                  <span
                    className="h-2.5 w-2.5 rounded-full"
                    style={{ background: t.swatch.accent }}
                  />
                </span>
                <span className="min-w-0 flex-1">
                  <span className="block font-medium text-foreground">{t.name}</span>
                </span>
                {selected && <Check className="h-4 w-4 shrink-0 text-primary" aria-hidden />}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
