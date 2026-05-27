import Link from 'next/link';
import { LocaleToggle } from '@/components/LocaleToggle';
import { ThemeToggle } from '@/components/theme-toggle';

const ANCHORS = [
  { href: '#demo', label: 'Démo' },
  { href: '#honnetete', label: 'Honnêteté' },
  { href: '#tarifs', label: 'Tarifs' },
  { href: '#faq', label: 'FAQ' },
] as const;

export function Nav() {
  return (
    <header className="sticky top-0 z-40 w-full border-b border-border/60 bg-background/85 backdrop-blur">
      <div className="container-prose flex h-14 items-center justify-between gap-4">
        <Link
          href="/"
          className="flex items-center gap-2 text-sm font-semibold tracking-tight"
          aria-label="MIA Markets — retour à l'accueil"
        >
          <span
            aria-hidden
            className="flex h-7 w-7 items-center justify-center rounded-md bg-gradient-to-br from-amber-400 to-amber-600 text-xs font-bold text-white shadow-sm"
          >
            M
          </span>
          <span>MIA Markets</span>
        </Link>

        <nav aria-label="Sections du site" className="hidden sm:block">
          <ul className="flex items-center gap-1 text-sm">
            {ANCHORS.map((a) => (
              <li key={a.href}>
                <Link
                  href={a.href}
                  className="rounded-md px-3 py-1.5 text-muted-foreground transition-colors hover:bg-accent hover:text-accent-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                >
                  {a.label}
                </Link>
              </li>
            ))}
          </ul>
        </nav>

        <div className="flex items-center gap-2">
          <LocaleToggle />
          <ThemeToggle />
        </div>
      </div>
    </header>
  );
}
