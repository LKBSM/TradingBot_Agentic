import Link from 'next/link';
import { ThemeToggle } from '@/components/theme-toggle';

const ANCHORS = [
  { href: '#demo', label: 'Démo' },
  { href: '#comment', label: 'Fonctionnement' },
  { href: '#tarifs', label: 'Tarifs' },
] as const;

export function Nav() {
  return (
    <header className="sticky top-0 z-40 w-full border-b border-border/60 bg-background/85 backdrop-blur">
      <div className="container-prose flex h-14 items-center justify-between gap-4">
        <Link
          href="/"
          className="text-sm font-semibold tracking-tight"
          aria-label="Smart Sentinel — retour à l'accueil"
        >
          Smart Sentinel
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

        <ThemeToggle />
      </div>
    </header>
  );
}
