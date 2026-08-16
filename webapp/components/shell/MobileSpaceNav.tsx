'use client';

import Link from 'next/link';
import { useTranslations } from 'next-intl';
import { CalendarClock, CandlestickChart, Layers, Radar, Settings } from 'lucide-react';
import { cn } from '@/lib/utils';
import { useLocalizedHref } from '@/lib/i18n/href';

/**
 * Mobile space navigation (UI-1b) — a sticky bottom tab bar shown ONLY below
 * 768px on the `.no-chat` surfaces (/zones, /scanner, /actualites, /compte),
 * where the left rail is collapsed. It carries the SAME five space links as
 * `ShellRail` (App · Scanner · Zones · Actualités · Compte), so a phone user is
 * never stranded once the rail is hidden — including a way back to /app.
 *
 * Navigation only: identical hrefs/labels to the rail, no state of its own.
 * Hidden at ≥768px via CSS (the rail returns there). On /app the bottom is owned
 * by MobileWorkspace's content tabs, so ProductShell does not render this there.
 */
export function MobileSpaceNav({ activeSpace }: { activeSpace: string }) {
  const t = useTranslations();
  const lh = useLocalizedHref();

  const spaces = [
    { key: 'app', href: lh('/app'), label: 'App', Icon: CandlestickChart },
    { key: 'scanner', href: lh('/scanner/decrire'), label: t('nav.scanner'), Icon: Radar },
    { key: 'zones', href: lh('/zones'), label: t('nav.zones'), Icon: Layers },
    { key: 'actualites', href: lh('/actualites'), label: t('nav.calendar'), Icon: CalendarClock },
    { key: 'compte', href: lh('/compte'), label: t('nav.account'), Icon: Settings },
  ];

  return (
    <nav className="mspace" aria-label={t('app.sidebar.navAria')}>
      {spaces.map(({ key, href, label, Icon }) => {
        const isActive = activeSpace === key;
        return (
          <Link
            key={key}
            href={href}
            className={cn('mspace-item', isActive && 'on')}
            aria-current={isActive ? 'page' : undefined}
          >
            <Icon aria-hidden />
            <span>{label}</span>
          </Link>
        );
      })}
    </nav>
  );
}
