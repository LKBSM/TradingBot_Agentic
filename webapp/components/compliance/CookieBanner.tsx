'use client';

import * as React from 'react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { cn } from '@/lib/utils';

/**
 * Cookie consent banner — CNIL "Lignes directrices et recommandation"
 * (mars 2020 + maj 2025). Four categories : nécessaires (toujours actives,
 * pas de consentement requis), fonctionnels, analytiques, marketing.
 *
 * V1 stores the choices in localStorage under `mia.cookie-consent.v1` so
 * the page doesn't re-prompt on every visit. The actual loading of the
 * tracking scripts (Plausible, Sentry, etc.) is gated on the categories
 * via the `useCookieConsent` hook — empty in V1 because we ship zero
 * trackers.
 *
 * LEGAL-PENDING: the textual descriptions below are placeholders pending
 * the legal terminal review (CNIL + RGPD compliant wording). The
 * structure (4 categories, granular toggles, refus aussi facile
 * qu'accepter) is correct by construction.
 */

const STORAGE_KEY = 'mia.cookie-consent.v1';

interface CookieConsent {
  necessary: true; // always true — required for the site to function
  functional: boolean;
  analytics: boolean;
  marketing: boolean;
  /** ISO timestamp of when the choice was made. */
  decidedAt: string;
}

const ALL_CONSENT: CookieConsent = {
  necessary: true,
  functional: true,
  analytics: true,
  marketing: true,
  decidedAt: '',
};

const NO_CONSENT: CookieConsent = {
  necessary: true,
  functional: false,
  analytics: false,
  marketing: false,
  decidedAt: '',
};

function readConsent(): CookieConsent | null {
  if (typeof window === 'undefined') return null;
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<CookieConsent>;
    if (typeof parsed.decidedAt !== 'string') return null;
    return {
      necessary: true,
      functional: Boolean(parsed.functional),
      analytics: Boolean(parsed.analytics),
      marketing: Boolean(parsed.marketing),
      decidedAt: parsed.decidedAt,
    };
  } catch {
    return null;
  }
}

function persistConsent(consent: CookieConsent): void {
  if (typeof window === 'undefined') return;
  window.localStorage.setItem(
    STORAGE_KEY,
    JSON.stringify({ ...consent, decidedAt: new Date().toISOString() }),
  );
}

export function CookieBanner() {
  const [mounted, setMounted] = React.useState(false);
  const [open, setOpen] = React.useState(false);
  const [showDetails, setShowDetails] = React.useState(false);
  const [draft, setDraft] = React.useState<CookieConsent>(NO_CONSENT);

  React.useEffect(() => {
    setMounted(true);
    const existing = readConsent();
    if (!existing) setOpen(true);
    else setDraft(existing);
  }, []);

  if (!mounted || !open) return null;

  function acceptAll() {
    persistConsent(ALL_CONSENT);
    setOpen(false);
  }

  function rejectAll() {
    persistConsent(NO_CONSENT);
    setOpen(false);
  }

  function saveChoice() {
    persistConsent(draft);
    setOpen(false);
  }

  return (
    <div
      role="dialog"
      aria-modal="false"
      aria-labelledby="cookie-banner-title"
      className="fixed inset-x-2 bottom-2 z-50 sm:inset-x-auto sm:bottom-4 sm:right-4 sm:max-w-md"
      data-legal-pending="cookie-banner-wording"
    >
      <Card className="border-border/80 p-4 shadow-xl sm:p-5">
        <h2 id="cookie-banner-title" className="text-sm font-semibold">
          Cookies &amp; vie privée
        </h2>
        {/* LEGAL-PENDING: wording final à valider par le terminal légal. */}
        <p className="mt-2 text-xs leading-relaxed text-muted-foreground">
          Nous utilisons des cookies pour faire fonctionner le site
          (nécessaires) et, sous réserve de votre consentement, pour mémoriser
          vos préférences, mesurer l&apos;audience et améliorer le produit.
          Vous pouvez accepter, refuser ou personnaliser à tout moment.
        </p>

        {showDetails && (
          <ul className="mt-3 space-y-2 text-xs">
            <CookieRow
              id="necessary"
              label="Nécessaires"
              description="Indispensables au fonctionnement du site (préférence de thème, navigation). Toujours actifs."
              checked
              disabled
            />
            <CookieRow
              id="functional"
              label="Fonctionnels"
              description="Mémorisent vos préférences (langue, sections dépliées, état du chatbot)."
              checked={draft.functional}
              onChange={(v) => setDraft((d) => ({ ...d, functional: v }))}
            />
            <CookieRow
              id="analytics"
              label="Analytiques"
              description="Statistiques d'usage anonymisées pour améliorer le produit (V2 : Plausible self-hosted, sans transfert hors UE)."
              checked={draft.analytics}
              onChange={(v) => setDraft((d) => ({ ...d, analytics: v }))}
            />
            <CookieRow
              id="marketing"
              label="Marketing"
              description="Personnalisation d'éventuelles communications. Aucun outil actif en V1."
              checked={draft.marketing}
              onChange={(v) => setDraft((d) => ({ ...d, marketing: v }))}
            />
          </ul>
        )}

        <div className="mt-4 flex flex-wrap gap-2">
          <Button size="sm" onClick={acceptAll} className="flex-1 sm:flex-none">
            Tout accepter
          </Button>
          <Button
            size="sm"
            variant="outline"
            onClick={rejectAll}
            className="flex-1 sm:flex-none"
          >
            Tout refuser
          </Button>
          <Button
            size="sm"
            variant="ghost"
            onClick={() => {
              if (showDetails) saveChoice();
              else setShowDetails(true);
            }}
            className="flex-1 sm:flex-none"
          >
            {showDetails ? 'Enregistrer mon choix' : 'Personnaliser'}
          </Button>
        </div>
      </Card>
    </div>
  );
}

function CookieRow({
  id,
  label,
  description,
  checked,
  onChange,
  disabled = false,
}: {
  id: string;
  label: string;
  description: string;
  checked: boolean;
  onChange?: (next: boolean) => void;
  disabled?: boolean;
}) {
  return (
    <li className="rounded-md border border-border/60 p-2">
      <label className={cn('flex items-start gap-2', !disabled && 'cursor-pointer')}>
        <input
          id={`cookie-${id}`}
          type="checkbox"
          checked={checked}
          disabled={disabled}
          onChange={(e) => onChange?.(e.target.checked)}
          className="mt-0.5 h-4 w-4 shrink-0 rounded border-input accent-primary disabled:opacity-60"
        />
        <span className="flex flex-col">
          <span className="text-xs font-medium text-foreground">{label}</span>
          <span className="text-[11px] leading-snug text-muted-foreground">
            {description}
          </span>
        </span>
      </label>
    </li>
  );
}

/**
 * Hook used by downstream code to gate analytics/marketing scripts.
 *
 *   const { analytics } = useCookieConsent();
 *   if (analytics) loadPlausible();
 */
export function useCookieConsent(): {
  necessary: boolean;
  functional: boolean;
  analytics: boolean;
  marketing: boolean;
  ready: boolean;
} {
  const [state, setState] = React.useState({
    necessary: true,
    functional: false,
    analytics: false,
    marketing: false,
    ready: false,
  });

  React.useEffect(() => {
    const existing = readConsent();
    if (existing) {
      setState({
        necessary: true,
        functional: existing.functional,
        analytics: existing.analytics,
        marketing: existing.marketing,
        ready: true,
      });
    } else {
      setState((s) => ({ ...s, ready: true }));
    }
  }, []);

  return state;
}
