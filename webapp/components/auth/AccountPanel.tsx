'use client';

import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { useLocale, useTranslations } from 'next-intl';
import * as React from 'react';
import { AuthError, updateProfile } from '@/lib/auth/api-client';
import { useAuth } from '@/lib/auth/store';
import { useLocalizedHref } from '@/lib/i18n/href';
import { useLocaleSwitch } from '@/lib/i18n/use-locale-switch';
import { LOCALE_LABELS, SUPPORTED_LOCALES } from '@/i18n';
import { useDesign } from '@/lib/theme/useDesign';
import type { ThemeMeta } from '@/lib/theme/themes';
import { FormError, FormSuccess } from './fields';

/**
 * Authenticated "Réglages" panel (UI-2) — restyled to the terminal reference:
 * a max-760 pagewrap of `.card` blocks (Compte · Apparence · Abonnement · Légal
 * & données), rendered inside ProductShell so the `.app-shell` tokens apply.
 *
 * REAL, wired flows are kept exactly as before: the session email display, the
 * email "Modifier" form (PATCH /api/auth/profile via updateProfile), logout, and
 * the legal document links. Rows with NO backend yet (change password, export
 * data, delete account) are rendered as DISABLED `.btn` with an honest
 * "à venir" note — no fake action is fired. Loi 25 will require real export /
 * delete before any billing is switched on.
 */
export function AccountPanel() {
  const t = useTranslations('app.account');
  const tAuth = useTranslations('auth');
  const { account, loading, probeFailed, logout, refresh } = useAuth();
  const router = useRouter();
  const lh = useLocalizedHref();

  const [editingEmail, setEditingEmail] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const [success, setSuccess] = React.useState<string | null>(null);
  const [saving, setSaving] = React.useState(false);
  const savingRef = React.useRef(false);

  // Redirect to login ONLY on a confirmed logged-out state (probe returned 401).
  // A network/5xx failure (probeFailed) must NOT eject a possibly-valid user —
  // we show a retry instead (AUTH-03/AUTH-09).
  React.useEffect(() => {
    if (!loading && account === null && !probeFailed) router.replace(lh('/connexion'));
  }, [loading, account, probeFailed, router, lh]);

  if (!loading && account === null && probeFailed) {
    return (
      <div className="pagewrap" style={{ maxWidth: 760 }}>
        <div className="card">
          <FormError message={tAuth('account.sessionError')} />
          <button type="button" className="btn" onClick={() => refresh()}>
            {tAuth('account.retry')}
          </button>
        </div>
      </div>
    );
  }

  if (loading || account === null) {
    return (
      <div className="pagewrap" style={{ maxWidth: 760 }}>
        <p className="text-sm text-muted-foreground">{tAuth('account.loading')}</p>
      </div>
    );
  }

  async function onSaveEmail(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    // Guard against a double submit (Enter + click) racing two requests before
    // React flips `saving` — the ref updates synchronously (AUTH-08).
    if (savingRef.current) return;
    savingRef.current = true;
    setError(null);
    setSuccess(null);
    const form = new FormData(e.currentTarget);
    setSaving(true);
    try {
      await updateProfile(String(form.get('email') ?? '').trim());
      // The server re-issued the session cookie on email change (AUTH-14), so a
      // follow-up refresh() reflects the new state without ever nulling the
      // account on a transient error (refresh is non-destructive now, AUTH-03).
      await refresh();
      setSuccess(tAuth('account.emailUpdated'));
      setEditingEmail(false);
    } catch (err) {
      setError(err instanceof AuthError ? err.message : tAuth('account.updateError'));
    } finally {
      savingRef.current = false;
      setSaving(false);
    }
  }

  async function onLogout() {
    await logout();
    // Invalidate the Router Cache before navigating so no cached authenticated
    // page survives the logout (inverse of the first-login fix — see LoginForm).
    router.refresh();
    router.push(lh('/'));
  }

  return (
    <div className="pagewrap" style={{ maxWidth: 760 }}>
      <div className="pghead">
        <div>
          <h1>{t('title')}</h1>
          <div className="sub">{t('subtitle')}</div>
        </div>
      </div>

      {/* ── Compte ─────────────────────────────────────────────────────── */}
      <div className="card">
        <div className="card-h">
          <UserIcon />
          <h3>{t('sectionAccount')}</h3>
        </div>

        <div className="setrow">
          <div className="sk">
            <b>{t('emailRow')}</b>
            <span>{account.email}</span>
          </div>
          <button
            type="button"
            className="btn"
            onClick={() => {
              setError(null);
              setSuccess(null);
              setEditingEmail((v) => !v);
            }}
          >
            {editingEmail ? t('editCancel') : t('edit')}
          </button>
        </div>

        {editingEmail && (
          <form onSubmit={onSaveEmail} className="setrow" noValidate>
            <div className="sk" style={{ display: 'grid', gap: 6 }}>
              <FormError message={error} />
              <input
                className="input"
                name="email"
                type="email"
                defaultValue={account.email}
                autoComplete="email"
                aria-label={t('emailRow')}
                required
              />
            </div>
            <button type="submit" className="btn primary" disabled={saving}>
              {saving ? tAuth('account.saving') : tAuth('account.save')}
            </button>
          </form>
        )}
        {success && !editingEmail && (
          <div className="setrow">
            <div className="sk">
              <FormSuccess message={success} />
            </div>
          </div>
        )}

        <div className="setrow">
          <div className="sk">
            <b>{t('passwordRow')}</b>
            <span>
              {t('passwordMasked')} · {t('comingSoon')}
            </span>
          </div>
          <button type="button" className="btn" disabled aria-disabled="true">
            {t('edit')}
          </button>
        </div>

        <div className="setrow">
          <div className="sk">
            <b>{t('sessionRow')}</b>
            <span>{t('sessionValue')}</span>
          </div>
          <button type="button" className="btn danger" onClick={onLogout}>
            {tAuth('account.logout')}
          </button>
        </div>
      </div>

      {/* ── Langue ─────────────────────────────────────────────────────── */}
      <div className="card">
        <div className="card-h">
          <GlobeIcon />
          <h3>{t('sectionLanguage')}</h3>
        </div>
        <LanguageGrid />
      </div>

      {/* ── Apparence ──────────────────────────────────────────────────── */}
      <div className="card">
        <div className="card-h">
          <PaletteIcon />
          <h3>{t('sectionAppearance')}</h3>
          <span className="badge2">{t('appearanceBadge')}</span>
        </div>
        <ThemeGrid />
      </div>

      {/* ── Abonnement ─────────────────────────────────────────────────── */}
      <div className="card">
        <div className="card-h">
          <CardIcon />
          <h3>{t('sectionSubscription')}</h3>
        </div>
        <div className="setrow">
          <div className="sk">
            <b>{t('planRow')}</b>
            <span>{t('planValue')}</span>
          </div>
          <span className="planbadge">{t('earlyAccessBadge')}</span>
        </div>
      </div>

      {/* ── Légal & données ────────────────────────────────────────────── */}
      <div className="card">
        <div className="card-h">
          <ShieldIcon />
          <h3>{t('sectionLegal')}</h3>
        </div>

        <div className="setrow">
          <div className="sk">
            <b>{t('docsRow')}</b>
            <span>{t('docsValue')}</span>
          </div>
          <Link href={lh('/conditions')} className="btn">
            {t('consult')}
          </Link>
        </div>

        <div className="setrow">
          <div className="sk">
            <b>{t('exportRow')}</b>
            <span>
              {t('exportValue')} · {t('comingSoon')}
            </span>
          </div>
          <button type="button" className="btn" disabled aria-disabled="true">
            {t('export')}
          </button>
        </div>

        <div className="setrow">
          <div className="sk">
            <b>{t('deleteRow')}</b>
            <span>
              {t('deleteValue')} · {t('comingSoon')}
            </span>
          </div>
          <button type="button" className="btn danger" disabled aria-disabled="true">
            {t('delete')}
          </button>
        </div>
      </div>
    </div>
  );
}

/**
 * Language selector in Réglages — the SAME switch action as the site nav's
 * LocaleToggle (`useLocaleSwitch`: writes NEXT_LOCALE + client-navigates the
 * current path under the chosen locale). One source of truth, so the choice is
 * persisted and applied on return, and the user stays on this page, translated.
 * Each chip is a real button with `aria-pressed`; the active one also carries a
 * check glyph (not colour alone).
 */
function LanguageGrid() {
  const t = useTranslations('nav');
  const active = useLocale();
  const switchLocale = useLocaleSwitch();
  return (
    <div className="fgrp" role="group" aria-label={t('language')}>
      {SUPPORTED_LOCALES.map((loc) => (
        <button
          key={loc}
          type="button"
          lang={loc}
          dir={loc === 'ar' ? 'rtl' : 'ltr'}
          className={active === loc ? 'fchip on' : 'fchip'}
          aria-pressed={active === loc}
          onClick={() => switchLocale(loc)}
        >
          {LOCALE_LABELS[loc]}
        </button>
      ))}
    </div>
  );
}

/**
 * The four theme tiles, wired to the existing design setter (`useDesign` →
 * next-themes, persists via `data-design` on <html>). Selecting a tile applies
 * immediately + persists; the active tile carries `.on` and shows "Actif" via
 * its `.chk`. Mount-guarded so the "on" ring never causes a hydration mismatch.
 */
function ThemeGrid() {
  const t = useTranslations('appearance');
  const { design, designs, setDesign } = useDesign();
  const [mounted, setMounted] = React.useState(false);
  React.useEffect(() => setMounted(true), []);
  const active = mounted ? design : null;

  return (
    <div className="themegrid" role="radiogroup" aria-label={t('groupLabel')}>
      {designs.map((theme) => (
        <ThemeTile
          key={theme.id}
          theme={theme}
          description={t(`descriptions.${theme.id}`)}
          activeLabel={t('active')}
          selected={active === theme.id}
          onSelect={() => setDesign(theme.id)}
        />
      ))}
    </div>
  );
}

function ThemeTile({
  theme,
  description,
  activeLabel,
  selected,
  onSelect,
}: {
  theme: ThemeMeta;
  description: string;
  activeLabel: string;
  selected: boolean;
  onSelect: () => void;
}) {
  const { bg, panel, accent } = theme.swatch;
  return (
    <button
      type="button"
      role="radio"
      aria-checked={selected}
      data-th={theme.id}
      onClick={onSelect}
      className={selected ? 'themetile on' : 'themetile'}
    >
      <span className="chk">{activeLabel}</span>
      <div className="sw3" aria-hidden>
        <i style={{ background: bg }} />
        <i style={{ background: panel }} />
        <i style={{ background: accent }} />
      </div>
      <div className="tt">
        <b>{theme.name}</b>
        <span>{description}</span>
      </div>
    </button>
  );
}

/* ── Inline card-head icons (stroke via `.card-h svg`) ─────────────────── */
function UserIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden>
      <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
      <circle cx="12" cy="7" r="4" />
    </svg>
  );
}
function GlobeIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden>
      <circle cx="12" cy="12" r="10" />
      <path d="M2 12h20" />
      <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10Z" />
    </svg>
  );
}
function PaletteIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden>
      <circle cx="13.5" cy="6.5" r="1.5" />
      <circle cx="17.5" cy="10.5" r="1.5" />
      <circle cx="8.5" cy="7.5" r="1.5" />
      <circle cx="6.5" cy="12.5" r="1.5" />
      <path d="M12 2a10 10 0 0 0 0 20c1.1 0 2-.9 2-2 0-.5-.2-1-.5-1.3-.3-.4-.5-.8-.5-1.2 0-1.1.9-2 2-2h2.3c2 0 3.7-1.6 3.7-3.7A9.6 9.6 0 0 0 12 2Z" />
    </svg>
  );
}
function CardIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden>
      <rect x="2" y="5" width="20" height="14" rx="2" />
      <path d="M2 10h20" />
    </svg>
  );
}
function ShieldIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden>
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10Z" />
    </svg>
  );
}
