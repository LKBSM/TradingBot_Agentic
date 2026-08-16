'use client';

import { useRouter, useSearchParams } from 'next/navigation';
import { useLocale, useTranslations } from 'next-intl';
import * as React from 'react';
import { Check, CreditCard, ShieldCheck } from 'lucide-react';
import {
  BillingError,
  fetchPricing,
  fetchSubscription,
  openPortal,
  startCheckout,
  type Plan,
  type Subscription,
} from '@/lib/billing/api-client';
import { useAuth } from '@/lib/auth/store';
import { useLocalizedHref } from '@/lib/i18n/href';
import { PRICING } from '@/lib/pricing.generated';
import { Button } from '@/components/ui/button';
import { FormError, FormSuccess } from '@/components/auth/fields';

const ACTIVE_STATUSES = new Set(['active', 'trialing']);

/**
 * The app-facing subscription states (PAY-1), derived from the Stripe status +
 * ``cancel_at_period_end``.
 */
type SubState = 'none' | 'active' | 'canceling' | 'grace' | 'suspended' | 'expired';

function deriveState(sub: Subscription | null): SubState {
  const status = sub?.status ?? null;
  if (!status) return 'none';
  if (ACTIVE_STATUSES.has(status)) {
    return sub?.cancel_at_period_end ? 'canceling' : 'active';
  }
  if (status === 'past_due') return 'grace';
  if (status === 'suspended') return 'suspended';
  return 'expired';
}

/** Whether a derived state currently grants product access (grace still does). */
function hasAccessState(s: SubState): boolean {
  return s === 'active' || s === 'canceling' || s === 'grace';
}

function stateHeading(state: SubState, t: (key: string) => string): string {
  switch (state) {
    case 'active':
    case 'canceling':
      return t('status.active');
    case 'grace':
      return t('status.pastDue');
    case 'suspended':
      return t('status.suspended');
    case 'expired':
      return t('status.expired');
    default:
      return t('status.none');
  }
}

function formatDate(epochSeconds: number | null, locale: string): string | null {
  if (!epochSeconds) return null;
  try {
    return new Date(epochSeconds * 1000).toLocaleDateString(locale, {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
    });
  } catch {
    return null;
  }
}

/**
 * Subscription panel — the single place an account activates or manages its
 * subscription. All payment UI is hosted by Stripe; this component only
 * redirects to URLs the backend returns.
 *
 * PAY-3b: for an account WITHOUT an active subscription there is no "no
 * subscription" resting state — paying is the only door in. The unsubscribed
 * view is a clean plan-choice ("activate your account") with the annual cadence
 * highlighted; the "current state" card only appears when there is a real
 * subscription to show.
 */
export function SubscriptionPanel() {
  const t = useTranslations('billing');
  const locale = useLocale();
  const { account, loading: authLoading } = useAuth();
  const router = useRouter();
  const lh = useLocalizedHref();
  const searchParams = useSearchParams();

  const [plans, setPlans] = React.useState<Plan[]>([]);
  const [sub, setSub] = React.useState<Subscription | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [busy, setBusy] = React.useState<string | null>(null);
  const [error, setError] = React.useState<string | null>(null);

  const checkoutStatus = searchParams.get('status');

  React.useEffect(() => {
    if (!authLoading && account === null) router.replace(lh('/connexion'));
  }, [authLoading, account, router, lh]);

  React.useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const [pricing, subscription] = await Promise.all([
          fetchPricing(),
          fetchSubscription(),
        ]);
        if (cancelled) return;
        setPlans(pricing.plans);
        setSub(subscription);
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof BillingError ? err.message : t('errorLoad'));
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    if (account) load();
    return () => {
      cancelled = true;
    };
  }, [account]);

  // After returning from Stripe Checkout the payment already succeeded, but
  // ACCESS is granted by the WEBHOOK, not by this redirect. Poll until active.
  const awaitingWebhook =
    checkoutStatus === 'success' && account !== null && !hasAccessState(deriveState(sub));

  React.useEffect(() => {
    if (!account || !awaitingWebhook) return;
    let cancelled = false;
    let tries = 0;
    let timer: ReturnType<typeof setTimeout>;
    const tick = async () => {
      tries += 1;
      try {
        const s = await fetchSubscription();
        if (cancelled) return;
        setSub(s);
        if (hasAccessState(deriveState(s))) {
          router.replace(lh('/app'));
          return;
        }
      } catch {
        /* transient — keep waiting for the webhook */
      }
      if (!cancelled && tries < 24) timer = setTimeout(tick, 2500);
    };
    timer = setTimeout(tick, 1500);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [account, awaitingWebhook, router, lh]);

  if (authLoading || account === null || loading) {
    return <p className="text-sm text-muted-foreground">{t('loading')}</p>;
  }

  // Confirming screen while the webhook lands (auto-redirects to /app on success).
  if (awaitingWebhook) {
    return (
      <div className="space-y-6">
        <h1 className="text-2xl font-semibold tracking-tight">{t('title')}</h1>
        <div
          className="flex flex-col items-center gap-4 rounded-2xl border border-border/60 bg-muted/10 p-10 text-center"
          aria-busy="true"
          aria-live="polite"
        >
          <div className="h-8 w-8 animate-spin rounded-full border-2 border-muted-foreground/30 border-t-primary" />
          <p className="text-sm text-foreground">{t('checkoutSuccess')}</p>
        </div>
      </div>
    );
  }

  async function onSubscribe(planKey: string) {
    setError(null);
    setBusy(planKey);
    try {
      const url = await startCheckout(planKey);
      window.location.href = url;
    } catch (err) {
      setError(err instanceof BillingError ? err.message : t('errorCheckout'));
      setBusy(null);
    }
  }

  async function onManage() {
    setError(null);
    setBusy('manage');
    try {
      const url = await openPortal();
      window.location.href = url;
    } catch (err) {
      setError(err instanceof BillingError ? err.message : t('errorPortal'));
      setBusy(null);
    }
  }

  const isOwner = account.role === 'owner';
  const state = deriveState(sub);
  const periodEnd = formatDate(sub?.current_period_end ?? null, locale);
  const currency = t('currency');
  const planKey = plans.find((p) => p.price_id === sub?.price_id)?.key ?? null;
  const nextAmount =
    planKey === 'MONTHLY' ? PRICING.monthly : planKey === 'ANNUAL' ? PRICING.annualPerYear : null;

  // Owner has unconditional access — no plans, no paywall.
  if (isOwner) {
    return (
      <div className="space-y-6">
        <h1 className="text-2xl font-semibold tracking-tight">{t('title')}</h1>
        <div className="inline-flex items-center gap-1.5 rounded-full border border-sentinel-warn/40 bg-sentinel-warn/10 px-3 py-1 text-xs font-medium text-sentinel-warn">
          <ShieldCheck className="h-3.5 w-3.5" aria-hidden />
          {t('ownerBadge')}
        </div>
      </div>
    );
  }

  // A real subscription to MANAGE (active / canceling / grace).
  if (state === 'active' || state === 'canceling' || state === 'grace') {
    let detail: string | null = null;
    if (state === 'active' && periodEnd) {
      detail =
        nextAmount !== null
          ? t('nextCharge', { date: periodEnd, amount: nextAmount, currency })
          : t('renewsOn', { date: periodEnd });
    } else if (state === 'canceling' && periodEnd) {
      detail = t('accessUntilNoRenewal', { date: periodEnd });
    } else if (state === 'grace') {
      detail = t('graceNotice');
    }
    return (
      <div className="space-y-8">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">{t('title')}</h1>
          <p className="text-sm text-muted-foreground">{t('intro')}</p>
        </div>
        <FormError message={error} />
        <section className="space-y-4 rounded-2xl border border-border/60 p-6">
          <div className="flex items-center gap-2">
            <span className="inline-flex h-2 w-2 rounded-full bg-sentinel-bull" aria-hidden />
            <h2 className="text-base font-medium text-foreground">{stateHeading(state, t)}</h2>
          </div>
          {detail && <p className="text-sm text-muted-foreground">{detail}</p>}
          <div className="space-y-2 pt-1">
            <Button variant="outline" onClick={onManage} disabled={busy !== null}>
              <CreditCard className="mr-2 h-4 w-4" aria-hidden />
              {t('manage')}
            </Button>
            <p className="text-xs text-muted-foreground">{t('manageHint')}</p>
          </div>
        </section>
      </div>
    );
  }

  // Unsubscribed: none / expired / suspended → the ACTIVATE (plan-choice) view.
  // No "no subscription" status card — paying is the only door in.
  const reactivating = state === 'expired' || state === 'suspended';
  const ctaLabel = reactivating ? t('reactivate') : t('subscribe');
  const savePerYear = PRICING.monthly * 12 - PRICING.annualPerYear;

  const mentions = [t('legalRenew'), t('legalTool'), t('legalRisk'), t('legalAge')];

  return (
    <div className="space-y-8">
      <header className="space-y-1.5">
        <h1 className="text-2xl font-semibold tracking-tight">
          {reactivating ? t('reactivateTitle') : t('activateTitle')}
        </h1>
        <p className="text-sm text-muted-foreground">
          {reactivating
            ? state === 'suspended'
              ? t('suspendedNotice')
              : t('expiredNotice')
            : t('activateIntro')}
        </p>
      </header>

      {checkoutStatus === 'success' && <FormSuccess message={t('checkoutSuccess')} />}
      {checkoutStatus === 'cancel' && <FormError message={t('checkoutCancel')} />}
      <FormError message={error} />

      {plans.length === 0 ? (
        <p className="text-sm text-muted-foreground">{t('noPlans')}</p>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2">
          {/* Monthly */}
          {plans.some((p) => p.key === 'MONTHLY') && (
            <PlanCard
              name={t('monthlyName')}
              amount={PRICING.monthly}
              currency={currency}
              perMonth={t('perMonth')}
              cta={ctaLabel}
              busy={busy === 'MONTHLY'}
              disabled={busy !== null}
              onClick={() => onSubscribe('MONTHLY')}
            />
          )}
          {/* Annual — highlighted */}
          {plans.some((p) => p.key === 'ANNUAL') && (
            <PlanCard
              name={t('annualName')}
              amount={PRICING.annualPerMonth}
              currency={currency}
              perMonth={t('perMonth')}
              note={t('annualBilledNote', { total: PRICING.annualPerYear, currency })}
              badge={t('bestValue')}
              save={savePerYear > 0 ? t('savePerYear', { amount: savePerYear, currency }) : undefined}
              highlighted
              cta={ctaLabel}
              busy={busy === 'ANNUAL'}
              disabled={busy !== null}
              onClick={() => onSubscribe('ANNUAL')}
            />
          )}
        </div>
      )}

      <p className="flex items-center gap-2 text-xs text-muted-foreground">
        <ShieldCheck className="h-4 w-4 shrink-0" aria-hidden />
        {t('securedByStripe')}
      </p>

      <ul className="space-y-1.5 border-t border-border/60 pt-4 text-xs text-muted-foreground">
        {mentions.map((m) => (
          <li key={m}>{m}</li>
        ))}
      </ul>
    </div>
  );
}

/** A single plan card. The highlighted one carries a "best value" badge + ring. */
function PlanCard({
  name,
  amount,
  currency,
  perMonth,
  note,
  badge,
  save,
  highlighted = false,
  cta,
  busy,
  disabled,
  onClick,
}: {
  name: string;
  amount: number;
  currency: string;
  perMonth: string;
  note?: string;
  badge?: string;
  save?: string;
  highlighted?: boolean;
  cta: string;
  busy: boolean;
  disabled: boolean;
  onClick: () => void;
}) {
  return (
    <div
      className={[
        'relative flex flex-col gap-4 rounded-2xl border p-6 transition',
        highlighted
          ? 'border-primary/60 bg-primary/[0.04] shadow-sm ring-1 ring-primary/20'
          : 'border-border/60 bg-muted/10 hover:border-border',
      ].join(' ')}
    >
      {badge && (
        <span className="absolute -top-2.5 right-5 rounded-full bg-primary px-2.5 py-0.5 text-[11px] font-semibold text-primary-foreground">
          {badge}
        </span>
      )}
      <div className="space-y-1">
        <h3 className="text-sm font-medium uppercase tracking-wider text-muted-foreground">
          {name}
        </h3>
        <p className="flex items-baseline gap-1.5">
          <span className="text-3xl font-semibold tracking-tight text-foreground">
            {amount} {currency}
          </span>
          <span className="text-sm text-muted-foreground">/ {perMonth}</span>
        </p>
        {note && <p className="text-xs text-muted-foreground">{note}</p>}
        {save && (
          <p className="inline-flex items-center gap-1 text-xs font-medium text-sentinel-bull">
            <Check className="h-3.5 w-3.5" aria-hidden />
            {save}
          </p>
        )}
      </div>
      <Button className="mt-auto w-full" onClick={onClick} disabled={disabled}>
        {busy ? '…' : cta}
      </Button>
    </div>
  );
}
