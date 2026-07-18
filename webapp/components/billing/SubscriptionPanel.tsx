'use client';

import { useRouter, useSearchParams } from 'next/navigation';
import { useLocale, useTranslations } from 'next-intl';
import * as React from 'react';
import { CreditCard, ShieldCheck } from 'lucide-react';
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
import { Button } from '@/components/ui/button';
import { FormError, FormSuccess } from '@/components/auth/fields';

const ACTIVE_STATUSES = new Set(['active', 'trialing']);

function statusLabel(
  status: string | null,
  t: (key: string) => string,
): string {
  switch (status) {
    case 'active':
      return t('status.active');
    case 'trialing':
      return t('status.trialing');
    case 'past_due':
      return t('status.pastDue');
    case 'canceled':
      return t('status.canceled');
    case null:
    case undefined:
      return t('status.none');
    default:
      return status;
  }
}

function formatDate(
  epochSeconds: number | null,
  locale: string,
): string | null {
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
 * Subscription management panel: shows the current state, lets the user start
 * Checkout for a configured plan, or open the Stripe Customer Portal to manage /
 * cancel. Redirects to /connexion when logged out. All payment UI is hosted by
 * Stripe — this component only redirects to URLs the backend returns.
 */
export function SubscriptionPanel() {
  const t = useTranslations('billing');
  const locale = useLocale();
  const { account, loading: authLoading } = useAuth();
  const router = useRouter();
  const searchParams = useSearchParams();

  const [plans, setPlans] = React.useState<Plan[]>([]);
  const [sub, setSub] = React.useState<Subscription | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);

  const checkoutStatus = searchParams.get('status');

  React.useEffect(() => {
    if (!authLoading && account === null) router.replace('/connexion');
  }, [authLoading, account, router]);

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
          setError(
            err instanceof BillingError
              ? err.message
              : t('errorLoad'),
          );
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

  if (authLoading || account === null || loading) {
    return <p className="text-sm text-muted-foreground">{t('loading')}</p>;
  }

  async function onSubscribe(planKey: string) {
    setError(null);
    setBusy(true);
    try {
      const url = await startCheckout(planKey);
      window.location.href = url;
    } catch (err) {
      setError(
        err instanceof BillingError
          ? err.message
          : t('errorCheckout'),
      );
      setBusy(false);
    }
  }

  async function onManage() {
    setError(null);
    setBusy(true);
    try {
      const url = await openPortal();
      window.location.href = url;
    } catch (err) {
      setError(
        err instanceof BillingError
          ? err.message
          : t('errorPortal'),
      );
      setBusy(false);
    }
  }

  const isActive = ACTIVE_STATUSES.has(sub?.status ?? '');
  const isOwner = account.role === 'owner';
  const periodEnd = formatDate(sub?.current_period_end ?? null, locale);

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">{t('title')}</h1>
        <p className="text-sm text-muted-foreground">
          {t('intro')}
        </p>
      </div>

      {checkoutStatus === 'success' && (
        <FormSuccess message={t('checkoutSuccess')} />
      )}
      {checkoutStatus === 'cancel' && (
        <FormError message={t('checkoutCancel')} />
      )}
      <FormError message={error} />

      {isOwner && (
        <div className="inline-flex items-center gap-1 rounded-full border border-amber-500/40 bg-amber-500/10 px-2.5 py-1 text-xs font-medium text-amber-600">
          <ShieldCheck className="h-3.5 w-3.5" aria-hidden />
          {t('ownerBadge')}
        </div>
      )}

      <section className="space-y-3 rounded-lg border border-border/60 p-5">
        <h2 className="text-sm font-medium uppercase tracking-wider text-muted-foreground">
          {t('currentStateTitle')}
        </h2>
        <div className="flex items-center justify-between gap-3">
          <span className="text-foreground">{statusLabel(sub?.status ?? null, t)}</span>
          {isActive && (
            <span className="text-xs text-muted-foreground">
              {sub?.cancel_at_period_end && periodEnd
                ? t('endsOn', { date: periodEnd })
                : periodEnd
                  ? t('renewsOn', { date: periodEnd })
                  : null}
            </span>
          )}
        </div>
        {sub?.status ? (
          <Button variant="outline" onClick={onManage} disabled={busy}>
            <CreditCard className="mr-2 h-4 w-4" aria-hidden />
            {t('manage')}
          </Button>
        ) : null}
      </section>

      {!isActive && (
        <section className="space-y-4 rounded-lg border border-border/60 p-5">
          <h2 className="text-sm font-medium uppercase tracking-wider text-muted-foreground">
            {t('choosePlanTitle')}
          </h2>
          {plans.length === 0 ? (
            <p className="text-sm text-muted-foreground">
              {t('noPlans')}
            </p>
          ) : (
            <ul className="space-y-3">
              {plans.map((plan) => (
                <li
                  key={plan.key}
                  className="flex items-center justify-between gap-3"
                >
                  <span className="font-medium text-foreground">{plan.key}</span>
                  <Button onClick={() => onSubscribe(plan.key)} disabled={busy}>
                    {t('subscribe')}
                  </Button>
                </li>
              ))}
            </ul>
          )}
        </section>
      )}
    </div>
  );
}
