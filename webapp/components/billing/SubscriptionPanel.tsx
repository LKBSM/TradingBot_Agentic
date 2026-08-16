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
import { useLocalizedHref } from '@/lib/i18n/href';
import { PRICING } from '@/lib/pricing.generated';
import { Button } from '@/components/ui/button';
import { FormError, FormSuccess } from '@/components/auth/fields';

const ACTIVE_STATUSES = new Set(['active', 'trialing']);

/**
 * The five app-facing subscription states (PAY-1), derived from the Stripe
 * status + ``cancel_at_period_end``. Each maps to a clear account-page display.
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
  // canceled / unpaid / incomplete / incomplete_expired
  return 'expired';
}

/**
 * Human label for a plan key (AUTH-15). The backend /pricing intentionally
 * returns only the key + price_id (no amount hard-coded server-side), so the
 * readable label + price live here — amounts come from `@/lib/pricing.generated`
 * (single source), the currency is always explicit US dollars. Unknown keys
 * fall back to the raw key rather than showing nothing.
 */
function planLabel(
  key: string,
  t: (k: string, values?: Record<string, string | number>) => string,
): string {
  const currency = t('currency');
  switch (key) {
    case 'MONTHLY':
      return t('planMonthly', { amount: PRICING.monthly, currency });
    case 'ANNUAL':
      return t('planAnnual', {
        total: PRICING.annualPerYear,
        perMonth: PRICING.annualPerMonth,
        currency,
      });
    default:
      return key;
  }
}

function stateHeading(
  state: SubState,
  t: (key: string) => string,
): string {
  switch (state) {
    case 'active':
      return t('status.active');
    case 'canceling':
      return t('status.active');
    case 'grace':
      return t('status.pastDue');
    case 'suspended':
      return t('status.suspended');
    case 'expired':
      return t('status.expired');
    case 'none':
    default:
      return t('status.none');
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
  const lh = useLocalizedHref();
  const searchParams = useSearchParams();

  const [plans, setPlans] = React.useState<Plan[]>([]);
  const [sub, setSub] = React.useState<Subscription | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [busy, setBusy] = React.useState(false);
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

  const isOwner = account.role === 'owner';
  const state = deriveState(sub);
  const periodEnd = formatDate(sub?.current_period_end ?? null, locale);
  const currency = t('currency');
  // Next charge amount, resolved from the subscription's price id via the
  // configured plans (amounts come from the single pricing source, never hard
  // coded, and always carry their currency).
  const planKey = plans.find((p) => p.price_id === sub?.price_id)?.key ?? null;
  const nextAmount =
    planKey === 'MONTHLY'
      ? PRICING.monthly
      : planKey === 'ANNUAL'
        ? PRICING.annualPerYear
        : null;
  // Plans are offered only when there is nothing active to manage.
  const showPlans = state === 'none' || state === 'expired' || state === 'suspended';

  // The one-line detail under the state heading, per state (PAY-1: the user
  // always knows where they stand and until when).
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
  } else if (state === 'suspended') {
    detail = t('suspendedNotice');
  } else if (state === 'expired') {
    detail = t('expiredNotice');
  }

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
        <div className="inline-flex items-center gap-1 rounded-full border border-sentinel-warn/40 bg-sentinel-warn/10 px-2.5 py-1 text-xs font-medium text-sentinel-warn">
          <ShieldCheck className="h-3.5 w-3.5" aria-hidden />
          {t('ownerBadge')}
        </div>
      )}

      <section className="space-y-3 rounded-lg border border-border/60 p-5">
        <h2 className="text-sm font-medium uppercase tracking-wider text-muted-foreground">
          {t('currentStateTitle')}
        </h2>
        <div className="flex items-center justify-between gap-3">
          <span className="text-foreground">{stateHeading(state, t)}</span>
        </div>
        {detail && <p className="text-xs text-muted-foreground">{detail}</p>}
        {sub?.status ? (
          <div className="space-y-2">
            <Button variant="outline" onClick={onManage} disabled={busy}>
              <CreditCard className="mr-2 h-4 w-4" aria-hidden />
              {t('manage')}
            </Button>
            <p className="text-xs text-muted-foreground">{t('manageHint')}</p>
          </div>
        ) : null}
      </section>

      {showPlans && (
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
                  <span className="font-medium text-foreground">{planLabel(plan.key, t)}</span>
                  <Button onClick={() => onSubscribe(plan.key)} disabled={busy}>
                    {state === 'expired' || state === 'suspended'
                      ? t('reactivate')
                      : t('subscribe')}
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
