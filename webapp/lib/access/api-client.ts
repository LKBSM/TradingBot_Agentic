/**
 * Access summary client — reads GET /api/access/me to learn what the current
 * account may see. Same-origin (the session cookie rides along automatically via
 * the `/api/*` rewrite). This drives the DISPLAY (lock vs. show); the server-side
 * guards remain the non-bypassable source of truth.
 */

export type AccessTier = 'visitor' | 'free' | 'subscriber' | 'owner';

export interface ChatQuota {
  /** Messages/day allowed; `null` ⇒ unlimited. */
  limit: number | null;
  used: number | null;
  remaining: number | null;
}

export interface AccessEntitlements {
  /** `null` ⇒ all instruments unlocked; otherwise the only allowed codes. */
  instruments: string[] | null;
  /** `null` ⇒ all timeframes unlocked; otherwise the only allowed codes. */
  timeframes: string[] | null;
  scanner: boolean;
  chat: ChatQuota;
}

export interface AccessSummary {
  authenticated: boolean;
  /** False during the personal-testing phase — everything is then open. */
  gate_enforced: boolean;
  /** Closed-beta login wall. When true the whole product API is 401 for anon. */
  beta_lockdown: boolean;
  /** Convenience: `beta_lockdown && !authenticated` — the UI must route to login. */
  must_login: boolean;
  tier: AccessTier;
  is_owner: boolean;
  has_full_access: boolean;
  entitlements: AccessEntitlements;
}

const ENDPOINT = '/api/access/me';

/** Fetch the caller's access summary. Never throws on 401 (returns the payload,
 * which says `authenticated:false`); throws only on transport/parse failure. */
export async function fetchAccess(signal?: AbortSignal): Promise<AccessSummary> {
  const res = await fetch(ENDPOINT, {
    method: 'GET',
    headers: { accept: 'application/json' },
    signal,
  });
  if (!res.ok) {
    throw new Error(`access summary unavailable (${res.status})`);
  }
  return (await res.json()) as AccessSummary;
}

/** Whether a given instrument/timeframe combo is unlocked for this account. */
export function comboAllowed(
  access: AccessSummary,
  instrument: string,
  timeframe: string,
): boolean {
  if (access.has_full_access) return true;
  const { instruments, timeframes } = access.entitlements;
  const instrumentOk =
    instruments === null || instruments.includes(instrument.toUpperCase());
  const timeframeOk =
    timeframes === null || timeframes.includes(timeframe.toUpperCase());
  return instrumentOk && timeframeOk;
}
