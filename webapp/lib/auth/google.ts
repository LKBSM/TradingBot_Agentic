/**
 * Google sign-in client. The flow itself is server-driven redirects
 * (`/api/auth/google/start` → Google → `/callback`); the frontend only needs to
 * (a) know whether to SHOW the button and (b) finalize a new-user signup after
 * the consent step.
 */

export const GOOGLE_START_URL = '/api/auth/google/start';

/** Whether Google sign-in is configured server-side (button shown only if so). */
export async function fetchGoogleEnabled(): Promise<boolean> {
  try {
    const res = await fetch('/api/auth/google/config', {
      headers: { accept: 'application/json' },
      credentials: 'same-origin',
    });
    if (!res.ok) return false;
    const body = (await res.json()) as { enabled?: boolean };
    return body.enabled === true;
  } catch {
    return false;
  }
}

export interface GoogleFinalizeInput {
  token: string;
  // PAY-3: no username — derived from the Google-verified email server-side.
  age_confirmed: boolean;
  accept_terms: boolean;
  accept_privacy: boolean;
}

/** Complete a Google signup after the consent step. Throws on non-2xx. */
export async function googleFinalize(input: GoogleFinalizeInput): Promise<void> {
  const res = await fetch('/api/auth/google/finalize', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    credentials: 'same-origin',
    body: JSON.stringify(input),
  });
  if (!res.ok) {
    let detail = 'Google sign-in failed.';
    try {
      const body = (await res.json()) as { detail?: unknown };
      if (typeof body?.detail === 'string') detail = body.detail;
    } catch {
      /* keep default */
    }
    throw new Error(detail);
  }
}
