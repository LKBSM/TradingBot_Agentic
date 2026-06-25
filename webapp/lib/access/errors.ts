/**
 * Shared access error for the freemium gate (mission ③).
 *
 * The feature endpoints answer 401 (login required) or 402 (subscription
 * required) when the gate is enforced. Feature clients throw this typed error so
 * components surface a CLEAN, actionable message (and can route to login / show
 * a paywall) instead of a generic "internal error". The message carried here is
 * the backend's own French upsell `detail` — never a raw server string.
 */
export class AccessError extends Error {
  readonly status: 401 | 402;

  constructor(status: 401 | 402, message: string) {
    super(message);
    this.status = status;
    this.name = 'AccessError';
  }

  /** 401 → the caller must authenticate; 402 → must subscribe. */
  get needsLogin(): boolean {
    return this.status === 401;
  }
}

/**
 * Map a 401/402 response to an {@link AccessError}, preferring the backend
 * `detail`. Returns `null` for any other status so callers keep their existing
 * branches. Consumes the response body only on a match.
 */
export async function accessErrorFromResponse(
  res: Response,
): Promise<AccessError | null> {
  if (res.status !== 401 && res.status !== 402) return null;
  let detail: string | null = null;
  try {
    const body = (await res.json()) as { detail?: unknown };
    if (typeof body?.detail === 'string') detail = body.detail;
  } catch {
    /* ignore — fall back to a default message below */
  }
  const fallback =
    res.status === 401
      ? 'Connecte-toi pour accéder à cette fonctionnalité.'
      : 'Cette fonctionnalité nécessite un abonnement.';
  return new AccessError(res.status, detail ?? fallback);
}
