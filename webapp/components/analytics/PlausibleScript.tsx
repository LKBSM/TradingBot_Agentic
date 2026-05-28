/**
 * DG-160 — Plausible analytics script loader.
 *
 * Renders a `<Script>` tag pointing at the Plausible instance defined by
 * ``NEXT_PUBLIC_PLAUSIBLE_DOMAIN`` + ``NEXT_PUBLIC_PLAUSIBLE_SRC``. When
 * either var is missing, the component is a server-side no-op so dev
 * and CI never load any analytics script.
 *
 * Operator config (V1 = self-hosted on Fly.io):
 *   NEXT_PUBLIC_PLAUSIBLE_DOMAIN=mia.markets
 *   NEXT_PUBLIC_PLAUSIBLE_SRC=https://plausible.internal.mia.markets/js/script.js
 *
 * Mount this component once in the root layout — multiple mounts would
 * record duplicate pageviews.
 */
import Script from 'next/script';

export function PlausibleScript() {
  const domain = process.env.NEXT_PUBLIC_PLAUSIBLE_DOMAIN;
  const src = process.env.NEXT_PUBLIC_PLAUSIBLE_SRC;
  if (!domain || !src) return null;
  return (
    <Script
      defer
      data-domain={domain}
      src={src}
      strategy="afterInteractive"
    />
  );
}
