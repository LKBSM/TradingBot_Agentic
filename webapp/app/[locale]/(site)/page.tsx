import { HashScroll } from '@/components/a11y/HashScroll';
import { HomeLanding } from '@/components/landing/lp1/HomeLanding';

/**
 * LP-1 — commercial home page. Structure mirrors the reference maquette
 * (docs/design/reference-accueil.html) but every figure and label is corrected
 * to the real product: 2 markets, 6 timeframes, 12 combinations, 22 conditions.
 *
 *   · Hero + real stats banner (single source: lib/landing/stats.ts)
 *   · Five tool feature rows (App / Scanner / Zones / News / M.I.A)
 *   · "Essaie sans créer de compte" — five interactive demos, no network
 *   · How it works · Who it's for · What sets us apart · Pricing · FAQ · CTA
 *
 * The whole page is static (illustration data only) — no API call. Nav + Footer
 * come from the (site) layout.
 */
export default function LandingPage() {
  return (
    <>
      <HashScroll />
      <HomeLanding />
    </>
  );
}
