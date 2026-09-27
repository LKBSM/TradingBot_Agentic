/**
 * The Stripe customer portal, reachable without our backend.
 *
 * `POST /api/billing/portal` mints a one-click portal session, but it only
 * works for an account that already has a Stripe customer id, and it fails
 * whenever the backend or Stripe is having a bad minute. The login-link portal
 * below has neither dependency: Stripe emails the customer a sign-in link for
 * the address they paid with. It is the escape hatch that must ALWAYS be
 * visible on the account page — cancelling, updating a card or downloading an
 * invoice is never allowed to depend on our own uptime.
 *
 * Configured in the Stripe dashboard (Billing → Customer portal → login page).
 * Not a secret: it is a public page that authenticates by emailed link, and it
 * reveals nothing until the customer proves they own the address.
 */
export const STRIPE_PORTAL_LOGIN_URL =
  'https://billing.stripe.com/p/login/5kQeVd3D45gl7sW55G67S00';
