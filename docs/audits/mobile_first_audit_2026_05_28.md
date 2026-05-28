# DG-103 — Mobile-first responsive audit (2026-05-28)

**Sprint** : 6 (Infra deploy + Analytique)
**Scope** : Verify the M.I.A. webapp meets Lighthouse mobile ≥ 90 *in principle* before the operator can run a full Lighthouse CI in prod.
**Target** : Lighthouse mobile Performance ≥ 90, Accessibility ≥ 90, Best Practices ≥ 90, SEO ≥ 90.

This audit walks the current webapp source against the standard Lighthouse mobile checklist. The actual numeric Lighthouse run happens against the deployed Vercel preview — this doc tells the operator what should pass and what to watch.

---

## 1. Viewport meta — ✅

`webapp/app/[locale]/layout.tsx:78-86` exports a `Viewport` object with `width: 'device-width'`, `initialScale: 1`, `maximumScale: 5`. Setting `maximumScale=5` (rather than the once-recommended `1`) preserves user zoom, which is an A11y requirement (WCAG 1.4.4).

## 2. Tailwind responsive breakpoints — ✅

All page-level surfaces use the standard Tailwind responsive prefixes:

- `mx-auto max-w-3xl px-4` on the track-record page → mobile-first centered container.
- `grid grid-cols-1 sm:grid-cols-3 gap-4` on the metrics grid → stacks on mobile, three columns from sm: (≥640px) up.
- Chatbot panel uses fixed-position with mobile-optimised slide-in (`fixed inset-x-0 bottom-0 sm:right-6 sm:bottom-6 sm:left-auto`) — verified in `webapp/components/chat/ChatPanel.tsx`.

No hardcoded `width: 1200px` or other desktop-only widths in the page templates.

## 3. Touch target sizes — ✅

Tailwind `py-2 px-3` is ~40px tall by default — under the 44×44 px Apple HIG minimum *but* still above the 24×24 WCAG 2.5.5 (Level AAA) minimum. We add explicit `min-h-[44px]` to the chatbot CTA button and the suggested-questions chips. Spot-checked at the hero + chatbot panel.

## 4. Font scaling — ✅

All typography uses Tailwind's responsive scale (`text-sm md:text-base`, `text-3xl md:text-4xl`). No `font-size: 14px` literals that would pin text size and break dynamic-type users.

## 5. Image responsiveness — ⚠️ Watch

V1 ships zero remote images (`webapp/next.config.js` images.remotePatterns is `[]`). Hero uses an SVG sparkline rendered server-side at fixed `width=720 height=200` with `class="w-full h-auto"` so it scales to container. Marketing assets added in Phase 2 must use `<Image>` with `sizes` set, or the Lighthouse perf score will drop.

## 6. JS bundle size — ⚠️ Watch

The landing page first-load JS measured 143 kB at commit `2a45967` (per session notes). Adding `@sentry/nextjs` in Sprint 4 inflates this by ~30 kB; the tradeoff is acceptable for error capture. Plausible analytics adds < 1 kB. Re-measure after Sprint 6 deploy.

## 7. CSP / Permissions-Policy — ✅

`next.config.js` ships a strict CSP (`default-src 'self'`), HSTS in prod, and `Permissions-Policy` denying all sensitive features except `fullscreen` and `web-share`. This is a Lighthouse "Best Practices" win.

## 8. PWA manifest — ✅

Verified by Sprint V2.0-V2.4 work — manifest + icons + apple-touch-icon meta all wired.

## 9. Accessibility quick-pass — ✅

- All form inputs in the chatbot have associated `<label>` (verified via Vitest accessibility-ish tests).
- The track-record SVG has `role="img"` and `aria-label` — screen-reader friendly.
- Headings use a single `<h1>` per page and nested `<h2>`/`<h3>` with `id`-anchored `aria-labelledby` on sections.
- Color contrast: Tailwind palette tokens (`text-muted-foreground` etc.) chosen for AA contrast against the canvas. Re-verify in dark mode after Sprint 6 with Lighthouse contrast checker.

## 10. SEO — ✅

- `metadata` exports declare `title`, `description`, `openGraph` and `twitter` blocks.
- Locale-specific routing via `next-intl` so each `[locale]` page has a unique URL.
- `robots.txt` and `sitemap.xml` are scaffolded under `webapp/app/`.

---

## Lighthouse runbook (post-deploy)

Once Vercel preview is live:

```bash
npx lighthouse https://preview.mia.markets \
  --preset=desktop --output=html --output-path=./lighthouse-desktop.html
npx lighthouse https://preview.mia.markets \
  --form-factor=mobile --throttling.cpuSlowdownMultiplier=4 \
  --output=html --output-path=./lighthouse-mobile.html
```

Acceptance gate: each of Performance / Accessibility / Best Practices / SEO ≥ 90 mobile.

If Performance < 90:
- check the first-load JS again (Sentry + Plausible should still leave headroom);
- verify image lazy-loading on hero;
- check `connect-src` CSP isn't blocking the chatbot stream (would cause noisy console errors that count as a "Best Practices" deduction).

If Accessibility < 90:
- check color contrast in dark mode;
- ensure every interactive element has a discernible name (most common deduction is the chatbot send button when icon-only).

---

## Sign-off

Static audit passed against the V1 source. **Lighthouse run is operator-side once the prod preview is reachable.** Re-open this audit if any value drops below 90.
