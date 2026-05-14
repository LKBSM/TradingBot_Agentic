# Smart Sentinel AI — Webapp (Phase 2B)

Next.js 14 + TypeScript + TailwindCSS + next-intl. Renders the public
landing, transparency page, glossary, pricing, dashboard, and AI chat.

## Local development

```bash
cd webapp
npm install
# Backend must run at NEXT_PUBLIC_API_BASE (default: http://localhost:8000)
npm run dev
```

Visit http://localhost:3000 — auto-redirects to /fr (default locale).

## Routes

- `/[locale]/` — landing page (UX-2B.1)
- `/[locale]/dashboard` — narrative cards + equity curve + regime timeline (UX-2B.1)
- `/[locale]/transparency` — paper-trading curve + stats (INFRA-2B.2 surface)
- `/[locale]/glossary` — 50 technical terms with tooltips (UX-2B.5)
- `/[locale]/pricing` — 4-tier FREE/LITE/PRO/PRO+ (INFRA-2B.3 surface)
- `/[locale]/chat` — Q&A chat with the LLM (UX-2B.3, fed by LLM-2B.5)

Locales: `fr` (default), `en`, `de`, `es` (UX-2B.6).

## Architecture

```
webapp/
  app/                  # Next.js app router
    [locale]/           # locale-scoped pages
      layout.tsx
      page.tsx          # landing
      dashboard/page.tsx
      transparency/page.tsx
      glossary/page.tsx
      pricing/page.tsx
      chat/page.tsx
    page.tsx            # root redirect to default locale
    globals.css
  components/           # shared React components
    Nav.tsx, Footer.tsx, NarrativeCard.tsx,
    EquityCurveChart.tsx, RegimeTimeline.tsx,
    ChatStream.tsx, Tooltip.tsx
  lib/
    api.ts              # SWR fetcher + postJson helper
    glossary.ts         # 50 glossary terms
  messages/             # i18n strings
    fr.json, en.json, de.json, es.json
  i18n.ts, middleware.ts
  next.config.js, tailwind.config.ts, tsconfig.json
```

## API integration

All `/api/*` requests are rewritten to the backend defined in
`NEXT_PUBLIC_API_BASE`. The four data endpoints the dashboard reads:

- `GET /api/v1/insights/history?limit=5` — latest narrative entries
- `GET /api/v1/forward-test/snapshot` — equity curve + stats
- `GET /api/v1/regime/timeline` — regime classifications
- `POST /api/v1/chat` — Q&A (LLM-2B.5)

## Compliance (UE 2024/2811)

Every locale's `disclaimer.short` string is rendered above the
dashboard + chat surfaces. Never use language like "buy" / "sell" /
"guaranteed" — the ComplianceChecker (RISK-2B.2) lints content before
publication.

## Mobile + PWA (UX-2B.4)

Tailwind breakpoints default to mobile-first (375px+). Navigation
collapses to a hamburger below `sm:` (not yet implemented — TODO).
PWA manifest + service worker are stubs to add when Lighthouse mobile
perf drops below 80 in production.

## Deployment

Frontend ships to Vercel; backend (FastAPI) ships to Railway. The
`next.config.js` rewrites `/api/*` to the backend so the same-origin
fetch works in both environments without CORS.
