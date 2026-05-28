# Tests — webapp/

## Unit (Vitest + RTL + jsdom)

```bash
npm run test          # one-shot
npm run test:watch    # interactive
```

Configuration : `vitest.config.ts` + `vitest.setup.ts`. Inclut tout fichier
`**/*.{test,spec}.{ts,tsx}` hors `tests/e2e/`.

Couverture actuelle : `lib/insight-formatters.test.ts` (40 tests, 1 fichier).

## E2E (Playwright)

```bash
npx playwright install chromium    # one-time après npm install
npm run test:e2e                   # headless
npm run test:e2e:ui                # interactive
```

Configuration : `playwright.config.ts`. Lance automatiquement `next dev`
sur `localhost:3000` puis exécute les specs sur `chromium-desktop` + `mobile-iphone-12`.

Specs (`tests/e2e/`) :

- `landing.spec.ts` — hero + 3 cartes + track record + pricing 4 tiers
- `chatbot.spec.ts` — ouverture panneau + question suggérée + refus pédagogique + input libre
- `sections.spec.ts` — ouverture des accordions Structure / Régime / Expert
- `theme-and-pwa.spec.ts` — toggle dark/light + manifest.webmanifest + icon endpoints + redirect 302 EN/DE/ES

## Lighthouse CI (perf + a11y + bonnes pratiques)

```bash
npm install -D @lhci/cli
npm run build
npm run test:lhci
```

Configuration : `.lighthouserc.cjs`. Cible mobile ≥ 90 en perf + a11y +
best-practices (assertions configurées en `error` pour a11y, `warn` pour
les autres). Reports temporairement publiés en URL publique.

## CI GitHub Actions

`.github/workflows/webapp-ci.yml` orchestre :

1. **lint-and-unit** : typecheck + lint + vitest (toutes PRs)
2. **e2e** : playwright sur chromium (dépend de 1)
3. **lighthouse** : LHCI gate (dépend de 1)

Trigger : push sur main/institutional-overhaul ou PR touchant `webapp/`.
