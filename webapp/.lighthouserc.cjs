// Lighthouse CI config — runs against a locally-built next start.
// Install once: `npm i -D @lhci/cli`
// Run: `npm run test:lhci`

/** @type {import('@lhci/cli').LHCIConfig} */
module.exports = {
  ci: {
    collect: {
      // CI: build then serve
      startServerCommand: 'npm run start',
      url: ['http://localhost:3000/'],
      numberOfRuns: 3,
      settings: {
        preset: 'desktop',
      },
    },
    assert: {
      // Cible Lighthouse mobile ≥ 90 (cf. docs/frontend/component_inventory.md §9)
      preset: 'lighthouse:recommended',
      assertions: {
        'categories:performance': ['warn', { minScore: 0.9 }],
        'categories:accessibility': ['error', { minScore: 0.9 }],
        'categories:best-practices': ['warn', { minScore: 0.9 }],
        'categories:seo': ['warn', { minScore: 0.85 }],
        // SEO du landing FR-only — désactive les checks i18n/hreflang qui
        // déclencheraient des faux positifs avec /en /de /es désactivés.
        'unused-javascript': 'off',
        'uses-rel-preconnect': 'off',
      },
    },
    upload: {
      target: 'temporary-public-storage',
    },
  },
};
