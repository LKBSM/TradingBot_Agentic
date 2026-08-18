import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

/**
 * VZ-3 — the gauge must carry NO direction colour: green reads « buy », red
 * reads « sell », and this product refuses both. This guards the CSS block so a
 * future edit can't quietly reintroduce --bull/--bear (or a raw green/red) into
 * the marker, band, bracket, or labels.
 */
describe('proximity gauge — no direction colour', () => {
  it('the .zgauge CSS block references neither bull/bear nor raw green/red', () => {
    // vitest runs with cwd = webapp/.
    const cssPath = resolve(process.cwd(), 'components/shell/pages.css');
    const css = readFileSync(cssPath, 'utf-8');

    const start = css.indexOf('VZ-3 — the proximity gauge');
    expect(start).toBeGreaterThan(-1);
    // The gauge block ends where the Confluence section begins.
    const end = css.indexOf('/* Confluence */', start);
    expect(end).toBeGreaterThan(start);
    // Strip comments — only the actual declarations are judged (prose may say
    // « centred », « measured », etc. which harmlessly contain "red").
    const block = css
      .slice(start, end)
      .replace(/\/\*[\s\S]*?\*\//g, '')
      .toLowerCase();

    for (const forbidden of ['--bull', '--bear', 'green', 'red', '--pos', '--neg']) {
      expect(block, `gauge CSS must not contain "${forbidden}"`).not.toContain(forbidden);
    }
  });
});
