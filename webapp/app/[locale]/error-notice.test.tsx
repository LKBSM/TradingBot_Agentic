import { render } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import NotFound from './not-found';
import LocaleError from './error';

/**
 * The error boundaries render OUTSIDE the (site)/(product) route groups, so they
 * inherit NEITHER the ProductShell rail disclaimer NOR the site Footer disclaimer
 * that cover every content page. They are the only surfaces that would otherwise
 * show ZERO regulatory notice — this guard fails if the hardcoded FR notice ever
 * disappears from them. (global-error.tsx renders its own <html>; it is verified
 * by inspection, not mounted here.)
 */

const NOTICE = /Ni signal de trading, ni conseil en investissement\.\s*18\+\./;

describe('error boundaries carry the regulatory notice — never zero', () => {
  it('the 404 page shows the notice', () => {
    const { container } = render(<NotFound />);
    expect(container.textContent).toMatch(NOTICE);
  });

  it('the route error boundary shows the notice', () => {
    const { container } = render(
      <LocaleError error={Object.assign(new Error('x'), { digest: undefined })} reset={() => {}} />,
    );
    expect(container.textContent).toMatch(NOTICE);
  });
});
