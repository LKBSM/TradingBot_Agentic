import { render } from '@/components/test-utils';
import { describe, expect, it } from 'vitest';
import { JsonLd } from '../JsonLd';

/**
 * UI-18 — the JSON-LD payload is injected via dangerouslySetInnerHTML. If a
 * string value ever contained `</script>` it would break out of the tag. We
 * escape `<` to < so that can never happen.
 */
describe('JsonLd', () => {
  it('escapes `<` so a payload cannot break out of the <script> tag', () => {
    const { container } = render(
      <JsonLd data={{ name: 'evil</script><script>alert(1)</script>' }} />,
    );
    const script = container.querySelector('script[type="application/ld+json"]');
    expect(script).not.toBeNull();
    const html = script!.innerHTML;
    // No raw closing tag survives…
    expect(html).not.toContain('</script>');
    // …it is escaped instead, and the value is still valid JSON.
    expect(html).toContain('\\u003c');
    expect(() => JSON.parse(html)).not.toThrow();
    expect(JSON.parse(html).name).toContain('</script>');
  });

  it('emits parseable JSON for a normal payload', () => {
    const { container } = render(
      <JsonLd data={{ '@type': 'Thing', name: 'MIA' }} />,
    );
    const script = container.querySelector('script[type="application/ld+json"]');
    expect(JSON.parse(script!.innerHTML)).toEqual({
      '@type': 'Thing',
      name: 'MIA',
    });
  });
});
