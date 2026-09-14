import { render } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { renderLegalMarkdown } from '../render-markdown';

/**
 * LEG-1 — the legal documents are hard-wrapped at ~80 columns for review, so
 * sentences straddle line breaks constantly. The renderer used to treat every
 * SOURCE line as a rendered line, which broke two things on the public pages:
 *
 *   1. a `**bold**` span crossing a wrap printed its asterisks literally
 *      (« est **responsable de la protection » was visible in production copy);
 *   2. the wrapped continuation of a bullet fell OUT of the list and became a
 *      stray paragraph underneath it.
 *
 * Markdown joins the lines of a paragraph; these tests hold that.
 */

function textOf(markdown: string): string {
  const { container } = render(<>{renderLegalMarkdown(markdown)}</>);
  return container.textContent ?? '';
}

function dom(markdown: string): HTMLElement {
  const { container } = render(<>{renderLegalMarkdown(markdown)}</>);
  return container;
}

describe('renderLegalMarkdown — wrapped source', () => {
  it('joins the lines of a paragraph instead of breaking them', () => {
    const html = dom('Une phrase coupée\npar un retour à la ligne.');
    expect(html.querySelectorAll('p')).toHaveLength(1);
    expect(html.querySelector('p')?.textContent).toBe(
      'Une phrase coupée par un retour à la ligne.',
    );
    expect(html.querySelectorAll('br')).toHaveLength(0);
  });

  it('renders a bold span that straddles a line break, without stray asterisks', () => {
    const html = dom('Il est **responsable de la protection\ndes renseignements**.');
    expect(html.textContent).not.toContain('*');
    expect(html.querySelector('strong')?.textContent).toBe(
      'responsable de la protection des renseignements',
    );
  });

  it('renders an italic span that straddles a line break', () => {
    const html = dom('Voir la *Loi sur la protection\ndu consommateur*.');
    expect(html.textContent).not.toContain('*');
    expect(html.querySelector('em')?.textContent).toBe(
      'Loi sur la protection du consommateur',
    );
  });

  it('keeps a wrapped bullet inside its list item', () => {
    const html = dom(
      ['- ton **adresse courriel** et ton identifiant ;', '  jamais ta carte ;', '- une autre puce ;'].join('\n'),
    );
    const items = html.querySelectorAll('li');
    expect(items).toHaveLength(2);
    expect(items[0]?.textContent).toBe(
      'ton adresse courriel et ton identifiant ; jamais ta carte ;',
    );
    // The continuation must NOT have escaped into a paragraph after the list.
    expect(html.querySelectorAll('p')).toHaveLength(0);
  });

  it('joins the lines of a blockquote', () => {
    const html = dom('> Version française, qui fait foi.\n> En cas de divergence.');
    expect(html.querySelector('blockquote')?.textContent).toBe(
      'Version française, qui fait foi. En cas de divergence.',
    );
  });

  it('still separates paragraphs on a blank line', () => {
    const html = dom('Premier paragraphe.\n\nSecond paragraphe.');
    expect(html.querySelectorAll('p')).toHaveLength(2);
  });

  it('still renders headings, and never uses innerHTML', () => {
    const html = dom('# Titre\n\n## 1. Section\n\nDu texte.');
    expect(html.querySelector('h1')?.textContent).toBe('Titre');
    expect(html.querySelector('h2')?.textContent).toBe('1. Section');
    // Trusted source, but escaped all the same.
    expect(textOf('Du <script>alert(1)</script> texte.')).toContain(
      '<script>alert(1)</script>',
    );
    expect(dom('Du <script>alert(1)</script> texte.').querySelector('script')).toBeNull();
  });
});

describe('renderLegalMarkdown — the real published documents', () => {
  // All SIX, not just the French ones: the English clause 4 wraps mid-bold
  // (« **United\nStates** »), which is exactly the case that used to leak
  // asterisks onto the page.
  const STEMS = ['conditions-utilisation', 'politique-confidentialite'] as const;
  const LOCALES = ['fr', 'en', 'es'] as const;

  for (const stem of STEMS) {
    for (const locale of LOCALES) {
      it(`${stem}.${locale} renders with no stray markdown syntax`, async () => {
        const { readFileSync } = await import('node:fs');
        const path = await import('node:path');
        const file = path.resolve(
          __dirname, '..', '..', '..', '..', 'docs', 'legal', `${stem}.${locale}.md`,
        );
        const body = dom(readFileSync(file, 'utf-8')).textContent ?? '';
        expect(body).not.toContain('**');
        expect(body).not.toContain('\n- ');
        expect(body.length).toBeGreaterThan(1000);
      });
    }
  }

  it('the terms show the territory as one phrase, bold intact', async () => {
    const { readFileSync } = await import('node:fs');
    const path = await import('node:path');
    const file = path.resolve(
      __dirname, '..', '..', '..', '..', 'docs', 'legal', 'conditions-utilisation.en.md',
    );
    const html = dom(readFileSync(file, 'utf-8'));
    const bold = Array.from(html.querySelectorAll('strong')).map((n) => n.textContent);
    expect(bold).toContain('United States');
  });
});
