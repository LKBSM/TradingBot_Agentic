import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { renderMarkdown } from '../markdown';

describe('renderMarkdown (chat bubble light Markdown)', () => {
  it('renders **bold** as <strong> without leaking asterisks', () => {
    const { container } = render(<div>{renderMarkdown('Voici un **point clé** ici.')}</div>);
    const strong = container.querySelector('strong');
    expect(strong).not.toBeNull();
    expect(strong!.textContent).toBe('point clé');
    expect(container.textContent).not.toContain('**');
  });

  it('renders *italic* and `code`', () => {
    const { container } = render(
      <div>{renderMarkdown('Un mot *en italique* et du `code`.')}</div>,
    );
    expect(container.querySelector('em')?.textContent).toBe('en italique');
    expect(container.querySelector('code')?.textContent).toBe('code');
    expect(container.textContent).not.toContain('`');
  });

  it('renders a bullet list as <ul><li> without raw dash markers', () => {
    const { container } = render(
      <div>
        {renderMarkdown('Structure :\n- BOS confirmé\n- FVG actif\n- Retest en cours')}
      </div>,
    );
    const items = container.querySelectorAll('ul > li');
    expect(items).toHaveLength(3);
    expect(items[0]!.textContent).toBe('BOS confirmé');
    // The leading "- " marker must not survive as text.
    expect(screen.queryByText(/^- BOS/)).toBeNull();
  });

  it('renders a numbered list as <ol><li>', () => {
    const { container } = render(
      <div>{renderMarkdown('1. Premier\n2. Deuxième')}</div>,
    );
    const items = container.querySelectorAll('ol > li');
    expect(items).toHaveLength(2);
    expect(items[1]!.textContent).toBe('Deuxième');
  });

  it('splits blank-line-separated blocks into separate paragraphs', () => {
    const { container } = render(
      <div>{renderMarkdown('Premier paragraphe.\n\nSecond paragraphe.')}</div>,
    );
    expect(container.querySelectorAll('p')).toHaveLength(2);
  });

  it('escapes content (no HTML injection)', () => {
    const { container } = render(
      <div>{renderMarkdown('Texte <img src=x onerror=alert(1)> fin')}</div>,
    );
    expect(container.querySelector('img')).toBeNull();
    expect(container.textContent).toContain('<img src=x onerror=alert(1)>');
  });
});
