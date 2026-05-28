import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ChatMessage } from './ChatMessage';

describe('ChatMessage — compliance badge (DG-112)', () => {
  it('does not render a badge when compliance is undefined', () => {
    render(<ChatMessage role="assistant" text="Hello" />);
    expect(screen.queryByTestId('compliance-badge')).toBeNull();
  });

  it('does not render a badge on a user bubble even if compliance is set', () => {
    render(
      <ChatMessage
        role="user"
        text="Achetez !"
        compliance={{ kind: 'refusal', category: 'prescriptive' }}
      />,
    );
    // User bubbles never carry the badge — refusals are emitted on the
    // assistant turn replying to the user, not on the user input itself.
    expect(screen.queryByTestId('compliance-badge')).toBeNull();
  });

  it('renders a refusal badge on an assistant bubble', () => {
    render(
      <ChatMessage
        role="assistant"
        text="Refus pédagogique."
        compliance={{ kind: 'refusal', category: 'prescriptive' }}
      />,
    );
    const badge = screen.getByTestId('compliance-badge');
    expect(badge).toBeInTheDocument();
    expect(badge.getAttribute('data-compliance-kind')).toBe('refusal');
    expect(badge.textContent?.toLowerCase()).toContain('refus');
  });

  it('renders a forbidden_token badge with the "filtré" wording', () => {
    render(
      <ChatMessage
        role="assistant"
        text="Sortie nettoyée."
        compliance={{ kind: 'forbidden_token', token: 'achetez' }}
      />,
    );
    const badge = screen.getByTestId('compliance-badge');
    expect(badge.getAttribute('data-compliance-kind')).toBe('forbidden_token');
    expect(badge.textContent?.toLowerCase()).toContain('filtrée');
  });
});
