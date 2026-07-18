import { render, screen } from '@/components/test-utils';
import { describe, expect, it } from 'vitest';
import { ChatMessage } from '../ChatMessage';

describe('ChatMessage', () => {
  it('renders the assistant text', () => {
    render(<ChatMessage role="assistant" text="Le marché consolide." />);
    expect(screen.getByText('Le marché consolide.')).toBeInTheDocument();
  });

  it('shows the discreet redirect badge when an assistant turn is blocked', () => {
    render(
      <ChatMessage
        role="assistant"
        text="Je décris les conditions du marché."
        blockedReason="trade_request"
      />,
    );
    expect(screen.getByText('Question recadrée')).toBeInTheDocument();
  });

  it('does NOT show the badge on a normal (unblocked) assistant turn', () => {
    render(<ChatMessage role="assistant" text="Réponse normale." />);
    expect(screen.queryByText('Question recadrée')).not.toBeInTheDocument();
  });

  it('never shows the badge on a user turn even if blockedReason is set', () => {
    render(
      <ChatMessage role="user" text="Dois-je acheter ?" blockedReason="trade_request" />,
    );
    expect(screen.queryByText('Question recadrée')).not.toBeInTheDocument();
  });
});
