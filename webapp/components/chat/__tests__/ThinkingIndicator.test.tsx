import { render, screen } from '@/components/test-utils';
import { describe, expect, it, vi } from 'vitest';
import { formatInstrument, formatTimeframe } from '@/lib/market-reading/formatters';

/**
 * MIA-1 — the waiting indicator narrates the REAL step honestly:
 *  - generic "thinking" → animated dots, screen-reader-only label (no invented
 *    step to fill the void);
 *  - a market read the backend actually started → a VISIBLE caption built from
 *    the tool's own arguments, never model prose.
 */

const state = vi.hoisted(() => ({ activity: null as unknown }));
vi.mock('../ChatProvider', () => ({ useChat: () => ({ activity: state.activity }) }));

import { ThinkingIndicator } from '../ThinkingIndicator';

describe('ThinkingIndicator — honest activity (MIA-1)', () => {
  it('generic thinking shows dots only (caption is screen-reader-only)', () => {
    state.activity = { kind: 'thinking' };
    render(<ThinkingIndicator />);
    expect(screen.getByRole('status')).toBeInTheDocument();
    expect(screen.queryByTestId('chat-activity')).toBeNull();
  });

  it('market read in progress shows an honest caption from the tool args', () => {
    state.activity = {
      kind: 'tool',
      tool: 'get_market_reading',
      instrument: 'XAUUSD',
      timeframe: 'M15',
    };
    render(<ThinkingIndicator />);
    const caption = screen.getByTestId('chat-activity');
    const expected = `Lecture de ${formatInstrument('XAUUSD')} ${formatTimeframe('M15')}…`;
    expect(caption.textContent).toBe(expected);
  });

  it('OB diagnostic shows its own honest caption', () => {
    state.activity = {
      kind: 'tool',
      tool: 'get_ob_diagnostic',
      instrument: 'XAUUSD',
      timeframe: 'M15',
    };
    render(<ThinkingIndicator />);
    expect(screen.getByTestId('chat-activity').textContent).toBe('Diagnostic Order Block…');
  });
});
