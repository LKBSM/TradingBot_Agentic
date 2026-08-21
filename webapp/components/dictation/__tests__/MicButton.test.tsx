import { describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import { MicButton } from '../MicButton';

/**
 * The shared mic affordance. Its `data-state` must reflect the REAL dictation
 * state (idle / listening / denied) so a refused permission is self-evident on
 * every surface, and a click always toggles (start when idle, retry when denied,
 * stop when listening).
 */

const labels = { startLabel: 'Dicter à la voix', stopLabel: 'Arrêter la dictée' };

describe('MicButton', () => {
  it('is idle by default and exposes the start label', () => {
    render(<MicButton listening={false} denied={false} onToggle={() => {}} {...labels} />);
    const btn = screen.getByTestId('mic-button');
    expect(btn).toHaveAttribute('data-state', 'idle');
    expect(btn).toHaveAttribute('aria-pressed', 'false');
    expect(btn).toHaveAttribute('aria-label', labels.startLabel);
  });

  it('reflects the listening state with the stop label', () => {
    render(<MicButton listening denied={false} onToggle={() => {}} {...labels} />);
    const btn = screen.getByTestId('mic-button');
    expect(btn).toHaveAttribute('data-state', 'listening');
    expect(btn).toHaveAttribute('aria-pressed', 'true');
    expect(btn).toHaveAttribute('aria-label', labels.stopLabel);
  });

  it('reflects the denied state (alert) while still being clickable to retry', () => {
    render(<MicButton listening={false} denied onToggle={() => {}} {...labels} />);
    const btn = screen.getByTestId('mic-button');
    expect(btn).toHaveAttribute('data-state', 'denied');
    // Denied is a visual alert only — the button offers to retry (start label).
    expect(btn).toHaveAttribute('aria-label', labels.startLabel);
    expect(btn).not.toBeDisabled();
  });

  it('calls onToggle on click', () => {
    const onToggle = vi.fn();
    render(<MicButton listening={false} denied={false} onToggle={onToggle} {...labels} />);
    fireEvent.click(screen.getByTestId('mic-button'));
    expect(onToggle).toHaveBeenCalledTimes(1);
  });
});
