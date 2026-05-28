import { afterEach, describe, expect, it, vi } from 'vitest';
import { lengthBucket, trackEvent } from './events';

describe('DG-161 trackEvent dispatcher', () => {
  afterEach(() => {
    delete (window as unknown as { plausible?: unknown }).plausible;
  });

  it('does nothing when plausible is undefined', () => {
    // No window.plausible — must not throw
    expect(() =>
      trackEvent('hero_view', { is_landing: true, locale: 'fr' }),
    ).not.toThrow();
  });

  it('calls plausible with the event name and props', () => {
    const spy = vi.fn();
    (window as unknown as { plausible: typeof spy }).plausible = spy;
    trackEvent('chatbot_open', { source: 'hero' });
    expect(spy).toHaveBeenCalledWith('chatbot_open', {
      props: { source: 'hero' },
    });
  });

  it('swallows exceptions thrown by plausible', () => {
    (window as unknown as { plausible: () => void }).plausible = () => {
      throw new Error('Plausible exploded');
    };
    expect(() =>
      trackEvent('track_record_view', { is_first_view: true }),
    ).not.toThrow();
  });
});

describe('DG-161 lengthBucket', () => {
  it('classifies tiny / short / medium / long', () => {
    expect(lengthBucket('hi')).toBe('tiny');
    expect(lengthBucket('a'.repeat(50))).toBe('short');
    expect(lengthBucket('a'.repeat(150))).toBe('medium');
    expect(lengthBucket('a'.repeat(500))).toBe('long');
  });

  it('trims whitespace before bucketing', () => {
    expect(lengthBucket('   hi   ')).toBe('tiny');
  });
});
