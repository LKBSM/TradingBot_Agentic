import { afterEach, describe, expect, it, vi } from 'vitest';
import { act, renderHook } from '@testing-library/react';
import { appendTranscript, useVoiceInput } from '../use-voice-input';

/**
 * Shared voice-input adapter (mission "Dictée vocale"). The load-bearing honesty
 * guarantee: dictation only ever APPENDS to the field the user controls, so what
 * M.I.A receives is exactly the visible text — no hidden transform. Degradation
 * (unsupported / denied) stays honest and the keyboard remains usable.
 */

afterEach(() => {
  delete (window as unknown as Record<string, unknown>).SpeechRecognition;
  delete (window as unknown as Record<string, unknown>).webkitSpeechRecognition;
  vi.restoreAllMocks();
});

/** Minimal fake capturing the live instance so a test can drive its callbacks. */
function installFakeRecognition() {
  let instance: FakeRec | null = null;
  class FakeRec {
    lang = '';
    continuous = false;
    interimResults = false;
    maxAlternatives = 1;
    onresult: ((e: unknown) => void) | null = null;
    onerror: ((e: { error: string }) => void) | null = null;
    onend: (() => void) | null = null;
    onstart: (() => void) | null = null;
    constructor() {
      instance = this;
    }
    start() {
      this.onstart?.();
    }
    stop() {
      this.onend?.();
    }
    abort() {}
  }
  (window as unknown as Record<string, unknown>).webkitSpeechRecognition = FakeRec;
  return () => instance;
}

function finalResult(rec: { onresult: ((e: unknown) => void) | null }, transcript: string) {
  rec.onresult?.({
    resultIndex: 0,
    results: { length: 1, 0: { isFinal: true, length: 1, 0: { transcript } } },
  });
}

describe('appendTranscript — exact, single-space composition', () => {
  it('returns the chunk verbatim on an empty field', () => {
    expect(appendTranscript('', 'order block jamais testé')).toBe('order block jamais testé');
  });
  it('joins with exactly one space and never doubles it', () => {
    expect(appendTranscript('mon plan', 'sur le CHOCH')).toBe('mon plan sur le CHOCH');
    expect(appendTranscript('mon plan  ', 'sur le CHOCH')).toBe('mon plan sur le CHOCH');
  });
});

describe('useVoiceInput — transcript is exactly what fills the field', () => {
  it('appends the finalised transcript with no hidden transform', () => {
    const getRec = installFakeRecognition();
    let value = 'ma stratégie';
    const onValueChange = vi.fn((next: string) => {
      value = next;
    });
    const { result, rerender } = renderHook((props: { value: string }) =>
      useVoiceInput({ locale: 'fr', value: props.value, onValueChange }), { initialProps: { value } });

    expect(result.current.supported).toBe(true);
    act(() => result.current.toggle());
    act(() => finalResult(getRec()!, 'sur le retest'));

    // The setter received exactly base + space + chunk — nothing more.
    expect(onValueChange).toHaveBeenCalledWith('ma stratégie sur le retest');
    rerender({ value });
    expect(value).toBe('ma stratégie sur le retest');
  });

  it('honours maxLength so the field never exceeds its cap', () => {
    const getRec = installFakeRecognition();
    const onValueChange = vi.fn();
    const { result } = renderHook(() =>
      useVoiceInput({ locale: 'fr', value: 'abcdefgh', onValueChange, maxLength: 10 }));
    act(() => result.current.toggle());
    act(() => finalResult(getRec()!, 'ijklmnop'));
    expect(onValueChange).toHaveBeenCalledWith('abcdefgh i'); // 'abcdefgh ijklmnop' sliced to 10
  });

  it('maps a denied permission to `denied` while leaving the field usable', () => {
    let instance: DenyRec | null = null;
    class DenyRec {
      lang = '';
      continuous = false;
      interimResults = false;
      maxAlternatives = 1;
      onresult: ((e: unknown) => void) | null = null;
      onerror: ((e: { error: string }) => void) | null = null;
      onend: (() => void) | null = null;
      onstart: (() => void) | null = null;
      constructor() {
        instance = this;
      }
      start() {
        this.onstart?.();
        this.onerror?.({ error: 'not-allowed' });
        this.onend?.();
      }
      stop() {
        this.onend?.();
      }
      abort() {}
    }
    (window as unknown as Record<string, unknown>).SpeechRecognition = DenyRec;
    void instance;

    const onValueChange = vi.fn();
    const { result } = renderHook(() => useVoiceInput({ locale: 'fr', value: '', onValueChange }));
    act(() => result.current.toggle());
    expect(result.current.denied).toBe(true);
    expect(result.current.error).toBe('not-allowed');
    expect(result.current.listening).toBe(false);
    // Dictation failing never wrote to the field.
    expect(onValueChange).not.toHaveBeenCalled();
  });

  it('reports supported === false when the browser lacks the API (mic hidden)', () => {
    const { result } = renderHook(() => useVoiceInput({ locale: 'en', value: '', onValueChange: () => {} }));
    expect(result.current.supported).toBe(false);
    expect(result.current.denied).toBe(false);
  });
});
