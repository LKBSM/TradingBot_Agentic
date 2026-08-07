import { afterEach, describe, expect, it, vi } from 'vitest';
import { act, renderHook } from '@testing-library/react';
import { useSpeechDictation } from '../use-speech-dictation';

/**
 * SC-2 — the dictation hook. The load-bearing guarantee for degradation: when the
 * browser cannot dictate, ``supported`` is false so the caller hides the mic
 * entirely (no dead button). When it can, an error is surfaced honestly and the
 * session never hangs.
 */

afterEach(() => {
  delete (window as unknown as Record<string, unknown>).SpeechRecognition;
  delete (window as unknown as Record<string, unknown>).webkitSpeechRecognition;
  vi.restoreAllMocks();
});

describe('useSpeechDictation — unsupported browser', () => {
  it('reports supported === false when no SpeechRecognition exists (mic hidden)', () => {
    const { result } = renderHook(() => useSpeechDictation({ lang: 'fr-FR', onFinal: () => {} }));
    expect(result.current.supported).toBe(false);
  });

  it('start() is a no-op when unsupported', () => {
    const { result } = renderHook(() => useSpeechDictation({ lang: 'fr-FR', onFinal: () => {} }));
    act(() => result.current.start());
    expect(result.current.listening).toBe(false);
  });
});

describe('useSpeechDictation — permission denied', () => {
  it('surfaces "not-allowed" and stops, keeping the field usable', () => {
    class FakeRecognition {
      lang = '';
      continuous = false;
      interimResults = false;
      maxAlternatives = 1;
      onresult: ((e: unknown) => void) | null = null;
      onerror: ((e: { error: string }) => void) | null = null;
      onend: (() => void) | null = null;
      onstart: (() => void) | null = null;
      start() {
        this.onstart?.();
        // Browser denies mic access.
        this.onerror?.({ error: 'not-allowed' });
        this.onend?.();
      }
      stop() {
        this.onend?.();
      }
      abort() {}
    }
    (window as unknown as Record<string, unknown>).SpeechRecognition = FakeRecognition;

    const { result } = renderHook(() => useSpeechDictation({ lang: 'fr-FR', onFinal: () => {} }));
    expect(result.current.supported).toBe(true);
    act(() => result.current.start());
    expect(result.current.error).toBe('not-allowed');
    expect(result.current.listening).toBe(false);
  });
});

describe('useSpeechDictation — transcription', () => {
  it('forwards a finalised transcript to onFinal', () => {
    let recognition: FakeRec | null = null;
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
        recognition = this;
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

    const onFinal = vi.fn();
    const { result } = renderHook(() => useSpeechDictation({ lang: 'fr-FR', onFinal }));
    act(() => result.current.start());
    act(() => {
      recognition!.onresult?.({
        resultIndex: 0,
        results: { length: 1, 0: { isFinal: true, length: 1, 0: { transcript: 'order block jamais testé' } } },
      });
    });
    expect(onFinal).toHaveBeenCalledWith('order block jamais testé');
  });
});
