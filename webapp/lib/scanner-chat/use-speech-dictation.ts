'use client';

import * as React from 'react';

/**
 * SC-2 — voice dictation via the browser's native Web Speech API.
 *
 * The transcription happens IN THE BROWSER; M.I.A only ever receives TEXT, as if
 * the user had typed it. No audio ever transits our backend, and none is stored.
 *
 * Degradation is deliberate and honest:
 *   · unsupported browser (e.g. Firefox) → ``supported === false`` so the caller
 *     hides the mic button entirely (no dead button, no error on click);
 *   · permission denied / no speech / network → a typed error the caller renders
 *     while the text field stays fully usable;
 *   · a listening timeout stops the session instead of listening forever.
 *
 * The recogniser is imperfect on « Order Block », « CHOCH », « Fair Value Gap »:
 * that is acceptable ONLY because the user sees the transcript before translating
 * and the condition cards before scanning — two safety nets. We never hide the
 * transcript and never auto-translate.
 */

export type DictationError =
  | 'not-allowed'
  | 'no-speech'
  | 'audio-capture'
  | 'network'
  | 'aborted'
  | 'timeout'
  | 'unknown';

export interface SpeechDictation {
  /** Feature-detected on mount (client-only). False ⇒ hide the mic entirely. */
  supported: boolean;
  listening: boolean;
  /** Live partial transcript shown while speaking (never hidden). */
  interim: string;
  error: DictationError | null;
  start(): void;
  stop(): void;
  clearError(): void;
}

// Minimal typings — the Web Speech API is not in the standard TS DOM lib.
interface SpeechRecognitionAlternativeLike {
  transcript: string;
}
interface SpeechRecognitionResultLike {
  readonly isFinal: boolean;
  readonly length: number;
  [index: number]: SpeechRecognitionAlternativeLike;
}
interface SpeechRecognitionEventLike {
  readonly resultIndex: number;
  readonly results: {
    readonly length: number;
    [index: number]: SpeechRecognitionResultLike;
  };
}
interface SpeechRecognitionErrorEventLike {
  readonly error: string;
}
interface SpeechRecognitionLike {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  maxAlternatives: number;
  start(): void;
  stop(): void;
  abort(): void;
  onresult: ((e: SpeechRecognitionEventLike) => void) | null;
  onerror: ((e: SpeechRecognitionErrorEventLike) => void) | null;
  onend: (() => void) | null;
  onstart: (() => void) | null;
}
type SpeechRecognitionCtor = new () => SpeechRecognitionLike;

function getRecognitionCtor(): SpeechRecognitionCtor | null {
  if (typeof window === 'undefined') return null;
  const w = window as unknown as {
    SpeechRecognition?: SpeechRecognitionCtor;
    webkitSpeechRecognition?: SpeechRecognitionCtor;
  };
  return w.SpeechRecognition ?? w.webkitSpeechRecognition ?? null;
}

const LISTEN_TIMEOUT_MS = 12_000;

export interface UseSpeechDictationOptions {
  /** BCP-47 language tag, derived from the active locale (fr-FR / en-US). */
  lang: string;
  /** Called with each finalised chunk (append to the text field). */
  onFinal(text: string): void;
}

export function useSpeechDictation({ lang, onFinal }: UseSpeechDictationOptions): SpeechDictation {
  const [supported, setSupported] = React.useState(false);
  const [listening, setListening] = React.useState(false);
  const [interim, setInterim] = React.useState('');
  const [error, setError] = React.useState<DictationError | null>(null);

  const recognitionRef = React.useRef<SpeechRecognitionLike | null>(null);
  const timeoutRef = React.useRef<ReturnType<typeof setTimeout> | null>(null);
  const onFinalRef = React.useRef(onFinal);
  onFinalRef.current = onFinal;

  // Feature-detect on mount only (avoids SSR / hydration mismatch: the button is
  // absent on the server render and appears once we know the browser supports it).
  React.useEffect(() => {
    setSupported(getRecognitionCtor() !== null);
  }, []);

  const clearTimer = React.useCallback(() => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }
  }, []);

  const stop = React.useCallback(() => {
    clearTimer();
    const rec = recognitionRef.current;
    if (rec) {
      try {
        rec.stop();
      } catch {
        /* already stopped */
      }
    }
  }, [clearTimer]);

  const start = React.useCallback(() => {
    const Ctor = getRecognitionCtor();
    if (!Ctor) return;
    // Never stack two sessions.
    if (recognitionRef.current) {
      try {
        recognitionRef.current.abort();
      } catch {
        /* noop */
      }
      recognitionRef.current = null;
    }
    setError(null);
    setInterim('');

    let rec: SpeechRecognitionLike;
    try {
      rec = new Ctor();
    } catch {
      setError('unknown');
      return;
    }
    rec.lang = lang;
    rec.continuous = false;
    rec.interimResults = true;
    rec.maxAlternatives = 1;

    let gotSpeech = false;

    rec.onstart = () => {
      setListening(true);
      clearTimer();
      timeoutRef.current = setTimeout(() => {
        // No speech within the window → stop and report it (never listen forever).
        if (!gotSpeech) setError('timeout');
        try {
          rec.stop();
        } catch {
          /* noop */
        }
      }, LISTEN_TIMEOUT_MS);
    };

    rec.onresult = (event) => {
      gotSpeech = true;
      let interimText = '';
      let finalText = '';
      for (let i = event.resultIndex; i < event.results.length; i += 1) {
        const result = event.results[i];
        const alt = result?.[0];
        if (!result || !alt) continue;
        if (result.isFinal) finalText += alt.transcript;
        else interimText += alt.transcript;
      }
      if (finalText) {
        setInterim('');
        onFinalRef.current(finalText.trim());
      } else {
        setInterim(interimText);
      }
    };

    rec.onerror = (event) => {
      const code = mapError(event.error);
      // A user-initiated stop surfaces as "aborted" — not an error worth showing.
      if (code !== 'aborted') setError(code);
    };

    rec.onend = () => {
      clearTimer();
      setListening(false);
      setInterim('');
      recognitionRef.current = null;
    };

    recognitionRef.current = rec;
    try {
      rec.start();
    } catch {
      // Calling start() twice throws — treat as a benign no-op.
      setListening(false);
    }
  }, [lang, clearTimer]);

  // Cleanup on unmount.
  React.useEffect(() => {
    return () => {
      clearTimer();
      const rec = recognitionRef.current;
      if (rec) {
        try {
          rec.abort();
        } catch {
          /* noop */
        }
      }
    };
  }, [clearTimer]);

  const clearError = React.useCallback(() => setError(null), []);

  return { supported, listening, interim, error, start, stop, clearError };
}

function mapError(raw: string): DictationError {
  switch (raw) {
    case 'not-allowed':
    case 'service-not-allowed':
      return 'not-allowed';
    case 'no-speech':
      return 'no-speech';
    case 'audio-capture':
      return 'audio-capture';
    case 'network':
      return 'network';
    case 'aborted':
      return 'aborted';
    default:
      return 'unknown';
  }
}
