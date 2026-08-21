'use client';

import * as React from 'react';
import { useSpeechDictation, type DictationError } from './use-speech-dictation';

/**
 * Shared voice-input adapter that turns the raw browser dictation hook
 * (`useSpeechDictation`) into a drop-in helper for ANY controlled text field —
 * the /app M.I.A Agent, the /zones panel, the /actualites publication chat and
 * the "Décris ta stratégie" console all consume this exact hook.
 *
 * It only ever APPENDS transcribed text to the field the user already controls:
 * what M.I.A receives is precisely the visible value, with no hidden transform.
 * The transcription itself happens in the browser (Web Speech API) — see the
 * privacy note surfaced next to each field.
 */
export interface VoiceInput {
  /** Feature-detected client-side. False ⇒ hide the mic (no dead button). */
  supported: boolean;
  listening: boolean;
  /** True while the last error is a denied microphone permission. */
  denied: boolean;
  /** Live partial transcript, shown while speaking (never hidden). */
  interim: string;
  error: DictationError | null;
  /** Start when idle, stop when listening — bind straight to the mic button. */
  toggle(): void;
  clearError(): void;
}

export interface UseVoiceInputOptions {
  /** Active app locale (e.g. "fr", "en-US") → mapped to a BCP-47 speech tag. */
  locale: string;
  /** Current field value (read at dictation time to append without clobbering). */
  value: string;
  /** Setter for the controlled field — called with the appended value. */
  onValueChange(next: string): void;
  /** Optional cap mirrored from the field's maxLength. */
  maxLength?: number;
}

export function useVoiceInput({
  locale,
  value,
  onValueChange,
  maxLength,
}: UseVoiceInputOptions): VoiceInput {
  const speechLang = locale.toLowerCase().startsWith('en') ? 'en-US' : 'fr-FR';

  // Read the freshest value / setter / cap at transcription time, so the
  // callback never appends onto a stale snapshot (the field stays the source
  // of truth).
  const valueRef = React.useRef(value);
  valueRef.current = value;
  const onChangeRef = React.useRef(onValueChange);
  onChangeRef.current = onValueChange;
  const maxRef = React.useRef(maxLength);
  maxRef.current = maxLength;

  const onFinal = React.useCallback((chunk: string) => {
    let next = appendTranscript(valueRef.current, chunk);
    if (maxRef.current != null) next = next.slice(0, maxRef.current);
    onChangeRef.current(next);
  }, []);

  const dictation = useSpeechDictation({ lang: speechLang, onFinal });

  const { listening, stop, start } = dictation;
  const toggle = React.useCallback(() => {
    if (listening) stop();
    else start();
  }, [listening, stop, start]);

  return {
    supported: dictation.supported,
    listening: dictation.listening,
    denied: dictation.error === 'not-allowed',
    interim: dictation.interim,
    error: dictation.error,
    toggle,
    clearError: dictation.clearError,
  };
}

/** Append a dictated chunk with a single separating space, trimming doubles. */
export function appendTranscript(current: string, chunk: string): string {
  const base = current.replace(/\s+$/, '');
  if (!base) return chunk;
  return `${base} ${chunk}`.replace(/\s{2,}/g, ' ');
}
