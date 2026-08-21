'use client';

import { useTranslations } from 'next-intl';
import type { DictationError } from './use-speech-dictation';

/**
 * Single place that resolves the dictation copy, so every surface (chat, zones,
 * publications, scanner) shows the SAME honest labels and the SAME privacy note
 * — and no component outside this file needs to know which i18n namespace the
 * strings live under.
 *
 * The privacy note deliberately says transcription "may transit through the
 * browser's servers": the Web Speech API is not guaranteed to be on-device, so
 * the statement stays true on every surface that reuses this hook.
 */
export interface DictationCopy {
  startLabel: string;
  stopLabel: string;
  listeningLabel: string;
  privacy: string;
  errorText(code: DictationError): string;
}

export function useDictationCopy(): DictationCopy {
  const t = useTranslations('scannerChat');
  return {
    startLabel: t('dictation.start'),
    stopLabel: t('dictation.stop'),
    listeningLabel: t('dictation.listening'),
    privacy: t('dictation.privacy'),
    errorText: (code: DictationError) => {
      const key = `dictation.errors.${code}` as const;
      return t.has(key) ? t(key) : t('dictation.errors.unknown');
    },
  };
}
