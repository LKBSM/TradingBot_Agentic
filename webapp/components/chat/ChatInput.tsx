'use client';

import { useTranslations } from 'next-intl';
import * as React from 'react';
import { ChatComposer } from './ChatComposer';
import { useDictationCopy } from '@/lib/scanner-chat/use-dictation-copy';
import { useChat } from './ChatProvider';

interface ChatInputProps {
  className?: string;
}

/**
 * Free-text input for the Sentinel chatbot. Thin wrapper that wires the shared
 * `ChatComposer` (CLN-1 §2) to the network chat: submits to the backend via
 * `useChat().askFreeForm()` and gates on an active signal + online state. The
 * composer owns the draft, dictation and DOM (same data-testids as before).
 */
export function ChatInput({ className }: ChatInputProps) {
  const t = useTranslations('chat');
  const { askFreeForm, isLoading, activeSignal, apiAvailable } = useChat();
  const dictationCopy = useDictationCopy();

  const offline = apiAvailable === false;

  function handleSubmit(question: string) {
    void askFreeForm(question).catch((err) => {
      // askFreeForm already pushes an error turn — nothing else to do here.
      console.error('chat submit failed', err);
    });
  }

  return (
    <ChatComposer
      className={className}
      onSubmit={handleSubmit}
      placeholder={offline ? t('inputPlaceholderOffline') : t('inputPlaceholder')}
      ariaLabel={t('inputAria')}
      hint={t('inputHint')}
      sendAria={t('sendAria')}
      sendLoadingAria={t('sendLoadingAria')}
      privacyNote={dictationCopy.privacy}
      isLoading={isLoading}
      disabled={offline}
      submitEnabled={activeSignal !== null}
    />
  );
}
