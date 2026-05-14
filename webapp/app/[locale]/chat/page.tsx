'use client';

import { useTranslations } from 'next-intl';
import { useState } from 'react';
import { ChatStream } from '@/components/ChatStream';

export default function ChatPage() {
  const t = useTranslations('chat');
  const td = useTranslations('disclaimer');
  const [messages, setMessages] = useState<
    { role: 'user' | 'assistant'; content: string; sources?: any[] }[]
  >([]);
  const [draft, setDraft] = useState('');
  const [busy, setBusy] = useState(false);

  async function send() {
    if (!draft.trim() || busy) return;
    const text = draft.trim();
    setDraft('');
    setBusy(true);
    setMessages((m) => [...m, { role: 'user', content: text }]);
    try {
      const r = await fetch('/api/v1/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: text }),
      });
      const data = await r.json();
      setMessages((m) => [
        ...m,
        {
          role: 'assistant',
          content: data.answer ?? '—',
          sources: data.sources_cited ?? [],
        },
      ]);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="container-prose py-12">
      <h1 className="text-2xl font-bold">{t('title')}</h1>
      <p className="mt-2 text-xs italic text-slate-500">{td('short')}</p>

      <div className="mt-6 space-y-4">
        {messages.map((m, i) => (
          <ChatStream key={i} message={m} sourcesLabel={t('sources_label')} />
        ))}
      </div>

      <div className="mt-8 flex gap-2">
        <input
          className="flex-1 rounded-xl border border-slate-300 px-4 py-3"
          placeholder={t('placeholder')}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && send()}
        />
        <button
          className="rounded-xl bg-sentinel-ink px-4 py-3 text-white disabled:opacity-50"
          onClick={send}
          disabled={busy}
        >
          {t('send')}
        </button>
      </div>
    </div>
  );
}
