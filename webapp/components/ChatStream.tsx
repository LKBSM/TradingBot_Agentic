type Msg = { role: 'user' | 'assistant'; content: string; sources?: any[] };

export function ChatStream({ message, sourcesLabel }: { message: Msg; sourcesLabel: string }) {
  const isUser = message.role === 'user';
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`max-w-[80%] rounded-2xl px-4 py-3 ${
          isUser ? 'bg-sentinel-ink text-white' : 'bg-slate-100'
        }`}
      >
        <p className="whitespace-pre-wrap text-sm">{message.content}</p>
        {message.sources && message.sources.length > 0 && (
          <details className="mt-2 text-xs">
            <summary className="cursor-pointer">{sourcesLabel}</summary>
            <ul className="mt-1 space-y-1">
              {message.sources.map((s: any, i: number) => (
                <li key={i} className="opacity-80">
                  • {s.source_id ?? s.id ?? JSON.stringify(s).slice(0, 60)}
                </li>
              ))}
            </ul>
          </details>
        )}
      </div>
    </div>
  );
}
