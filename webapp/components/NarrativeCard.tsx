type Entry = {
  seq: number;
  insight_id: string;
  inserted_at_utc: string;
  entry_hash: string;
};

export function NarrativeCard({ entry }: { entry: Entry }) {
  return (
    <article className="card">
      <header className="mb-2 flex justify-between text-xs text-slate-500">
        <span className="font-mono">#{entry.seq}</span>
        <time dateTime={entry.inserted_at_utc}>{entry.inserted_at_utc}</time>
      </header>
      <h3 className="font-semibold">Insight {entry.insight_id}</h3>
      <p className="mt-2 text-sm text-slate-500 font-mono">
        hash: {entry.entry_hash.slice(0, 12)}…
      </p>
    </article>
  );
}
