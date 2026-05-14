'use client';

type RegimeEntry = { ts: number; label: 'low_vol_trending' | 'low_vol_ranging' | 'high_vol_stress'; confidence: number };

const COLOR: Record<RegimeEntry['label'], string> = {
  low_vol_trending: '#16A34A',
  low_vol_ranging: '#94A3B8',
  high_vol_stress: '#DC2626',
};

export function RegimeTimeline({ payload }: { payload?: { entries?: RegimeEntry[] } }) {
  const entries = payload?.entries ?? [];
  if (entries.length === 0) {
    return <p className="text-sm text-slate-500">No regime data yet.</p>;
  }
  return (
    <div className="card">
      <div className="flex h-8 w-full overflow-hidden rounded-md">
        {entries.map((e, i) => (
          <div
            key={i}
            title={`${e.label} • conf=${e.confidence.toFixed(2)}`}
            style={{ flex: 1, backgroundColor: COLOR[e.label] }}
          />
        ))}
      </div>
      <div className="mt-2 flex gap-4 text-xs">
        {Object.entries(COLOR).map(([k, c]) => (
          <span key={k} className="flex items-center gap-1">
            <span className="inline-block h-2 w-2 rounded-sm" style={{ backgroundColor: c }} />
            {k}
          </span>
        ))}
      </div>
    </div>
  );
}
