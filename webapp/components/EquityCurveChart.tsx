'use client';

import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, Tooltip } from 'recharts';

type Point = { ts: number; equity_R: number };

export function EquityCurveChart({ payload }: { payload?: { equity_curve?: Point[] } }) {
  const data = (payload?.equity_curve ?? []).map((p) => ({
    t: new Date(p.ts * 1000).toLocaleDateString(),
    R: p.equity_R,
  }));
  if (data.length === 0) {
    return <p className="text-sm text-slate-500">No data yet.</p>;
  }
  return (
    <div className="card" style={{ height: 320 }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data}>
          <XAxis dataKey="t" fontSize={10} />
          <YAxis fontSize={10} />
          <Tooltip />
          <Line type="monotone" dataKey="R" stroke="#C9A14A" strokeWidth={2} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
