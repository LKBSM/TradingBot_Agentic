/**
 * DG-142 — public /track-record page.
 *
 * Server-rendered (Next 15 server component) so the marketing layer can
 * embed the headline PF + bootstrap CI + equity curve without any
 * client-side JS. The endpoint at /api/v1/track-record is public — no
 * auth required.
 *
 * Honesty rules (UE 2024/2811):
 *   - The page always renders ``edge_claim``: when false, the headline
 *     adds an "OOS validation in progress" caveat right next to the PF
 *     so a casual reader cannot mistake the figures for a promise.
 *   - All numbers carry their bootstrap CI bracket. The disclaimer from
 *     the backend is rendered verbatim.
 */
import type { Metadata } from 'next';

export const dynamic = 'force-dynamic';

interface TrackRecordPayload {
  n_trades: number;
  profit_factor: number | null;
  profit_factor_ci95: [number | null, number | null];
  hit_rate: number | null;
  equity_curve_r_multiples: number[];
  backtest_window: string;
  edge_claim: boolean;
  bootstrap: { n_iterations: number; alpha: number; seed: number };
  disclaimer: string;
}

async function fetchTrackRecord(baseUrl: string): Promise<TrackRecordPayload | null> {
  try {
    const res = await fetch(`${baseUrl}/api/v1/track-record`, {
      cache: 'no-store',
    });
    if (!res.ok) return null;
    return (await res.json()) as TrackRecordPayload;
  } catch {
    return null;
  }
}

export const metadata: Metadata = {
  title: 'Track record — M.I.A. Markets',
  description:
    'Synthèse du backtest paper-trading : profit factor, IC 95 % bootstrap, courbe d\'équité. Performance passée non garantie de performance future.',
};

export default async function TrackRecordPage() {
  const apiBase = process.env.NEXT_PUBLIC_API_BASE ?? 'http://localhost:8000';
  const data = await fetchTrackRecord(apiBase);

  if (!data) {
    return (
      <main className="mx-auto max-w-3xl px-4 py-16">
        <h1 className="text-3xl font-bold">Track record</h1>
        <p className="mt-4 text-muted-foreground">
          Les données ne sont pas disponibles pour le moment. La page se recharge
          automatiquement quand le snapshot quotidien est publié.
        </p>
      </main>
    );
  }

  const pf = data.profit_factor;
  const [lo, hi] = data.profit_factor_ci95;
  const hitPct = data.hit_rate !== null ? (data.hit_rate * 100).toFixed(1) : 'n/a';

  return (
    <main className="mx-auto max-w-3xl px-4 py-16 space-y-10">
      <header className="space-y-2">
        <h1 className="text-3xl font-bold">Track record paper-trading</h1>
        <p className="text-sm text-muted-foreground">
          {data.backtest_window} · {data.n_trades} trades
        </p>
        {!data.edge_claim ? (
          <p
            className="rounded bg-amber-100 px-3 py-2 text-sm text-amber-900"
            role="note"
          >
            Validation OOS indépendante en cours. Les chiffres ci-dessous ne
            constituent <strong>pas</strong> une preuve d&apos;edge tant que les
            gates empiriques (PF&nbsp;&gt;&nbsp;1.20, DSR&nbsp;&gt;&nbsp;1.0,
            PBO&nbsp;&lt;&nbsp;0.5) ne sont pas franchis.
          </p>
        ) : null}
      </header>

      <section
        aria-labelledby="metrics-heading"
        className="grid grid-cols-1 sm:grid-cols-3 gap-4"
      >
        <h2 id="metrics-heading" className="sr-only">
          Métriques principales
        </h2>
        <Metric
          label="Profit factor"
          value={pf !== null ? pf.toFixed(2) : 'n/a'}
          subtext={
            lo !== null && hi !== null
              ? `IC 95 % bootstrap [${lo.toFixed(2)} – ${hi.toFixed(2)}]`
              : 'IC indisponible'
          }
        />
        <Metric label="Hit rate observé" value={`${hitPct} %`} />
        <Metric label="Trades évalués" value={String(data.n_trades)} />
      </section>

      <section aria-labelledby="equity-heading" className="space-y-3">
        <h2 id="equity-heading" className="text-xl font-semibold">
          Courbe d&apos;équité (R multiples cumulés)
        </h2>
        <EquitySparkline points={data.equity_curve_r_multiples} />
        <p className="text-xs text-muted-foreground">
          Bootstrap : {data.bootstrap.n_iterations} itérations, α ={' '}
          {data.bootstrap.alpha}, seed = {data.bootstrap.seed}.
        </p>
      </section>

      <footer className="text-xs text-muted-foreground border-t pt-6">
        {data.disclaimer}
      </footer>
    </main>
  );
}

function Metric({
  label,
  value,
  subtext,
}: {
  label: string;
  value: string;
  subtext?: string;
}) {
  return (
    <div className="rounded-lg border p-4">
      <div className="text-xs uppercase text-muted-foreground">{label}</div>
      <div className="mt-2 text-3xl font-semibold">{value}</div>
      {subtext ? (
        <div className="mt-1 text-xs text-muted-foreground">{subtext}</div>
      ) : null}
    </div>
  );
}

/**
 * Tiny inline SVG sparkline — no client JS, no chart library.
 * The page renders the cumulative R-multiples as a single polyline so the
 * marketing page stays lightweight and SSR-only.
 */
function EquitySparkline({ points }: { points: number[] }) {
  if (points.length === 0) {
    return (
      <p className="text-sm text-muted-foreground">
        Aucune donnée d&apos;équité à afficher.
      </p>
    );
  }
  const width = 720;
  const height = 200;
  const padding = 8;
  const minY = Math.min(...points, 0);
  const maxY = Math.max(...points, 1);
  const rangeY = maxY - minY || 1;
  const stepX = (width - 2 * padding) / Math.max(points.length - 1, 1);
  const path = points
    .map((y, i) => {
      const px = padding + i * stepX;
      const py =
        height - padding - ((y - minY) / rangeY) * (height - 2 * padding);
      return `${i === 0 ? 'M' : 'L'}${px.toFixed(2)},${py.toFixed(2)}`;
    })
    .join(' ');
  return (
    <svg
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label="Courbe d'équité cumulée en R-multiples"
      className="w-full h-auto"
    >
      <rect x={0} y={0} width={width} height={height} fill="transparent" />
      <path d={path} fill="none" stroke="currentColor" strokeWidth={1.5} />
    </svg>
  );
}
