# DG-142 — Tableau performance public mensuel

> ⚠️ **OBSOLÈTE POST PIVOT 2026-05-27** — Tous claims PF 1.30 / 329 setups / IC 95 % retirés tant que validation OOS pending (Brier > +2 % AND DSR > 1.0 AND PBO < 0.5). Le tableau public peut être implémenté MAIS doit afficher uniquement : "Validation statistique OOS en cours, Sprint 1 du plan dev". Pas de chiffres performance affichés. Voir `docs/governance/decisions/2026-05-27_pivot_positioning_audit.md`.

**Effort** : ~14-20h · **Sprint** : S5 · **Owner** : code

---

## Objectif

Publier un **tableau de performance paper-trading public**, mis à jour automatiquement, accessible à `/track-record`. C'est la **preuve sociale honnête** qui substancie le PF 1.30 affiché en hero.

## Contexte

Sans track-record vérifiable, le PF 1.30 du hero n'est pas substanciable = revient à mentir au prospect. Avec un tableau public agrégé (PnL paper, win rate, drawdown, équité cumulée), la promesse devient prouvable.

## Périmètre

**IN** :
- Page `/track-record` accessible publique (sans auth)
- Tableau des trades clôturés (paper-trading) : date, instrument, direction, entry, exit, R-multiple, PnL paper
- Stats agrégées : PF, WR, DD max, n trades, exposure time moyen
- Graph équité cumulée (simple, line chart)
- Filtre par mois (last 1m / 3m / 6m / all)
- Mise à jour automatique quotidienne (cron J+1 23:59)
- Footer disclaimer : "Paper-trading uniquement. Pas un track-record réel."

**OUT** :
- Track-record live réel (pas en V1, edge_claim=False maintenu)
- Personnalisation par user
- Comparaison vs benchmark XAU spot (V2)
- Tear sheet PDF mensuel (DG-183, V3)

## Dépendances

- DG-025 scoring v2 stable
- SignalStore opérationnel avec trades clôturés
- Pipeline de calcul stats agrégées (à créer ou réutiliser)
- DG-072 Telegram public channel ouvert (cohérence wording)

## Fichiers à toucher

```
backend/
├── src/intelligence/
│   └── track_record/
│       ├── __init__.py
│       ├── aggregator.py             (à créer — calcule stats depuis signals.db)
│       ├── equity_curve.py           (à créer — série temporelle équité)
│       └── nightly_job.py            (à créer — cron J+1 23:59 UTC)
├── src/api/
│   └── routes/
│       └── track_record.py           (à créer — endpoint /api/v1/track-record)
└── tests/
    └── test_track_record.py

frontend/
├── app/
│   └── track-record/
│       └── page.tsx
├── components/
│   └── track-record/
│       ├── PerformanceHeader.tsx     (stats agrégées)
│       ├── TradesTable.tsx           (table trades clôturés)
│       ├── EquityChart.tsx           (line chart équité)
│       └── PeriodFilter.tsx          (1m / 3m / 6m / all)
└── lib/
    └── chart-utils.ts                (helpers chart sans dep lourde)
```

## Implémentation

### 1. Backend — aggregator

```python
# backend/src/intelligence/track_record/aggregator.py
from dataclasses import dataclass
from datetime import datetime, timedelta
from src.api.signal_store import SignalStore

@dataclass
class TrackRecordStats:
    period_label: str
    n_trades: int
    win_rate: float            # 0-1
    profit_factor: float
    profit_factor_ci95: tuple[float, float]
    drawdown_max_pct: float    # negative
    avg_exposure_hours: float
    total_r_multiples: float   # somme R cumulée
    instruments: list[str]


async def compute_stats(period_days: int | None = None) -> TrackRecordStats:
    """Calcule stats agrégées sur N derniers jours (None = all)."""
    cutoff = datetime.utcnow() - timedelta(days=period_days) if period_days else None
    trades = await SignalStore.get_closed_trades(since=cutoff)

    if not trades:
        return TrackRecordStats(
            period_label="N/A", n_trades=0, win_rate=0, profit_factor=0,
            profit_factor_ci95=(0, 0), drawdown_max_pct=0, avg_exposure_hours=0,
            total_r_multiples=0, instruments=[],
        )

    wins = [t for t in trades if t.r_multiple > 0]
    losses = [t for t in trades if t.r_multiple < 0]
    sum_wins = sum(t.r_multiple for t in wins)
    sum_losses = abs(sum(t.r_multiple for t in losses))

    pf = sum_wins / sum_losses if sum_losses > 0 else float('inf')
    wr = len(wins) / len(trades)

    # Bootstrap IC 95 % sur PF
    pf_ci = bootstrap_profit_factor_ci(trades, n_iter=1000, alpha=0.05)

    # Drawdown max sur equity curve
    equity_curve = compute_equity_curve(trades)
    dd_max = compute_max_drawdown(equity_curve)

    return TrackRecordStats(
        period_label=f"Last {period_days}d" if period_days else "All-time",
        n_trades=len(trades),
        win_rate=wr,
        profit_factor=pf,
        profit_factor_ci95=pf_ci,
        drawdown_max_pct=dd_max,
        avg_exposure_hours=sum(t.exposure_hours for t in trades) / len(trades),
        total_r_multiples=sum(t.r_multiple for t in trades),
        instruments=sorted(set(t.instrument for t in trades)),
    )


def bootstrap_profit_factor_ci(trades, n_iter=1000, alpha=0.05):
    import random
    pfs = []
    n = len(trades)
    for _ in range(n_iter):
        sample = [random.choice(trades) for _ in range(n)]
        wins = sum(t.r_multiple for t in sample if t.r_multiple > 0)
        losses = abs(sum(t.r_multiple for t in sample if t.r_multiple < 0))
        if losses > 0:
            pfs.append(wins / losses)
    pfs.sort()
    lower_idx = int(n_iter * alpha / 2)
    upper_idx = int(n_iter * (1 - alpha / 2))
    return (pfs[lower_idx], pfs[upper_idx])
```

### 2. Endpoint

```python
# backend/src/api/routes/track_record.py
from fastapi import APIRouter, Query
from src.intelligence.track_record.aggregator import compute_stats, compute_equity_curve

router = APIRouter(prefix="/api/v1/track-record")

@router.get("/")
async def get_track_record(period: str = Query("all", regex="^(1m|3m|6m|all)$")):
    period_map = {"1m": 30, "3m": 90, "6m": 180, "all": None}
    days = period_map[period]
    stats = await compute_stats(days)
    trades = await SignalStore.get_closed_trades(since=(datetime.utcnow() - timedelta(days=days)) if days else None)
    equity = compute_equity_curve(trades)

    return {
        "stats": asdict(stats),
        "trades": [
            {
                "date": t.closed_at.isoformat(),
                "instrument": t.instrument,
                "direction": t.direction,
                "entry": t.entry_price,
                "exit": t.exit_price,
                "r_multiple": t.r_multiple,
                "exposure_hours": t.exposure_hours,
            }
            for t in trades
        ],
        "equity_curve": equity,
        "last_updated_utc": datetime.utcnow().isoformat(),
        "disclaimer": "Paper-trading uniquement. Ne préjuge pas des performances futures.",
    }
```

### 3. Cron nightly

```python
# backend/src/intelligence/track_record/nightly_job.py
import asyncio
from datetime import datetime
from .aggregator import compute_stats, save_snapshot

async def nightly_recompute():
    """Cron J+1 23:59 UTC : recalcule stats + cache résultat."""
    for period in ["1m", "3m", "6m", "all"]:
        stats = await compute_stats({"1m":30,"3m":90,"6m":180,"all":None}[period])
        await save_snapshot(period, stats, datetime.utcnow())
```

Cron Fly.io :
```toml
# fly.toml
[processes]
  app = "uvicorn src.intelligence.main:app --host 0.0.0.0 --port 8000"
  cron_nightly = "python -m src.intelligence.track_record.nightly_job"

[[mounts]]
  source = "data"
  destination = "/app/data"
```

### 4. Frontend — page

```tsx
// frontend/app/track-record/page.tsx
'use client';
import { useState, useEffect } from 'react';
import PerformanceHeader from '@/components/track-record/PerformanceHeader';
import TradesTable from '@/components/track-record/TradesTable';
import EquityChart from '@/components/track-record/EquityChart';
import PeriodFilter from '@/components/track-record/PeriodFilter';

export default function TrackRecordPage() {
  const [period, setPeriod] = useState<'1m' | '3m' | '6m' | 'all'>('all');
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    fetch(`/api/v1/track-record/?period=${period}`)
      .then(r => r.json())
      .then(d => { setData(d); setLoading(false); });
  }, [period]);

  return (
    <main className="container mx-auto px-7 py-14 md:py-20">
      <div className="text-center mb-8">
        <div className="text-xs text-gold uppercase tracking-widest font-semibold mb-3">Track record honnête</div>
        <h1 className="text-3xl md:text-4xl font-semibold tracking-tight mb-3">Toutes nos lectures clôturées en paper-trading.</h1>
        <p className="text-base text-text-secondary max-w-2xl mx-auto">Aucune sélection, aucun cherry-pick. Forward only depuis le {data?.first_trade_date}. Mis à jour quotidiennement.</p>
      </div>

      <PeriodFilter value={period} onChange={setPeriod} />

      {loading ? <div>Chargement…</div> : (
        <>
          <PerformanceHeader stats={data.stats} />
          <EquityChart series={data.equity_curve} />
          <TradesTable trades={data.trades} />
        </>
      )}

      <p className="text-center text-xs text-text-muted mt-10 max-w-2xl mx-auto">
        <strong>Paper-trading uniquement.</strong> Les performances passées ne préjugent pas des performances futures. Ne constitue pas un conseil en investissement.
      </p>
    </main>
  );
}
```

### 5. EquityChart minimal sans dépendance lourde

```tsx
// frontend/components/track-record/EquityChart.tsx
import { useMemo } from 'react';

export default function EquityChart({ series }: { series: { date: string; cumulative_r: number }[] }) {
  const { points, maxR, minR } = useMemo(() => {
    if (series.length === 0) return { points: [], maxR: 0, minR: 0 };
    const maxR = Math.max(...series.map(s => s.cumulative_r));
    const minR = Math.min(...series.map(s => s.cumulative_r));
    const width = 800;
    const height = 240;
    const points = series.map((s, i) => {
      const x = (i / (series.length - 1)) * width;
      const y = height - ((s.cumulative_r - minR) / (maxR - minR + 0.001)) * (height - 20) - 10;
      return `${x},${y}`;
    }).join(' ');
    return { points, maxR, minR };
  }, [series]);

  return (
    <div className="bg-bg-card border border-border rounded-xl p-6 my-6">
      <h3 className="text-sm uppercase tracking-widest text-gold mb-3">Équité cumulée (R)</h3>
      <svg viewBox="0 0 800 240" className="w-full h-auto">
        <polyline points={points} fill="none" stroke="#c9a961" strokeWidth="2" />
        <text x="10" y="20" fill="#8b929c" fontSize="12">{maxR.toFixed(1)} R max</text>
        <text x="10" y="230" fill="#8b929c" fontSize="12">{minR.toFixed(1)} R min</text>
      </svg>
    </div>
  );
}
```

## Acceptance criteria

- [ ] Page `/track-record` accessible sans auth
- [ ] Affiche stats : n trades, win rate, PF avec IC 95 %, DD max, exposure time moyen
- [ ] Tableau trades clôturés : date, instrument, direction, entry/exit, R-multiple
- [ ] Filtre période 1m / 3m / 6m / all opérationnel
- [ ] Equity chart line affiché (SVG inline, pas de dep lourde Recharts/Chart.js)
- [ ] Disclaimer "Paper-trading uniquement" visible bas de page
- [ ] Mise à jour quotidienne via cron (J+1 23:59 UTC)
- [ ] Endpoint `/api/v1/track-record/?period=*` répond < 200 ms
- [ ] Aucune donnée user personnelle exposée (anonyme)
- [ ] Mobile responsive (table scrollable horizontalement)
- [ ] Aucun vocabulaire interdit (audit)

## Tests requis

```python
# backend/tests/test_track_record.py
@pytest.mark.asyncio
async def test_aggregator_with_known_trades():
    """Avec 4 trades connus (3 wins +1R/+2R/+1R, 1 loss -1R) : PF = 4/1 = 4.0"""
    trades = [
        Trade(r_multiple=1.0), Trade(r_multiple=2.0), Trade(r_multiple=1.0), Trade(r_multiple=-1.0)
    ]
    with patch_signalstore(trades):
        stats = await compute_stats(period_days=None)
    assert stats.n_trades == 4
    assert abs(stats.profit_factor - 4.0) < 0.01
    assert stats.win_rate == 0.75


@pytest.mark.asyncio
async def test_bootstrap_ci_includes_true_pf():
    """Sur 100 trades simulés PF≈1.3, l'IC bootstrap inclut bien 1.3"""
    trades = simulate_trades(n=100, true_pf=1.3)
    with patch_signalstore(trades):
        stats = await compute_stats(period_days=None)
    lo, hi = stats.profit_factor_ci95
    assert lo < 1.3 < hi
```

## Risques / pièges

- ❌ **Cacher les trades perdants** : violation totale de la posture "honest confidence". Tous les trades closed sont publics, sans sélection.
- ❌ **Wording "performance live"** : c'est du paper-trading. Disclaimer obligatoire.
- ❌ **Mise à jour temps réel** : pas utile + risque exposer trades en cours. Cron J+1 23:59 UTC = bonne discipline.
- ❌ **Authentification requise** : la page DOIT être publique sans login pour servir de preuve commerciale (prospect peut vérifier avant signup).
- ❌ **Dépendance lourde Recharts/Chart.js** : ralentit la page. SVG inline minimal suffit largement pour un line chart de 200 points.
- ❌ **Pas d'IC bootstrap** : PF nu = pas crédible. L'IC est essentiel.
