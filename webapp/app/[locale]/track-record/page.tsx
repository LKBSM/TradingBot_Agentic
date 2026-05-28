/**
 * DG-142 — /track-record (révision 2026-05-28 post-review).
 *
 * État V1 : la page rend une **vue méthodologie honnête**, jamais une
 * "vitrine commerciale" :
 *
 *  - Le feature flag ``NEXT_PUBLIC_TRACK_RECORD_PUBLIC`` doit valoir
 *    ``"true"`` pour que la page soit accessible. À défaut → ``notFound()``
 *    (Next 15 renvoie un vrai 404). Pas de redirection silencieuse, pas
 *    de page accessible sans intention opérateur explicite.
 *
 *  - Même quand le flag est ON, tant que l'API renvoie ``edge_claim = false``,
 *    la page n'affiche AUCUN chiffre brut (PF, hit rate, courbe d'équité).
 *    Elle décrit la méthodologie : gates empiriques, sources RAG, posture
 *    "validation OOS indépendante en attente". Cohérent avec UE 2024/2811 +
 *    pivot positioning 2026-05-27 (purge claim PF 1.30).
 *
 *  - Quand un jour ``edge_claim`` deviendra ``true`` (gates franchis,
 *    revue indépendante signée), on rebranche les chiffres ici. Le code
 *    de rendu numérique est conservé dans ``_NumbersPanel`` (commenté ci-
 *    dessous), mais NON appelé en V1.
 */
import type { Metadata } from 'next';
import { notFound } from 'next/navigation';

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

function flagIsOn(): boolean {
  return process.env.NEXT_PUBLIC_TRACK_RECORD_PUBLIC?.toLowerCase() === 'true';
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
  title: 'Méthodologie — M.I.A. Markets',
  description:
    "Critères empiriques d'évaluation : profit factor, IC 95 % bootstrap, deflated Sharpe, probability of backtest overfitting. Aucun chiffre publié tant que la validation OOS indépendante n'est pas signée.",
  robots: { index: false, follow: false },
};

export default async function TrackRecordPage() {
  // Hard 404 quand le flag n'est pas explicitement ON.
  if (!flagIsOn()) {
    notFound();
  }

  const apiBase = process.env.NEXT_PUBLIC_API_BASE ?? 'http://localhost:8000';
  const data = await fetchTrackRecord(apiBase);

  // Page méthodologie — toujours rendue, même quand edge_claim = false
  // (V1). Quand edge_claim devient true (post-gate review), on pourra
  // ré-introduire un panneau chiffres au-dessus de la méthodologie.
  return (
    <main className="mx-auto max-w-3xl px-4 py-16 space-y-10">
      <header className="space-y-2">
        <h1 className="text-3xl font-bold">Méthodologie M.I.A. Markets</h1>
        <p className="text-sm text-muted-foreground">
          Comment nous mesurons un edge. Pourquoi nous ne publions pas encore
          de chiffres.
        </p>
      </header>

      <section
        aria-labelledby="posture-heading"
        className="rounded-lg border border-amber-200 bg-amber-50 px-4 py-4"
      >
        <h2 id="posture-heading" className="text-lg font-semibold text-amber-900">
          Posture honnête
        </h2>
        <p className="mt-2 text-sm text-amber-900">
          Nous n&apos;affichons aucune métrique de performance tant que les
          quatre gates empiriques décrites ci-dessous ne sont pas franchies et
          revalidées par une revue indépendante. Performance passée non garantie
          de performance future. Pas de promesse de gain. Pas de signal d&apos;achat.
        </p>
      </section>

      <section aria-labelledby="gates-heading" className="space-y-3">
        <h2 id="gates-heading" className="text-xl font-semibold">
          Les quatre gates empiriques
        </h2>
        <ul className="space-y-3 text-sm">
          <li>
            <strong>Profit factor &gt; 1.20</strong> avec borne basse de l&apos;IC 95 %
            bootstrap (1000 itérations) strictement supérieure à 1.00. Une borne
            basse ≤ 1 signifie qu&apos;on ne peut pas exclure le hasard.
          </li>
          <li>
            <strong>Deflated Sharpe Ratio &gt; 1.0</strong> (Bailey &amp; López de Prado
            2014). Corrige le Sharpe observé pour la sélection multi-stratégies,
            la non-normalité, la taille d&apos;échantillon.
          </li>
          <li>
            <strong>Probability of Backtest Overfitting &lt; 0.5</strong>
            (Bailey-Borwein-López de Prado-Zhu 2014). Évalue la part de l&apos;edge
            qui pourrait s&apos;expliquer par l&apos;exploration de configurations.
          </li>
          <li>
            <strong>Walk-forward ≥ 2 ans hors-échantillon</strong> avec CPCV
            purgée + embargo (López de Prado, AFML ch. 7). Pas de paramètre
            calé sur la période d&apos;évaluation.
          </li>
        </ul>
      </section>

      <section aria-labelledby="rag-heading" className="space-y-3">
        <h2 id="rag-heading" className="text-xl font-semibold">
          Sources académiques sous-jacentes
        </h2>
        <p className="text-sm">
          Le chatbot s&apos;appuie sur un corpus de 22 références académiques curées
          (López de Prado, Corsi, Gibbs &amp; Candès, Angelopoulos &amp; Bates,
          Barndorff-Nielsen, Adams &amp; MacKay, Engle, Lo, Pedersen, Cont, etc.).
          Chaque réponse peut citer la source pertinente. La liste complète et
          le contenu indexé sont publiés dans le dépôt code.
        </p>
      </section>

      <section aria-labelledby="status-heading" className="space-y-3">
        <h2 id="status-heading" className="text-xl font-semibold">
          État actuel
        </h2>
        <p className="text-sm">
          {data?.backtest_window?.includes('pending')
            ? 'Validation OOS indépendante en cours. Aucune métrique de performance n\'est exposée publiquement.'
            : 'Validation OOS indépendante en cours. Les métriques internes ne franchissent pas encore les quatre gates. Aucun chiffre n\'est publié.'}
        </p>
        <p className="text-xs text-muted-foreground">
          La méthodologie de calcul (bootstrap 1000 itérations, seed déterministe,
          IC à 95 %) est implémentée et auditable dans le code source. Quand
          les gates seront franchies et la revue signée, les chiffres
          apparaîtront ici avec leur intervalle de confiance.
        </p>
      </section>

      <footer className="text-xs text-muted-foreground border-t pt-6">
        M.I.A. Markets — outil de compréhension augmentée. Aucune
        recommandation personnalisée. Aucun conseil en investissement.
      </footer>
    </main>
  );
}
