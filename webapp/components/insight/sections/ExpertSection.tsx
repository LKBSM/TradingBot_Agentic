import { Lock } from 'lucide-react';
import {
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion';
import { Badge } from '@/components/ui/badge';
import { ComponentWaterfall } from '@/components/insight/expert/ComponentWaterfall';
import { ConformalIntervalViz } from '@/components/insight/expert/ConformalIntervalViz';
import { directionBadgeVariant } from '@/lib/insight-formatters';
import type { InsightSignalV2 } from '@/types/insight';

/**
 * Section "Détail expert" — 6e accordéon ajouté en V2.4 pour exposer le
 * niveau de granularité algo institutionnel justifiant le tier Stratège
 * (79 €/mois) : waterfall des 8 composantes + visualisation conformelle.
 *
 * V1 : visible pour tous (pas encore d'auth tier-gated). Le badge "🔒
 * STRATEGIST+" matérialise visuellement le verrouillage à venir (cf.
 * DG-101-MODIFIED) sans bloquer l'usage actuel. Une fois auth + Stripe
 * branchés (V3), la section sera réellement gated et le contenu remplacé
 * par un teaser "upgrade".
 */
export function ExpertSection({ signal }: { signal: InsightSignalV2 }) {
  const tone = directionBadgeVariant(signal.direction);

  return (
    <AccordionItem value="expert">
      <AccordionTrigger className="text-left text-sm">
        <span className="flex items-center gap-2">
          <span aria-hidden>🔬</span>
          <span>Détail expert</span>
          <Badge variant="secondary" className="ml-1 text-[10px]">
            <Lock className="mr-1 h-3 w-3" aria-hidden />
            STRATEGIST+
          </Badge>
        </span>
      </AccordionTrigger>
      <AccordionContent>
        <div className="space-y-6">
          <ComponentWaterfall
            components={signal.breakdown_components}
            tone={tone}
          />

          <div className="rounded-lg border bg-muted/20 p-4">
            <p className="mb-3 text-xs font-medium uppercase tracking-wider text-muted-foreground">
              Visualisation de l&apos;incertitude
            </p>
            <ConformalIntervalViz
              score={signal.conviction_0_100}
              label={signal.conviction_label}
              direction={signal.direction}
              uncertainty={signal.uncertainty}
            />
          </div>

          <p className="text-xs italic text-muted-foreground">
            Pipeline de calibration : LightGBM (8 features) → Régression
            isotonique (alignement hit-rate empirique) → Adaptive Conformal
            Inference (Gibbs-Candès 2021) → intervalle distribution-free
            avec garantie ≥ 1−α sous échangeabilité ou drift.
          </p>
        </div>
      </AccordionContent>
    </AccordionItem>
  );
}
