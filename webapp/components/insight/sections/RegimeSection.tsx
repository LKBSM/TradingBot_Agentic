import {
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';
import {
  formatChangepointStability,
  formatHmmLabel,
  formatJumpDescriptor,
  formatRegimeGate,
} from '@/lib/insight-formatters';
import type { InsightSignalV2 } from '@/types/insight';

type Tone = 'ok' | 'warn' | 'block';

const TONE_COLOR: Record<Tone, string> = {
  ok: 'text-sentinel-bull',
  warn: 'text-sentinel-warn',
  block: 'text-sentinel-bear',
};

export function RegimeSection({ signal }: { signal: InsightSignalV2 }) {
  const r = signal.regime_readout;
  const gate = formatRegimeGate(r.regime_gate_decision);
  const stability = formatChangepointStability(r.bocpd_changepoint_prob);
  const jump = formatJumpDescriptor(r.jump_ratio);

  return (
    <AccordionItem value="regime">
      <AccordionTrigger className="text-left text-sm">
        <span className="flex items-center gap-2">
          <span aria-hidden>🌊</span>
          <span>Régime de marché</span>
        </span>
      </AccordionTrigger>
      <AccordionContent>
        <div className="space-y-4">
          <div className="flex flex-wrap items-center gap-2">
            <Badge variant="secondary">{formatHmmLabel(r.hmm_label)}</Badge>
            <span className="text-xs text-muted-foreground">
              certitude {Math.round(r.hmm_posterior * 100)} %
            </span>
          </div>

          <dl className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            <Row
              label="Stabilité du régime"
              value={stability.label}
              tone={stability.tone}
              hint={`Probabilité de rupture : ${(r.bocpd_changepoint_prob * 100).toFixed(1).replace('.', ',')} %`}
            />
            <Row
              label="Nature des mouvements"
              value={jump.label}
              tone={jump.tone}
              hint={`Part des sauts dans la variance : ${Math.round(r.jump_ratio * 100)} %`}
            />
            <Row
              label="Persistance attendue"
              value={`≈ ${Math.round(r.expected_run_length)} bougies avant changement`}
              hint="Espérance de durée du régime actuel (BOCPD)."
              className="sm:col-span-2"
            />
            <Row
              label="Décision interne"
              value={gate.label}
              tone={gate.tone}
              className="sm:col-span-2"
            />
          </dl>

          <p className="text-xs italic text-muted-foreground">
            La « décision interne » est utilisée par le moteur pour pondérer la
            conviction. Elle n'est pas une instruction adressée au trader.
          </p>
        </div>
      </AccordionContent>
    </AccordionItem>
  );
}

function Row({
  label,
  value,
  tone,
  hint,
  className,
}: {
  label: string;
  value: string;
  tone?: Tone;
  hint?: string;
  className?: string;
}) {
  return (
    <div className={className}>
      <dt className="text-xs uppercase tracking-wide text-muted-foreground">{label}</dt>
      <dd className={cn('mt-1 text-sm font-medium', tone ? TONE_COLOR[tone] : 'text-foreground')}>
        {value}
      </dd>
      {hint && <p className="mt-0.5 text-xs text-muted-foreground">{hint}</p>}
    </div>
  );
}
