import { useTranslations } from 'next-intl';
import type { MethodologyFormula } from '@/lib/methodology/content';

/**
 * Présentation descriptive d'un élément chiffré/qualitatif affiché dans l'UI
 * (importance d'un Order Block, statut d'un FVG, phase de marché, plage
 * d'incertitude) — Chantier 5.D.
 *
 * On explique CE QUE l'élément décrit et QUELLES VARIABLES le composent, en
 * langage simple. Niveau 1.5 : jamais une formule prédictive, jamais une
 * promesse de résultat.
 */
export function ScoreFormula({ formula }: { formula: MethodologyFormula }) {
  const t = useTranslations('methodology');

  return (
    <div id={formula.id} className="scroll-mt-24">
      <h3 className="text-base font-semibold tracking-tight">
        {t(`formulas.${formula.id}.title`)}
      </h3>
      <p className="mt-1.5 text-sm text-muted-foreground">
        {t(`formulas.${formula.id}.description`)}
      </p>
      <p className="mt-3 text-[11px] font-medium uppercase tracking-wide text-muted-foreground/80">
        {t('formulas.variablesLabel')}
      </p>
      <ul className="mt-1.5 space-y-1.5 text-sm text-foreground">
        {formula.variables.map((_, i) => (
          <li key={i} className="flex items-start gap-2">
            <span
              className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-primary/60"
              aria-hidden
            />
            <span>{t(`formulas.${formula.id}.variables.${i}`)}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
