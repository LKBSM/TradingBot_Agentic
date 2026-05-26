import { Eye, Layers, MessageCircleQuestion } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';

const STEPS = [
  {
    number: '1',
    icon: Eye,
    title: 'Le verdict en 5 secondes',
    body:
      "Une lecture haussière, baissière ou neutre — avec une jauge de conviction calibrée et la marge d'erreur honnêtement affichée. Pas de jargon non expliqué, pas de promesse.",
  },
  {
    number: '2',
    icon: Layers,
    title: 'Le détail si vous voulez',
    body:
      'Structure de marché, régime, volatilité prévisionnelle, contexte événementiel, historique des setups similaires (PF + IC 95 %). Cinq sections dépliables — vous ouvrez ce qui vous intéresse.',
  },
  {
    number: '3',
    icon: MessageCircleQuestion,
    title: 'Le chatbot si vous avez une question',
    body:
      "Pourquoi ce score ? C'est quoi un retest armé ? Le FOMC change quoi ? Sentinel répond avec le contexte de la lecture, refuse poliment les demandes d'instruction d'achat.",
  },
] as const;

export function HowItWorksSection() {
  return (
    <section
      id="comment"
      aria-labelledby="how-title"
      className="container-prose space-y-8 py-12 sm:py-16"
    >
      <header className="space-y-2">
        <h2
          id="how-title"
          className="text-balance text-2xl font-semibold tracking-tight sm:text-3xl"
        >
          Trois couches d&apos;information, dans l&apos;ordre qui compte.
        </h2>
        <p className="max-w-2xl text-muted-foreground">
          Architecture progressive : vous voyez d&apos;abord l&apos;essentiel,
          vous explorez le détail si vous le voulez, vous posez vos questions
          au chatbot quand vous voulez. Jamais l&apos;inverse.
        </p>
      </header>

      <div className="grid gap-4 sm:grid-cols-3">
        {STEPS.map((step) => {
          const Icon = step.icon;
          return (
            <Card key={step.number} className="border-border/60 shadow-sm">
              <CardContent className="space-y-3 p-5 sm:p-6">
                <div className="flex items-center gap-2">
                  <span className="flex h-7 w-7 items-center justify-center rounded-full bg-primary text-xs font-semibold text-primary-foreground tabular-nums">
                    {step.number}
                  </span>
                  <Icon className="h-4 w-4 text-muted-foreground" aria-hidden />
                </div>
                <h3 className="text-base font-semibold tracking-tight">
                  {step.title}
                </h3>
                <p className="text-sm text-muted-foreground">{step.body}</p>
              </CardContent>
            </Card>
          );
        })}
      </div>
    </section>
  );
}
