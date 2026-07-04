import { MessageCircle } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { ConversationReplayCard } from './ConversationReplayCard';
import { getChatbotScript } from '@/lib/chatbot';
import {
  getHeroLandingSample,
  LANDING_SAMPLES,
} from '@/lib/market-reading/landing-samples';

/**
 * Section 3 — « MIA répond aux vraies questions ».
 *
 * Trois conversations scriptées rejouables avec effet typing. Chacune
 * démontre une facette du moat conversationnel :
 *   1. Lecture vulgarisée (Qu'est-ce que cette lecture me dit ?)
 *   2. Contextualisation événementielle (Et si le FOMC ?)
 *   3. Refus pédagogique compliance (Donc je dois acheter ?)
 *
 * Le visiteur peut cliquer "Rejouer" pour relancer chaque conversation.
 */
export function ConversationReplaySection() {
  const xau = getHeroLandingSample();
  const eur = LANDING_SAMPLES[1];
  const xauScript = getChatbotScript(xau.id);
  const eurScript = eur ? getChatbotScript(eur.id) : null;

  if (!xauScript) return null;

  // Récupère les 3 questions ciblées dans les scripts existants.
  const whyScore = xauScript.questions.find((q) => q.id === 'why-this-score');
  const fomc = xauScript.questions.find((q) => q.id === 'fomc-impact');
  const buyOrNot = xauScript.questions.find((q) => q.id === 'buy-or-not');

  if (!whyScore || !fomc || !buyOrNot) return null;

  return (
    <section
      id="conversations"
      aria-labelledby="conversations-title"
      className="bg-muted/20 py-16 sm:py-20"
    >
      <div className="container-wide">
        <header className="mb-8 max-w-2xl">
          <Badge
            variant="secondary"
            className="mb-3 text-[11px] uppercase tracking-wider"
          >
            <MessageCircle className="mr-1 h-3 w-3" aria-hidden />
            Le chatbot · démonstration
          </Badge>
          <h2
            id="conversations-title"
            className="text-balance text-2xl font-semibold tracking-tight sm:text-3xl"
          >
            M.I.A Agent répond aux vraies questions.
          </h2>
          <p className="mt-3 text-pretty text-muted-foreground">
            Pas un script générique — un assistant qui connaît le contexte
            de chaque lecture, et qui refuse les questions qu&apos;il ne
            doit pas répondre.
          </p>
        </header>

        <div className="grid gap-5 lg:grid-cols-3 lg:gap-6">
          <ConversationReplayCard
            title="Comprendre une lecture"
            kicker="Pédagogie"
            question={whyScore.text}
            answer={whyScore.reply}
            instrument="XAU/USD M15"
          />
          <ConversationReplayCard
            title="Contextualiser un événement"
            kicker="Macro · blackout"
            question={fomc.text}
            answer={fomc.reply}
            instrument="XAU/USD M15"
          />
          <ConversationReplayCard
            title="Refuser un ordre"
            kicker="Refus pédagogique"
            question={buyOrNot.text}
            answer={buyOrNot.reply}
            instrument="XAU/USD M15"
            highlight="refusal"
          />
        </div>

        <p className="mt-6 text-xs italic text-muted-foreground">
          Les réponses ci-dessus sont scriptées pour la démo. En production,
          M.I.A Agent utilise Claude (Anthropic) avec le contexte InsightSignal
          v2.1.0 injecté et un prompt système qui interdit toute
          recommandation personnalisée.
        </p>
      </div>
    </section>
  );
}
