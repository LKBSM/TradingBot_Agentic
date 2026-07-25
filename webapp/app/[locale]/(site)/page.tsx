import { HashScroll } from '@/components/a11y/HashScroll';
import { BeforeAfterSection } from '@/components/landing/BeforeAfterSection';
import { ConversationReplaySection } from '@/components/landing/ConversationReplaySection';
import { FaqSection } from '@/components/landing/FaqSection';
import { HeroLive } from '@/components/landing/HeroLive';
import { HonestConfidenceSection } from '@/components/landing/HonestConfidenceSection';
import { HowItWorksSection } from '@/components/landing/HowItWorksSection';
import { MultiMarketSection } from '@/components/landing/MultiMarketSection';
import { PricingSection } from '@/components/landing/PricingSection';

/**
 * Landing commerciale ultime — composition par sections.
 *
 *   L2  · HeroLive                   (live MarketReadingCard + ChatPreview, no perf)
 *   L2  · HowItWorksSection          (définition produit : 4 étapes + points forts / démos)
 *   L3  · MultiMarketSection         (S2 multi-actifs XAU + EUR + Bientôt)
 *   L3  · ConversationReplaySection  (S3 conversations rejouables)
 *   L4  · BeforeAfterSection         (S4 avant/après — chaos vs lecture MIA)
 *   L4  · HonestConfidenceSection    (S5 vrais chiffres + citation imposée, full-width)
 *   L5  · PricingSection             (plan unique 49,99 $ mensuel / 39,99 $ annuel + contact B2B)
 *   L5  · FaqSection                 (6 questions clés)
 *   (Footer partagé dans <Footer /> du layout)
 */
export default function LandingPage() {
  return (
    <>
      <HashScroll />
      <HeroLive />
      <HowItWorksSection />
      <MultiMarketSection />
      <ConversationReplaySection />
      <BeforeAfterSection />
      <HonestConfidenceSection />
      <PricingSection />
      <FaqSection />
    </>
  );
}
